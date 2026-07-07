"""Gefen architecture smoke validation for audio models — ASR and TTS.

Memorization test (same contract as validate_llm.py): a fixed synthetic batch is
trained for N optimizer steps and the loss must fall by >30% with no NaN/Inf.
This exercises the parameter tensor shapes audio models produce — Conv1d feature
encoders, convolutional + recurrent TTS decoders, conformer conv-attention blocks,
and audio-codebook transformer heads — through Gefen's flattened block
partitioning and fused step. It is an optimizer/architecture check, not an audio
quality benchmark.

Models are built with RANDOM init from a small config (no pretrained download).

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idx> python validate_audio.py \
      --arch wav2vec2 --optimizer gefen --steps 60 --lr 1e-3 [--out results.jsonl]
  # --arch all  runs every registered architecture in sequence
"""

import argparse, json, sys, time, traceback

import torch
import torch.nn as nn
import torch.nn.functional as F


def _wav2vec2_ctc_setup(model_cls_name, batch=2, samples=16000, tgt_len=8):
    """transformers CTC ASR model (Conv1d feature encoder + transformer)."""

    def setup(device):
        import transformers as tf

        model_cls = getattr(tf, model_cls_name)
        cfg = model_cls.config_class(
            hidden_size=128, num_hidden_layers=2, num_attention_heads=4,
            intermediate_size=256, vocab_size=32,
            conv_dim=(64, 64, 64), conv_stride=(5, 2, 2), conv_kernel=(10, 3, 3),
            num_feat_extract_layers=3,
        )
        model = model_cls(cfg).to(device)
        model.train()
        torch.manual_seed(0)
        input_values = torch.randn(batch, samples, device=device)
        labels = torch.randint(1, cfg.vocab_size, (batch, tgt_len), device=device)

        def closure():
            return model(input_values=input_values, labels=labels).loss

        return model, closure

    return setup


def _tacotron2_setup(batch=2, n_symbols=64, tok_len=20, mel_len=80, n_mels=80):
    """torchaudio Tacotron2 TTS: conv encoder + LSTM decoder + attention."""

    def setup(device):
        from torchaudio.models import Tacotron2

        model = Tacotron2(n_symbol=n_symbols, n_mels=n_mels).to(device)
        model.train()
        torch.manual_seed(0)
        tokens = torch.randint(0, n_symbols, (batch, tok_len), device=device)
        token_lengths = torch.full((batch,), tok_len, device=device, dtype=torch.int32)
        mel_target = torch.randn(batch, n_mels, mel_len, device=device)
        mel_lengths = torch.full((batch,), mel_len, device=device, dtype=torch.int32)
        gate_target = torch.zeros(batch, mel_len, device=device)
        gate_target[:, -1] = 1.0

        def closure():
            mel_out, mel_post, gate_out, _ = model(
                tokens, token_lengths, mel_target, mel_lengths
            )
            return (
                F.mse_loss(mel_out, mel_target)
                + F.mse_loss(mel_post, mel_target)
                + F.binary_cross_entropy_with_logits(gate_out, gate_target)
            )

        return model, closure

    return setup


def _conformer_setup(batch=2, frames=100, in_dim=80, n_layers=4, vocab=32, tgt_len=8):
    """torchaudio Conformer encoder (conv + self-attention) with a CTC head."""

    def setup(device):
        from torchaudio.models import Conformer

        enc = Conformer(
            input_dim=in_dim, num_heads=4, ffn_dim=128,
            num_layers=n_layers, depthwise_conv_kernel_size=31,
        ).to(device)
        head = nn.Linear(in_dim, vocab).to(device)
        model = nn.ModuleList([enc, head])
        model.train()
        torch.manual_seed(0)
        x = torch.randn(batch, frames, in_dim, device=device)
        lengths = torch.full((batch,), frames, device=device, dtype=torch.int32)
        labels = torch.randint(1, vocab, (batch, tgt_len), device=device)
        label_lengths = torch.full((batch,), tgt_len, device=device, dtype=torch.int32)

        def closure():
            out, out_len = enc(x, lengths)
            logp = F.log_softmax(head(out), dim=-1).transpose(0, 1)  # (T,B,V)
            return F.ctc_loss(logp, labels, out_len, label_lengths, blank=0, zero_infinity=True)

        return model, closure

    return setup


def _dia_setup(batch=2, txt_len=16, audio_len=32):
    """transformers Dia TTS (text encoder + multi-codebook audio decoder)."""

    def setup(device):
        import transformers as tf

        cfg_cls = tf.DiaForConditionalGeneration.config_class
        cfg = cfg_cls()
        model = tf.DiaForConditionalGeneration(cfg).to(device)
        model.train()
        torch.manual_seed(0)
        enc = cfg.encoder_config
        dec = cfg.decoder_config
        n_cb = dec.num_channels
        input_ids = torch.randint(1, enc.vocab_size, (batch, txt_len), device=device)
        decoder_input_ids = torch.randint(
            1, dec.vocab_size, (batch, audio_len, n_cb), device=device
        )
        labels = decoder_input_ids.clone()

        def closure():
            return model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
            ).loss

        return model, closure

    return setup


ARCHES = {
    "wav2vec2": lambda: _wav2vec2_ctc_setup("Wav2Vec2ForCTC"),
    "hubert": lambda: _wav2vec2_ctc_setup("HubertForCTC"),
    "conformer": lambda: _conformer_setup(),
    "tacotron2": lambda: _tacotron2_setup(),
    "dia": lambda: _dia_setup(),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, help="architecture key, or 'all'")
    ap.add_argument("--optimizer", default="gefen", choices=["gefen", "hybrid", "adamw"])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip-grad-norm", type=float, default=None)
    ap.add_argument("--out", default="results_audio.jsonl")
    args = ap.parse_args()

    arches = list(ARCHES) if args.arch == "all" else [args.arch]
    any_fail = False
    for arch in arches:
        rec = run_one(arch, args)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec, indent=2))
        any_fail = any_fail or rec["status"] != "PASS"
    sys.exit(1 if any_fail else 0)


def run_one(arch, args):
    rec = {
        "task": "audio",
        "arch": arch,
        "optimizer": args.optimizer,
        "steps": args.steps,
        "lr": args.lr,
        "status": "FAIL",
        "error": None,
    }
    try:
        device = "cuda"
        torch.manual_seed(0)
        setup = ARCHES[arch]()
        model, closure = setup(device)
        rec["params_total_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        rec["params_trained_M"] = round(sum(p.numel() for _, p in named) / 1e6, 2)
        rec["param_ndims"] = sorted({p.ndim for _, p in named})

        import gefen

        if args.optimizer == "gefen":
            opt = gefen.Gefen(named, lr=args.lr)
        elif args.optimizer == "hybrid":
            muon_named, backup_named = gefen.split_params_for_muon(model)
            opt = gefen.GefenMuonHybrid(muon_named, backup_named, lr=args.lr)
        else:
            opt = torch.optim.AdamW([p for _, p in named], lr=args.lr)

        torch.cuda.reset_peak_memory_stats()
        losses = []
        t0 = time.time()
        for step in range(args.steps):
            loss = closure()
            if not torch.isfinite(loss):
                rec["losses"] = losses
                raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
            loss.backward()
            if args.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_([p for _, p in named], args.clip_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(round(loss.item(), 4))
        rec["train_s"] = round(time.time() - t0, 1)
        rec["losses"] = losses[:3] + ["..."] + losses[-3:] if len(losses) > 8 else losses
        rec["loss_first"], rec["loss_last"] = losses[0], losses[-1]
        rec["peak_vram_GiB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        drop = (losses[0] - losses[-1]) / max(abs(losses[0]), 1e-9)
        rec["loss_drop_pct"] = round(100 * drop, 1)
        rec["status"] = "PASS" if drop > 0.3 else "FAIL"
    except Exception:
        rec["error"] = traceback.format_exc()[-2000:]
        print(rec["error"], file=sys.stderr)
    return rec


if __name__ == "__main__":
    main()

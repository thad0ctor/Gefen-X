"""Gefen vs AdamW on real TTS — fine-tune pretrained Tacotron2 on LJSpeech.

Text-to-speech has no accuracy/mAP, so this reports held-out validation loss
(mel + post-net + gate) after fine-tuning the pretrained torchaudio Tacotron2 on
a real LJSpeech subset — a real-data convergence comparison against AdamW,
measured in loss. Same data / split / LR / epochs for both optimizers.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idx> python validate_tts_finetune.py \
      --optimizers adamw,gefen --epochs 8 --data <LJSpeech-1.1_dir> \
      [--n 400 --out results_tts.jsonl]
"""

import argparse, csv, json, os, sys, time, traceback, wave

import numpy as np
import torch
import torch.nn.functional as F


def load_wav(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr


def build_mel_transform(device):
    import torchaudio.transforms as T

    return T.MelSpectrogram(
        sample_rate=22050, n_fft=1024, win_length=1024, hop_length=256,
        n_mels=80, f_min=0.0, f_max=8000.0, power=1.0, norm="slaney", mel_scale="slaney",
    ).to(device)


def load_items(data, n):
    rows = []
    with open(os.path.join(data, "metadata.csv"), encoding="utf-8") as f:
        for parts in csv.reader(f, delimiter="|"):
            if len(parts) >= 2:
                text = parts[2] if len(parts) >= 3 and parts[2] else parts[1]
                rows.append((os.path.join(data, "wavs", parts[0] + ".wav"), text))
    return rows[:n]


def run_one(name, train, val, processor, tacotron_ctor, mel_tf, args, device):
    torch.manual_seed(0)
    model = tacotron_ctor().to(device)
    opt = build_optimizer(name, model, args.lr)

    def batchify(items):
        texts = [t for _, t in items]
        tokens, tok_lens = processor(texts)
        return tokens.to(device), tok_lens.to(device), [w for w, _ in items]

    def compute_loss(items):
        tokens, tok_lens, wavs = batchify(items)
        mels, mel_lens = [], []
        for w in wavs:
            a, _sr = load_wav(w)
            m = torch.log(torch.clamp(mel_tf(torch.from_numpy(a.copy()).to(device)), min=1e-5))
            mels.append(m)
            mel_lens.append(m.shape[-1])
        T = max(mel_lens)
        mel = torch.stack([F.pad(m, (0, T - m.shape[-1])) for m in mels])
        mel_lens = torch.tensor(mel_lens, device=device, dtype=torch.int32)
        gate = torch.zeros(len(items), T, device=device)
        for i, L in enumerate(mel_lens):
            gate[i, L - 1:] = 1.0
        # sort by token length desc (Tacotron2 packs padded sequences)
        order = torch.argsort(tok_lens, descending=True)
        tokens, tok_lens = tokens[order], tok_lens[order]
        mel, mel_lens, gate = mel[order], mel_lens[order], gate[order]
        mel_out, mel_post, gate_out, _ = model(tokens, tok_lens, mel, mel_lens)
        return (F.mse_loss(mel_out, mel) + F.mse_loss(mel_post, mel)
                + F.binary_cross_entropy_with_logits(gate_out, gate))

    torch.cuda.reset_peak_memory_stats()
    bs = args.batch
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(train)).tolist()
        for i in range(0, len(perm) - bs + 1, bs):
            batch = [train[j] for j in perm[i:i + bs]]
            opt.zero_grad(set_to_none=True)
            loss = compute_loss(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss ({name}, epoch {epoch})")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()

    model.eval()
    with torch.no_grad():
        losses = [compute_loss(val[i:i + bs]).item() for i in range(0, len(val) - bs + 1, bs)]
    if not losses:
        raise RuntimeError(f"no full validation batch available (val={len(val)}, batch={bs})")
    val_loss = sum(losses) / len(losses)
    peak = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    return val_loss, peak


def build_optimizer(name, model, lr):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if name == "gefen":
        import gefen
        return gefen.Gefen(named, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW([p for _, p in named], lr=lr)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizers", default="adamw,gefen")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--data", default="./LJSpeech-1.1")
    ap.add_argument("--out", default="results_tts.jsonl")
    args = ap.parse_args()

    import torchaudio

    device = "cuda"
    bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
    processor = bundle.get_text_processor()
    mel_tf = build_mel_transform(device)
    items = load_items(args.data, args.n)
    split = int(0.85 * len(items))
    train, val = items[:split], items[split:]
    print(f"train={len(train)} val={len(val)}")

    for name in [o.strip() for o in args.optimizers.split(",") if o.strip()]:
        rec = {
            "task": "tts-finetune", "arch": "tacotron2", "dataset": "ljspeech",
            "optimizer": name, "epochs": args.epochs, "lr": args.lr,
            "gpu": torch.cuda.get_device_name(0), "status": "FAIL", "error": None,
        }
        try:
            t0 = time.time()
            vloss, peak = run_one(name, train, val, processor, bundle.get_tacotron2, mel_tf, args, device)
            rec["val_loss"] = round(vloss, 4)
            rec["peak_vram_GiB"] = peak
            rec["train_s"] = round(time.time() - t0, 1)
            rec["status"] = "PASS"
            print(f"[tacotron2/{name}] val loss {vloss:.4f}  peak {peak} GiB  ({rec['train_s']}s)")
        except Exception:
            rec["error"] = traceback.format_exc()[-2000:]
            print(rec["error"], file=sys.stderr)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()

"""Gefen vs AdamW on real TTS — fine-tune pretrained Dia on LJSpeech.

Dia is a transformer text-to-speech model with a multi-codebook audio decoder.
This fine-tunes the pretrained Dia-1.6B on real (text, audio) LJSpeech pairs —
the DiaProcessor encodes each waveform to DAC codes as the training labels — and
reports held-out validation loss for Gefen vs AdamW (TTS has no accuracy/mAP).

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idx> python validate_dia_tts.py \
      --optimizers adamw,gefen --epochs 4 --n 96 --data <LJSpeech-1.1_dir> \
      [--out results_dia.jsonl]
"""

import argparse, csv, json, os, sys, time, traceback, wave

import numpy as np
import torch

DIA_SR = 44100


def load_wav_resampled(path, target_sr=DIA_SR):
    import torchaudio.functional as AF

    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    a = torch.from_numpy(np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0)
    if sr != target_sr:
        a = AF.resample(a, sr, target_sr)
    return a.numpy()


def load_items(data, n):
    rows = []
    with open(os.path.join(data, "metadata.csv"), encoding="utf-8") as f:
        for parts in csv.reader(f, delimiter="|"):
            if len(parts) >= 2:
                text = parts[2] if len(parts) >= 3 and parts[2] else parts[1]
                rows.append((os.path.join(data, "wavs", parts[0] + ".wav"), text))
    return rows[:n]


def build_optimizer(name, model, lr):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if name == "gefen":
        import gefen
        return gefen.Gefen(named, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW([p for _, p in named], lr=lr)
    raise ValueError(name)


def run_one(name, train, val, processor, model_ctor, args, device):
    torch.manual_seed(0)
    model = model_ctor().to(device)
    opt = build_optimizer(name, model, args.lr)

    def make_batch(items):
        texts = [t for _, t in items]
        audios = [load_wav_resampled(w) for w, _ in items]
        enc = processor(text=texts, audio=audios, output_labels=True, generation=False,
                        padding=True, return_tensors="pt", sampling_rate=DIA_SR)
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}

    def step_loss(items):
        return model(**make_batch(items)).loss

    torch.cuda.reset_peak_memory_stats()
    bs = args.batch
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(train)).tolist()
        for i in range(0, len(perm) - bs + 1, bs):
            batch = [train[j] for j in perm[i:i + bs]]
            opt.zero_grad(set_to_none=True)
            loss = step_loss(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss ({name}, epoch {epoch})")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()

    model.eval()
    with torch.no_grad():
        losses = [step_loss(val[i:i + bs]).item() for i in range(0, len(val) - bs + 1, bs)]
    if not losses:
        raise RuntimeError(f"no full validation batch available (val={len(val)}, batch={bs})")
    val_loss = sum(losses) / len(losses)
    peak = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    return val_loss, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizers", default="adamw,gefen")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--ckpt", default="nari-labs/Dia-1.6B-0626")
    ap.add_argument("--data", default="./LJSpeech-1.1")
    ap.add_argument("--out", default="results_dia.jsonl")
    args = ap.parse_args()

    import transformers as tf

    device = "cuda"
    processor = tf.DiaProcessor.from_pretrained(args.ckpt)
    items = load_items(args.data, args.n)
    split = int(0.85 * len(items))
    train, val = items[:split], items[split:]
    print(f"train={len(train)} val={len(val)}")

    def model_ctor():
        return tf.DiaForConditionalGeneration.from_pretrained(args.ckpt)

    for name in [o.strip() for o in args.optimizers.split(",") if o.strip()]:
        rec = {
            "task": "tts-finetune", "arch": "dia", "dataset": "ljspeech",
            "optimizer": name, "epochs": args.epochs, "lr": args.lr,
            "gpu": torch.cuda.get_device_name(0), "status": "FAIL", "error": None,
        }
        try:
            t0 = time.time()
            vloss, peak = run_one(name, train, val, processor, model_ctor, args, device)
            rec["val_loss"] = round(vloss, 4)
            rec["peak_vram_GiB"] = peak
            rec["train_s"] = round(time.time() - t0, 1)
            rec["status"] = "PASS"
            print(f"[dia/{name}] val loss {vloss:.4f}  peak {peak} GiB  ({rec['train_s']}s)")
        except Exception:
            rec["error"] = traceback.format_exc()[-2000:]
            print(rec["error"], file=sys.stderr)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()

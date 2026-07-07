"""Gefen vs AdamW on real speech — fine-tune pretrained ASR encoders.

Fine-tunes a pretrained Wav2Vec2 / HuBERT / Wav2Vec2-Conformer sequence
classifier on the real Speech Commands v2 keyword task and reports held-out test
accuracy, so each speech encoder is a real training comparison against AdamW.
The feature encoder is frozen (the standard wav2vec2 fine-tuning recipe); the
transformer/conformer body and the new head are trained.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idx> python validate_asr_finetune.py \
      --arch wav2vec2 --optimizers adamw,gefen --epochs 3 --data <sc_dir> \
      [--train-n 8000 --test-n 2000 --out results_asr.jsonl]
  # --arch all runs every registered encoder
"""

import argparse, json, os, sys, time, traceback, wave

import numpy as np
import torch
import torch.nn.functional as F

SAMPLE_RATE = 16000

ARCHES = {
    "wav2vec2": ("Wav2Vec2ForSequenceClassification", "facebook/wav2vec2-base"),
    "hubert": ("HubertForSequenceClassification", "facebook/hubert-base-ls960"),
    "conformer": ("Wav2Vec2ConformerForSequenceClassification",
                  "facebook/wav2vec2-conformer-rel-pos-large"),
}


def load_wav(path):
    with wave.open(path, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


class SC(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        a = load_wav(path)
        if len(a) < SAMPLE_RATE:
            a = np.pad(a, (0, SAMPLE_RATE - len(a)))
        else:
            a = a[:SAMPLE_RATE]
        return torch.from_numpy(a.copy()), y


def build_items(data, subset):
    from torchaudio.datasets import SPEECHCOMMANDS

    ds = SPEECHCOMMANDS(data, download=False, subset=subset)
    base = os.path.dirname(ds._path)
    items, labels = [], set()
    for i in range(len(ds)):
        rel, _sr, label, *_ = ds.get_metadata(i)
        items.append((os.path.join(base, rel), label))
        labels.add(label)
    return items, sorted(labels)


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.tensor([b[1] for b in batch])
    return xs, ys


def build_model(arch, num_labels, device):
    import transformers as tf

    cls_name, ckpt = ARCHES[arch]
    model = getattr(tf, cls_name).from_pretrained(
        ckpt, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    return model.to(device)


def build_optimizer(name, model, lr):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if name == "gefen":
        import gefen
        return gefen.Gefen(named, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW([p for _, p in named], lr=lr)
    raise ValueError(name)


def run_one(arch, name, train_ds, test_ds, n_labels, args, device):
    torch.manual_seed(0)
    model = build_model(arch, n_labels, device)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=6, collate_fn=collate, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch, shuffle=False, num_workers=6, collate_fn=collate
    )
    opt = build_optimizer(name, model, args.lr)

    torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = model(input_values=x, labels=y).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss ({arch}/{name}, epoch {epoch})")
            loss.backward()
            opt.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(input_values=x).logits.argmax(-1)
            correct += (pred == y).sum().item()
            total += y.numel()
    acc = 100.0 * correct / total
    peak = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    return acc, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, help="encoder key, or 'all'")
    ap.add_argument("--optimizers", default="adamw,gefen")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--data", default="./sc-data")
    ap.add_argument("--train-n", type=int, default=8000)
    ap.add_argument("--test-n", type=int, default=2000)
    ap.add_argument("--out", default="results_asr.jsonl")
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(0)
    train_items, labels = build_items(args.data, "training")
    test_items, _ = build_items(args.data, "testing")
    l2i = {l: i for i, l in enumerate(labels)}
    g = torch.Generator().manual_seed(0)
    tr_idx = torch.randperm(len(train_items), generator=g)[: args.train_n].tolist()
    te_idx = torch.randperm(len(test_items), generator=g)[: args.test_n].tolist()
    train_ds = SC([(train_items[i][0], l2i[train_items[i][1]]) for i in tr_idx])
    test_ds = SC([(test_items[i][0], l2i[test_items[i][1]]) for i in te_idx])
    print(f"labels={len(labels)} train={len(train_ds)} test={len(test_ds)}")

    arches = list(ARCHES) if args.arch == "all" else [args.arch]
    for arch in arches:
        for name in [o.strip() for o in args.optimizers.split(",") if o.strip()]:
            rec = {
                "task": "asr-finetune", "arch": arch, "dataset": "speechcommands",
                "optimizer": name, "epochs": args.epochs, "lr": args.lr,
                "gpu": torch.cuda.get_device_name(0), "status": "FAIL", "error": None,
            }
            try:
                t0 = time.time()
                acc, peak = run_one(arch, name, train_ds, test_ds, len(labels), args, device)
                rec["test_acc_pct"] = round(acc, 2)
                rec["peak_vram_GiB"] = peak
                rec["train_s"] = round(time.time() - t0, 1)
                rec["status"] = "PASS" if acc > 40.0 else "FAIL"
                print(f"[{arch}/{name}] test acc {acc:.2f}%  peak {peak} GiB  ({rec['train_s']}s)")
            except Exception:
                rec["error"] = traceback.format_exc()[-2000:]
                print(rec["error"], file=sys.stderr)
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()

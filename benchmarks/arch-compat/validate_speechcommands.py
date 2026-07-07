"""Gefen on real speech recognition — Speech Commands keyword recognition.

Trains an M5-style raw-waveform Conv1d recognizer on the real Speech Commands
v0.02 dataset (35 keyword classes) and reports held-out test accuracy, so it is
a genuine speech-recognition convergence comparison of Gefen against AdamW on a
1-D convolutional audio model (not a memorization smoke).

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idx> python validate_speechcommands.py \
      --optimizers adamw,gefen,hybrid --epochs 12 --seeds 0 --data <dir> \
      [--out results_speechcommands.jsonl]
"""

import argparse, json, os, sys, time, traceback, wave

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SAMPLE_RATE = 16000


def load_wav(path):
    """Read a 16 kHz mono PCM16 WAV to a float32 tensor (avoids torchaudio.load,
    which routes through torchcodec — an optional dep not installed here)."""
    with wave.open(path, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(a.copy())


class M5(nn.Module):
    """Raw-waveform Conv1d keyword recognizer (Dai et al. 2017)."""

    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, 80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, 3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, 3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, 3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = F.avg_pool1d(x, x.shape[-1]).permute(0, 2, 1)
        return F.log_softmax(self.fc1(x).squeeze(1), dim=-1)


class ClipDataset(torch.utils.data.Dataset):
    """(abs_path, label_idx) pairs; loads each 1-second clip on access."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label_idx = self.items[i]
        return load_wav(path), label_idx


def build_items(args, subset, label_to_idx=None):
    from torchaudio.datasets import SPEECHCOMMANDS

    ds = SPEECHCOMMANDS(args.data, download=args.download, subset=subset)
    base = os.path.dirname(ds._path)  # metadata paths include the archive-dir prefix
    items, labels = [], set()
    for i in range(len(ds)):
        rel, _sr, label, *_ = ds.get_metadata(i)
        items.append((os.path.join(base, rel), label))
        labels.add(label)
    return items, labels


def get_datasets(args):
    train_items, labels = build_items(args, "training")
    test_items, _ = build_items(args, "testing")
    labels = sorted(labels)
    l2i = {l: i for i, l in enumerate(labels)}
    train = ClipDataset([(p, l2i[l]) for p, l in train_items])
    test = ClipDataset([(p, l2i[l]) for p, l in test_items])
    return train, test, labels


def collate_factory(label_to_idx):
    def collate(batch):
        waves, targets = [], []
        for w, label_idx in batch:
            if w.numel() < SAMPLE_RATE:
                w = F.pad(w, (0, SAMPLE_RATE - w.numel()))
            else:
                w = w[:SAMPLE_RATE]
            waves.append(w)
            targets.append(label_idx)
        return torch.stack(waves).unsqueeze(1), torch.tensor(targets)

    return collate


def build_optimizer(name, model, lr):
    named = [(n, p) for n, p in model.named_parameters()]
    if name == "gefen":
        import gefen
        return gefen.Gefen(named, lr=lr)
    if name == "hybrid":
        import gefen
        muon_named, backup_named = gefen.split_params_for_muon(model)
        return gefen.GefenMuonHybrid(muon_named, backup_named, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW([p for _, p in named], lr=lr)
    raise ValueError(name)


def run_seed(name, seed, args, loaders, n_labels, device):
    train_loader, test_loader = loaders
    torch.manual_seed(seed)
    model = M5(n_output=n_labels).to(device)
    opt = build_optimizer(name, model, args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 3), gamma=0.3)

    torch.cuda.reset_peak_memory_stats()
    first = last = None
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.nll_loss(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss (seed {seed}, epoch {epoch})")
            loss.backward()
            opt.step()
            if first is None:
                first = round(loss.item(), 4)
            last = round(loss.item(), 4)
        sched.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(-1) == y).sum().item()
            total += y.numel()
    acc = 100.0 * correct / total
    peak = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    return acc, first, last, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizers", default="adamw,gefen,hybrid")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--data", default="./sc-data")
    ap.add_argument("--download", action="store_true", default=True)
    ap.add_argument("--no-download", dest="download", action="store_false")
    ap.add_argument("--out", default="results_speechcommands.jsonl")
    args = ap.parse_args()

    device = "cuda"
    train_ds, test_ds, labels = get_datasets(args)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    collate = collate_factory(label_to_idx)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=6,
        pin_memory=True, collate_fn=collate, drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=6,
        pin_memory=True, collate_fn=collate,
    )
    print(f"labels={len(labels)}  train={len(train_ds)}  test={len(test_ds)}")

    summary = {}
    for name in [o.strip() for o in args.optimizers.split(",") if o.strip()]:
        rec = {
            "task": "speechcommands", "model": "M5-conv1d", "optimizer": name,
            "epochs": args.epochs, "lr": args.lr, "n_labels": len(labels),
            "gpu": torch.cuda.get_device_name(0), "status": "FAIL", "error": None,
        }
        try:
            accs, peaks = [], []
            t0 = time.time()
            for seed in [int(s) for s in args.seeds.split(",") if s.strip() != ""]:
                acc, lf, ll, peak = run_seed(name, seed, args, (train_loader, test_loader), len(labels), device)
                accs.append(round(acc, 3))
                peaks.append(peak)
                print(f"[M5/{name}] seed {seed}: test acc {acc:.2f}%  loss {lf}->{ll}  peak {peak} GiB")
            mean = sum(accs) / len(accs)
            var = sum((a - mean) ** 2 for a in accs) / len(accs)
            rec["test_acc_pct"] = accs
            rec["test_acc_mean"] = round(mean, 3)
            rec["test_acc_std"] = round(var ** 0.5, 3)
            rec["peak_vram_GiB"] = round(max(peaks), 3)
            rec["train_s"] = round(time.time() - t0, 1)
            rec["status"] = "PASS" if mean > 60.0 else "FAIL"
            summary[name] = f"{mean:.2f} +/- {var**0.5:.2f}%"
        except Exception:
            rec["error"] = traceback.format_exc()[-2000:]
            print(rec["error"], file=sys.stderr)
        finally:
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")

    print(f"\n=== Speech Commands keyword recognition ({args.epochs} epochs) ===")
    for name, s in summary.items():
        print(f"  {name:8s}: {s}")


if __name__ == "__main__":
    main()

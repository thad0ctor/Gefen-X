"""Gefen on the official PyTorch MNIST CNN — a real end-to-end training run.

Unlike the memorization smoke tests in this directory, this trains the reference
CNN from the PyTorch examples repo to convergence and reports held-out test
accuracy, so it is a genuine "does Gefen learn as well as AdamW" check on a
convolutional network rather than an optimizer/architecture compatibility probe.

Recipe (matches the PyTorch examples default): two 3x3 conv layers with ReLU,
max-pool + dropout, two fully connected layers to a 10-way classifier; batch 64,
test batch 1000, 4 epochs, Adam-style betas (0.9, 0.999), lr 1e-3, StepLR with
step_size=1 and gamma=0.7. Standard deviations are over consecutive seeds.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idx> python validate_mnist_cnn.py \
      --optimizers adamw,gefen,hybrid --seeds 0,1,2 --epochs 4 --data <dir> \
      [--out results_mnist.jsonl]
"""

import argparse, json, os, sys, time, traceback

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """The PyTorch examples MNIST CNN, verbatim."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def build_optimizer(name, model, lr):
    named = [(n, p) for n, p in model.named_parameters()]
    if name == "gefen":
        import gefen
        return gefen.Gefen(named, lr=lr, betas=(0.9, 0.999))
    if name == "hybrid":
        import gefen
        muon_named, backup_named = gefen.split_params_for_muon(model)
        return gefen.GefenMuonHybrid(muon_named, backup_named, lr=lr, betas=(0.9, 0.999))
    if name == "adamw":
        return torch.optim.AdamW([p for _, p in named], lr=lr, betas=(0.9, 0.999))
    raise ValueError(name)


def run_seed(name, seed, args, loaders, device):
    train_loader, test_loader = loaders
    torch.manual_seed(seed)
    model = Net().to(device)
    opt = build_optimizer(name, model, args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.7)

    torch.cuda.reset_peak_memory_stats()
    first_loss = last_loss = None
    for epoch in range(args.epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.nll_loss(model(data), target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss (seed {seed}, epoch {epoch})")
            loss.backward()
            opt.step()
            if first_loss is None:
                first_loss = round(loss.item(), 4)
            last_loss = round(loss.item(), 4)
        sched.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.numel()
    acc = 100.0 * correct / total
    peak = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    return acc, first_loss, last_loss, peak


def make_loaders(args):
    from torchvision import datasets, transforms

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_ds = datasets.MNIST(args.data, train=True, download=args.download, transform=tf)
    test_ds = datasets.MNIST(args.data, train=False, download=args.download, transform=tf)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    return train_loader, test_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizers", default="adamw,gefen,hybrid")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--test-batch-size", type=int, default=1000)
    ap.add_argument("--data", default="./mnist-data")
    ap.add_argument("--download", action="store_true", default=True)
    ap.add_argument("--no-download", dest="download", action="store_false")
    ap.add_argument("--out", default="results_mnist.jsonl")
    args = ap.parse_args()

    device = "cuda"
    loaders = make_loaders(args)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    summary = {}
    for name in [o.strip() for o in args.optimizers.split(",") if o.strip()]:
        rec = {
            "task": "mnist-cnn",
            "optimizer": name,
            "epochs": args.epochs,
            "lr": args.lr,
            "seeds": seeds,
            "gpu": torch.cuda.get_device_name(0),
            "status": "FAIL",
            "error": None,
        }
        try:
            accs, losses_first, losses_last, peaks = [], [], [], []
            t0 = time.time()
            for seed in seeds:
                acc, lf, ll, peak = run_seed(name, seed, args, loaders, device)
                accs.append(round(acc, 3))
                losses_first.append(lf)
                losses_last.append(ll)
                peaks.append(peak)
                print(f"[{name}] seed {seed}: test acc {acc:.2f}%  loss {lf}->{ll}  peak {peak} GiB")
            mean = sum(accs) / len(accs)
            var = sum((a - mean) ** 2 for a in accs) / len(accs)
            rec["test_acc_pct"] = accs
            rec["test_acc_mean"] = round(mean, 3)
            rec["test_acc_std"] = round(var ** 0.5, 3)
            rec["loss_first"] = losses_first
            rec["loss_last"] = losses_last
            rec["peak_vram_GiB"] = round(max(peaks), 3)
            rec["train_s"] = round(time.time() - t0, 1)
            rec["status"] = "PASS" if mean > 95.0 else "FAIL"
            summary[name] = f"{mean:.2f} +/- {var**0.5:.2f}%"
        except Exception:
            rec["error"] = traceback.format_exc()[-2000:]
            print(rec["error"], file=sys.stderr)
        finally:
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec, indent=2))

    print("\n=== MNIST CNN test accuracy (mean +/- std over seeds) ===")
    for name, s in summary.items():
        print(f"  {name:8s}: {s}")


if __name__ == "__main__":
    main()

"""Gefen on real CIFAR-10 image classification — end-to-end training.

Trains a torchvision CNN on the real CIFAR-10 dataset and reports held-out test
accuracy, so it is a genuine convergence-quality comparison of Gefen against
AdamW on a convolutional network (not a memorization/compatibility smoke).

ResNet uses the standard CIFAR stem (3x3 stride-1 conv, no max-pool) so 32x32
inputs are not downsampled away before the residual stages.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idx> python validate_cifar10.py \
      --arch resnet18 --optimizers adamw,gefen,hybrid --epochs 20 --seeds 0 \
      --data <dir> [--out results_cifar10.jsonl]
"""

import argparse, json, sys, time, traceback

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_model(arch, num_classes=10):
    import torchvision.models as m

    model = getattr(m, arch)(weights=None, num_classes=num_classes)
    # adapt ResNet-family stems for 32x32 CIFAR inputs
    if arch.startswith("resnet"):
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    return model


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


def make_loaders(args):
    import os
    from torchvision import datasets, transforms

    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    # ImageFolder layout (<data>/cifar10/{train,test}/<class>/*.png) if present,
    # else fall back to torchvision's pickle CIFAR10 download.
    imgfolder = os.path.join(args.data, "cifar10")
    if os.path.isdir(os.path.join(imgfolder, "train")):
        train_ds = datasets.ImageFolder(os.path.join(imgfolder, "train"), transform=train_tf)
        test_ds = datasets.ImageFolder(os.path.join(imgfolder, "test"), transform=test_tf)
    else:
        train_ds = datasets.CIFAR10(args.data, train=True, download=args.download, transform=train_tf)
        test_ds = datasets.CIFAR10(args.data, train=False, download=args.download, transform=test_tf)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
    )
    return train_loader, test_loader


def run_seed(name, seed, args, loaders, device):
    train_loader, test_loader = loaders
    torch.manual_seed(seed)
    model = build_model(args.arch).to(device)
    opt = build_optimizer(name, model, args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    torch.cuda.reset_peak_memory_stats()
    first = last = None
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
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
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    acc = 100.0 * correct / total
    peak = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    return acc, first, last, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--optimizers", default="adamw,gefen,hybrid")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--data", default="./cifar-data")
    ap.add_argument("--download", action="store_true", default=True)
    ap.add_argument("--no-download", dest="download", action="store_false")
    ap.add_argument("--out", default="results_cifar10.jsonl")
    args = ap.parse_args()

    device = "cuda"
    loaders = make_loaders(args)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    summary = {}
    for name in [o.strip() for o in args.optimizers.split(",") if o.strip()]:
        rec = {
            "task": "cifar10", "arch": args.arch, "optimizer": name,
            "epochs": args.epochs, "lr": args.lr, "seeds": seeds,
            "gpu": torch.cuda.get_device_name(0), "status": "FAIL", "error": None,
        }
        try:
            accs, peaks = [], []
            t0 = time.time()
            for seed in seeds:
                acc, lf, ll, peak = run_seed(name, seed, args, loaders, device)
                accs.append(round(acc, 3))
                peaks.append(peak)
                print(f"[{args.arch}/{name}] seed {seed}: test acc {acc:.2f}%  loss {lf}->{ll}  peak {peak} GiB")
            mean = sum(accs) / len(accs)
            var = sum((a - mean) ** 2 for a in accs) / len(accs)
            rec["test_acc_pct"] = accs
            rec["test_acc_mean"] = round(mean, 3)
            rec["test_acc_std"] = round(var ** 0.5, 3)
            rec["peak_vram_GiB"] = round(max(peaks), 3)
            rec["train_s"] = round(time.time() - t0, 1)
            rec["status"] = "PASS" if mean > 70.0 else "FAIL"
            summary[name] = f"{mean:.2f} +/- {var**0.5:.2f}%"
        except Exception:
            rec["error"] = traceback.format_exc()[-2000:]
            print(rec["error"], file=sys.stderr)
        finally:
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")

    print(f"\n=== CIFAR-10 {args.arch} test accuracy ({args.epochs} epochs) ===")
    for name, s in summary.items():
        print(f"  {name:8s}: {s}")


if __name__ == "__main__":
    main()

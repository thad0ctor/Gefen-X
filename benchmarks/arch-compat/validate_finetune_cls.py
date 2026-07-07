"""Gefen vs AdamW on real image classification — fine-tune pretrained classifiers.

Fine-tunes an ImageNet-pretrained torchvision classifier on the real Imagenette
dataset (a 10-class ImageNet subset) and reports held-out test accuracy, so each
architecture is a real convergence comparison against AdamW rather than a
memorization smoke. Same model / data / LR / schedule / epochs for both.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idx> python validate_finetune_cls.py \
      --arch resnet50 --optimizers adamw,gefen --epochs 6 --data <imagenette_dir> \
      [--out results_ftcls.jsonl]
  # --arch all runs every registered classifier
"""

import argparse, json, sys, time, traceback

import torch
import torch.nn as nn
import torch.nn.functional as F


# arch -> (builder name, default weights enum name)
ARCHES = {
    "resnet50": "ResNet50_Weights",
    "convnext_tiny": "ConvNeXt_Tiny_Weights",
    "efficientnet_b0": "EfficientNet_B0_Weights",
    "mobilenet_v3_large": "MobileNet_V3_Large_Weights",
    "densenet121": "DenseNet121_Weights",
    "regnet_y_1_6gf": "RegNet_Y_1_6GF_Weights",
    "vit_b_16": "ViT_B_16_Weights",
    "swin_t": "Swin_T_Weights",
}


def build_pretrained(arch, num_classes=10):
    import torchvision.models as m

    weights = getattr(m, ARCHES[arch]).DEFAULT
    model = getattr(m, arch)(weights=weights)
    # swap the final classifier for the new label count; everything else keeps
    # its pretrained weights.
    if arch == "resnet50" or arch.startswith("regnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "densenet121":
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif arch in ("efficientnet_b0", "mobilenet_v3_large", "convnext_tiny"):
        last = model.classifier[-1]
        model.classifier[-1] = nn.Linear(last.in_features, num_classes)
    elif arch == "vit_b_16":
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif arch == "swin_t":
        model.head = nn.Linear(model.head.in_features, num_classes)
    else:
        raise ValueError(arch)
    return model, weights.transforms()


def build_optimizer(name, model, lr):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
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


def make_loaders(args, preprocess):
    import os
    from torchvision import datasets
    from torchvision.transforms import v2 as T

    # pretrained transforms give the eval pipeline (resize/crop/normalize); add
    # a light flip for the train split.
    train_tf = T.Compose([T.RandomHorizontalFlip(), preprocess])
    train_ds = datasets.ImageFolder(os.path.join(args.data, "train"), transform=train_tf)
    test_ds = datasets.ImageFolder(os.path.join(args.data, "val"), transform=preprocess)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
    )
    return train_loader, test_loader


def run_one(arch, name, args, device):
    torch.manual_seed(0)
    model, preprocess = build_pretrained(arch)
    model = model.to(device)
    train_loader, test_loader = make_loaders(args, preprocess)
    opt = build_optimizer(name, model, args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss ({arch}/{name}, epoch {epoch})")
            loss.backward()
            opt.step()
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
    return acc, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, help="classifier key, or 'all'")
    ap.add_argument("--optimizers", default="adamw,gefen")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--data", default="./imagenette2-160")
    ap.add_argument("--out", default="results_ftcls.jsonl")
    args = ap.parse_args()

    device = "cuda"
    arches = list(ARCHES) if args.arch == "all" else [args.arch]
    for arch in arches:
        for name in [o.strip() for o in args.optimizers.split(",") if o.strip()]:
            rec = {
                "task": "finetune-cls", "arch": arch, "dataset": "imagenette",
                "optimizer": name, "epochs": args.epochs, "lr": args.lr,
                "gpu": torch.cuda.get_device_name(0), "status": "FAIL", "error": None,
            }
            try:
                t0 = time.time()
                acc, peak = run_one(arch, name, args, device)
                rec["test_acc_pct"] = round(acc, 2)
                rec["peak_vram_GiB"] = peak
                rec["train_s"] = round(time.time() - t0, 1)
                rec["status"] = "PASS" if acc > 50.0 else "FAIL"
                print(f"[{arch}/{name}] test acc {acc:.2f}%  peak {peak} GiB  ({rec['train_s']}s)")
            except Exception:
                rec["error"] = traceback.format_exc()[-2000:]
                print(rec["error"], file=sys.stderr)
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()

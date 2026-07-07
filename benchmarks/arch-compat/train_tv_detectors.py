"""Gefen vs AdamW on real detection — fine-tune torchvision detectors on COCO128.

Fine-tunes a COCO-pretrained torchvision detector (Faster R-CNN, RetinaNet, SSD,
FCOS) on the real COCO128 dataset and reports COCO mAP (torchmetrics), so each
detector is a real training comparison against AdamW rather than a memorization
smoke. COCO128 labels are mapped to the pretrained models' COCO-91 label space so
the pretrained heads stay meaningful.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idx> python train_tv_detectors.py \
      --arch retinanet_resnet50_fpn --optimizers adamw,gefen --epochs 20 \
      --coco128 <coco128_dir> [--out results_tvdet.jsonl]
  # --arch all runs every registered detector
"""

import argparse, itertools, json, os, sys, time, traceback

import torch

# YOLO contiguous idx (0-79) -> COCO-91 category id, matching torchvision's
# pretrained detection label space.
COCO91_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]

ARCHES = ["fasterrcnn_resnet50_fpn", "retinanet_resnet50_fpn", "ssd300_vgg16", "fcos_resnet50_fpn"]


class Coco128(torch.utils.data.Dataset):
    """COCO128 images + YOLO labels -> (image tensor, target dict in COCO-91 ids)."""

    def __init__(self, coco128_dir):
        from torchvision.io import read_image

        self.read_image = read_image
        self.img_dir = os.path.join(coco128_dir, "images", "train2017")
        self.lbl_dir = os.path.join(coco128_dir, "labels", "train2017")
        self.files = sorted(f for f in os.listdir(self.img_dir) if f.lower().endswith(".jpg"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        import torchvision.transforms.v2.functional as TF

        fname = self.files[i]
        img = self.read_image(os.path.join(self.img_dir, fname))
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        img = TF.to_dtype(img[:3], torch.float32, scale=True)
        _, H, W = img.shape
        boxes, labels = [], []
        lbl = os.path.join(self.lbl_dir, os.path.splitext(fname)[0] + ".txt")
        if os.path.exists(lbl):
            for line in open(lbl):
                p = line.split()
                if len(p) < 5:
                    continue
                c, cx, cy, w, h = (float(x) for x in p[:5])
                bw, bh = w * W, h * H
                x0, y0 = (cx * W) - bw / 2, (cy * H) - bh / 2
                boxes.append([x0, y0, x0 + bw, y0 + bh])
                labels.append(COCO91_IDS[int(c)])
        if not boxes:
            boxes = torch.zeros((0, 4)); labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        return img, {"boxes": boxes, "labels": labels}


def build_detector(arch):
    import torchvision.models.detection as d

    return getattr(d, arch)(weights="DEFAULT")


def build_optimizer(name, model, lr):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if name == "gefen":
        import gefen
        return gefen.Gefen(named, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW([p for _, p in named], lr=lr)
    raise ValueError(name)


def find_lr(arch, name, ds, args, device):
    """Ramp the LR on a throwaway copy of this detector to pick each optimizer's
    own fair base LR (Gefen's fair LR is typically below AdamW's)."""
    from gefen.tools.lr_calibration import lr_range_test

    torch.manual_seed(0)
    model = build_detector(arch).to(device)
    model.train()
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True, num_workers=4,
        collate_fn=lambda b: tuple(zip(*b)), drop_last=True,
    )
    it = itertools.cycle(loader)

    def closure():
        imgs, targets = next(it)
        imgs = [im.to(device) for im in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss = sum(model(imgs, targets).values())
        loss.backward()
        return loss

    opt = build_optimizer(name, model, 1e-4)  # LR overridden by the ramp
    res = lr_range_test(opt, closure, num_iter=args.lrfind_iter,
                        start_lr=1e-6, end_lr=1e-1)
    print(f"[{arch}/{name}] lr-find: suggested {res.suggested_lr:.2e} "
          f"(min-loss {res.min_loss_lr:.2e})")
    return res.suggested_lr


def run_one(arch, name, ds, args, device):
    lr = find_lr(arch, name, ds, args, device) if args.lr_find else args.lr
    torch.manual_seed(0)
    model = build_detector(arch).to(device)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True, num_workers=4,
        collate_fn=lambda b: tuple(zip(*b)),
    )
    opt = build_optimizer(name, model, lr)

    torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        model.train()
        for imgs, targets in loader:
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            opt.zero_grad(set_to_none=True)
            loss = sum(model(imgs, targets).values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss ({arch}/{name}, epoch {epoch})")
            loss.backward()
            opt.step()

    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    metric = MeanAveragePrecision(box_format="xyxy")
    model.eval()
    eval_loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=False, num_workers=4,
        collate_fn=lambda b: tuple(zip(*b)),
    )
    with torch.no_grad():
        for imgs, targets in eval_loader:
            imgs = [im.to(device) for im in imgs]
            preds = model(imgs)
            preds = [{k: v.cpu() for k, v in p.items()} for p in preds]
            metric.update(preds, list(targets))
    res = metric.compute()
    peak = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    return float(res["map"]), float(res["map_50"]), peak, lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, help="detector key, or 'all'")
    ap.add_argument("--optimizers", default="adamw,gefen")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-find", action="store_true",
                    help="pick each optimizer's own base LR via lr_range_test before training. "
                         "NOTE: the ramp finder needs a descending loss curve; a COCO-pretrained "
                         "detector fine-tuned on in-distribution COCO128 starts near its loss "
                         "minimum, so the ramp only rises and the finder returns a degenerate "
                         "~start_lr. Use an explicit LR grid (loop --lr) for pretrained fine-tunes; "
                         "--lr-find is for from-scratch training.")
    ap.add_argument("--lrfind-iter", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--coco128", default="./coco128")
    ap.add_argument("--out", default="results_tvdet.jsonl")
    args = ap.parse_args()

    device = "cuda"
    ds = Coco128(args.coco128)
    arches = ARCHES if args.arch == "all" else [args.arch]
    for arch in arches:
        for name in [o.strip() for o in args.optimizers.split(",") if o.strip()]:
            rec = {
                "task": "tv-detector", "arch": arch, "dataset": "coco128",
                "optimizer": name, "epochs": args.epochs, "lr": args.lr,
                "lr_find": args.lr_find,
                "gpu": torch.cuda.get_device_name(0), "status": "FAIL", "error": None,
            }
            try:
                t0 = time.time()
                m, m50, peak, used_lr = run_one(arch, name, ds, args, device)
                rec["lr"] = used_lr
                rec["map50_95"] = round(m, 4)
                rec["map50"] = round(m50, 4)
                rec["peak_vram_GiB"] = peak
                rec["train_s"] = round(time.time() - t0, 1)
                rec["status"] = "PASS" if m > 0.05 else "FAIL"
                print(f"[{arch}/{name}] lr={used_lr:.2e} mAP50-95={rec['map50_95']} mAP50={rec['map50']} ({rec['train_s']}s)")
            except Exception:
                rec["error"] = traceback.format_exc()[-2000:]
                print(rec["error"], file=sys.stderr)
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()

"""Gefen vs AdamW on real RF-DETR detection — roboflow RF-DETR on COCO128.

Fine-tunes RF-DETR (Nano) on a real COCO-format detection dataset via rfdetr's
own PyTorch-Lightning training stack and reports mAP, so it is a real
transformer-detector training comparison (not a memorization smoke). Gefen is
injected by monkeypatching the Lightning module's `configure_optimizers`; both
optimizers use the same dataset / schedule / epochs.

`--make-dataset-from-coco128 <coco128_dir>` converts an ultralytics COCO128
checkout (YOLO-format labels) into the roboflow COCO layout RF-DETR expects
(`<out>/{train,valid,test}/_annotations.coco.json` + images), reusing the 128
images for every split (a small real-data overfit that still requires the
optimizer to actually fit boxes + classes).

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idx> python train_rfdetr_coco.py \
      --make-dataset-from-coco128 <coco128_dir> --dataset-dir <out> \
      --optimizers adamw,gefen --epochs 30 --lr 1e-4 [--out results_rfdetr.jsonl]
"""

import argparse, json, os, shutil, sys, time, traceback

_USE_GEFEN = {"on": False}
_LAST_METRICS = {}

# Standard 80-thing -> COCO-91 category id map (YOLO contiguous idx -> COCO id).
# Using these ids and pinning num_classes=90 keeps RF-DETR's pretrained COCO
# detection head (91-way class embed) instead of reinitializing it, so training
# fine-tunes a real detector from ~its pretrained mAP rather than from scratch.
COCO91_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]
PRETRAINED_NUM_CLASSES = 90

COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def make_dataset_from_coco128(coco128_dir, out_dir, keep_classes=None, preserve_head=False):
    """YOLO-format COCO128 -> roboflow COCO layout (train/valid/test all = the same images).

    keep_classes: optional list of YOLO class ids to keep, remapped to a
    contiguous 1..K range (dense few-class task).

    preserve_head: emit standard COCO-91 category ids + the full 80-category set
    so, with num_classes pinned to 90, RF-DETR keeps its pretrained COCO head and
    fine-tunes from ~its pretrained mAP instead of training a fresh head.
    """
    from PIL import Image

    img_dir = os.path.join(coco128_dir, "images", "train2017")
    lbl_dir = os.path.join(coco128_dir, "labels", "train2017")
    if keep_classes is not None:
        remap = {c: i + 1 for i, c in enumerate(keep_classes)}       # yolo id -> 1..K
        categories = [{"id": i + 1, "name": COCO80[c], "supercategory": "none"}
                      for i, c in enumerate(keep_classes)]
    elif preserve_head:
        remap = {c: COCO91_IDS[c] for c in range(len(COCO80))}       # yolo id -> COCO-91 id
        categories = [{"id": COCO91_IDS[c], "name": COCO80[c], "supercategory": "none"}
                      for c in range(len(COCO80))]
    else:
        remap = {c: c + 1 for c in range(len(COCO80))}               # yolo id -> 1..80
        categories = [{"id": i + 1, "name": n, "supercategory": "none"}
                      for i, n in enumerate(COCO80)]

    images, annotations = [], []
    ann_id = img_id = 0
    all_files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg")))
    for fname in all_files:
        lbl = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
        boxes = []
        if os.path.exists(lbl):
            for line in open(lbl):
                parts = line.split()
                if len(parts) < 5:
                    continue
                c = int(float(parts[0]))
                if c not in remap:
                    continue
                boxes.append((c, *(float(x) for x in parts[1:5])))
        if keep_classes is not None and not boxes:
            continue  # drop images with no kept boxes
        img_id += 1
        W, H = Image.open(os.path.join(img_dir, fname)).size
        images.append({"id": img_id, "file_name": fname, "width": W, "height": H})
        for c, cx, cy, w, h in boxes:
            bw, bh = w * W, h * H
            bx, by = (cx * W) - bw / 2, (cy * H) - bh / 2
            ann_id += 1
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": remap[c],
                "bbox": [round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2), "iscrowd": 0, "segmentation": [],
            })
    coco = {"images": images, "annotations": annotations, "categories": categories}
    for split in ("train", "valid", "test"):
        sdir = os.path.join(out_dir, split)
        os.makedirs(sdir, exist_ok=True)
        for im in images:
            dst = os.path.join(sdir, im["file_name"])
            if not os.path.exists(dst):
                shutil.copy(os.path.join(img_dir, im["file_name"]), dst)
        with open(os.path.join(sdir, "_annotations.coco.json"), "w") as f:
            json.dump(coco, f)
    print(f"dataset ready at {out_dir}: {len(images)} images, {len(annotations)} boxes, {len(categories)} classes")


def _make_gefen_lightning_cls():
    """Gefen subclass adapted to PyTorch-Lightning's closure-based optimizer step.

    Lightning automatic optimization calls ``optimizer.step(closure=...)`` where
    the closure runs forward+backward *inside* the step. Gefen learns its exact
    codebook at the top of ``step()`` from the current gradients, so the closure
    must run first (to populate grads) before Gefen's step logic. We run the
    closure, then delegate to Gefen with ``closure=None``. (RF-DETR trains in
    bf16-mixed with no GradScaler, so no unscaling step sits between the two.)
    """
    import gefen

    class GefenLightning(gefen.Gefen):
        def step(self, closure=None):
            loss = None
            if closure is not None:
                loss = closure()
            super().step(closure=None)
            return loss

    return GefenLightning


def install_patches(lr):
    import gefen
    from rfdetr.training.module_model import RFDETRModelModule
    from rfdetr.training.callbacks.coco_eval import COCOEvalCallback

    GefenLightning = _make_gefen_lightning_cls()
    orig_cfg = RFDETRModelModule.configure_optimizers

    def patched_cfg(self):
        cfg = orig_cfg(self)                       # RF-DETR's AdamW + LambdaLR
        if not _USE_GEFEN["on"]:
            return cfg
        import torch
        from rfdetr.utilities.logger import get_logger

        adamw = cfg["optimizer"] if isinstance(cfg, dict) else cfg
        # Reuse RF-DETR's exact param grouping — crucially the LAYER-WISE LRs that
        # keep the pretrained DINOv2 backbone at a low LR while the head trains
        # fast. A flat LR over every tensor cooks the pretrained backbone.
        groups = [
            {"params": list(g["params"]), "lr": g["lr"],
             "weight_decay": g.get("weight_decay", 0.0)}
            for g in adamw.param_groups
        ]
        opt = GefenLightning(groups, lr=adamw.param_groups[0]["lr"])
        get_logger(__name__).info(
            f"optimizer: Gefen (fused) over {len(groups)} layer-wise-LR groups")

        sched_cfg = cfg.get("lr_scheduler") if isinstance(cfg, dict) else None
        if sched_cfg is not None:
            old = sched_cfg["scheduler"] if isinstance(sched_cfg, dict) else sched_cfg
            lambdas = getattr(old, "lr_lambdas", None)
            if lambdas is not None and len(lambdas) == len(opt.param_groups):
                new_sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=list(lambdas))
            else:
                one = lambdas[0] if lambdas else (lambda step: 1.0)
                new_sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=one)
            if isinstance(sched_cfg, dict):
                return {"optimizer": opt, "lr_scheduler": {**sched_cfg, "scheduler": new_sched}}
            return {"optimizer": opt, "lr_scheduler": new_sched}
        return {"optimizer": opt}

    RFDETRModelModule.configure_optimizers = patched_cfg

    orig_log = COCOEvalCallback._compute_and_log

    def patched_log(self, trainer, pl_module, split, *, metric=None):
        orig_log(self, trainer, pl_module, split, metric=metric)
        cm = trainer.callback_metrics
        for k in (f"{split}/mAP_50_95", f"{split}/mAP_50"):
            if k in cm:
                _LAST_METRICS[k] = float(cm[k])

    COCOEvalCallback._compute_and_log = patched_log


def run_one(name, args):
    from rfdetr import RFDETRNano

    _USE_GEFEN["on"] = name == "gefen"
    _LAST_METRICS.clear()
    rec = {
        "task": "rfdetr-coco128", "model": "rfdetr-nano", "optimizer": name,
        "epochs": args.epochs, "lr": args.lr, "status": "FAIL", "error": None,
    }
    try:
        model = RFDETRNano(num_classes=PRETRAINED_NUM_CLASSES) if args.preserve_head else RFDETRNano()
        t0 = time.time()
        model.train(
            dataset_dir=args.dataset_dir, epochs=args.epochs, batch_size=args.batch,
            grad_accum_steps=1, lr=args.lr, num_workers=4,
            output_dir=os.path.join(args.project, name), device="cuda",
            tensorboard=False, wandb=False,
        )
        rec["train_s"] = round(time.time() - t0, 1)
        rec["map50_95"] = round(_LAST_METRICS.get("val/mAP_50_95", float("nan")), 4)
        rec["map50"] = round(_LAST_METRICS.get("val/mAP_50", float("nan")), 4)
        rec["status"] = "PASS" if rec["map50_95"] == rec["map50_95"] and rec["map50_95"] > 0.05 else "FAIL"
        print(f"[rfdetr-nano/{name}] mAP50-95={rec['map50_95']}  mAP50={rec['map50']}  ({rec['train_s']}s)")
    except Exception:
        rec["error"] = traceback.format_exc()[-2000:]
        print(rec["error"], file=sys.stderr)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizers", default="adamw,gefen")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dataset-dir", default="./rfdetr-coco")
    ap.add_argument("--make-dataset-from-coco128", default=None)
    ap.add_argument("--keep-classes", default=None,
                    help="comma-separated YOLO class ids to keep (e.g. '0' for person-only); "
                         "concentrates instances so a from-scratch head reaches a real mAP")
    ap.add_argument("--preserve-head", action="store_true",
                    help="use COCO-91 category ids + num_classes=90 so RF-DETR keeps its "
                         "pretrained COCO head and fine-tunes from ~pretrained mAP")
    ap.add_argument("--project", default="./rfdetr-runs")
    ap.add_argument("--out", default="results_rfdetr.jsonl")
    args = ap.parse_args()

    if args.make_dataset_from_coco128:
        keep = None
        if args.keep_classes:
            keep = [int(c) for c in args.keep_classes.split(",") if c.strip() != ""]
        make_dataset_from_coco128(args.make_dataset_from_coco128, args.dataset_dir,
                                  keep_classes=keep, preserve_head=args.preserve_head)

    install_patches(args.lr)
    summary = {}
    for name in [o.strip() for o in args.optimizers.split(",") if o.strip()]:
        rec = run_one(name, args)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if rec["status"] == "PASS":
            summary[name] = f"mAP50-95={rec['map50_95']}  mAP50={rec['map50']}"

    print(f"\n=== RF-DETR Nano on COCO128 ({args.epochs} epochs) ===")
    for name, s in summary.items():
        print(f"  {name:8s}: {s}")


if __name__ == "__main__":
    main()

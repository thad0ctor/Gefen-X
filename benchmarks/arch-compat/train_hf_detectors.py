"""Gefen vs AdamW on real detection — fine-tune HF transformer detectors on COCO128.

Fine-tunes a COCO-pretrained transformers detector (RT-DETR, YOLOS,
Deformable-DETR) on the real COCO128 dataset and reports COCO mAP (torchmetrics),
so each transformer detector is a real training comparison against AdamW. YOLO
boxes are already normalized cxcywh — the DETR target format — and class ids are
mapped into each model's label space (80- or 91-class) so its pretrained head
stays meaningful.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idx> python train_hf_detectors.py \
      --arch rtdetr --optimizers adamw,gefen --epochs 30 --coco128 <dir> \
      [--out results_hfdet.jsonl]
  # --arch all runs every registered detector
"""

import argparse, json, os, sys, time, traceback

import torch

COCO91_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]

# arch -> (ForObjectDetection class, pretrained checkpoint)
ARCHES = {
    "rtdetr": ("RTDetrForObjectDetection", "PekingU/rtdetr_r50vd"),
    "yolos": ("YolosForObjectDetection", "hustvl/yolos-tiny"),
    "deformable_detr": ("DeformableDetrForObjectDetection", "SenseTime/deformable-detr"),
}


def load_coco128(coco128_dir):
    """-> list of (PIL image, boxes_cxcywh_norm tensor, yolo_class list)."""
    from PIL import Image

    img_dir = os.path.join(coco128_dir, "images", "train2017")
    lbl_dir = os.path.join(coco128_dir, "labels", "train2017")
    items = []
    for fname in sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")):
        img = Image.open(os.path.join(img_dir, fname)).convert("RGB")
        boxes, cls = [], []
        lbl = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
        if os.path.exists(lbl):
            for line in open(lbl):
                p = line.split()
                if len(p) >= 5:
                    cls.append(int(float(p[0])))
                    boxes.append([float(x) for x in p[1:5]])  # cxcywh normalized
        items.append((img, torch.tensor(boxes).reshape(-1, 4), cls))
    return items


def build_optimizer(name, model, lr):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if name == "gefen":
        import gefen
        return gefen.Gefen(named, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW([p for _, p in named], lr=lr)
    raise ValueError(name)


def run_one(arch, name, items, args, device):
    import transformers as tf

    torch.manual_seed(0)
    cls_name, ckpt = ARCHES[arch]
    processor = tf.AutoImageProcessor.from_pretrained(ckpt)
    model = getattr(tf, cls_name).from_pretrained(ckpt).to(device)
    n_labels = model.config.num_labels
    # map YOLO contiguous class -> the model's label space
    def to_model_label(c):
        return c if n_labels <= 80 else COCO91_IDS[c]

    def make_batch(batch_items):
        images = [im for im, _, _ in batch_items]
        enc = processor(images=images, return_tensors="pt")
        labels = []
        for _, boxes, cls in batch_items:
            labels.append({
                "class_labels": torch.tensor([to_model_label(c) for c in cls], dtype=torch.int64),
                "boxes": boxes if len(boxes) else torch.zeros((0, 4)),
            })
        # pixel_mask marks valid (non-padding) pixels — required for models with
        # multiscale deformable attention when a batch pads to a common size.
        return enc["pixel_values"], enc.get("pixel_mask"), labels, images

    opt = build_optimizer(name, model, args.lr)
    bs = args.batch
    idx = list(range(len(items)))

    torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        model.train()
        g = torch.Generator().manual_seed(epoch)
        order = torch.randperm(len(idx), generator=g).tolist()
        for i in range(0, len(order) - bs + 1, bs):
            batch = [items[j] for j in order[i:i + bs]]
            pixel_values, pixel_mask, labels, _ = make_batch(batch)
            pixel_values = pixel_values.to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in labels]
            kw = {"pixel_values": pixel_values, "labels": labels}
            if pixel_mask is not None:
                kw["pixel_mask"] = pixel_mask.to(device)
            opt.zero_grad(set_to_none=True)
            loss = model(**kw).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss ({arch}/{name}, epoch {epoch})")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 0.1)
            opt.step()

    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    metric = MeanAveragePrecision(box_format="xyxy")
    model.eval()
    with torch.no_grad():
        for i in range(0, len(items), bs):
            batch = items[i:i + bs]
            pixel_values, pixel_mask, labels, images = make_batch(batch)
            kw = {"pixel_values": pixel_values.to(device)}
            if pixel_mask is not None:
                kw["pixel_mask"] = pixel_mask.to(device)
            outputs = model(**kw)
            sizes = torch.tensor([[im.size[1], im.size[0]] for im in images])
            post = processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=sizes)
            preds = [{"boxes": p["boxes"].cpu(), "scores": p["scores"].cpu(), "labels": p["labels"].cpu()}
                     for p in post]
            tgts = []
            for (im, boxes, cls) in batch:
                W, H = im.size
                if len(boxes):
                    cx, cy, w, h = boxes.unbind(-1)
                    xyxy = torch.stack([(cx - w / 2) * W, (cy - h / 2) * H,
                                        (cx + w / 2) * W, (cy + h / 2) * H], -1)
                else:
                    xyxy = torch.zeros((0, 4))
                tgts.append({"boxes": xyxy,
                             "labels": torch.tensor([to_model_label(c) for c in cls], dtype=torch.int64)})
            metric.update(preds, tgts)
    res = metric.compute()
    peak = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    return float(res["map"]), float(res["map_50"]), peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, help="detector key, or 'all'")
    ap.add_argument("--optimizers", default="adamw,gefen")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--coco128", default="./coco128")
    ap.add_argument("--out", default="results_hfdet.jsonl")
    args = ap.parse_args()

    device = "cuda"
    items = load_coco128(args.coco128)
    arches = list(ARCHES) if args.arch == "all" else [args.arch]
    for arch in arches:
        for name in [o.strip() for o in args.optimizers.split(",") if o.strip()]:
            rec = {
                "task": "hf-detector", "arch": arch, "dataset": "coco128",
                "optimizer": name, "epochs": args.epochs, "lr": args.lr,
                "gpu": torch.cuda.get_device_name(0), "status": "FAIL", "error": None,
            }
            try:
                t0 = time.time()
                m, m50, peak = run_one(arch, name, items, args, device)
                rec["map50_95"] = round(m, 4)
                rec["map50"] = round(m50, 4)
                rec["peak_vram_GiB"] = peak
                rec["train_s"] = round(time.time() - t0, 1)
                rec["status"] = "PASS" if m > 0.05 else "FAIL"
                print(f"[{arch}/{name}] mAP50-95={rec['map50_95']} mAP50={rec['map50']} ({rec['train_s']}s)")
            except Exception:
                rec["error"] = traceback.format_exc()[-2000:]
                print(rec["error"], file=sys.stderr)
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()

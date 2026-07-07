"""Gefen architecture smoke validation for vision models — CNNs and detectors.

Memorization test (same contract as validate_llm.py): a fixed synthetic batch is
trained for N optimizer steps and the loss must fall by >30% with no NaN/Inf.
This exercises the parameter tensor shapes each vision family produces —
Conv2d/Conv1d stacks, depthwise-separable and grouped convs, patch-embed convs,
FPN/detection heads, box-regression and objectness tensors, and transformer
object-query decoders — through Gefen's flattened block partitioning and fused
step. It is an optimizer/architecture check, not a detection-accuracy benchmark.

Models are built with RANDOM initialization (no pretrained download) unless a
setup needs weights; memorizing a fixed batch does not require a trained start.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idx> python validate_vision.py \
      --arch resnet50 --optimizer gefen --steps 60 --lr 1e-3 [--out results.jsonl]
  # --arch all  runs every registered architecture in sequence
"""

import argparse, json, os, sys, time, traceback

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# each setup_* returns (model, closure): closure() -> scalar loss tensor over a
# fixed batch captured in the closure. The harness discovers trainable params,
# builds the optimizer, and drives the memorization loop.
# --------------------------------------------------------------------------


def _cls_setup(builder, img_size=224, in_ch=3, n_classes=10, batch=8):
    """torchvision image classifier: memorize random image -> label."""

    def setup(device):
        model = builder(weights=None, num_classes=n_classes).to(device)
        x = torch.randn(batch, in_ch, img_size, img_size, device=device)
        y = torch.randint(0, n_classes, (batch,), device=device)

        def closure():
            out = model(x)
            out = out.logits if hasattr(out, "logits") else out
            return F.cross_entropy(out, y)

        return model, closure

    return setup


def _tv_detection_setup(builder, img_size=320, batch=2, n_boxes=3, n_classes=5):
    """torchvision detector: sum of the training loss dict on a fixed batch."""

    def setup(device):
        model = builder(
            weights=None, weights_backbone=None, num_classes=n_classes
        ).to(device)
        model.train()
        torch.manual_seed(0)
        images = [torch.rand(3, img_size, img_size, device=device) for _ in range(batch)]
        targets = []
        for _ in range(batch):
            # random but valid xyxy boxes inside the image
            x0 = torch.randint(0, img_size // 2, (n_boxes,))
            y0 = torch.randint(0, img_size // 2, (n_boxes,))
            w = torch.randint(10, img_size // 2, (n_boxes,))
            h = torch.randint(10, img_size // 2, (n_boxes,))
            boxes = torch.stack([x0, y0, x0 + w, y0 + h], 1).float().to(device)
            labels = torch.randint(1, n_classes, (n_boxes,), device=device)
            t = {"boxes": boxes, "labels": labels}
            if "maskrcnn" in getattr(builder, "__name__", ""):
                t["masks"] = (torch.rand(n_boxes, img_size, img_size, device=device) > 0.5).to(torch.uint8)
            targets.append(t)

        def closure():
            loss_dict = model(images, targets)
            return sum(loss_dict.values())

        return model, closure

    return setup


def _hf_detr_setup(model_cls_name, img_size=224, batch=2, n_boxes=3, n_labels=5):
    """transformers object detector built from a small random config."""

    def setup(device):
        import transformers as tf

        model_cls = getattr(tf, model_cls_name)
        cfg_cls = model_cls.config_class
        cfg = cfg_cls()
        # shrink to keep the smoke cheap where the config allows it. Covers both
        # DETR-style names (d_model / encoder_layers) and ViT-style names YOLOS
        # uses (hidden_size / num_hidden_layers) so every family runs small.
        for attr, val in (("num_labels", n_labels), ("d_model", 128),
                          ("encoder_layers", 2), ("decoder_layers", 2),
                          ("num_queries", 20), ("dim_feedforward", 256),
                          ("encoder_ffn_dim", 256), ("decoder_ffn_dim", 256),
                          ("hidden_size", 128), ("num_hidden_layers", 2),
                          ("num_attention_heads", 4), ("intermediate_size", 256),
                          ("image_size", [img_size, img_size])):
            if hasattr(cfg, attr):
                try:
                    setattr(cfg, attr, val)
                except Exception:
                    pass  # strict config rejects some shrinks; leave the default
        model = model_cls(cfg).to(device)
        model.train()
        torch.manual_seed(0)
        pixel_values = torch.randn(batch, 3, img_size, img_size, device=device)
        labels = []
        for _ in range(batch):
            boxes = torch.rand(n_boxes, 4, device=device) * 0.5 + 0.25  # cxcywh in [0.25,0.75]
            class_labels = torch.randint(0, n_labels, (n_boxes,), device=device)
            labels.append({"class_labels": class_labels, "boxes": boxes})

        def closure():
            return model(pixel_values=pixel_values, labels=labels).loss

        return model, closure

    return setup


def _ultralytics_yolo_setup(cfg="yolo11n.yaml", img_size=320, batch=2, n_boxes=3, n_classes=8):
    """ultralytics YOLO DetectionModel: loss on a fixed batch dict."""

    def setup(device):
        from ultralytics.nn.tasks import DetectionModel

        model = DetectionModel(cfg, nc=n_classes, verbose=False).to(device).float()
        model.train()
        model.args = _yolo_hyp()
        torch.manual_seed(0)
        img = torch.rand(batch, 3, img_size, img_size, device=device)
        idx, cls, boxes = [], [], []
        for b in range(batch):
            for _ in range(n_boxes):
                idx.append(b)
                cls.append(float(torch.randint(0, n_classes, (1,)).item()))
                # xywh normalized, kept away from the border
                boxes.append(torch.rand(4) * 0.4 + 0.3)
        batch_dict = {
            "img": img,
            "batch_idx": torch.tensor(idx, device=device).float(),
            "cls": torch.tensor(cls, device=device).view(-1, 1),
            "bboxes": torch.stack(boxes).to(device),
        }

        def closure():
            out = model(batch_dict)
            loss = out[0] if isinstance(out, (tuple, list)) else out
            return loss.sum()

        return model, closure

    return setup


def _yolo_hyp():
    from types import SimpleNamespace

    return SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)


def _rfdetr_setup(batch=2, n_boxes=3, n_labels=5):
    """RF-DETR (roboflow) LWDETR network + SetCriterion on a fixed batch."""

    def setup(device):
        from rfdetr import RFDETRNano
        from rfdetr.config import TrainConfig
        from rfdetr.models.lwdetr import build_criterion_from_config

        wrapper = RFDETRNano()
        net = wrapper.model.model.to(device)   # the LWDETR nn.Module
        mc = wrapper.get_model_config()
        criterion, _ = build_criterion_from_config(
            mc, TrainConfig(dataset_dir=".", output_dir=".")
        )
        criterion = criterion.to(device)
        net.train()
        img_size = getattr(mc, "resolution", 384)
        torch.manual_seed(0)
        samples = torch.randn(batch, 3, img_size, img_size, device=device)
        targets = []
        for _ in range(batch):
            boxes = torch.rand(n_boxes, 4, device=device) * 0.4 + 0.3  # cxcywh
            labels = torch.randint(0, n_labels, (n_boxes,), device=device)
            targets.append({"boxes": boxes, "labels": labels})

        def closure():
            outputs = net(samples, targets)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            return sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)

        return net, closure

    return setup


def _tv(name):
    import torchvision.models as m

    return getattr(m, name)


def _tvdet(name):
    import torchvision.models.detection as d

    return getattr(d, name)


ARCHES = {
    # --- classifiers (Conv2d / depthwise / grouped / patch-embed) ---
    "resnet50": lambda: _cls_setup(_tv("resnet50")),
    "convnext_tiny": lambda: _cls_setup(_tv("convnext_tiny")),
    "efficientnet_b0": lambda: _cls_setup(_tv("efficientnet_b0")),
    "mobilenet_v3_large": lambda: _cls_setup(_tv("mobilenet_v3_large")),
    "densenet121": lambda: _cls_setup(_tv("densenet121")),
    "regnet_y_1_6gf": lambda: _cls_setup(_tv("regnet_y_1_6gf")),
    "vit_b_16": lambda: _cls_setup(_tv("vit_b_16")),
    "swin_t": lambda: _cls_setup(_tv("swin_t")),
    # --- convolutional detectors (backbone + FPN + dense/2-stage heads) ---
    "fasterrcnn_resnet50_fpn": lambda: _tv_detection_setup(_tvdet("fasterrcnn_resnet50_fpn")),
    "retinanet_resnet50_fpn": lambda: _tv_detection_setup(_tvdet("retinanet_resnet50_fpn")),
    "ssd300_vgg16": lambda: _tv_detection_setup(_tvdet("ssd300_vgg16"), img_size=300),
    "fcos_resnet50_fpn": lambda: _tv_detection_setup(_tvdet("fcos_resnet50_fpn")),
    "yolo11n": lambda: _ultralytics_yolo_setup(),
    # --- transformer detectors (object-query decoders; RF-DETR family) ---
    "rtdetr": lambda: _hf_detr_setup("RTDetrForObjectDetection"),
    "yolos": lambda: _hf_detr_setup("YolosForObjectDetection"),
    "deformable_detr": lambda: _hf_detr_setup("DeformableDetrForObjectDetection"),
    "rfdetr": lambda: _rfdetr_setup(),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, help="architecture key, or 'all'")
    ap.add_argument("--optimizer", default="gefen", choices=["gefen", "hybrid", "adamw"])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip-grad-norm", type=float, default=None)
    ap.add_argument("--out", default="results_vision.jsonl")
    args = ap.parse_args()

    arches = list(ARCHES) if args.arch == "all" else [args.arch]
    any_fail = False
    for arch in arches:
        rec = run_one(arch, args)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec, indent=2))
        any_fail = any_fail or rec["status"] != "PASS"
    sys.exit(1 if any_fail else 0)


def run_one(arch, args):
    rec = {
        "task": "vision",
        "arch": arch,
        "optimizer": args.optimizer,
        "steps": args.steps,
        "lr": args.lr,
        "status": "FAIL",
        "error": None,
    }
    try:
        device = "cuda"
        torch.manual_seed(0)
        setup = ARCHES[arch]()
        model, closure = setup(device)
        rec["params_total_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)

        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        rec["params_trained_M"] = round(sum(p.numel() for _, p in named) / 1e6, 2)
        rec["param_ndims"] = sorted({p.ndim for _, p in named})

        import gefen

        if args.optimizer == "gefen":
            opt = gefen.Gefen(named, lr=args.lr)
        elif args.optimizer == "hybrid":
            muon_named, backup_named = gefen.split_params_for_muon(model)
            opt = gefen.GefenMuonHybrid(muon_named, backup_named, lr=args.lr)
        else:
            opt = torch.optim.AdamW([p for _, p in named], lr=args.lr)

        torch.cuda.reset_peak_memory_stats()
        losses = []
        t0 = time.time()
        for step in range(args.steps):
            loss = closure()
            if not torch.isfinite(loss):
                rec["losses"] = losses
                raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
            loss.backward()
            if args.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_([p for _, p in named], args.clip_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(round(loss.item(), 4))
        rec["train_s"] = round(time.time() - t0, 1)
        rec["losses"] = losses[:3] + ["..."] + losses[-3:] if len(losses) > 8 else losses
        rec["loss_first"], rec["loss_last"] = losses[0], losses[-1]
        rec["peak_vram_GiB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        drop = (losses[0] - losses[-1]) / max(abs(losses[0]), 1e-9)
        rec["loss_drop_pct"] = round(100 * drop, 1)
        rec["status"] = "PASS" if drop > 0.3 else "FAIL"
    except Exception:
        rec["error"] = traceback.format_exc()[-2000:]
        print(rec["error"], file=sys.stderr)
    return rec


if __name__ == "__main__":
    main()

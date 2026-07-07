"""Gefen vs AdamW on real YOLO detection — ultralytics YOLO11n on COCO128.

Fine-tunes the pretrained YOLO11n on the real COCO128 dataset with ultralytics'
own training loop and reports mAP, so it is a real object-detection training
comparison (not a memorization smoke). Gefen is injected by monkeypatching the
trainer's `build_optimizer`; both optimizers start from the same pretrained
weights, same lr0/schedule/epochs, for a fair comparison.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idx> python train_yolo_coco.py \
      --optimizers adamw,gefen --epochs 40 --lr0 1e-3 [--out results_yolo.jsonl]
"""

import argparse, json, os, sys, time, traceback

_USE_GEFEN = {"on": False, "lr": 1e-3}


def _install_gefen_patch():
    import torch.nn as nn
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.utils import LOGGER
    from ultralytics.utils.torch_utils import unwrap_model

    orig = BaseTrainer.build_optimizer

    def patched(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        if _USE_GEFEN["on"]:
            import gefen
            eff_lr = lr if lr else _USE_GEFEN["lr"]
            # Mirror ultralytics' weight-decay grouping so the comparison is fair:
            # 2-D+ weights get `decay`, biases and norm params get none.
            bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
            decay_params, nodecay_params = [], []
            for _, module in unwrap_model(model).named_modules():
                for pn, p in module.named_parameters(recurse=False):
                    if not p.requires_grad:
                        continue
                    if p.ndim >= 2 and not isinstance(module, bn):
                        decay_params.append(p)
                    else:
                        nodecay_params.append(p)
            groups = [
                {"params": decay_params, "lr": eff_lr, "weight_decay": decay},
                {"params": nodecay_params, "lr": eff_lr, "weight_decay": 0.0},
            ]
            n = len(decay_params) + len(nodecay_params)
            LOGGER.info(f"optimizer: Gefen (fused) over {n} params, lr={eff_lr}, decay={decay}")
            return gefen.Gefen(groups, lr=eff_lr)
        return orig(self, model, name, lr, momentum, decay, iterations)

    BaseTrainer.build_optimizer = patched


def run_one(name, args, project):
    from ultralytics import YOLO

    _USE_GEFEN["on"] = name == "gefen"
    _USE_GEFEN["lr"] = args.lr0
    rec = {
        "task": "yolo-coco128", "model": "yolo11n", "optimizer": name,
        "epochs": args.epochs, "lr0": args.lr0, "status": "FAIL", "error": None,
    }
    try:
        model = YOLO("yolo11n.pt")
        t0 = time.time()
        model.train(
            data="coco128.yaml", epochs=args.epochs, imgsz=args.imgsz,
            batch=args.batch, optimizer="AdamW",  # name ignored when Gefen patch is on
            lr0=args.lr0, lrf=0.01, warmup_epochs=1.0, seed=0, deterministic=False,
            project=project, name=name, exist_ok=True, verbose=False, plots=False,
            val=True, save=False,
        )
        metrics = model.val(data="coco128.yaml", imgsz=args.imgsz, verbose=False,
                            project=project, name=f"{name}_val", exist_ok=True)
        rec["train_s"] = round(time.time() - t0, 1)
        rec["map50_95"] = round(float(metrics.box.map), 4)
        rec["map50"] = round(float(metrics.box.map50), 4)
        rec["status"] = "PASS" if rec["map50_95"] > 0.10 else "FAIL"
        print(f"[yolo11n/{name}] mAP50-95={rec['map50_95']}  mAP50={rec['map50']}  ({rec['train_s']}s)")
    except Exception:
        rec["error"] = traceback.format_exc()[-2000:]
        print(rec["error"], file=sys.stderr)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizers", default="adamw,gefen")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr0", type=float, default=1e-3)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--project", default="./yolo-runs")
    ap.add_argument("--out", default="results_yolo.jsonl")
    args = ap.parse_args()

    _install_gefen_patch()
    requested = [o.strip() for o in args.optimizers.split(",") if o.strip()]
    invalid = set(requested) - {"adamw", "gefen"}
    if invalid:
        raise SystemExit(f"unknown optimizer(s): {sorted(invalid)} (expected adamw/gefen)")
    summary = {}
    for name in requested:
        rec = run_one(name, args, args.project)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if rec["status"] == "PASS":
            summary[name] = f"mAP50-95={rec['map50_95']}  mAP50={rec['map50']}"

    print(f"\n=== YOLO11n on COCO128 ({args.epochs} epochs) ===")
    for name, s in summary.items():
        print(f"  {name:8s}: {s}")


if __name__ == "__main__":
    main()

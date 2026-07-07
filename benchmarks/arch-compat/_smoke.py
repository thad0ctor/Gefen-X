"""Shared scaffold for the fixed-batch architecture smoke validators.

`validate_vision.py` and `validate_audio.py` only differ in their per-architecture
`setup(device) -> (model, closure)` builders (registered in an `ARCHES` dict);
the record shape, optimizer construction, memorization loop, PASS/FAIL loss-drop
criterion, and JSONL output are identical and live here so a fix lands in one
place instead of drifting across copies.
"""

import argparse
import json
import sys
import time
import traceback

import torch


def run_smoke(task, arch, arches, args):
    """Memorize a fixed batch for one architecture and return a result record."""
    rec = {
        "task": task,
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
        setup = arches[arch]()
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


def main_smoke(task, arches, default_out):
    """CLI entry point: run one architecture or `--arch all`, append JSONL records."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, help="architecture key, or 'all'")
    ap.add_argument("--optimizer", default="gefen", choices=["gefen", "hybrid", "adamw"])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip-grad-norm", type=float, default=None)
    ap.add_argument("--out", default=default_out)
    args = ap.parse_args()

    selected = list(arches) if args.arch == "all" else [args.arch]
    any_fail = False
    for arch in selected:
        rec = run_smoke(task, arch, arches, args)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec, indent=2))
        any_fail = any_fail or rec["status"] != "PASS"
    sys.exit(1 if any_fail else 0)

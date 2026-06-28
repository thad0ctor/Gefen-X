"""Auto-determine the optimal base learning rate for a Gefen-family optimizer.

The recommended way to set Gefen's LR: don't hand-pick a "~0.6-0.8x your AdamW
LR" factor -- run this finder on a few real batches of YOUR model/data and use
the number it returns. Works for plain ``Gefen``, ``GefenMuon``,
``GefenMuonHybrid``, and ``torch.optim.AdamW`` (so you can find each one's LR).

Two methods:

  * ``method="range_test"`` (fast, default) -- one exponential LR ramp over a few
    hundred steps; the loss-vs-LR curve's steepest-descent point is the suggested
    base LR (Leslie Smith / fastai ``lr_find``, built on the noise-robust
    ``tools.lr_calibration.lr_range_test``). ~1 short run.

  * ``method="sweep"`` (precise) -- short real fine-tunes at a few LRs, pick the
    lowest HELD-OUT eval loss (the gold standard). Needs eval data; ~N short runs.

Both are non-destructive: the model's weights are snapshotted and restored, so the
finder leaves the model exactly as it found it (``restore=False`` to disable).

Programmatic:

    from tools.find_lr import find_lr
    res = find_lr(model, train_blocks, optimizer="hybrid",
                  eval_blocks=eval_blocks, method="sweep")
    print(res.lr)               # recommended base LR
    optimizer = ...(lr=res.lr)  # use it

CLI: see ``python -m gefen.tools.find_lr --help`` (model + dataset paths via args).
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

import torch

from gefen import Gefen, GefenMuon, GefenMuonHybrid
from gefen.tools.lr_calibration import lr_range_test, LRRangeResult
from gefen.tools.run_model_calibration import split_params_for_muon


OPTIMIZERS = ("gefen", "gefen_muon", "hybrid", "adamw")


@dataclass
class FindLRResult:
    lr: float                      # recommended base LR
    optimizer: str
    method: str
    min_loss_lr: Optional[float] = None       # range_test: too-hot upper bound
    diverged_at: Optional[float] = None       # range_test
    range_result: Optional[LRRangeResult] = None
    sweep: Dict[float, float] = field(default_factory=dict)  # lr -> eval loss

    def summary(self) -> str:
        lines = [f"find_lr[{self.optimizer}/{self.method}] -> recommended LR = {self.lr:.3e}"]
        if self.method == "range_test":
            if self.min_loss_lr is not None:
                lines.append(f"  min-loss LR (too hot): {self.min_loss_lr:.3e}")
            if self.diverged_at is not None:
                lines.append(f"  diverged at ~ {self.diverged_at:.3e}")
        else:
            for lr in sorted(self.sweep):
                mark = "  <- best" if lr == self.lr else ""
                lines.append(f"    {lr:.2e}  eval {self.sweep[lr]:.4f}{mark}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Optimizer construction (one place, so every family is found consistently)
# --------------------------------------------------------------------------- #
def build_optimizer(
    model,
    optimizer: str,
    lr: float,
    *,
    weight_decay: float = 0.0,
    fused: bool = True,
    adjust_lr_fn: str = "match_rms_adamw",
):
    """Construct the requested optimizer family at ``lr``. ``adjust_lr_fn`` applies
    to the Muon paths only (default ``match_rms_adamw`` -- the magnitude-matching
    v2/auto path is intentionally NOT used here)."""
    if optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if optimizer == "gefen":
        return Gefen(named, lr=lr, betas=(0.9, 0.999), eps=1e-8,
                     weight_decay=weight_decay, fused=fused)
    if optimizer == "hybrid":
        muon, backup = split_params_for_muon(model)
        return GefenMuonHybrid(muon, backup, lr=lr, weight_decay=weight_decay,
                               fused=fused, adjust_lr_fn=adjust_lr_fn)
    if optimizer == "gefen_muon":
        # GefenMuon only accepts 2D hidden matrices; the rest of the model is left
        # frozen for the duration of the finder (the Muon path's stability/optimum
        # still drives the curve). Use the hybrid to also train embeddings/norms.
        muon, _ = split_params_for_muon(model)
        if not muon:
            raise ValueError("gefen_muon: model has no 2D hidden matrices to optimize")
        return GefenMuon(muon, lr=lr, weight_decay=weight_decay, fused=fused,
                         adjust_lr_fn=adjust_lr_fn)
    raise ValueError(f"unknown optimizer {optimizer!r}; choose from {OPTIMIZERS}")


# --------------------------------------------------------------------------- #
# Data adapters
# --------------------------------------------------------------------------- #
def _closure_from_blocks(model, blocks, device, bs, seed):
    g = torch.Generator().manual_seed(seed)
    st = {"order": torch.randperm(blocks.size(0), generator=g), "ptr": 0}

    def closure():
        n = blocks.size(0)
        if st["ptr"] + bs > n:
            st["order"] = torch.randperm(n, generator=g)
            st["ptr"] = 0
        idx = st["order"][st["ptr"]:st["ptr"] + bs]
        st["ptr"] += bs
        ids = blocks[idx].to(device)
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        return out.loss.detach()

    return closure


@torch.no_grad()
def _eval_loss(model, eval_blocks, device, bs):
    model.eval()
    losses = []
    for i in range(0, eval_blocks.size(0), bs):
        ids = eval_blocks[i:i + bs].to(device)
        losses.append(float(model(input_ids=ids, labels=ids).loss))
    model.train()
    return sum(losses) / len(losses)


def _snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def find_lr(
    model,
    train_data: Union[torch.Tensor, Callable[[], torch.Tensor]],
    *,
    optimizer: str = "gefen",
    method: str = "range_test",
    eval_blocks: Optional[torch.Tensor] = None,
    device: Optional[Union[str, torch.device]] = None,
    weight_decay: float = 0.0,
    fused: bool = True,
    adjust_lr_fn: str = "match_rms_adamw",
    restore: bool = True,
    # range-test knobs
    num_iter: int = 100,
    start_lr: float = 1e-6,
    end_lr: float = 1e-1,
    smooth_beta: float = 0.9,
    diverge_factor: float = 5.0,
    # sweep knobs
    sweep_lrs: Optional[List[float]] = None,
    sweep_steps: int = 80,
    warmup: int = 10,
    bs: int = 8,
    seed: int = 0,
    verbose: bool = True,
) -> FindLRResult:
    """Recommend a base LR for ``optimizer`` on ``model`` + a few real batches.

    ``train_data`` is either a ``[N, seq]`` LongTensor of token blocks (a cycling
    closure is built from it) or a ready-made closure (forward + ``backward``,
    returns loss). ``eval_blocks`` (``[N, seq]``) is required for ``method="sweep"``.
    Returns a :class:`FindLRResult` whose ``.lr`` is the recommendation.
    """
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
    if method not in ("range_test", "sweep"):
        raise ValueError("method must be 'range_test' or 'sweep'")
    if device is None:
        device = next(model.parameters()).device
    model.train()
    snap = _snapshot(model) if restore else None

    try:
        if method == "range_test":
            closure = (
                train_data if callable(train_data)
                else _closure_from_blocks(model, train_data, device, bs, seed)
            )
            opt = build_optimizer(model, optimizer, 1e-4, weight_decay=weight_decay,
                                  fused=fused, adjust_lr_fn=adjust_lr_fn)
            res = lr_range_test(opt, closure, num_iter=num_iter, start_lr=start_lr,
                                end_lr=end_lr, smooth_beta=smooth_beta,
                                diverge_factor=diverge_factor)
            out = FindLRResult(
                lr=res.suggested_lr, optimizer=optimizer, method=method,
                min_loss_lr=res.min_loss_lr, diverged_at=res.diverged_at,
                range_result=res,
            )
        else:  # sweep
            if eval_blocks is None:
                raise ValueError("method='sweep' requires eval_blocks")
            if callable(train_data):
                raise ValueError("method='sweep' needs train_data as a [N, seq] tensor")
            if snap is None:
                snap = _snapshot(model)  # sweep needs to reset between LRs
            grid = sweep_lrs or [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
            sweep: Dict[float, float] = {}
            for lr in grid:
                model.load_state_dict(snap)
                opt = build_optimizer(model, optimizer, lr, weight_decay=weight_decay,
                                      fused=fused, adjust_lr_fn=adjust_lr_fn)
                _short_finetune(model, train_data, opt, device, steps=sweep_steps,
                                warmup=warmup, bs=bs, seed=seed)
                sweep[lr] = _eval_loss(model, eval_blocks, device, bs)
                if verbose:
                    print(f"    [{optimizer}] lr {lr:.2e} -> eval {sweep[lr]:.4f}")
                del opt
                torch.cuda.empty_cache()
            best = min(sweep, key=sweep.get)
            out = FindLRResult(lr=best, optimizer=optimizer, method=method, sweep=sweep)
    finally:
        if restore and snap is not None:
            model.load_state_dict(snap)

    if verbose:
        print(out.summary())
    return out


def _short_finetune(model, train_blocks, optimizer, device, *, steps, warmup, bs, seed):
    import math
    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    g = torch.Generator().manual_seed(seed + 1)
    n = train_blocks.size(0)
    order = torch.randperm(n, generator=g)
    ptr = 0
    model.train()
    for _ in range(steps):
        if ptr + bs > n:
            order = torch.randperm(n, generator=g)
            ptr = 0
        ids = train_blocks[order[ptr:ptr + bs]].to(device)
        ptr += bs
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        optimizer.step()
        sched.step()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    from gefen.tools.run_model_calibration import load_model
    from gefen.tools.train_real_validation import build_token_stream

    ap = argparse.ArgumentParser(description="Auto-find a base LR for a Gefen-family optimizer.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--optimizer", choices=OPTIMIZERS, default="gefen")
    ap.add_argument("--method", choices=["range_test", "sweep"], default="range_test")
    ap.add_argument("--dataset", default="tatsu-lab/alpaca")
    ap.add_argument("--split", default="train")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--eval-blocks", type=int, default=48)
    # range-test
    ap.add_argument("--num-iter", type=int, default=100)
    ap.add_argument("--start-lr", type=float, default=1e-6)
    ap.add_argument("--end-lr", type=float, default=1e-1)
    # sweep
    ap.add_argument("--sweep-lrs", type=float, nargs="+", default=None)
    ap.add_argument("--sweep-steps", type=int, default=80)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    tok, model = load_model(args.model, args.device, getattr(torch, args.dtype))
    train_blocks, eval_blocks = build_token_stream(tok, args)
    res = find_lr(
        model, train_blocks, optimizer=args.optimizer, method=args.method,
        eval_blocks=eval_blocks, device=args.device, weight_decay=args.weight_decay,
        num_iter=args.num_iter, start_lr=args.start_lr, end_lr=args.end_lr,
        sweep_lrs=args.sweep_lrs, sweep_steps=args.sweep_steps, warmup=args.warmup,
        bs=args.bs, seed=args.seed,
    )
    print(f"\nRECOMMENDED base LR for {args.optimizer}: {res.lr:.3e}")


if __name__ == "__main__":
    main()

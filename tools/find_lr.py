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
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

import torch

# NOTE: the gefen OPTIMIZER imports (gefen.gefen / the CUDA kernel-build guard,
# which needs an nvcc matching torch.version.cuda) are deferred to the functions
# that build optimizers, so the CLI can re-exec under a target venv
# (--venv/--python) BEFORE that build runs. ``gefen.tools.lr_calibration`` is
# imported eagerly below ON PURPOSE: it pulls in only ``torch`` (no kernel build),
# and ``LRRangeResult`` must exist in module globals so the FindLRResult
# annotation resolves at runtime (e.g. typing.get_type_hints), not just under
# TYPE_CHECKING.
from gefen.tools.lr_calibration import LRRangeResult, lr_range_test


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
    v2/auto path is intentionally NOT used here).

    The gefen imports live INSIDE each branch so the ``adamw`` path never imports
    gefen (hence never triggers the CUDA kernel build) -- you can find an AdamW LR
    on a box without a matching nvcc."""
    if optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if optimizer == "gefen":
        from gefen import Gefen
        return Gefen(named, lr=lr, betas=(0.9, 0.999), eps=1e-8,
                     weight_decay=weight_decay, fused=fused)
    if optimizer == "hybrid":
        from gefen import GefenMuonHybrid
        from gefen.tools.run_model_calibration import split_params_for_muon
        muon, backup = split_params_for_muon(model)
        return GefenMuonHybrid(muon, backup, lr=lr, weight_decay=weight_decay,
                               fused=fused, adjust_lr_fn=adjust_lr_fn)
    if optimizer == "gefen_muon":
        # GefenMuon only accepts 2D hidden matrices; the rest of the model is left
        # frozen for the duration of the finder (the Muon path's stability/optimum
        # still drives the curve). Use the hybrid to also train embeddings/norms.
        from gefen import GefenMuon
        from gefen.tools.run_model_calibration import split_params_for_muon
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
    was_training = model.training
    model.eval()
    try:
        losses = []
        for i in range(0, eval_blocks.size(0), bs):
            ids = eval_blocks[i:i + bs].to(device)
            losses.append(float(model(input_ids=ids, labels=ids).loss))
        return sum(losses) / len(losses)
    finally:
        model.train(was_training)


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
    was_training = model.training  # restore the caller's mode on exit
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
        model.train(was_training)

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
# Parallel multi-GPU sweep (each LR arm is independent -> fan across GPUs)
# --------------------------------------------------------------------------- #
def distribute_lrs(lrs: List[float], devices: List[str]) -> Dict[str, List[float]]:
    """Round-robin assign LR arms to devices (pure / unit-testable). Device i gets
    ``lrs[i], lrs[i+len(devices)], ...`` so the grid is spread as evenly as
    possible and the same selection is reproducible regardless of #GPUs."""
    if not devices:
        raise ValueError("no devices given")
    assignment: Dict[str, List[float]] = {d: [] for d in devices}
    for i, lr in enumerate(lrs):
        assignment[devices[i % len(devices)]].append(lr)
    # Drop devices that got no LRs (more GPUs than grid points).
    return {d: ls for d, ls in assignment.items() if ls}


def _prewarm_kernels():
    """Trigger the (cached) CUDA kernel build once in the parent so concurrent
    spawn workers load the compiled .so instead of all racing to compile it."""
    try:
        import gefen.gefen  # noqa: F401  -- import builds/loads the fused kernels
    except Exception as exc:  # noqa: BLE001 -- best-effort warm-up, never fatal
        # torch's extension lock makes concurrent builds safe even if this no-ops;
        # the workers will (re)build on their own first import.
        print(f"[find_lr] kernel pre-warm skipped ({type(exc).__name__}: {exc})",
              flush=True)


def _sweep_worker(device: str, lr_list: List[float], cfg: dict, q) -> None:
    """Spawned worker: load the model FROM PATH on ``device`` (the in-memory model
    can't cross processes), run a short fine-tune + held-out eval at each assigned
    LR, and push ``("RESULT", lr, eval)`` per LR, then ``("DONE", device, None)``.
    Any exception is reported as ``("ERROR", device, traceback)`` so the parent
    never hangs on a dead worker."""
    import traceback
    import types

    try:
        import torch as _torch
        from gefen.tools.run_model_calibration import load_model

        dtype = getattr(_torch, cfg["dtype"])
        tok, model = load_model(cfg["model_path"], device, dtype)
        ds_args = types.SimpleNamespace(
            seq=cfg["seq"], dataset=cfg["dataset"], split=cfg["split"],
            max_rows=cfg["max_rows"], eval_blocks=cfg["eval_blocks"],
        )
        train_blocks, eval_blocks = build_token_stream(tok, ds_args)
        snap = _snapshot(model)
        for lr in lr_list:
            model.load_state_dict(snap)
            opt = build_optimizer(
                model, cfg["optimizer"], lr, weight_decay=cfg["weight_decay"],
                fused=cfg["fused"], adjust_lr_fn=cfg["adjust_lr_fn"],
            )
            _short_finetune(model, train_blocks, opt, device,
                            steps=cfg["sweep_steps"], warmup=cfg["warmup"],
                            bs=cfg["bs"], seed=cfg["seed"])
            ev = _eval_loss(model, eval_blocks, device, cfg["bs"])
            print(f"[find_lr] lr {lr:.2e} on {device} -> eval {ev:.4f}", flush=True)
            q.put(("RESULT", lr, ev))
            del opt
            _torch.cuda.empty_cache()
        q.put(("DONE", device, None))
    except Exception:  # noqa: BLE001 -- surface to the parent, don't hang
        q.put(("ERROR", device, traceback.format_exc()))


def find_lr_parallel_sweep(
    model_path: str,
    devices: List[str],
    *,
    optimizer: str = "gefen",
    sweep_lrs: Optional[List[float]] = None,
    dtype: str = "bfloat16",
    dataset: str = "tatsu-lab/alpaca",
    split: str = "train",
    seq: int = 512,
    max_rows: int = 0,
    eval_blocks: int = 48,
    weight_decay: float = 0.0,
    fused: bool = True,
    adjust_lr_fn: str = "match_rms_adamw",
    sweep_steps: int = 80,
    warmup: int = 10,
    bs: int = 8,
    seed: int = 0,
    verbose: bool = True,
    worker_timeout: float = 3600.0,
) -> FindLRResult:
    """CLI-level parallel ``method="sweep"``: fan the LR grid across ``devices``
    (one spawn worker per device, each loading the model from ``model_path`` on its
    GPU). Selection is identical to the sequential path (lowest held-out eval).

    Single device -> just delegates to the sequential in-process path.
    """
    import torch.multiprocessing as mp

    grid = list(sweep_lrs or [1e-5, 3e-5, 1e-4, 3e-4, 1e-3])
    assignment = distribute_lrs(grid, list(devices))
    cfg = dict(
        model_path=model_path, dtype=dtype, optimizer=optimizer,
        weight_decay=weight_decay, fused=fused, adjust_lr_fn=adjust_lr_fn,
        dataset=dataset, split=split, seq=seq, max_rows=max_rows,
        eval_blocks=eval_blocks, sweep_steps=sweep_steps, warmup=warmup,
        bs=bs, seed=seed,
    )

    _prewarm_kernels()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for dev, lrs in assignment.items():
        p = ctx.Process(target=_sweep_worker, args=(dev, lrs, cfg, q))
        p.start()
        procs.append(p)
        if verbose:
            print(f"[find_lr] device {dev} <- LRs {['%.1e' % x for x in lrs]}", flush=True)

    import queue as _pyqueue
    import time as _time
    results: Dict[float, float] = {}
    errors: List = []
    done = 0
    deadline = _time.monotonic() + worker_timeout  # OVERALL deadline, not per-get
    timed_out = False
    while done < len(procs):
        remaining = deadline - _time.monotonic()
        if remaining <= 0:
            timed_out = True
            break  # a worker is alive but silent past the deadline -> give up
        try:
            tag, a, b = q.get(timeout=min(5.0, remaining))
        except _pyqueue.Empty:
            if not any(p.is_alive() for p in procs):
                break  # all workers died without signalling -> stop waiting
            continue
        if tag == "RESULT":
            results[a] = b
        elif tag == "DONE":
            done += 1
        elif tag == "ERROR":
            errors.append((a, b))
            done += 1
    for p in procs:
        p.join(timeout=15)
        if p.is_alive():
            p.terminate()

    if timed_out:
        raise RuntimeError(
            f"parallel sweep exceeded the {worker_timeout:.0f}s deadline with a "
            f"worker still alive but silent; terminated workers."
        )
    if errors:
        dev, tb = errors[0]
        raise RuntimeError(f"parallel sweep worker on {dev} failed:\n{tb}")
    missing = [lr for lr in grid if lr not in results]
    if missing:
        raise RuntimeError(f"parallel sweep produced no result for LRs {missing}")

    best = min(results, key=results.get)
    out = FindLRResult(lr=best, optimizer=optimizer, method="sweep", sweep=results)
    if verbose:
        print(out.summary())
    return out


# --------------------------------------------------------------------------- #
# Model-SHARDED sweep (FSDP2 / fully_shard): one model split across GPUs per arm
# --------------------------------------------------------------------------- #
def _decoder_layers(model):
    """Return the model's per-layer ModuleList to wrap with FSDP2 (Llama/Qwen ->
    ``model.model.layers``), or None if not found. Kept tiny + pure so the
    discovery can be unit-tested without a GPU."""
    inner = getattr(model, "model", model)
    for attr in ("layers", "h", "blocks"):
        cand = getattr(inner, attr, None)
        if cand is not None and hasattr(cand, "__len__"):
            return cand
    return None


def _apply_fully_shard(model, mesh, fully_shard):
    """Standard FSDP2 idiom: shard each decoder block, then the whole model."""
    layers = _decoder_layers(model)
    if layers is not None:
        for blk in layers:
            fully_shard(blk, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return layers is not None


def _free_port() -> str:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return str(s.getsockname()[1])


def _shard_worker(rank: int, world: int, lr: float, cfg: dict, port: str, q) -> None:
    """One rank of an FSDP2-sharded fine-tune at a single LR. Each rank holds only
    a 1/world shard of every parameter; forward/backward/eval are collective, so
    every rank runs them in lockstep. rank 0 reports (lr, eval); all ranks report
    their local shard size + peak memory (evidence of true sharding). The process
    group is torn down in ``finally`` so a failed arm can't leak a group or hang."""
    import os as _os
    import sys as _sys
    import traceback

    # Drop any sys.path entry exposing a flat ``gefen.py`` that would shadow the
    # ``gefen/`` package (same guard the FSDP2 parity test uses for spawned ranks).
    _sys.path[:] = [
        p for p in _sys.path
        if not _os.path.isfile(_os.path.join(p or ".", "gefen.py"))
    ]
    try:
        import types
        import torch as _torch
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        _os.environ["MASTER_ADDR"] = "127.0.0.1"
        _os.environ["MASTER_PORT"] = port
        _os.environ["RANK"] = str(rank)
        _os.environ["WORLD_SIZE"] = str(world)
        _torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world)
        try:
            mesh = init_device_mesh("cuda", (world,))
            dev = f"cuda:{rank}"
            from gefen.tools.run_model_calibration import load_model

            dtype = getattr(_torch, cfg["dtype"])
            tok, model = load_model(cfg["model_path"], dev, dtype)
            # FSDP2 + reentrant grad-checkpointing can deadlock; the models we
            # sweep fit without it once sharded, so disable it for the sweep.
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            full_params = sum(p.numel() for p in model.parameters())
            _apply_fully_shard(model, mesh, fully_shard)

            ds_args = types.SimpleNamespace(
                seq=cfg["seq"], dataset=cfg["dataset"], split=cfg["split"],
                max_rows=cfg["max_rows"], eval_blocks=cfg["eval_blocks"],
            )
            train_blocks, eval_blocks = build_token_stream(tok, ds_args)
            opt = build_optimizer(
                model, cfg["optimizer"], lr, weight_decay=cfg["weight_decay"],
                fused=cfg["fused"], adjust_lr_fn=cfg["adjust_lr_fn"],
            )
            _torch.cuda.reset_peak_memory_stats(rank)
            _short_finetune(model, train_blocks, opt, dev, steps=cfg["sweep_steps"],
                            warmup=cfg["warmup"], bs=cfg["bs"], seed=cfg["seed"])
            ev = _eval_loss(model, eval_blocks, dev, cfg["bs"])

            local_params = 0
            for p in model.parameters():
                local_params += p.to_local().numel() if hasattr(p, "to_local") else p.numel()
            peak_mb = _torch.cuda.max_memory_allocated(rank) / (1024 ** 2)
            tag = "RESULT" if rank == 0 else "RANK"
            q.put((tag, rank, lr, float(ev), int(local_params), int(full_params), float(peak_mb)))
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
    except Exception:  # noqa: BLE001 -- surface, never hang
        q.put(("ERROR", rank, traceback.format_exc()))


def find_lr_sharded_sweep(
    model_path: str,
    devices: List[str],
    *,
    optimizer: str = "gefen",
    sweep_lrs: Optional[List[float]] = None,
    dtype: str = "bfloat16",
    dataset: str = "tatsu-lab/alpaca",
    split: str = "train",
    seq: int = 512,
    max_rows: int = 0,
    eval_blocks: int = 48,
    weight_decay: float = 0.0,
    fused: bool = True,
    adjust_lr_fn: str = "match_rms_adamw",
    sweep_steps: int = 80,
    warmup: int = 10,
    bs: int = 8,
    seed: int = 0,
    verbose: bool = True,
    worker_timeout: float = 3600.0,
) -> FindLRResult:
    """Sweep LRs SEQUENTIALLY, each arm running ONE model sharded across
    ``len(devices)`` GPUs via FSDP2 ``fully_shard``. For each LR a fresh nccl
    process group is spun up (one spawn process per GPU), the model is sharded,
    short-fine-tuned + held-out-evaluated, then the group is torn down before the
    next LR. Selection is identical to the other modes (lowest eval).

    ``devices`` is the per-rank visible-device list (e.g. ``["cuda:0","cuda:1"]``);
    restrict the physical GPUs with ``CUDA_VISIBLE_DEVICES`` so rank r -> cuda:r.
    """
    import queue as _pyqueue
    import time as _time
    import torch.multiprocessing as mp

    world = len(devices)
    if world < 2:
        raise ValueError("--shard needs >=2 devices")
    grid = list(sweep_lrs or [1e-5, 3e-5, 1e-4, 3e-4, 1e-3])
    cfg = dict(
        model_path=model_path, dtype=dtype, optimizer=optimizer,
        weight_decay=weight_decay, fused=fused, adjust_lr_fn=adjust_lr_fn,
        dataset=dataset, split=split, seq=seq, max_rows=max_rows,
        eval_blocks=eval_blocks, sweep_steps=sweep_steps, warmup=warmup,
        bs=bs, seed=seed,
    )
    _prewarm_kernels()
    ctx = mp.get_context("spawn")
    results: Dict[float, float] = {}

    for lr in grid:
        q = ctx.Queue()
        port = _free_port()
        procs = [ctx.Process(target=_shard_worker, args=(r, world, lr, cfg, port, q))
                 for r in range(world)]
        for p in procs:
            p.start()

        msgs: List = []
        errors: List = []
        got = 0
        deadline = _time.monotonic() + worker_timeout  # OVERALL deadline per arm
        timed_out = False
        while got < world:
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                timed_out = True
                break  # a rank is alive but silent past the deadline -> give up
            try:
                m = q.get(timeout=min(5.0, remaining))
            except _pyqueue.Empty:
                if not any(p.is_alive() for p in procs):
                    break
                continue
            got += 1
            (errors if m[0] == "ERROR" else msgs).append(m)
        for p in procs:
            p.join(timeout=30)
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)

        if timed_out:
            raise RuntimeError(
                f"sharded sweep arm lr {lr:.2e} exceeded the {worker_timeout:.0f}s "
                f"deadline with a rank still alive but silent; terminated ranks."
            )
        if errors:
            raise RuntimeError(
                f"sharded sweep worker (rank {errors[0][1]}) failed:\n{errors[0][2]}")
        res = [m for m in msgs if m[0] == "RESULT"]
        if not res:
            raise RuntimeError(f"sharded sweep produced no rank-0 result for lr {lr:.2e}")
        _, _, _, ev, lp0, full, peak0 = res[0]
        results[lr] = ev
        if verbose:
            frac = lp0 / full if full else float("nan")
            print(f"[find_lr][shard] lr {lr:.2e} -> eval {ev:.4f}  "
                  f"(rank0 shard {lp0:,}/{full:,} params = {frac:.2f}x, peak {peak0:.0f} MiB)",
                  flush=True)
            for m in sorted((m for m in msgs if m[0] == "RANK"), key=lambda x: x[1]):
                print(f"                 rank{m[1]} shard {m[4]:,} params, peak {m[6]:.0f} MiB",
                      flush=True)

    best = min(results, key=results.get)
    out = FindLRResult(lr=best, optimizer=optimizer, method="sweep", sweep=results)
    if verbose:
        print(out.summary())
    return out


# --------------------------------------------------------------------------- #
# Dataset -> packed token blocks (self-contained; HF name OR local file)
# --------------------------------------------------------------------------- #
def _row_text(r):
    """Best-effort text extraction from a dataset row (str / common fields /
    alpaca-style instruction+input+output)."""
    if isinstance(r, str):
        return r
    if isinstance(r, dict):
        if r.get("instruction"):
            parts = [r.get("instruction", ""), r.get("input", ""), r.get("output", "")]
            return "\n".join(p for p in parts if p)
        for k in ("text", "content", "output", "completion", "body"):
            if r.get(k):
                return r[k]
    return ""


def _load_texts(dataset, split, max_rows):
    """Load a list of text strings from a local file (.txt/.json/.jsonl) or, if
    ``dataset`` isn't a path, an installed HuggingFace dataset by name."""
    import json
    import os
    if os.path.exists(dataset):
        if dataset.endswith(".jsonl"):
            with open(dataset) as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            texts = [_row_text(r) for r in rows]
        elif dataset.endswith(".json"):
            with open(dataset) as fh:
                texts = [_row_text(r) for r in json.load(fh)]
        else:  # plain text file
            with open(dataset) as fh:
                texts = [fh.read()]
    else:
        from datasets import load_dataset
        texts = [_row_text(r) for r in load_dataset(dataset, split=split)]
    texts = [t for t in texts if t]
    if max_rows and max_rows > 0:
        texts = texts[:max_rows]
    return texts


def build_token_stream(tok, args):
    """Tokenize ``args.dataset`` and pack into ``[N, args.seq]`` blocks; hold out
    the last ``args.eval_blocks`` for eval. Returns (train_blocks, eval_blocks)."""
    seq = args.seq
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0
    ids: List[int] = []
    for t in _load_texts(args.dataset, args.split, args.max_rows):
        ids.extend(tok(t, add_special_tokens=False)["input_ids"])
        ids.append(eos)
    n_blocks = len(ids) // seq
    if n_blocks < 2:
        raise ValueError(
            f"only {len(ids)} tokens -> {n_blocks} blocks at seq={seq}; need >=2. "
            f"Use a bigger --dataset, smaller --seq, or raise --max-rows."
        )
    blocks = torch.tensor(ids[: n_blocks * seq], dtype=torch.long).view(n_blocks, seq)
    n_eval = max(1, min(args.eval_blocks, n_blocks // 5))
    return blocks[:-n_eval], blocks[-n_eval:]


# --------------------------------------------------------------------------- #
# Optional re-exec under a target venv (so the kernel build sees a matching nvcc)
# --------------------------------------------------------------------------- #
def _strip_flags(argv, flags):
    """Drop ``--flag value`` and ``--flag=value`` occurrences from an argv list."""
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        hit = False
        for fl in flags:
            if a == fl:                 # "--flag value": skip the flag and its value
                i += 2
                hit = True
                break
            if a.startswith(fl + "="):  # "--flag=value": skip the single token
                i += 1
                hit = True
                break
        if not hit:
            out.append(a)
            i += 1
    return out


def _reexec_command(args, current_exe, argv, environ):
    """Pure/testable: return (target_python, new_argv, new_env) to re-exec under,
    or None to run in the current interpreter.

    Re-exec happens only when --python/--venv names a *different* interpreter and
    we're not already the re-exec'd child. If neither is given (the common case --
    already inside a venv/conda/uv), returns None and we run as-is.
    """
    import os
    if environ.get("GEFEN_FINDLR_REEXEC"):
        return None
    target = args.python
    if not target and args.venv:
        target = os.path.join(args.venv, "bin", "python")
    if not target:
        return None
    if os.path.realpath(target) == os.path.realpath(current_exe):
        return None                                   # already this interpreter
    if not os.path.exists(target):
        raise SystemExit(f"--python/--venv interpreter not found: {target}")
    env = dict(environ)
    env["GEFEN_FINDLR_REEXEC"] = "1"
    if args.cuda_home:
        env["CUDA_HOME"] = args.cuda_home
        env["PATH"] = os.path.join(args.cuda_home, "bin") + os.pathsep + env.get("PATH", "")
    new_argv = _strip_flags(argv, ("--python", "--venv", "--cuda-home"))
    return target, ["-m", "gefen.tools.find_lr", *new_argv], env


def _maybe_reexec(args):
    import os
    import sys
    plan = _reexec_command(args, sys.executable, sys.argv[1:], os.environ)
    if plan is None:
        return
    target, new_argv, env = plan
    note = f" (CUDA_HOME={env['CUDA_HOME']})" if args.cuda_home else ""
    print(f"[find_lr] re-exec under {target}{note}", flush=True)
    # Intentional process replacement so the kernel build sees a matching nvcc.
    os.execve(target, [target, *new_argv], env)  # noqa: S606 -- deliberate re-exec


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Auto-find a base LR for a Gefen-family optimizer.")
    ap.add_argument("--model", required=True)
    # Optional environment redirection (skip if you already run inside the right
    # venv/conda/uv env). --venv <dir> uses <dir>/bin/python; --cuda-home points
    # the kernel build at a matching nvcc.
    ap.add_argument("--venv", default=None, help="path to a venv/conda env dir; re-exec under <dir>/bin/python")
    ap.add_argument("--python", default=None, help="path to a python interpreter to re-exec under")
    ap.add_argument("--cuda-home", default=None, help="CUDA toolkit dir (sets CUDA_HOME + PATH for the kernel build)")
    ap.add_argument("--optimizer", choices=OPTIMIZERS, default="gefen")
    ap.add_argument("--method", choices=["range_test", "sweep"], default="range_test")
    ap.add_argument("--dataset", default="tatsu-lab/alpaca")
    ap.add_argument("--split", default="train")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--devices", nargs="+", default=None,
        help="multiple devices for a PARALLEL sweep (e.g. --devices cuda:0 cuda:1); "
        "each LR arm runs on its own GPU. Only used for --method sweep with >1 device.",
    )
    ap.add_argument(
        "--shard", action="store_true",
        help="SHARDED sweep: each LR arm runs ONE model split across --devices via "
        "FSDP2 (arms sequential). For models too big for one GPU. Restrict physical "
        "GPUs with CUDA_VISIBLE_DEVICES so rank r -> cuda:r.",
    )
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

    # If --venv/--python names a different interpreter, re-exec under it BEFORE
    # importing gefen (kernel build). No-op when already in the right env.
    _maybe_reexec(args)

    devices = args.devices or [args.device]

    # Sharded sweep: each LR arm runs ONE model sharded across --devices (FSDP2),
    # arms sequential. For models too big for a single GPU.
    if args.shard:
        if len(devices) < 2:
            raise SystemExit("--shard needs >=2 --devices (e.g. --devices cuda:0 cuda:1)")
        res = find_lr_sharded_sweep(
            args.model, devices, optimizer=args.optimizer, sweep_lrs=args.sweep_lrs,
            dtype=args.dtype, dataset=args.dataset, split=args.split, seq=args.seq,
            max_rows=args.max_rows, eval_blocks=args.eval_blocks,
            weight_decay=args.weight_decay, sweep_steps=args.sweep_steps,
            warmup=args.warmup, bs=args.bs, seed=args.seed,
        )
        print(f"\nRECOMMENDED base LR for {args.optimizer} (sharded): {res.lr:.3e}")
        return

    # Parallel sweep: fan the LR grid across GPUs (loads the model per worker from
    # --model, so this is CLI-level, not the in-memory find_lr path).
    if args.method == "sweep" and len(devices) > 1:
        res = find_lr_parallel_sweep(
            args.model, devices, optimizer=args.optimizer, sweep_lrs=args.sweep_lrs,
            dtype=args.dtype, dataset=args.dataset, split=args.split, seq=args.seq,
            max_rows=args.max_rows, eval_blocks=args.eval_blocks,
            weight_decay=args.weight_decay, sweep_steps=args.sweep_steps,
            warmup=args.warmup, bs=args.bs, seed=args.seed,
        )
        print(f"\nRECOMMENDED base LR for {args.optimizer}: {res.lr:.3e}")
        return

    from gefen.tools.run_model_calibration import load_model
    tok, model = load_model(args.model, devices[0], getattr(torch, args.dtype))
    train_blocks, eval_blocks = build_token_stream(tok, args)
    res = find_lr(
        model, train_blocks, optimizer=args.optimizer, method=args.method,
        eval_blocks=eval_blocks, device=devices[0], weight_decay=args.weight_decay,
        num_iter=args.num_iter, start_lr=args.start_lr, end_lr=args.end_lr,
        sweep_lrs=args.sweep_lrs, sweep_steps=args.sweep_steps, warmup=args.warmup,
        bs=args.bs, seed=args.seed,
    )
    print(f"\nRECOMMENDED base LR for {args.optimizer}: {res.lr:.3e}")


if __name__ == "__main__":
    main()

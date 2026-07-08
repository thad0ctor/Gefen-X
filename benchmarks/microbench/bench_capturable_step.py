"""Benchmark capturable optimizer-step modes on a synthetic 386M-param census.

This is the reproducible harness for the capturable table in COMPATIBILITY.md:
it builds a Qwen-ish collection of 2D transformer matrices, primes optimizer
state eagerly, then times steady-state ``opt.step()`` variants:

  * default eager (``capturable=False``)
  * eager capturable
  * manual ``torch.cuda.CUDAGraph`` replay
  * ``torch.compile(mode="reduce-overhead", dynamic=False)``

Both CUDA-event time and synchronized wall time are recorded per step. The
headline is the tail window (default: last 100 of 500 measured steps), so compile
warmup/cudagraph-tree setup and early autotune noise do not contaminate the
steady-state number.

Run on the 3090 Ti timing GPU, from outside the repo checkout:

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      PYTHONPATH=/path/to/Gefen/src python benchmarks/microbench/bench_capturable_step.py
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch


def make_shapes(hidden: int, inter: int, layers: int, kv_groups: int) -> list[tuple[int, int]]:
    kv = hidden // kv_groups
    per_layer = [
        (hidden, hidden),  # q_proj
        (kv, hidden),      # k_proj
        (kv, hidden),      # v_proj
        (hidden, hidden),  # o_proj
        (inter, hidden),   # gate_proj
        (inter, hidden),   # up_proj
        (hidden, inter),   # down_proj
    ]
    return per_layer * layers


def named_params(
    shapes: list[tuple[int, int]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> list[tuple[str, torch.nn.Parameter]]:
    gen = torch.Generator(device=device).manual_seed(seed)
    params = []
    for idx, (rows, cols) in enumerate(shapes):
        p = torch.randn(rows, cols, device=device, dtype=dtype, generator=gen)
        p.mul_(0.02)
        params.append((f"layers.{idx}.weight", torch.nn.Parameter(p)))
    return params


def attach_static_grads(
    params: list[tuple[str, torch.nn.Parameter]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> None:
    gen = torch.Generator(device=device).manual_seed(seed)
    for _name, p in params:
        p.grad = torch.randn(p.shape, device=device, dtype=dtype, generator=gen).mul_(1e-3)


def build_optimizer(
    optimizer: str,
    params: list[tuple[str, torch.nn.Parameter]],
    *,
    lr,
    capturable: bool,
):
    if optimizer == "gefen":
        from gefen import Gefen

        return Gefen(params, lr=lr, weight_decay=0.01, fused=True, capturable=capturable)
    if optimizer == "hybrid":
        from gefen import GefenMuonHybrid

        return GefenMuonHybrid(
            params,
            [],
            lr=lr,
            weight_decay=0.01,
            fused=True,
            capturable=capturable,
        )
    raise ValueError(f"unknown optimizer {optimizer!r}")


def timed_steps(fn: Callable[[], None], *, steps: int) -> tuple[list[float], list[float]]:
    cuda_ms = []
    wall_ms = []
    for _ in range(steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall0 = time.perf_counter()
        start.record()
        fn()
        end.record()
        end.synchronize()
        wall_ms.append((time.perf_counter() - wall0) * 1000.0)
        cuda_ms.append(start.elapsed_time(end))
    return cuda_ms, wall_ms


def stats(values: list[float], tail: int) -> dict[str, float]:
    if tail <= 0 or tail > len(values):
        raise ValueError(f"tail must be in [1, {len(values)}], got {tail}")
    tail_values = values[-tail:]
    return {
        "all_mean_ms": float(statistics.mean(values)),
        "tail_mean_ms": float(statistics.mean(tail_values)),
        "tail_median_ms": float(statistics.median(tail_values)),
        "tail_min_ms": float(min(tail_values)),
        "tail_max_ms": float(max(tail_values)),
    }


def finite_params(params: list[tuple[str, torch.nn.Parameter]]) -> bool:
    return all(torch.isfinite(p).all().item() for _name, p in params)


def run_case(args, optimizer: str, mode: str) -> dict[str, object]:
    if mode == "default_eager":
        capturable = False
        lr = args.lr
    else:
        capturable = True
        lr = torch.tensor(args.lr, device=args.device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(args.device)
    shapes = make_shapes(args.hidden, args.inter, args.layers, args.kv_groups)
    params = named_params(shapes, device=args.device, dtype=args.dtype, seed=args.seed)
    attach_static_grads(params, device=args.device, dtype=args.dtype, seed=args.seed + 1)
    opt = build_optimizer(optimizer, params, lr=lr, capturable=capturable)

    def eager_step():
        opt.step()

    for _ in range(args.eager_warmup):
        eager_step()
    torch.cuda.synchronize()

    if mode in ("default_eager", "capturable_eager"):
        step_fn = eager_step
    elif mode == "manual_graph":
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            opt.step()
        torch.cuda.synchronize()

        def step_fn(graph=graph):
            graph.replay()

    elif mode == "compile":
        compiled_step = torch.compile(opt.step, mode="reduce-overhead", dynamic=False)

        def step_fn(compiled_step=compiled_step):
            torch.compiler.cudagraph_mark_step_begin()
            compiled_step()

    else:
        raise ValueError(f"unknown mode {mode!r}")

    cuda_ms, wall_ms = timed_steps(step_fn, steps=args.steps)
    torch.cuda.synchronize()
    peak = float(torch.cuda.max_memory_allocated(args.device))
    allocated = float(torch.cuda.memory_allocated(args.device))
    ok_finite = finite_params(params) if args.check_finite else True

    row = {
        "variant": args.variant,
        "optimizer": optimizer,
        "mode": mode,
        "param_count": int(sum(p.numel() for _name, p in params)),
        "num_tensors": len(params),
        "steps": args.steps,
        "tail": args.tail,
        "eager_warmup": args.eager_warmup,
        "cuda": stats(cuda_ms, args.tail),
        "wall": stats(wall_ms, args.tail),
        "peak_memory_gib": peak / 2**30,
        "allocated_after_run_gib": allocated / 2**30,
        "finite": bool(ok_finite),
    }

    del opt, params, cuda_ms, wall_ms
    gc.collect()
    torch.cuda.empty_cache()
    return row


def parse_csv(value: str, allowed: tuple[str, ...]) -> list[str]:
    if value == "all":
        return list(allowed)
    got = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(got) - set(allowed))
    if unknown:
        raise ValueError(f"unknown value(s) {unknown}; allowed: {allowed} or all")
    return got


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="unknown")
    parser.add_argument("--optimizers", default="gefen,hybrid")
    parser.add_argument("--modes", default="default_eager,capturable_eager,manual_graph,compile")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--tail", type=int, default=100)
    parser.add_argument("--eager-warmup", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--inter", type=int, default=6144)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--kv-groups", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"))
    parser.add_argument("--json", default=None, help="write rows as JSONL")
    parser.add_argument("--no-check-finite", dest="check_finite", action="store_false")
    parser.set_defaults(check_finite=True)
    ns = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    ns.device = torch.device("cuda", 0)
    torch.cuda.set_device(ns.device)
    ns.dtype = torch.bfloat16 if ns.dtype == "bf16" else torch.float32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    optimizers = parse_csv(ns.optimizers, ("gefen", "hybrid"))
    modes = parse_csv(ns.modes, ("default_eager", "capturable_eager", "manual_graph", "compile"))
    if ns.tail > ns.steps:
        raise SystemExit("--tail must be <= --steps")

    device_name = torch.cuda.get_device_name(ns.device)
    print(
        f"=== capturable step benchmark: {device_name}, torch={torch.__version__}, "
        f"variant={ns.variant}, steps={ns.steps}, tail={ns.tail} ==="
    )
    print(f"{'optimizer':>8} {'mode':>17} {'cuda_tail_ms':>13} {'wall_tail_ms':>13} "
          f"{'peak_gib':>9} {'finite':>7}")

    rows = []
    for optimizer in optimizers:
        for mode in modes:
            row = run_case(ns, optimizer, mode)
            rows.append(row)
            print(
                f"{optimizer:>8} {mode:>17} "
                f"{row['cuda']['tail_mean_ms']:13.3f} "
                f"{row['wall']['tail_mean_ms']:13.3f} "
                f"{row['peak_memory_gib']:9.3f} {str(row['finite']):>7}",
                flush=True,
            )
            if not row["finite"]:
                raise SystemExit(f"{optimizer}/{mode} produced non-finite params")

    if ns.json:
        path = Path(ns.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

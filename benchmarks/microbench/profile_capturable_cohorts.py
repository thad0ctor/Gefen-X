"""Profile capturable scalar batching across common parameter-group layouts.

The three cases keep parameter geometry identical and vary only the scalar
cohorts: one uniform stack, a decay/no-decay split, and two independently
scheduled tensor learning rates. Reported CUDA activities include kernels and
device copies but exclude the profiler's synthetic ``Optimizer.step`` event.
"""

from __future__ import annotations

import argparse
import time

import torch

from gefen import Gefen


def build_case(case: str, count: int, size: int):
    torch.manual_seed(1234)
    params = [
        torch.nn.Parameter(
            (torch.randn(size, size, device="cuda") * 0.02).bfloat16()
        )
        for _ in range(count)
    ]
    for p in params:
        p.grad = torch.randn_like(p) * 1e-3
    lr_a = torch.tensor(3e-5, device="cuda")
    lr_b = torch.tensor(2e-5, device="cuda")
    half = count // 2
    if case == "uniform":
        groups = [
            {"params": params[:half], "lr": lr_a, "weight_decay": 0.1},
            {"params": params[half:], "lr": lr_a, "weight_decay": 0.1},
        ]
    elif case == "wd_split":
        groups = [
            {"params": params[:half], "lr": lr_a, "weight_decay": 0.1},
            {"params": params[half:], "lr": lr_a, "weight_decay": 0.0},
        ]
    elif case == "lr_split":
        groups = [
            {"params": params[:half], "lr": lr_a, "weight_decay": 0.1},
            {"params": params[half:], "lr": lr_b, "weight_decay": 0.1},
        ]
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(case)
    return params, Gefen(groups, capturable=True)


def warm(opt, steps: int) -> None:
    for _ in range(steps):
        opt.step()
    torch.cuda.synchronize()


def profile_cuda_activities(fn) -> int:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
    torch.cuda.synchronize()
    return sum(
        event.device_type == torch.autograd.DeviceType.CUDA
        and not event.name.startswith("Optimizer.step")
        for event in prof.events()
    )


def time_calls(fn, calls: int) -> tuple[float, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start.record()
    for _ in range(calls):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / calls, (time.perf_counter() - wall_start) * 1e3 / calls


def run(case: str, mode: str, args) -> dict[str, object]:
    params, opt = build_case(case, args.params, args.size)
    warm(opt, args.warmup)
    if mode == "eager":
        fn = opt.step
        calls = args.steps
    else:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            opt.step()
        fn = graph.replay
        calls = args.replays
    activities = profile_cuda_activities(fn)
    cuda_ms, wall_ms = time_calls(fn, calls)
    cohorts = sum(len(device_stacks) for device_stacks in opt._capt_stacks.values())
    del opt, params
    torch.cuda.empty_cache()
    return {
        "case": case,
        "mode": mode,
        "cohorts": cohorts,
        "cuda_activities": activities,
        "cuda_ms": cuda_ms,
        "wall_ms": wall_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=int, default=64)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--replays", type=int, default=500)
    args = parser.parse_args()
    if args.params < 2 or args.params % 2:
        parser.error("--params must be an even integer >= 2")
    if min(args.size, args.warmup, args.steps, args.replays) <= 0:
        parser.error("size/warmup/steps/replays must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    props = torch.cuda.get_device_properties(0)
    print(f"gpu={props.name} uuid={props.uuid} torch={torch.__version__}")
    print(f"{'case':>9} {'mode':>6} {'cohorts':>7} {'activities':>10} {'cuda_ms':>10} {'wall_ms':>10}")
    for mode in ("eager", "graph"):
        for case in ("uniform", "wd_split", "lr_split"):
            row = run(case, mode, args)
            print(
                f"{case:>9} {mode:>6} {row['cohorts']:7d} "
                f"{row['cuda_activities']:10d} {row['cuda_ms']:10.4f} "
                f"{row['wall_ms']:10.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()

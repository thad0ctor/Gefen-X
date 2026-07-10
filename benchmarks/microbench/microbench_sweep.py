"""Occupancy sweep of the fused_update kernel: v1 vs v2 across num_blocks.

For each numel, sweep period over powers of two so num_blocks goes from numel
(period==1) down to 2 (one huge period). CUDA-event time both kernels, record
num_blocks, num_blocks / SM_count, and the v1/v2 ratio. Used to calibrate the
occupancy-driven dispatch threshold (num_blocks as a multiple of SM count).

Run: GEFEN env + CUDA_VISIBLE_DEVICES=<gpu> python microbench_sweep.py
"""
import torch

from gefen.kernels.automatic_gefen_fused import _load_extension

DEV = "cuda"
mod = _load_extension()
cap = torch.cuda.get_device_capability()
name = torch.cuda.get_device_name()
sm = torch.cuda.get_device_properties(0).multi_processor_count
codebook = torch.sort(torch.randn(256, device=DEV)).values.float().contiguous()

NUMELS = [1 << 20, 1 << 22, 1 << 24]  # 1M, 4M, 16M elements


def make(numel, period):
    nb = numel // period
    g = torch.Generator(device=DEV).manual_seed(7)
    p = (torch.randn(numel, device=DEV, dtype=torch.bfloat16, generator=g) * 0.02).contiguous()
    gv = (torch.randn(nb, period, device=DEV, dtype=torch.bfloat16, generator=g) * 1e-3).contiguous()
    ms = torch.randint(0, 256, (numel,), device=DEV, dtype=torch.uint8)
    mm = (torch.rand(nb, 1, device=DEV, dtype=torch.float32, generator=g) * 1e-3).contiguous()
    ss = (torch.rand(nb, 1, device=DEV, dtype=torch.float32, generator=g) + 0.5).contiguous()
    return p, gv, ms, mm, ss


def bench(fn, args, iters=300, warmup=80):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn(*args)
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def run_v1(a):
    mod.automatic_gefen_fused_update_cuda(
        a[0], a[1], a[2], a[3], a[4], codebook, False, 0.9, 5e-5
    )


def run_v2(a):
    mod.automatic_gefen_fused_update_v2_cuda(
        a[0], a[1], a[2], a[3], a[4], codebook, False, 0.9, 5e-5
    )


print(f"=== {name}  sm_{cap[0]}{cap[1]}  SMs={sm} ===")
print(f"{'numel':>9} {'period':>9} {'nb':>9} {'nb/SM':>9} {'v1 ms':>9} {'v2 ms':>9} {'v1/v2':>8} {'win':>4}")
for numel in NUMELS:
    period = 1
    rows = []
    while period <= numel // 2:
        a = make(numel, period)
        # Clone each mutable input ONCE per variant, outside the timed loop.
        # Cloning p/sign/magnitude inside run_v* used to charge tens of MiB of
        # allocation + D2D copies to every kernel call and corrupted the
        # crossover this script exists to measure.
        a1 = tuple(x.clone() for x in a)
        a2 = tuple(x.clone() for x in a)
        t1 = bench(run_v1, (a1,))
        t2 = bench(run_v2, (a2,))
        nb = numel // period
        ratio = t1 / t2
        win = "v2" if ratio > 1.0 else "v1"
        rows.append((period, nb, nb / sm, t1, t2, ratio, win))
        period <<= 1
    for period, nb, nbsm, t1, t2, ratio, win in rows:
        print(f"{numel:>9} {period:>9} {nb:>9} {nbsm:>9.1f} {t1:>9.4f} {t2:>9.4f} {ratio:>8.2f} {win:>4}")
    # lower crossover: largest nb (smallest period>1) where v2 still wins
    v1rows = [r for r in rows if r[0] > 1 and r[6] == "v1"]
    v2rows = [r for r in rows if r[0] > 1 and r[6] == "v2"]
    if v1rows and v2rows:
        v2_max_nbsm = max(r[2] for r in v2rows)
        v1_min_nbsm = min(r[2] for r in v1rows)
        print(f"   -> numel={numel}: v2(p>1) up to nb/SM={v2_max_nbsm:.1f}, "
              f"v1 starts nb/SM={v1_min_nbsm:.1f}")
    print()

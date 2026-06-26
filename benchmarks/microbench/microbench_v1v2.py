"""Direct CUDA-event microbench of the fused_update kernel: v1 vs v2.
Isolates the kernel from training/flash-attn/contention noise.
Run: GEFEN env + CUDA_VISIBLE_DEVICES=<gpu> python microbench_v1v2.py
"""
import torch
from gefen.kernels.automatic_gefen_fused import _load_extension

DEV = "cuda"
mod = _load_extension()
cap = torch.cuda.get_device_capability()
name = torch.cuda.get_device_name()
codebook = torch.sort(torch.randn(256, device=DEV)).values.float().contiguous()

# (numel, period) across occupancy regimes; the few-block (large period) cases
# dominate real full-FT models (~65% of fused_update time at 0.6B).
CASES = [
    (1024 * 2048, 1048576),   # nb=2   few-block (dominant)
    (1024 * 1024, 524288),    # nb=2   few-block (dominant)
    (1024 * 3072, 98304),     # nb=32
    (3072 * 1024, 512),       # nb=6144 sweet spot
    (2048 * 1024, 1),         # period==1
]


def make(numel, period):
    nb = numel // period
    g = torch.Generator(device=DEV).manual_seed(7)
    p = (torch.randn(numel, device=DEV, dtype=torch.bfloat16, generator=g) * 0.02).contiguous()
    gv = (torch.randn(nb, period, device=DEV, dtype=torch.bfloat16, generator=g) * 1e-3).contiguous()
    ms = torch.randint(0, 256, (numel,), device=DEV, dtype=torch.uint8)
    mm = (torch.rand(nb, 1, device=DEV, dtype=torch.float32, generator=g) * 1e-3).contiguous()
    ss = (torch.rand(nb, 1, device=DEV, dtype=torch.float32, generator=g) + 0.5).contiguous()
    return p, gv, ms, mm, ss


def bench(fn, args, iters=200, warmup=50):
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
    return e0.elapsed_time(e1) / iters  # ms/call


print(f"=== {name}  sm_{cap[0]}{cap[1]} ===")
print(f"{'numel':>10} {'period':>9} {'nb':>7} {'v1 ms':>9} {'v2 ms':>9} {'v2 speedup':>11}")
tot1 = tot2 = 0.0
for numel, period in CASES:
    a = make(numel, period)
    t1 = bench(mod.automatic_gefen_fused_update_cuda, (a[0].clone(), a[1], a[2].clone(), a[3].clone(), a[4], codebook, False, 0.9, 5e-5))
    t2 = bench(mod.automatic_gefen_fused_update_v2_cuda, (a[0].clone(), a[1], a[2].clone(), a[3].clone(), a[4], codebook, False, 0.9, 5e-5))
    tot1 += t1
    tot2 += t2
    print(f"{numel:>10} {period:>9} {numel//period:>7} {t1:>9.4f} {t2:>9.4f} {t1/t2:>10.2f}x")
print(f"{'TOTAL':>10} {'':>9} {'':>7} {tot1:>9.4f} {tot2:>9.4f} {tot1/tot2:>10.2f}x")

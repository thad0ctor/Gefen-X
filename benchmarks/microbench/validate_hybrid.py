"""Validate the occupancy hybrid dispatch of the fused_update kernel.

(1) Per-regime: hybrid (wrapper auto) must be within ~5% of min(v1, v2) in every
    regime -- few-block, sweet-spot, small-period, period==1.
(2) Real distribution: a representative full-FT param distribution (element mass
    per occupancy bucket from fused_bench/PERF.md, Qwen3-0.6B). Sum fused_update
    time for pure-v1, pure-v2, and hybrid -> hybrid TOTAL must be <= both.

Run: GEFEN env + CUDA_VISIBLE_DEVICES=<gpu> python validate_hybrid.py
"""
import torch

from gefen.kernels.automatic_gefen_fused import (
    _should_use_v2,
    _sm_count,
    automatic_gefen_fused_update_cuda,
)

DEV = torch.device("cuda", 0)
sm = _sm_count(DEV)
name = torch.cuda.get_device_name()
cap = torch.cuda.get_device_capability()
codebook = torch.sort(torch.randn(256, device=DEV)).values.float().contiguous()


def make(numel, period):
    nb = numel // period
    g = torch.Generator(device=DEV).manual_seed(7)
    p = (torch.randn(numel, device=DEV, dtype=torch.bfloat16, generator=g) * 0.02).contiguous()
    gv = (torch.randn(nb, period, device=DEV, dtype=torch.bfloat16, generator=g) * 1e-3).contiguous()
    ms = torch.randint(0, 256, (numel,), device=DEV, dtype=torch.uint8, generator=g)
    mm = (torch.rand(nb, 1, device=DEV, dtype=torch.float32, generator=g) * 1e-3).contiguous()
    ss = (torch.rand(nb, 1, device=DEV, dtype=torch.float32, generator=g) + 0.5).contiguous()
    return p, gv, ms, mm, ss


def _fn(args, mode):
    # mode: True=force v2, False=force v1, None=hybrid (wrapper auto-dispatch).
    return lambda: automatic_gefen_fused_update_cuda(
        args[0], args[1], args[2], args[3], args[4], codebook, False, 0.9, 5e-5, use_v2=mode
    )


def bench3(numel, period, iters=200, warmup=60):
    # Time v1 / v2 / hybrid interleaved on fresh per-variant buffers, so they
    # share thermal/clock state (sequential blocks bias whichever runs last) and
    # in-place mutation can't favor one variant. Returns (t1, t2, th) ms/call.
    fns = [_fn(make(numel, period), m) for m in (False, True, None)]
    for f in fns:
        for _ in range(warmup):
            f()
    torch.cuda.synchronize()
    acc = [0.0, 0.0, 0.0]
    for _ in range(iters):
        for i, f in enumerate(fns):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            f()
            e1.record()
            e1.synchronize()
            acc[i] += e0.elapsed_time(e1)
    return acc[0] / iters, acc[1] / iters, acc[2] / iters


print(f"=== {name}  sm_{cap[0]}{cap[1]}  SMs={sm} ===\n")

# ---- (1) per-regime: hybrid within 5% of min(v1,v2) ----
REGIMES = [
    ("few-block",   1 << 21, 1 << 20),   # nb=2
    ("few-block-32", 3145728, 98304),    # nb=32
    ("small-period", 1 << 22, 4),        # period=4, huge nb
    ("sweet-spot",  3145728, 512),       # nb=6144
    ("sweet-spot2", 1 << 24, 2048),      # nb=8192, big param
    ("period==1",   1 << 21, 1),         # nb=numel
]
print(f"{'regime':>13} {'nb':>9} {'period':>9} {'pick':>4} {'v1 ms':>9} {'v2 ms':>9} {'hyb ms':>9} {'hyb/min':>8} {'ok':>4}")
all_ok = True
for label, numel, period in REGIMES:
    t1, t2, th = bench3(numel, period)
    pick = "v2" if _should_use_v2(numel // period, period, DEV) else "v1"
    mn = min(t1, t2)
    ratio = th / mn
    ok = ratio <= 1.05
    all_ok = all_ok and ok
    print(f"{label:>13} {numel//period:>9} {period:>9} {pick:>4} {t1:>9.4f} {t2:>9.4f} {th:>9.4f} {ratio:>8.3f} {'OK' if ok else 'BAD':>4}")
print(f"\nper-regime hybrid<=1.05*min(v1,v2): {'PASS' if all_ok else 'FAIL'}\n")

# ---- (2) representative full-FT distribution (PERF.md Qwen3-0.6B buckets) ----
# (count, nb, period) reproducing the element mass per occupancy bucket:
#   few-block nb<=8 : 72 params, ~118M elem ; nb<=64 : 12 params, ~23M elem ;
#   sweet-spot       : 124 params, ~212M elem ; period==1 : 102 params, ~242M elem.
DIST = [
    ("few-block (nb<=8)",   72,  4, 409600),   # numel 1.638M  -> 118M
    ("mid (nb<=64)",        12, 32,  60416),   # numel 1.933M  -> 23.2M
    ("sweet-spot",         124, 2048,   836),  # numel 1.712M  -> 212M
    ("period==1",          102,  1, 2373632),  # numel 2.374M  -> 242M
]
print(f"{'bucket':>20} {'cnt':>4} {'nb':>8} {'period':>8} {'pick':>4} {'v1/p ms':>8} {'v2/p ms':>8} {'hyb/p ms':>9}")
tot1 = tot2 = toth = 0.0
elems = 0
for label, cnt, nb, period in DIST:
    numel = nb * period
    elems += cnt * numel
    t1, t2, th = bench3(numel, period)
    pick = "v2" if _should_use_v2(nb, period, DEV) else "v1"
    tot1 += cnt * t1
    tot2 += cnt * t2
    toth += cnt * th
    print(f"{label:>20} {cnt:>4} {nb:>8} {period:>8} {pick:>4} {t1:>8.4f} {t2:>8.4f} {th:>9.4f}")

print(f"\ntotal elements: {elems/1e6:.0f}M across {sum(d[1] for d in DIST)} params")
print(f"{'':>20} {'pure-v1':>10} {'pure-v2':>10} {'hybrid':>10}")
print(f"{'fused_update total ms':>20} {tot1:>10.3f} {tot2:>10.3f} {toth:>10.3f}")
print(f"{'hybrid vs v1':>20} {tot1/toth:>9.2f}x")
print(f"{'hybrid vs v2':>20} {tot2/toth:>9.2f}x")
win = toth <= tot1 * 1.001 and toth <= tot2 * 1.001
print(f"\nhybrid TOTAL <= both pure totals: {'PASS' if win else 'FAIL'}")
raise SystemExit(0 if (all_ok and win) else 1)

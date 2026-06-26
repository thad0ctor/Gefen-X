"""
Parity + correctness tests for the fused Gefen update kernels (v1 single-pass vs
v2 occupancy-flexible two-phase) and the vmean period==1 fast path.

Coverage (each runs on the current CUDA device — for a multi-arch sweep use
tests/run_parity_all_gpus.sh, which launches one process per GPU):

  1. v1 vs v2 bit-parity across occupancy regimes, dtypes (bf16/fp16/fp32), and
     codebook distributions.
  2. independent fp32 PyTorch oracle for the *full* update (so "matches v1" is
     backed by "matches a from-scratch reference", not just the other kernel).
  3. NaN/Inf gradient edge inputs: v1 and v2 must stay bit-identical (this is
     what the atomic_max_nonneg NaN guard exists for).
  4. v2 input-validation guards raise (dtype mismatch, codebook size, packed
     indices, non-float32 aux tensors).
  5. determinism: v2 repeated on identical inputs is bit-identical run-to-run.
  6. vmean period==1 kernel vs PyTorch reference.

Run on the current GPU:  python tests/test_fused_update_v2_parity.py
Sweep every GPU:         bash tests/run_parity_all_gpus.sh
"""
import argparse

import torch

try:
    import pytest
except ImportError:  # pragma: no cover - allows the script path without pytest
    pytest = None

from gefen.kernels.automatic_gefen_fused import _load_extension
from gefen.kernels.automatic_vmean import automatic_vmean_update_cuda
from gefen.gefen import gefen_nearest_codebook_indices

BETA1, LR = 0.9, 5e-5

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# (numel, period) covering every occupancy regime.
CASES = [
    (1024 * 2048, 1048576),  # num_blocks = 2  (huge period, few blocks)
    (1024 * 1024, 524288),   # num_blocks = 2
    (1024 * 3072, 98304),    # num_blocks = 32
    (3072 * 1024, 512),      # num_blocks = 6144 (sweet spot)
    (3072 * 1024, 64),       # num_blocks = 49152
    (2048 * 1024, 1),        # period == 1
    (1024 * 1024, 1),        # period == 1
    (128, 1),                # tiny
]
# Smaller subset for the dtype/codebook/oracle sweeps to bound runtime.
SWEEP = [(1024 * 3072, 98304), (3072 * 1024, 512), (2048 * 1024, 1), (128, 1)]


def _codebook(kind, device):
    g = torch.Generator(device=device).manual_seed(0)
    if kind == "gauss":
        cb = torch.randn(256, device=device, generator=g)
    elif kind == "uniform":
        cb = torch.rand(256, device=device, generator=g) * 2.0 - 1.0
    elif kind == "skewed":
        cb = torch.randn(256, device=device, generator=g).abs() ** 2
    else:
        raise ValueError(kind)
    return torch.sort(cb).values.float().contiguous()


def _make(numel, period, dtype, device, seed, scale=1.0):
    g = torch.Generator(device=device).manual_seed(seed)
    nb = numel // period
    p = (torch.randn(numel, device=device, dtype=dtype, generator=g) * 0.02 * scale).contiguous()
    gv = (torch.randn(nb, period, device=device, dtype=dtype, generator=g) * 1e-3 * scale).contiguous()
    ms = torch.randint(0, 256, (numel,), device=device, dtype=torch.uint8, generator=g)
    mm = (torch.rand(nb, 1, device=device, dtype=torch.float32, generator=g) * 1e-3).contiguous()
    ss = (torch.rand(nb, 1, device=device, dtype=torch.float32, generator=g) + 0.5).contiguous()
    return p, gv, ms, mm, ss


def _bits(t):
    # Reinterpret float bits as same-width ints so the comparison is truly
    # bit-for-bit -- no collapsing of +0/-0, NaN payloads, or signed NaNs, and no
    # bf16/fp16 information loss from an intermediate float() cast. Integer dtypes
    # (e.g. the uint8 indices) pass through unchanged.
    t = t.contiguous()
    if t.dtype in (torch.bfloat16, torch.float16):
        return t.view(torch.int16)
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    if t.dtype == torch.float64:
        return t.view(torch.int64)
    return t


def _bit_equal(a, b):
    # v1 and v2 run the identical float ops, so their outputs must match to the
    # exact bit pattern (including NaN payloads / signed zeros), in native dtype.
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return bool((_bits(a) == _bits(b)).all().item())


def _run(mod, v2, p, gv, ms, mm, ss, cb):
    p, ms, mm = p.clone(), ms.clone(), mm.clone()
    fn = mod.automatic_gefen_fused_update_v2_cuda if v2 else mod.automatic_gefen_fused_update_cuda
    fn(p, gv, ms, mm, ss, cb, False, BETA1, LR)
    return p, ms, mm


# --------------------------------------------------------------------------- 1
def parity(device):
    mod = _load_extension()
    ok = True
    fails = 0
    combos = []
    # primary: full case list, bf16, gaussian codebook (the original coverage)
    combos += [("bf16", "gauss", c) for c in CASES]
    # dtype sweep + codebook sweep on the smaller subset
    combos += [(dt, "gauss", c) for dt in ("fp16", "fp32") for c in SWEEP]
    combos += [("bf16", cb, c) for cb in ("uniform", "skewed") for c in SWEEP]
    for dt_name, cb_kind, (numel, period) in combos:
        cb = _codebook(cb_kind, device)
        p, gv, ms, mm, ss = _make(numel, period, DTYPES[dt_name], device, 100 + (numel % 97))
        p1, ms1, mm1 = _run(mod, False, p, gv, ms, mm, ss, cb)
        p2, ms2, mm2 = _run(mod, True, p, gv, ms, mm, ss, cb)
        good = _bit_equal(p1, p2) and _bit_equal(mm1, mm2) and _bit_equal(ms1, ms2)
        ok = ok and good
        if not good:
            fails += 1
            print(f"  [parity FAIL] dt={dt_name} cb={cb_kind} numel={numel} period={period}")
    print(f"[1] v1 vs v2 parity: {len(combos)} combos (dtypes/codebooks/regimes) -> "
          f"{'PASS' if ok else f'FAIL ({fails})'}")
    return ok


# --------------------------------------------------------------------------- 2
def _oracle(p, gv, ms, mm, ss, cb):
    """Pure-torch fp32 reimplementation of the fused update (independent of the
    CUDA kernel). Mirrors automatic_gefen_fused_kernel.cu exactly."""
    nb, period = gv.shape
    coeff = cb[ms.long()].view(nb, period)
    current_m = mm.float() * coeff
    updated = BETA1 * current_m + (1.0 - BETA1) * gv.float()
    new_mag = updated.abs().amax(dim=1, keepdim=True)
    normalized = torch.where(new_mag > 0, updated / new_mag, torch.zeros_like(updated))
    new_idx = gefen_nearest_codebook_indices(cb, normalized)
    quantized = cb[new_idx.long()] * new_mag
    new_p = p.float() - (quantized * ss.float() * LR).reshape(-1)
    return new_p, new_mag, new_idx.reshape(-1)


def oracle(device):
    mod = _load_extension()
    cb = _codebook("gauss", device)
    ok = True
    worst_p = worst_m = 0.0
    for dt_name in ("bf16", "fp32"):
        dt = DTYPES[dt_name]
        for numel, period in SWEEP:
            p, gv, ms, mm, ss = _make(numel, period, dt, device, 7)
            pk, msk, mmk = _run(mod, False, p, gv, ms, mm, ss, cb)  # v1 (v2==v1 by test 1)
            op, omag, oidx = _oracle(p, gv, ms.clone(), mm, ss, cb)
            # Magnitude: kernel and oracle both compute `updated` in fp32 with the
            # same ops, so this matches to ~1 ULP. Use abs+rel tol so per-element
            # period==1 blocks with near-zero magnitude don't blow up the ratio.
            mag_ok = torch.allclose(mmk, omag, rtol=1e-3, atol=1e-6)
            # p: only fp32 is above the param-dtype noise floor (these intentionally
            # tiny updates underflow bf16/fp16, which would make the check trivial).
            if dt_name == "fp32":
                p_ok = torch.allclose(pk, op.to(dt), rtol=1e-4, atol=1e-7)
            else:
                p_ok = True
            worst_m = max(worst_m, ((mmk - omag).abs() / (omag.abs() + 1e-6)).max().item())
            if dt_name == "fp32":
                worst_p = max(worst_p, ((pk - op).abs() / (op.abs() + 1e-9)).max().item())
            good = mag_ok and p_ok
            ok = ok and good
            if not good:
                print(f"  [oracle FAIL] dt={dt_name} numel={numel} period={period} "
                      f"mag_ok={mag_ok} p_ok={p_ok}")
    print(f"[2] independent fp32 oracle: worst rel|p|(fp32)={worst_p:.2e} "
          f"worst |mag| rel={worst_m:.2e} -> {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- 3
def edge_parity(device):
    mod = _load_extension()
    cb = _codebook("gauss", device)
    ok = True
    for numel, period in SWEEP:
        p, gv, ms, mm, ss = _make(numel, period, torch.bfloat16, device, 11)
        gv = gv.clone()
        flat = gv.view(-1)
        n = flat.numel()
        flat[0] = float("nan")
        flat[n // 2] = float("inf")
        flat[n - 1] = float("-inf")
        if n > 3:
            flat[n // 3] = float("nan")
        p1, ms1, mm1 = _run(mod, False, p, gv, ms, mm, ss, cb)
        p2, ms2, mm2 = _run(mod, True, p, gv, ms, mm, ss, cb)
        good = _bit_equal(p1, p2) and _bit_equal(mm1, mm2) and _bit_equal(ms1, ms2)
        ok = ok and good
        if not good:
            print(f"  [edge FAIL] numel={numel} period={period}")
    print(f"[3] NaN/Inf grad edge — v1 vs v2 bit-identical: {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- 4
def guards(device):
    mod = _load_extension()
    cb = _codebook("gauss", device)
    p, gv, ms, mm, ss = _make(2048, 64, torch.bfloat16, device, 3)
    checks = []

    def expect_raise(label, fn):
        try:
            fn()
            checks.append((label, False))
            print(f"  [guard FAIL] {label}: no exception")
        except Exception:  # noqa: BLE001 - pybind maps the C++ throw to some Python error
            checks.append((label, True))

    expect_raise("grad dtype != p", lambda: mod.automatic_gefen_fused_update_v2_cuda(
        p.clone(), gv.float(), ms.clone(), mm.clone(), ss, cb, False, BETA1, LR))
    expect_raise("codebook > 256", lambda: mod.automatic_gefen_fused_update_v2_cuda(
        p.clone(), gv, ms.clone(), mm.clone(), ss,
        torch.sort(torch.randn(300, device=device)).values.float().contiguous(), False, BETA1, LR))
    expect_raise("empty codebook", lambda: mod.automatic_gefen_fused_update_v2_cuda(
        p.clone(), gv, ms.clone(), mm.clone(), ss,
        torch.empty(0, device=device, dtype=torch.float32), False, BETA1, LR))
    expect_raise("packed indices", lambda: mod.automatic_gefen_fused_update_v2_cuda(
        p.clone(), gv, ms.clone(), mm.clone(), ss, cb, True, BETA1, LR))
    expect_raise("m_magnitude not f32", lambda: mod.automatic_gefen_fused_update_v2_cuda(
        p.clone(), gv, ms.clone(), mm.clone().half(), ss, cb, False, BETA1, LR))
    expect_raise("stepsize not f32", lambda: mod.automatic_gefen_fused_update_v2_cuda(
        p.clone(), gv, ms.clone(), mm.clone(), ss.half(), cb, False, BETA1, LR))

    ok = all(passed for _, passed in checks)
    print(f"[4] v2 input guards raise: {sum(p for _, p in checks)}/{len(checks)} -> {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- 5
def repeatability(device, runs=5):
    mod = _load_extension()
    cb = _codebook("gauss", device)
    ok = True
    for numel, period in SWEEP:
        p, gv, ms, mm, ss = _make(numel, period, torch.bfloat16, device, 5)
        ref = _run(mod, True, p, gv, ms, mm, ss, cb)
        for _ in range(runs - 1):
            cur = _run(mod, True, p, gv, ms, mm, ss, cb)
            same = all(_bit_equal(a, b) for a, b in zip(ref, cur))
            ok = ok and same
            if not same:
                print(f"  [repeat FAIL] numel={numel} period={period}")
                break
    print(f"[5] v2 determinism ({runs}x identical inputs): {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- 6
def vmean_period1(device):
    ok = True
    for numel in (128, 1024, 1024 * 1024, 2048 * 1024):
        g = torch.Generator(device=device).manual_seed(7)
        gv = (torch.randn(numel, 1, device=device, dtype=torch.bfloat16, generator=g) * 1e-3).contiguous()
        vm = (torch.rand(numel, 1, device=device, dtype=torch.float32, generator=g) * 1e-3).contiguous()
        gf = gv.float()
        ref = 0.999 * vm + (1.0 - 0.999) * (gf * gf)
        automatic_vmean_update_cuda(vm, gv, 0.999)
        d = (vm - ref).abs().max().item()
        ok = ok and d <= 1e-6
    print(f"[6] vmean period==1 vs reference: {'PASS' if ok else 'FAIL'}")
    return ok


def run_suite(device):
    cap = torch.cuda.get_device_capability(device)
    print(f"=== device {device} ({torch.cuda.get_device_name(device)}, sm_{cap[0]}{cap[1]}) ===")
    results = [
        parity(device), oracle(device), edge_parity(device),
        guards(device), repeatability(device), vmean_period1(device),
    ]
    ok = all(results)
    print(f"OVERALL [{device}]: {'PASS' if ok else 'FAIL'}\n")
    return ok


if pytest is not None:
    requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

    @requires_cuda
    def test_fused_update_v2_parity():
        assert run_suite(torch.device("cuda"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-gpus", action="store_true",
                    help="run on every visible GPU in-process (only works if the "
                         "kernel .so was built fat; prefer run_parity_all_gpus.sh)")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "CUDA required"
    devs = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] if args.all_gpus \
        else [torch.device("cuda")]
    all_ok = all(run_suite(d) for d in devs)
    print("V2 PARITY OVERALL:", "PASS" if all_ok else "FAIL")
    raise SystemExit(0 if all_ok else 1)

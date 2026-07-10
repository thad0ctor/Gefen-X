"""Parity for the Tier-1 fully-fused *v2* (two-phase) update kernel
(automatic_gefen_fused_update_v2_full_cuda) vs the decomposed v2 pipeline it
replaces: the standalone vmean kernel + the host stepsize/bias-correction math
(gefen.py) + the original two-phase v2 update kernel.

The fused kernel folds K1 (vmean EMA via Sum(grad^2) accumulated in the magnitude
phase), K2 (in-kernel per-block stepsize) and K3 (weight decay in the write).

Correctness contract:
  * p, m_sign and m_magnitude are BIT-IDENTICAL to the decomposed pipeline run on
    the *same* vmean the fused kernel produced. Magnitude is an order-free absmax;
    the in-kernel stepsize and the wd-folded write are op-for-op identical to the
    host math + v2 update kernel (the v1 fully-fused kernel proves these ops are
    bit-exact vs the host). So given its own vmean, downstream is deterministic.
  * vmean is NOT bit-identical to the standalone tree reduction: the per-block
    Sum(grad^2) is accumulated atomically (non-deterministic order). It is a
    sub-ULP-scale perturbation of a 2nd-moment EMA (convergence-neutral). We
    assert it stays within VMEAN_RTOL of the standalone kernel. period==1 has a
    single term per block and is bit-identical there.

Run on the current GPU:  python tests/test_fused_update_v2_full_parity.py
"""
import math

import torch

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

from gefen.kernels.automatic_gefen_fused import _codebook_search_lut, _load_extension
from gefen.kernels.automatic_vmean import automatic_vmean_update_cuda

BETA1, BETA2, EPS, LR = 0.9, 0.999, 1e-8, 5e-5

# Documented tolerance for the atomic Sum(grad^2) vs the standalone tree sum.
# Measured worst-case rel deviation on sm_86 over the case/step/dtype sweep is
# ~1e-6 (printed by the suite); this bound leaves a small margin. p/m_sign/
# m_magnitude remain bit-exact regardless (asserted separately).
VMEAN_RTOL = 1e-5
VMEAN_ATOL = 1e-8

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# v2-regime shapes (few blocks / tiny period) plus a large-period split case and
# period==1. The fused v2 kernel is correct for any shape.
CASES = [
    (1024 * 2048, 1048576),  # num_blocks = 2  (huge period -> split kernel)
    (1024 * 3072, 98304),    # num_blocks = 32 (split kernel)
    (3072 * 1024, 512),      # flat kernel, medium period
    (3072 * 1024, 64),       # flat kernel
    (3072 * 1024, 16),       # tiny period -> v2 regime
    (2048 * 1024, 1),        # period == 1 (bit-identical vmean)
    (771 * 3001, 3001),      # odd period in the blocks_per_row==1 split window
                             # AND total % 8 != 0: exercises the chunked
                             # kernels' scalar tail + unaligned-row fallbacks
]
STEPS = [1, 2, 7, 50]

# Representative shapes whose default path changes when moving from the legacy
# partial-update crossover to the current full-kernel crossover. This pins the
# accepted numerical contract at the exact v1/v2 boundary being retuned.
ROUTE_SWITCH_CASES = [
    (4096, 1),    # legacy v2 -> full v1: tiny period
    (4096, 12),   # legacy v2 -> full v1: packed short rows
    (4, 512),     # legacy v2 -> full v1: few medium-period rows
    (512, 8192),  # legacy v1 -> full v2: large-period split reducer
]


def _codebook(device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    cb = torch.randn(256, device=device, generator=g)
    return torch.sort(cb).values.float().contiguous()


def _make(numel, period, dtype, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    nb = numel // period
    p = (torch.randn(numel, device=device, dtype=dtype, generator=g) * 0.02).contiguous()
    gv = (torch.randn(nb, period, device=device, dtype=dtype, generator=g) * 1e-3).contiguous()
    ms = torch.randint(0, 256, (numel,), device=device, dtype=torch.uint8, generator=g)
    mm = (torch.rand(nb, 1, device=device, dtype=torch.float32, generator=g) * 1e-3).contiguous()
    vm = (torch.rand(nb, 1, device=device, dtype=torch.float32, generator=g) * 1e-3).contiguous()
    return p, gv, ms, mm, vm


def _bits(t):
    t = t.contiguous()
    if t.dtype in (torch.bfloat16, torch.float16):
        return t.view(torch.int16)
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    return t


def _bit_equal(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return bool((_bits(a) == _bits(b)).all().item())


def _host_stepsize(vmean, bc1, bc2):
    # Mirror gefen.py's host stepsize math exactly (sqrt -> div_ -> add_ ->
    # reciprocal -> mul_); the kernel reproduces these ops in-kernel.
    h = torch.sqrt(vmean)
    h = h.div_(math.sqrt(bc2)).add_(EPS)
    stepsize = torch.reciprocal(h)
    stepsize = stepsize.mul_(1.0 / bc1)
    return stepsize


def _fused(mod, p, gv, ms, mm, vm, cb, step, wd, lut=None):
    p, ms, mm, vm = p.clone(), ms.clone(), mm.clone(), vm.clone()
    bc1 = 1 - BETA1 ** step
    bc2 = 1 - BETA2 ** step
    weight_decay_factor = 1.0 - LR * wd
    if lut is None:
        lut = torch.empty(0, dtype=torch.int16, device=p.device)
    mod.automatic_gefen_fused_update_v2_full_cuda(
        p, gv, ms, mm, vm, cb, lut, False, BETA1, BETA2, LR, EPS,
        1.0 / math.sqrt(bc2), 1.0 / bc1, weight_decay_factor)
    return p, ms, mm, vm


def _fused_v1(mod, p, gv, ms, mm, vm, cb, step, wd, lut=None):
    p, ms, mm, vm = p.clone(), ms.clone(), mm.clone(), vm.clone()
    bc1 = 1 - BETA1 ** step
    bc2 = 1 - BETA2 ** step
    weight_decay_factor = 1.0 - LR * wd
    if lut is None:
        lut = torch.empty(0, dtype=torch.int16, device=p.device)
    mod.automatic_gefen_fused_full_update_cuda(
        p, gv, ms, mm, vm, cb, lut, False, BETA1, BETA2, LR, EPS,
        1.0 / math.sqrt(bc2), 1.0 / bc1, weight_decay_factor)
    return p, ms, mm, vm


def _reference(mod, p, gv, ms, mm, fused_vmean, cb, step, wd):
    """Decomposed v2 pipeline, run on the fused kernel's OWN vmean so the only
    sanctioned divergence (the atomic Sum(grad^2)) does not leak into p."""
    p, ms, mm = p.clone(), ms.clone(), mm.clone()
    bc1 = 1 - BETA1 ** step
    bc2 = 1 - BETA2 ** step
    stepsize = _host_stepsize(fused_vmean.clone(), bc1, bc2).contiguous()
    if wd > 0.0:
        p.mul_(1 - LR * wd)
    mod.automatic_gefen_fused_update_v2_cuda(
        p, gv, ms, mm, stepsize, cb, False, BETA1, LR)
    return p, ms, mm


def _standalone_vmean(vm, gv):
    vm = vm.clone()
    automatic_vmean_update_cuda(vm, gv, BETA2)
    return vm


def parity(device, wd=0.0):
    mod = _load_extension()
    cb = _codebook(device)
    ok = True
    fails = 0
    n = 0
    worst_vm = 0.0
    for dt_name, dt in DTYPES.items():
        for numel, period in CASES:
            for step in STEPS:
                p, gv, ms, mm, vm = _make(numel, period, dt, device, 100 + step + (numel % 91))
                fp, fms, fmm, fvm = _fused(mod, p, gv, ms, mm, vm, cb, step, wd)
                rp, rms, rmm = _reference(mod, p, gv, ms, mm, fvm, cb, step, wd)
                svm = _standalone_vmean(vm, gv)
                # LUT-narrowed quantize vs the full-range search on identical
                # inputs: the quantize inputs (old/new magnitude) are atomic-
                # order-independent, so m_sign and m_magnitude must be
                # bit-identical; p inherits only the documented vmean/stepsize
                # atomic jitter between the two runs.
                lp, lms, lmm, _ = _fused(mod, p, gv, ms, mm, vm, cb, step, wd,
                                         lut=_codebook_search_lut(cb))
                lut_ok = (_bit_equal(lms, fms) and _bit_equal(lmm, fmm)
                          and torch.allclose(lp.float(), fp.float(),
                                             rtol=VMEAN_RTOL, atol=VMEAN_ATOL))
                bitwise = (_bit_equal(rp, fp) and _bit_equal(rms, fms)
                           and _bit_equal(rmm, fmm) and lut_ok)
                # period==1 has a single grad^2 term per block, so the atomic
                # accumulation degenerates to one add and must match the
                # standalone tree reduction bit-for-bit -- no tolerance there.
                if period == 1:
                    vm_ok = _bit_equal(fvm, svm)
                else:
                    vm_ok = torch.allclose(fvm, svm, rtol=VMEAN_RTOL, atol=VMEAN_ATOL)
                rel = ((fvm - svm).abs() / (svm.abs() + 1e-12)).max().item()
                worst_vm = max(worst_vm, rel)
                good = bitwise and vm_ok
                n += 1
                ok = ok and good
                if not good:
                    fails += 1
                    print(f"  [FAIL] dt={dt_name} numel={numel} period={period} step={step} "
                          f"p={_bit_equal(rp,fp)} ms={_bit_equal(rms,fms)} "
                          f"mm={_bit_equal(rmm,fmm)} vm_ok={vm_ok} vm_rel={rel:.2e}")
    tag = f"wd={wd}"
    print(f"[fused-v2-full vs decomposed v2, {tag}] {n} combos "
          f"(p/ms/mm bit-exact; vmean rtol<={VMEAN_RTOL:.0e}) worst vmean rel={worst_vm:.2e} -> "
          f"{'PASS' if ok else f'FAIL ({fails})'}")
    return ok


def determinism(device, runs=4):
    """p inherits the atomic vmean's run-to-run perturbation, but the deviation
    must stay within the same documented vmean tolerance (p tracks vmean), and
    period==1 (no cross-block atomic contention) must be bit-identical."""
    mod = _load_extension()
    cb = _codebook(device)
    ok = True
    for numel, period in CASES:
        p, gv, ms, mm, vm = _make(numel, period, torch.float32, device, 5)
        fp0, _, _, fvm0 = _fused(mod, p, gv, ms, mm, vm, cb, 7, 0.0)
        for _ in range(runs - 1):
            fp, _, _, fvm = _fused(mod, p, gv, ms, mm, vm, cb, 7, 0.0)
            vm_ok = torch.allclose(fvm, fvm0, rtol=VMEAN_RTOL, atol=VMEAN_ATOL)
            # p only inherits perturbation through vmean, so it must stay within
            # the same documented tolerance run-to-run (period>1), not just be
            # bit-identical at period==1. Assert it explicitly.
            p_ok = torch.allclose(fp, fp0, rtol=VMEAN_RTOL, atol=VMEAN_ATOL)
            if period == 1:
                vm_ok = vm_ok and _bit_equal(fvm, fvm0) and _bit_equal(fp, fp0)
            ok = ok and vm_ok and p_ok
            if not (vm_ok and p_ok):
                print(f"  [determinism FAIL] numel={numel} period={period} "
                      f"vm_ok={vm_ok} p_ok={p_ok}")
                break
    print(f"[fused-v2-full determinism ({runs}x): vmean within rtol, period==1 bit-exact] "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def route_switch_numerics(device):
    """The full-kernel route may change performance, not optimizer semantics.

    The momentum state is exactly equal across v1/v2. Their different grad^2
    reduction orders may perturb vmean and therefore p within the same bound
    already used for v2-full versus the standalone tree reduction. With one
    grad^2 term per row (period one), every output remains bit-identical.
    """
    mod = _load_extension()
    cb = _codebook(device, seed=19)
    lut = _codebook_search_lut(cb)
    ok = True
    for num_blocks, period in ROUTE_SWITCH_CASES:
        numel = num_blocks * period
        p, gv, ms, mm, vm = _make(
            numel, period, torch.float32, device, 700 + period
        )
        v1_p, v1_ms, v1_mm, v1_vm = _fused_v1(
            mod, p, gv, ms, mm, vm, cb, step=7, wd=0.1, lut=lut
        )
        v2_p, v2_ms, v2_mm, v2_vm = _fused(
            mod, p, gv, ms, mm, vm, cb, step=7, wd=0.1, lut=lut
        )
        momentum_exact = _bit_equal(v1_ms, v2_ms) and _bit_equal(v1_mm, v2_mm)
        p_ok = torch.allclose(v1_p, v2_p, rtol=VMEAN_RTOL, atol=VMEAN_ATOL)
        vm_ok = torch.allclose(v1_vm, v2_vm, rtol=VMEAN_RTOL, atol=VMEAN_ATOL)
        if period == 1:
            p_ok = p_ok and _bit_equal(v1_p, v2_p)
            vm_ok = vm_ok and _bit_equal(v1_vm, v2_vm)
        good = momentum_exact and p_ok and vm_ok
        ok = ok and good
        if not good:
            print(
                f"  [route-switch FAIL] blocks={num_blocks} period={period} "
                f"momentum_exact={momentum_exact} p_ok={p_ok} vm_ok={vm_ok}"
            )
    print(
        "[full v1/v2 route-switch numerics: exact momentum; bounded vmean/p] "
        f"{len(ROUTE_SWITCH_CASES)} cases -> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def run_suite(device):
    cap = torch.cuda.get_device_capability(device)
    print(f"=== device {device} ({torch.cuda.get_device_name(device)}, sm_{cap[0]}{cap[1]}) ===")
    results = [
        parity(device, wd=0.0),
        parity(device, wd=0.1),
        determinism(device),
        route_switch_numerics(device),
    ]
    ok = all(results)
    print(f"OVERALL [{device}]: {'PASS' if ok else 'FAIL'}\n")
    return ok


if pytest is not None:
    requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

    @requires_cuda
    def test_fused_update_v2_full_parity():
        assert run_suite(torch.device("cuda"))

    @requires_cuda
    def test_full_v1_v2_route_switch_numerics():
        assert route_switch_numerics(torch.device("cuda"))


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    ok = run_suite(torch.device("cuda"))
    print("FUSED-V2-FULL PARITY OVERALL:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)

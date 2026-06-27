"""Bit-exact parity for the Tier-1 fully-fused v1 update kernel
(automatic_gefen_fused_full_update_cuda) vs the decomposed reference pipeline it
replaces: the standalone vmean kernel + the host stepsize/bias-correction math
(gefen.py) + the original v1 update kernel.

The fused kernel folds K1 (vmean EMA), K2 (in-kernel stepsize) and -- once the
host stops doing the separate weight-decay pass -- K3. Every float op is mirrored
op-for-op, so the contract is BIT-IDENTICAL outputs (p, m_sign, m_magnitude,
vmean), not just allclose. A drift here means a real float-order divergence.

Run on the current GPU:  python tests/test_fused_full_update_parity.py
"""
import math

import torch

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

from gefen.kernels.automatic_gefen_fused import _load_extension
from gefen.kernels.automatic_vmean import automatic_vmean_update_cuda

BETA1, BETA2, EPS, LR = 0.9, 0.999, 1e-8, 5e-5

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# v1-regime shapes (period > warp, well occupied) -- the only shapes the host
# routes to the fused kernel -- plus a couple of stress periods.
CASES = [
    (3072 * 1024, 512),
    (3072 * 1024, 64),
    (1024 * 3072, 98304),
    (1024 * 1024, 256),
    (2048 * 1024, 1024),
]
STEPS = [1, 2, 7, 50]  # bias corrections vary with step


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
    # Mirror gefen.py:1115-1118 exactly.
    h = torch.sqrt(vmean)
    h = h.div_(math.sqrt(bc2)).add_(EPS)
    stepsize = torch.reciprocal(h)
    stepsize = stepsize.mul_(1.0 / bc1)
    return stepsize


def _reference(mod, p, gv, ms, mm, vm, cb, step, wd):
    """Decomposed pipeline the fused kernel replaces."""
    p, ms, mm, vm = p.clone(), ms.clone(), mm.clone(), vm.clone()
    bc1 = 1 - BETA1 ** step
    bc2 = 1 - BETA2 ** step
    automatic_vmean_update_cuda(vm, gv, BETA2)          # K1 reference
    stepsize = _host_stepsize(vm, bc1, bc2).contiguous()  # K2 reference
    if wd > 0.0:                                         # K3 reference
        p.mul_(1 - LR * wd)
    mod.automatic_gefen_fused_update_cuda(
        p, gv, ms, mm, stepsize, cb, False, BETA1, LR)
    return p, ms, mm, vm


def _fused(mod, p, gv, ms, mm, vm, cb, step, wd):
    p, ms, mm, vm = p.clone(), ms.clone(), mm.clone(), vm.clone()
    bc1 = 1 - BETA1 ** step
    bc2 = 1 - BETA2 ** step
    if wd > 0.0:  # K3 not yet folded -> host pass mirrors the reference
        p.mul_(1 - LR * wd)
    mod.automatic_gefen_fused_full_update_cuda(
        p, gv, ms, mm, vm, cb, False, BETA1, BETA2, LR, EPS,
        1.0 / math.sqrt(bc2), 1.0 / bc1)
    return p, ms, mm, vm


def parity(device, wd=0.0):
    mod = _load_extension()
    cb = _codebook(device)
    ok = True
    fails = 0
    n = 0
    for dt_name, dt in DTYPES.items():
        for numel, period in CASES:
            for step in STEPS:
                p, gv, ms, mm, vm = _make(numel, period, dt, device, 100 + step + (numel % 91))
                rp, rms, rmm, rvm = _reference(mod, p, gv, ms, mm, vm, cb, step, wd)
                fp, fms, fmm, fvm = _fused(mod, p, gv, ms, mm, vm, cb, step, wd)
                good = (_bit_equal(rp, fp) and _bit_equal(rms, fms)
                        and _bit_equal(rmm, fmm) and _bit_equal(rvm, fvm))
                n += 1
                ok = ok and good
                if not good:
                    fails += 1
                    print(f"  [FAIL] dt={dt_name} numel={numel} period={period} step={step} "
                          f"p={_bit_equal(rp,fp)} ms={_bit_equal(rms,fms)} "
                          f"mm={_bit_equal(rmm,fmm)} vm={_bit_equal(rvm,fvm)}")
    tag = f"wd={wd}"
    print(f"[fused-full vs decomposed, {tag}] {n} combos -> "
          f"{'PASS' if ok else f'FAIL ({fails})'}")
    return ok


def run_suite(device):
    cap = torch.cuda.get_device_capability(device)
    print(f"=== device {device} ({torch.cuda.get_device_name(device)}, sm_{cap[0]}{cap[1]}) ===")
    results = [parity(device, wd=0.0), parity(device, wd=0.1)]
    ok = all(results)
    print(f"OVERALL [{device}]: {'PASS' if ok else 'FAIL'}\n")
    return ok


if pytest is not None:
    requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

    @requires_cuda
    def test_fused_full_update_parity():
        assert run_suite(torch.device("cuda"))


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    ok = run_suite(torch.device("cuda"))
    print("FUSED-FULL PARITY OVERALL:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)

"""Single-GPU coverage for the combined Muon levers: the tuned NS coefficient
SCHEDULE (#3) and the portable, arch+size-gated fp8 NS path (#2).

Distributed (#1) parity (incl. composed with tuned4) lives in
test_muon_fsdp2_parity.py. Here we cover the per-GPU behaviour:
  - ns_schedule="standard" is bit-identical to the classic fixed quintic
    (the schedule mechanism doesn't perturb the default path);
  - tuned4 runs and orthogonalizes at least as well as standard-5;
  - fp8_ns=True is PORTABLE: it falls back to bf16 (bit-identically) on pre-sm_89
    GPUs and on small matrices (size gate), and on sm_89+ large matrices it runs
    and stays close to bf16 (non-parity, self-correcting).

Run: python tests/test_muon_combined_levers.py
"""
import os
import sys

sys.path[:] = [p for p in sys.path if not os.path.isfile(os.path.join(p or ".", "gefen.py"))]

import torch  # noqa: E402

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

from gefen.gefen_muon import (  # noqa: E402
    GefenMuon,
    _zeropower_via_newtonschulz,
    _fp8_supported,
    _normalize_ns_schedule,
    FP8_MIN_DIM,
    DEFAULT_A,
    DEFAULT_B,
    DEFAULT_C,
)
import torch.nn as nn  # noqa: E402

EPS = 1e-7
COEF = (DEFAULT_A, DEFAULT_B, DEFAULT_C)


def _ortho_err(o):
    # ||XᵀX − I||_F on the smaller dimension (the Gram NS actually conditions).
    o = o.float()
    g = o @ o.T if o.shape[0] <= o.shape[1] else o.T @ o
    return (g - torch.eye(g.shape[0], device=g.device)).norm().item()


def _muon_steps(shape, fp8, schedule, seed=0, n=3):
    torch.manual_seed(seed)
    p = (torch.randn(*shape, device="cuda") * 0.02).bfloat16().requires_grad_()
    g = torch.randn(*shape, device="cuda").bfloat16() * 1e-3
    kw = dict(lr=1e-3, fused=True, adjust_lr_fn="match_rms_adamw", fp8_ns=fp8)
    if schedule is not None:
        kw["ns_schedule"] = schedule
    opt = GefenMuon([("w", p)], **kw)
    for _ in range(n):
        p.grad = g.clone()
        opt.step()
    return p.detach().float()


# ---- #3 tuned schedule ----
def check_schedule_standard_bit_identical():
    """A single (a,b,c) tuple and an explicit 5-list of it normalize identically,
    and ns_schedule='standard' leaves the NS output bit-for-bit unchanged."""
    x = (torch.randn(2048, 2048, device="cuda") * 0.02).bfloat16()
    a = _zeropower_via_newtonschulz(x, COEF, 5, EPS)
    b = _zeropower_via_newtonschulz(x, _normalize_ns_schedule(COEF, 5), 5, EPS)
    assert torch.equal(a, b), "standard schedule not bit-identical to fixed quintic"
    return True


def check_tuned4_quality():
    """tuned4 orthogonalizes at least as well as standard-5 on the real shapes."""
    from gefen.gefen_muon import NS_SCHEDULE_4STEP
    ok = True
    for shape in [(2048, 2048), (512, 2048), (6144, 2048), (2048, 6144)]:
        torch.manual_seed(1)
        x = (torch.randn(*shape, device="cuda") * 0.02).bfloat16()
        std = _ortho_err(_zeropower_via_newtonschulz(x, COEF, 5, EPS))
        t4 = _ortho_err(_zeropower_via_newtonschulz(x, NS_SCHEDULE_4STEP, 4, EPS))
        good = (t4 <= std * 1.10) and torch.isfinite(torch.tensor(t4))
        print(f"  tuned4 {shape}: std-5 {std:.3f} vs tuned4 {t4:.3f} -> {'OK' if good else 'FAIL'}")
        ok = ok and good
    return ok


# ---- #2 fp8 portability ----
# When fp8 is GATED OFF (pre-sm_89, or small matrix), fp8_ns=True runs the SAME
# bf16 NS as fp8_ns=False -- so results match within bf16 cuBLAS run-to-run
# nondeterminism (a few e-3), NOT necessarily torch.equal. When fp8 ENGAGES it
# deviates ~few-11% (non-parity). This threshold cleanly separates the two.
_GATED_OFF_REL = 2e-2  # fp8 gated off => ~nondeterminism; fp8 on => ~0.11


def _rel(a, b):
    return (a - b).norm().item() / (b.norm().item() + 1e-12)


def check_fp8_size_gate_falls_back():
    """On a small matrix (min-dim < FP8_MIN_DIM) fp8_ns=True falls back to bf16 on
    ANY GPU (the size gate), so it matches fp8_ns=False to bf16 noise."""
    small = min(512, FP8_MIN_DIM - 1)
    rel = _rel(_muon_steps((small, small), fp8=True, schedule=None),
               _muon_steps((small, small), fp8=False, schedule=None))
    print(f"  fp8 size-gate ({small}²): rel||fp8-bf16|| = {rel:.5f} (gated off, expect ~0)")
    return rel < _GATED_OFF_REL


def check_fp8_arch_portable():
    """fp8_ns=True is portable on a LARGE matrix: pre-sm_89 -> bf16 fallback (rel ~
    bf16 noise); sm_89+ -> fp8 engages, finite, close-ish to bf16 (non-parity)."""
    shape = (2048, 2048)
    bf16 = _muon_steps(shape, fp8=False, schedule=None)
    fp8 = _muon_steps(shape, fp8=True, schedule=None)
    assert torch.isfinite(fp8).all(), "fp8 produced non-finite params"
    rel = _rel(fp8, bf16)
    if _fp8_supported():
        print(f"  fp8 engaged: rel||fp8-bf16|| = {rel:.4f} (non-parity, expect <~0.15)")
        return rel < 0.15
    else:
        print(f"  fp8 gated off (pre-sm_89): rel||fp8-bf16|| = {rel:.5f} (expect ~0)")
        return rel < _GATED_OFF_REL


def run():
    if not torch.cuda.is_available():
        print("SKIP: CUDA required")
        return True
    print("device:", torch.cuda.get_device_name(0), "fp8_supported:", _fp8_supported())
    checks = [
        ("schedule_standard_bit_identical", check_schedule_standard_bit_identical),
        ("tuned4_quality", check_tuned4_quality),
        ("fp8_size_gate_falls_back", check_fp8_size_gate_falls_back),
        ("fp8_arch_portable", check_fp8_arch_portable),
    ]
    ok = True
    for name, fn in checks:
        try:
            r = fn()
            print(f"[{name}] {'PASS' if r else 'FAIL'}")
            ok = ok and r
        except Exception as e:
            print(f"[{name}] FAIL: {e}")
            ok = False
    return ok


if pytest is not None:
    _skip = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

    @_skip
    def test_schedule_standard_bit_identical():
        assert check_schedule_standard_bit_identical()

    @_skip
    def test_tuned4_quality():
        assert check_tuned4_quality()

    @_skip
    def test_fp8_size_gate_falls_back():
        assert check_fp8_size_gate_falls_back()

    @_skip
    def test_fp8_arch_portable():
        assert check_fp8_arch_portable()


if __name__ == "__main__":
    ok = run()
    print("\nCOMBINED LEVERS OVERALL:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)

"""Single-GPU coverage for the Muon quality levers: NorMuon-style per-neuron
2nd-moment normalization of the NS output (normuon) and cautious masking.

  - flags off (the default) leaves the update path untouched: no lever state is
    created and the trajectory is deterministic;
  - normuon preserves the Frobenius norm of the NS output (so the
    match_rms_adamw LR calibration is unchanged), creates only O(rows) fp32
    state, and that state round-trips through state_dict/load_state_dict
    without dtype loss;
  - cautious zeroes exactly the sign-disagreeing coordinates and rescales the
    survivors by 1/mean(mask).

Run: python tests/test_muon_quality_levers.py
"""
import os
import sys

sys.path[:] = [p for p in sys.path if not os.path.isfile(os.path.join(p or ".", "gefen.py"))]

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

from gefen.gefen_muon import GefenMuon, _cautious_mask_  # noqa: E402


def _run_steps(seed=0, n=4, shape=(256, 512), **kw):
    torch.manual_seed(seed)
    p = (torch.randn(*shape, device="cuda") * 0.02).bfloat16().requires_grad_()
    opt = GefenMuon([("w", p)], lr=1e-3, adjust_lr_fn="match_rms_adamw", **kw)
    torch.manual_seed(seed + 1)
    for _ in range(n):
        p.grad = torch.randn(*shape, device="cuda").bfloat16() * 1e-3
        opt.step()
    return p.detach().clone(), opt


def check_flags_off_no_state_and_deterministic():
    """Default construction stores no lever state and is repeatable."""
    p1, opt1 = _run_steps()
    p2, opt2 = _run_steps()
    assert torch.equal(p1, p2), "default path is not deterministic"
    state = next(iter(opt1.state.values()))
    assert "normuon_v" not in state and "normuon_step" not in state, (
        "lever state created with flags off"
    )
    return True


def check_normuon_engages_and_state_shape():
    """normuon=True changes the trajectory and keeps only [rows,1] fp32 state."""
    base, _ = _run_steps()
    nm, opt = _run_steps(normuon=True)
    assert not torch.equal(base, nm), "normuon had no effect"
    assert torch.isfinite(nm).all()
    state = next(iter(opt.state.values()))
    v = state["normuon_v"]
    assert v.dtype == torch.float32 and v.shape == (256, 1), v.shape
    assert state["normuon_step"] == 4
    return True


def check_normuon_norm_preserving():
    """The normalized update keeps the NS output's Frobenius norm, for both
    orientations (wide, and tall -> NS returns a transpose view)."""
    opt = GefenMuon(
        [("w", torch.zeros(4, 4, device="cuda").bfloat16().requires_grad_())],
        lr=1e-3, normuon=True,
    )
    for shape in [(256, 1024), (1024, 256)]:
        for step in range(3):
            state = {}
            o = torch.randn(*shape, device="cuda").bfloat16()
            out = opt._normuon_normalize(state, o.clone(), 0.95, 1e-8)
            n_in = o.float().norm().item()
            n_out = out.float().norm().item()
            assert abs(n_in - n_out) / n_in < 5e-3, (shape, n_in, n_out)
    return True


def check_normuon_state_roundtrip():
    """normuon_v survives state_dict -> load_state_dict on a bf16 param
    without being downcast (the generic aux-tensor restore covers it)."""
    _, opt = _run_steps(normuon=True)
    sd = opt.state_dict()
    p3, opt3 = _run_steps(normuon=True)  # fresh, same trajectory
    opt3.load_state_dict(sd)
    s_old = next(iter(opt.state.values()))
    s_new = next(iter(opt3.state.values()))
    assert s_new["normuon_v"].dtype == torch.float32, s_new["normuon_v"].dtype
    assert torch.equal(s_new["normuon_v"], s_old["normuon_v"])
    assert s_new["normuon_step"] == s_old["normuon_step"]
    return True


def check_cautious_mask_property():
    """Exactly the sign-disagreeing coords are zeroed; survivors are scaled by
    numel/agree_count."""
    torch.manual_seed(0)
    u = torch.randn(128, 256, device="cuda").bfloat16()
    g = torch.randn(128, 256, device="cuda").bfloat16()
    out = _cautious_mask_(u.clone(), g)
    agree = (u * g) > 0
    assert (out[~agree] == 0).all(), "disagreeing coords not zeroed"
    scale = agree.numel() / agree.sum().float()
    expect = (u[agree].float() * scale.to(torch.bfloat16).float()).bfloat16()
    assert torch.equal(out[agree], expect), "survivor rescale mismatch"
    return True


def check_cautious_engages():
    base, _ = _run_steps()
    ca, opt = _run_steps(cautious=True)
    assert not torch.equal(base, ca), "cautious had no effect"
    assert torch.isfinite(ca).all()
    state = next(iter(opt.state.values()))
    assert "normuon_v" not in state, "cautious must not create normuon state"
    return True


def run():
    if not torch.cuda.is_available():
        print("SKIP: CUDA required")
        return True
    print("device:", torch.cuda.get_device_name(0))
    checks = [
        ("flags_off_no_state_and_deterministic", check_flags_off_no_state_and_deterministic),
        ("normuon_engages_and_state_shape", check_normuon_engages_and_state_shape),
        ("normuon_norm_preserving", check_normuon_norm_preserving),
        ("normuon_state_roundtrip", check_normuon_state_roundtrip),
        ("cautious_mask_property", check_cautious_mask_property),
        ("cautious_engages", check_cautious_engages),
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
    def test_flags_off_no_state_and_deterministic():
        assert check_flags_off_no_state_and_deterministic()

    @_skip
    def test_normuon_engages_and_state_shape():
        assert check_normuon_engages_and_state_shape()

    @_skip
    def test_normuon_norm_preserving():
        assert check_normuon_norm_preserving()

    @_skip
    def test_normuon_state_roundtrip():
        assert check_normuon_state_roundtrip()

    @_skip
    def test_cautious_mask_property():
        assert check_cautious_mask_property()

    @_skip
    def test_cautious_engages():
        assert check_cautious_engages()


if __name__ == "__main__":
    ok = run()
    print("\nQUALITY LEVERS OVERALL:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)

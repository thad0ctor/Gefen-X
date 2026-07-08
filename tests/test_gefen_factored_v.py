"""Coverage for the factored-second-moment (Adafactor-style) Gefen path
(factored_v_2d): fused kernel vs decomposed fallback consistency, single-step
quantize adjacency, bounded transient memory, and state/flag hygiene.

The fused kernel is the canonical numerics for this mode. The decomposed
fallback differs benignly: it dequantizes momentum coefficients through a bf16
template (the nonfused-path convention) while the kernel reads the fp32
codebook directly (the fused-path convention), and the momentum op association
differs (lerp vs fma). The 8-bit quantizer discretizes both, so single-step
index flips must land on ADJACENT codewords and multi-step trajectories stay
close in aggregate, not bit-for-bit.

Run: python tests/test_gefen_factored_v.py
"""

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

from gefen import Gefen  # noqa: E402


def _run(fused, steps=6, shape=(768, 512), seed=3):
    torch.manual_seed(seed)
    p = nn.Parameter((torch.randn(*shape, device="cuda") * 0.02).bfloat16())
    opt = Gefen([("w", p)], lr=1e-3, fused=fused, factored_v_2d=True,
                weight_decay=0.1)
    torch.manual_seed(seed + 1)
    for _ in range(steps):
        p.grad = torch.randn(*shape, device="cuda").bfloat16() * 1e-3
        opt.step()
    return p.detach().float(), opt


def check_kernel_vs_fallback_aggregate():
    pk, ok = _run(fused=True)
    pf, of = _run(fused=False)
    sk = next(iter(ok.state.values()))
    sf = next(iter(of.state.values()))
    assert torch.allclose(sk["v_row"], sf["v_row"], rtol=1e-5, atol=1e-12)
    assert torch.allclose(sk["v_col"], sf["v_col"], rtol=1e-5, atol=1e-12)
    rel_mean = ((pk - pf).abs().mean() / pf.abs().mean()).item()
    rel_max = ((pk - pf).abs().max() / pf.abs().max()).item()
    print(f"  6-step kernel-vs-fallback rel diff: mean {rel_mean:.2e} max {rel_max:.2e}")
    assert torch.isfinite(pk).all()
    assert torch.isfinite(pf).all()
    assert rel_mean < 2e-3, rel_mean
    assert rel_max < 5e-2, rel_max
    return True


def check_single_step_adjacency():
    _, ok = _run(fused=True, steps=1)
    _, of = _run(fused=False, steps=1)
    sk = next(iter(ok.state.values()))
    sf = next(iter(of.state.values()))
    didx = (sk["m_codebook"].short() - sf["m_codebook"].short()).abs()
    adjacent = (didx <= 1).float().mean().item()
    print(f"  single-step m_sign within-1-codeword: {adjacent*100:.4f}%")
    assert adjacent > 0.9999, adjacent
    return True


def check_transient_memory_bounded():
    shape = (4096, 4096)
    p = nn.Parameter((torch.randn(*shape, device="cuda") * 0.02).bfloat16())
    opt = Gefen([("w", p)], lr=1e-3, fused=True, factored_v_2d=True)
    p.grad = torch.randn(*shape, device="cuda").bfloat16() * 1e-3
    opt.step()  # state init
    p.grad = torch.randn(*shape, device="cuda").bfloat16() * 1e-3
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    opt.step()
    torch.cuda.synchronize()
    extra = (torch.cuda.max_memory_allocated() - base) / 2**20
    full_fp32 = shape[0] * shape[1] * 4 / 2**20
    print(f"  transient {extra:.1f} MiB (full fp32 tensor = {full_fp32:.0f} MiB)")
    assert extra < 0.35 * full_fp32, extra
    return True


def check_state_and_flag_hygiene():
    _, opt = _run(fused=True, steps=2)
    st = next(iter(opt.state.values()))
    assert st["v_row"].dtype == torch.float32 and st["v_row"].shape == (768,)
    assert st["v_col"].dtype == torch.float32 and st["v_col"].shape == (512,)
    assert "vmean" not in st, "factored param must not carry block vmean"
    # factored_v_2d is DEFAULT-ON: default construction carries factored
    # state; the explicit opt-out restores the legacy block-vmean state.
    torch.manual_seed(3)
    p = nn.Parameter((torch.randn(64, 32, device="cuda") * 0.02).bfloat16())
    opt2 = Gefen([("w", p)], lr=1e-3, fused=True)
    p.grad = torch.randn(64, 32, device="cuda").bfloat16() * 1e-3
    opt2.step()
    st2 = next(iter(opt2.state.values()))
    assert "v_row" in st2 and "vmean" not in st2, "default must be factored"
    torch.manual_seed(3)
    p3 = nn.Parameter((torch.randn(64, 32, device="cuda") * 0.02).bfloat16())
    opt3 = Gefen([("w", p3)], lr=1e-3, fused=True, factored_v_2d=False)
    p3.grad = torch.randn(64, 32, device="cuda").bfloat16() * 1e-3
    opt3.step()
    st3 = next(iter(opt3.state.values()))
    assert "v_row" not in st3 and "vmean" in st3
    return True


def check_tall_matrix_grid_stride():
    """rows > 65535 * TILE_ROWS exceeds CUDA's grid.y with one tile per
    block-row; the stats kernel must cover the tail via its y-grid-stride.
    Validate row/col sums against a torch reference on a skinny tall param."""
    rows, cols = 600_000, 8  # > 524_280 rows forces the strided path
    torch.manual_seed(9)
    p = nn.Parameter((torch.randn(rows, cols, device="cuda") * 0.02).bfloat16())
    opt = Gefen([("w", p)], lr=1e-3, fused=True)
    g = torch.randn(rows, cols, device="cuda").bfloat16() * 1e-3
    p.grad = g.clone()
    opt.step()
    st = next(iter(opt.state.values()))
    beta2 = 0.999
    ref_row = g.float().square().mean(dim=1) * (1 - beta2)
    ref_col = g.float().square().mean(dim=0) * (1 - beta2)
    assert torch.allclose(st["v_row"], ref_row, rtol=1e-4, atol=1e-12)
    assert torch.allclose(st["v_col"], ref_col, rtol=1e-4, atol=1e-12)
    assert torch.isfinite(p).all()
    return True


def check_pre_factored_checkpoint_resume():
    """A checkpoint saved with factored_v_2d=False must resume under the
    factored default: v_row/v_col init lazily and the v bias correction uses
    the EMA's own age (factored_step), not the restored global step."""
    torch.manual_seed(5)
    p = nn.Parameter((torch.randn(96, 64, device="cuda") * 0.02).bfloat16())
    opt_old = Gefen([("w", p)], lr=1e-3, fused=True, factored_v_2d=False)
    torch.manual_seed(6)
    for _ in range(5):
        p.grad = torch.randn(96, 64, device="cuda").bfloat16() * 1e-3
        opt_old.step()
    sd = opt_old.state_dict()

    torch.manual_seed(5)
    p2 = nn.Parameter((torch.randn(96, 64, device="cuda") * 0.02).bfloat16())
    opt_new = Gefen([("w", p2)], lr=1e-3, fused=True)  # factored default
    opt_new.load_state_dict(sd)
    torch.manual_seed(7)
    for _ in range(3):
        p2.grad = torch.randn(96, 64, device="cuda").bfloat16() * 1e-3
        opt_new.step()
    st = next(iter(opt_new.state.values()))
    assert st["v_row"].shape == (96,) and st["factored_step"] == 3
    assert st["step"] == 8, st["step"]
    assert torch.isfinite(p2).all()
    return True


def run():
    if not torch.cuda.is_available():
        print("SKIP: CUDA required")
        return True
    print("device:", torch.cuda.get_device_name(0))
    checks = [
        ("kernel_vs_fallback_aggregate", check_kernel_vs_fallback_aggregate),
        ("single_step_adjacency", check_single_step_adjacency),
        ("transient_memory_bounded", check_transient_memory_bounded),
        ("state_and_flag_hygiene", check_state_and_flag_hygiene),
        ("pre_factored_checkpoint_resume", check_pre_factored_checkpoint_resume),
        ("tall_matrix_grid_stride", check_tall_matrix_grid_stride),
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
    def test_kernel_vs_fallback_aggregate():
        assert check_kernel_vs_fallback_aggregate()

    @_skip
    def test_single_step_adjacency():
        assert check_single_step_adjacency()

    @_skip
    def test_transient_memory_bounded():
        assert check_transient_memory_bounded()

    @_skip
    def test_state_and_flag_hygiene():
        assert check_state_and_flag_hygiene()

    @_skip
    def test_pre_factored_checkpoint_resume():
        assert check_pre_factored_checkpoint_resume()

    @_skip
    def test_tall_matrix_grid_stride():
        assert check_tall_matrix_grid_stride()


if __name__ == "__main__":
    ok = run()
    print("\nFACTORED-V OVERALL:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)

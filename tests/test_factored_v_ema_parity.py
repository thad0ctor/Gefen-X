"""
Bit-parity of the fused factored-v EMA kernel against the aten op chain it
replaced inside the factored launcher.

The launcher used to advance the Adafactor-style row/col second-moment EMAs
with six aten launches per 2D param per step:

    v_row.mul_(beta2).add_(row_sq.div_(cols), 1 - beta2)
    v_col.mul_(beta2).add_(col_sq.div_(rows), 1 - beta2)

gefen_factored_v_ema_cuda runs the same update as ONE kernel. The default
(capturable=False) fused step is under a bit-exact-vs-history contract, so the
kernel must reproduce the aten chain's exact rounding sequence per element:
divide-by-scalar is a reciprocal multiply with the reciprocal formed in fp32,
mul_ rounds to memory, and add_'s a + alpha*b contracts to a single fma. This
test pins that equality elementwise over randomized shapes and magnitudes from
denormal to 1e10 (where the fma-vs-mul+add distinction measurably differs).

Run on the current GPU:  python tests/test_factored_v_ema_parity.py
"""
import torch

try:
    import pytest
except ImportError:  # pragma: no cover - allows the script path without pytest
    pytest = None

CUDA_OK = torch.cuda.is_available()

if pytest is not None:
    pytestmark = pytest.mark.skipif(not CUDA_OK, reason="fused EMA needs CUDA")


def test_factored_v_ema_bit_parity():
    from gefen.kernels.automatic_gefen_fused import _load_extension

    mod = _load_extension()
    torch.manual_seed(0)
    total = 0
    for _ in range(60):
        rows = int(torch.randint(1, 5000, (1,)).item())
        cols = int(torch.randint(1, 5000, (1,)).item())
        beta2 = float(torch.rand(()) * 0.2 + 0.8)
        scale_r = torch.logspace(-30, 10, rows, device="cuda")
        scale_c = torch.logspace(-30, 10, cols, device="cuda")
        v_row = (torch.randn(rows, device="cuda") * scale_r).abs()
        v_col = (torch.randn(cols, device="cuda") * scale_c).abs()
        row_sq = (torch.randn(rows, device="cuda") * scale_r).square()
        col_sq = (torch.randn(cols, device="cuda") * scale_c).square()

        ref_r = v_row.clone().mul_(beta2).add_(
            row_sq.clone().div_(float(cols)), alpha=1 - beta2
        )
        ref_c = v_col.clone().mul_(beta2).add_(
            col_sq.clone().div_(float(rows)), alpha=1 - beta2
        )

        got_r, got_c = v_row.clone(), v_col.clone()
        mod.gefen_factored_v_ema_cuda(got_r, got_c, row_sq, col_sq, beta2)

        assert torch.equal(got_r, ref_r), (
            "v_row EMA diverged from the aten chain "
            "(rows={}, cols={}, beta2={})".format(rows, cols, beta2)
        )
        assert torch.equal(got_c, ref_c), (
            "v_col EMA diverged from the aten chain "
            "(rows={}, cols={}, beta2={})".format(rows, cols, beta2)
        )
        total += rows + cols
    assert total > 100_000  # the sweep actually covered a broad element count


if __name__ == "__main__":
    if not CUDA_OK:
        print("CUDA unavailable; skipping fused EMA parity test")
    else:
        test_factored_v_ema_bit_parity()
        print("factored-v EMA kernel: bit-identical to the aten chain")

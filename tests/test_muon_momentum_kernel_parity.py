"""Bit-exact parity for the Muon single-pass quantized-momentum kernel
(gefen_quantized_momentum_update_cuda) vs the ORIGINAL recipe it replaced.

The old GefenMuon momentum step was a two-part hack:
  1. run the generic fused update at lr==0 (quantizes m_sign / m_magnitude into
     state, writes nothing to p), then
  2. rebuild the dense momentum with a separate full-size codebook gather:
     `dequantize(m_sign) .mul_(m_magnitude)`.

The new kernel does both in one pass. This test pins the new kernel to that old
recipe and asserts BIT-IDENTICAL outputs -- the advanced m_sign, the advanced
m_magnitude, and the dense momentum fed to Newton-Schulz -- across dtypes and
occupancy regimes (periods). A drift here is a real correctness regression in the
"bit-exact" claim, not a tolerance issue, so the contract is torch.equal.

Run on the current GPU:  python tests/test_muon_momentum_kernel_parity.py
"""
import torch

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

from gefen.gefen import (
    gefen_automatic_fused_update,
    gefen_dequantize_unpacked_indices,
)
from gefen.kernels.automatic_gefen_fused import gefen_quantized_momentum_update_cuda

BETA1 = 0.9
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# (numel, period) -- a spread of occupancy regimes (period==1 elementwise, small
# warp-sized blocks, big few-block tensors) like the fused-update parity suite.
CASES = [
    (3072 * 1024, 1),
    (3072 * 1024, 64),
    (1024 * 1024, 256),
    (2048 * 1024, 1024),
    (1024 * 3072, 98304),
]


def _codebook(device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    cb = torch.randn(256, device=device, generator=g)
    return torch.sort(cb).values.float().contiguous()


def _make(numel, period, dtype, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    nb = numel // period
    gv = (torch.randn(nb, period, device=device, dtype=dtype, generator=g) * 1e-3).contiguous()
    # initial quantized state: m_sign matches grad_view's (nb, period) shape
    ms = torch.randint(0, 256, (nb, period), device=device, dtype=torch.uint8, generator=g)
    mm = (torch.rand(nb, 1, device=device, dtype=torch.float32, generator=g) * 1e-3).contiguous()
    return gv, ms, mm


def _old_recipe(grad_view, m_sign, m_magnitude, codebook, beta1):
    """generic fused update @ lr==0  ->  dequantize(m_sign) * m_magnitude."""
    ms = m_sign.clone()
    mm = m_magnitude.clone()
    # the old code's p scratch + dummy stepsize; lr==0 so p is never written
    p = grad_view.reshape(-1).clone()
    stepsize = grad_view[:, :1].detach().float().contiguous()
    gefen_automatic_fused_update(
        p=p,
        grad_view=grad_view,
        m_codebook=ms,
        m_magnitude=mm,
        stepsize=stepsize,
        codebook=codebook,
        packed_indices=False,
        beta1=beta1,
        lr=0.0,
    )
    mom = gefen_dequantize_unpacked_indices(codebook, ms, grad_view)
    mom = mom.mul_(mm)
    return ms, mm, mom


def _new_kernel(grad_view, m_sign, m_magnitude, codebook, beta1):
    ms = m_sign.clone()
    mm = m_magnitude.clone()
    mom = torch.empty(grad_view.shape, dtype=grad_view.dtype, device=grad_view.device)
    gefen_quantized_momentum_update_cuda(grad_view, ms, mm, codebook, mom, beta1)
    return ms, mm, mom


def _check(numel, period, dtype_name, device):
    dtype = DTYPES[dtype_name]
    cb = _codebook(device)
    gv, ms0, mm0 = _make(numel, period, dtype, device, seed=period + len(dtype_name))

    old_sign, old_mag, old_mom = _old_recipe(gv, ms0, mm0, cb, BETA1)
    new_sign, new_mag, new_mom = _new_kernel(gv, ms0, mm0, cb, BETA1)

    assert torch.equal(new_sign, old_sign), (
        f"m_sign mismatch [{dtype_name} numel={numel} period={period}]: "
        f"{(new_sign != old_sign).sum().item()} elems differ"
    )
    assert torch.equal(new_mag, old_mag), (
        f"m_magnitude mismatch [{dtype_name} numel={numel} period={period}]: "
        f"max|d|={(new_mag - old_mag).abs().max().item():.3e}"
    )
    assert torch.equal(new_mom, old_mom), (
        f"dense momentum mismatch [{dtype_name} numel={numel} period={period}]: "
        f"max|d|={(new_mom.float() - old_mom.float()).abs().max().item():.3e}"
    )
    return True


def run():
    if not torch.cuda.is_available():
        print("SKIP: CUDA required")
        return True
    device = "cuda"
    ok = True
    for dtype_name in DTYPES:
        for numel, period in CASES:
            try:
                _check(numel, period, dtype_name, device)
                print(f"[{dtype_name} numel={numel} period={period}] -> PASS")
            except AssertionError as e:
                ok = False
                print(f"[{dtype_name} numel={numel} period={period}] -> FAIL: {e}")
    return ok


if pytest is not None:
    import itertools

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.parametrize(
        "dtype_name,case", list(itertools.product(DTYPES, CASES))
    )
    def test_muon_momentum_kernel_bit_exact(dtype_name, case):
        assert _check(case[0], case[1], dtype_name, "cuda")


if __name__ == "__main__":
    ok = run()
    print("\nMUON MOMENTUM KERNEL PARITY OVERALL:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)

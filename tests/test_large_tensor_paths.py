"""Regression tests for the >2G-element parameter fixes (Gemma 4 per-layer
embeddings are 2.35e9 elements and hit both paths):

1. period_variance CUDA kernel: grid-strided launch so numel/period can exceed
   the 2^31-1 grid.x limit, and the thread-count clamp no longer overflows int
   for periods > 2^30. Small-size results must stay identical to the torch
   reference backend.
2. Exact-codebook histogram: chunked accumulation must be bit-exact vs the
   whole-tensor form (per-block histogram counts are independent and are
   accumulated as int64).

The huge-tensor cases allocate ~4.6 GiB on the GPU; they are skipped when
there is not enough free VRAM.
"""

import math

import pytest
import torch

import gefen.gefen as gefen_mod
from gefen.kernels.period_variance import average_within_block_variance_cuda_kernel
from gefen.partitioning import average_within_block_variance_torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

HUGE_NUMEL = 2_440_000_000  # > 2^31 - 1


def _enough_vram(bytes_needed):
    if not torch.cuda.is_available():
        return False
    free, _ = torch.cuda.mem_get_info()
    return free > bytes_needed


@requires_cuda
@pytest.mark.parametrize("input_is_squared", [True, False])
def test_period_variance_matches_torch_reference(input_is_squared):
    torch.manual_seed(0)
    for n, p in [(1024, 8), (1_000_000, 100), (4_194_304, 256), (999_936, 96)]:
        x = torch.rand(n, device="cuda", dtype=torch.float32) + 0.1
        kernel = average_within_block_variance_cuda_kernel(
            x, p, input_is_squared=input_is_squared
        )
        ref_values = x if input_is_squared else x.square()
        ref = float(average_within_block_variance_torch(ref_values, p))
        assert kernel == pytest.approx(ref, rel=1e-5), (n, p, input_is_squared)


@requires_cuda
@pytest.mark.skipif(
    not _enough_vram(int(HUGE_NUMEL * 2 * 1.2)),
    reason="needs ~5.5 GiB free VRAM for the >2^31-element tensor",
)
def test_period_variance_grid_overflow_huge_tensor():
    # Pre-fix: p=1 launches numel/1 > 2^31-1 blocks -> cudaErrorInvalidValue.
    # HUGE_NUMEL // 2 = 1.22e9 > 2^30 also covers the threads clamp, whose
    # next_power_of_two int shift overflowed pre-fix.
    x = torch.rand(HUGE_NUMEL, device="cuda", dtype=torch.bfloat16)
    for period in (1, 2, HUGE_NUMEL // 2):
        val = average_within_block_variance_cuda_kernel(
            x, period, input_is_squared=True
        )
        assert math.isfinite(val)  # no launch error, no inf/nan
    del x
    torch.cuda.empty_cache()


@requires_cuda
def test_exact_histogram_chunked_bit_exact(monkeypatch):
    # The chunked upcast must produce the identical codebook: per-block
    # histogram counts are independent, so any block-aligned chunking is exact.
    torch.manual_seed(0)
    period = 96
    flat = (torch.randn(4096 * period, device="cuda", dtype=torch.bfloat16)) * 1e-3
    grad_periods = [("w", flat, period, None)]

    kwargs = dict(
        codebook_device=torch.device("cuda"),
        num_codebooks=16,
        force_endpoints=False,
        verbose=False,
        compute_mse_logging=False,
    )
    whole = gefen_mod.learn_gefen_exact_codebook_from_grad_periods(
        grad_periods=grad_periods, **kwargs
    )
    monkeypatch.setattr(gefen_mod, "GEFEN_EXACT_HISTOGRAM_CHUNK", 5 * period)
    chunked = gefen_mod.learn_gefen_exact_codebook_from_grad_periods(
        grad_periods=grad_periods, **kwargs
    )
    assert torch.equal(whole, chunked)

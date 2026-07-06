"""CPU-only coverage for period partitioning and codebook learning.

Covers the CPU backend of find_period_by_block_variance (plus its backend
dispatch errors), the memory-safe period-1 fallback, and exact_dp's numba /
pure-Python parity and validation guards. All pure CPU math, so it runs on
CPU-only CI.

Run: pytest tests/test_partitioning_quantization_cpu.py
"""
import numpy as np
import pytest
import torch

import gefen.quantization as quantization
from gefen.partitioning import (
    FALLBACK_PERIOD_CAP,
    ZeroBlockMeanError,
    average_within_block_variance_cpu,
    divisors,
    _finalize_period_results,
    find_period_by_block_variance,
    memory_safe_fallback_period,
)
from gefen.quantization import exact_dp


# ---------------------------------------------------------------------------
# partitioning
# ---------------------------------------------------------------------------

def test_divisors_excludes_n_itself():
    # period == numel would be one block covering the whole tensor; the
    # candidate list deliberately stops short of n.
    assert divisors(12) == [1, 2, 3, 4, 6]
    assert divisors(13) == [1]
    assert divisors(1) == []
    assert divisors(16) == [1, 2, 4, 8]


def test_zero_block_mean_error_is_value_error():
    assert issubclass(ZeroBlockMeanError, ValueError)


def test_average_within_block_variance_cpu():
    values = [1.0, 1.0, 2.0, 2.0]
    assert average_within_block_variance_cpu(values, 2) == pytest.approx(0.0)
    assert average_within_block_variance_cpu(values, 4) > 0.0


def test_find_period_cpu_backend_engages_memory_safe_fallback():
    # Repeated-block data drives the weighted-error rule to a tiny period,
    # which _finalize_period_results clamps to 1; MEMORY_SAFE_FALLBACK then
    # replaces the ~9 B/param period-1 outcome with the fallback period.
    rng = np.random.default_rng(0)
    block_values = rng.uniform(0.5, 2.0, size=512)
    values = np.repeat(block_values, 64).astype(np.float64)
    n = values.size
    period = find_period_by_block_variance(
        values, print_results=False, parameter_shape=(512, 64), backend="cpu"
    )
    assert period == memory_safe_fallback_period(n, (512, 64))
    assert period > 1 and n % period == 0


def test_find_period_unknown_backend_raises():
    with pytest.raises(ValueError, match="backend"):
        find_period_by_block_variance(
            np.ones(64), print_results=False, backend="warp-drive"
        )


def test_find_period_gpu_backend_requires_cuda_tensor():
    with pytest.raises(ValueError):
        find_period_by_block_variance(
            torch.ones(64), print_results=False, backend="gpu"
        )


def test_memory_safe_fallback_period():
    # tiny tensors stay at period 1
    assert memory_safe_fallback_period(4, (4,)) == 1
    # 2D weight: in_features is a candidate, but the largest capped divisor
    # wins when it is bigger (6400 -> 1600 <= cap 2048)
    assert memory_safe_fallback_period(6400, (100, 64)) == 1600
    # in_features wins when the capped divisor is smaller
    assert memory_safe_fallback_period(2048 * 3, (3, 2048)) == 2048
    # a large prime has no divisor in [8, cap] and no usable in_features
    assert memory_safe_fallback_period(104729, None) == 1
    # every returned period must divide n
    for n, shape in ((6400, (100, 64)), (2048 * 3, (3, 2048)), (7 * 11 * 13, None)):
        p = memory_safe_fallback_period(n, shape)
        assert n % p == 0


def test_fallback_period_cap_respected():
    p = memory_safe_fallback_period(2**20, None)
    assert p <= FALLBACK_PERIOD_CAP and 2**20 % p == 0


def test_finalize_period_results_picks_largest_error_drop():
    # Entries are scored by the err delta vs the previous candidate (the first
    # gets a positive sentinel); the single largest drop wins: 10 -> 4 at p=8.
    results = [(1, 10.0), (8, 4.0), (16, 1.0), (32, 0.9)]
    assert _finalize_period_results(results, print_results=False) == 8


def test_finalize_period_results_clamps_small_periods_to_one():
    results = [(1, 10.0), (2, 0.5), (4, 0.4)]
    assert _finalize_period_results(results, print_results=False) == 1


def test_finalize_period_results_empty_raises():
    with pytest.raises(ValueError):
        _finalize_period_results([], print_results=False)


# ---------------------------------------------------------------------------
# quantization / exact_dp
# ---------------------------------------------------------------------------

def _hist(centers, counts):
    return {
        "active_centers": torch.tensor(centers, dtype=torch.float32),
        "active_counts": torch.tensor(counts, dtype=torch.float32),
    }


def test_exact_dp_rejects_empty_histogram():
    with pytest.raises(ValueError, match="empty"):
        exact_dp(_hist([], []), num_codebooks=4)


def test_exact_dp_rejects_non_positive_counts():
    with pytest.raises(ValueError, match="positive"):
        exact_dp(_hist([0.0, 1.0], [3.0, 0.0]), num_codebooks=2)


def test_exact_dp_pads_when_codebook_exceeds_active_bins():
    centers = [-1.0, 0.0, 1.0]
    codebook = exact_dp(_hist(centers, [1.0, 1.0, 1.0]), num_codebooks=8)
    assert codebook.numel() == 8
    values = codebook.tolist()
    assert values == sorted(values)
    # the true centers must survive padding
    for c in centers:
        assert any(abs(v - c) < 1e-6 for v in values)


def test_exact_dp_recovers_separated_clusters():
    # two tight clusters + 2 codebooks -> centers land on the cluster means
    centers = [-1.01, -1.0, -0.99, 0.99, 1.0, 1.01]
    counts = [1.0, 2.0, 1.0, 1.0, 2.0, 1.0]
    codebook = exact_dp(_hist(centers, counts), num_codebooks=2, force_endpoints=False)
    assert codebook.numel() == 2
    assert codebook[0].item() == pytest.approx(-1.0, abs=0.02)
    assert codebook[1].item() == pytest.approx(1.0, abs=0.02)


def test_exact_dp_numba_and_python_paths_agree(monkeypatch):
    torch.manual_seed(0)
    centers = torch.linspace(-2.0, 2.0, 64).tolist()
    counts = (torch.rand(64) * 10 + 1).tolist()
    hist = _hist(centers, counts)

    via_numba = exact_dp(hist, num_codebooks=16)

    def _no_numba():
        raise ModuleNotFoundError("numba disabled for parity test")

    monkeypatch.setattr(quantization, "_get_exact_dp_numba_kernel", _no_numba)
    via_python = exact_dp(hist, num_codebooks=16)

    assert torch.allclose(
        via_numba.to(torch.float64), via_python.to(torch.float64), atol=1e-12
    ), "numba kernel and pure-Python exact-DP fallback diverged"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

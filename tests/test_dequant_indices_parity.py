"""
Bit-parity test for gefen_dequantize_unpacked_indices (gefen.py).

The production implementation was reformulated from `codebook[stored.long()]`
(which makes a full-size int64 copy of the uint8 indices on top of the full-size
fp32 gather) into a chunked index_select that writes straight into a preallocated
output, bounding the int64/fp32 scratch to one chunk. This test pins the
contract: it embeds the ORIGINAL advanced-indexing algorithm as a self-contained
reference and asserts the current function matches it *exactly* (torch.equal)
across sizes, both output dtypes (bf16/fp32), small codebooks, a non-contiguous
index tensor, 2D shape preservation, and a cross-device (CUDA->CPU) case. (The
defensive DTensor from_local branch is deliberately not unit-tested here; see the
note in _run.)

The function is on the per-step fused path and runs on GPU, so the suite is
skipped cleanly when CUDA is unavailable.

Run on the current GPU:  python tests/test_dequant_indices_parity.py
"""
import torch

try:
    import pytest
except ImportError:  # pragma: no cover - allows the script path without pytest
    pytest = None

from gefen.gefen import (
    GEFEN_DEQUANT_GATHER_CHUNK,
    gefen_dequantize_unpacked_indices,
)


def _reference_dequantize(codebook, stored_indices, like_tensor):
    """Original advanced-indexing algorithm, kept verbatim as the parity oracle."""
    coeff = codebook[stored_indices.long()].to(
        device=like_tensor.device, dtype=like_tensor.dtype
    )
    if hasattr(like_tensor, "device_mesh") and hasattr(like_tensor, "placements"):
        coeff_cls = type(like_tensor)
        if hasattr(coeff_cls, "from_local"):
            return coeff_cls.from_local(
                coeff, like_tensor.device_mesh, like_tensor.placements
            )
    return coeff


def _sorted_codebook(k, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    cb = torch.empty(k, device=device).uniform_(-1.0, 1.0, generator=g)
    return torch.sort(torch.unique(cb)).values.float().contiguous()


def _assert_parity(codebook, stored, like):
    ref = _reference_dequantize(codebook, stored, like)
    got = gefen_dequantize_unpacked_indices(codebook, stored, like)
    assert got.dtype == like.dtype
    assert got.shape == stored.shape
    # bit-for-bit equality (covers fp32->bf16 rounding of the gathered codeword).
    assert torch.equal(got, ref), (
        f"dequant mismatch: k={codebook.numel()} shape={tuple(stored.shape)} "
        f"dtype={like.dtype} disagree={(got != ref).sum().item()}"
    )


def _run():
    device = torch.device("cuda")

    # --- sizes x output dtypes (bf16 spans >1 search chunk at 4096^2/8192^2) ----
    for rows, cols in [(4096, 4096), (8192, 8192), (4096, 14336), (257, 129)]:
        cb = _sorted_codebook(256, device, seed=0)
        g = torch.Generator(device=device).manual_seed(rows + cols)
        stored = torch.randint(
            0, cb.numel(), (rows, cols), device=device, dtype=torch.uint8, generator=g
        )
        for dt in (torch.bfloat16, torch.float32):
            like = torch.empty(rows, cols, device=device, dtype=dt)
            _assert_parity(cb, stored, like)

    # --- ragged: spans >2 chunks and is NOT a multiple of the chunk -------------
    # The shapes above are all exact multiples of GEFEN_DEQUANT_GATHER_CHUNK, so
    # the short final-chunk branch (start:stop overshooting n) is otherwise
    # untested.
    cb = _sorted_codebook(256, device, seed=2)
    ragged_n = 2 * GEFEN_DEQUANT_GATHER_CHUNK + 1234567
    g = torch.Generator(device=device).manual_seed(99)
    stored = torch.randint(
        0, cb.numel(), (ragged_n,), device=device, dtype=torch.uint8, generator=g
    )
    like = torch.empty(ragged_n, device=device, dtype=torch.bfloat16)
    _assert_parity(cb, stored, like)

    # --- small codebooks k = 1, 2 ----------------------------------------------
    for k in (1, 2):
        cb = _sorted_codebook(max(k, 2), device, seed=10 + k)[:k].contiguous()
        g = torch.Generator(device=device).manual_seed(20 + k)
        stored = torch.randint(
            0, k, (64, 64), device=device, dtype=torch.uint8, generator=g
        )
        like = torch.empty(64, 64, device=device, dtype=torch.bfloat16)
        _assert_parity(cb, stored, like)

    # --- non-contiguous index tensor (transposed view) -------------------------
    cb = _sorted_codebook(256, device, seed=3)
    g = torch.Generator(device=device).manual_seed(4)
    stored_nc = torch.randint(
        0, cb.numel(), (128, 256), device=device, dtype=torch.uint8, generator=g
    ).t()
    assert not stored_nc.is_contiguous()
    like = torch.empty(stored_nc.shape, device=device, dtype=torch.bfloat16)
    _assert_parity(cb, stored_nc, like)

    # --- 2D shape preservation (explicit) --------------------------------------
    cb = _sorted_codebook(256, device, seed=7)
    g = torch.Generator(device=device).manual_seed(8)
    stored = torch.randint(
        0, cb.numel(), (300, 101), device=device, dtype=torch.uint8, generator=g
    )
    like = torch.empty(300, 101, device=device, dtype=torch.float32)
    got = gefen_dequantize_unpacked_indices(cb, stored, like)
    assert got.shape == (300, 101) and got.dtype == torch.float32

    # --- cross-device: codebook/indices on CUDA, like_tensor on CPU -------------
    # The new code allocates the output on like_tensor.device and copy_'s the
    # CUDA gather into it, where the original used `.to(device=...)`. Pin parity.
    cb = _sorted_codebook(256, device, seed=5)
    g = torch.Generator(device=device).manual_seed(6)
    stored = torch.randint(
        0, cb.numel(), (200, 100), device=device, dtype=torch.uint8, generator=g
    )
    like_cpu = torch.empty(200, 100, device="cpu", dtype=torch.bfloat16)
    got = gefen_dequantize_unpacked_indices(cb, stored, like_cpu)
    assert got.device.type == "cpu"
    _assert_parity(cb, stored, like_cpu)

    # NOTE on the DTensor from_local branch: it is intentionally NOT unit-tested
    # here. Constructing a DTensor needs a live process group, and initializing
    # one in-process leaves global distributed state that breaks sibling tests
    # (e.g. test_fused_fsdp2_noncontig, which runs its own single-rank group) when
    # the suite shares a process. The branch is also defensive: every caller of
    # gefen_dequantize_unpacked_indices passes a to_local()'d (plain) tensor as
    # like_tensor, so the branch is unreached by current code paths. Its body is a
    # trivial pass-through to the exact `from_local` the original used (preserved
    # verbatim in the rewrite and in the oracle above), so chunking cannot change
    # its result. The real DTensor-through-optimizer behavior on the fused path is
    # covered end-to-end by test_fused_fsdp2_noncontig.

    print("[dequant indices parity] all cases PASS")
    return True


if pytest is not None:
    requires_cuda = pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required"
    )

    @requires_cuda
    def test_dequant_indices_parity():
        assert _run()


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    _run()

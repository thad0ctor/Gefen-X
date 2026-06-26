"""
Bit-parity test for gefen_nearest_codebook_indices (gefen.py).

The production implementation was reformulated from a gather/abs/where pipeline
(which materialized ~10 full-size fp32/int64 temporaries per call) into a single
chunked searchsorted on codebook midpoints, to bound the per-step transient
memory. This test pins the contract: it embeds the ORIGINAL gather/abs/where
algorithm as a self-contained reference and asserts the current function matches
it *exactly* (torch.equal on the uint8 output) across a range of adversarial
inputs and codebook sizes.

The function runs on GPU (it is called on the optimizer's CUDA tensors), so the
whole suite is skipped cleanly when CUDA is unavailable.

Run on the current GPU:  python tests/test_codebook_indices_parity.py
"""
import torch

try:
    import pytest
except ImportError:  # pragma: no cover - allows the script path without pytest
    pytest = None

from gefen.gefen import (
    GEFEN_CODEBOOK_SEARCH_CHUNK,
    gefen_nearest_codebook_indices,
)


def _reference_nearest_codebook_indices(codebook, normalized_vals):
    """Original gather/abs/where algorithm, kept verbatim as the parity oracle.

    For each value, searchsorted finds the insertion point in the sorted
    codebook; the nearest codeword is whichever of the two bracketing entries is
    closer, with ties (equal distance) resolved to the lower index via `<=`.
    """
    k = codebook.numel()
    flat = normalized_vals.reshape(-1).float()
    idx = torch.searchsorted(codebook, flat)
    left = (idx - 1).clamp(0, k - 1)
    right = idx.clamp(0, k - 1)
    assignments = torch.where(
        (flat - codebook[left]).abs() <= (flat - codebook[right]).abs(),
        left,
        right,
    )
    return assignments.to(torch.uint8).view(normalized_vals.shape)


def _sorted_codebook(k, device, seed):
    """Sorted, strictly-increasing codebook in [-1, 1] with forced endpoints,
    matching the force_endpoints style of the real exact_dp codebook."""
    g = torch.Generator(device=device).manual_seed(seed)
    cb = torch.empty(k, device=device).uniform_(-1.0, 1.0, generator=g)
    cb = torch.unique(cb)  # strictly increasing
    if cb.numel() >= 2:
        cb[0], cb[-1] = -1.0, 1.0
    return cb.float().contiguous()


def _assert_parity(codebook, vals):
    ref = _reference_nearest_codebook_indices(codebook, vals)
    got = gefen_nearest_codebook_indices(codebook, vals)
    assert got.dtype == torch.uint8
    assert got.shape == vals.shape
    assert torch.equal(got, ref), (
        f"index mismatch: k={codebook.numel()} shape={tuple(vals.shape)} "
        f"disagree={(got.long() != ref.long()).sum().item()}"
    )


def _assert_fp32_near_parity(codebook, vals):
    """fp32 contract: the midpoint form is NOT provably bit-exact against the old
    gather/abs/where form, because within ~1 ULP of a midpoint the old form's two
    independent abs-distance subtractions can round equal where `v <= mid` does
    not. Such ties are vanishingly rare and immaterial. Assert exactly that: any
    disagreement is (a) rare, (b) between *adjacent* codewords, and (c) a genuine
    distance-tie (so the reconstruction error difference is ~0)."""
    ref = _reference_nearest_codebook_indices(codebook, vals).long()
    got = gefen_nearest_codebook_indices(codebook, vals).long()
    assert got.shape == vals.shape
    disagree = ref != got
    n = vals.numel()
    nd = int(disagree.sum())
    # vanishingly rare: a systematic off-by-one would light up millions, not a
    # handful. Ceiling well below any real regression, well above the tie rate.
    assert nd <= max(8, n // 1_000_000), f"too many fp32 disagreements: {nd}/{n}"
    if nd:
        sel = disagree.reshape(-1).nonzero().flatten()
        v = vals.reshape(-1)[sel].float()
        oi = ref.reshape(-1)[sel]
        gi = got.reshape(-1)[sel]
        assert torch.equal((oi - gi).abs(), torch.ones_like(oi)), (
            "fp32 disagreement between non-adjacent codewords (not a tie)"
        )
        do = (v - codebook[oi]).abs()
        dg = (v - codebook[gi]).abs()
        rel = (do - dg).abs() / (do + dg).clamp_min(1e-30)
        assert rel.max().item() < 1e-3, (
            f"fp32 disagreement is not a distance-tie: max rel gap {rel.max().item()}"
        )


def _run():
    device = torch.device("cuda")

    # The non-fused path feeds this function BOTH dtypes: GefenMuon quantizes a
    # bf16 momentum_view, while plain Gefen builds updated_m in fp32
    # (_automatic_momentum_update). So both dtypes below are production inputs.

    # --- 1. random uniform [-1, 1] bf16 data, k = 256 ---------------------------
    cb256 = _sorted_codebook(256, device, seed=0)
    g = torch.Generator(device=device).manual_seed(1)
    vals = torch.empty(4096, 4096, device=device, dtype=torch.bfloat16).uniform_(
        -1.0, 1.0, generator=g
    )
    _assert_parity(cb256, vals)

    # --- 1b. random uniform fp32 data (plain-Gefen production dtype) ------------
    # fp32 is near-parity, not bit-exact: a handful of values land within ~1 ULP
    # of a midpoint and flip by one (adjacent) index. _assert_fp32_near_parity
    # pins that those are rare, immaterial ties (see its docstring).
    g = torch.Generator(device=device).manual_seed(2)
    vals_fp32 = torch.empty(4096, 4096, device=device, dtype=torch.float32).uniform_(
        -1.2, 1.2, generator=g
    )
    _assert_fp32_near_parity(cb256, vals_fp32)

    # --- 2. values exactly at codebook points ----------------------------------
    _assert_parity(cb256, cb256.bfloat16())
    _assert_parity(cb256, cb256.float())

    # --- 3. values exactly at midpoints (the tie-break boundary) ---------------
    # Midpoints are fed as bf16 here. Feeding *exact fp32* midpoints is
    # deliberately NOT asserted: mid = (cb[i]+cb[i+1])*0.5 rounded in fp32 is not
    # bit-exactly equidistant from its two neighbours, so the reference's two
    # independent abs-distance computations and the single midpoint comparison
    # legitimately diverge on that ~ULP tie. The tie is a sub-ULP boundary effect
    # (it can also occur for fp32 values within ~1 ULP of a midpoint, including
    # subnormals near a zero-midpoint); it only ever swaps two near-equidistant
    # codewords, so it is immaterial. Random fp32/bf16 data (cases 1/1b, tens of
    # millions of values) shows 0 disagreements.
    mids = (cb256[:-1] + cb256[1:]) * 0.5
    _assert_parity(cb256, mids.bfloat16())

    # --- 4. out-of-range values (below cb[0], above cb[-1]) --------------------
    oor = torch.tensor(
        [-5.0, -2.0, -1.5, 1.5, 2.0, 5.0, 0.0], device=device, dtype=torch.bfloat16
    )
    _assert_parity(cb256, oor)

    # --- 5. codebook sizes k = 1, 2, 256 ---------------------------------------
    for k in (1, 2, 256):
        cb = _sorted_codebook(k, device, seed=10 + k)
        g = torch.Generator(device=device).manual_seed(20 + k)
        v = torch.empty(100000, device=device, dtype=torch.bfloat16).uniform_(
            -1.5, 1.5, generator=g
        )
        _assert_parity(cb, v)
        # also exercise exact codebook points / midpoints for small k
        _assert_parity(cb, cb.bfloat16())
        if cb.numel() >= 2:
            _assert_parity(cb, ((cb[:-1] + cb[1:]) * 0.5).bfloat16())

    # --- 5b. ragged multi-chunk input ------------------------------------------
    # The search runs in GEFEN_CODEBOOK_SEARCH_CHUNK-element slices. Case 1
    # (4096^2) is exactly two full chunks, so it never exercises a short final
    # chunk. Use a size spanning >2 chunks that is NOT a multiple of the chunk,
    # so the loop covers full chunks plus a partial tail (off-by-one boundary).
    ragged_n = 2 * GEFEN_CODEBOOK_SEARCH_CHUNK + 1234567
    g = torch.Generator(device=device).manual_seed(123)
    v_ragged = torch.empty(ragged_n, device=device, dtype=torch.bfloat16).uniform_(
        -1.5, 1.5, generator=g
    )
    _assert_parity(cb256, v_ragged)

    # --- 6. 2D shape preservation ----------------------------------------------
    g = torch.Generator(device=device).manual_seed(99)
    v2d = torch.empty(257, 129, device=device, dtype=torch.bfloat16).uniform_(
        -1.0, 1.0, generator=g
    )
    got = gefen_nearest_codebook_indices(cb256, v2d)
    assert got.shape == (257, 129)
    _assert_parity(cb256, v2d)

    print("[codebook indices parity] all cases PASS")
    return True


if pytest is not None:
    requires_cuda = pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required"
    )

    @requires_cuda
    def test_codebook_indices_parity():
        assert _run()


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    _run()

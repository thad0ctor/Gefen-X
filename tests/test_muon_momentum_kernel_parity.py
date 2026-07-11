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
occupancy regimes (periods). It also pins optional fused Nesterov emission to
the former two host TensorIterator kernels while proving that Nesterov never
changes the stored EMA state. A drift here is a real correctness regression in
the "bit-exact" claim, not a tolerance issue, so the contract is torch.equal.

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
from gefen.kernels.automatic_gefen_fused import (
    _codebook_search_lut,
    gefen_quantized_momentum_update_cuda,
)

BETA1 = 0.9
NESTEROV_BETAS = (0.875, 0.9, 0.95, 0.99)
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# (numel, period) -- a spread of occupancy regimes (period==1 elementwise, small
# warp-sized blocks, big few-block tensors) like the fused-update parity suite.
CASES = [
    (3072 * 1024, 1),
    (3072 * 1024, 64),
    (1024 * 1024, 256),
    (2048 * 1024, 1024),
    (1024 * 3072, 98304),
    (1009 * 127, 127),  # non-power-of-eight period + tail chunk
    (513 * 65, 65),     # a chunk crosses block boundaries repeatedly
    (65536 * 128, 128),  # above the chunked grid-stride cap (~5.5M on sm86):
                         # every thread takes a second grid-stride pass
]

# Focused exact-Nesterov coverage around the scalar/chunked emitter boundary.
# These are intentionally smaller than CASES because every entry is crossed
# with three dtypes, four betas, and LUT/no-LUT modes.
NESTEROV_CASES = [
    (257 * 64, 64),   # scalar emitter (cutoff is inclusive)
    (513 * 65, 65),   # chunked emitter; chunks repeatedly cross EMA blocks
    (1009 * 127, 127),  # odd chunk tail + non-power-of-two period
]

# Chunked-emitter tail remainders + degenerate tiny tensors.  CASES only hits
# numel % 8 in {0, 1, 7}; nb=7 with periods 66..70 walks the remaining tail
# remainders 6..2 through the 8-per-thread chunked emitter, and the tiny
# shapes (numel 1..16) pin the launch/grid math at the small extreme.  Each
# entry is checked plain AND Nesterov (single beta) so both emitters see the
# tail.
TAIL_TINY_CASES = [
    (7 * 66, 66),   # numel % 8 == 6
    (7 * 67, 67),   # numel % 8 == 5
    (7 * 68, 68),   # numel % 8 == 4
    (7 * 69, 69),   # numel % 8 == 3
    (7 * 70, 70),   # numel % 8 == 2
    (1 * 65, 65),   # one block, one partial chunk
    (1, 1),
    (3, 1),
    (1, 3),
    (1, 16),
    (3 * 5, 5),
    (16, 1),
    (2 * 7, 7),
]

# (tensor-to-offset, dtype, offset in elements) -- each variant breaks the
# 16B (8B for m_sign) alignment the vectorized path requires, forcing the
# scalar chunk fallback.  The bf16-grad case lives in
# _check_unaligned_contiguous_view; these cover the remaining pointers/dtypes.
MISALIGNED_VARIANTS = [
    ("grad", "fp32", 1),
    ("grad", "fp16", 1),
    ("m_sign", "bf16", 4),
    ("mom_out", "fp32", 1),
    ("mom_out", "bf16", 1),
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


def _old_recipe(
    grad_view, m_sign, m_magnitude, codebook, beta1, *, nesterov=False
):
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
    if nesterov:
        mom.mul_(beta1).add_(grad_view, alpha=1 - beta1)
    return ms, mm, mom


def _new_kernel(
    grad_view, m_sign, m_magnitude, codebook, beta1, codebook_lut=None,
    *, stochastic_round=False, rng_seed=0, seed_dev=None, nesterov=False,
):
    ms = m_sign.clone()
    mm = m_magnitude.clone()
    mom = torch.empty(grad_view.shape, dtype=grad_view.dtype, device=grad_view.device)
    gefen_quantized_momentum_update_cuda(
        grad_view, ms, mm, codebook, mom, beta1,
        stochastic_round=stochastic_round,
        rng_seed=rng_seed,
        seed_dev=seed_dev,
        codebook_lut=codebook_lut,
        nesterov=nesterov,
    )
    return ms, mm, mom


def _check(numel, period, dtype_name, device):
    dtype = DTYPES[dtype_name]
    cb = _codebook(device)
    gv, ms0, mm0 = _make(numel, period, dtype, device, seed=period + len(dtype_name))

    old_sign, old_mag, old_mom = _old_recipe(gv, ms0, mm0, cb, BETA1)
    lut_real = _codebook_search_lut(cb)
    lut_empty = torch.empty(0, dtype=torch.int16, device=device)
    for lut_name, lut in (("lut", lut_real), ("nolut", lut_empty)):
        new_sign, new_mag, new_mom = _new_kernel(
            gv, ms0, mm0, cb, BETA1, codebook_lut=lut
        )

        assert torch.equal(new_sign, old_sign), (
            f"m_sign mismatch [{dtype_name} numel={numel} period={period} "
            f"{lut_name}]: {(new_sign != old_sign).sum().item()} elems differ"
        )
        assert torch.equal(new_mag, old_mag), (
            f"m_magnitude mismatch [{dtype_name} numel={numel} period={period} "
            f"{lut_name}]: max|d|={(new_mag - old_mag).abs().max().item():.3e}"
        )
        assert torch.equal(new_mom, old_mom), (
            f"dense momentum mismatch [{dtype_name} numel={numel} period={period} "
            f"{lut_name}]: max|d|="
            f"{(new_mom.float() - old_mom.float()).abs().max().item():.3e}"
        )
    return True


def _check_stochastic_lut_parity(device):
    cb = _codebook(device, seed=11)
    gv, ms0, mm0 = _make(1009 * 127, 127, torch.bfloat16, device, seed=19)
    real = _codebook_search_lut(cb)
    empty = torch.empty(0, dtype=torch.int16, device=device)
    for nesterov in (False, True):
        a = _new_kernel(
            gv, ms0, mm0, cb, BETA1, real,
            stochastic_round=True, rng_seed=12345, nesterov=nesterov,
        )
        b = _new_kernel(
            gv, ms0, mm0, cb, BETA1, empty,
            stochastic_round=True, rng_seed=12345, nesterov=nesterov,
        )
        assert all(torch.equal(x, y) for x, y in zip(a, b))
    return True


def _check_nesterov(numel, period, dtype_name, beta1, device):
    """The folded output must equal the two real host kernels bit-for-bit.

    Run the same quantized-state advance with Nesterov disabled/enabled.  This
    simultaneously proves that Nesterov does not leak into persistent state and
    pins the emitted tensor to the exact scalar-dtype rounding sequence used by
    ``mul_().add_()``.
    """
    dtype = DTYPES[dtype_name]
    cb = _codebook(device, seed=31)
    gv, ms0, mm0 = _make(
        numel, period, dtype, device,
        seed=period + len(dtype_name) + round(beta1 * 1000),
    )
    real = _codebook_search_lut(cb)
    empty = torch.empty(0, dtype=torch.int16, device=device)
    for lut_name, lut in (("lut", real), ("nolut", empty)):
        plain_sign, plain_mag, plain_mom = _new_kernel(
            gv, ms0, mm0, cb, beta1, lut, nesterov=False
        )
        nest_sign, nest_mag, nest_mom = _new_kernel(
            gv, ms0, mm0, cb, beta1, lut, nesterov=True
        )
        expected = plain_mom.clone()
        expected.mul_(beta1).add_(gv, alpha=1 - beta1)

        assert torch.equal(nest_sign, plain_sign), (
            f"Nesterov changed m_sign [{dtype_name} beta={beta1} "
            f"period={period} {lut_name}]"
        )
        assert torch.equal(nest_mag, plain_mag), (
            f"Nesterov changed m_magnitude [{dtype_name} beta={beta1} "
            f"period={period} {lut_name}]"
        )
        assert torch.equal(nest_mom, expected), (
            f"Nesterov output mismatch [{dtype_name} beta={beta1} "
            f"period={period} {lut_name}]: "
            f"{(nest_mom != expected).sum().item()} elems differ; max|d|="
            f"{(nest_mom.float() - expected.float()).abs().max().item():.3e}"
        )
    return True


def _check_unaligned_contiguous_view(device):
    """A contiguous storage-offset view must take the scalar chunk fallback."""
    cb = _codebook(device, seed=23)
    gv, ms0, mm0 = _make(1009 * 127, 127, torch.bfloat16, device, seed=29)
    backing = torch.empty(gv.numel() + 1, dtype=gv.dtype, device=device)
    gv_offset = backing[1:].view_as(gv)
    gv_offset.copy_(gv)
    assert gv_offset.is_contiguous() and gv_offset.data_ptr() % 16 != 0
    for nesterov in (False, True):
        old = _old_recipe(
            gv_offset, ms0, mm0, cb, BETA1, nesterov=nesterov
        )
        new = _new_kernel(
            gv_offset, ms0, mm0, cb, BETA1, _codebook_search_lut(cb),
            nesterov=nesterov,
        )
        assert all(torch.equal(x, y) for x, y in zip(old, new))
    return True


def _check_misaligned(which, dtype_name, off_elems, device):
    """Offsetting grad / m_sign / momentum_out off vector alignment must drop
    the kernel to the scalar chunk fallback with bit-identical results."""
    dtype = DTYPES[dtype_name]
    numel, period = 101 * 67, 67  # chunked emitter; tail numel % 8 == 7
    cb = _codebook(device, seed=53)
    gv, ms0, mm0 = _make(numel, period, dtype, device, seed=59)
    lut = _codebook_search_lut(cb)

    grad = gv
    if which == "grad":
        backing = torch.empty(numel + off_elems, dtype=dtype, device=device)
        grad = backing[off_elems:].view_as(gv)
        grad.copy_(gv)
        assert grad.data_ptr() % 16 != 0, "grad view unexpectedly 16B-aligned"

    for nesterov in (False, True):
        if which == "m_sign":
            backing = torch.empty(
                numel + off_elems, dtype=torch.uint8, device=device
            )
            ms = backing[off_elems:].view_as(ms0)
            ms.copy_(ms0)
            assert ms.data_ptr() % 8 != 0, "m_sign view unexpectedly 8B-aligned"
        else:
            ms = ms0.clone()
        mm = mm0.clone()
        if which == "mom_out":
            backing = torch.empty(numel + off_elems, dtype=dtype, device=device)
            mom = backing[off_elems:].view_as(gv)
            assert mom.data_ptr() % 16 != 0, "mom view unexpectedly 16B-aligned"
        else:
            mom = torch.empty_like(gv)

        gefen_quantized_momentum_update_cuda(
            grad, ms, mm, cb, mom, BETA1, codebook_lut=lut, nesterov=nesterov,
        )
        old = _old_recipe(gv, ms0, mm0, cb, BETA1, nesterov=nesterov)
        for name, o, n in zip(
            ("m_sign", "m_magnitude", "momentum"), old, (ms, mm, mom)
        ):
            assert torch.equal(n, o), (
                f"misaligned-{which} [{dtype_name} off_elems={off_elems} "
                f"nesterov={nesterov}] {name} differs from aligned reference: "
                f"{(n != o).sum().item()} elems differ"
            )
    return True


def _check_seed_dev_chunked_parity(device):
    """seed_dev (capturable device-tensor rounding seed) must be bit-identical
    to the same seed passed as host ``rng_seed`` through the chunked emitter
    path, including the grid-stride second pass -- and must override the host
    seed (host rng_seed is a decoy 0 whenever seed_dev is given)."""
    seed = 987654321
    cb = _codebook(device, seed=61)
    lut = _codebook_search_lut(cb)
    # chunked emitter with odd tail; then above the grid-stride cap (~5.5M)
    for numel, period in ((1009 * 127, 127), (65536 * 128, 128)):
        gv, ms0, mm0 = _make(numel, period, torch.bfloat16, device, seed=67)
        for nesterov in (False, True):
            host = _new_kernel(
                gv, ms0, mm0, cb, BETA1, lut,
                stochastic_round=True, rng_seed=seed, nesterov=nesterov,
            )
            dev = _new_kernel(
                gv, ms0, mm0, cb, BETA1, lut,
                stochastic_round=True, rng_seed=0,
                seed_dev=torch.tensor(seed, dtype=torch.int64, device=device),
                nesterov=nesterov,
            )
            for name, h, d in zip(("m_sign", "m_magnitude", "momentum"), host, dev):
                assert torch.equal(d, h), (
                    f"seed_dev != host rng_seed [{name} numel={numel} "
                    f"period={period} nesterov={nesterov}]"
                )
            # Sanity that the equality above has teeth: a different device
            # seed must actually change the stochastic rounding.
            other = _new_kernel(
                gv, ms0, mm0, cb, BETA1, lut,
                stochastic_round=True, rng_seed=0,
                seed_dev=torch.tensor(seed + 1, dtype=torch.int64, device=device),
                nesterov=nesterov,
            )
            assert not torch.equal(other[0], host[0]), (
                f"different seed_dev produced identical m_sign [numel={numel} "
                f"period={period} nesterov={nesterov}] -- seed not consumed"
            )
    return True


def _check_nesterov_graph_capture(device):
    """Direct raw-kernel capture/replay keeps the exact folded semantics."""
    beta1 = 0.95
    cb = _codebook(device, seed=37)
    gv, ms0, mm0 = _make(513 * 65, 65, torch.bfloat16, device, seed=41)
    lut = _codebook_search_lut(cb)
    expected = _new_kernel(
        gv, ms0, mm0, cb, beta1, lut, nesterov=True
    )

    ms = ms0.clone()
    mm = mm0.clone()
    mom = torch.empty_like(gv)
    # Warm the extension/allocator before capture, then capture only the raw op.
    _new_kernel(gv, ms0, mm0, cb, beta1, lut, nesterov=True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gefen_quantized_momentum_update_cuda(
            gv, ms, mm, cb, mom, beta1,
            codebook_lut=lut, nesterov=True,
        )
    ms.copy_(ms0)
    mm.copy_(mm0)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(ms, expected[0])
    assert torch.equal(mm, expected[1])
    assert torch.equal(mom, expected[2])
    return True


def _check_nesterov_compile(device):
    """Dynamo must preserve the new bool through the registered custom op."""
    if not hasattr(torch, "compile"):
        return True
    beta1 = 0.99
    cb = _codebook(device, seed=43)
    gv, ms0, mm0 = _make(513 * 65, 65, torch.float32, device, seed=47)
    lut = _codebook_search_lut(cb)
    expected = _new_kernel(
        gv, ms0, mm0, cb, beta1, lut, nesterov=True
    )
    ms = ms0.clone()
    mm = mm0.clone()
    mom = torch.empty_like(gv)

    def invoke(grad, sign, magnitude, codebook, out, codebook_lut):
        gefen_quantized_momentum_update_cuda(
            grad, sign, magnitude, codebook, out, beta1,
            codebook_lut=codebook_lut, nesterov=True,
        )
        return out

    compiled = torch.compile(invoke, backend="eager", fullgraph=True)
    compiled(gv, ms, mm, cb, mom, lut)
    torch.cuda.synchronize()
    assert torch.equal(ms, expected[0])
    assert torch.equal(mm, expected[1])
    assert torch.equal(mom, expected[2])
    torch._dynamo.reset()
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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.parametrize(
        "dtype_name,beta1,case",
        list(itertools.product(DTYPES, NESTEROV_BETAS, NESTEROV_CASES)),
    )
    def test_muon_momentum_nesterov_bit_exact(dtype_name, beta1, case):
        assert _check_nesterov(case[0], case[1], dtype_name, beta1, "cuda")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.parametrize(
        "dtype_name,case", list(itertools.product(DTYPES, TAIL_TINY_CASES))
    )
    def test_muon_momentum_tail_and_tiny_bit_exact(dtype_name, case):
        assert _check(case[0], case[1], dtype_name, "cuda")
        assert _check_nesterov(case[0], case[1], dtype_name, BETA1, "cuda")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.parametrize("which,dtype_name,off_elems", MISALIGNED_VARIANTS)
    def test_muon_momentum_misaligned_scalar_fallback(
        which, dtype_name, off_elems
    ):
        assert _check_misaligned(which, dtype_name, off_elems, "cuda")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_muon_momentum_seed_dev_chunked_parity():
        assert _check_seed_dev_chunked_parity("cuda")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_muon_momentum_stochastic_lut_parity():
        assert _check_stochastic_lut_parity("cuda")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_muon_momentum_unaligned_contiguous_view():
        assert _check_unaligned_contiguous_view("cuda")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_muon_momentum_nesterov_graph_capture():
        assert _check_nesterov_graph_capture("cuda")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_muon_momentum_nesterov_compile():
        assert _check_nesterov_compile("cuda")


if __name__ == "__main__":
    ok = run()
    print("\nMUON MOMENTUM KERNEL PARITY OVERALL:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)

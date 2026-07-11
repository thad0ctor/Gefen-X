"""Regression tests for per-device codebook copies (HF `device_map` sharding).

Pre-fix, the single learned codebook lived on the first param's device and the
fused kernels raised "Expected all tensors on the same device as p." for any
param on another GPU. The fix hands each step a cached per-device copy and
clears the cache at every codebook (re)assignment.
"""

import pytest
import torch

from gefen import Gefen

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
requires_two_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs 2 CUDA devices"
)


def _make_param(device, seed):
    torch.manual_seed(seed)
    p = torch.nn.Parameter(
        torch.randn(64, 96, device=device, dtype=torch.bfloat16)
    )
    torch.manual_seed(seed + 100)
    p.grad = torch.randn_like(p) * 0.01
    return p


@requires_two_gpus
def test_step_with_params_on_two_devices():
    p0 = _make_param("cuda:0", 0)
    p1 = _make_param("cuda:1", 1)
    before0, before1 = p0.detach().clone(), p1.detach().clone()

    opt = Gefen([("a", p0), ("b", p1)], lr=5e-5)
    for _ in range(3):
        opt.step()
        torch.manual_seed(7)
        p0.grad = torch.randn_like(p0) * 0.01
        p1.grad = torch.randn_like(p1) * 0.01

    assert not torch.equal(p0.detach(), before0)
    assert not torch.equal(p1.detach(), before1)
    # cached copy exists for the second device and matches the master codebook
    assert opt._gefen_codebook is not None
    cached = opt._gefen_codebook_on(p1.device)
    assert cached.device == p1.device
    assert torch.equal(cached.cpu(), opt._gefen_codebook.cpu())


@requires_two_gpus
def test_second_device_update_matches_single_device_oracle():
    # The same (param, grad) stepped on cuda:1 inside a two-device optimizer
    # must produce exactly the update a single-device cuda:0 optimizer makes:
    # the codebook copy cannot change numerics.
    pa = _make_param("cuda:0", 0)
    pb = _make_param("cuda:1", 0)  # identical values to pa, other device
    opt_multi = Gefen([("a", pa), ("b", pb)], lr=5e-5)
    opt_multi.step()

    oracle = _make_param("cuda:0", 0)
    opt_single = Gefen([("a", oracle)], lr=5e-5)
    opt_single.step()

    assert torch.equal(pb.detach().cpu(), oracle.detach().cpu())


@requires_cuda
def test_lut_cache_stays_fresh_across_codebook_refresh():
    # codebook_refresh_every=2 relearns the codebook at steps 2 and 4. The
    # cached per-device search LUT (_gefen_codebook_lut_by_device) must always
    # describe the LIVE codebook: every codebook assignment site clears the
    # cache, so a lookup after a refresh rebuilds from the new table instead
    # of serving a stale one.
    from gefen.kernels.automatic_gefen_fused import _LUT_BUCKETS

    def expected_lut(codebook):
        # Independent reconstruction from the LUT definition:
        # lut[b] = number of codebook entries whose bucket < b.
        cb = codebook.detach().float().cpu()
        buckets = torch.clamp(
            torch.floor((cb + 1.0) * (_LUT_BUCKETS / 2)), 0, _LUT_BUCKETS - 1
        )
        return torch.tensor(
            [int((buckets < b).sum()) for b in range(_LUT_BUCKETS + 1)],
            dtype=torch.int16,
        )

    device = torch.device("cuda", 0)
    torch.manual_seed(0)
    p = torch.nn.Parameter(
        (torch.randn(64, 64, device=device) * 0.02).bfloat16()
    )
    opt = Gefen([("a", p)], lr=1e-3, codebook_refresh_every=2)

    refresh_changed_codebook = False
    prev_codebook = None
    for step in range(1, 6):
        # Shift the grad distribution each step so the periodic refresh
        # learns a genuinely different codebook.
        torch.manual_seed(step)
        p.grad = torch.randn_like(p) * (0.01 * step)
        opt.step()
        lut = opt._gefen_codebook_lut_on(device)
        codebook = opt._gefen_codebook_on(device)
        assert torch.equal(lut.cpu(), expected_lut(codebook)), (
            "cached LUT is stale after step {}".format(step)
        )
        # Repeated lookups must return the same cached tensor: the eager Muon
        # momentum path relies on a stable per-device object, and capturable
        # compile marks exactly this tensor static.
        assert opt._gefen_codebook_lut_on(device) is lut
        if prev_codebook is not None and not torch.equal(
            codebook.cpu(), prev_codebook
        ):
            refresh_changed_codebook = True
        prev_codebook = codebook.detach().cpu().clone()
    # The scenario must have exercised a real refresh (a codebook that never
    # changed would make the staleness assertions vacuous).
    assert refresh_changed_codebook


@requires_cuda
def test_per_device_cache_cleared_on_codebook_assignment():
    p = _make_param("cuda:0", 0)
    opt = Gefen([("a", p)], lr=5e-5)
    opt.step()
    assert opt._gefen_codebook is not None
    # simulate a stale cache entry, then force the refresh path
    opt._gefen_codebook_by_device[torch.device("cpu")] = opt._gefen_codebook.cpu()
    torch.manual_seed(9)
    p.grad = torch.randn_like(p) * 0.01
    opt._refresh_codebook_with_requant()
    assert opt._gefen_codebook_by_device == {}


if __name__ == "__main__":
    # Delegate so the skipif marks (CUDA / 2-GPU requirements) still apply
    # when the file is executed directly.
    raise SystemExit(pytest.main([__file__]))

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

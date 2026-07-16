"""CPU-only regression coverage for plain Gefen under DeepSpeed ZeRO CPU-offload.

Under ZeRO with ``offload_optimizer`` (and, at stage 3, ``offload_param``) set to
CPU, DeepSpeed hands the *client* optimizer CPU-resident fp32 flat 1-D
partitions and calls ``optimizer.step()`` on them (requires
``zero_force_ds_cpu_optimizer: false``). That is exactly Gefen's non-fused
CPU-resident path. This file locks the "Gefen steps CPU offload partitions"
contract WITHOUT a DeepSpeed dependency or a GPU, by reproducing the
partition-swap DeepSpeed performs: replace each param group's params with a
flat CPU fp32 partition carrying a CPU gradient, then step.

A live end-to-end DeepSpeed ZeRO-2/ZeRO-3 CPU-offload run (Gefen confirmed as
the client optimizer, stepping CPU partitions, loss matching a non-offloaded
GPU run) is validated separately on GPU hardware; this is the CI guard.

Run: pytest tests/test_zero_offload_cpu.py
"""
import pytest
import torch
import torch.nn as nn

from gefen import Gefen


def _zero_offload_flat_swap(groups, *, device="cpu", grad_fill=1e-3):
    """Mirror DeepSpeed ZeRO + CPU offload: replace each group's params with one
    fresh flat 1-D fp32 partition on ``device`` carrying a co-located gradient.
    param_names / state / hypers are left untouched, as DeepSpeed leaves them.
    """
    dev = torch.device(device)
    flats = []
    for group in groups:
        numel = sum(p.numel() for p in group["params"])
        flat = nn.Parameter(torch.randn(numel, dtype=torch.float32, device=dev) * 0.1)
        flat.grad = torch.full_like(flat, grad_fill)
        group["params"] = [flat]
        flats.append(flat)
    return flats


@pytest.mark.parametrize("factored", [True, False])
def test_gefen_steps_cpu_offload_flat_partitions(factored):
    """Plain Gefen must step a CPU flat fp32 partition (the ZeRO-offload shape)
    and train, with all optimizer state staying on CPU."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 48), nn.Tanh(), nn.Linear(48, 16))
    opt = Gefen(
        list(model.named_parameters()), lr=1e-3, fused=False, factored_v_2d=factored
    )
    flats = _zero_offload_flat_swap(opt.param_groups)
    before = [f.detach().clone() for f in flats]

    for _ in range(4):
        for f in flats:
            f.grad = torch.full_like(f, 1e-3)
        opt.step()

    for b, f in zip(before, flats):
        assert f.device.type == "cpu"
        assert torch.isfinite(f).all()
        assert not torch.equal(b, f.detach()), "Gefen did not step the CPU partition"
        for v in opt.state[f].values():
            if torch.is_tensor(v):
                assert v.device.type == "cpu", "optimizer state left the CPU partition"


def test_cpu_offload_partition_does_not_trip_grad_device_guard():
    """The Phase-1 co-location guard must PASS for ZeRO-offload partitions: the
    flat partition and its gradient are both CPU, so step() must not raise."""
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(6, 10))
    opt = Gefen([("w", p)], lr=1e-3, fused=False)
    (flat,) = _zero_offload_flat_swap(opt.param_groups)
    assert flat.device.type == "cpu" and flat.grad.device.type == "cpu"
    opt.step()  # co-located CPU param+grad -> no RuntimeError
    assert torch.isfinite(flat).all()


def test_fused_true_downgrades_on_cpu_offload_partition():
    """axolotl's gefenx builds Gefen with fused=True. Under CPU offload the
    partition is a CPU tensor, so the fused CUDA kernels must be bypassed and
    the pure-torch path used instead of crashing in a kernel launch. Both
    downgrade routes are exercised depending on the host: with CUDA present the
    constructor keeps fused=True and the per-tensor grad_view.is_cuda gate routes
    the CPU partition to the pure-torch path; on a CPU-only host the constructor
    downgrades fused at build time. Either way the step must complete."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(24, 24), nn.GELU(), nn.Linear(24, 8))
    opt = Gefen(list(model.named_parameters()), lr=1e-3, fused=True)
    flats = _zero_offload_flat_swap(opt.param_groups)
    for _ in range(3):
        for f in flats:
            f.grad = torch.full_like(f, 1e-3)
        opt.step()  # must not raise despite fused=True on CPU partitions
    for f in flats:
        assert f.device.type == "cpu"
        assert torch.isfinite(f).all()

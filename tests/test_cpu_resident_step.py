"""CPU-resident optimizer stepping: the correctness baseline for CPU offload.

Gefen's non-fused (``fused=False``) path already steps correctly when the
parameter, gradient, and optimizer state all live on CPU (covered by
``test_core_correctness_cpu`` and ``test_cpu_step_checkpoint``). This file
targets the *dynamic* device scenarios a CPU-offload trainer introduces, which
those suites do not exercise:

  * a framework relocating a parameter between iterations (CPU->CUDA->CPU),
    which would otherwise strand the per-param state on the previous device and
    trip Gefen's codebook/state device-equality guards -- guarded by the new
    ``Gefen._colocate_param_state_`` migration;
  * the fail-fast contract when a gradient and its parameter live on different
    devices;
  * a checkpoint saved on CUDA and reloaded onto CPU (``map_location='cpu'``);
  * CPU-vs-CUDA non-fused parity (convergence-level, not bit-exact).

The pure-CPU cases run on CPU-only CI. The cross-device cases are gated on CUDA.

Run: pytest tests/test_cpu_resident_step.py
"""
import io

import pytest
import torch
import torch.nn as nn

from gefen import Gefen


CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cross-device offload cases need a CUDA GPU"
)


def _state_tensor_devices(opt, p):
    """Devices of every tensor-valued state entry for parameter ``p``."""
    return {
        k: v.device
        for k, v in opt.state[p].items()
        if torch.is_tensor(v)
    }


# ---------------------------------------------------------------------------
# CPU-only: run on any CI runner, no GPU required.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factored", [True, False])
def test_grad_none_subset_is_skipped(factored):
    """A None gradient on part of a group leaves those params untouched while
    the rest step -- the CPU-offload trainer commonly drives a subset."""
    torch.manual_seed(0)
    stepped = nn.Parameter(torch.randn(16, 16))
    idle = nn.Parameter(torch.randn(16, 16))
    opt = Gefen(
        [("stepped", stepped), ("idle", idle)],
        lr=1e-2,
        fused=False,
        factored_v_2d=factored,
    )
    stepped.grad = torch.randn(16, 16) * 0.1
    idle.grad = None  # framework produced no gradient for this shard this step
    before_idle = idle.detach().clone()
    before_stepped = stepped.detach().clone()

    opt.step()

    assert torch.equal(idle.detach(), before_idle), "None-grad param was mutated"
    assert idle not in opt.state or "step" not in opt.state[idle]
    assert not torch.equal(stepped.detach(), before_stepped)
    assert torch.isfinite(stepped.detach()).all()


def test_periodic_codebook_refresh_on_cpu():
    """codebook_refresh_every re-fits the exact-DP codebook and re-quantizes the
    stored momentum indices; on an all-CPU run this must complete without ever
    hitting the codebook/state device-equality guards."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 48), nn.Tanh(), nn.Linear(48, 16))
    opt = Gefen(
        list(model.named_parameters()),
        lr=1e-3,
        fused=False,
        codebook_refresh_every=2,
    )
    gen = torch.Generator().manual_seed(7)
    for _ in range(5):  # crosses two refresh boundaries (steps 2 and 4)
        for p in model.parameters():
            p.grad = torch.randn(p.shape, generator=gen) * 1e-2
        opt.step()
        opt.zero_grad()

    assert opt._gefen_codebook.device.type == "cpu"
    for p in model.parameters():
        assert torch.isfinite(p).all()
        for dev in _state_tensor_devices(opt, p).values():
            assert dev.type == "cpu"


@pytest.mark.parametrize("factored", [True, False])
def test_colocate_is_noop_on_same_device(factored):
    """The zero-diff guarantee: on a same-device step the colocation helper must
    not reallocate/reassign any state tensor (pure device compare early-out)."""
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(16, 16))
    opt = Gefen([("w", p)], lr=1e-2, fused=False, factored_v_2d=factored)
    p.grad = torch.randn(16, 16) * 0.1
    opt.step()  # materialize all state on CPU

    state = opt.state[p]
    before = {k: v for k, v in state.items() if torch.is_tensor(v)}
    assert before, "expected materialized tensor state after one step"

    opt._colocate_param_state_(state, torch.device("cpu"))

    for k, v in before.items():
        assert state[k] is v, f"state[{k!r}] was reassigned on a same-device call"


# ---------------------------------------------------------------------------
# CUDA-required: the genuine offload / cross-device behavior.
# ---------------------------------------------------------------------------

@CUDA
@pytest.mark.parametrize("factored", [True, False])
def test_param_paged_cpu_cuda_cpu_across_steps(factored):
    """Core regression for _colocate_param_state_: a parameter paged
    CPU->CUDA->CPU between steps must not raise, its state must track the param's
    device each step, and the final CPU parameter must be finite and updated."""
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(16, 16))
    opt = Gefen([("w", p)], lr=1e-2, fused=False, factored_v_2d=factored)
    gen = torch.Generator().manual_seed(3)

    def _step_on(device):
        p.data = p.data.to(device)
        p.grad = (torch.randn(16, 16, generator=gen) * 0.1).to(device)
        opt.step()
        opt.zero_grad(set_to_none=True)
        for dev in _state_tensor_devices(opt, p).values():
            assert dev.type == device.type, "state did not follow the param device"

    before = p.detach().cpu().clone()
    _step_on(torch.device("cpu"))
    _step_on(torch.device("cuda"))
    _step_on(torch.device("cpu"))

    assert p.device.type == "cpu"
    assert torch.isfinite(p.detach()).all()
    assert not torch.equal(before, p.detach().cpu())


@CUDA
def test_batched_group_paged_cpu_cuda_cpu():
    """The merged (foreach) non-fused path only engages for several same-geometry
    small params on CUDA. Paging such a batch CPU->CUDA->CPU must not stack
    cross-device state: params whose state is momentarily stranded fall back to
    the per-param colocating path, so the batch never mixes devices."""
    torch.manual_seed(0)
    # Several identically-shaped small params so they batch together.
    params = [(f"w{i}", nn.Parameter(torch.randn(8, 12))) for i in range(6)]
    opt = Gefen(params, lr=1e-2, fused=False, factored_v_2d=False)
    gen = torch.Generator().manual_seed(5)

    def _step_on(device):
        for _, p in params:
            p.data = p.data.to(device)
            p.grad = (torch.randn(8, 12, generator=gen) * 0.1).to(device)
        opt.step()
        opt.zero_grad(set_to_none=True)
        for _, p in params:
            for dev in _state_tensor_devices(opt, p).values():
                assert dev.type == device.type

    _step_on(torch.device("cpu"))
    _step_on(torch.device("cuda"))  # paged onto CUDA with CPU-stranded state
    _step_on(torch.device("cuda"))  # now steady-state on CUDA (batched path)
    _step_on(torch.device("cpu"))

    for _, p in params:
        assert p.device.type == "cpu"
        assert torch.isfinite(p.detach()).all()


@CUDA
def test_grad_on_different_device_than_param_raises():
    """The Phase-1 co-location contract. PyTorch's ``.grad`` setter already
    blocks assigning a cross-device gradient, so the realistic way this mismatch
    reaches the optimizer is the classic offload footgun: the parameter's
    storage is paged to another device *after* its gradient was populated,
    leaving grad and param split. step() must reject it with a message naming
    both devices instead of failing opaquely inside a kernel/add_."""
    p = nn.Parameter(torch.randn(8, 8).cuda())  # CUDA param + CUDA grad (valid)
    opt = Gefen([("w", p)], lr=1e-2, fused=False)
    p.grad = torch.randn(8, 8, device="cuda") * 0.1
    p.data = p.data.cpu()  # page the param to CPU but leave its grad on CUDA

    assert p.device.type == "cpu" and p.grad.device.type == "cuda"
    with pytest.raises(RuntimeError, match="same device"):
        opt.step()


@CUDA
@pytest.mark.parametrize("factored", [True, False])
def test_checkpoint_saved_on_cuda_loads_and_steps_on_cpu(factored):
    """save on CUDA -> torch.load(map_location='cpu') -> load into a CPU
    optimizer -> CPU step: state and codebook must land co-located on CPU."""
    torch.manual_seed(0)
    model_cuda = nn.Sequential(nn.Linear(32, 48), nn.Tanh(), nn.Linear(48, 16)).cuda()
    opt_cuda = Gefen(
        list(model_cuda.named_parameters()),
        lr=1e-3,
        fused=False,
        factored_v_2d=factored,
    )
    gen = torch.Generator(device="cuda").manual_seed(11)
    for _ in range(3):
        for p in model_cuda.parameters():
            p.grad = torch.randn(p.shape, generator=gen, device="cuda") * 1e-2
        opt_cuda.step()
        opt_cuda.zero_grad()

    # Round-trip the checkpoint through disk semantics onto CPU.
    buf = io.BytesIO()
    torch.save(opt_cuda.state_dict(), buf)
    buf.seek(0)
    cpu_state = torch.load(buf, map_location="cpu")

    torch.manual_seed(0)
    model_cpu = nn.Sequential(nn.Linear(32, 48), nn.Tanh(), nn.Linear(48, 16))
    opt_cpu = Gefen(
        list(model_cpu.named_parameters()),
        lr=1e-3,
        fused=False,
        factored_v_2d=factored,
    )
    opt_cpu.load_state_dict(cpu_state)

    assert opt_cpu._gefen_codebook.device.type == "cpu"
    for p in model_cpu.parameters():
        for dev in _state_tensor_devices(opt_cpu, p).values():
            assert dev.type == "cpu", "restored state stranded off the CPU param"

    # And a subsequent CPU step must run cleanly on the restored state.
    gen_cpu = torch.Generator().manual_seed(12)
    for p in model_cpu.parameters():
        p.grad = torch.randn(p.shape, generator=gen_cpu) * 1e-2
    opt_cpu.step()
    for p in model_cpu.parameters():
        assert torch.isfinite(p).all()


@CUDA
@pytest.mark.parametrize(
    "dtype,rtol,atol",
    [(torch.float32, 1e-4, 1e-5), (torch.bfloat16, 2e-2, 2e-2)],
)
def test_cpu_vs_cuda_nonfused_parity(dtype, rtol, atol):
    """Non-fused stepping on CPU vs CUDA must agree to convergence-level
    tolerance. Bit-exactness is NOT achievable: the grad^2 reductions feeding
    the second moment (vmean EMA / factored row-col sums / outer-sqrt-div) use
    different accumulation orders and backend math on CPU vs CUDA, so the
    per-block stepsize drifts at sub-ULP scale and compounds across steps. We
    therefore assert closeness, not equality."""
    torch.manual_seed(0)
    init = torch.randn(24, 32, dtype=dtype)
    grads = [torch.randn(24, 32, dtype=dtype) * 0.1 for _ in range(6)]

    def _run(device):
        p = nn.Parameter(init.detach().clone().to(device))
        opt = Gefen([("w", p)], lr=1e-2, fused=False)
        for g in grads:
            p.grad = g.detach().clone().to(device)
            opt.step()
            opt.zero_grad(set_to_none=True)
        return p.detach().float().cpu()

    p_cpu = _run(torch.device("cpu"))
    p_cuda = _run(torch.device("cuda"))

    assert torch.isfinite(p_cpu).all() and torch.isfinite(p_cuda).all()
    assert torch.allclose(p_cpu, p_cuda, rtol=rtol, atol=atol), (
        (p_cpu - p_cuda).abs().max().item()
    )

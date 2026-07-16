"""Atomic movement and synchronous CPU-offload coverage."""

import copy

import pytest
import torch

from gefen import Gefen, GefenMuon


def _initialized(optimizer_type=Gefen, *, device="cpu", factored=False):
    values = torch.linspace(-0.5, 0.5, 8, device=device)
    if optimizer_type is GefenMuon:
        values = values.reshape(2, 4)
    parameter = torch.nn.Parameter(values)
    kwargs = {"fused": False, "lr": 2e-3}
    if optimizer_type is Gefen:
        kwargs["factored_v_2d"] = factored
    else:
        kwargs.update(ns_steps=1, normuon=False, weight_decay=0.0)
    optimizer = optimizer_type([("layer.weight", parameter)], **kwargs)
    optimizer._resolve_automatic_period = lambda *args, **kwargs: 4
    optimizer._predict_period_from_grad_sq = lambda *args, **kwargs: 4
    parameter.grad = torch.linspace(-1.0, 1.0, 8, device=device).reshape_as(parameter)
    optimizer.step()
    return optimizer, parameter


def _persistent_snapshot(optimizer, parameter):
    return {
        key: value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in optimizer.state[parameter].items()
        if key not in {"stepsize", "_h_buf"}
    }


@pytest.mark.parametrize("optimizer_type", [Gefen, GefenMuon])
def test_move_state_cpu_makes_fresh_tight_copies(optimizer_type):
    optimizer, parameter = _initialized(optimizer_type)
    before = _persistent_snapshot(optimizer, parameter)
    old_state = optimizer.state
    old_tensors = {key: value for key, value in optimizer.state[parameter].items() if torch.is_tensor(value)}

    optimizer.move_state_("cpu")

    assert optimizer.state is not old_state
    for key, value in optimizer.state[parameter].items():
        if not torch.is_tensor(value):
            continue
        assert value is not old_tensors[key]
        assert value.device == torch.device("cpu")
        assert value.is_contiguous() and value.storage_offset() == 0
        assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_move_state_failure_is_atomic(monkeypatch):
    optimizer, parameter = _initialized()
    state_before = optimizer.state
    parameter_state_before = optimizer.state[parameter]
    codebook_before = optimizer._gefen_codebook
    calls = 0
    original = optimizer._copy_state_tensor_for_move

    def fail_late(tensor, device):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected state movement failure")
        return original(tensor, device)

    monkeypatch.setattr(optimizer, "_copy_state_tensor_for_move", fail_late)
    with pytest.raises(RuntimeError, match="injected"):
        optimizer.move_state_("cpu")
    assert optimizer.state is state_before
    assert optimizer.state[parameter] is parameter_state_before
    assert optimizer._gefen_codebook is codebook_before


def test_move_state_rejects_undeclared_tensor_and_preserves_state():
    optimizer, parameter = _initialized()
    optimizer.state[parameter]["extension"] = torch.ones(1)
    state_before = optimizer.state
    with pytest.raises(RuntimeError, match="metadata"):
        optimizer.move_state_()
    assert optimizer.state is state_before


def test_cpu_offload_rejects_cpu_parameters_and_muon():
    optimizer, _ = _initialized()
    with pytest.raises(RuntimeError, match="CUDA"):
        optimizer.offload_state_()
    muon, _ = _initialized(GefenMuon)
    with pytest.raises(RuntimeError, match="plain Gefen"):
        muon.offload_state_()


def _zero_style_flat_swap(optimizer, *, device):
    """Replace registered parameters the way ZeRO-1/2 replaces client groups."""

    flat_parameters = []
    for group in optimizer.param_groups:
        numel = sum(parameter.numel() for parameter in group["params"])
        flat = torch.zeros(numel, dtype=torch.float32, device=device, requires_grad=True)
        group["params"] = [flat]
        flat_parameters.append(flat)
    return flat_parameters


CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@CUDA_REQUIRED
def test_offload_rejects_zero_style_parameter_takeover_atomically():
    optimizer, _ = _initialized(device="cuda")
    state_before = optimizer.state
    codebook_before = optimizer._gefen_codebook
    _zero_style_flat_swap(optimizer, device="cuda")

    with pytest.raises(RuntimeError, match="parameter ownership changed.*DeepSpeed ZeRO"):
        optimizer.offload_state_()

    assert optimizer.state is state_before
    assert optimizer._gefen_codebook is codebook_before
    assert not optimizer.state_offload_active


@CUDA_REQUIRED
def test_offload_rejects_zero_style_takeover_after_inner_optimizer_resume():
    optimizer, _ = _initialized(device="cuda")
    (flat_parameter,) = _zero_style_flat_swap(optimizer, device="cuda")
    flat_parameter.grad = torch.linspace(-0.4, 0.5, flat_parameter.numel(), device="cuda")
    optimizer.step()

    # Saving a ZeRO-owned flat parameter withholds the unreachable originals;
    # after load, live-state equality alone cannot identify the takeover.
    checkpoint = copy.deepcopy(optimizer.state_dict())
    optimizer.load_state_dict(checkpoint)
    assert set(optimizer.state) == {flat_parameter}

    with pytest.raises(RuntimeError, match="parameter ownership changed.*DeepSpeed ZeRO"):
        optimizer.offload_state_()
    assert not optimizer.state_offload_active


@CUDA_REQUIRED
def test_active_offload_rejects_late_zero_style_takeover_before_mutation():
    optimizer, original_parameter = _initialized(device="cuda")
    optimizer.offload_state_()
    state_before = optimizer.state
    original_state_before = optimizer.state[original_parameter]
    global_step_before = optimizer._gefen_global_step

    (flat_parameter,) = _zero_style_flat_swap(optimizer, device="cuda")
    flat_before = flat_parameter.detach().clone()
    flat_parameter.grad = torch.linspace(-0.4, 0.5, flat_parameter.numel(), device="cuda")

    with pytest.raises(RuntimeError, match="parameter ownership changed.*DeepSpeed ZeRO"):
        optimizer.step()

    assert torch.equal(flat_parameter, flat_before)
    assert optimizer.state is state_before
    assert optimizer.state[original_parameter] is original_state_before
    assert optimizer._gefen_global_step == global_step_before
    assert optimizer.state_offload_active


@CUDA_REQUIRED
def test_offload_rejects_zero3_style_cleared_parameter_groups():
    optimizer, _ = _initialized(device="cuda")
    state_before = optimizer.state
    for group in optimizer.param_groups:
        group["params"] = []

    with pytest.raises(RuntimeError, match="parameter ownership changed.*DeepSpeed ZeRO"):
        optimizer.offload_state_()

    assert optimizer.state is state_before
    assert not optimizer.state_offload_active


@CUDA_REQUIRED
def test_legitimate_add_param_group_extends_offload_ownership():
    optimizer, _ = _initialized(device="cuda")
    added = torch.nn.Parameter(torch.linspace(-0.2, 0.3, 6, device="cuda"))
    optimizer.add_param_group({"params": [("added.weight", added)]})

    optimizer.offload_state_()

    assert optimizer.state_offload_active
    assert added in optimizer.state


@CUDA_REQUIRED
@pytest.mark.parametrize("factored,shape", [(False, (8,)), (True, (2, 4))])
@pytest.mark.parametrize("activate_before_first_step", [False, True])
def test_offload_step_is_bit_exact_and_keeps_cpu_authority(factored, shape, activate_before_first_step):
    values = torch.linspace(-0.5, 0.5, 8, device="cuda").reshape(shape)
    source_parameter = torch.nn.Parameter(values.clone())
    target_parameter = torch.nn.Parameter(values.clone())
    source = Gefen(
        [("layer.weight", source_parameter)],
        lr=2e-3,
        fused=False,
        factored_v_2d=factored,
        codebook_refresh_every=2,
    )
    target = Gefen(
        [("layer.weight", target_parameter)],
        lr=2e-3,
        fused=False,
        factored_v_2d=factored,
        codebook_refresh_every=2,
    )
    source._resolve_automatic_period = lambda *args, **kwargs: 4
    target._resolve_automatic_period = lambda *args, **kwargs: 4
    if activate_before_first_step:
        target.offload_state_()
    else:
        initial_grad = torch.linspace(-1.0, 1.0, 8, device="cuda").reshape(shape)
        source_parameter.grad = initial_grad.clone()
        target_parameter.grad = initial_grad.clone()
        source.step()
        target.step()
        target.offload_state_()

    for step in range(3):
        grad = torch.linspace(-0.8 + step * 0.1, 0.7, 8, device="cuda").reshape(shape)
        source_parameter.grad = grad.clone()
        target_parameter.grad = grad.clone()
        source.step()
        target.step()
        torch.testing.assert_close(target_parameter, source_parameter, rtol=0, atol=0)
        assert all(
            not torch.is_tensor(value) or value.device.type == "cpu"
            for key, value in target.state[target_parameter].items()
            if key not in {"name", "automatic_period"}
        )


@CUDA_REQUIRED
def test_offload_activation_and_restore_copy_failures_are_atomic(monkeypatch):
    optimizer, parameter = _initialized(device="cuda")
    state_before = optimizer.state
    parameter_state_before = optimizer.state[parameter]

    def fail_cpu_copy(_tensor):
        raise RuntimeError("injected activation copy failure")

    monkeypatch.setattr(optimizer, "_copy_state_tensor_to_offload_cpu", fail_cpu_copy)
    with pytest.raises(RuntimeError, match="activation copy failure"):
        optimizer.offload_state_()
    assert optimizer.state is state_before
    assert optimizer.state[parameter] is parameter_state_before
    assert not optimizer.state_offload_active

    monkeypatch.undo()
    optimizer.offload_state_()
    state_before = optimizer.state
    parameter_state_before = optimizer.state[parameter]

    def fail_restore(_tensor, _device):
        raise RuntimeError("injected restore copy failure")

    monkeypatch.setattr(optimizer, "_copy_state_tensor_for_move", fail_restore)
    with pytest.raises(RuntimeError, match="restore copy failure"):
        optimizer.restore_state_()
    assert optimizer.state is state_before
    assert optimizer.state[parameter] is parameter_state_before
    assert optimizer.state_offload_active


@CUDA_REQUIRED
@pytest.mark.parametrize("optimizer_type", [Gefen, GefenMuon])
def test_move_state_follows_parameter_across_cpu_and_cuda(optimizer_type):
    optimizer, parameter = _initialized(optimizer_type, device="cuda")
    module = torch.nn.Module()
    module.register_parameter("weight", parameter)

    module.to("cpu")
    assert module.weight is parameter
    optimizer.move_state_()
    assert optimizer._gefen_codebook.device.type == "cpu"
    assert all(
        not torch.is_tensor(value) or value.device.type == "cpu" for value in optimizer.state[parameter].values()
    )

    module.to("cuda")
    assert module.weight is parameter
    optimizer.move_state_()
    assert optimizer._gefen_codebook.device.type == "cuda"
    assert all(
        not torch.is_tensor(value) or value.device.type == "cuda" for value in optimizer.state[parameter].values()
    )


@CUDA_REQUIRED
def test_active_offload_survives_load_and_restore():
    optimizer, parameter = _initialized(device="cuda")
    checkpoint = copy.deepcopy(optimizer.state_dict())
    optimizer.offload_state_()
    optimizer.load_state_dict(checkpoint)
    assert optimizer.state_offload_active
    assert not optimizer.state_offload_poisoned
    assert all(
        not torch.is_tensor(value) or value.device.type == "cpu" for value in optimizer.state[parameter].values()
    )
    optimizer.restore_state_()
    assert not optimizer.state_offload_active
    assert all(
        not torch.is_tensor(value) or value.device.type == "cuda" for value in optimizer.state[parameter].values()
    )


@CUDA_REQUIRED
def test_copyback_failure_poison_is_sticky(monkeypatch):
    optimizer, parameter = _initialized(device="cuda")
    optimizer.offload_state_()
    parameter.grad = torch.linspace(0.7, -0.6, 8, device="cuda")

    def fail_copyback(_tensor):
        raise RuntimeError("injected copyback failure")

    monkeypatch.setattr(optimizer, "_copy_state_tensor_to_offload_cpu", fail_copyback)
    with pytest.raises(RuntimeError, match="known-good checkpoint"):
        optimizer.step()
    assert optimizer.state_offload_poisoned
    with pytest.raises(RuntimeError, match="cannot export optimizer state"):
        optimizer.state_dict()
    with pytest.raises(RuntimeError, match="poisoned"):
        optimizer.step()


@CUDA_REQUIRED
def test_offload_rejects_persistent_storage_aliases():
    optimizer, parameter = _initialized(device="cuda")
    optimizer.state[parameter]["vmean"] = optimizer.state[parameter]["m_magnitude"]
    state_before = optimizer.state
    with pytest.raises(RuntimeError, match="aliases"):
        optimizer.offload_state_()
    assert optimizer.state is state_before


@CUDA_REQUIRED
def test_known_good_load_clears_poison_even_after_restore(monkeypatch):
    optimizer, parameter = _initialized(device="cuda")
    known_good = copy.deepcopy(optimizer.state_dict())
    optimizer.offload_state_()
    parameter.grad = torch.linspace(0.7, -0.6, 8, device="cuda")

    def fail_copyback(_tensor):
        raise RuntimeError("injected copyback failure")

    monkeypatch.setattr(optimizer, "_copy_state_tensor_to_offload_cpu", fail_copyback)
    with pytest.raises(RuntimeError, match="known-good checkpoint"):
        optimizer.step()
    monkeypatch.undo()
    assert optimizer.state_offload_poisoned

    # Turning offload off first must not strand the poison flag: a full
    # known-good load is the documented recovery and clears it unconditionally.
    optimizer.restore_state_()
    assert not optimizer.state_offload_active
    assert optimizer.state_offload_poisoned

    optimizer.load_state_dict(copy.deepcopy(known_good))
    assert not optimizer.state_offload_poisoned
    optimizer.state_dict()
    parameter.grad = torch.linspace(-1.0, 1.0, 8, device="cuda")
    optimizer.step()


@CUDA_REQUIRED
def test_active_offload_load_colocates_cpu_mapped_codebook():
    optimizer, parameter = _initialized(device="cuda")
    checkpoint = copy.deepcopy(optimizer.state_dict())

    def to_cpu(value):
        if torch.is_tensor(value):
            return value.cpu()
        if isinstance(value, dict):
            return {key: to_cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [to_cpu(item) for item in value]
        return value

    cpu_mapped = to_cpu(checkpoint)
    optimizer.offload_state_()
    optimizer.load_state_dict(cpu_mapped)

    assert optimizer.state_offload_active
    assert optimizer._gefen_codebook.device.type == "cuda"
    parameter.grad = torch.linspace(-1.0, 1.0, 8, device="cuda")
    optimizer.step()
    assert all(
        not torch.is_tensor(value) or value.device.type == "cpu"
        for value in optimizer.state[parameter].values()
    )


def _to_cpu_tree(value):
    if torch.is_tensor(value):
        return value.cpu()
    if isinstance(value, dict):
        return {key: _to_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_tree(item) for item in value]
    return value


@CUDA_REQUIRED
def test_offload_activation_colocates_cpu_mapped_codebook():
    # Inactive map_location='cpu' load leaves the codebook on CPU; enabling
    # offload afterwards must restore the CUDA-resident invariant so the very
    # next step succeeds (the common resume-then-enable-offload flow).
    optimizer, parameter = _initialized(device="cuda")
    cpu_mapped = _to_cpu_tree(copy.deepcopy(optimizer.state_dict()))
    optimizer.load_state_dict(cpu_mapped)
    assert optimizer._gefen_codebook.device.type == "cpu"
    assert not optimizer.state_offload_active

    optimizer.offload_state_()
    assert optimizer.state_offload_active
    assert optimizer._gefen_codebook.device.type == "cuda"

    parameter.grad = torch.linspace(-1.0, 1.0, 8, device="cuda")
    optimizer.step()
    assert all(
        not torch.is_tensor(value) or value.device.type == "cpu"
        for value in optimizer.state[parameter].values()
    )


@CUDA_REQUIRED
@pytest.mark.parametrize("factored", [False, True])
def test_offload_end_to_end_training_loss_matches_baseline(factored):
    # A real forward/backward/step loop over a small MLP: the CPU-authoritative
    # offloaded run must track a non-offloaded run's loss bit-for-bit at every
    # step, and keep per-parameter state on CPU throughout.
    torch.manual_seed(0)
    reference_model = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4)
    ).cuda()
    offloaded_model = copy.deepcopy(reference_model)

    def make_optimizer(model):
        optimizer = Gefen(
            model.named_parameters(), lr=3e-3, fused=False, factored_v_2d=factored
        )
        optimizer._resolve_automatic_period = lambda *args, **kwargs: 4
        return optimizer

    reference_optimizer = make_optimizer(reference_model)
    offloaded_optimizer = make_optimizer(offloaded_model)
    offloaded_optimizer.offload_state_()

    generator = torch.Generator(device="cuda").manual_seed(7)
    batches = [
        (
            torch.randn(32, 8, generator=generator, device="cuda"),
            torch.randn(32, 4, generator=generator, device="cuda"),
        )
        for _ in range(8)
    ]

    reference_losses = []
    offloaded_losses = []
    for inputs, targets in batches:
        for model, optimizer, losses in (
            (reference_model, reference_optimizer, reference_losses),
            (offloaded_model, offloaded_optimizer, offloaded_losses),
        ):
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(inputs) - targets) ** 2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())

    for offloaded_loss, reference_loss in zip(offloaded_losses, reference_losses):
        torch.testing.assert_close(offloaded_loss, reference_loss, rtol=0, atol=0)
    assert offloaded_optimizer.state_offload_active
    assert all(
        not torch.is_tensor(value) or value.device.type == "cpu"
        for parameter_state in offloaded_optimizer.state.values()
        for value in parameter_state.values()
    )

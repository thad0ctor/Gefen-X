"""Focused CPU/CUDA coverage for native plain-Gefen parameter-state offload."""

import copy
import io
import types

import pytest
import torch

from gefen import (
    CheckpointProcessGroupBinding,
    CheckpointTransport,
    Gefen,
    GefenMuon,
    PortableStateLimits,
    StateOffloadProvider,
)
from gefen.codebook import CodebookProcessGroupBinding
from gefen.contracts import (
    LogicalSlice,
    ParameterIdentity,
    ParameterLayout,
    PlacementKind,
    ProcessGroupIdentity,
    ShardIdentity,
    ShardPlacement,
    ShardingManifest,
)
from gefen.rebinding import ParameterRebinding


_PERSISTENT_TENSOR_KEYS = frozenset({"m_codebook", "m_magnitude", "vmean", "v_row", "v_col"})
_CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _make_gefen(parameter, *, factored=False, fused=False):
    optimizer = Gefen(
        [("layer.weight", parameter)],
        lr=2.0e-3,
        fused=fused,
        factored_v_2d=factored,
        deterministic=True,
    )
    optimizer._resolve_automatic_period = lambda *args, **kwargs: 4
    return optimizer


def _persistent_snapshot(optimizer, parameter):
    result = {}
    for key, value in optimizer.state[parameter].items():
        if key in _PERSISTENT_TENSOR_KEYS:
            result[key] = value.detach().cpu().clone()
        elif key in {"automatic_period", "step", "vmean_step", "factored_step"}:
            result[key] = value
    return result


def _assert_persistent_equal(left, right):
    assert set(left) == set(right)
    for key in left:
        if torch.is_tensor(left[key]):
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        else:
            assert left[key] == right[key]


def _assert_cpu_boundary(optimizer):
    assert optimizer.state_offload_active
    assert optimizer.state_offload_device == torch.device("cpu")
    assert not optimizer.state_offload_poisoned
    for parameter_state in optimizer.state.values():
        assert "stepsize" not in parameter_state
        assert "_h_buf" not in parameter_state
        for key, value in parameter_state.items():
            if key in _PERSISTENT_TENSOR_KEYS:
                assert type(value) is torch.Tensor
                assert value.device.type == "cpu"
                assert value.is_contiguous()
                assert value.storage_offset() == 0
                assert value.untyped_storage().nbytes() == value.numel() * value.element_size()


def test_state_offload_visibility_is_read_only_and_cpu_parameters_fail_closed():
    parameter = torch.nn.Parameter(torch.ones(8))
    optimizer = _make_gefen(parameter)

    assert not optimizer.state_offload_active
    assert optimizer.state_offload_device is None
    assert not optimizer.state_offload_poisoned
    assert isinstance(optimizer, StateOffloadProvider)
    assert not optimizer.optimizer_contract().capabilities.state_offload
    with pytest.raises(AttributeError):
        optimizer.state_offload_active = True
    with pytest.raises(AttributeError):
        optimizer.state_offload_device = torch.device("cpu")
    with pytest.raises(AttributeError):
        optimizer.state_offload_poisoned = True
    with pytest.raises(RuntimeError, match="ordinary replicated CUDA"):
        optimizer.offload_state_()


def test_state_offload_rejects_non_cpu_targets_and_muon():
    parameter = torch.nn.Parameter(torch.ones(2, 4))
    optimizer = _make_gefen(parameter)
    muon = GefenMuon(
        [("layer.weight", parameter)],
        fused=False,
        ns_steps=1,
    )

    with pytest.raises(ValueError, match="only CPU"):
        optimizer.offload_state_("meta")
    with pytest.raises(RuntimeError, match="only by plain Gefen"):
        muon.offload_state_()


@_CUDA_REQUIRED
@pytest.mark.parametrize("factored", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_pristine_offload_multistep_is_exact_and_cpu_authoritative(factored, fused):
    shape = (2, 4) if factored else (8,)
    initial = torch.linspace(-0.7, 0.6, 8, device="cuda").reshape(shape)
    reference_parameter = torch.nn.Parameter(initial.clone())
    offloaded_parameter = torch.nn.Parameter(initial.clone())
    reference = _make_gefen(reference_parameter, factored=factored, fused=fused)
    offloaded = _make_gefen(offloaded_parameter, factored=factored, fused=fused)

    assert offloaded.optimizer_contract().capabilities.state_offload
    offloaded.offload_state_()
    assert offloaded.optimizer_contract().capabilities.state_offload
    _assert_cpu_boundary(offloaded)
    assert offloaded._gefen_codebook is None

    for step in range(3):
        grad = torch.linspace(
            -1.1 + 0.2 * step,
            0.9 - 0.1 * step,
            8,
            device="cuda",
        ).reshape(shape)
        reference_parameter.grad = grad.clone()
        offloaded_parameter.grad = grad.clone()
        reference.step()
        offloaded.step()

        torch.testing.assert_close(offloaded_parameter, reference_parameter, rtol=0, atol=0)
        _assert_persistent_equal(
            _persistent_snapshot(offloaded, offloaded_parameter),
            _persistent_snapshot(reference, reference_parameter),
        )
        assert offloaded._gefen_global_step == reference._gefen_global_step
        _assert_cpu_boundary(offloaded)
        assert offloaded._gefen_codebook.device.type == "cuda"


@_CUDA_REQUIRED
def test_initialized_activation_keeps_common_codebook_resident_and_move_disables():
    parameter = torch.nn.Parameter(torch.linspace(-0.4, 0.8, 8, device="cuda"))
    optimizer = _make_gefen(parameter)
    parameter.grad = torch.linspace(-1.0, 1.0, 8, device="cuda")
    optimizer.step()
    codebook = optimizer._gefen_codebook
    cache = optimizer._gefen_codebook_by_device
    cache[parameter.device] = codebook

    optimizer.offload_state_("cpu:0")

    _assert_cpu_boundary(optimizer)
    assert optimizer._gefen_codebook is codebook
    assert optimizer._gefen_codebook_by_device is cache
    assert optimizer._gefen_codebook_by_device[parameter.device] is codebook

    optimizer.move_state_()

    assert not optimizer.state_offload_active
    assert optimizer.state_offload_device is None
    assert not optimizer.state_offload_poisoned
    assert all(
        value.device == parameter.device
        for key, value in optimizer.state[parameter].items()
        if key in _PERSISTENT_TENSOR_KEYS
    )


@_CUDA_REQUIRED
def test_runtime_state_is_private_and_only_one_parameter_is_staged():
    first = torch.nn.Parameter(torch.arange(8, device="cuda", dtype=torch.float32))
    second = torch.nn.Parameter(torch.arange(8, 16, device="cuda", dtype=torch.float32))
    optimizer = Gefen(
        [("first", first), ("second", second)],
        fused=False,
        factored_v_2d=False,
    )
    optimizer._resolve_automatic_period = lambda *args, **kwargs: 4
    optimizer.offload_state_()
    observed = []
    original = optimizer._step_automatic

    def inspected(self, group, name, parameter, grad, *, state=None):
        assert state is not self.state[parameter]
        assert all(
            value.device.type == "cpu"
            for published in self.state.values()
            for key, value in published.items()
            if key in _PERSISTENT_TENSOR_KEYS
        )
        assert all(value.device == parameter.device for key, value in state.items() if key in _PERSISTENT_TENSOR_KEYS)
        observed.append(parameter)
        return original(group, name, parameter, grad, state=state)

    optimizer._step_automatic = types.MethodType(inspected, optimizer)
    first.grad = torch.linspace(-1.0, 1.0, 8, device="cuda")
    second.grad = torch.linspace(1.0, -1.0, 8, device="cuda")
    optimizer.step()

    assert observed == [first, second]
    _assert_cpu_boundary(optimizer)


@_CUDA_REQUIRED
def test_later_parameter_corruption_is_caught_before_earlier_parameter_mutates():
    # Regression: the offloaded step processes parameters sequentially
    # (stage -> update -> copyback -> commit per parameter). If a later
    # parameter's offloaded state is corrupted in a way that preserves the
    # state/param-group container identities and the layout version, a cached
    # step-readiness verdict would let step() begin, mutate the first
    # parameter, and only reject when the second parameter is staged --
    # violating fail-before-mutation. The readiness scan therefore runs in
    # full on every step, so the corruption is caught at step entry and the
    # first parameter is left byte-for-byte untouched.
    first = torch.nn.Parameter(torch.arange(8, device="cuda", dtype=torch.float32))
    second = torch.nn.Parameter(torch.arange(8, 16, device="cuda", dtype=torch.float32))
    optimizer = Gefen(
        [("first", first), ("second", second)],
        fused=False,
        factored_v_2d=False,
    )
    optimizer._resolve_automatic_period = lambda *args, **kwargs: 4
    optimizer.offload_state_()

    first.grad = torch.linspace(-1.0, 1.0, 8, device="cuda")
    second.grad = torch.linspace(1.0, -1.0, 8, device="cuda")
    optimizer.step()

    first_state_before = _persistent_snapshot(optimizer, first)
    first_value_before = first.detach().clone()

    # Corrupt the SECOND parameter's offloaded state with a non-tight CPU view.
    # self.state, self.state[second], and the layout version are all unchanged.
    corrupt = optimizer.state[second]["m_magnitude"].repeat_interleave(2)[::2]
    assert not corrupt.is_contiguous()
    optimizer.state[second]["m_magnitude"] = corrupt

    first.grad = torch.linspace(-1.0, 1.0, 8, device="cuda")
    second.grad = torch.linspace(1.0, -1.0, 8, device="cuda")
    with pytest.raises(RuntimeError, match="cannot step"):
        optimizer.step()

    _assert_persistent_equal(_persistent_snapshot(optimizer, first), first_state_before)
    assert torch.equal(first.detach(), first_value_before)


@_CUDA_REQUIRED
def test_active_load_preserves_target_policy_and_exact_continuation(monkeypatch):
    source_parameter = torch.nn.Parameter(torch.linspace(-0.5, 0.5, 8, device="cuda"))
    source = _make_gefen(source_parameter)
    source_parameter.grad = torch.linspace(-1.0, 0.7, 8, device="cuda")
    source.step()
    checkpoint = copy.deepcopy(source.state_dict())

    target_parameter = torch.nn.Parameter(source_parameter.detach().clone())
    target = _make_gefen(target_parameter)
    target.offload_state_()

    def reject_parameter_device_cast(*_args, **_kwargs):
        raise AssertionError("active offload load used PyTorch's parameter-device cast")

    monkeypatch.setattr(
        torch.optim.Optimizer,
        "_process_value_according_to_param_policy",
        reject_parameter_device_cast,
    )
    target.load_state_dict(checkpoint)

    _assert_cpu_boundary(target)
    _assert_persistent_equal(
        _persistent_snapshot(source, source_parameter),
        _persistent_snapshot(target, target_parameter),
    )

    continuation = torch.linspace(0.8, -0.6, 8, device="cuda")
    source_parameter.grad = continuation.clone()
    target_parameter.grad = continuation.clone()
    source.step()
    target.step()

    torch.testing.assert_close(target_parameter, source_parameter, rtol=0, atol=0)
    _assert_persistent_equal(
        _persistent_snapshot(source, source_parameter),
        _persistent_snapshot(target, target_parameter),
    )
    _assert_cpu_boundary(target)


@_CUDA_REQUIRED
@pytest.mark.parametrize("activate_before_load", [False, True])
def test_cpu_mapped_checkpoint_keeps_common_codebook_cuda_resident(
    activate_before_load,
):
    source_parameter = torch.nn.Parameter(torch.linspace(-0.5, 0.5, 8, device="cuda"))
    source = _make_gefen(source_parameter)
    source_parameter.grad = torch.linspace(-1.0, 0.7, 8, device="cuda")
    source.step()
    serialized = io.BytesIO()
    torch.save(source.state_dict(), serialized)
    serialized.seek(0)
    checkpoint = torch.load(
        serialized,
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["gefen_codebook"].device.type == "cpu"

    target_parameter = torch.nn.Parameter(source_parameter.detach().clone())
    target = _make_gefen(target_parameter)
    if activate_before_load:
        target.offload_state_()
    target.load_state_dict(checkpoint)
    if not activate_before_load:
        assert target._gefen_codebook.device.type == "cpu"
        target.offload_state_()

    _assert_cpu_boundary(target)
    assert target._gefen_codebook.device == target_parameter.device
    torch.testing.assert_close(
        target._gefen_codebook,
        source._gefen_codebook,
        rtol=0,
        atol=0,
    )

    continuation = torch.linspace(0.8, -0.6, 8, device="cuda")
    source_parameter.grad = continuation.clone()
    target_parameter.grad = continuation.clone()
    source.step()
    target.step()
    torch.testing.assert_close(target_parameter, source_parameter, rtol=0, atol=0)
    _assert_cpu_boundary(target)


@_CUDA_REQUIRED
def test_activation_and_restore_copy_failures_are_atomic(monkeypatch):
    parameter = torch.nn.Parameter(torch.linspace(-0.5, 0.5, 8, device="cuda"))
    optimizer = _make_gefen(parameter)
    parameter.grad = torch.linspace(-1.0, 1.0, 8, device="cuda")
    optimizer.step()
    state_before = optimizer.state
    parameter_state_before = optimizer.state[parameter]
    codebook_before = optimizer._gefen_codebook
    codebook_value_before = optimizer._gefen_codebook.detach().cpu().clone()
    persistent_before = _persistent_snapshot(optimizer, parameter)
    global_step_before = optimizer._gefen_global_step

    def fail_cpu_copy(_tensor):
        raise RuntimeError("injected activation copy failure")

    monkeypatch.setattr(optimizer, "_copy_state_tensor_to_offload_cpu", fail_cpu_copy)
    with pytest.raises(RuntimeError, match="injected activation"):
        optimizer.offload_state_()
    assert optimizer.state is state_before
    assert optimizer.state[parameter] is parameter_state_before
    assert optimizer._gefen_codebook is codebook_before
    assert not optimizer.state_offload_active
    _assert_persistent_equal(_persistent_snapshot(optimizer, parameter), persistent_before)
    assert torch.equal(optimizer._gefen_codebook.detach().cpu(), codebook_value_before)
    assert optimizer._gefen_global_step == global_step_before

    monkeypatch.undo()
    optimizer.offload_state_()
    state_before = optimizer.state
    parameter_state_before = optimizer.state[parameter]
    persistent_before = _persistent_snapshot(optimizer, parameter)

    def fail_move(_tensor, _device):
        raise RuntimeError("injected restore copy failure")

    monkeypatch.setattr(optimizer, "_copy_state_tensor_for_move", fail_move)
    with pytest.raises(RuntimeError, match="injected restore"):
        optimizer.restore_state_()
    assert optimizer.state is state_before
    assert optimizer.state[parameter] is parameter_state_before
    assert optimizer.state_offload_active
    _assert_cpu_boundary(optimizer)
    _assert_persistent_equal(_persistent_snapshot(optimizer, parameter), persistent_before)
    assert torch.equal(optimizer._gefen_codebook.detach().cpu(), codebook_value_before)
    assert optimizer._gefen_global_step == global_step_before


@_CUDA_REQUIRED
def test_copyback_failure_poison_is_sticky_until_successful_load(monkeypatch):
    parameter = torch.nn.Parameter(torch.linspace(-0.4, 0.7, 8, device="cuda"))
    optimizer = _make_gefen(parameter)
    parameter.grad = torch.linspace(-1.0, 1.0, 8, device="cuda")
    optimizer.step()
    checkpoint = copy.deepcopy(optimizer.state_dict())
    optimizer.offload_state_()
    published_state = optimizer.state[parameter]
    parameter.grad = torch.linspace(0.9, -0.8, 8, device="cuda")

    def fail_copyback(_tensor):
        raise RuntimeError("injected copyback failure")

    monkeypatch.setattr(optimizer, "_copy_state_tensor_to_offload_cpu", fail_copyback)
    with pytest.raises(RuntimeError, match="known-good checkpoint"):
        optimizer.step()

    assert optimizer.state_offload_active
    assert optimizer.state_offload_poisoned
    assert not optimizer.optimizer_contract().capabilities.state_offload
    with pytest.raises(RuntimeError, match="cannot export optimizer state"):
        optimizer.state_dict()
    assert optimizer.state[parameter] is published_state
    assert all(value.device.type == "cpu" for key, value in published_state.items() if key in _PERSISTENT_TENSOR_KEYS)
    parameter_after_failure = parameter.detach().clone()
    with pytest.raises(RuntimeError, match="poisoned"):
        optimizer.step()
    torch.testing.assert_close(parameter, parameter_after_failure, rtol=0, atol=0)

    monkeypatch.undo()
    optimizer.load_state_dict(checkpoint)
    assert optimizer.state_offload_active
    assert not optimizer.state_offload_poisoned
    assert optimizer.optimizer_contract().capabilities.state_offload
    _assert_cpu_boundary(optimizer)


@_CUDA_REQUIRED
def test_active_offload_blocks_rebinding_and_portable_global_state():
    parameter = torch.nn.Parameter(torch.ones(8, device="cuda"))
    optimizer = _make_gefen(parameter)
    del optimizer._resolve_automatic_period
    identity = ParameterIdentity("Model.Weight", (8,))
    group = ProcessGroupIdentity("singleton", ("rank:0",))
    shard = ShardIdentity(
        identity,
        ParameterLayout.REPLICATED,
        LogicalSlice.full(identity),
        placements=(ShardPlacement("checkpoint", PlacementKind.REPLICATE, 0, 1),),
        process_group=group,
        local_member="rank:0",
    )
    manifest = ShardingManifest((shard,))
    codebook_binding = CodebookProcessGroupBinding(
        group,
        "rank:0",
        None,
        parameter.device,
    )
    checkpoint_binding = CheckpointProcessGroupBinding(
        group,
        "rank:0",
        None,
        parameter.device,
    )

    optimizer.offload_state_()
    with pytest.raises(RuntimeError, match="restore state first"):
        optimizer.post_sharding(
            (ParameterRebinding(parameter, parameter, shard),),
            manifest=manifest,
            codebook_process_group=codebook_binding,
        )
    assert not optimizer._canonical_identity_ready()

    optimizer.restore_state_()
    optimizer.post_sharding(
        (ParameterRebinding(parameter, parameter, shard),),
        manifest=manifest,
        codebook_process_group=codebook_binding,
    )
    optimizer.offload_state_()
    assert all(
        support.transport is not CheckpointTransport.CANONICAL_GLOBAL
        for support in optimizer.optimizer_contract().capabilities.checkpoints
    )
    with pytest.raises(RuntimeError, match="active optimizer-state offload"):
        optimizer.export_portable_state(
            checkpoint_process_group=checkpoint_binding,
            transaction_id="active-offload-reject-v1",
            limits=PortableStateLimits(
                max_fragment_tensor_bytes=1 << 20,
                max_collective_tensor_bytes=4 << 20,
                max_collective_metadata_bytes=4 << 20,
                max_metadata_bytes=1 << 20,
            ),
        )


@_CUDA_REQUIRED
def test_state_offload_rejects_persistent_aliases_and_multi_member_scope():
    parameter = torch.nn.Parameter(torch.linspace(-0.5, 0.5, 8, device="cuda"))
    optimizer = _make_gefen(parameter)
    parameter.grad = torch.linspace(-1.0, 1.0, 8, device="cuda")
    optimizer.step()
    parameter_state = optimizer.state[parameter]
    parameter_state["vmean"] = parameter_state["m_magnitude"]
    state_before = optimizer.state

    assert not optimizer.optimizer_contract().capabilities.state_offload
    with pytest.raises(RuntimeError, match="storage aliases"):
        optimizer.offload_state_()
    assert optimizer.state is state_before
    assert parameter_state["vmean"] is parameter_state["m_magnitude"]

    parameter_state["vmean"] = parameter_state["m_magnitude"].clone()
    group = ProcessGroupIdentity("multi", ("rank:0", "rank:1"))
    optimizer._gefen_codebook_process_group = CodebookProcessGroupBinding(
        group,
        "rank:0",
        object(),
        parameter.device,
    )
    with pytest.raises(RuntimeError, match="multi-member"):
        optimizer.offload_state_()


@_CUDA_REQUIRED
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices are required")
def test_state_offload_checks_capture_on_every_parameter_device(monkeypatch):
    first = torch.nn.Parameter(torch.ones(8, device="cuda:0"))
    second = torch.nn.Parameter(torch.ones(8, device="cuda:1"))
    optimizer = Gefen(
        [("first", first), ("second", second)],
        fused=False,
        factored_v_2d=False,
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        lambda: torch.cuda.current_device() == second.device.index,
    )

    assert not optimizer.optimizer_contract().capabilities.atomic_state_movement
    with pytest.raises(RuntimeError, match="CUDA graph capture"):
        optimizer.move_state_()
    with pytest.raises(RuntimeError, match="CUDA graph capture"):
        optimizer.offload_state_()
    assert not optimizer.state_offload_active

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    optimizer.offload_state_()
    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        lambda: torch.cuda.current_device() == second.device.index,
    )
    with pytest.raises(RuntimeError, match="CUDA graph capture"):
        optimizer.restore_state_()
    assert optimizer.state_offload_active


@_CUDA_REQUIRED
def test_operation_error_copies_state_back_without_poisoning(monkeypatch):
    parameter = torch.nn.Parameter(torch.linspace(-0.4, 0.7, 8, device="cuda"))
    optimizer = _make_gefen(parameter)
    parameter.grad = torch.linspace(-1.0, 1.0, 8, device="cuda")
    optimizer.step()
    optimizer.offload_state_()
    original = optimizer._step_automatic

    def update_then_fail(group, name, live_parameter, grad, *, state=None):
        original(group, name, live_parameter, grad, state=state)
        raise RuntimeError("injected update failure")

    monkeypatch.setattr(optimizer, "_step_automatic", update_then_fail)
    parameter.grad = torch.linspace(0.8, -0.6, 8, device="cuda")
    with pytest.raises(RuntimeError, match="injected update"):
        optimizer.step()

    assert not optimizer.state_offload_poisoned
    _assert_cpu_boundary(optimizer)
    assert optimizer.state[parameter]["step"] == 2


@_CUDA_REQUIRED
def test_custom_dtensor_like_and_finalized_nonreplicated_state_fail_closed():
    custom_parameter = torch.nn.Parameter(torch.ones(8, device="cuda"))
    custom = _make_gefen(custom_parameter)
    custom.state[custom_parameter]["extension"] = {"value": 1}
    state_before = custom.state
    with pytest.raises(RuntimeError, match="custom per-parameter state"):
        custom.offload_state_()
    assert custom.state is state_before
    assert not custom.state_offload_active

    dtensor_like_parameter = torch.nn.Parameter(torch.ones(8, device="cuda"))
    dtensor_like_parameter.to_local = lambda: dtensor_like_parameter
    dtensor_like_parameter.placements = ()
    dtensor_like = _make_gefen(dtensor_like_parameter)
    with pytest.raises(RuntimeError, match="ordinary replicated CUDA"):
        dtensor_like.offload_state_()

    flat_parameter = torch.nn.Parameter(torch.ones(8, device="cuda"))
    flattened = _make_gefen(flat_parameter)
    identity = ParameterIdentity("layer.weight", (16,))
    process_group = ProcessGroupIdentity("data_parallel", ("rank:0", "rank:1"))
    shards = tuple(
        ShardIdentity(
            identity,
            ParameterLayout.FLATTENED_ELEMENT_SHARD,
            LogicalSlice(index * 8, 8),
            placements=(
                ShardPlacement(
                    "data_parallel",
                    PlacementKind.FLAT_SHARD,
                    index,
                    2,
                ),
            ),
            process_group=process_group,
            local_member=member,
        )
        for index, member in enumerate(process_group.ordered_members)
    )
    shard = shards[0]
    flattened.rebind_shard(
        flat_parameter,
        flat_parameter,
        shard=shard,
        manifest=ShardingManifest(shards),
    )
    with pytest.raises(RuntimeError, match="finalized replicated"):
        flattened.offload_state_()


@_CUDA_REQUIRED
def test_capturable_compile_and_capture_states_fail_closed(monkeypatch):
    parameter = torch.nn.Parameter(torch.ones(8, device="cuda"))
    capturable = Gefen(
        [("layer.weight", parameter)],
        fused=False,
        factored_v_2d=False,
        capturable=True,
    )
    with pytest.raises(RuntimeError, match="capturable"):
        capturable.offload_state_()

    optimizer = _make_gefen(torch.nn.Parameter(torch.ones(8, device="cuda")))
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    with pytest.raises(RuntimeError, match="torch.compile"):
        optimizer.offload_state_()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="CUDA graph capture"):
        optimizer.offload_state_()


def _replicated_shard(fqn, shape):
    identity = ParameterIdentity(fqn, shape)
    return ShardIdentity(
        identity,
        ParameterLayout.REPLICATED,
        LogicalSlice.full(identity),
    )


def _finalized_replicated_cuda_optimizer(count=3):
    parameters = [
        torch.nn.Parameter(torch.full((4,), float(index + 1), device="cuda"))
        for index in range(count)
    ]
    optimizer = Gefen(
        [
            ("weight{}".format(index), parameter)
            for index, parameter in enumerate(parameters)
        ],
        fused=False,
        factored_v_2d=False,
    )
    optimizer._resolve_automatic_period = lambda *args, **kwargs: 4
    shards = tuple(
        _replicated_shard("Model.Weight{}".format(index), (4,))
        for index in range(count)
    )
    optimizer.post_sharding(
        tuple(
            ParameterRebinding(parameter, parameter, shard)
            for parameter, shard in zip(parameters, shards)
        ),
        manifest=ShardingManifest(shards),
    )
    return optimizer, parameters


def _step_offload(optimizer, parameters):
    for parameter in parameters:
        parameter.grad = torch.full_like(parameter, 0.5)
    optimizer.step()


def _count_full_layout_passes(monkeypatch):
    calls = {"count": 0}
    original = Gefen._finalized_binding_layout_matches_full

    def counted(self):
        calls["count"] += 1
        return original(self)

    monkeypatch.setattr(Gefen, "_finalized_binding_layout_matches_full", counted)
    return calls


def _count_manifest_digest_computes(monkeypatch):
    calls = {"count": 0}
    original = Gefen._compute_codebook_manifest_fingerprint

    def counted(self, manifest):
        calls["count"] += 1
        return original(self, manifest)

    monkeypatch.setattr(Gefen, "_compute_codebook_manifest_fingerprint", counted)
    return calls


@_CUDA_REQUIRED
def test_offload_steady_state_reuses_cached_layout_forensics(monkeypatch):
    # The finalized layout is immutable across steps, so the per-step offload
    # readiness path (called ~2x/step) must reuse the memoized layout-forensics
    # verdict instead of rebuilding the full layout + recomputing the manifest
    # digest on every step. The per-tensor offload scan still runs every step;
    # only the immutable layout forensics is cached.
    optimizer, parameters = _finalized_replicated_cuda_optimizer()
    optimizer.offload_state_()

    layout = _count_full_layout_passes(monkeypatch)
    digest = _count_manifest_digest_computes(monkeypatch)

    # The first offloaded step still fully validates the layout once, because
    # offload_state_ invalidated the cached verdict; that single full pass
    # recomputes the manifest digest once.
    _step_offload(optimizer, parameters)
    first_layout = layout["count"]
    first_digest = digest["count"]
    assert first_layout >= 1
    assert first_digest >= 1

    # Steady state: neither the full layout rebuild nor the manifest digest is
    # recomputed again. On the pre-fix code each of the two per-step readiness
    # calls forced a full rebuild + digest recompute, so these counts grew by
    # four per step.
    for _ in range(3):
        _step_offload(optimizer, parameters)
    assert layout["count"] == first_layout
    assert digest["count"] == first_digest


@_CUDA_REQUIRED
def test_offload_layout_change_is_still_caught_with_a_warm_verdict():
    optimizer, parameters = _finalized_replicated_cuda_optimizer()
    optimizer.offload_state_()
    _step_offload(optimizer, parameters)  # warm the cached layout verdict

    # Replace a finalized registry container so the fast-path tokens diverge:
    # the offload readiness path must fall back to the full forensic rebuild and
    # reject the step before any parameter is staged or mutated.
    first_before = parameters[0].detach().clone()
    optimizer._gefen_local_shard_bindings = tuple(
        reversed(optimizer._gefen_local_shard_bindings)
    )
    for parameter in parameters:
        parameter.grad = torch.full_like(parameter, 0.5)
    with pytest.raises(RuntimeError, match="cannot step"):
        optimizer.step()
    assert torch.equal(parameters[0].detach(), first_before)

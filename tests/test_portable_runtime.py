"""Warning-strict CPU coverage for the optimizer-facing portable runtime."""

import copy

import pytest
import torch

from gefen.checkpoint import CheckpointProcessGroupBinding
from gefen.codebook import CodebookProcessGroupBinding
from gefen.contracts import (
    CheckpointTransport,
    LogicalSlice,
    ParameterIdentity,
    ParameterLayout,
    PlacementKind,
    PortableStateProvider,
    ProcessGroupScope,
    ProcessGroupIdentity,
    ShardIdentity,
    ShardPlacement,
    ShardingManifest,
    TopologyChange,
)
from gefen.gefen import Gefen
from gefen.gefen_muon import GefenMuon
from gefen.portable import _decode_quantized_momentum, _recompress_dense_momentum
from gefen.portable_runtime import (
    _export_portable_state,
    _import_portable_state,
    _optimizer_implementation,
    _portable_runtime_layouts,
)
from gefen.portable_state import PortableStateLimits
from gefen.rebinding import ParameterRebinding


_MEMBER = "rank:0"


def _limits():
    return PortableStateLimits(
        max_fragment_tensor_bytes=4 << 20,
        max_collective_tensor_bytes=16 << 20,
        max_collective_metadata_bytes=64 << 20,
    )


def _bindings(group):
    return (
        CodebookProcessGroupBinding(group, _MEMBER, None, torch.device("cpu")),
        CheckpointProcessGroupBinding(group, _MEMBER, None, torch.device("cpu")),
    )


def _finalize(optimizer, parameter, *, layout, global_shape=None):
    group = ProcessGroupIdentity("checkpoint", (_MEMBER,))
    identity = ParameterIdentity(
        "layer.weight",
        tuple(parameter.shape) if global_shape is None else global_shape,
    )
    if layout is ParameterLayout.REPLICATED:
        kind = PlacementKind.REPLICATE
        owner = None
    elif layout is ParameterLayout.FLATTENED_ELEMENT_SHARD:
        kind = PlacementKind.FLAT_SHARD
        owner = None
    elif layout is ParameterLayout.WHOLE_PARAMETER_OWNER:
        kind = PlacementKind.WHOLE_PARAMETER_OWNER
        owner = _MEMBER
    else:
        raise AssertionError("unsupported test layout")
    shard = ShardIdentity(
        identity,
        layout,
        LogicalSlice.full(identity),
        placements=(ShardPlacement("checkpoint", kind, 0, 1),),
        process_group=group,
        local_member=_MEMBER,
        owner=owner,
    )
    codebook_binding, checkpoint_binding = _bindings(group)
    optimizer.post_sharding(
        (ParameterRebinding(parameter, parameter, shard),),
        manifest=ShardingManifest((shard,)),
        codebook_process_group=codebook_binding,
    )
    return checkpoint_binding


def _plain(*, factored, deterministic, layout=ParameterLayout.REPLICATED):
    values = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
    parameter = torch.nn.Parameter(
        values.reshape(-1) if layout is ParameterLayout.FLATTENED_ELEMENT_SHARD else values
    )
    optimizer = Gefen(
        [("weight", parameter)],
        fused=False,
        factored_v_2d=factored,
        force_2d_period_one=True,
        deterministic=deterministic,
    )
    binding = _finalize(
        optimizer,
        parameter,
        layout=layout,
        global_shape=(2, 3),
    )
    return optimizer, parameter, binding


def _muon(*, deterministic):
    parameter = torch.nn.Parameter(torch.arange(1, 7, dtype=torch.float32).reshape(2, 3))
    optimizer = GefenMuon(
        [("weight", parameter)],
        fused=False,
        sharded_mode="distributed",
        normuon=True,
        deterministic=deterministic,
    )
    binding = _finalize(
        optimizer,
        parameter,
        layout=ParameterLayout.WHOLE_PARAMETER_OWNER,
    )
    return optimizer, parameter, binding


def _initialize(optimizer, parameter, *, variant):
    optimizer._gefen_global_step = 4
    optimizer._gefen_codebook = torch.linspace(-1.0, 1.0, 256, dtype=torch.float32)
    state = optimizer.state[parameter]
    state.update(
        {
            "automatic_period": 1,
            "step": 4,
            "m_codebook": torch.tensor([[0], [255], [128], [64], [192], [0]], dtype=torch.uint8),
            "m_magnitude": torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]], dtype=torch.float32),
        }
    )
    if variant == "block":
        state.update(
            {
                "vmean": torch.arange(1, 7, dtype=torch.float32).reshape(6, 1),
                "vmean_step": 3,
            }
        )
    elif variant == "factored":
        state.update(
            {
                "v_row": torch.tensor([2.0, 3.0], dtype=torch.float32),
                "v_col": torch.tensor([5.0, 7.0, 11.0], dtype=torch.float32),
                "factored_step": 3,
            }
        )
    elif variant == "normuon":
        state.update(
            {
                "normuon_v": torch.tensor([[2.0], [3.0]], dtype=torch.float32),
                "normuon_step": 2,
            }
        )
    else:
        raise AssertionError("unknown test variant")


@pytest.mark.parametrize("variant", ["block", "factored", "normuon"])
def test_initialized_singleton_export_import_round_trip(variant):
    if variant == "normuon":
        source, source_parameter, source_binding = _muon(deterministic=True)
        target, target_parameter, target_binding = _muon(deterministic=False)
    else:
        source, source_parameter, source_binding = _plain(
            factored=variant == "factored",
            deterministic=True,
        )
        target, target_parameter, target_binding = _plain(
            factored=variant == "factored",
            deterministic=False,
        )
    _initialize(source, source_parameter, variant=variant)
    assert isinstance(source, PortableStateProvider)
    support = next(
        item
        for item in source.optimizer_contract().capabilities.checkpoints
        if item.transport is CheckpointTransport.CANONICAL_GLOBAL
    )
    assert support.requires_collective
    assert support.atomic_load
    assert support.process_group_scope is ProcessGroupScope.ADAPTER_DEFINED
    expected_layout = ParameterLayout.WHOLE_PARAMETER_OWNER if variant == "normuon" else ParameterLayout.REPLICATED
    assert expected_layout in support.same_topology
    if variant == "factored":
        assert not support.topology_changing
    elif variant == "block":
        assert support.topology_changing == frozenset(
            {
                    ParameterLayout.REPLICATED,
                    ParameterLayout.FLATTENED_ELEMENT_SHARD,
                    ParameterLayout.DTENSOR_1D_DEFAULT_WORLD,
                }
            )
        assert support.topology_change_kinds == frozenset({TopologyChange.PLACEMENT_RESHARD})
    else:
        assert support.topology_changing == frozenset(
            {
                ParameterLayout.REPLICATED,
                ParameterLayout.WHOLE_PARAMETER_OWNER,
            }
        )
        assert support.topology_change_kinds == frozenset(
            {
                TopologyChange.PLACEMENT_RESHARD,
                TopologyChange.WORLD_SIZE_OWNER_REDISTRIBUTION,
            }
        )

    document = _export_portable_state(
        source,
        checkpoint_process_group=source_binding,
        transaction_id="export-{}".format(variant),
        limits=_limits(),
    )
    _import_portable_state(
        target,
        document,
        checkpoint_process_group=target_binding,
        transaction_id="import-{}".format(variant),
        limits=_limits(),
    )

    assert target._gefen_global_step == 4
    assert target._deterministic is True
    assert torch.equal(target._gefen_codebook, document["common"]["gefen_codebook"])
    target_state = target.state[target_parameter]
    target_momentum = _decode_quantized_momentum(
        target._gefen_codebook,
        target_state["m_codebook"],
        target_state["m_magnitude"],
        logical_shape=tuple(target_parameter.shape),
        period=1,
        step=target_state["step"],
    )
    record = document["parameters"]["layer.weight"]
    assert torch.equal(target_momentum.view(torch.int32), record["state"]["momentum"].view(torch.int32))
    if variant == "block":
        assert torch.equal(target_state["vmean"].reshape(2, 3), record["state"]["second_moment"])
        assert target_state["vmean_step"] == 3
    elif variant == "factored":
        assert torch.equal(target_state["v_row"], record["state"]["v_row"])
        assert torch.equal(target_state["v_col"], record["state"]["v_col"])
        assert target_state["factored_step"] == 3
    else:
        assert torch.equal(target_state["normuon_v"], record["state"]["normuon_v"])
        assert target_state["normuon_step"] == 2


def test_factored_to_flat_block_import_matches_target_topology_reference_continuation():
    source, source_parameter, source_binding = _plain(factored=True, deterministic=True)
    target, target_parameter, target_binding = _plain(
        factored=False,
        deterministic=False,
        layout=ParameterLayout.FLATTENED_ELEMENT_SHARD,
    )
    reference, reference_parameter, _ = _plain(
        factored=False,
        deterministic=True,
        layout=ParameterLayout.FLATTENED_ELEMENT_SHARD,
    )
    _initialize(source, source_parameter, variant="factored")
    document = _export_portable_state(
        source,
        checkpoint_process_group=source_binding,
        transaction_id="export-factored-to-block",
        limits=_limits(),
    )
    assert "factored_to_block_live_fp32" in document["policy"]["second_moment_projection"]

    _import_portable_state(
        target,
        document,
        checkpoint_process_group=target_binding,
        transaction_id="import-factored-to-block",
        limits=_limits(),
    )

    record = document["parameters"]["layer.weight"]
    expected_second = torch.outer(record["state"]["v_row"], record["state"]["v_col"]).div_(
        record["state"]["v_row"].mean().clamp_(min=torch.finfo(torch.float32).tiny)
    )
    target_state = target.state[target_parameter]
    assert torch.equal(target_state["vmean"].reshape(2, 3).view(torch.int32), expected_second.view(torch.int32))
    assert target_state["vmean_step"] == record["state"]["factored_step"]

    reference._gefen_global_step = document["common"]["gefen_global_step"]
    reference._gefen_codebook = document["common"]["gefen_codebook"].clone()
    reference._deterministic = document["common"]["gefen_deterministic"]
    reference_indices, reference_magnitudes = _recompress_dense_momentum(
        record["state"]["momentum"],
        reference._gefen_codebook,
        period=1,
        step=record["state"]["step"],
    )
    reference.state[reference_parameter].update(
        {
            "automatic_period": 1,
            "step": record["state"]["step"],
            "m_codebook": reference_indices,
            "m_magnitude": reference_magnitudes,
            "vmean": expected_second.reshape(-1, 1).clone(),
            "vmean_step": record["state"]["factored_step"],
        }
    )

    gradient = torch.tensor([[-0.75, 0.5, 2.0], [3.25, -1.5, 0.125]], dtype=torch.float32)
    target_parameter.grad = gradient.reshape(-1).clone()
    reference_parameter.grad = gradient.reshape(-1).clone()
    target.step()
    reference.step()

    assert torch.equal(target_parameter.view(torch.int32), reference_parameter.view(torch.int32))
    assert target._gefen_global_step == reference._gefen_global_step == 5
    reference_state = reference.state[reference_parameter]
    for key in ("automatic_period", "step", "vmean_step"):
        assert target_state[key] == reference_state[key]
    assert torch.equal(target_state["m_codebook"], reference_state["m_codebook"])
    for key in ("m_magnitude", "vmean"):
        assert torch.equal(target_state[key].view(torch.int32), reference_state[key].view(torch.int32))


def test_factored_to_block_projection_overflow_fails_before_live_mutation():
    source, source_parameter, source_binding = _plain(factored=True, deterministic=True)
    target, target_parameter, target_binding = _plain(factored=False, deterministic=False)
    _initialize(source, source_parameter, variant="factored")
    _initialize(target, target_parameter, variant="block")
    maximum = torch.finfo(torch.float32).max
    source.state[source_parameter]["v_row"] = torch.full((2,), maximum)
    source.state[source_parameter]["v_col"] = torch.full((3,), maximum)
    document = _export_portable_state(
        source,
        checkpoint_process_group=source_binding,
        transaction_id="export-factored-overflow",
        limits=_limits(),
    )
    before = target._canonical_import_live_token()

    with pytest.raises(RuntimeError, match="cannot be represented"):
        _import_portable_state(
            target,
            document,
            checkpoint_process_group=target_binding,
            transaction_id="import-factored-overflow",
            limits=_limits(),
        )

    assert target._canonical_import_live_token() == before


def test_block_to_factored_import_is_rejected_before_live_mutation():
    source, source_parameter, source_binding = _plain(factored=False, deterministic=True)
    target, target_parameter, target_binding = _plain(factored=True, deterministic=False)
    _initialize(source, source_parameter, variant="block")
    _initialize(target, target_parameter, variant="factored")
    document = _export_portable_state(
        source,
        checkpoint_process_group=source_binding,
        transaction_id="export-block-for-reverse-rejection",
        limits=_limits(),
    )
    before = target._canonical_import_live_token()

    with pytest.raises(RuntimeError, match="block-to-factored"):
        _import_portable_state(
            target,
            document,
            checkpoint_process_group=target_binding,
            transaction_id="import-block-to-factored",
            limits=_limits(),
        )

    assert target._canonical_import_live_token() == before


def test_import_rejects_corrupt_document_without_live_mutation():
    source, source_parameter, source_binding = _plain(factored=False, deterministic=True)
    target, _, target_binding = _plain(factored=False, deterministic=False)
    _initialize(source, source_parameter, variant="block")
    document = _export_portable_state(
        source,
        checkpoint_process_group=source_binding,
        transaction_id="export-corruption-source",
        limits=_limits(),
    )
    corrupt = copy.deepcopy(document)
    corrupt["completion"]["digest"] = "0" * 64
    before = target._canonical_import_live_token()

    with pytest.raises(RuntimeError, match="digest"):
        _import_portable_state(
            target,
            corrupt,
            checkpoint_process_group=target_binding,
            transaction_id="import-corruption",
            limits=_limits(),
        )

    assert target._canonical_import_live_token() == before


def test_export_rejects_non_period_one_and_unknown_group_state():
    optimizer, parameter, binding = _plain(factored=False, deterministic=False)
    optimizer._gefen_global_step = 2
    optimizer._gefen_codebook = torch.linspace(-1.0, 1.0, 256, dtype=torch.float32)
    optimizer.state[parameter].update(
        {
            "automatic_period": 2,
            "step": 2,
            "m_codebook": torch.zeros((3, 2), dtype=torch.uint8),
            "m_magnitude": torch.ones((3, 1), dtype=torch.float32),
            "vmean": torch.ones((3, 1), dtype=torch.float32),
            "vmean_step": 2,
        }
    )
    with pytest.raises(RuntimeError, match="period one"):
        _export_portable_state(
            optimizer,
            checkpoint_process_group=binding,
            transaction_id="export-period-two",
            limits=_limits(),
        )

    optimizer.state[parameter] = {"name": "weight"}
    optimizer.param_groups[0]["extension"] = 1
    with pytest.raises(RuntimeError, match="unknown keys"):
        _export_portable_state(
            optimizer,
            checkpoint_process_group=binding,
            transaction_id="export-unknown-group-option",
            limits=_limits(),
        )


def test_exact_type_and_full_process_group_identity_are_required():
    class GefenSubclass(Gefen):
        pass

    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
    optimizer = GefenSubclass([("weight", parameter)], fused=False)
    with pytest.raises(TypeError, match="exact Gefen"):
        _optimizer_implementation(optimizer)

    exact, _, _ = _plain(factored=False, deterministic=False)
    wrong_group = ProcessGroupIdentity("different-checkpoint", (_MEMBER,))
    _, wrong_binding = _bindings(wrong_group)
    with pytest.raises(RuntimeError, match="identities must match"):
        _export_portable_state(
            exact,
            checkpoint_process_group=wrong_binding,
            transaction_id="export-wrong-process-group",
            limits=_limits(),
        )


def test_runtime_readiness_fails_closed_on_state_group_and_subclass_mutation():
    optimizer, parameter, binding = _plain(factored=False, deterministic=False)
    assert _portable_runtime_layouts(optimizer) == frozenset({ParameterLayout.REPLICATED})

    optimizer.state[parameter]["automatic_period"] = 2
    assert not _portable_runtime_layouts(optimizer)
    assert all(
        support.transport is not CheckpointTransport.CANONICAL_GLOBAL
        for support in optimizer.optimizer_contract().capabilities.checkpoints
    )
    optimizer.state[parameter] = {"name": "weight"}
    optimizer.param_groups[0]["extension"] = 1
    assert not _portable_runtime_layouts(optimizer)
    optimizer.param_groups[0].pop("extension")

    _initialize(optimizer, parameter, variant="block")
    optimizer.state[parameter]["m_magnitude"] = optimizer.state[parameter]["m_magnitude"].requires_grad_()
    assert not _portable_runtime_layouts(optimizer)
    with pytest.raises(RuntimeError, match="plain materialized"):
        _export_portable_state(
            optimizer,
            checkpoint_process_group=binding,
            transaction_id="export-grad-state",
            limits=_limits(),
        )

    class GefenSubclass(Gefen):
        pass

    subclass_parameter = torch.nn.Parameter(torch.ones(2))
    subclassed = GefenSubclass([("weight", subclass_parameter)], fused=False)
    assert not _portable_runtime_layouts(subclassed)


def test_muon_readiness_rejects_nonexact_beta_and_unsafe_raw_schedule():
    optimizer, _, binding = _muon(deterministic=False)
    optimizer.param_groups[0]["beta2"] = torch.tensor(0.0)
    assert not _portable_runtime_layouts(optimizer)
    with pytest.raises(RuntimeError, match="exact floats"):
        _export_portable_state(
            optimizer,
            checkpoint_process_group=binding,
            transaction_id="export-tensor-beta",
            limits=_limits(),
        )

    optimizer.param_groups[0]["beta2"] = 0.0
    optimizer.param_groups[0]["ns_schedule"] = lambda: None
    assert not _portable_runtime_layouts(optimizer)
    with pytest.raises(RuntimeError, match="weights-only-safe"):
        _export_portable_state(
            optimizer,
            checkpoint_process_group=binding,
            transaction_id="export-callable-schedule",
            limits=_limits(),
        )


def test_invalid_collective_arguments_fail_through_the_status_gate():
    optimizer, _, binding = _plain(factored=False, deterministic=False)
    with pytest.raises(RuntimeError, match="PortableStateLimits"):
        _export_portable_state(
            optimizer,
            checkpoint_process_group=binding,
            transaction_id="invalid-limits",
            limits=object(),
        )
    with pytest.raises(RuntimeError, match="transaction_id"):
        _export_portable_state(
            optimizer,
            checkpoint_process_group=binding,
            transaction_id="",
            limits=_limits(),
        )
    with pytest.raises(RuntimeError, match="trimmed"):
        _export_portable_state(
            optimizer,
            checkpoint_process_group=binding,
            transaction_id=" invalid-transaction ",
            limits=_limits(),
        )

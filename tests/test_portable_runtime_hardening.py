"""Warning-strict CPU regressions for portable runtime publication guards."""

import pytest
import torch

import gefen.portable_runtime as portable_runtime
from gefen.checkpoint import CheckpointProcessGroupBinding
from gefen.codebook import CodebookProcessGroupBinding
from gefen.contracts import (
    CheckpointTransport,
    LogicalSlice,
    ParameterIdentity,
    ParameterLayout,
    PlacementKind,
    ProcessGroupIdentity,
    ShardIdentity,
    ShardPlacement,
    ShardingManifest,
)
from gefen.gefen import Gefen
from gefen.portable_state import PortableStateLimits
from gefen.rebinding import ParameterRebinding


_MEMBER = "rank:0"


class _ParameterIdentitySubclass(ParameterIdentity):
    pass


class _ProcessGroupIdentitySubclass(ProcessGroupIdentity):
    pass


class _ShardPlacementSubclass(ShardPlacement):
    pass


class _StringSubclass(str):
    pass


def _limits(*, max_fragment_tensor_bytes=4 << 20):
    return PortableStateLimits(
        max_fragment_tensor_bytes=max_fragment_tensor_bytes,
        max_collective_tensor_bytes=16 << 20,
        max_collective_metadata_bytes=64 << 20,
    )


@pytest.mark.parametrize("subclass_kind", ["parameter", "process_group", "placement"])
def test_nested_identity_subclasses_are_rejected(subclass_kind):
    parameter_type = _ParameterIdentitySubclass if subclass_kind == "parameter" else ParameterIdentity
    group_type = _ProcessGroupIdentitySubclass if subclass_kind == "process_group" else ProcessGroupIdentity
    placement_type = _ShardPlacementSubclass if subclass_kind == "placement" else ShardPlacement
    identity = parameter_type("layer.weight", (2, 3))
    group = group_type("checkpoint", (_MEMBER,))
    shard = ShardIdentity(
        identity,
        ParameterLayout.REPLICATED,
        LogicalSlice.full(identity),
        placements=(
            placement_type(
                "checkpoint",
                PlacementKind.REPLICATE,
                0,
                1,
            ),
        ),
        process_group=group,
        local_member=_MEMBER,
    )

    with pytest.raises(TypeError, match="exact"):
        portable_runtime._validate_exact_shard_identity(shard)


def _optimizer(*, layout, deterministic):
    identity = ParameterIdentity("layer.weight", (2, 3))
    if layout is ParameterLayout.REPLICATED:
        parameter = torch.nn.Parameter(torch.arange(1, 7, dtype=torch.float32).reshape(identity.global_shape))
        placement_kind = PlacementKind.REPLICATE
    elif layout is ParameterLayout.FLATTENED_ELEMENT_SHARD:
        parameter = torch.nn.Parameter(torch.arange(1, 7, dtype=torch.float32))
        placement_kind = PlacementKind.FLAT_SHARD
    else:
        raise AssertionError("unsupported test layout")

    optimizer = Gefen(
        [("weight", parameter)],
        fused=False,
        factored_v_2d=False,
        force_2d_period_one=True,
        deterministic=deterministic,
    )
    group = ProcessGroupIdentity("checkpoint", (_MEMBER,))
    shard = ShardIdentity(
        identity,
        layout,
        LogicalSlice.full(identity),
        placements=(ShardPlacement("checkpoint", placement_kind, 0, 1),),
        process_group=group,
        local_member=_MEMBER,
    )
    optimizer.post_sharding(
        (ParameterRebinding(parameter, parameter, shard),),
        manifest=ShardingManifest((shard,)),
        codebook_process_group=CodebookProcessGroupBinding(
            group,
            _MEMBER,
            None,
            torch.device("cpu"),
        ),
    )
    binding = CheckpointProcessGroupBinding(
        group,
        _MEMBER,
        None,
        torch.device("cpu"),
    )
    return optimizer, parameter, binding


def _initialize(optimizer, parameter):
    optimizer._gefen_global_step = 4
    optimizer._gefen_codebook = torch.linspace(-1.0, 1.0, 256, dtype=torch.float32)
    optimizer.state[parameter].update(
        {
            "automatic_period": 1,
            "step": 4,
            "m_codebook": torch.tensor(
                [[0], [255], [128], [64], [192], [0]],
                dtype=torch.uint8,
            ),
            "m_magnitude": torch.tensor(
                [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]],
                dtype=torch.float32,
            ),
            "vmean": torch.arange(1, 7, dtype=torch.float32).reshape(6, 1),
            "vmean_step": 3,
        }
    )


def _has_canonical_global(optimizer):
    return any(
        support.transport is CheckpointTransport.CANONICAL_GLOBAL
        for support in optimizer.optimizer_contract().capabilities.checkpoints
    )


@pytest.mark.parametrize(
    ("descriptor", "field", "value"),
    [
        ("manifest", "schema_version", 2),
        ("shard", "schema_version", 2),
        ("parameter", "schema_version", 2),
        ("process_group", "schema_version", 2),
        ("process_group", "semantic_name", ""),
        ("process_group", "semantic_name", " checkpoint"),
        ("process_group", "semantic_name", "check\x00point"),
        ("parameter", "fqn", ""),
        ("parameter", "fqn", " layer.weight"),
        ("parameter", "fqn", "layer..weight"),
        ("logical_slice", "flat_offset", -1),
        ("placement", "mesh_axis", " checkpoint"),
        ("placement", "coordinate", -1),
        ("placement", "parts", 0),
    ],
)
def test_mutated_descriptor_invariants_disable_portable_capability(
    descriptor,
    field,
    value,
):
    optimizer, _, _ = _optimizer(
        layout=ParameterLayout.REPLICATED,
        deterministic=True,
    )
    manifest = optimizer._gefen_sharding_manifest
    shard = manifest.shards[0]
    target = {
        "manifest": manifest,
        "shard": shard,
        "parameter": shard.parameter,
        "process_group": shard.process_group,
        "logical_slice": shard.logical_slice,
        "placement": shard.placements[0],
    }[descriptor]
    object.__setattr__(target, field, value)

    with pytest.raises((TypeError, ValueError)):
        portable_runtime._validate_exact_manifest(manifest)
    assert not _has_canonical_global(optimizer)


def test_codebook_binding_local_member_subclass_disables_portable_capability():
    optimizer, _, _ = _optimizer(
        layout=ParameterLayout.REPLICATED,
        deterministic=True,
    )
    scope = optimizer._gefen_codebook_process_group
    object.__setattr__(scope, "local_member", _StringSubclass(scope.local_member))

    assert not _has_canonical_global(optimizer)


def _drift_storage(parameter, drift):
    if drift == "replicated_shape":
        parameter.data = parameter.detach().reshape(3, 2)
    elif drift == "flat_shape":
        parameter.data = parameter.detach().reshape(2, 3)
    elif drift == "flat_noncontiguous":
        parameter.data = torch.arange(12, dtype=torch.float32)[::2]
    elif drift == "complex_dtype":
        parameter.data = parameter.detach().to(dtype=torch.complex64)
    else:
        raise AssertionError("unknown storage drift")


@pytest.mark.parametrize(
    ("layout", "drift"),
    [
        (ParameterLayout.REPLICATED, "replicated_shape"),
        (ParameterLayout.FLATTENED_ELEMENT_SHARD, "flat_shape"),
        (ParameterLayout.FLATTENED_ELEMENT_SHARD, "flat_noncontiguous"),
        (ParameterLayout.REPLICATED, "complex_dtype"),
    ],
)
def test_finalized_parameter_storage_drift_disables_and_rejects_portable_io(
    layout,
    drift,
):
    source, source_parameter, source_binding = _optimizer(
        layout=layout,
        deterministic=True,
    )
    _initialize(source, source_parameter)
    document = source.export_portable_state(
        checkpoint_process_group=source_binding,
        transaction_id="hardening-source-export-{}".format(drift),
        limits=_limits(),
    )

    _drift_storage(source_parameter, drift)
    assert not _has_canonical_global(source)
    with pytest.raises(RuntimeError, match="geometry|dtype"):
        source.export_portable_state(
            checkpoint_process_group=source_binding,
            transaction_id="hardening-drifted-export-{}".format(drift),
            limits=_limits(),
        )

    target, target_parameter, target_binding = _optimizer(
        layout=layout,
        deterministic=False,
    )
    _drift_storage(target_parameter, drift)
    assert not _has_canonical_global(target)
    state_mapping = target.state
    parameter_state = target.state[target_parameter]
    defaults = target.defaults
    with pytest.raises(RuntimeError, match="geometry|dtype"):
        target.import_portable_state(
            document,
            checkpoint_process_group=target_binding,
            transaction_id="hardening-drifted-import-{}".format(drift),
            limits=_limits(),
        )
    assert target.state is state_mapping
    assert target.state[target_parameter] is parameter_state
    assert parameter_state == {"name": "weight"}
    assert target.defaults is defaults
    assert target._gefen_global_step == 0
    assert target._gefen_codebook is None
    assert target._deterministic is False


class _UpdateBombDict(dict):
    update_calls = 0

    def update(self, *args, **kwargs):
        self.update_calls += 1
        self["publication_started"] = True
        raise AssertionError("defaults publication must not start")


def test_defaults_dict_subclass_is_rejected_before_publication():
    source, source_parameter, source_binding = _optimizer(
        layout=ParameterLayout.REPLICATED,
        deterministic=True,
    )
    _initialize(source, source_parameter)
    document = source.export_portable_state(
        checkpoint_process_group=source_binding,
        transaction_id="hardening-defaults-source",
        limits=_limits(),
    )
    target, target_parameter, target_binding = _optimizer(
        layout=ParameterLayout.REPLICATED,
        deterministic=False,
    )
    hostile_defaults = dict.__new__(_UpdateBombDict)
    dict.update(hostile_defaults, target.defaults)
    hostile_defaults.update_calls = 0
    target.defaults = hostile_defaults
    before_defaults = dict(hostile_defaults)
    state_mapping = target.state
    parameter_state = target.state[target_parameter]

    assert not _has_canonical_global(target)
    with pytest.raises(RuntimeError, match="exact built-in optimizer defaults"):
        target.import_portable_state(
            document,
            checkpoint_process_group=target_binding,
            transaction_id="hardening-defaults-import",
            limits=_limits(),
        )

    assert target.defaults is hostile_defaults
    assert dict(hostile_defaults) == before_defaults
    assert hostile_defaults.update_calls == 0
    assert "publication_started" not in hostile_defaults
    assert target.state is state_mapping
    assert target.state[target_parameter] is parameter_state
    assert parameter_state == {"name": "weight"}
    assert target._gefen_global_step == 0
    assert target._gefen_codebook is None
    assert target._deterministic is False


def test_foreign_state_inserted_after_prepare_fails_freshness_without_publication(
    monkeypatch,
):
    source, source_parameter, source_binding = _optimizer(
        layout=ParameterLayout.REPLICATED,
        deterministic=True,
    )
    _initialize(source, source_parameter)
    document = source.export_portable_state(
        checkpoint_process_group=source_binding,
        transaction_id="hardening-freshness-source",
        limits=_limits(),
    )
    target, target_parameter, target_binding = _optimizer(
        layout=ParameterLayout.REPLICATED,
        deterministic=False,
    )
    assert _has_canonical_global(target)
    state_mapping = target.state
    parameter_state = target.state[target_parameter]
    foreign_parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    foreign_state = {"name": "foreign"}
    injected = False
    original_status = portable_runtime._collective_unanimous_status

    def status_with_concurrent_insertion(*args, **kwargs):
        nonlocal injected
        result = original_status(*args, **kwargs)
        if kwargs["operation"] == "portable_import_prepare" and not injected:
            target.state[foreign_parameter] = foreign_state
            injected = True
        return result

    monkeypatch.setattr(
        portable_runtime,
        "_collective_unanimous_status",
        status_with_concurrent_insertion,
    )
    with pytest.raises(RuntimeError, match="foreign parameter keys|changed after"):
        target.import_portable_state(
            document,
            checkpoint_process_group=target_binding,
            transaction_id="hardening-freshness-import",
            limits=_limits(),
        )

    assert injected
    assert target.state is state_mapping
    assert target.state[target_parameter] is parameter_state
    assert parameter_state == {"name": "weight"}
    assert target.state[foreign_parameter] is foreign_state
    assert target._gefen_global_step == 0
    assert target._gefen_codebook is None
    assert target._deterministic is False


def test_fragment_limit_rejects_before_dense_momentum_materialization(monkeypatch):
    optimizer, parameter, binding = _optimizer(
        layout=ParameterLayout.REPLICATED,
        deterministic=True,
    )
    _initialize(optimizer, parameter)
    reached_decode = False
    reached_live_token = False

    def fail_if_decoded(*args, **kwargs):
        nonlocal reached_decode
        reached_decode = True
        raise AssertionError("dense momentum decode was reached")

    def fail_if_tokenized(*args, **kwargs):
        nonlocal reached_live_token
        reached_live_token = True
        raise AssertionError("content-bearing live token was reached")

    monkeypatch.setattr(
        portable_runtime,
        "_decode_local_momentum",
        fail_if_decoded,
    )
    monkeypatch.setattr(
        portable_runtime,
        "_portable_live_token",
        fail_if_tokenized,
    )
    with pytest.raises(RuntimeError, match="max_fragment_tensor_bytes"):
        optimizer.export_portable_state(
            checkpoint_process_group=binding,
            transaction_id="hardening-fragment-budget",
            limits=_limits(max_fragment_tensor_bytes=1),
        )
    assert not reached_decode
    assert not reached_live_token


def test_pristine_import_does_not_bound_or_materialize_obsolete_target_payload():
    source, _source_parameter, source_binding = _optimizer(
        layout=ParameterLayout.REPLICATED,
        deterministic=True,
    )
    tiny_limits = _limits(max_fragment_tensor_bytes=1)
    document = source.export_portable_state(
        checkpoint_process_group=source_binding,
        transaction_id="hardening-pristine-source",
        limits=tiny_limits,
    )

    target, target_parameter, target_binding = _optimizer(
        layout=ParameterLayout.REPLICATED,
        deterministic=False,
    )
    _initialize(target, target_parameter)
    old_state = target.state[target_parameter]
    target.import_portable_state(
        document,
        checkpoint_process_group=target_binding,
        transaction_id="hardening-pristine-target",
        limits=tiny_limits,
    )

    assert target.state[target_parameter] is not old_state
    assert target.state[target_parameter] == {"name": "weight"}
    assert target._gefen_global_step == 0
    assert target._gefen_codebook is None
    assert target._deterministic is True

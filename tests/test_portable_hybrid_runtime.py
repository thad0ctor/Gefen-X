"""Warning-strict runtime coverage for Gefen-backed Hybrid portable state."""

import copy
import warnings

import pytest
import torch
import torch.distributed.checkpoint as dcp

from gefen import (
    CheckpointProcessGroupBinding,
    CheckpointTransport,
    GefenMuonHybrid,
    PortableStateLimits,
    PortableStateProvider,
    load_portable_dcp,
    save_portable_dcp,
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
from gefen.portable_schema import portable_state_digest


_MEMBER = "rank:0"
_SINGLE_PROCESS_DCP_WARNING = (
    r"^torch\.distributed is (?:disabled, )?unavailable or uninitialized, "
    r"assuming the intent is to (?:save|load) in a single process\.$"
)
_TYPED_STORAGE_DCP_WARNING = (
    r"^TypedStorage is deprecated\. It will be removed in the future and UntypedStorage will be the only storage class\. "
    r"This should only matter to you if you are using storages directly\.  To access UntypedStorage directly, use "
    r"tensor\.untyped_storage\(\) instead of tensor\.storage\(\)$"
)


@pytest.fixture(autouse=True)
def _warning_strict():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings(
            "ignore",
            message=_SINGLE_PROCESS_DCP_WARNING,
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=_TYPED_STORAGE_DCP_WARNING,
            category=UserWarning,
        )
        yield


def _limits():
    return PortableStateLimits(
        max_fragment_tensor_bytes=4 << 20,
        max_collective_tensor_bytes=16 << 20,
        max_collective_metadata_bytes=64 << 20,
    )


def _grouped_replicated(fqn, shape, group):
    identity = ParameterIdentity(fqn, shape)
    return ShardIdentity(
        identity,
        ParameterLayout.REPLICATED,
        LogicalSlice.full(identity),
        placements=(
            ShardPlacement(
                "checkpoint",
                PlacementKind.REPLICATE,
                0,
                1,
            ),
        ),
        process_group=group,
        local_member=_MEMBER,
    )


def _make_hybrid(matrix, bias, *, deterministic):
    optimizer = GefenMuonHybrid(
        [("layer.weight", matrix)],
        [("layer.bias", bias)],
        lr=1e-3,
        fused=False,
        sharded_mode="distributed",
        backup_optimizer="gefen",
        backup_1d_period_one=True,
        normuon=True,
        deterministic=deterministic,
    )
    group = ProcessGroupIdentity("hybrid_checkpoint", (_MEMBER,))
    matrix_shard = _grouped_replicated(
        "Model.Layer.Weight",
        tuple(matrix.shape),
        group,
    )
    bias_shard = _grouped_replicated(
        "Model.Layer.Bias",
        tuple(bias.shape),
        group,
    )
    manifest = ShardingManifest((matrix_shard, bias_shard))
    codebook_binding = CodebookProcessGroupBinding(
        group,
        _MEMBER,
        None,
        torch.device("cpu"),
    )
    checkpoint_binding = CheckpointProcessGroupBinding(
        group,
        _MEMBER,
        None,
        torch.device("cpu"),
    )
    optimizer.post_sharding(
        (
            ParameterRebinding(matrix, matrix, matrix_shard),
            ParameterRebinding(bias, bias, bias_shard),
        ),
        manifest=manifest,
        codebook_process_group=codebook_binding,
    )
    return optimizer, checkpoint_binding


def _set_grads(matrix, bias, offset):
    matrix.grad = (
        torch.tensor(
            [[0.25, -0.5], [0.75, -1.0]],
            dtype=matrix.dtype,
        )
        + offset
    )
    bias.grad = torch.tensor([0.2, -0.4, 0.6, -0.8], dtype=bias.dtype) - offset


def _equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left.cpu(), right.cpu())
        )
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(_equal(left[key], right[key]) for key in left)
    if type(left) in {list, tuple}:
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def _redigest(document):
    payload = {key: document[key] for key in document if key != "completion"}
    document["completion"]["digest"] = portable_state_digest(payload)


def _assert_same_optimizer_state(left, right):
    assert _equal(left.muon.state_dict(), right.muon.state_dict())
    assert _equal(left.backup.state_dict(), right.backup.state_dict())
    assert left._deterministic is right._deterministic


def _initialized_source():
    matrix = torch.nn.Parameter(torch.tensor([[1.0, -2.0], [3.0, -4.0]]))
    bias = torch.nn.Parameter(torch.tensor([0.5, -1.5, 2.5, -3.5]))
    optimizer, binding = _make_hybrid(matrix, bias, deterministic=True)
    _set_grads(matrix, bias, 0.0)
    optimizer.step()
    return optimizer, matrix, bias, binding


def _target_from_source(matrix, bias, *, deterministic=False):
    target_matrix = torch.nn.Parameter(matrix.detach().clone())
    target_bias = torch.nn.Parameter(bias.detach().clone())
    optimizer, binding = _make_hybrid(
        target_matrix,
        target_bias,
        deterministic=deterministic,
    )
    return optimizer, target_matrix, target_bias, binding


def test_hybrid_portable_round_trip_and_exact_next_step():
    source, source_matrix, source_bias, source_binding = _initialized_source()
    document = source.export_portable_state(
        checkpoint_process_group=source_binding,
        transaction_id="hybrid-export-v1",
        limits=_limits(),
    )
    target, target_matrix, target_bias, target_binding = _target_from_source(
        source_matrix,
        source_bias,
    )

    target.import_portable_state(
        document,
        checkpoint_process_group=target_binding,
        transaction_id="hybrid-import-v1",
        limits=_limits(),
    )

    assert isinstance(target, PortableStateProvider)
    assert document["routing"] == {
        "Model.Layer.Bias": "backup",
        "Model.Layer.Weight": "muon",
    }
    assert document["children"]["muon"]["implementation"] == "gefen.GefenMuon"
    assert document["children"]["backup"]["implementation"] == "gefen.Gefen"
    assert target._deterministic is True
    _assert_same_optimizer_state(source, target)
    support = tuple(
        item
        for item in target.optimizer_contract().capabilities.checkpoints
        if item.transport is CheckpointTransport.CANONICAL_GLOBAL
    )
    assert len(support) == 1
    assert support[0].atomic_load

    _set_grads(source_matrix, source_bias, 0.125)
    _set_grads(target_matrix, target_bias, 0.125)
    source.step()
    target.step()

    assert torch.equal(source_matrix, target_matrix)
    assert torch.equal(source_bias, target_bias)
    _assert_same_optimizer_state(source, target)


def test_hybrid_portable_import_stages_every_child_before_mutation():
    source, source_matrix, source_bias, source_binding = _initialized_source()
    document = source.export_portable_state(
        checkpoint_process_group=source_binding,
        transaction_id="hybrid-atomic-export-v1",
        limits=_limits(),
    )
    target, _target_matrix, _target_bias, target_binding = _target_from_source(
        source_matrix,
        source_bias,
    )
    target.backup.param_groups[0]["eps"] = 1e-7
    muon_before = copy.deepcopy(target.muon.state_dict())
    backup_before = copy.deepcopy(target.backup.state_dict())
    muon_state_object = target.muon.state
    backup_state_object = target.backup.state

    with pytest.raises((RuntimeError, ValueError), match="policy|option|portable"):
        target.import_portable_state(
            document,
            checkpoint_process_group=target_binding,
            transaction_id="hybrid-atomic-import-v1",
            limits=_limits(),
        )

    assert target.muon.state is muon_state_object
    assert target.backup.state is backup_state_object
    assert _equal(target.muon.state_dict(), muon_before)
    assert _equal(target.backup.state_dict(), backup_before)


def test_hybrid_portable_import_rejects_semantically_noncanonical_child():
    source, source_matrix, source_bias, source_binding = _initialized_source()
    document = source.export_portable_state(
        checkpoint_process_group=source_binding,
        transaction_id="hybrid-semantic-export-v1",
        limits=_limits(),
    )
    document["children"]["backup"]["provenance"] = {}
    _redigest(document["children"]["backup"])
    _redigest(document)
    target, _target_matrix, _target_bias, target_binding = _target_from_source(
        source_matrix,
        source_bias,
    )
    muon_before = copy.deepcopy(target.muon.state_dict())
    backup_before = copy.deepcopy(target.backup.state_dict())

    with pytest.raises(RuntimeError, match="provenance"):
        target.import_portable_state(
            document,
            checkpoint_process_group=target_binding,
            transaction_id="hybrid-semantic-import-v1",
            limits=_limits(),
        )

    assert _equal(target.muon.state_dict(), muon_before)
    assert _equal(target.backup.state_dict(), backup_before)


def test_hybrid_portable_composite_uses_collective_tensor_limit():
    source, source_matrix, source_bias, source_binding = _initialized_source()
    limits = PortableStateLimits(
        max_fragment_tensor_bytes=1536,
        max_collective_tensor_bytes=4096,
        max_collective_metadata_bytes=64 << 20,
    )

    document = source.export_portable_state(
        checkpoint_process_group=source_binding,
        transaction_id="hybrid-aggregate-export-v1",
        limits=limits,
    )
    target, _target_matrix, _target_bias, target_binding = _target_from_source(
        source_matrix,
        source_bias,
    )
    target.import_portable_state(
        document,
        checkpoint_process_group=target_binding,
        transaction_id="hybrid-aggregate-import-v1",
        limits=limits,
    )

    _assert_same_optimizer_state(source, target)


def test_hybrid_portable_dcp_round_trip(tmp_path):
    source, source_matrix, source_bias, source_binding = _initialized_source()
    checkpoint = tmp_path / "hybrid-portable"
    save_portable_dcp(
        source,
        checkpoint_process_group=source_binding,
        storage_writer=dcp.FileSystemWriter(checkpoint),
        transaction_id="hybrid-dcp-save-v1",
        limits=_limits(),
    )
    target, target_matrix, target_bias, target_binding = _target_from_source(
        source_matrix,
        source_bias,
    )

    load_portable_dcp(
        target,
        checkpoint_process_group=target_binding,
        storage_reader=dcp.FileSystemReader(checkpoint),
        transaction_id="hybrid-dcp-load-v1",
        limits=_limits(),
    )

    _assert_same_optimizer_state(source, target)
    _set_grads(source_matrix, source_bias, -0.0625)
    _set_grads(target_matrix, target_bias, -0.0625)
    source.step()
    target.step()
    assert torch.equal(source_matrix, target_matrix)
    assert torch.equal(source_bias, target_bias)
    _assert_same_optimizer_state(source, target)


def test_adamw_backed_hybrid_remains_explicitly_nonportable():
    matrix = torch.nn.Parameter(torch.ones(2, 2))
    bias = torch.nn.Parameter(torch.ones(4))
    optimizer = GefenMuonHybrid(
        [("layer.weight", matrix)],
        [("layer.bias", bias)],
        lr=1e-3,
        fused=False,
        sharded_mode="distributed",
        backup_optimizer="adamw",
    )
    assert not optimizer.optimizer_contract().capabilities.canonical_state_io
    assert all(
        item.transport is not CheckpointTransport.CANONICAL_GLOBAL
        for item in optimizer.optimizer_contract().capabilities.checkpoints
    )
    group = ProcessGroupIdentity("hybrid_checkpoint", (_MEMBER,))
    binding = CheckpointProcessGroupBinding(
        group,
        _MEMBER,
        None,
        torch.device("cpu"),
    )
    with pytest.raises(RuntimeError, match="AdamW"):
        optimizer.export_portable_state(
            checkpoint_process_group=binding,
            transaction_id="adamw-hybrid-reject-v1",
            limits=_limits(),
        )


def test_hybrid_canonical_global_checkpoint_guarantees_are_intersected(monkeypatch):
    # A composite canonical-global save/load processes EVERY child, so the
    # Hybrid may advertise a checkpoint guarantee only when both children
    # support it. The children's guarantee sets must intersect, not union.
    from types import SimpleNamespace

    from gefen.contracts import (
        CheckpointSupport,
        ProcessGroupScope,
        TopologyChange,
    )
    from gefen.portable_hybrid import _hybrid_portable_contract_support

    source, _matrix, _bias, _binding = _initialized_source()

    def _support(same_topology, kinds):
        return CheckpointSupport(
            transport=CheckpointTransport.CANONICAL_GLOBAL,
            same_topology=same_topology,
            topology_changing=frozenset({ParameterLayout.REPLICATED}),
            process_group_scope=ProcessGroupScope.DEFAULT_WORLD,
            topology_change_kinds=kinds,
            atomic_load=True,
        )

    muon_support = _support(
        frozenset(
            {ParameterLayout.REPLICATED, ParameterLayout.FLATTENED_ELEMENT_SHARD}
        ),
        frozenset(
            {
                TopologyChange.WORLD_SIZE_OWNER_REDISTRIBUTION,
                TopologyChange.PLACEMENT_RESHARD,
            }
        ),
    )
    backup_support = _support(
        frozenset({ParameterLayout.REPLICATED}),
        frozenset({TopologyChange.WORLD_SIZE_OWNER_REDISTRIBUTION}),
    )

    def _stub(support):
        # Patch at the class level: portable readiness rejects instance-level
        # method shadows. The muon and backup children are distinct classes.
        return lambda _self: SimpleNamespace(
            capabilities=SimpleNamespace(checkpoints=(support,))
        )

    assert type(source.muon) is not type(source.backup)
    monkeypatch.setattr(type(source.muon), "optimizer_contract", _stub(muon_support))
    monkeypatch.setattr(
        type(source.backup), "optimizer_contract", _stub(backup_support)
    )

    same_topology, topology_changing, topology_change_kinds = (
        _hybrid_portable_contract_support(source)
    )

    assert same_topology == frozenset({ParameterLayout.REPLICATED})
    assert topology_changing == frozenset({ParameterLayout.REPLICATED})
    assert topology_change_kinds == frozenset(
        {TopologyChange.WORLD_SIZE_OWNER_REDISTRIBUTION}
    )

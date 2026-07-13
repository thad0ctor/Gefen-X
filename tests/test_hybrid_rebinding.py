"""Focused CPU coverage for atomic Gefen-backed Hybrid rebinding."""

import pytest
import torch

from gefen import GefenMuonHybrid
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

from _state_snapshot import assert_deep_state_snapshot, deep_state_snapshot


_MEMBER = "rank:0"


def _optimizer(*, backup_optimizer="gefen", sharded_mode="distributed"):
    matrix = torch.nn.Parameter(torch.arange(4, dtype=torch.float32).reshape(2, 2))
    bias = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    optimizer = GefenMuonHybrid(
        [("layer.weight", matrix)],
        [("layer.bias", bias)],
        lr=1e-3,
        fused=False,
        sharded_mode=sharded_mode,
        backup_optimizer=backup_optimizer,
    )
    return optimizer, matrix, bias


def _ungrouped_replicated(fqn, shape):
    identity = ParameterIdentity(fqn, shape)
    return ShardIdentity(
        identity,
        ParameterLayout.REPLICATED,
        LogicalSlice.full(identity),
    )


def _grouped_replicated(fqn, shape, group, member):
    identity = ParameterIdentity(fqn, shape)
    coordinate = group.ordered_members.index(member)
    return ShardIdentity(
        identity,
        ParameterLayout.REPLICATED,
        LogicalSlice.full(identity),
        placements=(
            ShardPlacement(
                "checkpoint",
                PlacementKind.REPLICATE,
                coordinate,
                len(group.ordered_members),
            ),
        ),
        process_group=group,
        local_member=member,
    )


def _flat_shards(fqn, shape, group, lengths):
    identity = ParameterIdentity(fqn, shape)
    offset = 0
    shards = []
    for coordinate, (member, length) in enumerate(zip(group.ordered_members, lengths)):
        shards.append(
            ShardIdentity(
                identity,
                ParameterLayout.FLATTENED_ELEMENT_SHARD,
                LogicalSlice(offset, length),
                placements=(
                    ShardPlacement(
                        "checkpoint",
                        PlacementKind.FLAT_SHARD,
                        coordinate,
                        len(group.ordered_members),
                    ),
                ),
                process_group=group,
                local_member=member,
            )
        )
        offset += length
    return tuple(shards)


def _owner_shards(fqn, shape, group, owner):
    identity = ParameterIdentity(fqn, shape)
    shards = []
    for coordinate, member in enumerate(group.ordered_members):
        shards.append(
            ShardIdentity(
                identity,
                ParameterLayout.WHOLE_PARAMETER_OWNER,
                (LogicalSlice.full(identity) if member == owner else LogicalSlice(0, 0)),
                placements=(
                    ShardPlacement(
                        "checkpoint",
                        PlacementKind.WHOLE_PARAMETER_OWNER,
                        coordinate,
                        len(group.ordered_members),
                    ),
                ),
                process_group=group,
                local_member=member,
                owner=owner,
            )
        )
    return tuple(shards)


def _snapshot(optimizer):
    # Each child is snapshotted deeply (container identities plus bitwise
    # clones of per-parameter state, group['params'] entries, and options)
    # so a failed composite transaction that partially rebound a child in
    # place cannot pass on top-level identity alone.
    return {
        "children": tuple((child, deep_state_snapshot(child)) for child in optimizer._subopts),
        "owner": optimizer._state_param_owner,
        "finalized": optimizer._hybrid_post_sharding_finalized,
        "manifest": optimizer._hybrid_sharding_manifest,
        "local": optimizer._hybrid_local_shard_bindings,
        "roles": optimizer._hybrid_fqn_roles,
        "binding": optimizer._hybrid_codebook_process_group,
        "slots": optimizer._hybrid_finalized_slots,
    }


def _assert_snapshot(optimizer, snapshot):
    assert optimizer._state_param_owner is snapshot["owner"]
    assert optimizer._hybrid_post_sharding_finalized is snapshot["finalized"]
    assert optimizer._hybrid_sharding_manifest is snapshot["manifest"]
    assert optimizer._hybrid_local_shard_bindings is snapshot["local"]
    assert optimizer._hybrid_fqn_roles is snapshot["roles"]
    assert optimizer._hybrid_codebook_process_group is snapshot["binding"]
    assert optimizer._hybrid_finalized_slots is snapshot["slots"]
    assert len(optimizer._subopts) == len(snapshot["children"])
    for live, (expected_child, expected_snapshot) in zip(optimizer._subopts, snapshot["children"]):
        assert live is expected_child
        assert_deep_state_snapshot(live, expected_snapshot)


def test_composite_post_sharding_publishes_children_and_rebuilds_routing():
    optimizer, old_matrix, old_bias = _optimizer()
    matrix = torch.nn.Parameter(torch.full((2, 2), 7.0))
    bias = torch.nn.Parameter(torch.full((4,), 11.0))
    group = ProcessGroupIdentity("checkpoint", (_MEMBER,))
    binding = CodebookProcessGroupBinding(
        group,
        _MEMBER,
        None,
        torch.device("cpu"),
    )
    matrix_shard = _grouped_replicated(
        "Model.Layer.Weight",
        (2, 2),
        group,
        _MEMBER,
    )
    bias_shard = _grouped_replicated(
        "Model.Layer.Bias",
        (4,),
        group,
        _MEMBER,
    )
    manifest = ShardingManifest((matrix_shard, bias_shard))

    optimizer.post_sharding(
        (
            ParameterRebinding(old_bias, bias, bias_shard),
            ParameterRebinding(old_matrix, matrix, matrix_shard),
        ),
        manifest=manifest,
        codebook_process_group=binding,
    )

    assert optimizer._canonical_identity_ready()
    assert optimizer._codebook_scope_ready()
    assert optimizer.sharding_manifest() is manifest
    assert optimizer.codebook_process_group_binding() is binding
    assert optimizer.muon.codebook_process_group_binding() is binding
    assert optimizer.backup.codebook_process_group_binding() is binding
    assert optimizer._hybrid_fqn_roles == (
        ("Model.Layer.Bias", "backup"),
        ("Model.Layer.Weight", "muon"),
    )
    assert optimizer.parameter_routing() == optimizer._hybrid_fqn_roles
    assert optimizer.muon.sharding_manifest().shards == (matrix_shard,)
    assert optimizer.backup.sharding_manifest().shards == (bias_shard,)
    assert tuple(shard.parameter.fqn for _, shard in optimizer.shard_bindings()) == (
        "Model.Layer.Bias",
        "Model.Layer.Weight",
    )
    assert optimizer.parameter_identity(matrix) == matrix_shard.parameter
    assert optimizer.shard_identity(bias) == bias_shard
    with pytest.raises(KeyError, match="no finalized"):
        optimizer.shard_identity(old_matrix)

    optimizer.state[bias]["sentinel"] = 13
    optimizer.state[matrix]["sentinel"] = 17
    assert optimizer.backup.state[bias]["sentinel"] == 13
    assert optimizer.muon.state[matrix]["sentinel"] == 17
    assert set(optimizer._state_param_owner) == {id(matrix), id(bias)}


def test_composite_supports_muon_nonowner_and_flat_backup_without_binding():
    optimizer, old_matrix, old_bias = _optimizer()
    group = ProcessGroupIdentity("data_parallel", ("rank:0", "rank:1"))
    matrix_shards = _owner_shards(
        "Model.Layer.Weight",
        (2, 2),
        group,
        owner="rank:1",
    )
    bias_shards = _flat_shards(
        "Model.Layer.Bias",
        (4,),
        group,
        lengths=(2, 2),
    )
    local_matrix = matrix_shards[0]
    local_bias = bias_shards[0]
    bias = torch.nn.Parameter(torch.full((2,), 5.0))
    manifest = ShardingManifest(matrix_shards + bias_shards)

    optimizer.post_sharding(
        (
            ParameterRebinding(old_matrix, None, local_matrix),
            ParameterRebinding(old_bias, bias, local_bias),
        ),
        manifest=manifest,
    )

    assert optimizer._canonical_identity_ready()
    assert not optimizer._codebook_scope_ready()
    assert optimizer.muon.param_groups[0]["params"] == []
    assert optimizer.muon.state == {}
    assert optimizer.backup.param_groups[0]["params"] == [bias]
    assert optimizer.codebook_process_group_binding() is None
    assert optimizer.shard_bindings() == (
        (bias, local_bias),
        (None, local_matrix),
    )
    assert set(optimizer._state_param_owner) == {id(bias)}


def test_second_child_staging_failure_leaves_both_children_unchanged():
    optimizer, old_matrix, old_bias = _optimizer()
    matrix = torch.nn.Parameter(torch.full((2, 2), 3.0))
    invalid_bias = torch.nn.Parameter(torch.full((3,), 4.0))
    matrix_shard = _ungrouped_replicated("Model.Layer.Weight", (2, 2))
    bias_shard = _ungrouped_replicated("Model.Layer.Bias", (4,))
    manifest = ShardingManifest((matrix_shard, bias_shard))
    snapshot = _snapshot(optimizer)
    matrix_before = matrix.detach().clone()
    bias_before = invalid_bias.detach().clone()

    with pytest.raises(ValueError, match="complete parameter storage"):
        optimizer.post_sharding(
            (
                ParameterRebinding(old_matrix, matrix, matrix_shard),
                ParameterRebinding(old_bias, invalid_bias, bias_shard),
            ),
            manifest=manifest,
        )

    _assert_snapshot(optimizer, snapshot)
    assert torch.equal(matrix, matrix_before)
    assert torch.equal(invalid_bias, bias_before)
    assert not optimizer._canonical_identity_ready()


def test_external_child_parameter_replacement_can_be_finalized_atomically():
    optimizer, old_matrix, old_bias = _optimizer()
    matrix = torch.nn.Parameter(torch.full((2, 2), 3.0))
    bias = torch.nn.Parameter(torch.full((4,), 4.0))
    optimizer.muon.param_groups[0]["params"][0] = matrix
    optimizer.backup.param_groups[0]["params"][0] = bias
    matrix_shard = _ungrouped_replicated("Model.Layer.Weight", (2, 2))
    bias_shard = _ungrouped_replicated("Model.Layer.Bias", (4,))

    optimizer.post_sharding(
        (
            ParameterRebinding(old_matrix, matrix, matrix_shard),
            ParameterRebinding(old_bias, bias, bias_shard),
        ),
        manifest=ShardingManifest((matrix_shard, bias_shard)),
    )

    assert optimizer._canonical_identity_ready()
    assert optimizer.muon.param_groups[0]["params"] == [matrix]
    assert optimizer.backup.param_groups[0]["params"] == [bias]
    assert set(optimizer._state_param_owner) == {id(matrix), id(bias)}


def test_cross_child_target_storage_overlap_is_rejected_atomically():
    optimizer, old_matrix, old_bias = _optimizer()
    storage = torch.arange(8, dtype=torch.float32)
    matrix = torch.nn.Parameter(storage[:4].view(2, 2))
    bias = torch.nn.Parameter(storage[2:6])
    assert matrix.untyped_storage().data_ptr() == bias.untyped_storage().data_ptr()
    matrix_shard = _ungrouped_replicated("Model.Layer.Weight", (2, 2))
    bias_shard = _ungrouped_replicated("Model.Layer.Bias", (4,))
    snapshot = _snapshot(optimizer)

    with pytest.raises(ValueError, match="storage ranges must not overlap"):
        optimizer.post_sharding(
            (
                ParameterRebinding(old_matrix, matrix, matrix_shard),
                ParameterRebinding(old_bias, bias, bias_shard),
            ),
            manifest=ShardingManifest((matrix_shard, bias_shard)),
        )

    _assert_snapshot(optimizer, snapshot)


def test_global_source_target_and_fqn_guards_run_before_publication():
    optimizer, old_matrix, old_bias = _optimizer()
    matrix = torch.nn.Parameter(torch.ones(2, 2))
    bias = torch.nn.Parameter(torch.ones(4))
    matrix_shard = _ungrouped_replicated("Model.Shared", (2, 2))
    duplicate_fqn = _ungrouped_replicated("Model.Shared", (4,))
    snapshot = _snapshot(optimizer)

    with pytest.raises(ValueError, match="FQNs must be unique"):
        optimizer.post_sharding(
            (
                ParameterRebinding(old_matrix, matrix, matrix_shard),
                ParameterRebinding(old_bias, bias, duplicate_fqn),
            ),
            manifest=ShardingManifest((matrix_shard,)),
        )
    _assert_snapshot(optimizer, snapshot)

    matrix_shard = _ungrouped_replicated("Model.Layer.Weight", (2, 2))
    bias_shard = _ungrouped_replicated("Model.Layer.Bias", (4,))
    with pytest.raises(ValueError, match="source tensors must be unique"):
        optimizer.post_sharding(
            (
                ParameterRebinding(old_matrix, matrix, matrix_shard),
                ParameterRebinding(old_matrix, bias, bias_shard),
            ),
            manifest=ShardingManifest((matrix_shard, bias_shard)),
        )
    _assert_snapshot(optimizer, snapshot)

    with pytest.raises(ValueError, match="cannot steal"):
        optimizer.post_sharding(
            (
                ParameterRebinding(old_matrix, old_bias, matrix_shard),
                ParameterRebinding(old_bias, bias, bias_shard),
            ),
            manifest=ShardingManifest((matrix_shard, bias_shard)),
        )
    _assert_snapshot(optimizer, snapshot)


def test_shared_binding_must_match_every_child_shard_before_staging():
    optimizer, old_matrix, old_bias = _optimizer()
    shard_group = ProcessGroupIdentity("shards", (_MEMBER,))
    binding_group = ProcessGroupIdentity("checkpoint", (_MEMBER,))
    binding = CodebookProcessGroupBinding(
        binding_group,
        _MEMBER,
        None,
        torch.device("cpu"),
    )
    matrix_shard = _grouped_replicated(
        "Model.Layer.Weight",
        (2, 2),
        shard_group,
        _MEMBER,
    )
    bias_shard = _grouped_replicated(
        "Model.Layer.Bias",
        (4,),
        shard_group,
        _MEMBER,
    )
    snapshot = _snapshot(optimizer)

    with pytest.raises(ValueError, match="shared codebook"):
        optimizer.post_sharding(
            (
                ParameterRebinding(
                    old_matrix,
                    torch.nn.Parameter(torch.ones(2, 2)),
                    matrix_shard,
                ),
                ParameterRebinding(
                    old_bias,
                    torch.nn.Parameter(torch.ones(4)),
                    bias_shard,
                ),
            ),
            manifest=ShardingManifest((matrix_shard, bias_shard)),
            codebook_process_group=binding,
        )

    _assert_snapshot(optimizer, snapshot)


@pytest.mark.parametrize("role", ["muon", "backup"])
def test_one_child_hybrids_support_convenience_rebinding(role):
    if role == "muon":
        old = torch.nn.Parameter(torch.ones(2, 2))
        new = torch.nn.Parameter(torch.full((2, 2), 2.0))
        optimizer = GefenMuonHybrid(
            [("layer.weight", old)],
            [],
            lr=1e-3,
            fused=False,
        )
        identity = ParameterIdentity("Model.Layer.Weight", (2, 2))
    else:
        old = torch.nn.Parameter(torch.ones(4))
        new = torch.nn.Parameter(torch.full((4,), 2.0))
        optimizer = GefenMuonHybrid(
            [],
            [("layer.bias", old)],
            lr=1e-3,
            fused=False,
        )
        identity = ParameterIdentity("Model.Layer.Bias", (4,))

    optimizer.rebind_parameter(old, new, identity=identity)

    assert optimizer._canonical_identity_ready()
    assert optimizer.parameter_identity(new) == identity
    assert optimizer.shard_bindings() == ((new, optimizer.shard_identity(new)),)
    assert set(optimizer._state_param_owner) == {id(new)}


def test_adamw_backup_rejects_before_any_child_or_hybrid_mutation():
    optimizer, old_matrix, old_bias = _optimizer(backup_optimizer="adamw")
    matrix_shard = _ungrouped_replicated("Model.Layer.Weight", (2, 2))
    bias_shard = _ungrouped_replicated("Model.Layer.Bias", (4,))
    snapshot = _snapshot(optimizer)

    with pytest.raises(NotImplementedError, match="AdamW"):
        optimizer.post_sharding(
            (
                ParameterRebinding(
                    old_matrix,
                    torch.nn.Parameter(torch.ones(2, 2)),
                    matrix_shard,
                ),
                ParameterRebinding(
                    old_bias,
                    torch.nn.Parameter(torch.ones(4)),
                    bias_shard,
                ),
            ),
            manifest=ShardingManifest((matrix_shard, bias_shard)),
        )

    _assert_snapshot(optimizer, snapshot)
    assert not optimizer._canonical_identity_ready()


def test_composite_guard_fails_closed_after_owner_metadata_corruption():
    optimizer, old_matrix, old_bias = _optimizer()
    matrix = torch.nn.Parameter(torch.ones(2, 2))
    bias = torch.nn.Parameter(torch.ones(4))
    matrix_shard = _ungrouped_replicated("Model.Layer.Weight", (2, 2))
    bias_shard = _ungrouped_replicated("Model.Layer.Bias", (4,))
    optimizer.post_sharding(
        (
            ParameterRebinding(old_matrix, matrix, matrix_shard),
            ParameterRebinding(old_bias, bias, bias_shard),
        ),
        manifest=ShardingManifest((matrix_shard, bias_shard)),
    )
    assert optimizer._canonical_identity_ready()

    optimizer._state_param_owner = {}

    assert not optimizer._canonical_identity_ready()
    with pytest.raises(RuntimeError, match="changed outside post_sharding"):
        optimizer.sharding_manifest()
    with pytest.raises(RuntimeError, match="changed outside post_sharding"):
        _ = optimizer.state

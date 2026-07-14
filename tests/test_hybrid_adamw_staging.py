"""Focused stable-binding and native-load staging tests for Hybrid AdamW."""

import copy

import pytest
import torch

from gefen import GefenMuonHybrid
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


def _shard(fqn, shape):
    identity = ParameterIdentity(fqn, shape)
    return ShardIdentity(
        identity,
        ParameterLayout.REPLICATED,
        LogicalSlice.full(identity),
    )


def _flat_shards(fqn, shape, group, lengths):
    identity = ParameterIdentity(fqn, shape)
    offset = 0
    shards = []
    for coordinate, (member, length) in enumerate(
        zip(group.ordered_members, lengths)
    ):
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
    return tuple(
        ShardIdentity(
            identity,
            ParameterLayout.WHOLE_PARAMETER_OWNER,
            (
                LogicalSlice.full(identity)
                if member == owner
                else LogicalSlice(0, 0)
            ),
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
        for coordinate, member in enumerate(group.ordered_members)
    )


def _finalized(*, backup_optimizer="adamw", prefix="Model", matrix=None, bias=None):
    old_matrix = torch.nn.Parameter(torch.arange(4, dtype=torch.float32).reshape(2, 2))
    old_bias = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    optimizer = GefenMuonHybrid(
        [("layer.weight", old_matrix)],
        [("layer.bias", old_bias)],
        lr=1e-3,
        fused=False,
        backup_optimizer=backup_optimizer,
    )
    matrix = torch.nn.Parameter(old_matrix.detach().clone() if matrix is None else matrix.detach().clone())
    bias = torch.nn.Parameter(old_bias.detach().clone() if bias is None else bias.detach().clone())
    matrix_shard = _shard(prefix + ".Layer.Weight", (2, 2))
    bias_shard = _shard(prefix + ".Layer.Bias", (4,))
    manifest = ShardingManifest((matrix_shard, bias_shard))
    optimizer.post_sharding(
        (
            ParameterRebinding(old_matrix, matrix, matrix_shard),
            ParameterRebinding(old_bias, bias, bias_shard),
        ),
        manifest=manifest,
    )
    return optimizer, matrix, bias


def _unfinalized(*, backup_optimizer="adamw", matrix=None, bias=None):
    matrix = torch.nn.Parameter(
        torch.arange(4, dtype=torch.float32).reshape(2, 2)
        if matrix is None
        else matrix.detach().clone()
    )
    bias = torch.nn.Parameter(
        torch.arange(4, dtype=torch.float32)
        if bias is None
        else bias.detach().clone()
    )
    return (
        GefenMuonHybrid(
            [("layer.weight", matrix)],
            [("layer.bias", bias)],
            lr=1e-3,
            fused=False,
            backup_optimizer=backup_optimizer,
        ),
        matrix,
        bias,
    )


def _step_backup(optimizer, bias, gradient):
    optimizer.zero_grad()
    bias.grad = gradient.clone()
    optimizer.step()
    optimizer.zero_grad()


def _assert_nested_exact(actual, expected, path="value"):
    assert type(actual) is type(expected), path
    if torch.is_tensor(expected):
        assert actual.dtype == expected.dtype, path
        assert tuple(actual.shape) == tuple(expected.shape), path
        assert torch.equal(actual, expected), path
    elif isinstance(expected, dict):
        assert set(actual) == set(expected), path
        for key in expected:
            _assert_nested_exact(actual[key], expected[key], "{}[{!r}]".format(path, key))
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected), path
        for index, (live, saved) in enumerate(zip(actual, expected)):
            _assert_nested_exact(live, saved, "{}[{}]".format(path, index))
    else:
        assert actual == expected, path


def _hybrid_snapshot(optimizer):
    return {
        "children": tuple(
            (child, deep_state_snapshot(child)) for child in optimizer._subopts
        ),
        "owner": optimizer._state_param_owner,
        "owner_items": tuple(optimizer._state_param_owner.items()),
        "shards": optimizer._hybrid_shard_bindings,
        "shard_items": tuple(optimizer._hybrid_shard_bindings.items()),
        "manifest": optimizer._hybrid_sharding_manifest,
        "local": optimizer._hybrid_local_shard_bindings,
        "roles": optimizer._hybrid_fqn_roles,
        "slots": optimizer._hybrid_finalized_slots,
    }


def _assert_hybrid_snapshot(optimizer, snapshot):
    assert optimizer._state_param_owner is snapshot["owner"]
    assert optimizer._hybrid_shard_bindings is snapshot["shards"]
    assert optimizer._hybrid_sharding_manifest is snapshot["manifest"]
    assert optimizer._hybrid_local_shard_bindings is snapshot["local"]
    assert optimizer._hybrid_fqn_roles is snapshot["roles"]
    assert optimizer._hybrid_finalized_slots is snapshot["slots"]
    assert len(optimizer._state_param_owner) == len(snapshot["owner_items"])
    for (live_key, live_value), (saved_key, saved_value) in zip(
        optimizer._state_param_owner.items(), snapshot["owner_items"]
    ):
        assert live_key == saved_key
        assert live_value[0] is saved_value[0]
        assert live_value[1] is saved_value[1]
    for (live_parameter, live_shard), (saved_parameter, saved_shard) in zip(
        optimizer._hybrid_shard_bindings.items(), snapshot["shard_items"]
    ):
        assert live_parameter is saved_parameter
        assert live_shard == saved_shard
    for live_child, (saved_child, child_snapshot) in zip(
        optimizer._subopts, snapshot["children"]
    ):
        assert live_child is saved_child
        assert_deep_state_snapshot(live_child, child_snapshot)


def test_finalized_adamw_native_checkpoint_continues_exactly():
    source, source_matrix, source_bias = _finalized()
    _step_backup(source, source_bias, torch.tensor([0.5, -0.25, 0.75, -1.0]))
    checkpoint = copy.deepcopy(source.state_dict())
    assert set(checkpoint) == {
        "muon",
        "backup",
        "backup_optimizer",
        "finalized_binding",
    }
    assert checkpoint["finalized_binding"]["format_version"] == 1

    target, target_matrix, target_bias = _finalized(
        matrix=source_matrix,
        bias=source_bias,
    )
    target.load_state_dict(checkpoint)
    _assert_nested_exact(target.state_dict(), source.state_dict())

    next_gradient = torch.tensor([-0.5, 0.125, 0.25, 0.875])
    _step_backup(source, source_bias, next_gradient)
    _step_backup(target, target_bias, next_gradient)
    assert torch.equal(target_matrix, source_matrix)
    assert torch.equal(target_bias, source_bias)
    _assert_nested_exact(target.state_dict(), source.state_dict())


def test_adamw_native_checkpoint_accepts_scheduler_initial_lr():
    source, source_matrix, source_bias = _finalized()
    torch.optim.lr_scheduler.LambdaLR(source, lambda _step: 1.0)
    assert source.backup.param_groups[0]["initial_lr"] == 1e-3
    _step_backup(source, source_bias, torch.tensor([0.5, -0.25, 0.75, -1.0]))
    checkpoint = copy.deepcopy(source.state_dict())

    target, target_matrix, target_bias = _finalized(
        matrix=source_matrix,
        bias=source_bias,
    )
    torch.optim.lr_scheduler.LambdaLR(target, lambda _step: 1.0)
    target.load_state_dict(checkpoint)
    _assert_nested_exact(target.state_dict(), source.state_dict())

    next_gradient = torch.tensor([-0.5, 0.125, 0.25, 0.875])
    _step_backup(source, source_bias, next_gradient)
    _step_backup(target, target_bias, next_gradient)
    assert torch.equal(target_matrix, source_matrix)
    assert torch.equal(target_bias, source_bias)


def test_adamw_backup_supports_flat_shards_while_muon_nonowner_is_pruned():
    old_matrix = torch.nn.Parameter(torch.ones(2, 2))
    old_bias = torch.nn.Parameter(torch.ones(4))
    optimizer = GefenMuonHybrid(
        [("layer.weight", old_matrix)],
        [("layer.bias", old_bias)],
        lr=1e-3,
        fused=False,
        backup_optimizer="adamw",
    )
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
    bias = torch.nn.Parameter(torch.full((2,), 3.0))
    optimizer.post_sharding(
        (
            ParameterRebinding(old_matrix, None, local_matrix),
            ParameterRebinding(old_bias, bias, local_bias),
        ),
        manifest=ShardingManifest(matrix_shards + bias_shards),
    )

    assert optimizer._canonical_identity_ready()
    assert optimizer.muon.param_groups[0]["params"] == []
    assert optimizer.backup.param_groups[0]["params"] == [bias]
    assert optimizer.shard_identity(bias) == local_bias
    assert optimizer.shard_bindings() == ((bias, local_bias), (None, local_matrix))
    assert set(optimizer._hybrid_shard_bindings) == {bias}


@pytest.mark.parametrize(
    "damage",
    ["extra_key", "wrong_shape", "negative_variance", "fractional_step"],
)
def test_invalid_adamw_state_rejects_before_any_child_publication(damage):
    source, _matrix, source_bias = _finalized()
    _step_backup(source, source_bias, torch.tensor([0.5, -0.25, 0.75, -1.0]))
    checkpoint = copy.deepcopy(source.state_dict())
    (parameter_id,) = checkpoint["backup"]["state"]
    state = checkpoint["backup"]["state"][parameter_id]
    if damage == "extra_key":
        state["foreign"] = torch.ones(1)
    elif damage == "wrong_shape":
        state["exp_avg"] = torch.ones(3)
    elif damage == "negative_variance":
        state["exp_avg_sq"][0] = -1.0
    else:
        state["step"] = torch.tensor(1.5)

    target, _target_matrix, target_bias = _finalized()
    _step_backup(target, target_bias, torch.tensor([-0.5, 0.25, -0.75, 1.0]))
    snapshot = _hybrid_snapshot(target)
    with pytest.raises((TypeError, ValueError), match="AdamW"):
        target.load_state_dict(checkpoint)
    _assert_hybrid_snapshot(target, snapshot)


def test_finalized_binding_guard_rejects_changed_identity_before_child_hooks():
    source, _matrix, source_bias = _finalized(prefix="Source")
    _step_backup(source, source_bias, torch.ones(4))
    checkpoint = copy.deepcopy(source.state_dict())
    target, _target_matrix, _target_bias = _finalized(prefix="Target")
    snapshot = _hybrid_snapshot(target)
    calls = []
    target.muon.register_load_state_dict_pre_hook(lambda *_args: calls.append("muon"))
    target.backup.register_load_state_dict_pre_hook(lambda *_args: calls.append("backup"))

    with pytest.raises(ValueError, match="binding or routing differs"):
        target.load_state_dict(checkpoint)

    assert calls == []
    _assert_hybrid_snapshot(target, snapshot)


@pytest.mark.parametrize("backup_optimizer", ["gefen", "adamw"])
def test_all_child_core_is_published_before_any_load_post_hook(backup_optimizer):
    source, _source_matrix, _source_bias = _finalized(
        backup_optimizer=backup_optimizer
    )
    source.muon.param_groups[0]["lr"] = 0.006
    source.backup.param_groups[0]["lr"] = 0.007
    checkpoint = copy.deepcopy(source.state_dict())
    target, _target_matrix, _target_bias = _finalized(
        backup_optimizer=backup_optimizer
    )
    events = []

    def muon_post(_child):
        assert target.muon.param_groups[0]["lr"] == 0.006
        assert target.backup.param_groups[0]["lr"] == 0.007
        events.append("muon")

    def backup_post(_child):
        assert target.muon.param_groups[0]["lr"] == 0.006
        assert target.backup.param_groups[0]["lr"] == 0.007
        events.append("backup")

    target.muon.register_load_state_dict_post_hook(muon_post)
    target.backup.register_load_state_dict_post_hook(backup_post)
    target.register_load_state_dict_post_hook(lambda _optimizer: events.append("hybrid"))
    target.load_state_dict(checkpoint)
    assert events == ["muon", "backup", "hybrid"]


@pytest.mark.parametrize("backup_optimizer", ["gefen", "adamw"])
def test_throwing_first_child_post_hook_observes_fully_committed_core(backup_optimizer):
    source, _source_matrix, _source_bias = _finalized(
        backup_optimizer=backup_optimizer
    )
    source.muon.param_groups[0]["lr"] = 0.004
    source.backup.param_groups[0]["lr"] = 0.008
    checkpoint = copy.deepcopy(source.state_dict())
    target, _target_matrix, _target_bias = _finalized(
        backup_optimizer=backup_optimizer
    )

    def stop_after_observation(_child):
        assert target.muon.param_groups[0]["lr"] == 0.004
        assert target.backup.param_groups[0]["lr"] == 0.008
        raise RuntimeError("post-hook sentinel")

    target.muon.register_load_state_dict_post_hook(stop_after_observation)
    with pytest.raises(RuntimeError, match="post-hook sentinel"):
        target.load_state_dict(checkpoint)
    assert target.muon.param_groups[0]["lr"] == 0.004
    assert target.backup.param_groups[0]["lr"] == 0.008


def test_unfinalized_legacy_adamw_checkpoint_schema_and_continuation_are_preserved():
    source, source_matrix, source_bias = _unfinalized()
    _step_backup(source, source_bias, torch.tensor([0.5, -0.25, 0.75, -1.0]))
    checkpoint = copy.deepcopy(source.state_dict())
    assert set(checkpoint) == {"muon", "backup", "backup_optimizer"}

    target, target_matrix, target_bias = _unfinalized(
        matrix=source_matrix,
        bias=source_bias,
    )
    target.load_state_dict(checkpoint)
    next_gradient = torch.tensor([-0.5, 0.125, 0.25, 0.875])
    _step_backup(source, source_bias, next_gradient)
    _step_backup(target, target_bias, next_gradient)
    assert torch.equal(target_matrix, source_matrix)
    assert torch.equal(target_bias, source_bias)
    _assert_nested_exact(target.state_dict(), source.state_dict())


def test_finalized_and_legacy_native_checkpoint_schemas_do_not_cross_load():
    finalized, _matrix, _bias = _finalized()
    legacy, _legacy_matrix, _legacy_bias = _unfinalized()
    finalized_checkpoint = copy.deepcopy(finalized.state_dict())
    legacy_checkpoint = copy.deepcopy(legacy.state_dict())

    with pytest.raises(ValueError, match="requires a versioned native binding guard"):
        finalized.load_state_dict(legacy_checkpoint)
    with pytest.raises(ValueError, match="cannot load finalized native shard state"):
        legacy.load_state_dict(finalized_checkpoint)

import copy
from dataclasses import FrozenInstanceError
import io

import pytest
import torch

from gefen import (
    CodebookProcessGroupBinding,
    Gefen,
    GefenMuon,
    LogicalSlice,
    ParameterIdentity,
    ParameterLayout,
    ParameterRebinding,
    ParameterStateRole,
    PlacementKind,
    ProcessGroupIdentity,
    ProcessGroupScope,
    ShardIdentity,
    ShardPlacement,
    ShardingManifest,
    StateExtent,
)
import gefen.gefen as gefen_module

from _state_snapshot import assert_deep_state_snapshot, deep_state_snapshot


def _replicated_shard(parameter, group, member):
    coordinate = group.ordered_members.index(member)
    return ShardIdentity(
        parameter,
        ParameterLayout.REPLICATED,
        LogicalSlice.full(parameter),
        process_group=group,
        local_member=member,
        placements=(
            ShardPlacement(
                "data_parallel",
                PlacementKind.REPLICATE,
                coordinate,
                len(group.ordered_members),
            ),
        ),
    )


def _flat_shard(parameter, group, member):
    coordinate = group.ordered_members.index(member)
    return ShardIdentity(
        parameter,
        ParameterLayout.FLATTENED_ELEMENT_SHARD,
        LogicalSlice.full(parameter),
        process_group=group,
        local_member=member,
        placements=(
            ShardPlacement(
                "data_parallel",
                PlacementKind.FLAT_SHARD,
                coordinate,
                len(group.ordered_members),
            ),
        ),
    )


def _whole_owner_shard(parameter, group, member, owner):
    coordinate = group.ordered_members.index(member)
    return ShardIdentity(
        parameter,
        ParameterLayout.WHOLE_PARAMETER_OWNER,
        LogicalSlice.full(parameter) if member == owner else LogicalSlice(0, 0),
        process_group=group,
        local_member=member,
        owner=owner,
        placements=(
            ShardPlacement(
                "data_parallel",
                PlacementKind.WHOLE_PARAMETER_OWNER,
                coordinate,
                len(group.ordered_members),
            ),
        ),
    )


def _single_member_binding(group, member="rank:0"):
    return CodebookProcessGroupBinding(group, member, None, torch.device("cpu"))


def _finalize_replicated(optimizer, parameter, *, fqn="Layer.Weight"):
    group = ProcessGroupIdentity("data_parallel", ("rank:0",))
    identity = ParameterIdentity(fqn, tuple(parameter.shape))
    shard = _replicated_shard(identity, group, "rank:0")
    binding = _single_member_binding(group)
    optimizer.post_sharding(
        (ParameterRebinding(parameter, parameter, shard),),
        manifest=ShardingManifest((shard,)),
        codebook_process_group=binding,
    )
    return binding


def _snapshot(optimizer):
    return {
        "dict": optimizer.__dict__.copy(),
        "groups": optimizer.param_groups,
        "state": optimizer.state,
        "codebook": optimizer._gefen_codebook,
        "binding": optimizer._gefen_codebook_process_group,
        "validated": optimizer._gefen_codebook_scope_validated,
        # Bitwise clones of every reachable state tensor, param-group option, and
        # mutable registry so a rejected transaction that mutates contents in
        # place (without swapping the top-level container) is still caught.
        "deep": deep_state_snapshot(optimizer),
    }


def _assert_snapshot_identity(optimizer, snapshot):
    assert optimizer.param_groups is snapshot["groups"]
    assert optimizer.state is snapshot["state"]
    assert optimizer._gefen_codebook is snapshot["codebook"]
    assert optimizer._gefen_codebook_process_group is snapshot["binding"]
    assert optimizer._gefen_codebook_scope_validated is snapshot["validated"]
    assert optimizer.__dict__.keys() == snapshot["dict"].keys()
    for key, value in snapshot["dict"].items():
        assert optimizer.__dict__[key] is value
    assert_deep_state_snapshot(optimizer, snapshot["deep"])


def _replace_native_guard(checkpoint, guard):
    checkpoint["gefen_native_local_shards"] = copy.deepcopy(guard)
    for group in checkpoint["param_groups"]:
        group["_gefen_checkpoint_metadata"]["native_local_shards"] = copy.deepcopy(guard)


def test_deep_snapshot_detects_per_device_cache_membership_clear():
    # A per-device cache clear preserves the dict attribute identity, and the
    # retained cache tensors still match their tensor-value clones, so only a
    # membership-aware snapshot catches the removal.
    parameter = torch.nn.Parameter(torch.randn(4, 4))
    optimizer = Gefen([("Layer.Weight", parameter)], fused=False)
    device = torch.device("cpu")
    optimizer._gefen_codebook_by_device[device] = torch.randn(8)

    snapshot = deep_state_snapshot(optimizer)
    assert_deep_state_snapshot(optimizer, snapshot)  # unchanged membership passes

    optimizer._gefen_codebook_by_device.clear()
    with pytest.raises(AssertionError):
        assert_deep_state_snapshot(optimizer, snapshot)


def test_codebook_process_group_binding_is_public_frozen_and_ordered():
    group = ProcessGroupIdentity("replica", ("worker:b", "worker:a"))
    binding = CodebookProcessGroupBinding(group, "worker:b", object(), torch.device("cpu"))

    assert binding.identity is group
    assert binding.local_member == "worker:b"
    assert binding.sort_key == ("replica", ("worker:b", "worker:a"))
    with pytest.raises(FrozenInstanceError):
        binding.local_member = "worker:a"
    with pytest.raises(TypeError, match="ProcessGroupIdentity"):
        CodebookProcessGroupBinding(object(), "worker:b", None, "cpu")
    with pytest.raises(ValueError, match="local_member"):
        CodebookProcessGroupBinding(group, "missing", None, "cpu")
    with pytest.raises(ValueError, match="materialized"):
        CodebookProcessGroupBinding(group, "worker:b", None, "meta")


def test_scoped_one_member_initialization_matches_unscoped_first_step():
    initial = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
    scoped_param = torch.nn.Parameter(initial.clone())
    reference_param = torch.nn.Parameter(initial.clone())
    scoped = Gefen([("Layer.Weight", scoped_param)], fused=False, factored_v_2d=False)
    reference = Gefen([("Layer.Weight", reference_param)], fused=False, factored_v_2d=False)
    binding = _finalize_replicated(scoped, scoped_param)
    gradient = torch.linspace(-2, 3, initial.numel()).reshape_as(initial)
    scoped_param.grad = gradient.clone()
    reference_param.grad = gradient.clone()

    assert scoped.initialize_codebook()
    assert not scoped.initialize_codebook()
    scoped.step()
    reference.step()

    assert scoped.codebook_process_group_binding() is binding
    assert scoped.optimizer_contract().capabilities.explicit_process_group_codebook_scope
    assert torch.equal(scoped._gefen_codebook, reference._gefen_codebook)
    assert torch.equal(scoped_param, reference_param)
    for key in scoped.state[scoped_param]:
        left = scoped.state[scoped_param][key]
        right = reference.state[reference_param][key]
        if torch.is_tensor(left):
            assert torch.equal(left, right)
        else:
            assert left == right
    frozen_codebook = scoped._gefen_codebook
    frozen_indices = scoped.state[scoped_param]["m_codebook"].detach().clone()
    scoped_param.grad = None
    assert not scoped.refresh_codebook()
    assert scoped._gefen_codebook is frozen_codebook
    assert torch.equal(scoped.state[scoped_param]["m_codebook"], frozen_indices)


def test_scope_binding_validation_is_part_of_atomic_post_sharding():
    parameter = torch.nn.Parameter(torch.ones(8))
    optimizer = Gefen([("p", parameter)], fused=False, factored_v_2d=False)
    group = ProcessGroupIdentity("dp", ("rank:0",))
    wrong_group = ProcessGroupIdentity("other", ("rank:0",))
    identity = ParameterIdentity("P", (8,))
    shard = _replicated_shard(identity, group, "rank:0")
    manifest = ShardingManifest((shard,))
    snapshot = _snapshot(optimizer)

    with pytest.raises(ValueError, match="process-group identity"):
        optimizer.post_sharding(
            (ParameterRebinding(parameter, parameter, shard),),
            manifest=manifest,
            codebook_process_group=_single_member_binding(wrong_group),
        )

    _assert_snapshot_identity(optimizer, snapshot)
    assert not optimizer._gefen_post_sharding_finalized


def test_scope_binding_rejects_implicit_default_world_for_one_member():
    parameter = torch.nn.Parameter(torch.ones(4))
    optimizer = Gefen([parameter], fused=False, factored_v_2d=False)
    group = ProcessGroupIdentity("dp", ("rank:0",))
    identity = ParameterIdentity("P", (4,))
    shard = _replicated_shard(identity, group, "rank:0")

    with pytest.raises(ValueError, match="process_group=None"):
        optimizer.post_sharding(
            (ParameterRebinding(parameter, parameter, shard),),
            manifest=ShardingManifest((shard,)),
            codebook_process_group=CodebookProcessGroupBinding(group, "rank:0", object(), "cpu"),
        )

    assert not optimizer._gefen_post_sharding_finalized


def test_initialization_failure_does_not_publish_periods_or_codebook(monkeypatch):
    parameter = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    optimizer = Gefen([("p", parameter)], fused=False, factored_v_2d=False)
    _finalize_replicated(optimizer, parameter, fqn="P")
    parameter.grad = torch.arange(1, 9, dtype=torch.float32)
    before_state = copy.deepcopy(optimizer.state[parameter])
    before_cache = optimizer._gefen_codebook_by_device

    def fail_exact_dp(*args, **kwargs):
        raise RuntimeError("injected exact-DP failure")

    monkeypatch.setattr(gefen_module.quantization_module, "exact_dp", fail_exact_dp)
    with pytest.raises(RuntimeError, match="injected exact-DP failure"):
        optimizer.initialize_codebook()

    assert optimizer.state[parameter] == before_state
    assert optimizer._gefen_codebook is None
    assert optimizer._gefen_codebook_by_device is before_cache
    assert not optimizer._gefen_codebook_scope_validated
    assert optimizer._gefen_global_step == 0


def test_scoped_native_checkpoint_uses_primitive_scope_and_requires_exact_binding():
    source_param = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    source = Gefen([("p", source_param)], fused=False, factored_v_2d=False)
    _finalize_replicated(source, source_param, fqn="P")
    source_param.grad = torch.arange(1, 9, dtype=torch.float32)
    source.step()
    checkpoint = source.state_dict()

    scope = checkpoint["gefen_codebook_scope"]
    assert scope == {
        "format_version": 1,
        "semantic_name": "data_parallel",
        "ordered_members": ["rank:0"],
        "refresh_every": 0,
    }
    assert all(group["_gefen_checkpoint_metadata"]["codebook_scope"] == scope for group in checkpoint["param_groups"])
    assert all(group["_gefen_checkpoint_metadata"]["format_version"] == 4 for group in checkpoint["param_groups"])
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=True)
    assert loaded["gefen_codebook_scope"] == scope

    target_param = torch.nn.Parameter(source_param.detach().clone())
    target = Gefen([("p", target_param)], fused=False, factored_v_2d=False)
    _finalize_replicated(target, target_param, fqn="P")
    live_binding = target.codebook_process_group_binding()
    target.load_state_dict(copy.deepcopy(checkpoint))
    assert target.codebook_process_group_binding() is live_binding
    assert not target._gefen_codebook_scope_validated
    assert torch.equal(target._gefen_codebook, source._gefen_codebook)
    continuation_grad = torch.linspace(-1, 1, 8)
    source_param.grad = continuation_grad.clone()
    target_param.grad = continuation_grad.clone()
    source.step()
    target.step()
    assert torch.equal(target_param, source_param)
    assert torch.equal(target._gefen_codebook, source._gefen_codebook)

    conflicting_param = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    conflicting_target = Gefen([("p", conflicting_param)], fused=False, factored_v_2d=False)
    _finalize_replicated(conflicting_target, conflicting_param, fqn="P")
    conflicting = copy.deepcopy(checkpoint)
    conflicting["gefen_codebook_scope"] = dict(conflicting["gefen_codebook_scope"])
    conflicting["gefen_codebook_scope"]["semantic_name"] = "other"
    conflicting_before = _snapshot(conflicting_target)
    with pytest.raises(ValueError, match="scopes disagree"):
        conflicting_target.load_state_dict(conflicting)
    _assert_snapshot_identity(conflicting_target, conflicting_before)

    unsafe_version = copy.deepcopy(checkpoint)
    unsafe_version["param_groups"][0]["_gefen_checkpoint_metadata"]["format_version"] = 1
    with pytest.raises(ValueError, match="format_version 4"):
        conflicting_target.load_state_dict(unsafe_version)
    _assert_snapshot_identity(conflicting_target, conflicting_before)

    unbound_param = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    unbound = Gefen([("p", unbound_param)], fused=False, factored_v_2d=False)
    before = _snapshot(unbound)
    with pytest.raises(ValueError, match="does not match"):
        unbound.load_state_dict(checkpoint)
    _assert_snapshot_identity(unbound, before)

    schedule_param = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    schedule_target = Gefen(
        [("p", schedule_param)],
        fused=False,
        factored_v_2d=False,
        codebook_refresh_every=1,
    )
    _finalize_replicated(schedule_target, schedule_param, fqn="P")
    schedule_before = _snapshot(schedule_target)
    with pytest.raises(ValueError, match="does not match"):
        schedule_target.load_state_dict(checkpoint)
    _assert_snapshot_identity(schedule_target, schedule_before)


@pytest.mark.parametrize(
    "second_layout",
    [
        ParameterLayout.FLATTENED_ELEMENT_SHARD,
        ParameterLayout.REPLICATED,
    ],
)
def test_native_flat_checkpoint_guard_rejects_reordered_equal_shape_slots_atomically(
    second_layout,
):
    group = ProcessGroupIdentity("data_parallel", ("rank:0",))
    first_identity = ParameterIdentity("Model.First", (4,))
    second_identity = ParameterIdentity("Model.Second", (4,))
    first_shard = _flat_shard(first_identity, group, "rank:0")
    second_shard = (
        _flat_shard(second_identity, group, "rank:0")
        if second_layout is ParameterLayout.FLATTENED_ELEMENT_SHARD
        else _replicated_shard(second_identity, group, "rank:0")
    )
    manifest = ShardingManifest((first_shard, second_shard))

    source_first = torch.nn.Parameter(torch.zeros(4))
    source_second = torch.nn.Parameter(torch.zeros(4))
    source = Gefen(
        [("first", source_first), ("second", source_second)],
        fused=False,
        factored_v_2d=False,
    )
    source.post_sharding(
        (
            ParameterRebinding(source_first, source_first, first_shard),
            ParameterRebinding(source_second, source_second, second_shard),
        ),
        manifest=manifest,
        codebook_process_group=_single_member_binding(group),
    )
    source._resolve_automatic_period = lambda *args: 4
    source_first.grad = torch.tensor([1.0, 2.0, 3.0, 4.0])
    source_second.grad = torch.tensor([9.0, -1.0, -2.0, -3.0])
    source.step()
    checkpoint = source.state_dict()
    source_guard = checkpoint["gefen_native_local_shards"]
    source_slots = source_guard["logical_slots"]
    assert source_guard["format_version"] == 2
    assert (
        checkpoint["param_groups"][0]["_gefen_checkpoint_metadata"]["native_local_shards"]
        is source_guard
    )
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    assert torch.load(buffer, weights_only=True)["gefen_native_local_shards"] == checkpoint["gefen_native_local_shards"]
    malformed_version = copy.deepcopy(checkpoint)
    malformed_version["gefen_native_local_shards"]["format_version"] = True
    for group_record in malformed_version["param_groups"]:
        group_record["_gefen_checkpoint_metadata"]["native_local_shards"]["format_version"] = True
    source_before = _snapshot(source)
    with pytest.raises(ValueError, match="format_version"):
        source.load_state_dict(malformed_version)
    _assert_snapshot_identity(source, source_before)
    assert [
        (
            record["group_index"],
            record["original_slot_index"],
            record["compatibility_name"],
            record["shard"]["parameter"]["fqn"],
        )
        for record in source_slots
    ] == [
        (0, 0, "first", "Model.First"),
        (0, 1, "second", "Model.Second"),
    ]
    if second_layout is ParameterLayout.REPLICATED:
        assert source_slots[1]["shard"]["layout"] == ParameterLayout.REPLICATED.value

    legacy_checkpoint = copy.deepcopy(checkpoint)
    legacy_guard = source._serialized_native_local_shards_v1()
    legacy_checkpoint["gefen_native_local_shards"] = legacy_guard
    for group_record in legacy_checkpoint["param_groups"]:
        group_record["_gefen_checkpoint_metadata"]["native_local_shards"] = copy.deepcopy(legacy_guard)
    legacy_first = torch.nn.Parameter(torch.zeros(4))
    legacy_second = torch.nn.Parameter(torch.zeros(4))
    legacy_target = Gefen(
        [("target_first", legacy_first), ("target_second", legacy_second)],
        fused=False,
        factored_v_2d=False,
    )
    legacy_target.post_sharding(
        (
            ParameterRebinding(legacy_first, legacy_first, first_shard),
            ParameterRebinding(legacy_second, legacy_second, second_shard),
        ),
        manifest=manifest,
        codebook_process_group=_single_member_binding(group),
    )
    legacy_target.load_state_dict(copy.deepcopy(legacy_checkpoint))
    assert torch.equal(legacy_target._gefen_codebook, source._gefen_codebook)
    assert legacy_target.param_groups[0]["param_names"] == [
        "target_first",
        "target_second",
    ]
    assert legacy_target.state[legacy_first]["name"] == "target_first"
    assert legacy_target.state[legacy_second]["name"] == "target_second"

    mirror_only_checkpoint = copy.deepcopy(legacy_checkpoint)
    mirror_only_checkpoint.pop("gefen_native_local_shards")
    legacy_target.load_state_dict(mirror_only_checkpoint)
    assert torch.equal(legacy_target._gefen_codebook, source._gefen_codebook)

    top_only_checkpoint = copy.deepcopy(legacy_checkpoint)
    for group_record in top_only_checkpoint["param_groups"]:
        group_record.pop("_gefen_checkpoint_metadata")
    legacy_target.load_state_dict(top_only_checkpoint)
    assert torch.equal(legacy_target._gefen_codebook, source._gefen_codebook)

    mixed_version_checkpoint = copy.deepcopy(checkpoint)
    for group_record in mixed_version_checkpoint["param_groups"]:
        group_record["_gefen_checkpoint_metadata"]["native_local_shards"] = (
            copy.deepcopy(legacy_guard)
        )
    source_before = _snapshot(source)
    with pytest.raises(ValueError, match="local shards disagree"):
        source.load_state_dict(mixed_version_checkpoint)
    _assert_snapshot_identity(source, source_before)

    target_second = torch.nn.Parameter(torch.zeros(4))
    target_first = torch.nn.Parameter(torch.zeros(4))
    target = Gefen(
        [("second", target_second), ("first", target_first)],
        fused=False,
        factored_v_2d=False,
    )
    target.post_sharding(
        (
            ParameterRebinding(target_second, target_second, second_shard),
            ParameterRebinding(target_first, target_first, first_shard),
        ),
        manifest=manifest,
        codebook_process_group=_single_member_binding(group),
    )
    target_before = _snapshot(target)

    with pytest.raises(ValueError, match="local-shard identity"):
        target.load_state_dict(copy.deepcopy(checkpoint))

    _assert_snapshot_identity(target, target_before)


def test_native_v2_guard_retains_pruned_logical_positions_and_rejects_corruption_atomically(
    monkeypatch,
):
    monkeypatch.setattr(
        Gefen,
        "_validate_codebook_runtime_binding",
        lambda self, binding: None,
    )
    group = ProcessGroupIdentity("data_parallel", ("rank:0", "rank:1"))

    def build():
        owned = torch.nn.Parameter(torch.ones(2, 2))
        remote = torch.nn.Parameter(torch.full((2, 2), 2.0))
        replicated = torch.nn.Parameter(torch.full((2, 2), 3.0))
        optimizer = GefenMuon(
            [
                {"params": [("owned", owned), ("remote", remote)]},
                {"params": [("replicated", replicated)]},
            ],
            fused=False,
        )
        owned_identity = ParameterIdentity("Model.Owned", (2, 2))
        remote_identity = ParameterIdentity("Model.Remote", (2, 2))
        replicated_identity = ParameterIdentity("Model.Replicated", (2, 2))
        owned_records = tuple(
            _whole_owner_shard(owned_identity, group, member, "rank:0")
            for member in group.ordered_members
        )
        remote_records = tuple(
            _whole_owner_shard(remote_identity, group, member, "rank:1")
            for member in group.ordered_members
        )
        replicated_records = tuple(
            _replicated_shard(replicated_identity, group, member)
            for member in group.ordered_members
        )
        local_owned = owned_records[0]
        local_remote = remote_records[0]
        local_replicated = replicated_records[0]
        optimizer.post_sharding(
            (
                ParameterRebinding(remote, None, local_remote),
                ParameterRebinding(replicated, replicated, local_replicated),
                ParameterRebinding(owned, owned, local_owned),
            ),
            manifest=ShardingManifest(
                owned_records + remote_records + replicated_records
            ),
            codebook_process_group=CodebookProcessGroupBinding(
                group,
                "rank:0",
                object(),
                torch.device("cpu"),
            ),
        )
        return optimizer

    source = build()
    checkpoint = source.state_dict()
    guard = checkpoint["gefen_native_local_shards"]
    assert guard["format_version"] == 2
    assert [
        (
            record["group_index"],
            record["original_slot_index"],
            record["compatibility_name"],
            record["shard"]["parameter"]["fqn"],
        )
        for record in guard["logical_slots"]
    ] == [
        (0, 0, "owned", "Model.Owned"),
        (0, 1, "remote", "Model.Remote"),
        (1, 0, "replicated", "Model.Replicated"),
    ]
    assert source.param_groups[0]["param_names"] == ["owned"]

    matching = build()
    matching.load_state_dict(copy.deepcopy(checkpoint))
    assert matching._gefen_logical_slots == source._gefen_logical_slots

    malformed_second_mirror = copy.deepcopy(checkpoint)
    second_metadata = copy.deepcopy(
        malformed_second_mirror["param_groups"][1][
            "_gefen_checkpoint_metadata"
        ]
    )
    second_metadata["native_local_shards"]["logical_slots"] = tuple(
        second_metadata["native_local_shards"]["logical_slots"]
    )
    malformed_second_mirror["param_groups"][1][
        "_gefen_checkpoint_metadata"
    ] = second_metadata
    malformed_target = build()
    malformed_before = _snapshot(malformed_target)
    with pytest.raises(ValueError, match="logical_slots must be a nonempty list"):
        malformed_target.load_state_dict(malformed_second_mirror)
    _assert_snapshot_identity(malformed_target, malformed_before)

    corruptions = []

    class GuardDict(dict):
        pass

    corruptions.append((GuardDict(copy.deepcopy(guard)), "invalid schema"))

    invalid_schema = copy.deepcopy(guard)
    invalid_schema["logical_slots"][0]["extra"] = None
    corruptions.append((invalid_schema, "invalid schema"))

    bool_position = copy.deepcopy(guard)
    bool_position["logical_slots"][0]["group_index"] = False
    corruptions.append((bool_position, "logical slot is invalid"))

    skipped_position = copy.deepcopy(guard)
    skipped_position["logical_slots"][2]["group_index"] = 2
    corruptions.append((skipped_position, "positions must be contiguous"))

    uppercase_name = copy.deepcopy(guard)
    uppercase_name["logical_slots"][1]["compatibility_name"] = "Remote"
    corruptions.append((uppercase_name, "logical slot is invalid"))

    duplicate_fqn = copy.deepcopy(guard)
    duplicate_fqn["logical_slots"][1]["shard"]["parameter"] = copy.deepcopy(
        duplicate_fqn["logical_slots"][0]["shard"]["parameter"]
    )
    corruptions.append((duplicate_fqn, "FQNs must be unique"))

    missing_group = copy.deepcopy(guard)
    missing_group["logical_slots"][0]["shard"]["process_group"] = None
    corruptions.append((missing_group, "requires a process-group identity"))

    changed_layout = copy.deepcopy(guard)
    changed_layout["logical_slots"][0]["shard"]["layout"] = (
        ParameterLayout.REPLICATED.value
    )
    changed_layout["logical_slots"][0]["shard"]["owner"] = None
    changed_layout["logical_slots"][0]["shard"]["placements"][0]["kind"] = (
        PlacementKind.REPLICATE.value
    )
    corruptions.append((changed_layout, "does not match"))

    changed_fqn = copy.deepcopy(guard)
    changed_fqn["logical_slots"][1]["shard"]["parameter"]["fqn"] = (
        "Model.OtherRemote"
    )
    corruptions.append((changed_fqn, "does not match"))

    changed_name = copy.deepcopy(guard)
    changed_name["logical_slots"][1]["compatibility_name"] = "other"
    corruptions.append((changed_name, "does not match"))

    for corrupted_guard, match in corruptions:
        target = build()
        before = _snapshot(target)
        corrupted = copy.deepcopy(checkpoint)
        _replace_native_guard(corrupted, corrupted_guard)
        with pytest.raises(ValueError, match=match):
            target.load_state_dict(corrupted)
        _assert_snapshot_identity(target, before)

    def build_all_nonowner():
        remote = torch.nn.Parameter(torch.full((2, 2), 2.0))
        optimizer = GefenMuon([("remote", remote)], fused=False)
        identity = ParameterIdentity("Model.Remote", (2, 2))
        records = tuple(
            _whole_owner_shard(identity, group, member, "rank:1")
            for member in group.ordered_members
        )
        optimizer.post_sharding(
            (ParameterRebinding(remote, None, records[0]),),
            manifest=ShardingManifest(records),
            codebook_process_group=CodebookProcessGroupBinding(
                group,
                "rank:0",
                object(),
                torch.device("cpu"),
            ),
        )
        return optimizer

    all_nonowner = build_all_nonowner()
    all_nonowner_checkpoint = all_nonowner.state_dict()
    all_nonowner_guard = all_nonowner_checkpoint["gefen_native_local_shards"]
    assert all_nonowner.param_groups[0]["params"] == []
    assert [
        (
            record["group_index"],
            record["original_slot_index"],
            record["compatibility_name"],
            record["shard"]["parameter"]["fqn"],
        )
        for record in all_nonowner_guard["logical_slots"]
    ] == [(0, 0, "remote", "Model.Remote")]
    all_nonowner_target = build_all_nonowner()
    all_nonowner_target.load_state_dict(copy.deepcopy(all_nonowner_checkpoint))
    assert all_nonowner_target._gefen_logical_slots == all_nonowner._gefen_logical_slots


def test_unscoped_native_checkpoint_does_not_serialize_a_none_scope():
    parameter = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    optimizer = Gefen([parameter], fused=False)
    checkpoint = optimizer.state_dict()

    assert "gefen_codebook_scope" not in checkpoint
    assert all("codebook_scope" not in group["_gefen_checkpoint_metadata"] for group in checkpoint["param_groups"])


def test_capturable_optimizer_rejects_manual_codebook_replacement():
    parameter = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    optimizer = Gefen(
        [("p", parameter)],
        fused=False,
        factored_v_2d=False,
        capturable=True,
    )
    _finalize_replicated(optimizer, parameter, fqn="P")
    parameter.grad = torch.arange(1, 9, dtype=torch.float32)
    assert optimizer.initialize_codebook()

    with pytest.raises(RuntimeError, match="capturable=True"):
        optimizer.refresh_codebook()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_capturable_scope_rejects_first_active_step_inside_graph_capture():
    parameter = torch.nn.Parameter(torch.arange(8, dtype=torch.float32, device="cuda"))
    optimizer = Gefen(
        [("p", parameter)],
        fused=False,
        factored_v_2d=False,
        capturable=True,
    )
    _finalize_replicated(optimizer, parameter, fqn="P")
    parameter.grad = torch.arange(1, 9, dtype=torch.float32, device="cuda")
    graph = torch.cuda.CUDAGraph()

    with pytest.raises(RuntimeError, match="eager warmup"):
        with torch.cuda.graph(graph):
            optimizer.step()

    assert optimizer._gefen_codebook is None
    assert optimizer._gefen_global_step == 0


def test_capturable_multi_member_scope_binding_is_atomic_rejection():
    parameter = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    optimizer = Gefen(
        [("p", parameter)],
        fused=False,
        factored_v_2d=False,
        capturable=True,
    )
    group = ProcessGroupIdentity("dp", ("rank:0", "rank:1"))
    identity = ParameterIdentity("P", (8,))
    records = tuple(_replicated_shard(identity, group, member) for member in group.ordered_members)
    before = _snapshot(optimizer)

    with pytest.raises(ValueError, match="capturable=True"):
        optimizer.post_sharding(
            (ParameterRebinding(parameter, parameter, records[0]),),
            manifest=ShardingManifest(records),
            codebook_process_group=CodebookProcessGroupBinding(group, "rank:0", object(), "cpu"),
        )

    _assert_snapshot_identity(optimizer, before)


def test_muon_whole_owner_scope_enables_owner_update_with_sync_requirement():
    parameter = torch.nn.Parameter(torch.ones(4, 4))
    optimizer = GefenMuon([("matrix", parameter)], fused=False)
    group = ProcessGroupIdentity("owner_group", ("rank:0",))
    identity = ParameterIdentity("Matrix", (4, 4))
    shard = _whole_owner_shard(identity, group, "rank:0", "rank:0")
    binding = _single_member_binding(group)
    optimizer.post_sharding(
        (ParameterRebinding(parameter, parameter, shard),),
        manifest=ShardingManifest((shard,)),
        codebook_process_group=binding,
    )
    before = parameter.detach().clone()
    parameter.grad = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)

    optimizer.step()

    support = next(
        item
        for item in optimizer.optimizer_contract().capabilities.training
        if item.layout is ParameterLayout.WHOLE_PARAMETER_OWNER
    )
    assert support.process_group_scope is ProcessGroupScope.ADAPTER_DEFINED
    assert support.requires_complete_parameter_storage
    assert support.requires_post_step_parameter_sync
    assert all(
        ParameterLayout.WHOLE_PARAMETER_OWNER not in checkpoint.same_topology
        for checkpoint in optimizer.optimizer_contract().capabilities.checkpoints
    )
    owner_variants = [
        variant
        for variant in optimizer.optimizer_contract().state_layout.parameter_variants
        if ParameterLayout.WHOLE_PARAMETER_OWNER in variant.layouts and variant.initialized
    ]
    assert owner_variants
    assert all(
        variant.extent is StateExtent.OWNER_PARAMETER and variant.role is ParameterStateRole.OWNER
        for variant in owner_variants
    )
    assert not torch.equal(parameter, before)
    assert optimizer._gefen_global_step == 1

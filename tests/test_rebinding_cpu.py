"""CPU coverage for atomic pre-initialization parameter rebinding."""

import copy
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
import gc
import weakref

import pytest
import torch

from gefen import (
    Gefen,
    GefenMuon,
    GefenMuonHybrid,
    LogicalSlice,
    ParameterIdentity,
    ParameterLayout,
    ParameterRebinding,
    PlacementKind,
    ProcessGroupIdentity,
    ShardIdentity,
    ShardPlacement,
    ShardingManifest,
)
from gefen.rebinding import LogicalSlotBinding


def _replicated(parameter, fqn, shape=None):
    identity = ParameterIdentity(fqn, tuple(parameter.shape) if shape is None else shape)
    shard = ShardIdentity(
        identity,
        ParameterLayout.REPLICATED,
        LogicalSlice.full(identity),
    )
    return shard


def _flat_manifest(fqn, shape, lengths, local_member):
    identity = ParameterIdentity(fqn, shape)
    members = tuple("rank:{}".format(index) for index in range(len(lengths)))
    group = ProcessGroupIdentity("data_parallel", members)
    shards = []
    offset = 0
    for coordinate, (member, length) in enumerate(zip(members, lengths)):
        shards.append(
            ShardIdentity(
                identity,
                ParameterLayout.FLATTENED_ELEMENT_SHARD,
                LogicalSlice(offset, length),
                placements=(
                    ShardPlacement(
                        "data_parallel",
                        PlacementKind.FLAT_SHARD,
                        coordinate,
                        len(members),
                    ),
                ),
                process_group=group,
                local_member=member,
            )
        )
        offset += length
    manifest = ShardingManifest(tuple(reversed(shards)))
    local = next(item for item in manifest.shards if item.local_member == local_member)
    return local, manifest


def _owner_shard(identity, group, local_member, owner):
    coordinate = group.ordered_members.index(local_member)
    return ShardIdentity(
        identity,
        ParameterLayout.WHOLE_PARAMETER_OWNER,
        LogicalSlice.full(identity) if local_member == owner else LogicalSlice(0, 0),
        placements=(
            ShardPlacement(
                "data_parallel",
                PlacementKind.WHOLE_PARAMETER_OWNER,
                coordinate,
                len(group.ordered_members),
            ),
        ),
        process_group=group,
        local_member=local_member,
        owner=owner,
    )


def _nested_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            _nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _nested_equal(left_item, right_item)
    elif torch.is_tensor(left):
        assert torch.equal(left, right)
    else:
        assert left == right


def _contains_tensor(value):
    if torch.is_tensor(value):
        return True
    if is_dataclass(value) and not isinstance(value, type):
        return any(
            _contains_tensor(getattr(value, field.name)) for field in fields(value)
        )
    if isinstance(value, dict):
        return any(
            _contains_tensor(key) or _contains_tensor(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_tensor(item) for item in value)
    return False


def _snapshot(optimizer):
    return {
        "state": optimizer.state,
        "state_items": tuple(
            (parameter, pstate, copy.deepcopy(pstate)) for parameter, pstate in optimizer.state.items()
        ),
        "param_groups": optimizer.param_groups,
        "groups": tuple(
            (
                group,
                group["params"],
                tuple(group["params"]),
                copy.deepcopy({key: value for key, value in group.items() if key != "params"}),
            )
            for group in optimizer.param_groups
        ),
        "param_names": optimizer._param_names,
        "param_names_value": dict(optimizer._param_names),
        "bindings": optimizer._gefen_shard_bindings,
        # Copy the mapping so an in-place add/remove/replace of a binding that
        # keeps the registry object identity is still caught.
        "bindings_contents": dict(optimizer._gefen_shard_bindings),
        "local_bindings": optimizer._gefen_local_shard_bindings,
        "manifest": optimizer._gefen_sharding_manifest,
        "finalized": optimizer._gefen_post_sharding_finalized,
        "finalized_slots": optimizer._gefen_finalized_slots,
        "logical_slots": optimizer._gefen_logical_slots,
        "caches": tuple(
            (name, getattr(optimizer, name))
            for name in (
                "_gefen_codebook_by_device",
                "_gefen_codebook_lut_by_device",
                "_sr_seed_by_device",
                "_gefen_global_step_by_device",
            )
        ),
        # Bitwise clone of each device-cache entry so an in-place copy_() into a
        # cached tensor that preserves the cache identity is still caught.
        "caches_contents": tuple(
            (
                name,
                {
                    device: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
                    for device, value in getattr(optimizer, name).items()
                },
            )
            for name in (
                "_gefen_codebook_by_device",
                "_gefen_codebook_lut_by_device",
                "_sr_seed_by_device",
                "_gefen_global_step_by_device",
            )
        ),
        "capt_stacks": optimizer._capt_stacks,
        "static_mark_sig": optimizer._static_mark_sig,
        "global_step": optimizer._gefen_global_step,
        "codebook": optimizer._gefen_codebook,
    }


def _assert_snapshot(optimizer, snapshot):
    assert optimizer.state is snapshot["state"]
    assert optimizer.param_groups is snapshot["param_groups"]
    assert optimizer._param_names is snapshot["param_names"]
    assert optimizer._param_names == snapshot["param_names_value"]
    assert optimizer._gefen_shard_bindings is snapshot["bindings"]
    expected_bindings = snapshot["bindings_contents"]
    assert optimizer._gefen_shard_bindings.keys() == expected_bindings.keys()
    for key, expected_binding in expected_bindings.items():
        assert optimizer._gefen_shard_bindings[key] is expected_binding
    assert optimizer._gefen_local_shard_bindings is snapshot["local_bindings"]
    assert optimizer._gefen_sharding_manifest is snapshot["manifest"]
    assert optimizer._gefen_post_sharding_finalized is snapshot["finalized"]
    assert optimizer._gefen_finalized_slots is snapshot["finalized_slots"]
    assert optimizer._gefen_logical_slots is snapshot["logical_slots"]
    assert optimizer._capt_stacks is snapshot["capt_stacks"]
    assert optimizer._static_mark_sig is snapshot["static_mark_sig"]
    assert optimizer._gefen_global_step is snapshot["global_step"]
    assert optimizer._gefen_codebook is snapshot["codebook"]
    for parameter, state_ref, expected in snapshot["state_items"]:
        assert optimizer.state[parameter] is state_ref
        _nested_equal(optimizer.state[parameter], expected)
    for group, params_ref, expected_params, expected_values in snapshot["groups"]:
        assert group in optimizer.param_groups
        assert group["params"] is params_ref
        assert tuple(group["params"]) == expected_params
        _nested_equal(
            {key: value for key, value in group.items() if key != "params"},
            expected_values,
        )
    for name, cache_ref in snapshot["caches"]:
        assert getattr(optimizer, name) is cache_ref
    for name, expected_cache in snapshot["caches_contents"]:
        live_cache = getattr(optimizer, name)
        assert live_cache.keys() == expected_cache.keys()
        for device, expected_value in expected_cache.items():
            live_value = live_cache[device]
            if torch.is_tensor(expected_value):
                assert torch.is_tensor(live_value)
                assert torch.equal(live_value, expected_value)
            else:
                assert live_value == expected_value


@pytest.mark.parametrize(
    ("changes", "error", "match"),
    [
        ({"group_index": True}, TypeError, "group_index must be an integer"),
        ({"group_index": -1}, ValueError, "group_index must be nonnegative"),
        (
            {"original_slot_index": 1.0},
            TypeError,
            "original_slot_index must be an integer",
        ),
        (
            {"original_slot_index": -1},
            ValueError,
            "original_slot_index must be nonnegative",
        ),
        (
            {"compatibility_name": object()},
            TypeError,
            "compatibility_name must be a string",
        ),
        (
            {"compatibility_name": "MixedCase"},
            ValueError,
            "compatibility_name must be lowercase",
        ),
        ({"shard": object()}, TypeError, "shard must be a ShardIdentity"),
    ],
)
def test_logical_slot_binding_strict_validation(changes, error, match):
    parameter = torch.nn.Parameter(torch.ones(4))
    values = {
        "group_index": 0,
        "original_slot_index": 0,
        "compatibility_name": "",
        "shard": _replicated(parameter, "Weight"),
    }
    values.update(changes)

    with pytest.raises(error, match=match):
        LogicalSlotBinding(**values)


def _interleaved_logical_slot_optimizer():
    sources = [
        torch.nn.Parameter(torch.full((2, 2), float(index + 1)))
        for index in range(5)
    ]
    source_refs = tuple(weakref.ref(parameter) for parameter in sources)
    names = (
        "group0.live0",
        "group0.remote",
        "group0.live2",
        "group1.remote",
        "group1.live1",
    )
    optimizer = GefenMuon(
        [
            {"params": list(zip(names[:3], sources[:3]))},
            {"params": list(zip(names[3:], sources[3:]))},
        ],
        fused=False,
    )
    process_group = ProcessGroupIdentity("data_parallel", ("rank:0", "rank:1"))
    owners = ("rank:0", "rank:1", "rank:0", "rank:1", "rank:0")
    records = []
    local_shards = []
    for index, owner in enumerate(owners):
        identity = ParameterIdentity("Model.Parameter{}".format(index), (2, 2))
        parameter_records = tuple(
            _owner_shard(identity, process_group, member, owner)
            for member in process_group.ordered_members
        )
        records.extend(parameter_records)
        local_shards.append(
            next(
                shard
                for shard in parameter_records
                if shard.local_member == "rank:0"
            )
        )
    targets = {
        index: torch.nn.Parameter(sources[index].detach().clone())
        for index in (0, 2, 4)
    }
    rebindings = [
        ParameterRebinding(
            sources[index],
            targets.get(index),
            local_shards[index],
        )
        for index in (4, 1, 3, 0, 2)
    ]
    optimizer.post_sharding(
        rebindings,
        manifest=ShardingManifest(tuple(records)),
    )
    return optimizer, source_refs, targets, tuple(local_shards), names


def test_logical_slot_registry_preserves_original_interleaved_structure_without_tensors():
    optimizer, source_refs, targets, local_shards, names = (
        _interleaved_logical_slot_optimizer()
    )
    gc.collect()

    assert all(reference() is None for reference in source_refs)
    assert optimizer.param_groups[0]["params"] == [targets[0], targets[2]]
    assert optimizer.param_groups[1]["params"] == [targets[4]]
    assert optimizer.param_groups[0]["param_names"] == [names[0], names[2]]
    assert optimizer.param_groups[1]["param_names"] == [names[4]]
    assert tuple(
        (slot.group_index, slot.original_slot_index)
        for slot in optimizer._gefen_logical_slots
    ) == ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1))
    assert tuple(
        slot.compatibility_name for slot in optimizer._gefen_logical_slots
    ) == names
    assert tuple(
        slot.shard for slot in optimizer._gefen_logical_slots
    ) == local_shards
    assert not _contains_tensor(optimizer._gefen_logical_slots)
    assert all(
        not hasattr(slot, "__dict__") for slot in optimizer._gefen_logical_slots
    )
    with pytest.raises(FrozenInstanceError):
        optimizer._gefen_logical_slots[0].compatibility_name = "changed"
    assert optimizer.optimizer_contract().capabilities.stable_shard_identity


@pytest.mark.parametrize(
    "corruption",
    [
        "logical_slots",
        "manifest",
        "finalized_slots",
        "local_bindings",
        "group_names",
        "name_cache",
        "state_name",
    ],
)
def test_finalized_layout_guard_checks_logical_slot_caches_and_names(corruption):
    parameters = [
        torch.nn.Parameter(torch.full((4,), float(index + 1)))
        for index in range(2)
    ]
    optimizer = Gefen(
        [("first", parameters[0]), ("second", parameters[1])],
        fused=False,
        factored_v_2d=False,
    )
    shards = tuple(
        _replicated(parameter, "Model.Parameter{}".format(index))
        for index, parameter in enumerate(parameters)
    )
    optimizer.post_sharding(
        tuple(
            ParameterRebinding(parameter, parameter, shard)
            for parameter, shard in zip(parameters, shards)
        ),
        manifest=ShardingManifest(shards),
    )

    if corruption == "logical_slots":
        slots = optimizer._gefen_logical_slots
        optimizer._gefen_logical_slots = (
            replace(slots[0], compatibility_name="changed"),
            slots[1],
        )
    elif corruption == "manifest":
        optimizer._gefen_sharding_manifest = ShardingManifest(
            (
                _replicated(parameters[0], "Model.Parameter0", shape=(5,)),
                shards[1],
            )
        )
    elif corruption == "finalized_slots":
        optimizer._gefen_finalized_slots = (tuple(reversed(parameters)),)
    elif corruption == "local_bindings":
        optimizer._gefen_local_shard_bindings = tuple(
            reversed(optimizer._gefen_local_shard_bindings)
        )
    elif corruption == "group_names":
        optimizer.param_groups[0]["param_names"].reverse()
    elif corruption == "name_cache":
        optimizer._param_names[parameters[0]] = "changed"
    else:
        optimizer.state[parameters[0]]["name"] = "changed"

    assert not optimizer.optimizer_contract().capabilities.stable_shard_identity
    with pytest.raises(RuntimeError, match="changed outside post_sharding"):
        optimizer.state_dict()


def test_replicated_rebind_preserves_legacy_name_and_enables_identity_contract():
    old = torch.nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 4))
    new = torch.nn.Parameter(old.detach().clone())
    optimizer = Gefen([("Encoder.MixedCase.Weight", old)], fused=False)
    old_group_options = {key: value for key, value in optimizer.param_groups[0].items() if key != "params"}
    identity = ParameterIdentity("Encoder.MixedCase.Weight", (4, 4))

    optimizer.rebind_parameter(old, new, identity=identity)

    assert optimizer.param_groups[0]["params"] == [new]
    assert {key: value for key, value in optimizer.param_groups[0].items() if key != "params"} == old_group_options
    assert old not in optimizer.state
    assert optimizer.state[new] == {"name": "encoder.mixedcase.weight"}
    assert optimizer.parameter_identity(new) == identity
    assert optimizer.shard_identity(new).parameter.fqn == "Encoder.MixedCase.Weight"
    assert optimizer.shard_bindings() == ((new, optimizer.shard_identity(new)),)
    assert optimizer.sharding_manifest().shards == (optimizer.shard_identity(new),)
    contract = optimizer.optimizer_contract()
    assert contract.capabilities.canonical_parameter_fqns
    assert contract.capabilities.stable_shard_identity
    assert contract.capabilities.shard_rebinding
    assert contract.capabilities.post_sharding
    assert contract.capabilities.canonical_state_io
    assert contract.capabilities.explicit_process_group_codebook_scope
    with pytest.raises(RuntimeError, match="already finalized"):
        optimizer.rebind_parameter(new, new, identity=identity)
    with pytest.raises(RuntimeError, match="cannot add"):
        optimizer.add_param_group({"params": [torch.nn.Parameter(torch.ones(2))]})


def test_same_tensor_identity_only_rebinding_is_supported():
    parameter = torch.nn.Parameter(torch.ones(4, 4))
    optimizer = Gefen([("legacy.name", parameter)], fused=False)
    identity = ParameterIdentity("Exact.FQN", (4, 4))

    optimizer.rebind_parameter(parameter, parameter, identity=identity)

    assert optimizer.param_groups[0]["params"][0] is parameter
    assert optimizer.state[parameter]["name"] == "legacy.name"
    assert optimizer.parameter_identity(parameter).fqn == "Exact.FQN"


def test_replicated_rebinding_accepts_nonoverlapping_strided_storage():
    old = torch.nn.Parameter(torch.ones(4))
    target = torch.nn.Parameter(torch.arange(8, dtype=torch.float32)[::2])
    optimizer = Gefen([("weight", old)], fused=False, factored_v_2d=False)

    optimizer.rebind_parameter(
        old,
        target,
        identity=ParameterIdentity("Weight", (4,)),
    )

    assert optimizer.param_groups[0]["params"] == [target]
    assert optimizer.optimizer_contract().capabilities.stable_shard_identity


def test_rebinding_after_external_parameter_replacement_removes_orphan_state():
    old = torch.nn.Parameter(torch.ones(4, 4))
    new = torch.nn.Parameter(torch.full((4, 4), 2.0))
    optimizer = Gefen([("layer.weight", old)], fused=False)
    optimizer.param_groups[0]["params"][0] = new

    optimizer.rebind_parameter(
        old,
        new,
        identity=ParameterIdentity("layer.weight", (4, 4)),
    )

    assert old not in optimizer.state
    assert old not in optimizer._param_names
    assert optimizer.state[new] == {"name": "layer.weight"}


def test_external_replacement_reorder_preserves_each_source_compatibility_name():
    old_a = torch.nn.Parameter(torch.ones(4))
    old_b = torch.nn.Parameter(torch.ones(4) * 2)
    new_a = torch.nn.Parameter(torch.ones(4) * 3)
    new_b = torch.nn.Parameter(torch.ones(4) * 4)
    optimizer = Gefen(
        [("source.a", old_a), ("source.b", old_b)],
        fused=False,
        factored_v_2d=False,
    )
    optimizer.param_groups[0]["params"][:] = [new_b, new_a]
    shard_a = _replicated(new_a, "Canonical.A")
    shard_b = _replicated(new_b, "Canonical.B")

    optimizer.post_sharding(
        (
            ParameterRebinding(old_a, new_a, shard_a),
            ParameterRebinding(old_b, new_b, shard_b),
        ),
        manifest=ShardingManifest((shard_b, shard_a)),
    )

    assert optimizer.param_groups[0]["params"] == [new_b, new_a]
    assert optimizer.param_groups[0]["param_names"] == ["source.b", "source.a"]
    assert optimizer.state[new_a]["name"] == "source.a"
    assert optimizer.state[new_b]["name"] == "source.b"


def test_flattened_plain_gefen_rebind_steps_like_direct_local_shard():
    old = torch.nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 4))
    rebound = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    oracle = torch.nn.Parameter(rebound.detach().clone())
    optimizer = Gefen(
        [("layer.weight", old)],
        lr=1e-3,
        fused=False,
        factored_v_2d=False,
    )
    reference = Gefen(
        [("layer.weight", oracle)],
        lr=1e-3,
        fused=False,
        factored_v_2d=False,
    )
    local, manifest = _flat_manifest("Model.Layer.Weight", (4, 4), (8, 8), "rank:0")
    optimizer.post_sharding((ParameterRebinding(old, rebound, local),), manifest=manifest)
    grad = torch.linspace(-1, 1, 8)

    for parameter, current in ((rebound, optimizer), (oracle, reference)):
        parameter.grad = grad.clone()
        current.step()
        current.zero_grad()

    assert torch.equal(rebound, oracle)
    _nested_equal(optimizer.state[rebound], reference.state[oracle])
    assert torch.equal(optimizer._gefen_codebook, reference._gefen_codebook)


def test_native_resume_preserves_target_flat_shard_identity_and_continuation():
    def build():
        old = torch.nn.Parameter(torch.zeros(4, 4))
        local_param = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
        optimizer = Gefen(
            [("layer.weight", old)],
            lr=1e-3,
            fused=False,
            factored_v_2d=False,
        )
        local, manifest = _flat_manifest("Model.Layer.Weight", (4, 4), (8, 8), "rank:0")
        optimizer.post_sharding((ParameterRebinding(old, local_param, local),), manifest=manifest)
        return optimizer, local_param, local

    source, source_param, _ = build()
    target, target_param, target_identity = build()
    for grad in (torch.linspace(-1, 1, 8), torch.linspace(1, -1, 8)):
        source_param.grad = grad.clone()
        source.step()
        source.zero_grad()
    checkpoint = copy.deepcopy(source.state_dict())
    with torch.no_grad():
        target_param.copy_(source_param)

    target.load_state_dict(checkpoint)

    assert target.shard_identity(target_param) == target_identity
    assert target.optimizer_contract().capabilities.stable_shard_identity
    next_grad = torch.linspace(-0.5, 0.5, 8)
    for parameter, optimizer in (
        (source_param, source),
        (target_param, target),
    ):
        parameter.grad = next_grad.clone()
        optimizer.step()
        optimizer.zero_grad()
    assert torch.equal(source_param, target_param)
    _nested_equal(source.state_dict(), target.state_dict())


def test_factored_logical_matrix_flattening_rejects_atomically():
    old = torch.nn.Parameter(torch.ones(4, 4))
    new = torch.nn.Parameter(torch.ones(8))
    optimizer = Gefen([("layer.weight", old)], fused=False, factored_v_2d=True)
    local, manifest = _flat_manifest("layer.weight", (4, 4), (8, 8), "rank:0")
    snapshot = _snapshot(optimizer)

    with pytest.raises(ValueError, match="factored_v_2d=False"):
        optimizer.post_sharding((ParameterRebinding(old, new, local),), manifest=manifest)

    _assert_snapshot(optimizer, snapshot)


def test_late_batch_failure_is_atomic_and_continuation_is_bit_exact():
    def build():
        params = [
            torch.nn.Parameter(torch.arange(8, dtype=torch.float32)),
            torch.nn.Parameter(torch.arange(8, dtype=torch.float32) + 1),
        ]
        return Gefen(
            [("first", params[0]), ("second", params[1])],
            fused=False,
            factored_v_2d=False,
        ), params

    target, target_params = build()
    control, control_params = build()
    replacements = [
        torch.nn.Parameter(torch.arange(8, dtype=torch.float32) + 2),
        torch.nn.Parameter(torch.arange(7, dtype=torch.float32) + 3),
    ]
    first = _replicated(replacements[0], "First", (8,))
    second = _replicated(torch.empty(8), "Second", (8,))
    manifest = ShardingManifest((second, first))
    snapshot = _snapshot(target)

    with pytest.raises(ValueError, match="complete parameter storage"):
        target.post_sharding(
            (
                ParameterRebinding(target_params[0], replacements[0], first),
                ParameterRebinding(target_params[1], replacements[1], second),
            ),
            manifest=manifest,
        )

    _assert_snapshot(target, snapshot)
    grads = (torch.linspace(-1, 1, 8), torch.linspace(1, -1, 8))
    for params, optimizer in ((target_params, target), (control_params, control)):
        for parameter, grad in zip(params, grads):
            parameter.grad = grad.clone()
        optimizer.step()
        optimizer.zero_grad()
    for target_param, control_param in zip(target_params, control_params):
        assert torch.equal(target_param, control_param)
    _nested_equal(target.state_dict(), control.state_dict())


def test_checkpoint_schema_preparation_failure_is_atomic(monkeypatch):
    old = torch.nn.Parameter(torch.ones(4))
    new = torch.nn.Parameter(torch.ones(4) * 2)
    optimizer = Gefen([("weight", old)], fused=False, factored_v_2d=False)
    shard = _replicated(new, "Weight")
    snapshot = _snapshot(optimizer)

    def reject_schema():
        raise RuntimeError("schema preparation failed")

    monkeypatch.setattr(
        optimizer,
        "_install_rank_local_checkpoint_schema",
        reject_schema,
    )
    with pytest.raises(RuntimeError, match="schema preparation failed"):
        optimizer.post_sharding(
            (ParameterRebinding(old, new, shard),),
            manifest=ShardingManifest((shard,)),
        )

    _assert_snapshot(optimizer, snapshot)


def test_whole_optimizer_pristine_guard_rejects_after_partial_initialization():
    params = [
        torch.nn.Parameter(torch.ones(8)),
        torch.nn.Parameter(torch.ones(8) * 2),
    ]
    optimizer = Gefen(
        [("first", params[0]), ("second", params[1])],
        fused=False,
        factored_v_2d=False,
    )
    params[0].grad = torch.ones_like(params[0])
    optimizer.step()
    optimizer.zero_grad()
    replacements = [torch.nn.Parameter(param.detach().clone()) for param in params]
    shards = (
        _replicated(replacements[0], "First"),
        _replicated(replacements[1], "Second"),
    )
    snapshot = _snapshot(optimizer)

    with pytest.raises(RuntimeError, match="before optimizer mutation"):
        optimizer.post_sharding(
            tuple(ParameterRebinding(old, new, shard) for old, new, shard in zip(params, replacements, shards)),
            manifest=ShardingManifest(shards),
        )

    _assert_snapshot(optimizer, snapshot)


@pytest.mark.parametrize("failure", ["duplicate_target", "unknown_source", "gradient", "manifest"])
def test_rebinding_structural_failures_are_no_ops(failure):
    old = torch.nn.Parameter(torch.ones(4))
    new = torch.nn.Parameter(torch.ones(4) * 2)
    optimizer = Gefen([("weight", old)], fused=False, factored_v_2d=False)
    shard = _replicated(new, "Weight")
    manifest = ShardingManifest((shard,))
    binding = ParameterRebinding(old, new, shard)
    if failure == "duplicate_target":
        other = torch.nn.Parameter(torch.ones(4))
        optimizer = Gefen(
            [("weight", old), ("other", other)],
            fused=False,
            factored_v_2d=False,
        )
        other_shard = _replicated(new, "Other")
        manifest = ShardingManifest((shard, other_shard))
        bindings = (
            binding,
            ParameterRebinding(other, new, other_shard),
        )
    elif failure == "unknown_source":
        bindings = (ParameterRebinding(torch.nn.Parameter(torch.ones(4)), new, shard),)
    elif failure == "gradient":
        new.grad = torch.ones_like(new)
        bindings = (binding,)
    else:
        other_shard = _replicated(new, "Other")
        manifest = ShardingManifest((other_shard,))
        bindings = (binding,)
    snapshot = _snapshot(optimizer)

    with pytest.raises((RuntimeError, ValueError)):
        optimizer.post_sharding(bindings, manifest=manifest)

    _assert_snapshot(optimizer, snapshot)


def test_rebinding_rejects_distinct_targets_with_overlapping_storage():
    old = [
        torch.nn.Parameter(torch.ones(8)),
        torch.nn.Parameter(torch.ones(8) * 2),
    ]
    optimizer = Gefen(
        [("first", old[0]), ("second", old[1])],
        fused=False,
        factored_v_2d=False,
    )
    storage = torch.arange(12, dtype=torch.float32)
    targets = [
        torch.nn.Parameter(storage[:8]),
        torch.nn.Parameter(storage[4:]),
    ]
    shards = (
        _replicated(targets[0], "First"),
        _replicated(targets[1], "Second"),
    )
    snapshot = _snapshot(optimizer)

    with pytest.raises(ValueError, match="must not overlap"):
        optimizer.post_sharding(
            (
                ParameterRebinding(old[0], targets[0], shards[0]),
                ParameterRebinding(old[1], targets[1], shards[1]),
            ),
            manifest=ShardingManifest(shards),
        )

    _assert_snapshot(optimizer, snapshot)


@pytest.mark.parametrize("target_kind", ["noncontiguous", "integer", "internal_overlap"])
def test_rebinding_rejects_invalid_target_storage_atomically(target_kind):
    old = torch.nn.Parameter(torch.ones(4))
    optimizer = Gefen([("weight", old)], fused=False, factored_v_2d=False)
    if target_kind == "noncontiguous":
        target = torch.nn.Parameter(torch.arange(8, dtype=torch.float32)[::2])
        local, manifest = _flat_manifest("Weight", (8,), (4, 4), "rank:0")
        error = "contiguous physical storage"
    elif target_kind == "integer":
        target = torch.nn.Parameter(torch.ones(4, dtype=torch.int64), requires_grad=False)
        local = _replicated(target, "Weight")
        manifest = ShardingManifest((local,))
        error = "floating-point storage"
    else:
        target = torch.nn.Parameter(torch.ones(1).expand(4))
        local = _replicated(target, "Weight")
        manifest = ShardingManifest((local,))
        error = "internal storage overlap"
    snapshot = _snapshot(optimizer)
    before = target.detach().clone()

    with pytest.raises(ValueError, match=error):
        optimizer.post_sharding((ParameterRebinding(old, target, local),), manifest=manifest)

    _assert_snapshot(optimizer, snapshot)
    assert torch.equal(target, before)


def test_rebinding_rejects_mixed_local_members_for_one_process_group():
    old = [torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(2))]
    targets = [torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(2))]
    optimizer = Gefen(
        [("first", old[0]), ("second", old[1])],
        fused=False,
        factored_v_2d=False,
    )
    first_local, first_manifest = _flat_manifest("First", (4,), (2, 2), "rank:0")
    second_local, second_manifest = _flat_manifest("Second", (4,), (2, 2), "rank:1")
    manifest = ShardingManifest(first_manifest.shards + second_manifest.shards)
    snapshot = _snapshot(optimizer)

    with pytest.raises(ValueError, match="one member"):
        optimizer.post_sharding(
            (
                ParameterRebinding(old[0], targets[0], first_local),
                ParameterRebinding(old[1], targets[1], second_local),
            ),
            manifest=manifest,
        )

    _assert_snapshot(optimizer, snapshot)


def test_external_replacement_rejects_duplicate_stale_sources():
    old = [torch.nn.Parameter(torch.ones(4)), torch.nn.Parameter(torch.ones(4))]
    targets = [torch.nn.Parameter(torch.ones(4)), torch.nn.Parameter(torch.ones(4))]
    optimizer = Gefen(
        [("first", old[0]), ("second", old[1])],
        fused=False,
        factored_v_2d=False,
    )
    optimizer.param_groups[0]["params"][:] = targets
    shards = (
        _replicated(targets[0], "First"),
        _replicated(targets[1], "Second"),
    )
    snapshot = _snapshot(optimizer)

    with pytest.raises(ValueError, match="source tensors must be unique"):
        optimizer.post_sharding(
            (
                ParameterRebinding(old[0], targets[0], shards[0]),
                ParameterRebinding(old[0], targets[1], shards[1]),
            ),
            manifest=ShardingManifest(shards),
        )

    _assert_snapshot(optimizer, snapshot)


@pytest.mark.parametrize("mutation", ["replace", "reorder"])
def test_finalized_layout_guard_rejects_direct_parameter_group_mutation(mutation):
    old = [torch.nn.Parameter(torch.ones(4)), torch.nn.Parameter(torch.ones(4) * 2)]
    new = [torch.nn.Parameter(torch.ones(4) * 3), torch.nn.Parameter(torch.ones(4) * 4)]
    optimizer = Gefen(
        [("first", old[0]), ("second", old[1])],
        fused=False,
        factored_v_2d=False,
    )
    shards = (_replicated(new[0], "First"), _replicated(new[1], "Second"))
    optimizer.post_sharding(
        (
            ParameterRebinding(old[0], new[0], shards[0]),
            ParameterRebinding(old[1], new[1], shards[1]),
        ),
        manifest=ShardingManifest(shards),
    )
    before = [parameter.detach().clone() for parameter in new]
    if mutation == "replace":
        rogue = torch.nn.Parameter(torch.ones(4) * 5)
        optimizer.param_groups[0]["params"][0] = rogue
    else:
        optimizer.param_groups[0]["params"].reverse()
    assert not optimizer.optimizer_contract().capabilities.stable_shard_identity

    with pytest.raises(RuntimeError, match="changed outside post_sharding"):
        optimizer.step()
    with pytest.raises(RuntimeError, match="changed outside post_sharding"):
        optimizer.state_dict()
    with pytest.raises(RuntimeError, match="changed outside post_sharding"):
        optimizer.shard_bindings()
    with pytest.raises(RuntimeError, match="changed outside post_sharding"):
        optimizer.sharding_manifest()
    for parameter, expected in zip(new, before):
        assert torch.equal(parameter, expected)


@pytest.mark.parametrize("optimizer_type", ["plain", "muon"])
def test_finalized_layout_guard_rechecks_after_closure(optimizer_type):
    shape = (4,) if optimizer_type == "plain" else (2, 2)
    old = torch.nn.Parameter(torch.ones(shape))
    bound = torch.nn.Parameter(torch.ones(shape) * 2)
    rogue = torch.nn.Parameter(torch.ones(shape) * 3)
    if optimizer_type == "plain":
        optimizer = Gefen([("weight", old)], fused=False, factored_v_2d=False)
    else:
        optimizer = GefenMuon([("weight", old)], fused=False)
    optimizer.rebind_parameter(
        old,
        bound,
        identity=ParameterIdentity("Weight", shape),
    )
    bound_before = bound.detach().clone()
    rogue_before = rogue.detach().clone()

    def mutate_layout():
        optimizer.param_groups[0]["params"][0] = rogue
        rogue.grad = torch.ones_like(rogue)
        return torch.tensor(1.0)

    with pytest.raises(RuntimeError, match="changed outside post_sharding"):
        optimizer.step(mutate_layout)

    assert torch.equal(bound, bound_before)
    assert torch.equal(rogue, rogue_before)
    assert optimizer.state[bound] == {"name": "weight"}


def test_finalized_layout_guard_rechecks_after_load_pre_hook():
    old = torch.nn.Parameter(torch.ones(4))
    bound = torch.nn.Parameter(torch.ones(4) * 2)
    rogue = torch.nn.Parameter(torch.ones(4) * 3)
    optimizer = Gefen([("weight", old)], fused=False, factored_v_2d=False)
    optimizer.rebind_parameter(
        old,
        bound,
        identity=ParameterIdentity("Weight", (4,)),
    )
    checkpoint = copy.deepcopy(optimizer.state_dict())
    bound_before = bound.detach().clone()
    rogue_before = rogue.detach().clone()

    def mutate_layout(current, state_dict):
        current.param_groups[0]["params"][0] = rogue

    optimizer.register_load_state_dict_pre_hook(mutate_layout)
    with pytest.raises(RuntimeError, match="changed outside post_sharding"):
        optimizer.load_state_dict(checkpoint)

    assert torch.equal(bound, bound_before)
    assert torch.equal(rogue, rogue_before)
    assert optimizer.state[bound] == {"name": "weight"}


def test_muon_replicated_rebind_steps_like_direct_optimizer():
    old = torch.nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 4))
    rebound = torch.nn.Parameter(old.detach().clone())
    oracle = torch.nn.Parameter(old.detach().clone())
    optimizer = GefenMuon([("layer.weight", old)], fused=False)
    reference = GefenMuon([("layer.weight", oracle)], fused=False)
    identity = ParameterIdentity("Layer.Weight", (4, 4))
    optimizer.rebind_parameter(old, rebound, identity=identity)
    grad = torch.linspace(-1, 1, 16).reshape(4, 4)

    for parameter, current in ((rebound, optimizer), (oracle, reference)):
        parameter.grad = grad.clone()
        current.step()
        current.zero_grad()

    assert torch.equal(rebound, oracle)
    _nested_equal(optimizer.state[rebound], reference.state[oracle])


def test_muon_whole_owner_post_sharding_prunes_nonowner_and_blocks_training():
    old_owner = torch.nn.Parameter(torch.ones(4, 4))
    old_nonowner = torch.nn.Parameter(torch.ones(4, 4) * 2)
    new_owner = torch.nn.Parameter(torch.ones(4, 4) * 3)
    optimizer = GefenMuon(
        [("owner.weight", old_owner), ("remote.weight", old_nonowner)],
        fused=False,
    )
    group = ProcessGroupIdentity("data_parallel", ("rank:0", "rank:1"))
    owner_identity = ParameterIdentity("Owner.Weight", (4, 4))
    remote_identity = ParameterIdentity("Remote.Weight", (4, 4))
    owner_records = tuple(_owner_shard(owner_identity, group, member, "rank:0") for member in group.ordered_members)
    remote_records = tuple(_owner_shard(remote_identity, group, member, "rank:1") for member in group.ordered_members)
    manifest = ShardingManifest(owner_records + remote_records)
    local_owner = next(item for item in owner_records if item.local_member == "rank:0")
    local_nonowner = next(item for item in remote_records if item.local_member == "rank:0")

    optimizer.post_sharding(
        (
            ParameterRebinding(old_nonowner, None, local_nonowner),
            ParameterRebinding(old_owner, new_owner, local_owner),
        ),
        manifest=manifest,
    )

    assert optimizer.param_groups[0]["params"] == [new_owner]
    assert set(optimizer.state) == {new_owner}
    assert old_owner not in optimizer._param_names
    assert old_nonowner not in optimizer._param_names
    assert optimizer.shard_bindings() == tuple(
        sorted(
            ((new_owner, local_owner), (None, local_nonowner)),
            key=lambda item: item[1].sort_key,
        )
    )
    contract = optimizer.optimizer_contract()
    assert contract.capabilities.canonical_parameter_fqns
    assert contract.capabilities.stable_shard_identity
    assert contract.capabilities.explicit_process_group_codebook_scope
    assert all(
        support.layout is not ParameterLayout.WHOLE_PARAMETER_OWNER for support in contract.capabilities.training
    )
    snapshot = copy.deepcopy(optimizer.state_dict())
    parameter_snapshot = new_owner.detach().clone()
    closure_calls = []
    with pytest.raises(RuntimeError, match="codebook scope"):
        optimizer.step(lambda: closure_calls.append(True))
    assert closure_calls == []
    assert torch.equal(new_owner, parameter_snapshot)
    _nested_equal(optimizer.state_dict(), snapshot)


def test_muon_all_nonowner_post_sharding_retains_manifest_without_fake_state():
    old = torch.nn.Parameter(torch.ones(4, 4))
    optimizer = GefenMuon([("remote.weight", old)], fused=False)
    group = ProcessGroupIdentity("data_parallel", ("rank:0", "rank:1"))
    identity = ParameterIdentity("Remote.Weight", (4, 4))
    records = tuple(_owner_shard(identity, group, member, "rank:1") for member in group.ordered_members)
    local = next(item for item in records if item.local_member == "rank:0")
    manifest = ShardingManifest(records)

    optimizer.post_sharding((ParameterRebinding(old, None, local),), manifest=manifest)

    assert optimizer.param_groups[0]["params"] == []
    assert optimizer.param_groups[0]["param_names"] == []
    assert optimizer.state == {}
    assert optimizer.shard_bindings() == ((None, local),)
    assert optimizer.sharding_manifest() == manifest
    assert optimizer.optimizer_contract().capabilities.stable_shard_identity


def test_muon_flattened_rebinding_rejects_without_mutation():
    old = torch.nn.Parameter(torch.ones(4, 4))
    new = torch.nn.Parameter(torch.ones(8))
    optimizer = GefenMuon([("layer.weight", old)], fused=False)
    local, manifest = _flat_manifest("Layer.Weight", (4, 4), (8, 8), "rank:0")
    snapshot = _snapshot(optimizer)

    with pytest.raises(ValueError, match="complete matrices"):
        optimizer.post_sharding((ParameterRebinding(old, new, local),), manifest=manifest)

    _assert_snapshot(optimizer, snapshot)


def test_gefen_backed_hybrid_contract_claims_composite_rebinding():
    matrix = torch.nn.Parameter(torch.ones(4, 4))
    bias = torch.nn.Parameter(torch.ones(4))
    optimizer = GefenMuonHybrid(
        [("layer.weight", matrix)],
        [("layer.bias", bias)],
        lr=1e-3,
        fused=False,
    )
    contract = optimizer.optimizer_contract()
    assert contract.capabilities.shard_rebinding
    assert contract.capabilities.post_sharding
    assert contract.capabilities.explicit_process_group_codebook_scope
    assert not contract.capabilities.canonical_parameter_fqns
    assert not contract.capabilities.stable_shard_identity
    assert optimizer.state_dict()["backup_optimizer"] == "gefen"

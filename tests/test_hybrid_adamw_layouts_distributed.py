"""Two-rank AdamW Hybrid coverage for flattened and whole-owner layouts."""

from datetime import timedelta
import copy
import multiprocessing as mp
import os
import queue as queue_module
import shutil
import tempfile
import traceback

import pytest
import torch
import torch.distributed as dist

from _state_snapshot import assert_deep_state_snapshot, deep_state_snapshot


_WORLD = 2
_MEMBERS = ("rank:0", "rank:1")
_OWNER = "rank:1"
_SHAPE = (2, 3)
_DTYPES = (
    ("float16", torch.float16),
    ("bfloat16", torch.bfloat16),
    ("float32", torch.float32),
    ("float64", torch.float64),
)


def _manifest(layout_name):
    from gefen import (
        LogicalSlice,
        ParameterIdentity,
        ParameterLayout,
        PlacementKind,
        ProcessGroupIdentity,
        ShardIdentity,
        ShardPlacement,
        ShardingManifest,
    )

    group = ProcessGroupIdentity("data_parallel", _MEMBERS)
    identity = ParameterIdentity("Model.Backup.Weight", _SHAPE)
    shards = []
    for coordinate, member in enumerate(_MEMBERS):
        if layout_name == "flattened":
            layout = ParameterLayout.FLATTENED_ELEMENT_SHARD
            logical_slice = LogicalSlice(coordinate * 3, 3)
            placement_kind = PlacementKind.FLAT_SHARD
            owner = None
        else:
            layout = ParameterLayout.WHOLE_PARAMETER_OWNER
            logical_slice = LogicalSlice.full(identity) if member == _OWNER else LogicalSlice(0, 0)
            placement_kind = PlacementKind.WHOLE_PARAMETER_OWNER
            owner = _OWNER
        shards.append(
            ShardIdentity(
                identity,
                layout,
                logical_slice,
                placements=(
                    ShardPlacement(
                        "dp",
                        placement_kind,
                        coordinate,
                        _WORLD,
                    ),
                ),
                process_group=group,
                local_member=member,
                owner=owner,
            )
        )
    return ShardingManifest(tuple(shards)), tuple(shards)


def _optimizer(parameter):
    from gefen import GefenMuonHybrid

    return GefenMuonHybrid(
        [],
        [("backup.weight", parameter)],
        lr=2e-3,
        betas=(0.8, 0.95),
        eps=1e-6,
        weight_decay=0.03,
        fused=False,
        backup_optimizer="adamw",
    )


def _reference(parameter):
    return torch.optim.AdamW(
        [parameter],
        lr=2e-3,
        betas=(0.8, 0.95),
        eps=1e-6,
        weight_decay=0.03,
        fused=False,
    )


def _build(rank, full_value, layout_name):
    from torch import nn

    from gefen import ParameterRebinding

    manifest, shards = _manifest(layout_name)
    shard = shards[rank]
    old_parameter = nn.Parameter(full_value.clone())
    if layout_name == "flattened":
        start = shard.logical_slice.flat_offset
        target = nn.Parameter(full_value.reshape(-1)[start : start + shard.logical_slice.length].clone())
    elif _MEMBERS[rank] == _OWNER:
        target = nn.Parameter(full_value.clone())
    else:
        target = None
    optimizer = _optimizer(old_parameter)
    optimizer.post_sharding(
        (ParameterRebinding(old_parameter, target, shard),),
        manifest=manifest,
    )
    return optimizer, target, shard


def _training_support(contract, layout):
    matches = [item for item in contract.capabilities.training if item.layout is layout]
    assert len(matches) == 1
    return matches[0]


def _assert_contract(optimizer, layout_name):
    from gefen import (
        CheckpointTransport,
        ParameterLayout,
        Precision,
        ProcessGroupScope,
    )

    contract = optimizer.optimizer_contract()
    assert contract.capabilities.precisions == frozenset(
        {Precision.FLOAT16, Precision.BFLOAT16, Precision.FLOAT32, Precision.FLOAT64}
    )
    assert {item.layout for item in contract.capabilities.training} == {
        ParameterLayout.REPLICATED,
        ParameterLayout.FLATTENED_ELEMENT_SHARD,
        ParameterLayout.WHOLE_PARAMETER_OWNER,
        ParameterLayout.DTENSOR_1D_DEFAULT_WORLD,
    }
    flattened = _training_support(contract, ParameterLayout.FLATTENED_ELEMENT_SHARD)
    assert flattened.process_group_scope is ProcessGroupScope.NONE
    assert not flattened.requires_complete_parameter_storage
    assert not flattened.requires_complete_logical_matrix
    assert not flattened.requires_post_step_parameter_sync
    owner = _training_support(contract, ParameterLayout.WHOLE_PARAMETER_OWNER)
    assert owner.process_group_scope is ProcessGroupScope.ADAPTER_DEFINED
    assert owner.requires_complete_parameter_storage
    assert not owner.requires_complete_logical_matrix
    assert owner.requires_post_step_parameter_sync

    expected_layout = (
        ParameterLayout.FLATTENED_ELEMENT_SHARD
        if layout_name == "flattened"
        else ParameterLayout.WHOLE_PARAMETER_OWNER
    )
    native = [
        item
        for item in contract.capabilities.checkpoints
        if item.transport is CheckpointTransport.COMPOSITE_NATIVE
    ]
    assert len(native) == 1
    assert native[0].same_topology == frozenset({ParameterLayout.REPLICATED, expected_layout})
    assert not native[0].topology_changing
    assert native[0].process_group_scope is ProcessGroupScope.NONE
    assert not native[0].requires_collective
    assert native[0].atomic_load
    assert all(
        item.transport is not CheckpointTransport.CANONICAL_GLOBAL
        for item in contract.capabilities.checkpoints
    )


def _assign_gradient(target, full_gradient, shard):
    if target is None:
        return
    if target.ndim == 1:
        start = shard.logical_slice.flat_offset
        target.grad = full_gradient.reshape(-1)[start : start + shard.logical_slice.length].clone()
    else:
        target.grad = full_gradient.clone()


def _step(optimizer, target, full_gradient, shard):
    _assign_gradient(target, full_gradient, shard)
    optimizer.step()
    optimizer.zero_grad()


def _tree_exact(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and left.dtype == right.dtype
            and left.layout == right.layout
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(_tree_exact(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            _tree_exact(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


def _assert_reference_exact(optimizer, target, shard, reference_optimizer, reference_parameter):
    if target is None:
        assert optimizer.backup.param_groups[0]["params"] == []
        assert not optimizer.backup.state
        return
    reference_value = reference_parameter.detach()
    if target.ndim == 1:
        start = shard.logical_slice.flat_offset
        reference_value = reference_value.reshape(-1)[start : start + shard.logical_slice.length]
    assert torch.equal(target.detach(), reference_value)

    state = optimizer.backup.state[target]
    reference_state = reference_optimizer.state[reference_parameter]
    assert torch.equal(state["step"], reference_state["step"])
    for name in ("exp_avg", "exp_avg_sq"):
        expected = reference_state[name]
        if target.ndim == 1:
            start = shard.logical_slice.flat_offset
            expected = expected.reshape(-1)[start : start + shard.logical_slice.length]
        assert torch.equal(state[name], expected)


def _corrupt_checkpoint(checkpoint, target):
    invalid = copy.deepcopy(checkpoint)
    backup = invalid["backup"]
    if target is None:
        backup["param_groups"].append(copy.deepcopy(backup["param_groups"][0]))
    else:
        (parameter_id,) = backup["state"]
        backup["state"][parameter_id]["exp_avg"] = torch.zeros(1, dtype=target.dtype)
    return invalid


def _exercise_case(rank, dtype, layout_name):
    full_value = torch.linspace(-0.75, 0.625, 6, dtype=dtype).reshape(_SHAPE)
    first_gradient = torch.tensor(
        [[0.5, -0.25, 0.75], [-1.0, 0.375, -0.625]],
        dtype=dtype,
    )
    second_gradient = torch.tensor(
        [[-0.5, 0.125, 0.25], [0.875, -0.75, 0.625]],
        dtype=dtype,
    )

    optimizer, target, shard = _build(rank, full_value, layout_name)
    _assert_contract(optimizer, layout_name)
    reference_parameter = torch.nn.Parameter(full_value.clone())
    reference_optimizer = _reference(reference_parameter)

    _step(optimizer, target, first_gradient, shard)
    reference_parameter.grad = first_gradient.clone()
    reference_optimizer.step()
    reference_optimizer.zero_grad()
    _assert_reference_exact(
        optimizer,
        target,
        shard,
        reference_optimizer,
        reference_parameter,
    )
    checkpoint = copy.deepcopy(optimizer.state_dict())

    resumed, resumed_target, resumed_shard = _build(
        rank,
        reference_parameter.detach(),
        layout_name,
    )
    resumed.load_state_dict(checkpoint)
    assert _tree_exact(resumed.state_dict(), optimizer.state_dict())
    if target is not None:
        assert torch.equal(resumed_target, target)

    _step(optimizer, target, second_gradient, shard)
    _step(resumed, resumed_target, second_gradient, resumed_shard)
    reference_parameter.grad = second_gradient.clone()
    reference_optimizer.step()
    reference_optimizer.zero_grad()
    assert _tree_exact(resumed.state_dict(), optimizer.state_dict())
    if target is not None:
        assert torch.equal(resumed_target, target)
    _assert_reference_exact(
        optimizer,
        target,
        shard,
        reference_optimizer,
        reference_parameter,
    )
    _assert_reference_exact(
        resumed,
        resumed_target,
        resumed_shard,
        reference_optimizer,
        reference_parameter,
    )

    invalid = _corrupt_checkpoint(checkpoint, resumed_target)
    child_snapshot = deep_state_snapshot(resumed.backup)
    state_before = copy.deepcopy(resumed.state_dict())
    target_before = resumed_target.detach().clone() if resumed_target is not None else None
    finalized_before = (
        resumed._hybrid_sharding_manifest,
        resumed._hybrid_local_shard_bindings,
        resumed._hybrid_shard_bindings,
        resumed._hybrid_fqn_roles,
        resumed._hybrid_finalized_slots,
    )
    try:
        resumed.load_state_dict(invalid)
    except (RuntimeError, TypeError, ValueError):
        pass
    else:
        raise AssertionError("corrupt AdamW checkpoint was accepted")
    assert_deep_state_snapshot(resumed.backup, child_snapshot)
    assert _tree_exact(resumed.state_dict(), state_before)
    if resumed_target is not None:
        assert torch.equal(resumed_target, target_before)
    finalized_after = (
        resumed._hybrid_sharding_manifest,
        resumed._hybrid_local_shard_bindings,
        resumed._hybrid_shard_bindings,
        resumed._hybrid_fqn_roles,
        resumed._hybrid_finalized_slots,
    )
    assert all(after is before for after, before in zip(finalized_after, finalized_before))


def _worker(rank, init_file, layout_name, result_queue):
    try:
        torch.set_num_threads(1)
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=_WORLD,
            timeout=timedelta(seconds=120),
        )
        completed = []
        for dtype_name, dtype in _DTYPES:
            _exercise_case(rank, dtype, layout_name)
            completed.append(dtype_name)
        result_queue.put({"rank": rank, "completed": tuple(completed)})
    except BaseException:
        result_queue.put({"rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run(layout_name):
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    directory = tempfile.mkdtemp(prefix="gefen-hybrid-adamw-{}-".format(layout_name))
    init_file = os.path.join(directory, "store")
    processes = [
        context.Process(
            target=_worker,
            args=(rank, init_file, layout_name, result_queue),
        )
        for rank in range(_WORLD)
    ]
    results = []
    try:
        for process in processes:
            process.start()
        try:
            for _ in processes:
                results.append(result_queue.get(timeout=180))
        except queue_module.Empty:
            pass
        for process in processes:
            process.join(timeout=10)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        result_queue.close()
        result_queue.join_thread()
        shutil.rmtree(directory, ignore_errors=True)
    assert len(results) == _WORLD, (results, [process.exitcode for process in processes])
    assert all(process.exitcode == 0 for process in processes), results
    return results


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="distributed Hybrid AdamW rebinding requires Gloo",
)
@pytest.mark.parametrize("layout_name", ["flattened", "whole_owner"])
def test_hybrid_adamw_sharded_native_continuation_is_exact_and_atomic(layout_name):
    results = _run(layout_name)
    assert all("traceback" not in result for result in results), results
    expected = tuple(name for name, _dtype in _DTYPES)
    assert all(result["completed"] == expected for result in results), results

"""Two-rank CPU coverage for Gefen-family 1-D DTensor rebinding."""

from __future__ import annotations

import copy
from datetime import timedelta
import multiprocessing as mp
import os
import queue as queue_module
import shutil
import tempfile
import traceback

import pytest
import torch
import torch.distributed as dist


_WORLD = 2
_MEMBERS = ("rank:0", "rank:1")
_STATE_KEYS = {
    "name",
    "automatic_period",
    "step",
    "m_codebook",
    "m_magnitude",
    "vmean",
    "vmean_step",
    "v_row",
    "v_col",
    "factored_step",
}


def _local(value):
    if hasattr(value, "to_local"):
        value = value.to_local()
    if hasattr(value, "wait"):
        value = value.wait()
    return value


def _raw_bytes(value):
    value = _local(value).detach().contiguous()
    return value.reshape(-1).view(torch.uint8).clone()


def _tensor_unchanged(value, snapshot):
    local = _local(value)
    return (
        local.dtype == snapshot[0]
        and local.layout == snapshot[1]
        and tuple(local.shape) == snapshot[2]
        and torch.equal(_raw_bytes(local), snapshot[3])
    )


def _tensor_snapshot(value):
    local = _local(value)
    return local.dtype, local.layout, tuple(local.shape), _raw_bytes(local)


def _assert_value_exact(actual, expected, path="value"):
    if torch.is_tensor(expected):
        assert torch.is_tensor(actual), path
        actual = _local(actual)
        expected = _local(expected)
        assert actual.dtype == expected.dtype, path
        assert actual.layout == expected.layout, path
        assert tuple(actual.shape) == tuple(expected.shape), path
        assert torch.equal(_raw_bytes(actual), _raw_bytes(expected)), path
        return
    assert type(actual) is type(expected), path
    if isinstance(expected, dict):
        assert set(actual) == set(expected), path
        for key in expected:
            _assert_value_exact(actual[key], expected[key], "{}.{}".format(path, key))
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected), path
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_value_exact(actual_item, expected_item, "{}[{}]".format(path, index))
    else:
        assert actual == expected, path


def _persistent_snapshot(optimizer, parameter):
    return {
        "global_step": optimizer._gefen_global_step,
        "codebook": None if optimizer._gefen_codebook is None else optimizer._gefen_codebook.detach().clone(),
        "state": {
            key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
            for key, value in optimizer.state[parameter].items()
            if key in _STATE_KEYS
        },
    }


def _assert_optimizer_exact(actual_optimizer, actual_parameter, expected_optimizer, expected_parameter, path):
    _assert_value_exact(
        _persistent_snapshot(actual_optimizer, actual_parameter),
        _persistent_snapshot(expected_optimizer, expected_parameter),
        path,
    )


def _region(shape, coordinate, parts, dimension):
    from gefen import LogicalRegion

    global_length = shape[dimension]
    chunk = (global_length + parts - 1) // parts
    offset = min(coordinate * chunk, global_length)
    length = max(0, min(global_length, offset + chunk) - offset)
    offsets = [0] * len(shape)
    lengths = list(shape)
    offsets[dimension] = offset
    lengths[dimension] = length
    return LogicalRegion(tuple(offsets), tuple(lengths))


def _manifest(
    fqn,
    shape,
    *,
    kind,
    dimension=None,
    members=_MEMBERS,
    semantic_name="data_parallel",
    layout=None,
):
    from gefen import (
        LogicalRegion,
        LogicalSlice,
        ParameterIdentity,
        ParameterLayout,
        PlacementKind,
        ProcessGroupIdentity,
        ShardIdentity,
        ShardPlacement,
        ShardingManifest,
    )

    identity = ParameterIdentity(fqn, shape)
    group = ProcessGroupIdentity(semantic_name, members)
    if layout is None:
        layout = ParameterLayout.DTENSOR_1D_DEFAULT_WORLD
    shards = []
    for coordinate, member in enumerate(members):
        if layout is ParameterLayout.REPLICATED:
            logical_slice = LogicalSlice.full(identity)
        elif kind is PlacementKind.REPLICATE:
            logical_slice = LogicalRegion.full(identity)
        else:
            logical_slice = _region(shape, coordinate, len(members), dimension)
        shards.append(
            ShardIdentity(
                identity,
                layout,
                logical_slice,
                placements=(
                    ShardPlacement(
                        "dp",
                        kind,
                        coordinate,
                        len(members),
                        parameter_dimension=dimension,
                    ),
                ),
                process_group=group,
                local_member=member,
            )
        )
    return ShardingManifest(tuple(shards)), tuple(shards)


def _combined_manifest(*manifests):
    from gefen import ShardingManifest

    return ShardingManifest(tuple(shard for manifest in manifests for shard in manifest.shards))


def _optimizer(parameter, name="legacy.weight"):
    from gefen import Gefen

    return Gefen(
        [(name, parameter)],
        lr=3e-3,
        betas=(0.8, 0.97),
        eps=2e-8,
        weight_decay=0.02,
        fused=False,
        deterministic=True,
        force_2d_period_one=True,
        factored_v_2d=False,
    )


def _two_parameter_optimizer(first, second):
    from gefen import Gefen

    return Gefen(
        [("legacy.first", first), ("legacy.second", second)],
        lr=3e-3,
        betas=(0.8, 0.97),
        eps=2e-8,
        weight_decay=0.02,
        fused=False,
        deterministic=True,
        force_2d_period_one=True,
        factored_v_2d=False,
    )


def _bind(optimizer, old_parameter, target, shard, manifest):
    from gefen import ParameterRebinding

    optimizer.post_sharding(
        (ParameterRebinding(old_parameter, target, shard),),
        manifest=manifest,
    )


def _model(parameter):
    from torch import nn

    model = nn.Module()
    model.register_parameter("weight", parameter)
    return model


def _local_slice(value, region):
    slices = tuple(slice(offset, offset + length) for offset, length in zip(region.offsets, region.lengths))
    return value[slices].clone()


def _placement(kind, dimension):
    from torch.distributed.tensor import Replicate, Shard

    return Replicate() if kind == "replicate" else Shard(dimension)


def _placement_kind(kind):
    from gefen import PlacementKind

    return PlacementKind.REPLICATE if kind == "replicate" else PlacementKind.DIMENSION_SHARD


def _contract_is_exact(optimizer):
    from gefen import CheckpointTransport, ParameterLayout, ProcessGroupScope

    contract = optimizer.optimizer_contract()
    rank_local = [
        support
        for support in contract.capabilities.checkpoints
        if support.transport is CheckpointTransport.PYTORCH_RANK_LOCAL
    ]
    native_dtensor = [
        support
        for support in contract.capabilities.checkpoints
        if support.transport is CheckpointTransport.NATIVE_OPTIMIZER
        and ParameterLayout.DTENSOR_1D_DEFAULT_WORLD in support.same_topology
    ]
    return (
        contract.capabilities.canonical_parameter_fqns
        and contract.capabilities.stable_shard_identity
        and contract.capabilities.shard_rebinding
        and contract.capabilities.post_sharding
        and optimizer.codebook_process_group_binding() is None
        and len(rank_local) == 1
        and rank_local[0].same_topology == frozenset({ParameterLayout.DTENSOR_1D_DEFAULT_WORLD})
        and rank_local[0].topology_changing == frozenset()
        and rank_local[0].process_group_scope is ProcessGroupScope.DEFAULT_WORLD
        and rank_local[0].requires_collective
        and rank_local[0].atomic_load
        and not native_dtensor
    )


def _run_valid_case(rank, mesh, *, shape, kind, dimension):
    from torch import nn
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_optimizer_state_dict,
        set_optimizer_state_dict,
    )
    from torch.distributed.tensor import distribute_tensor

    placement = _placement(kind, dimension)
    placement_kind = _placement_kind(kind)
    full = torch.linspace(-0.75, 0.5, torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
    first_gradient = torch.linspace(0.6, -0.4, full.numel(), dtype=torch.float32).reshape(shape)
    second_gradient = torch.arange(full.numel(), dtype=torch.float32).reshape(shape).cos()
    manifest, shards = _manifest(
        "Model.Weight",
        shape,
        kind=placement_kind,
        dimension=dimension,
    )
    shard = shards[rank]

    old = nn.Parameter(full.clone())
    target = nn.Parameter(distribute_tensor(full.clone(), mesh, [placement]))
    optimizer = _optimizer(old)
    _bind(optimizer, old, target, shard, manifest)
    model = _model(target)
    assert optimizer.param_groups[0]["params"] == [target]
    assert old not in optimizer.state
    assert optimizer.shard_identity(target) == shard
    assert optimizer.shard_bindings() == ((target, shard),)
    assert optimizer.sharding_manifest() == manifest
    assert _contract_is_exact(optimizer)

    local_initial = _local(target).detach().clone()
    local_gradient = (
        first_gradient.clone() if kind == "replicate" else _local_slice(first_gradient, shard.logical_region)
    )
    local_reference_parameter = nn.Parameter(local_initial)
    local_reference = _optimizer(local_reference_parameter)

    target.grad = distribute_tensor(first_gradient.clone(), mesh, [placement])
    local_reference_parameter.grad = local_gradient
    optimizer.step()
    local_reference.step()
    assert torch.equal(_local(target.detach()), local_reference_parameter.detach())
    _assert_optimizer_exact(optimizer, target, local_reference, local_reference_parameter, "first local oracle")

    native_checkpoint = copy.deepcopy(optimizer.state_dict())
    public_checkpoint = get_optimizer_state_dict(
        model,
        optimizer,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    current_full = target.detach().full_tensor().clone()

    native_old = nn.Parameter(current_full.clone())
    native_target = nn.Parameter(distribute_tensor(current_full.clone(), mesh, [placement]))
    native_optimizer = _optimizer(native_old)
    _bind(native_optimizer, native_old, native_target, shard, manifest)
    native_optimizer.load_state_dict(native_checkpoint)

    public_old = nn.Parameter(current_full.clone())
    public_target = nn.Parameter(distribute_tensor(current_full.clone(), mesh, [placement]))
    public_optimizer = _optimizer(public_old)
    _bind(public_optimizer, public_old, public_target, shard, manifest)
    public_model = _model(public_target)
    set_optimizer_state_dict(
        public_model,
        public_optimizer,
        public_checkpoint if rank == 0 else {},
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
    )

    assert torch.equal(_local(native_target.detach()), _local(target.detach()))
    assert torch.equal(_local(public_target.detach()), _local(target.detach()))
    _assert_optimizer_exact(native_optimizer, native_target, optimizer, target, "native restore")
    _assert_optimizer_exact(public_optimizer, public_target, optimizer, target, "public DCP restore")

    second_local_gradient = (
        second_gradient.clone() if kind == "replicate" else _local_slice(second_gradient, shard.logical_region)
    )
    target.grad = distribute_tensor(second_gradient.clone(), mesh, [placement])
    native_target.grad = distribute_tensor(second_gradient.clone(), mesh, [placement])
    public_target.grad = distribute_tensor(second_gradient.clone(), mesh, [placement])
    local_reference_parameter.grad = second_local_gradient
    optimizer.step()
    native_optimizer.step()
    public_optimizer.step()
    local_reference.step()

    for label, candidate_optimizer, candidate_parameter in (
        ("source continuation", optimizer, target),
        ("native continuation", native_optimizer, native_target),
        ("public DCP continuation", public_optimizer, public_target),
    ):
        assert torch.equal(_local(candidate_parameter.detach()), local_reference_parameter.detach()), label
        _assert_optimizer_exact(
            candidate_optimizer, candidate_parameter, local_reference, local_reference_parameter, label
        )

    empty = shard.logical_region.numel == 0
    if empty:
        assert set(_persistent_snapshot(optimizer, target)["state"]) == {"name"}
        assert optimizer._gefen_codebook is None
    return {
        "shape": shape,
        "kind": kind,
        "dimension": dimension,
        "empty": empty,
    }


def _valid_worker(rank, init_file, result_queue):
    from torch.distributed.tensor import init_device_mesh

    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=_WORLD,
            timeout=timedelta(seconds=90),
        )
        mesh = init_device_mesh("cpu", (_WORLD,), mesh_dim_names=("dp",))
        cases = [
            ((5, 4), "shard", 0),
            ((1, 4), "shard", 0),
            ((4, 5), "shard", 1),
            ((4, 1), "shard", 1),
            ((5, 4), "replicate", None),
        ]
        results = [
            _run_valid_case(rank, mesh, shape=shape, kind=kind, dimension=dimension) for shape, kind, dimension in cases
        ]
        result_queue.put({"rank": rank, "cases": results})
    except BaseException:
        result_queue.put({"rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _scoped_distributed_muon_contract_worker(rank, init_file, result_queue):
    from gefen import (
        CheckpointTransport,
        CodebookProcessGroupBinding,
        GefenMuon,
        ParameterLayout,
        ParameterRebinding,
        PlacementKind,
        ProcessGroupScope,
    )
    from torch import nn
    from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=_WORLD,
            timeout=timedelta(seconds=90),
        )
        mesh = init_device_mesh("cpu", (_WORLD,), mesh_dim_names=("dp",))
        full = torch.linspace(-0.75, 0.5, 16, dtype=torch.float32).reshape(4, 4)
        gradient = torch.arange(16, dtype=torch.float32).reshape(4, 4).sin()
        manifest, shards = _manifest(
            "Model.Weight",
            full.shape,
            kind=PlacementKind.DIMENSION_SHARD,
            dimension=0,
        )

        def build(values):
            old = nn.Parameter(values.clone())
            target = nn.Parameter(distribute_tensor(values.clone(), mesh, [Shard(0)]))
            optimizer = GefenMuon(
                [("legacy.weight", old)],
                lr=3e-3,
                fused=False,
                deterministic=True,
                sharded_mode="distributed",
            )
            optimizer.post_sharding(
                (ParameterRebinding(old, target, shards[rank]),),
                manifest=manifest,
                codebook_process_group=CodebookProcessGroupBinding(
                    shards[rank].process_group,
                    _MEMBERS[rank],
                    dist.group.WORLD,
                    torch.device("cpu"),
                ),
            )
            return optimizer, target

        source, source_parameter = build(full)
        supports = [
            support
            for support in source.optimizer_contract().capabilities.checkpoints
            if support.transport is CheckpointTransport.NATIVE_OPTIMIZER
            and ParameterLayout.DTENSOR_1D_DEFAULT_WORLD in support.same_topology
        ]
        contract_exact = (
            len(supports) == 1
            and supports[0].same_topology == frozenset({ParameterLayout.DTENSOR_1D_DEFAULT_WORLD})
            and not supports[0].topology_changing
            and not supports[0].topology_change_kinds
            and supports[0].process_group_scope is ProcessGroupScope.INFERRED_DEVICE_MESH
            and supports[0].requires_collective
            and supports[0].atomic_load
        )

        source_parameter.grad = distribute_tensor(gradient.clone(), mesh, [Shard(0)])
        source.step()
        checkpoint = copy.deepcopy(source.state_dict())
        current = source_parameter.detach().full_tensor().clone()
        resumed, resumed_parameter = build(current)
        resumed.load_state_dict(checkpoint)
        common_state_restored = resumed._gefen_global_step == source._gefen_global_step and torch.equal(
            resumed._gefen_codebook, source._gefen_codebook
        )
        owner_state_restored = True
        if rank == 0:
            try:
                _assert_optimizer_exact(
                    resumed,
                    resumed_parameter,
                    source,
                    source_parameter,
                    "scoped distributed Muon native restore",
                )
            except AssertionError:
                owner_state_restored = False

        continuation_gradient = gradient.cos()
        source_parameter.grad = distribute_tensor(continuation_gradient.clone(), mesh, [Shard(0)])
        resumed_parameter.grad = distribute_tensor(continuation_gradient.clone(), mesh, [Shard(0)])
        source.step()
        resumed.step()
        continued_exactly = torch.equal(
            _local(source_parameter.detach()),
            _local(resumed_parameter.detach()),
        )
        if rank == 0:
            try:
                _assert_optimizer_exact(
                    resumed,
                    resumed_parameter,
                    source,
                    source_parameter,
                    "scoped distributed Muon native continuation",
                )
            except AssertionError:
                continued_exactly = False
        result_queue.put(
            {
                "rank": rank,
                "contract_exact": contract_exact,
                "common_state_restored": common_state_restored,
                "owner_state_restored": owner_state_restored,
                "continued_exactly": continued_exactly,
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _rejection_result(optimizer, targets, operation):
    from _state_snapshot import assert_deep_state_snapshot, deep_state_snapshot

    if torch.is_tensor(targets):
        targets = (targets,)
    else:
        targets = tuple(targets)
    optimizer_snapshot = deep_state_snapshot(optimizer)
    target_snapshots = tuple(_tensor_snapshot(target) for target in targets)
    error = None
    try:
        operation()
    except (TypeError, ValueError, RuntimeError) as exc:
        error = "{}: {}".format(type(exc).__name__, exc)
    try:
        assert_deep_state_snapshot(optimizer, optimizer_snapshot)
        unchanged = all(_tensor_unchanged(target, snapshot) for target, snapshot in zip(targets, target_snapshots))
    except AssertionError:
        unchanged = False
    return {"rejected": error is not None, "unchanged": unchanged, "error": error}


def _run_rejections(rank, mesh):
    from gefen import ParameterLayout, ParameterRebinding, PlacementKind
    from torch import nn
    from torch.distributed.tensor import DeviceMesh, Replicate, Shard, distribute_tensor

    full = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    results = {}

    manifest, shards = _manifest(
        "Model.Weight",
        full.shape,
        kind=PlacementKind.REPLICATE,
        layout=ParameterLayout.REPLICATED,
    )
    old = nn.Parameter(full.clone())
    target = nn.Parameter(distribute_tensor(full.clone(), mesh, [Shard(0)]))
    optimizer = _optimizer(old)
    results["wrong_layout"] = _rejection_result(
        optimizer,
        target,
        lambda: _bind(optimizer, old, target, shards[rank], manifest),
    )

    manifest, shards = _manifest(
        "Model.Weight",
        full.shape,
        kind=PlacementKind.DIMENSION_SHARD,
        dimension=0,
    )
    old = nn.Parameter(full.clone())
    target = nn.Parameter(full[rank * 2 : (rank + 1) * 2].clone())
    optimizer = _optimizer(old)
    results["wrong_type"] = _rejection_result(
        optimizer,
        target,
        lambda: _bind(optimizer, old, target, shards[rank], manifest),
    )

    three_members = ("rank:0", "rank:1", "rank:2")
    manifest, shards = _manifest(
        "Model.Weight",
        full.shape,
        kind=PlacementKind.REPLICATE,
        members=three_members,
    )
    old = nn.Parameter(full.clone())
    target = nn.Parameter(distribute_tensor(full.clone(), mesh, [Replicate()]))
    optimizer = _optimizer(old)
    results["wrong_world"] = _rejection_result(
        optimizer,
        target,
        lambda: _bind(optimizer, old, target, shards[rank], manifest),
    )

    manifest, shards = _manifest(
        "Model.Weight",
        full.shape,
        kind=PlacementKind.DIMENSION_SHARD,
        dimension=0,
    )
    old = nn.Parameter(full.clone())
    target = nn.Parameter(distribute_tensor(full.clone(), mesh, [Shard(1)]))
    optimizer = _optimizer(old)
    results["wrong_placement"] = _rejection_result(
        optimizer,
        target,
        lambda: _bind(optimizer, old, target, shards[rank], manifest),
    )

    old = nn.Parameter(full.clone())
    target = nn.Parameter(distribute_tensor(full.clone(), mesh, [Shard(0)]))
    optimizer = _optimizer(old)
    results["wrong_coordinate"] = _rejection_result(
        optimizer,
        target,
        lambda: _bind(optimizer, old, target, shards[1 - rank], manifest),
    )

    reverse_mesh = DeviceMesh("cpu", [1, 0], mesh_dim_names=("reverse",))
    old = nn.Parameter(full.clone())
    target = nn.Parameter(distribute_tensor(full.clone(), reverse_mesh, [Shard(0)]))
    optimizer = _optimizer(old)
    results["wrong_mesh"] = _rejection_result(
        optimizer,
        target,
        lambda: _bind(optimizer, old, target, shards[rank], manifest),
    )

    first_manifest, first_shards = _manifest(
        "Model.First",
        full.shape,
        kind=PlacementKind.DIMENSION_SHARD,
        dimension=0,
        semantic_name="shared_mesh",
    )
    second_manifest, second_shards = _manifest(
        "Model.Second",
        full.shape,
        kind=PlacementKind.DIMENSION_SHARD,
        dimension=0,
        semantic_name="shared_mesh",
    )
    combined = _combined_manifest(first_manifest, second_manifest)
    first_old = nn.Parameter(full.clone())
    second_old = nn.Parameter(full.neg().clone())
    first_target = nn.Parameter(distribute_tensor(full.clone(), mesh, [Shard(0)]))
    second_target = nn.Parameter(distribute_tensor(full.neg().clone(), reverse_mesh, [Shard(0)]))
    mixed = _two_parameter_optimizer(first_old, second_old)
    results["shared_mesh"] = _rejection_result(
        mixed,
        (first_target, second_target),
        lambda: mixed.post_sharding(
            (
                ParameterRebinding(first_old, first_target, first_shards[rank]),
                ParameterRebinding(second_old, second_target, second_shards[rank]),
            ),
            manifest=combined,
        ),
    )
    return results


def _rejection_worker(rank, init_file, result_queue):
    from torch.distributed.tensor import init_device_mesh

    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=_WORLD,
            timeout=timedelta(seconds=90),
        )
        mesh = init_device_mesh("cpu", (_WORLD,), mesh_dim_names=("dp",))
        result_queue.put({"rank": rank, "results": _run_rejections(rank, mesh)})
    except BaseException:
        result_queue.put({"rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _scoped_replicated_dtensor_worker(rank, init_file, result_queue):
    from gefen import (
        CodebookProcessGroupBinding,
        ParameterRebinding,
        PlacementKind,
    )
    from gefen.gefen import learn_gefen_exact_codebook_from_grad_periods
    from torch import nn
    from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor, init_device_mesh

    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=_WORLD,
            timeout=timedelta(seconds=90),
        )
        mesh = init_device_mesh("cpu", (_WORLD,), mesh_dim_names=("dp",))
        replicated_manifest, replicated_shards = _manifest(
            "Model.Replicated",
            (4,),
            kind=PlacementKind.REPLICATE,
            semantic_name="scoped_dtensor",
        )
        sharded_manifest, sharded_shards = _manifest(
            "Model.Sharded",
            (4,),
            kind=PlacementKind.DIMENSION_SHARD,
            dimension=0,
            semantic_name="scoped_dtensor",
        )
        manifest = _combined_manifest(replicated_manifest, sharded_manifest)
        binding = CodebookProcessGroupBinding(
            replicated_shards[rank].process_group,
            _MEMBERS[rank],
            dist.group.WORLD,
            torch.device("cpu"),
        )

        replicated_old = nn.Parameter(torch.zeros(4))
        sharded_old = nn.Parameter(torch.zeros(4))
        replicated = nn.Parameter(distribute_tensor(torch.zeros(4), mesh, [Replicate()]))
        sharded = nn.Parameter(distribute_tensor(torch.zeros(4), mesh, [Shard(0)]))
        optimizer = _two_parameter_optimizer(replicated_old, sharded_old)
        optimizer.post_sharding(
            (
                ParameterRebinding(replicated_old, replicated, replicated_shards[rank]),
                ParameterRebinding(sharded_old, sharded, sharded_shards[rank]),
            ),
            manifest=manifest,
            codebook_process_group=binding,
        )
        optimizer._resolve_automatic_period = lambda *args: 2
        replicated_gradients = (
            torch.tensor([-4.0, -1.0, 2.0, 8.0]),
            torch.tensor([-9.0, 3.0, 5.0, 7.0]),
        )
        sharded_gradients = (
            torch.tensor([-3.0, 6.0]),
            torch.tensor([2.0, 11.0]),
        )
        replicated.grad = DTensor.from_local(
            replicated_gradients[rank].clone(),
            mesh,
            [Replicate()],
            run_check=False,
        )
        sharded.grad = DTensor.from_local(
            sharded_gradients[rank].clone(),
            mesh,
            [Shard(0)],
            run_check=False,
        )
        initialized = optimizer.initialize_codebook()
        intended_oracle = learn_gefen_exact_codebook_from_grad_periods(
            grad_periods=(
                (
                    "Model.Replicated",
                    replicated_gradients[0],
                    2,
                    replicated_gradients[0],
                ),
                *tuple(("Model.Sharded", gradient, 2, gradient) for gradient in sharded_gradients),
            ),
            codebook_device=torch.device("cpu"),
            num_codebooks=256,
            force_endpoints=True,
            verbose=False,
            compute_mse_logging=False,
            use_fused_histogram=False,
        )
        duplicated_oracle = learn_gefen_exact_codebook_from_grad_periods(
            grad_periods=(
                *tuple(("Model.Replicated", gradient, 2, gradient) for gradient in replicated_gradients),
                *tuple(("Model.Sharded", gradient, 2, gradient) for gradient in sharded_gradients),
            ),
            codebook_device=torch.device("cpu"),
            num_codebooks=256,
            force_endpoints=True,
            verbose=False,
            compute_mse_logging=False,
            use_fused_histogram=False,
        )
        logical_once = torch.equal(optimizer._gefen_codebook, intended_oracle)
        duplicate_would_differ = not torch.equal(intended_oracle, duplicated_oracle)
        gathered = [torch.empty_like(optimizer._gefen_codebook) for _ in range(_WORLD)]
        dist.all_gather(gathered, optimizer._gefen_codebook)
        agreed = all(torch.equal(item, gathered[0]) for item in gathered[1:])

        mismatch_replicated_old = nn.Parameter(torch.zeros(4))
        mismatch_sharded_old = nn.Parameter(torch.zeros(4))
        mismatch_replicated = nn.Parameter(distribute_tensor(torch.zeros(4), mesh, [Replicate()]))
        mismatch_sharded = nn.Parameter(distribute_tensor(torch.zeros(4), mesh, [Shard(0)]))
        mismatch_optimizer = _two_parameter_optimizer(
            mismatch_replicated_old,
            mismatch_sharded_old,
        )
        mismatch_optimizer.post_sharding(
            (
                ParameterRebinding(
                    mismatch_replicated_old,
                    mismatch_replicated,
                    replicated_shards[rank],
                ),
                ParameterRebinding(
                    mismatch_sharded_old,
                    mismatch_sharded,
                    sharded_shards[rank],
                ),
            ),
            manifest=manifest,
            codebook_process_group=binding,
        )

        def mismatched_period(_name, parameter, _gradient):
            if parameter is mismatch_replicated:
                return 2 if rank == 0 else 4
            return 2

        mismatch_optimizer._resolve_automatic_period = mismatched_period
        mismatch_replicated.grad = DTensor.from_local(
            replicated_gradients[0].clone(),
            mesh,
            [Replicate()],
            run_check=False,
        )
        mismatch_sharded.grad = DTensor.from_local(
            sharded_gradients[rank].clone(),
            mesh,
            [Shard(0)],
            run_check=False,
        )
        mismatch_before = (
            _persistent_snapshot(mismatch_optimizer, mismatch_replicated),
            _persistent_snapshot(mismatch_optimizer, mismatch_sharded),
        )
        try:
            mismatch_optimizer.initialize_codebook()
            period_mismatch_rejected = False
        except RuntimeError as exc:
            period_mismatch_rejected = "automatic periods" in str(exc)
        try:
            _assert_value_exact(
                (
                    _persistent_snapshot(mismatch_optimizer, mismatch_replicated),
                    _persistent_snapshot(mismatch_optimizer, mismatch_sharded),
                ),
                mismatch_before,
                "period mismatch atomicity",
            )
            mismatch_atomic = True
        except AssertionError:
            mismatch_atomic = False
        result_queue.put(
            {
                "rank": rank,
                "initialized": initialized,
                "logical_once": logical_once,
                "duplicate_would_differ": duplicate_would_differ,
                "agreed": agreed,
                "period_mismatch_rejected": period_mismatch_rejected,
                "mismatch_atomic": mismatch_atomic,
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _scoped_muon_codebook_worker(rank, init_file, result_queue):
    from gefen import (
        CodebookProcessGroupBinding,
        GefenMuon,
        ParameterRebinding,
        PlacementKind,
    )
    from gefen.gefen import learn_gefen_exact_codebook_from_grad_periods
    from torch import nn
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor, init_device_mesh

    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=_WORLD,
            timeout=timedelta(seconds=90),
        )
        mesh = init_device_mesh("cpu", (_WORLD,), mesh_dim_names=("dp",))
        shape = (16, 32)
        generator = torch.Generator().manual_seed(0)
        replicated_gradient = torch.randn(shape, generator=generator)
        sharded_gradient = torch.randn(shape, generator=generator).mul_(3.0)
        replicated_manifest, replicated_shards = _manifest(
            "Model.Replicated",
            shape,
            kind=PlacementKind.REPLICATE,
            semantic_name="scoped_muon_dtensor",
        )
        sharded_manifest, sharded_shards = _manifest(
            "Model.Sharded",
            shape,
            kind=PlacementKind.DIMENSION_SHARD,
            dimension=0,
            semantic_name="scoped_muon_dtensor",
        )
        manifest = _combined_manifest(replicated_manifest, sharded_manifest)

        def build(mode, *, scoped=True):
            replicated_old = nn.Parameter(torch.zeros(shape))
            sharded_old = nn.Parameter(torch.zeros(shape))
            replicated = nn.Parameter(distribute_tensor(torch.zeros(shape), mesh, [Replicate()]))
            sharded = nn.Parameter(distribute_tensor(torch.zeros(shape), mesh, [Shard(0)]))
            optimizer = GefenMuon(
                [
                    ("legacy.replicated", replicated_old),
                    ("legacy.sharded", sharded_old),
                ],
                lr=3e-3,
                fused=False,
                deterministic=True,
                sharded_mode=mode,
            )
            post_sharding_options = {}
            if scoped:
                post_sharding_options["codebook_process_group"] = (
                    CodebookProcessGroupBinding(
                        replicated_shards[rank].process_group,
                        _MEMBERS[rank],
                        dist.group.WORLD,
                        torch.device("cpu"),
                    )
                )
            optimizer.post_sharding(
                (
                    ParameterRebinding(
                        replicated_old,
                        replicated,
                        replicated_shards[rank],
                    ),
                    ParameterRebinding(
                        sharded_old,
                        sharded,
                        sharded_shards[rank],
                    ),
                ),
                manifest=manifest,
                **post_sharding_options,
            )
            return optimizer, replicated, sharded

        def set_gradients(replicated, sharded, *, sharded_active=True):
            replicated.grad = distribute_tensor(
                replicated_gradient.clone(), mesh, [Replicate()]
            )
            sharded_value = distribute_tensor(
                sharded_gradient.clone(), mesh, [Shard(0)]
            )
            sharded.grad = sharded_value if sharded_active else None

        def solve(items):
            return learn_gefen_exact_codebook_from_grad_periods(
                grad_periods=items,
                codebook_device=torch.device("cpu"),
                num_codebooks=256,
                force_endpoints=True,
                verbose=False,
                compute_mse_logging=False,
                use_fused_histogram=False,
            )

        def values_equal(actual, expected, label):
            try:
                _assert_value_exact(actual, expected, label)
            except AssertionError:
                return False
            return True

        exact, exact_replicated, exact_sharded = build("exact")
        exact._predict_period_from_grad_sq = lambda *args: 16
        set_gradients(exact_replicated, exact_sharded)
        exact_initialized = exact.initialize_codebook()
        exact_oracle = solve(
            (
                ("Model.Replicated", replicated_gradient, 16, replicated_gradient),
                ("Model.Sharded", sharded_gradient, 16, sharded_gradient),
            )
        )
        duplicated_shard_oracle = solve(
            (
                ("Model.Replicated", replicated_gradient, 16, replicated_gradient),
                ("Model.Sharded", sharded_gradient, 16, sharded_gradient),
                ("Model.Sharded", sharded_gradient, 16, sharded_gradient),
            )
        )
        checkpoint_hook = None
        if rank == 0:
            def reject_checkpoint(_optimizer):
                raise RuntimeError("scoped checkpoint hook failure")

            checkpoint_hook = exact.register_state_dict_pre_hook(
                reject_checkpoint
            )
        try:
            exact.state_dict()
            scoped_checkpoint_hook_rejected = False
        except RuntimeError as exc:
            scoped_checkpoint_hook_rejected = (
                "checkpoint state-dict pre-hook failed" in str(exc)
            )
        finally:
            if checkpoint_hook is not None:
                checkpoint_hook.remove()

        approximate, approximate_replicated, approximate_sharded = build("approx")
        approximate._predict_period_from_grad_sq = lambda *args: 16
        set_gradients(approximate_replicated, approximate_sharded)
        approximate_initialized = approximate.initialize_codebook()
        approximate_oracle = solve(
            (
                ("Model.Replicated", replicated_gradient, 16, replicated_gradient),
                (
                    "Model.Sharded",
                    sharded_gradient[: shape[0] // 2],
                    16,
                    sharded_gradient[: shape[0] // 2],
                ),
                (
                    "Model.Sharded",
                    sharded_gradient[shape[0] // 2 :],
                    16,
                    sharded_gradient[shape[0] // 2 :],
                ),
            )
        )

        mismatch, mismatch_replicated, mismatch_sharded = build("distributed")

        def mismatched_period(_name, parameter, _gradient):
            if parameter is mismatch_sharded:
                return 16 if rank == 0 else 32
            return 16

        mismatch._predict_period_from_grad_sq = mismatched_period
        set_gradients(mismatch_replicated, mismatch_sharded)
        mismatch_before = (
            _persistent_snapshot(mismatch, mismatch_replicated),
            _persistent_snapshot(mismatch, mismatch_sharded),
        )
        try:
            mismatch.initialize_codebook()
            period_mismatch_rejected = False
        except RuntimeError as exc:
            period_mismatch_rejected = "automatic periods" in str(exc)
        mismatch_after = (
            _persistent_snapshot(mismatch, mismatch_replicated),
            _persistent_snapshot(mismatch, mismatch_sharded),
        )

        presence, presence_replicated, presence_sharded = build("exact")
        presence._predict_period_from_grad_sq = lambda *args: 16
        set_gradients(
            presence_replicated,
            presence_sharded,
            sharded_active=rank == 0,
        )
        presence_before = (
            _persistent_snapshot(presence, presence_replicated),
            _persistent_snapshot(presence, presence_sharded),
        )
        try:
            presence.initialize_codebook()
            initialize_presence_rejected = False
        except RuntimeError as exc:
            initialize_presence_rejected = "identical gradient presence" in str(exc)
        presence_after = (
            _persistent_snapshot(presence, presence_replicated),
            _persistent_snapshot(presence, presence_sharded),
        )

        refresh, refresh_replicated, refresh_sharded = build("exact")
        refresh._predict_period_from_grad_sq = lambda *args: 16
        set_gradients(refresh_replicated, refresh_sharded)
        refresh.initialize_codebook()
        refresh_before = (
            _persistent_snapshot(refresh, refresh_replicated),
            _persistent_snapshot(refresh, refresh_sharded),
        )
        set_gradients(
            refresh_replicated,
            refresh_sharded,
            sharded_active=rank == 0,
        )
        try:
            refresh.refresh_codebook()
            refresh_presence_rejected = False
        except RuntimeError as exc:
            refresh_presence_rejected = "identical gradient presence" in str(exc)
        refresh_after = (
            _persistent_snapshot(refresh, refresh_replicated),
            _persistent_snapshot(refresh, refresh_sharded),
        )

        divergent, divergent_replicated, divergent_sharded = build(
            "approx" if rank else "exact"
        )
        divergent._predict_period_from_grad_sq = lambda *args: 16
        set_gradients(divergent_replicated, divergent_sharded)
        try:
            divergent.initialize_codebook()
            mode_mismatch_rejected = False
        except RuntimeError as exc:
            mode_mismatch_rejected = "policy" in str(exc)

        unscoped_divergent, unscoped_replicated, unscoped_sharded = build(
            "approx" if rank else "exact",
            scoped=False,
        )
        unscoped_divergent._predict_period_from_grad_sq = lambda *args: 16
        set_gradients(unscoped_replicated, unscoped_sharded)
        try:
            unscoped_divergent.initialize_codebook()
            unscoped_mode_mismatch_rejected = False
        except RuntimeError as exc:
            unscoped_mode_mismatch_rejected = (
                "identical sharded_mode collective intent" in str(exc)
            )

        result_queue.put(
            {
                "rank": rank,
                "exact_initialized": exact_initialized,
                "exact_logical_once": torch.equal(exact._gefen_codebook, exact_oracle),
                "duplicate_would_differ": not torch.equal(
                    exact_oracle, duplicated_shard_oracle
                ),
                "scoped_checkpoint_hook_rejected": (
                    scoped_checkpoint_hook_rejected
                ),
                "approximate_initialized": approximate_initialized,
                "approximate_local_shards": torch.equal(
                    approximate._gefen_codebook, approximate_oracle
                ),
                "period_mismatch_rejected": period_mismatch_rejected,
                "period_mismatch_atomic": values_equal(
                    mismatch_after, mismatch_before, "period mismatch atomicity"
                ),
                "initialize_presence_rejected": initialize_presence_rejected,
                "initialize_presence_atomic": values_equal(
                    presence_after, presence_before, "initialize presence atomicity"
                ),
                "refresh_presence_rejected": refresh_presence_rejected,
                "refresh_presence_atomic": values_equal(
                    refresh_after, refresh_before, "refresh presence atomicity"
                ),
                "mode_mismatch_rejected": mode_mismatch_rejected,
                "unscoped_mode_mismatch_rejected": (
                    unscoped_mode_mismatch_rejected
                ),
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _unscoped_muon_mode_transition_worker(rank, init_file, result_queue):
    from gefen import GefenMuon, ParameterRebinding, PlacementKind
    from torch import nn
    from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=_WORLD,
            timeout=timedelta(seconds=90),
        )
        mesh = init_device_mesh("cpu", (_WORLD,), mesh_dim_names=("dp",))
        shape = (4, 4)
        full = torch.arange(16, dtype=torch.float32).reshape(shape).div_(16)
        gradient = torch.arange(16, dtype=torch.float32).reshape(shape).sin()
        manifest, shards = _manifest(
            "Model.Weight",
            shape,
            kind=PlacementKind.DIMENSION_SHARD,
            dimension=0,
            semantic_name="unscoped_muon_transition",
        )
        old = nn.Parameter(full.clone())
        parameter = nn.Parameter(distribute_tensor(full.clone(), mesh, [Shard(0)]))
        optimizer = GefenMuon(
            [("legacy.weight", old)],
            lr=3e-3,
            fused=False,
            deterministic=True,
            sharded_mode="approx",
        )
        optimizer.post_sharding(
            (ParameterRebinding(old, parameter, shards[rank]),),
            manifest=manifest,
        )
        optimizer._predict_period_from_grad_sq = lambda *args: 1
        parameter.grad = distribute_tensor(gradient.clone(), mesh, [Shard(0)])
        initialized = optimizer.initialize_codebook()
        parameter.grad = distribute_tensor(gradient.cos(), mesh, [Shard(0)])
        optimizer.step()
        warm_global_step = optimizer._gefen_global_step
        warm_state_initialized = "m_codebook" in optimizer.state[parameter]
        parameter_before = _tensor_snapshot(parameter)
        state_before = _persistent_snapshot(optimizer, parameter)
        if rank == 0:
            optimizer.param_groups[0]["sharded_mode"] = "exact"
        try:
            optimizer.step()
            asymmetric_rejected = False
        except RuntimeError as exc:
            asymmetric_rejected = (
                "identical sharded_mode collective intent" in str(exc)
            )
        asymmetric_parameter_unchanged = _tensor_unchanged(
            parameter, parameter_before
        )
        try:
            _assert_value_exact(
                _persistent_snapshot(optimizer, parameter),
                state_before,
                "asymmetric post-warmup mode transition atomicity",
            )
            asymmetric_state_unchanged = True
        except AssertionError:
            asymmetric_state_unchanged = False

        optimizer.param_groups[0]["sharded_mode"] = "distributed"
        try:
            optimizer.step()
            unanimous_rejected = False
        except RuntimeError as exc:
            unanimous_rejected = "post-validation DTensor" in str(exc)
        unanimous_parameter_unchanged = _tensor_unchanged(
            parameter, parameter_before
        )
        try:
            _assert_value_exact(
                _persistent_snapshot(optimizer, parameter),
                state_before,
                "unanimous post-warmup mode transition atomicity",
            )
            unanimous_state_unchanged = True
        except AssertionError:
            unanimous_state_unchanged = False

        optimizer.param_groups[0]["sharded_mode"] = "approx"
        replacement = nn.Parameter(
            distribute_tensor((full + 1).clone(), mesh, [Shard(0)])
        )
        group = optimizer.param_groups[0]
        original_params = list(group["params"])
        original_names = list(group["param_names"])

        def unchanged(label):
            if not _tensor_unchanged(parameter, parameter_before):
                return False
            try:
                state_after = _persistent_snapshot(optimizer, parameter)
                _assert_value_exact(
                    state_after,
                    state_before,
                    label,
                )
                return True
            except AssertionError:
                return False

        if rank == 0:
            group["sharded_mode"] = "invalid"
        try:
            optimizer.step()
            invalid_mode_rejected = False
        except RuntimeError as exc:
            invalid_mode_rejected = "requires sharded_mode" in str(exc)
        invalid_mode_unchanged = unchanged("invalid-mode header atomicity")
        group["sharded_mode"] = "approx"

        if rank == 0:
            group["params"] = []
            group["param_names"] = []
        try:
            optimizer.step()
            membership_rejected = False
        except RuntimeError as exc:
            membership_rejected = "identical DTensor parameter membership" in str(
                exc
            )
        membership_unchanged = unchanged("membership header atomicity")
        group["params"] = list(original_params)
        group["param_names"] = list(original_names)

        if rank == 0:
            group["params"] = None
        try:
            optimizer.refresh_codebook()
            malformed_rejected = False
        except RuntimeError as exc:
            malformed_rejected = "structurally valid and identical" in str(exc)
        malformed_unchanged = unchanged("malformed-group header atomicity")
        group["params"] = list(original_params)

        if rank == 0:
            group["params"] = [replacement]
        try:
            optimizer.step()
            rebind_rejected = False
        except RuntimeError as exc:
            rebind_rejected = "post-validation DTensor" in str(exc)
        rebind_unchanged = unchanged("rebind header atomicity")
        group["params"] = list(original_params)

        group["sharded_mode"] = "exact"
        baseline_before_add = optimizer._gefen_muon_collective_intent_baseline
        capture_plan_before_add = optimizer._gefen_muon_capture_plan
        try:
            optimizer.add_param_group(
                {
                    "params": [("extra", replacement)],
                    "sharded_mode": "approx",
                }
            )
            warm_add_rejected = False
        except RuntimeError as exc:
            warm_add_rejected = "cannot add a parameter group" in str(exc)
        warm_add_preserved_plan = (
            optimizer._gefen_muon_collective_intent_baseline
            is baseline_before_add
            and optimizer._gefen_muon_capture_plan is capture_plan_before_add
            and len(optimizer.param_groups) == 1
        )
        try:
            optimizer.step()
            add_bypass_transition_rejected = False
        except RuntimeError as exc:
            add_bypass_transition_rejected = "post-validation DTensor" in str(exc)
        add_bypass_unchanged = unchanged("warm add-group bypass atomicity")
        group["sharded_mode"] = "approx"

        checkpoint = optimizer.state_dict()
        old_capture_plan = optimizer._gefen_muon_capture_plan
        optimizer.load_state_dict(checkpoint)
        loaded_group = optimizer.param_groups[0]
        fresh_loaded_plan = optimizer._gefen_muon_capture_plan
        load_refroze_fresh_plan = (
            optimizer._gefen_muon_collective_intent_baseline is not None
            and fresh_loaded_plan is not None
            and fresh_loaded_plan is not old_capture_plan
            and fresh_loaded_plan[0][0][0] is loaded_group
        )
        loaded_group["sharded_mode"] = "exact"
        try:
            optimizer.add_param_group(
                {
                    "params": [("post_load_extra", replacement)],
                    "sharded_mode": "approx",
                }
            )
            post_load_add_rejected = False
        except RuntimeError as exc:
            post_load_add_rejected = "cannot add a parameter group" in str(exc)
        post_load_add_preserved_plan = (
            optimizer._gefen_muon_capture_plan is fresh_loaded_plan
            and len(optimizer.param_groups) == 1
        )
        loaded_group["sharded_mode"] = "approx"
        load_plan_unchanged = unchanged("post-load route-plan atomicity")
        result_queue.put(
            {
                "rank": rank,
                "initialized": initialized,
                "warm_global_step": warm_global_step,
                "warm_state_initialized": warm_state_initialized,
                "asymmetric_rejected": asymmetric_rejected,
                "asymmetric_parameter_unchanged": (
                    asymmetric_parameter_unchanged
                ),
                "asymmetric_state_unchanged": asymmetric_state_unchanged,
                "unanimous_rejected": unanimous_rejected,
                "unanimous_parameter_unchanged": unanimous_parameter_unchanged,
                "unanimous_state_unchanged": unanimous_state_unchanged,
                "invalid_mode_rejected": invalid_mode_rejected,
                "invalid_mode_unchanged": invalid_mode_unchanged,
                "membership_rejected": membership_rejected,
                "membership_unchanged": membership_unchanged,
                "malformed_rejected": malformed_rejected,
                "malformed_unchanged": malformed_unchanged,
                "rebind_rejected": rebind_rejected,
                "rebind_unchanged": rebind_unchanged,
                "warm_add_rejected": warm_add_rejected,
                "warm_add_preserved_plan": warm_add_preserved_plan,
                "add_bypass_transition_rejected": (
                    add_bypass_transition_rejected
                ),
                "add_bypass_unchanged": add_bypass_unchanged,
                "load_refroze_fresh_plan": load_refroze_fresh_plan,
                "post_load_add_rejected": post_load_add_rejected,
                "post_load_add_preserved_plan": (
                    post_load_add_preserved_plan
                ),
                "load_plan_unchanged": load_plan_unchanged,
                "global_step": optimizer._gefen_global_step,
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _muon_checkpoint_routing_worker(rank, init_file, result_queue):
    from gefen import GefenMuon
    from torch import nn
    from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=_WORLD,
            timeout=timedelta(seconds=90),
        )
        mesh = init_device_mesh("cpu", (_WORLD,), mesh_dim_names=("dp",))
        full = torch.arange(16, dtype=torch.float32).reshape(4, 4).div_(16)
        gradient = torch.arange(16, dtype=torch.float32).reshape(4, 4).sin()
        parameter = nn.Parameter(
            distribute_tensor(full.clone(), mesh, [Shard(0)])
        )
        optimizer = GefenMuon(
            [("weight", parameter)],
            lr=3e-3,
            fused=False,
            deterministic=True,
            ns_steps=1,
            sharded_mode="approx",
        )
        optimizer._predict_period_from_grad_sq = lambda *args: 1
        parameter.grad = distribute_tensor(gradient.clone(), mesh, [Shard(0)])
        optimizer.step()
        parameter.grad = None

        checkpoint = copy.deepcopy(optimizer.state_dict())
        valid_export = (
            checkpoint["param_groups"][0]["sharded_mode"] == "approx"
            and optimizer._gefen_global_step == 1
        )
        optimizer.param_groups[0]["sharded_mode"] = "exact"
        try:
            optimizer.optimizer_contract()
            frozen_contract_rejected = False
        except RuntimeError as exc:
            frozen_contract_rejected = "frozen DTensor collective routing" in str(
                exc
            )
        try:
            optimizer._canonical_state_variant_layout()
            frozen_layout_rejected = False
        except RuntimeError as exc:
            frozen_layout_rejected = "frozen DTensor collective routing" in str(
                exc
            )
        optimizer.param_groups[0]["sharded_mode"] = "invalid"
        try:
            optimizer.optimizer_contract()
            invalid_contract_rejected = False
        except RuntimeError as exc:
            invalid_contract_rejected = "invalid sharded_mode" in str(exc)
        optimizer.param_groups[0]["sharded_mode"] = "approx"
        valid_contract = optimizer.optimizer_contract() is not None
        valid_layout = optimizer._canonical_state_variant_layout() is not None
        parameter_before = _tensor_snapshot(parameter)
        state_before = _persistent_snapshot(optimizer, parameter)

        def unchanged(label):
            if not _tensor_unchanged(parameter, parameter_before):
                return False
            try:
                _assert_value_exact(
                    _persistent_snapshot(optimizer, parameter),
                    state_before,
                    label,
                )
                return True
            except AssertionError:
                return False

        if rank == 0:
            optimizer.param_groups[0]["sharded_mode"] = "distributed"
        try:
            optimizer.state_dict()
            asymmetric_export_rejected = False
        except RuntimeError as exc:
            asymmetric_export_rejected = (
                "identical sharded_mode collective intent" in str(exc)
            )
        asymmetric_export_atomic = unchanged("asymmetric export atomicity")

        optimizer.param_groups[0]["sharded_mode"] = "distributed"
        try:
            optimizer.state_dict()
            unanimous_export_rejected = False
        except RuntimeError as exc:
            unanimous_export_rejected = "post-validation DTensor" in str(exc)
        unanimous_export_atomic = unchanged("unanimous export atomicity")
        optimizer.param_groups[0]["sharded_mode"] = "approx"

        def mutate_route_in_pre_hook(_optimizer):
            if rank == 0:
                _optimizer.param_groups[0]["sharded_mode"] = "distributed"

        hook = optimizer.register_state_dict_pre_hook(mutate_route_in_pre_hook)
        try:
            optimizer.state_dict()
            hook_export_rejected = False
        except RuntimeError as exc:
            hook_export_rejected = (
                "identical sharded_mode collective intent" in str(exc)
            )
        finally:
            hook.remove()
        hook_export_atomic = unchanged("pre-hook export atomicity")
        optimizer.param_groups[0]["sharded_mode"] = "approx"

        if rank == 0:
            optimizer.param_groups[0]["sharded_mode"] = "distributed"
        try:
            optimizer.load_state_dict(copy.deepcopy(checkpoint))
            asymmetric_load_rejected = False
        except RuntimeError as exc:
            asymmetric_load_rejected = (
                "identical sharded_mode collective intent" in str(exc)
            )
        asymmetric_load_atomic = unchanged("asymmetric load atomicity")

        optimizer.param_groups[0]["sharded_mode"] = "distributed"
        try:
            optimizer.load_state_dict(copy.deepcopy(checkpoint))
            unanimous_load_rejected = False
        except RuntimeError as exc:
            unanimous_load_rejected = "post-validation DTensor" in str(exc)
        unanimous_load_atomic = unchanged("unanimous load atomicity")
        optimizer.param_groups[0]["sharded_mode"] = "approx"

        drift_checkpoint = copy.deepcopy(checkpoint)
        drift_checkpoint["param_groups"][0]["sharded_mode"] = "exact"
        try:
            optimizer.load_state_dict(drift_checkpoint)
            checkpoint_route_rejected = False
        except RuntimeError as exc:
            checkpoint_route_rejected = "post-validation DTensor" in str(exc)
        checkpoint_route_atomic = unchanged("checkpoint route atomicity")

        optimizer.load_state_dict(copy.deepcopy(checkpoint))
        valid_load = (
            optimizer.param_groups[0]["sharded_mode"] == "approx"
            and unchanged("valid unchanged load")
        )
        result_queue.put(
            {
                "rank": rank,
                "valid_export": valid_export,
                "frozen_contract_rejected": frozen_contract_rejected,
                "frozen_layout_rejected": frozen_layout_rejected,
                "invalid_contract_rejected": invalid_contract_rejected,
                "valid_contract": valid_contract,
                "valid_layout": valid_layout,
                "asymmetric_export_rejected": asymmetric_export_rejected,
                "asymmetric_export_atomic": asymmetric_export_atomic,
                "unanimous_export_rejected": unanimous_export_rejected,
                "unanimous_export_atomic": unanimous_export_atomic,
                "hook_export_rejected": hook_export_rejected,
                "hook_export_atomic": hook_export_atomic,
                "asymmetric_load_rejected": asymmetric_load_rejected,
                "asymmetric_load_atomic": asymmetric_load_atomic,
                "unanimous_load_rejected": unanimous_load_rejected,
                "unanimous_load_atomic": unanimous_load_atomic,
                "checkpoint_route_rejected": checkpoint_route_rejected,
                "checkpoint_route_atomic": checkpoint_route_atomic,
                "valid_load": valid_load,
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _muon_checkpoint_callback_failure_worker(rank, init_file, result_queue):
    from gefen import GefenMuon
    from torch import nn
    from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=_WORLD,
            timeout=timedelta(seconds=90),
        )
        mesh = init_device_mesh("cpu", (_WORLD,), mesh_dim_names=("dp",))
        full = torch.arange(16, dtype=torch.float32).reshape(4, 4).div_(16)
        gradient = torch.arange(16, dtype=torch.float32).reshape(4, 4).sin()
        cases = []
        for mode in ("exact", "distributed"):
            parameter = nn.Parameter(
                distribute_tensor(full.clone(), mesh, [Shard(0)])
            )
            optimizer = GefenMuon(
                [("weight", parameter)],
                lr=3e-3,
                fused=False,
                deterministic=True,
                ns_steps=1,
                sharded_mode=mode,
            )
            optimizer._predict_period_from_grad_sq = lambda *args: 1
            parameter.grad = distribute_tensor(
                gradient.clone(), mesh, [Shard(0)]
            )
            optimizer.step()
            parameter.grad = None
            checkpoint = copy.deepcopy(optimizer.state_dict())
            parameter_before = _tensor_snapshot(parameter)
            state_before = _persistent_snapshot(optimizer, parameter)

            def rejected(register, hook, action, phase):
                handle = register(hook) if rank == 0 else None
                try:
                    action()
                    matched = False
                except RuntimeError as exc:
                    matched = "checkpoint {} failed".format(phase) in str(exc)
                finally:
                    if handle is not None:
                        handle.remove()
                dist.barrier()
                return matched

            state_pre = rejected(
                optimizer.register_state_dict_pre_hook,
                lambda _optimizer: (_ for _ in ()).throw(
                    RuntimeError("state pre-hook failure")
                ),
                optimizer.state_dict,
                "state-dict pre-hook",
            )
            state_post = rejected(
                optimizer.register_state_dict_post_hook,
                lambda _optimizer, _state: (_ for _ in ()).throw(
                    RuntimeError("state post-hook failure")
                ),
                optimizer.state_dict,
                "state-dict post-hook",
            )
            load_pre = rejected(
                optimizer.register_load_state_dict_pre_hook,
                lambda _optimizer, _state: (_ for _ in ()).throw(
                    RuntimeError("load pre-hook failure")
                ),
                lambda: optimizer.load_state_dict(copy.deepcopy(checkpoint)),
                "load pre-hook",
            )
            try:
                _assert_value_exact(
                    _persistent_snapshot(optimizer, parameter),
                    state_before,
                    "pre-commit checkpoint callback atomicity",
                )
                precommit_state_atomic = True
            except AssertionError:
                precommit_state_atomic = False
            load_post = rejected(
                optimizer.register_load_state_dict_post_hook,
                lambda _optimizer: (_ for _ in ()).throw(
                    RuntimeError("load post-hook failure")
                ),
                lambda: optimizer.load_state_dict(copy.deepcopy(checkpoint)),
                "load post-hook",
            )
            if rank == 0:
                optimizer._gefen_global_step_by_device = {
                    torch.device("cpu"): torch.tensor([1, 2])
                }
            try:
                optimizer.state_dict()
                counter_preparation_rejected = False
            except RuntimeError as exc:
                counter_preparation_rejected = (
                    "checkpoint local serialization preparation failed"
                    in str(exc)
                )
            optimizer._gefen_global_step_by_device = {}
            dist.barrier()

            if mode == "distributed":
                if rank == 0:
                    optimizer.state[parameter]["unpickleable"] = lambda: None
                try:
                    optimizer.state_dict()
                    owner_preparation_rejected = False
                except RuntimeError as exc:
                    owner_preparation_rejected = (
                        "checkpoint distributed serialization preparation failed"
                        in str(exc)
                    )
                optimizer.state[parameter].pop("unpickleable", None)
                dist.barrier()
            else:
                owner_preparation_rejected = True
            valid_checkpoint = optimizer.state_dict()
            optimizer.load_state_dict(copy.deepcopy(valid_checkpoint))
            parameter_atomic = _tensor_unchanged(parameter, parameter_before)
            cases.append(
                {
                    "mode": mode,
                    "state_pre": state_pre,
                    "state_post": state_post,
                    "load_pre": load_pre,
                    "load_post": load_post,
                    "counter_preparation": counter_preparation_rejected,
                    "owner_preparation": owner_preparation_rejected,
                    "valid": (
                        valid_checkpoint["param_groups"][0]["sharded_mode"]
                        == mode
                    ),
                    "precommit_state_atomic": precommit_state_atomic,
                    "parameter_atomic": parameter_atomic,
                }
            )
            dist.barrier()
        result_queue.put({"rank": rank, "cases": cases})
    except BaseException:
        result_queue.put({"rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _spawn(worker):
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    directory = tempfile.mkdtemp(prefix="gefen-dtensor-rebinding-")
    init_file = os.path.join(directory, "store")
    processes = [context.Process(target=worker, args=(rank, init_file, result_queue)) for rank in range(_WORLD)]
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
    return sorted(results, key=lambda item: item["rank"])


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DTensor rebinding requires Gloo",
)
def test_dtensor_rebinding_matches_rank_local_oracles_and_continues_exactly():
    results = _spawn(_valid_worker)
    assert all("traceback" not in result for result in results), results
    assert all(len(result["cases"]) == 5 for result in results), results
    assert any(case["empty"] for result in results for case in result["cases"]), results


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="scoped distributed Muon DTensor coverage requires Gloo",
)
def test_scoped_distributed_muon_native_contract_is_same_topology_and_loads_exactly():
    results = _spawn(_scoped_distributed_muon_contract_worker)
    assert all("traceback" not in result for result in results), results
    assert all(
        result["contract_exact"]
        and result["common_state_restored"]
        and result["owner_state_restored"]
        and result["continued_exactly"]
        for result in results
    ), results


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DTensor rebinding requires Gloo",
)
def test_dtensor_rebinding_rejects_invalid_runtime_topology_without_mutation():
    results = _spawn(_rejection_worker)
    assert all("traceback" not in result for result in results), results
    failures = {
        (result["rank"], name): details
        for result in results
        for name, details in result["results"].items()
        if not details["rejected"] or not details["unchanged"]
    }
    assert not failures, failures


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="scoped DTensor codebook learning requires Gloo",
)
def test_scoped_replicated_dtensor_contributes_once_and_requires_period_agreement():
    results = _spawn(_scoped_replicated_dtensor_worker)
    assert all("traceback" not in result for result in results), results
    for result in results:
        assert result["initialized"], result
        assert result["logical_once"], result
        assert result["duplicate_would_differ"], result
        assert result["agreed"], result
        assert result["period_mismatch_rejected"], result
        assert result["mismatch_atomic"], result


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="scoped Muon DTensor codebook coverage requires Gloo",
)
def test_scoped_muon_dtensor_codebook_uses_one_logical_input_and_safe_manual_preflights():
    results = _spawn(_scoped_muon_codebook_worker)
    assert all("traceback" not in result for result in results), results
    for result in results:
        assert result["exact_initialized"], result
        assert result["exact_logical_once"], result
        assert result["duplicate_would_differ"], result
        assert result["scoped_checkpoint_hook_rejected"], result
        assert result["approximate_initialized"], result
        assert result["approximate_local_shards"], result
        assert result["period_mismatch_rejected"], result
        assert result["period_mismatch_atomic"], result
        assert result["initialize_presence_rejected"], result
        assert result["initialize_presence_atomic"], result
        assert result["refresh_presence_rejected"], result
        assert result["refresh_presence_atomic"], result
        assert result["mode_mismatch_rejected"], result
        assert result["unscoped_mode_mismatch_rejected"], result


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="unscoped Muon DTensor transition coverage requires Gloo",
)
def test_unscoped_muon_post_warmup_mode_transition_rejects_symmetrically_without_mutation():
    results = _spawn(_unscoped_muon_mode_transition_worker)
    assert all("traceback" not in result for result in results), results
    assert all(
        result["initialized"]
        and result["warm_global_step"] == 1
        and result["warm_state_initialized"]
        and result["asymmetric_rejected"]
        and result["asymmetric_parameter_unchanged"]
        and result["asymmetric_state_unchanged"]
        and result["unanimous_rejected"]
        and result["unanimous_parameter_unchanged"]
        and result["unanimous_state_unchanged"]
        and result["invalid_mode_rejected"]
        and result["invalid_mode_unchanged"]
        and result["membership_rejected"]
        and result["membership_unchanged"]
        and result["malformed_rejected"]
        and result["malformed_unchanged"]
        and result["rebind_rejected"]
        and result["rebind_unchanged"]
        and result["warm_add_rejected"]
        and result["warm_add_preserved_plan"]
        and result["add_bypass_transition_rejected"]
        and result["add_bypass_unchanged"]
        and result["load_refroze_fresh_plan"]
        and result["post_load_add_rejected"]
        and result["post_load_add_preserved_plan"]
        and result["load_plan_unchanged"]
        and result["global_step"] == 1
        for result in results
    ), results


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Muon checkpoint routing coverage requires Gloo",
)
def test_muon_checkpoint_rejects_post_warmup_route_drift_before_save_and_load():
    results = _spawn(_muon_checkpoint_routing_worker)
    assert all("traceback" not in result for result in results), results
    assert all(
        result["valid_export"]
        and result["frozen_contract_rejected"]
        and result["frozen_layout_rejected"]
        and result["invalid_contract_rejected"]
        and result["valid_contract"]
        and result["valid_layout"]
        and result["asymmetric_export_rejected"]
        and result["asymmetric_export_atomic"]
        and result["unanimous_export_rejected"]
        and result["unanimous_export_atomic"]
        and result["hook_export_rejected"]
        and result["hook_export_atomic"]
        and result["asymmetric_load_rejected"]
        and result["asymmetric_load_atomic"]
        and result["unanimous_load_rejected"]
        and result["unanimous_load_atomic"]
        and result["checkpoint_route_rejected"]
        and result["checkpoint_route_atomic"]
        and result["valid_load"]
        for result in results
    ), results


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Muon checkpoint callback failure coverage requires Gloo",
)
def test_muon_checkpoint_synchronizes_exact_and_distributed_callback_failures():
    results = _spawn(_muon_checkpoint_callback_failure_worker)
    assert all("traceback" not in result for result in results), results
    for result in results:
        assert [case["mode"] for case in result["cases"]] == [
            "exact",
            "distributed",
        ]
        for case in result["cases"]:
            assert case["state_pre"], (result["rank"], case)
            assert case["state_post"], (result["rank"], case)
            assert case["load_pre"], (result["rank"], case)
            assert case["load_post"], (result["rank"], case)
            assert case["counter_preparation"], (result["rank"], case)
            assert case["owner_preparation"], (result["rank"], case)
            assert case["valid"], (result["rank"], case)
            assert case["precommit_state_atomic"], (result["rank"], case)
            assert case["parameter_atomic"], (result["rank"], case)

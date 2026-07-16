"""Two-rank portable-v3 coverage for exact public DTensor storage."""

from __future__ import annotations

import copy
from datetime import timedelta
import math
import multiprocessing as mp
import os
import queue as queue_module
import shutil
import tempfile
import time
import traceback

import pytest
import torch
import torch.distributed as dist


_WORLD = 2
_FQN = "model.weight"


def _limits():
    from gefen import PortableStateLimits

    return PortableStateLimits(
        max_fragment_tensor_bytes=2 << 20,
        max_collective_tensor_bytes=8 << 20,
        max_collective_metadata_bytes=8 << 20,
        chunk_bytes=29,
        max_members=4,
        max_metadata_bytes=2 << 20,
        max_tree_nodes=20_000,
        max_tree_depth=32,
        max_container_items=20_000,
        max_string_bytes=16 << 10,
        max_integer_bytes=128,
        max_tensors=512,
        max_tensor_rank=8,
        diagnostic_bytes=1024,
    )


def _members():
    return tuple("rank:{}".format(rank) for rank in range(_WORLD))


def _bindings(group, rank):
    from gefen import CheckpointProcessGroupBinding, CodebookProcessGroupBinding

    member = _members()[rank]
    return (
        CodebookProcessGroupBinding(group, member, dist.group.WORLD, torch.device("cpu")),
        CheckpointProcessGroupBinding(group, member, dist.group.WORLD, torch.device("cpu")),
    )


def _standard_region(shape, coordinate, dimension):
    from gefen import LogicalRegion

    chunk = (shape[dimension] + _WORLD - 1) // _WORLD
    offset = min(coordinate * chunk, shape[dimension])
    length = max(0, min(shape[dimension], offset + chunk) - offset)
    offsets = [0] * len(shape)
    lengths = list(shape)
    offsets[dimension] = offset
    lengths[dimension] = length
    return LogicalRegion(tuple(offsets), tuple(lengths))


def _dtensor_manifest(identity, group, *, placement_kind, dimension=None):
    from gefen import (
        LogicalRegion,
        ParameterLayout,
        PlacementKind,
        ShardIdentity,
        ShardPlacement,
        ShardingManifest,
    )

    shards = []
    for coordinate, member in enumerate(group.ordered_members):
        region = (
            LogicalRegion.full(identity)
            if placement_kind is PlacementKind.REPLICATE
            else _standard_region(identity.global_shape, coordinate, dimension)
        )
        shards.append(
            ShardIdentity(
                identity,
                ParameterLayout.DTENSOR_1D_DEFAULT_WORLD,
                region,
                placements=(
                    ShardPlacement(
                        "dp",
                        placement_kind,
                        coordinate,
                        _WORLD,
                        parameter_dimension=dimension,
                    ),
                ),
                process_group=group,
                local_member=member,
            )
        )
    return ShardingManifest(tuple(shards)), tuple(shards)


def _replicated_manifest(identity, group):
    from gefen import LogicalSlice, ParameterLayout, PlacementKind, ShardIdentity, ShardPlacement, ShardingManifest

    shards = tuple(
        ShardIdentity(
            identity,
            ParameterLayout.REPLICATED,
            LogicalSlice.full(identity),
            placements=(ShardPlacement("dp", PlacementKind.REPLICATE, coordinate, _WORLD),),
            process_group=group,
            local_member=member,
        )
        for coordinate, member in enumerate(group.ordered_members)
    )
    return ShardingManifest(shards), shards


def _flat_manifest(identity, group):
    from gefen import LogicalSlice, ParameterLayout, PlacementKind, ShardIdentity, ShardPlacement, ShardingManifest

    first = max(0, identity.numel // 3)
    lengths = (first, identity.numel - first)
    offset = 0
    shards = []
    for coordinate, (member, length) in enumerate(zip(group.ordered_members, lengths)):
        shards.append(
            ShardIdentity(
                identity,
                ParameterLayout.FLATTENED_ELEMENT_SHARD,
                LogicalSlice(offset, length),
                placements=(ShardPlacement("dp", PlacementKind.FLAT_SHARD, coordinate, _WORLD),),
                process_group=group,
                local_member=member,
            )
        )
        offset += length
    return ShardingManifest(tuple(shards)), tuple(shards)


def _topology_manifest(identity, group, topology):
    from gefen import PlacementKind

    if topology == "replicated":
        return _replicated_manifest(identity, group)
    if topology == "flat":
        return _flat_manifest(identity, group)
    if topology == "dtensor_shard0":
        return _dtensor_manifest(identity, group, placement_kind=PlacementKind.DIMENSION_SHARD, dimension=0)
    if topology == "dtensor_shard1":
        return _dtensor_manifest(identity, group, placement_kind=PlacementKind.DIMENSION_SHARD, dimension=1)
    if topology == "dtensor_replicate":
        return _dtensor_manifest(identity, group, placement_kind=PlacementKind.REPLICATE)
    raise AssertionError("unknown topology")


def _region_index(region):
    return tuple(slice(offset, offset + length) for offset, length in zip(region.offsets, region.lengths))


def _local_projection(value, shard):
    from gefen import ParameterLayout, PlacementKind

    if shard.layout is ParameterLayout.REPLICATED:
        return value.clone()
    if shard.layout is ParameterLayout.FLATTENED_ELEMENT_SHARD:
        start = shard.logical_slice.flat_offset
        return value.reshape(-1)[start : start + shard.logical_slice.length].clone()
    if shard.layout is ParameterLayout.DTENSOR_1D_DEFAULT_WORLD:
        if shard.placements[0].kind is PlacementKind.REPLICATE:
            return value.clone()
        return value[_region_index(shard.logical_region)].clone()
    raise AssertionError("unsupported test projection")


def _local_tensor(parameter):
    value = parameter.to_local() if hasattr(parameter, "to_local") else parameter
    if hasattr(value, "wait"):
        value = value.wait()
    return value


def _make_optimizer(rank, mesh, group, topology, full_parameter, *, factored_v_2d=False, deterministic=True):
    from gefen import Gefen, ParameterIdentity, ParameterLayout, ParameterRebinding
    from torch import nn
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor

    identity = ParameterIdentity(_FQN, tuple(full_parameter.shape))
    manifest, shards = _topology_manifest(identity, group, topology)
    shard = shards[rank]
    if topology.startswith("dtensor_"):
        if topology == "dtensor_shard0":
            placement = Shard(0)
        elif topology == "dtensor_shard1":
            placement = Shard(1)
        else:
            placement = Replicate()
        parameter = nn.Parameter(distribute_tensor(full_parameter.clone(), mesh, [placement]))
        original = parameter
    elif topology == "replicated":
        parameter = nn.Parameter(full_parameter.clone())
        original = parameter
    else:
        original = nn.Parameter(full_parameter.clone())
        parameter = nn.Parameter(_local_projection(full_parameter, shard))
    optimizer = Gefen(
        [("weight", original)],
        lr=2.5e-3,
        betas=(0.8, 0.97),
        eps=2.0e-8,
        weight_decay=0.03,
        fused=False,
        force_2d_period_one=True,
        factored_v_2d=factored_v_2d,
        deterministic=deterministic,
    )
    codebook_binding, checkpoint_binding = _bindings(group, rank)
    optimizer.post_sharding(
        (ParameterRebinding(original, parameter, shard),),
        manifest=manifest,
        codebook_process_group=codebook_binding,
    )
    if shard.layout is ParameterLayout.DTENSOR_1D_DEFAULT_WORLD:
        assert tuple(_local_tensor(parameter).shape) == shard.logical_region.lengths
    return optimizer, parameter, shard, checkpoint_binding


def _codebook():
    return torch.linspace(-1.0, 1.0, 256, dtype=torch.float32)


def _quantize_period_one(momentum):
    flat = momentum.reshape(-1)
    indices = torch.where(
        torch.signbit(flat),
        torch.zeros(flat.numel(), dtype=torch.uint8),
        torch.full((flat.numel(),), 255, dtype=torch.uint8),
    )
    return indices.reshape(-1, 1), flat.abs().reshape(-1, 1).clone()


def _seed_block_state(optimizer, parameter, shard, momentum, second_moment):
    optimizer._gefen_global_step = 9
    optimizer._gefen_codebook = _codebook()
    local_momentum = _local_projection(momentum, shard)
    if local_momentum.numel() == 0:
        return
    local_second = _local_projection(second_moment, shard)
    indices, magnitudes = _quantize_period_one(local_momentum)
    optimizer.state[parameter].update(
        {
            "automatic_period": 1,
            "step": 7,
            "m_codebook": indices,
            "m_magnitude": magnitudes,
            "vmean": local_second.reshape(-1, 1),
            "vmean_step": 6,
        }
    )


def _seed_factored_state(optimizer, parameter, momentum, row, column):
    optimizer._gefen_global_step = 9
    optimizer._gefen_codebook = _codebook()
    indices, magnitudes = _quantize_period_one(momentum)
    optimizer.state[parameter].update(
        {
            "automatic_period": 1,
            "step": 7,
            "m_codebook": indices,
            "m_magnitude": magnitudes,
            "v_row": row.clone(),
            "v_col": column.clone(),
            "factored_step": 6,
        }
    )


def _decode_local_state(optimizer, parameter, shard):
    from gefen import ParameterLayout
    from gefen.portable import _decode_quantized_momentum

    state = optimizer.state[parameter]
    if shard.layout is ParameterLayout.DTENSOR_1D_DEFAULT_WORLD and shard.logical_region.numel == 0:
        return None, None
    momentum = _decode_quantized_momentum(
        optimizer._gefen_codebook,
        state["m_codebook"],
        state["m_magnitude"],
        logical_shape=tuple(_local_tensor(parameter).shape),
        period=1,
        step=state["step"],
    )
    return momentum, state["vmean"].reshape(_local_tensor(parameter).shape)


def _bits_equal(left, right):
    return (
        left.dtype == right.dtype
        and tuple(left.shape) == tuple(right.shape)
        and torch.equal(left.detach().contiguous().reshape(-1).view(torch.uint8), right.detach().contiguous().reshape(-1).view(torch.uint8))
    )


def _assign_gradient(parameter, shard, mesh, full_gradient):
    from gefen import ParameterLayout
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor

    if shard.layout is ParameterLayout.DTENSOR_1D_DEFAULT_WORLD:
        descriptor = shard.placements[0]
        placement = Replicate() if descriptor.parameter_dimension is None else Shard(descriptor.parameter_dimension)
        parameter.grad = distribute_tensor(full_gradient.clone(), mesh, [placement])
    else:
        parameter.grad = _local_projection(full_gradient, shard)


def _state_snapshot(optimizer, parameter, *, include_containers=False):
    def clone(value):
        if torch.is_tensor(value):
            return value.detach().clone()
        if type(value) is dict:
            return {key: clone(item) for key, item in value.items()}
        if type(value) is list:
            return [clone(item) for item in value]
        if type(value) is tuple:
            return tuple(clone(item) for item in value)
        return copy.deepcopy(value)

    state = {
        key: clone(value)
        for key, value in optimizer.state[parameter].items()
    }
    snapshot = {
        "global_step": optimizer._gefen_global_step,
        "codebook": None if optimizer._gefen_codebook is None else optimizer._gefen_codebook.detach().clone(),
        "deterministic": optimizer._deterministic,
        "parameter": _local_tensor(parameter).detach().clone(),
        "state": state,
    }
    if include_containers:
        snapshot["groups"] = tuple(
            {key: clone(value) for key, value in group.items() if key != "params"}
            for group in optimizer.param_groups
        )
        snapshot["containers"] = (
            id(optimizer.state),
            id(optimizer.param_groups),
            id(optimizer.state[parameter]),
            id(optimizer._param_names),
            id(optimizer._gefen_shard_bindings),
        )
    return snapshot


def _rank_local_checkpoint_schema_is_valid(optimizer):
    state_dict = optimizer.state_dict()
    metadata = state_dict["param_groups"][0]["_gefen_checkpoint_metadata"]
    marker = metadata.get("rank_local_sharded_state")
    expected_carriers = {
        "_gefen_rank_local_payload_{}".format(rank)
        for rank in range(_WORLD)
    }
    carrier_sets = [
        {
            key
            for key in state
            if type(key) is str and key.startswith("_gefen_rank_local_payload_")
        }
        for state in state_dict["state"].values()
    ]
    return (
        type(marker) is dict
        and marker.get("format") == "rank_local_dtensor_v2"
        and carrier_sets.count(expected_carriers) == 1
        and all(not keys or keys == expected_carriers for keys in carrier_sets)
    )


def _live_carrier_schema_is_valid(state):
    expected_keys = {
        "name",
        "_gefen_rank_local_payload_0",
        "_gefen_rank_local_payload_1",
    }
    if set(state) != expected_keys:
        return False
    for rank in range(_WORLD):
        value = state["_gefen_rank_local_payload_{}".format(rank)]
        if (
            type(value) is not torch.Tensor
            or value.layout is not torch.strided
            or value.device.type != "cpu"
            or value.dtype != torch.uint8
            or tuple(value.shape) != (1,)
            or value.requires_grad
            or not value.is_contiguous()
            or value.storage_offset() != 0
            or value.untyped_storage().nbytes() != 1
            or int(value.item()) != 0
        ):
            return False
    return True


def _assert_nested_exact(actual, expected, path="value"):
    if torch.is_tensor(expected):
        assert torch.is_tensor(actual), path
        assert _bits_equal(actual, expected), path
        return
    assert type(actual) is type(expected), path
    if isinstance(expected, dict):
        assert set(actual) == set(expected), path
        for key in expected:
            _assert_nested_exact(actual[key], expected[key], "{}.{}".format(path, key))
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected), path
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_nested_exact(actual_item, expected_item, "{}[{}]".format(path, index))
    else:
        assert actual == expected, path


def _run_transition(rank, mesh, group, *, source_topology, target_topology, shape, label, factored_projection=False):
    from gefen import ParameterLayout

    full_parameter = torch.linspace(-0.8, 0.7, torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
    momentum = torch.linspace(-1.5, 1.25, full_parameter.numel(), dtype=torch.float32).reshape(shape)
    second = torch.arange(1, full_parameter.numel() + 1, dtype=torch.float32).reshape(shape).div_(7.0)
    gradient = torch.arange(full_parameter.numel(), dtype=torch.float32).reshape(shape).cos()

    source, source_parameter, source_shard, source_binding = _make_optimizer(
        rank,
        mesh,
        group,
        source_topology,
        full_parameter,
        factored_v_2d=factored_projection,
        deterministic=True,
    )
    if factored_projection:
        row = torch.linspace(1.0, 2.0, shape[0], dtype=torch.float32)
        column = torch.linspace(0.5, 1.5, shape[1], dtype=torch.float32)
        _seed_factored_state(source, source_parameter, momentum, row, column)
        projected_second = torch.outer(row, column).div_(row.mean().clamp_(min=torch.finfo(torch.float32).tiny))
    else:
        _seed_block_state(source, source_parameter, source_shard, momentum, second)
        projected_second = second
    document = source.export_portable_state(
        checkpoint_process_group=source_binding,
        transaction_id="{}-export".format(label),
        limits=_limits(),
    )
    record = document["parameters"][_FQN]
    document_exact = _bits_equal(record["state"]["momentum"], momentum)
    if factored_projection:
        document_exact = document_exact and _bits_equal(record["state"]["v_row"], row) and _bits_equal(record["state"]["v_col"], column)
    else:
        document_exact = document_exact and _bits_equal(record["state"]["second_moment"], second)

    target, target_parameter, target_shard, target_binding = _make_optimizer(
        rank,
        mesh,
        group,
        target_topology,
        full_parameter,
        factored_v_2d=factored_projection,
        deterministic=False,
    )
    target.import_portable_state(
        document,
        checkpoint_process_group=target_binding,
        transaction_id="{}-import".format(label),
        limits=_limits(),
    )
    expected_local_momentum = _local_projection(momentum, target_shard)
    expected_local_second = _local_projection(projected_second, target_shard)
    if expected_local_momentum.numel() == 0:
        projected_exact = _live_carrier_schema_is_valid(
            target.state[target_parameter]
        )
    else:
        decoded, local_second = _decode_local_state(target, target_parameter, target_shard)
        projected_exact = (
            _bits_equal(decoded, expected_local_momentum)
            and _bits_equal(local_second, expected_local_second)
            and target.state[target_parameter]["step"] == 7
            and target.state[target_parameter]["vmean_step"] == 6
            and target._gefen_global_step == 9
            and target._deterministic is True
        )
    checkpoint_schema_exact = (
        _rank_local_checkpoint_schema_is_valid(target)
        if target_shard.layout is ParameterLayout.DTENSOR_1D_DEFAULT_WORLD
        else True
    )

    reference, reference_parameter, reference_shard, _ = _make_optimizer(
        rank,
        mesh,
        group,
        target_topology,
        full_parameter,
        factored_v_2d=factored_projection,
        deterministic=True,
    )
    _seed_block_state(reference, reference_parameter, reference_shard, momentum, projected_second)
    _assert_nested_exact(
        _state_snapshot(target, target_parameter),
        _state_snapshot(reference, reference_parameter),
        "{} imported state".format(label),
    )
    _assign_gradient(target_parameter, target_shard, mesh, gradient)
    _assign_gradient(reference_parameter, reference_shard, mesh, gradient)
    target.step()
    reference.step()
    continuation_exact = True
    try:
        _assert_nested_exact(
            _state_snapshot(target, target_parameter),
            _state_snapshot(reference, reference_parameter),
            "{} continuation".format(label),
        )
    except AssertionError:
        continuation_exact = False
    return {
        "label": label,
        "document_exact": document_exact,
        "projected_exact": projected_exact,
        "checkpoint_schema_exact": checkpoint_schema_exact,
        "continuation_exact": continuation_exact,
        "empty": expected_local_momentum.numel() == 0,
    }, document


def _mismatch_result(rank, mesh, group, document):
    target_values = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    target, parameter, _, binding = _make_optimizer(
        rank,
        mesh,
        group,
        "dtensor_shard0",
        target_values,
        deterministic=False,
    )
    before = _state_snapshot(target, parameter, include_containers=True)
    error = None
    try:
        target.import_portable_state(
            document,
            checkpoint_process_group=binding,
            transaction_id="dtensor-identity-mismatch",
            limits=_limits(),
        )
    except RuntimeError as exc:
        error = str(exc)
    unchanged = True
    try:
        _assert_nested_exact(
            _state_snapshot(target, parameter, include_containers=True),
            before,
            "mismatched import",
        )
    except AssertionError:
        unchanged = False
    return {"rejected": error is not None, "unchanged": unchanged, "error": error}


def _dcp_transition_result(rank, mesh, group, checkpoint_dir):
    from gefen import load_portable_dcp, save_portable_dcp
    from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter

    shape = (4, 1)
    full_parameter = torch.linspace(-0.8, 0.7, math.prod(shape), dtype=torch.float32).reshape(shape)
    momentum = torch.linspace(-1.5, 1.25, math.prod(shape), dtype=torch.float32).reshape(shape)
    second = torch.arange(1, math.prod(shape) + 1, dtype=torch.float32).reshape(shape).div_(7.0)
    gradient = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape).cos()
    source, source_parameter, source_shard, source_binding = _make_optimizer(
        rank,
        mesh,
        group,
        "dtensor_shard0",
        full_parameter,
        deterministic=True,
    )
    _seed_block_state(source, source_parameter, source_shard, momentum, second)

    optimizer_type = type(source)
    original_export = optimizer_type.export_portable_state

    def reject_dense_export(*_args, **_kwargs):
        raise AssertionError("sharded DCP must not call dense portable export")

    optimizer_type.export_portable_state = reject_dense_export
    try:
        save_portable_dcp(
            source,
            checkpoint_process_group=source_binding,
            storage_writer=FileSystemWriter(checkpoint_dir),
            transaction_id="dtensor-dcp-save",
            limits=_limits(),
        )
    finally:
        optimizer_type.export_portable_state = original_export

    checkpoint_metadata = FileSystemReader(checkpoint_dir).read_metadata()
    entries = checkpoint_metadata.state_dict_metadata
    v2_key = "optimizer.__gefen_portable_metadata_v2__"
    field_entries = [
        value
        for key, value in entries.items()
        if key.startswith("optimizer.__gefen_portable_field_")
    ]
    sharded_envelope = (
        v2_key in entries
        and "optimizer.__gefen_portable_metadata_v1__" not in entries
        and bool(field_entries)
        and all(len(entry.chunks) == _WORLD for entry in field_entries)
    )

    target, target_parameter, target_shard, target_binding = _make_optimizer(
        rank,
        mesh,
        group,
        "dtensor_shard1",
        full_parameter,
        deterministic=False,
    )
    load_portable_dcp(
        target,
        checkpoint_process_group=target_binding,
        storage_reader=FileSystemReader(checkpoint_dir),
        transaction_id="dtensor-dcp-load",
        limits=_limits(),
    )
    reference, reference_parameter, reference_shard, _ = _make_optimizer(
        rank,
        mesh,
        group,
        "dtensor_shard1",
        full_parameter,
        deterministic=True,
    )
    _seed_block_state(reference, reference_parameter, reference_shard, momentum, second)
    imported_exact = True
    try:
        _assert_nested_exact(
            _state_snapshot(target, target_parameter),
            _state_snapshot(reference, reference_parameter),
            "DTensor DCP imported state",
        )
    except AssertionError:
        imported_exact = False
    _assign_gradient(target_parameter, target_shard, mesh, gradient)
    _assign_gradient(reference_parameter, reference_shard, mesh, gradient)
    target.step()
    reference.step()
    continuation_exact = True
    try:
        _assert_nested_exact(
            _state_snapshot(target, target_parameter),
            _state_snapshot(reference, reference_parameter),
            "DTensor DCP continuation",
        )
    except AssertionError:
        continuation_exact = False
    mismatch, mismatch_parameter, _mismatch_shard, mismatch_binding = _make_optimizer(
        rank,
        mesh,
        group,
        "dtensor_shard1",
        full_parameter,
        deterministic=False,
    )
    if rank == 0:
        mismatch.param_groups[0]["eps"] = 1.0e-7
    mismatch_before = _state_snapshot(
        mismatch,
        mismatch_parameter,
        include_containers=True,
    )
    mismatch_error = None
    try:
        load_portable_dcp(
            mismatch,
            checkpoint_process_group=mismatch_binding,
            storage_reader=FileSystemReader(checkpoint_dir),
            transaction_id="dtensor-dcp-target-mismatch",
            limits=_limits(),
        )
    except RuntimeError as exc:
        mismatch_error = str(exc)
    mismatch_unchanged = True
    try:
        _assert_nested_exact(
            _state_snapshot(
                mismatch,
                mismatch_parameter,
                include_containers=True,
            ),
            mismatch_before,
            "DTensor DCP mismatched target",
        )
    except AssertionError:
        mismatch_unchanged = False
    corrupted, corrupted_parameter, _corrupted_shard, corrupted_binding = _make_optimizer(
        rank,
        mesh,
        group,
        "dtensor_shard1",
        full_parameter,
        deterministic=False,
    )
    corrupted_before = _state_snapshot(
        corrupted,
        corrupted_parameter,
        include_containers=True,
    )
    from gefen import portable_dcp_sharded

    original_local_tensor = portable_dcp_sharded._canonical_local_tensor
    corrupted_once = False

    def corrupt_one_rank(value):
        nonlocal corrupted_once
        local = original_local_tensor(value)
        if rank == 0 and not corrupted_once and local.dtype is torch.float32 and local.numel():
            local.reshape(-1)[0] = float("nan")
            corrupted_once = True
        return local

    portable_dcp_sharded._canonical_local_tensor = corrupt_one_rank
    corruption_error = None
    try:
        load_portable_dcp(
            corrupted,
            checkpoint_process_group=corrupted_binding,
            storage_reader=FileSystemReader(checkpoint_dir),
            transaction_id="dtensor-dcp-asymmetric-corruption",
            limits=_limits(),
        )
    except RuntimeError as exc:
        corruption_error = str(exc)
    finally:
        portable_dcp_sharded._canonical_local_tensor = original_local_tensor
    corruption_unchanged = True
    try:
        _assert_nested_exact(
            _state_snapshot(
                corrupted,
                corrupted_parameter,
                include_containers=True,
            ),
            corrupted_before,
            "DTensor DCP asymmetric corruption",
        )
    except AssertionError:
        corruption_unchanged = False
    return {
        "sharded_envelope": sharded_envelope,
        "imported_exact": imported_exact,
        "continuation_exact": continuation_exact,
        "empty": target_shard.logical_region.numel == 0,
        "mismatch_error": mismatch_error,
        "mismatch_unchanged": mismatch_unchanged,
        "corruption_error": corruption_error,
        "corruption_unchanged": corruption_unchanged,
    }


def _malformed_transport_result(rank, mesh, group):
    values = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    optimizer, parameter, _, binding = _make_optimizer(
        rank,
        mesh,
        group,
        "dtensor_shard0",
        values,
    )
    optimizer.state[parameter]["_gefen_rank_local_payload_evil"] = torch.tensor(
        [17],
        dtype=torch.uint8,
    )
    before = _state_snapshot(optimizer, parameter, include_containers=True)
    error = None
    try:
        optimizer.export_portable_state(
            checkpoint_process_group=binding,
            transaction_id="dtensor-malformed-transport",
            limits=_limits(),
        )
    except RuntimeError as exc:
        error = str(exc)
    unchanged = True
    try:
        _assert_nested_exact(
            _state_snapshot(optimizer, parameter, include_containers=True),
            before,
            "malformed transport export",
        )
    except AssertionError:
        unchanged = False
    return {"rejected": error is not None, "unchanged": unchanged, "error": error}


def _mixed_factored_contract_result(rank, mesh, group):
    from gefen import (
        CheckpointProjectionQualifier,
        CheckpointStateRepresentation,
        CheckpointStateTransitionKind,
        CheckpointTransport,
        Gefen,
        ParameterIdentity,
        ParameterLayout,
        ParameterRebinding,
        PlacementKind,
        ShardingManifest,
    )
    from torch import nn
    from torch.distributed.tensor import Shard, distribute_tensor

    factored_identity = ParameterIdentity("model.factored", (2, 4))
    block_identity = ParameterIdentity("model.block", (4, 4))
    factored_manifest, factored_shards = _replicated_manifest(
        factored_identity, group
    )
    block_manifest, block_shards = _dtensor_manifest(
        block_identity,
        group,
        placement_kind=PlacementKind.DIMENSION_SHARD,
        dimension=0,
    )
    manifest = ShardingManifest(
        tuple(factored_manifest.shards) + tuple(block_manifest.shards)
    )
    factored = nn.Parameter(torch.arange(8, dtype=torch.float32).reshape(2, 4))
    block_old = nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 4))
    block = nn.Parameter(
        distribute_tensor(block_old.detach().clone(), mesh, [Shard(0)])
    )
    optimizer = Gefen(
        [("factored", factored), ("block", block_old)],
        fused=False,
        deterministic=True,
        force_2d_period_one=True,
        factored_v_2d=True,
    )
    codebook_binding, _ = _bindings(group, rank)
    optimizer.post_sharding(
        (
            ParameterRebinding(factored, factored, factored_shards[rank]),
            ParameterRebinding(block_old, block, block_shards[rank]),
        ),
        manifest=manifest,
        codebook_process_group=codebook_binding,
    )
    support = next(
        item
        for item in optimizer.optimizer_contract().capabilities.checkpoints
        if item.transport is CheckpointTransport.CANONICAL_GLOBAL
    )
    factored_exact = next(
        item
        for item in support.state_transitions
        if item.source is CheckpointStateRepresentation.FACTORED_SECOND_MOMENT
        and item.target is CheckpointStateRepresentation.FACTORED_SECOND_MOMENT
    )
    block_exact = next(
        item
        for item in support.state_transitions
        if item.source is CheckpointStateRepresentation.BLOCK_SECOND_MOMENT
        and item.target is CheckpointStateRepresentation.BLOCK_SECOND_MOMENT
    )
    projection = next(
        item
        for item in support.state_transitions
        if item.source is CheckpointStateRepresentation.FACTORED_SECOND_MOMENT
        and item.target is CheckpointStateRepresentation.BLOCK_SECOND_MOMENT
    )
    return {
        "layouts": support.same_topology
        == frozenset(
            {
                ParameterLayout.REPLICATED,
                ParameterLayout.DTENSOR_1D_DEFAULT_WORLD,
            }
        ),
        "factored_exact": (
            factored_exact.kind is CheckpointStateTransitionKind.EXACT
            and factored_exact.same_topology
            == frozenset({ParameterLayout.REPLICATED})
        ),
        "block_exact": (
            block_exact.kind is CheckpointStateTransitionKind.EXACT
            and block_exact.same_topology
            == frozenset({ParameterLayout.DTENSOR_1D_DEFAULT_WORLD})
        ),
        "exact_union": (
            factored_exact.same_topology | block_exact.same_topology
            == support.same_topology
        ),
        "projection": (
            projection.kind is CheckpointStateTransitionKind.DEFINED_PROJECTION
            and projection.qualifier
            is CheckpointProjectionQualifier.FACTORED_TO_BLOCK_LIVE_FP32_TARGET_PERIOD_ONE_V1
        ),
    }


def _worker(rank, init_file, result_queue):
    from gefen import ProcessGroupIdentity
    from torch.distributed.tensor import init_device_mesh

    try:
        torch.set_num_threads(1)
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=_WORLD,
            timeout=timedelta(seconds=45),
        )
        mesh = init_device_mesh("cpu", (_WORLD,), mesh_dim_names=("dp",))
        group = ProcessGroupIdentity("portable_dtensor", _members())
        cases = []
        last_document = None
        for transition in (
            {
                "source_topology": "dtensor_shard0",
                "target_topology": "replicated",
                "shape": (5, 4),
                "label": "dtensor-shard0-to-replicated",
            },
            {
                "source_topology": "replicated",
                "target_topology": "dtensor_shard1",
                "shape": (4, 1),
                "label": "replicated-to-dtensor-shard1-empty",
            },
            {
                "source_topology": "dtensor_replicate",
                "target_topology": "flat",
                "shape": (5, 4),
                "label": "dtensor-replicate-to-flat",
            },
            {
                "source_topology": "replicated",
                "target_topology": "dtensor_replicate",
                "shape": (5, 4),
                "label": "replicated-to-dtensor-replicate",
            },
            {
                "source_topology": "flat",
                "target_topology": "dtensor_shard0",
                "shape": (5, 4),
                "label": "flat-to-dtensor-shard0",
            },
            {
                "source_topology": "dtensor_shard0",
                "target_topology": "dtensor_shard1",
                "shape": (5, 4),
                "label": "dtensor-shard0-to-shard1",
            },
            {
                "source_topology": "replicated",
                "target_topology": "dtensor_shard0",
                "shape": (5, 4),
                "label": "factored-replicated-to-dtensor-block",
                "factored_projection": True,
            },
        ):
            result, last_document = _run_transition(rank, mesh, group, **transition)
            cases.append(result)
            dist.barrier()
        dcp = _dcp_transition_result(
            rank,
            mesh,
            group,
            os.path.join(os.path.dirname(init_file), "portable-dcp"),
        )
        dist.barrier()
        malformed_transport = _malformed_transport_result(rank, mesh, group)
        dist.barrier()
        mismatch = _mismatch_result(rank, mesh, group, last_document)
        mixed_factored_contract = _mixed_factored_contract_result(
            rank, mesh, group
        )
        result_queue.put(
            {
                "rank": rank,
                "cases": cases,
                "dcp": dcp,
                "malformed_transport": malformed_transport,
                "mismatch": mismatch,
                "mixed_factored_contract": mixed_factored_contract,
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_workers():
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    directory = tempfile.mkdtemp(prefix="gefen-portable-dtensor-")
    init_file = os.path.join(directory, "store")
    processes = [context.Process(target=_worker, args=(rank, init_file, result_queue)) for rank in range(_WORLD)]
    results = []
    deadline = time.monotonic() + 120
    try:
        for process in processes:
            process.start()
        while len(results) < _WORLD:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                results.append(result_queue.get(timeout=remaining))
            except queue_module.Empty:
                break
        for process in processes:
            process.join(timeout=max(0.0, min(5.0, deadline - time.monotonic())))
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
    assert all(process.exitcode == 0 for process in processes), (results, [process.exitcode for process in processes])
    results = sorted(results, key=lambda item: item["rank"])
    assert all("fatal" not in result for result in results), results
    return results


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="portable DTensor coverage requires Gloo")
def test_portable_dtensor_reshards_all_public_1d_placements_and_continues_exactly():
    results = _run_workers()
    assert all(len(result["cases"]) == 7 for result in results), results
    assert any(case["empty"] for result in results for case in result["cases"]), results
    assert all(
        case["document_exact"]
        and case["projected_exact"]
        and case["checkpoint_schema_exact"]
        and case["continuation_exact"]
        for result in results
        for case in result["cases"]
    ), results
    assert any(result["dcp"]["empty"] for result in results), results
    assert all(
        result["dcp"]["sharded_envelope"]
        and result["dcp"]["imported_exact"]
        and result["dcp"]["continuation_exact"]
        and result["dcp"]["mismatch_error"] is not None
        and result["dcp"]["mismatch_unchanged"]
        and result["dcp"]["corruption_error"] is not None
        and result["dcp"]["corruption_unchanged"]
        for result in results
    ), results
    mismatch_messages = [result["dcp"]["mismatch_error"] for result in results]
    assert mismatch_messages[0] == mismatch_messages[1]
    corruption_messages = [result["dcp"]["corruption_error"] for result in results]
    assert corruption_messages[0] == corruption_messages[1]
    assert all(
        result["malformed_transport"]["rejected"]
        and result["malformed_transport"]["unchanged"]
        for result in results
    ), results
    assert all(result["mismatch"]["rejected"] and result["mismatch"]["unchanged"] for result in results), results
    assert all(
        all(result["mixed_factored_contract"].values()) for result in results
    ), results


def test_public_rebinding_and_portable_runtime_reject_dtensor_like_tensor_subclasses():
    from gefen import Gefen, LogicalRegion, ParameterIdentity, ParameterLayout, ParameterRebinding, PlacementKind, ProcessGroupIdentity, ShardIdentity, ShardPlacement, ShardingManifest
    from gefen.portable_runtime import _parameter_storage_token, _parameter_supported
    from torch import nn

    class DTensorLike(torch.Tensor):
        @staticmethod
        def __new__(cls):
            return torch.Tensor._make_subclass(cls, torch.zeros(2, 2), False)

        def to_local(self):
            return self.as_subclass(torch.Tensor)

        @property
        def placements(self):
            return ()

        @property
        def device_mesh(self):
            return None

    identity = ParameterIdentity(_FQN, (2, 2))
    group = ProcessGroupIdentity("portable_dtensor_subclass", ("rank:0",))
    shard = ShardIdentity(
        identity,
        ParameterLayout.DTENSOR_1D_DEFAULT_WORLD,
        LogicalRegion.full(identity),
        placements=(ShardPlacement("dp", PlacementKind.REPLICATE, 0, 1),),
        process_group=group,
        local_member="rank:0",
    )
    value = DTensorLike()
    original = nn.Parameter(torch.zeros(2, 2))
    optimizer = Gefen(
        [("weight", original)],
        fused=False,
        force_2d_period_one=True,
    )
    before = (
        id(optimizer.state),
        id(optimizer.param_groups),
        id(optimizer.param_groups[0]),
        optimizer.param_groups[0]["params"][0],
        tuple(optimizer.state),
    )
    with pytest.raises(TypeError, match="DTensor-like tensor subclasses"):
        optimizer.post_sharding(
            (ParameterRebinding(original, value, shard),),
            manifest=ShardingManifest((shard,)),
        )
    after = (
        id(optimizer.state),
        id(optimizer.param_groups),
        id(optimizer.param_groups[0]),
        optimizer.param_groups[0]["params"][0],
        tuple(optimizer.state),
    )
    assert after == before
    assert not _parameter_supported(value, shard)
    with pytest.raises(TypeError, match="DTensor-like tensor subclasses"):
        _parameter_storage_token(value)

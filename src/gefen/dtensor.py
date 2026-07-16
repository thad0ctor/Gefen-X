"""PyTorch DTensor validation for stable optimizer rebinding."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard

from gefen.contracts import (
    LogicalRegion,
    ParameterLayout,
    PlacementKind,
    ShardIdentity,
    ShardingManifest,
)


def is_exact_dtensor(value) -> bool:
    """Return whether ``value`` is the public, exact PyTorch DTensor type."""

    return type(value) is DTensor


def looks_like_dtensor(value) -> bool:
    """Return whether a tensor advertises DTensor-like runtime attributes."""

    return isinstance(value, DTensor) or (
        isinstance(value, torch.Tensor)
        and hasattr(value, "to_local")
        and hasattr(value, "placements")
        and hasattr(value, "device_mesh")
    )


def resolve_local_tensor(value) -> torch.Tensor:
    """Resolve an exact DTensor to an ordinary materialized local tensor."""

    if not is_exact_dtensor(value):
        raise TypeError("DTensor rebinding requires an exact torch.distributed.tensor.DTensor")
    local = value.to_local()
    if hasattr(local, "wait"):
        local = local.wait()
    if type(local) is not torch.Tensor:
        raise TypeError("DTensor rebinding requires an ordinary local torch.Tensor")
    if local.layout is not torch.strided:
        raise ValueError("DTensor rebinding requires dense strided local storage")
    if local.is_meta:
        raise ValueError("DTensor rebinding requires materialized local storage")
    return local


def expected_1d_region(shard: ShardIdentity) -> LogicalRegion:
    """Return PyTorch's standard 1-D placement region for ``shard``."""

    if shard.layout is not ParameterLayout.DTENSOR_1D_DEFAULT_WORLD:
        raise ValueError("expected_1d_region requires a DTensor shard identity")
    if len(shard.placements) != 1:
        raise ValueError("one-dimensional DTensor identity requires one placement")
    placement = shard.placements[0]
    if placement.kind is PlacementKind.REPLICATE:
        return LogicalRegion.full(shard.parameter)
    if placement.kind is not PlacementKind.DIMENSION_SHARD:
        raise ValueError("one-dimensional DTensor identity requires Shard or Replicate")
    dimension = placement.parameter_dimension
    if type(dimension) is not int:
        raise ValueError("sharded DTensor identity requires a parameter dimension")
    global_length = shard.parameter.global_shape[dimension]
    chunk = (global_length + placement.parts - 1) // placement.parts
    offset = min(placement.coordinate * chunk, global_length)
    length = max(0, min(global_length, offset + chunk) - offset)
    offsets = [0] * len(shard.parameter.global_shape)
    lengths = list(shard.parameter.global_shape)
    offsets[dimension] = offset
    lengths[dimension] = length
    return LogicalRegion(tuple(offsets), tuple(lengths))


def _global_group_ranks(group) -> Tuple[int, ...]:
    try:
        size = dist.get_world_size(group)
        return tuple(dist.get_global_rank(group, rank) for rank in range(size))
    except Exception as exc:
        raise ValueError("DTensor rebinding could not resolve the DeviceMesh process group") from exc


def _validate_manifest_dtensor_shards(shards: Tuple[ShardIdentity, ...]) -> Tuple[object, str]:
    process_group = shards[0].process_group
    mesh_axis = shards[0].placements[0].mesh_axis
    for shard in shards:
        if shard.process_group != process_group:
            raise ValueError("DTensor manifest shards must share one stable process-group identity")
        if shard.placements[0].mesh_axis != mesh_axis:
            raise ValueError("DTensor manifest shards must share one semantic mesh axis")
        if shard.logical_region != expected_1d_region(shard):
            raise ValueError("DTensor manifest logical region does not match standard one-dimensional chunk geometry")
    return process_group, mesh_axis


def _validate_live_dtensor(target, shard: ShardIdentity, mesh_axis: str):
    if not is_exact_dtensor(target):
        raise TypeError("DTensor shard identity requires an exact live DTensor")
    if tuple(target.shape) != shard.parameter.global_shape:
        raise ValueError("DTensor rebinding global shape does not match its parameter identity")

    local = resolve_local_tensor(target)
    if tuple(local.shape) != shard.logical_region.lengths:
        raise ValueError("DTensor rebinding local shape does not match its logical region")
    if local.numel() != shard.logical_region.numel:
        raise ValueError("DTensor rebinding local storage size does not match its logical region")

    placements = tuple(target.placements)
    if len(placements) != 1:
        raise ValueError("DTensor rebinding requires exactly one live placement")
    live_placement = placements[0]
    descriptor = shard.placements[0]
    if descriptor.kind is PlacementKind.REPLICATE:
        if type(live_placement) is not Replicate:
            raise ValueError("DTensor rebinding placement does not match the replicated identity")
    elif (
        descriptor.kind is not PlacementKind.DIMENSION_SHARD
        or type(live_placement) is not Shard
        or type(live_placement.dim) is not int
        or live_placement.dim != descriptor.parameter_dimension
    ):
        raise ValueError("DTensor rebinding placement does not match the dimension-shard identity")

    mesh = target.device_mesh
    if tuple(mesh.shape) != (descriptor.parts,):
        raise ValueError("DTensor rebinding requires one mesh dimension matching the identity parts")
    dim_names = tuple(mesh.mesh_dim_names or ())
    if dim_names and dim_names != (mesh_axis,):
        raise ValueError("DTensor rebinding DeviceMesh dimension name does not match the semantic mesh axis")
    coordinate = mesh.get_coordinate()
    if coordinate is None or list(coordinate) != [descriptor.coordinate]:
        raise ValueError("DTensor rebinding mesh coordinate does not match the shard identity")
    return local, mesh, mesh.get_group()


def validate_dtensor_rebinding_plan(
    rebindings: Iterable, manifest: ShardingManifest
) -> Optional[Tuple[object, object]]:
    """Validate all declared and live DTensors before rebinding publication."""

    rebindings = tuple(rebindings)
    for rebinding in rebindings:
        target = rebinding.new_parameter
        declared = rebinding.shard.layout is ParameterLayout.DTENSOR_1D_DEFAULT_WORLD
        exact = is_exact_dtensor(target)
        if looks_like_dtensor(target) and not exact:
            raise TypeError("DTensor-like tensor subclasses are not supported by stable rebinding")
        if exact != declared:
            raise ValueError("live DTensor targets and DTensor shard identities must match exactly")

    manifest_shards = tuple(
        shard for shard in manifest.shards if shard.layout is ParameterLayout.DTENSOR_1D_DEFAULT_WORLD
    )
    if not manifest_shards:
        return None
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("DTensor rebinding requires an initialized default distributed process group")

    process_group, mesh_axis = _validate_manifest_dtensor_shards(manifest_shards)
    world = dist.get_world_size()
    if len(process_group.ordered_members) != world:
        raise ValueError("DTensor stable process-group identity must span the default world")

    shared_mesh = None
    shared_mesh_group = None
    for rebinding in rebindings:
        if rebinding.shard.layout is not ParameterLayout.DTENSOR_1D_DEFAULT_WORLD:
            continue
        descriptor = rebinding.shard.placements[0]
        if descriptor.parts != world:
            raise ValueError("DTensor placement parts must equal the default-world size")
        _, mesh, mesh_group = _validate_live_dtensor(rebinding.new_parameter, rebinding.shard, mesh_axis)
        if dist.get_world_size(mesh_group) != world:
            raise ValueError("DTensor DeviceMesh process group must span the default world")
        if set(_global_group_ranks(mesh_group)) != set(range(world)):
            raise ValueError("DTensor DeviceMesh process group must contain exactly the default-world ranks")
        mesh_ranks = tuple(int(rank) for rank in mesh.mesh.detach().cpu().reshape(-1).tolist())
        if set(mesh_ranks) != set(range(world)):
            raise ValueError("DTensor DeviceMesh must contain exactly the default-world ranks")
        if shared_mesh is None:
            shared_mesh = mesh
            shared_mesh_group = mesh_group
        elif mesh is not shared_mesh or mesh_group is not shared_mesh_group:
            raise ValueError("all rebound DTensors must share one DeviceMesh and process-group handle")
    return shared_mesh, shared_mesh_group


__all__ = [
    "expected_1d_region",
    "is_exact_dtensor",
    "looks_like_dtensor",
    "resolve_local_tensor",
    "validate_dtensor_rebinding_plan",
]

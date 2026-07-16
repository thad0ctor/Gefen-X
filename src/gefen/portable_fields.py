"""Pure assembly and projection for dense logical optimizer-state fields."""

from collections.abc import Sequence

import torch

from gefen.contracts import (
    LogicalRegion,
    ParameterIdentity,
    ParameterLayout,
    PlacementKind,
    ShardIdentity,
    ShardingManifest,
)
from gefen.portable import _element_chunks, _read_flat_chunk, _validate_plain_tensor


def _validate_parameter(parameter) -> ParameterIdentity:
    if not isinstance(parameter, ParameterIdentity):
        raise TypeError("parameter must be a ParameterIdentity")
    # Reconstruct the frozen descriptor so an object modified through low-level
    # attribute access cannot bypass its invariants at a checkpoint boundary.
    return ParameterIdentity(
        parameter.fqn,
        parameter.global_shape,
        schema_version=parameter.schema_version,
    )


def _validate_manifest(manifest) -> ShardingManifest:
    if not isinstance(manifest, ShardingManifest):
        raise TypeError("manifest must be a ShardingManifest")
    return ShardingManifest(
        manifest.shards,
        schema_version=manifest.schema_version,
    )


def _validate_dense_tensor(value, *, name: str, shape) -> torch.Tensor:
    value = _validate_plain_tensor(
        value,
        name=name,
        dtype=torch.float32,
    )
    if value.device.type != "cpu":
        raise ValueError("{} must be on CPU".format(name))
    if tuple(value.shape) != tuple(shape):
        raise ValueError("{} must have shape {}".format(name, tuple(shape)))
    return value


def _finish_tight(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if (
        value.device.type != "cpu"
        or value.dtype != torch.float32
        or value.requires_grad
        or not value.is_contiguous()
        or value.storage_offset() != 0
        or value.untyped_storage().nbytes() != value.numel() * value.element_size()
    ):
        raise RuntimeError("{} did not produce tight detached CPU fp32 state".format(name))
    return value


def _tight_clone(value: torch.Tensor, *, name: str) -> torch.Tensor:
    result = torch.empty(tuple(value.shape), dtype=torch.float32, device="cpu")
    flat = result.reshape(-1)
    for start, stop in _element_chunks(value.numel()):
        flat[start:stop].copy_(_read_flat_chunk(value, start, stop).resolve_conj().resolve_neg())
    return _finish_tight(result, name=name)


def _tensor_bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    for start, stop in _element_chunks(left.numel()):
        left_bytes = (
            _read_flat_chunk(left, start, stop)
            .resolve_conj()
            .resolve_neg()
            .contiguous()
            .view(torch.uint8)
        )
        right_bytes = (
            _read_flat_chunk(right, start, stop)
            .resolve_conj()
            .resolve_neg()
            .contiguous()
            .view(torch.uint8)
        )
        if not torch.equal(left_bytes, right_bytes):
            return False
    return True


def _tight_flat_slice(
    value: torch.Tensor,
    *,
    flat_offset: int,
    length: int,
    name: str,
) -> torch.Tensor:
    result = torch.empty((length,), dtype=torch.float32, device="cpu")
    for start, stop in _element_chunks(length):
        result[start:stop].copy_(
            _read_flat_chunk(
                value,
                flat_offset + start,
                flat_offset + stop,
            )
            .resolve_conj()
            .resolve_neg()
        )
    return _finish_tight(result, name=name)


def _full_output(parameter: ParameterIdentity) -> torch.Tensor:
    return torch.empty(
        parameter.global_shape,
        dtype=torch.float32,
        device="cpu",
    )


def _region_index(region: LogicalRegion):
    return tuple(slice(offset, offset + length) for offset, length in zip(region.offsets, region.lengths))


def _normalize_fragments(fragments):
    if isinstance(fragments, (str, bytes, bytearray)):
        raise TypeError("fragments must be a sequence of shard/payload pairs")
    try:
        fragments = tuple(fragments)
    except TypeError as exc:
        raise TypeError("fragments must be a sequence of shard/payload pairs") from exc

    normalized = []
    for item in fragments:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes, bytearray)) or len(item) != 2:
            raise TypeError("each fragment must be a shard/payload pair")
        shard, payload = item
        if not isinstance(shard, ShardIdentity):
            raise TypeError("each fragment identity must be a ShardIdentity")
        normalized.append((shard, payload))
    return tuple(normalized)


def _validate_fragment_set(
    manifest: ShardingManifest,
    parameter: ParameterIdentity,
    fragments,
):
    expected = manifest.for_parameter(parameter.fqn)
    if not expected:
        raise ValueError("manifest does not contain the requested parameter")
    if any(shard.parameter != parameter for shard in expected):
        raise ValueError("manifest parameter identity does not match parameter")

    fragments = _normalize_fragments(fragments)
    identities = tuple(shard for shard, _ in fragments)
    if len(set(identities)) != len(identities):
        raise ValueError("fragments must not contain duplicate shard identities")
    if set(identities) != set(expected):
        raise ValueError("fragments must exactly cover the requested parameter manifest")
    return expected, {shard: payload for shard, payload in fragments}


def _assemble_replicated(
    parameter: ParameterIdentity,
    shards,
    payload_by_shard,
    *,
    name: str,
) -> torch.Tensor:
    reference = None
    for shard in shards:
        payload = payload_by_shard[shard]
        if payload is None:
            raise ValueError("{} replicas require a payload on every shard".format(name))
        payload = _validate_dense_tensor(
            payload,
            name="{} fragment".format(name),
            shape=parameter.global_shape,
        )
        if reference is None:
            reference = payload
        elif not _tensor_bits_equal(reference, payload):
            raise ValueError("{} replicas disagree".format(name))
    if reference is None:  # ShardingManifest forbids this, but keep failure local.
        raise ValueError("{} requires at least one replica".format(name))
    return _tight_clone(reference, name="assembled {}".format(name))


def _assemble_flattened(
    parameter: ParameterIdentity,
    shards,
    payload_by_shard,
) -> torch.Tensor:
    result = _full_output(parameter)
    flat = result.reshape(-1)
    for shard in shards:
        logical_slice = shard.logical_slice
        payload = payload_by_shard[shard]
        if logical_slice.length == 0:
            if payload is not None:
                raise ValueError("an empty flattened fragment must have payload None")
            continue
        if payload is None:
            raise ValueError("a nonempty flattened fragment requires a payload")
        payload = _validate_dense_tensor(
            payload,
            name="flattened fragment",
            shape=(logical_slice.length,),
        )
        start = logical_slice.flat_offset
        flat[start : start + logical_slice.length].copy_(payload.detach())
    return _finish_tight(result, name="assembled flattened field")


def _assemble_owner(
    parameter: ParameterIdentity,
    shards,
    payload_by_shard,
) -> torch.Tensor:
    owner_payload = None
    for shard in shards:
        payload = payload_by_shard[shard]
        if shard.local_member == shard.owner:
            if payload is None:
                raise ValueError("the whole-parameter owner requires a payload")
            owner_payload = _validate_dense_tensor(
                payload,
                name="whole-parameter owner fragment",
                shape=parameter.global_shape,
            )
        elif payload is not None:
            raise ValueError("whole-parameter non-owners must have payload None")
    if owner_payload is None:
        raise ValueError("whole-parameter fragments do not contain the owner payload")
    return _tight_clone(owner_payload, name="assembled whole-parameter field")


def _assemble_dtensor_dimension_shards(
    parameter: ParameterIdentity,
    shards,
    payload_by_shard,
) -> torch.Tensor:
    result = _full_output(parameter)
    for shard in shards:
        region = shard.logical_region
        payload = payload_by_shard[shard]
        if region.numel == 0:
            if payload is not None:
                raise ValueError("an empty DTensor fragment must have payload None")
            continue
        if payload is None:
            raise ValueError("a nonempty DTensor fragment requires a payload")
        payload = _validate_dense_tensor(
            payload,
            name="DTensor fragment",
            shape=region.lengths,
        )
        result[_region_index(region)].copy_(payload.detach())
    return _finish_tight(result, name="assembled DTensor field")


def _assemble_dense_logical_field(
    manifest,
    parameter,
    fragments,
) -> torch.Tensor:
    """Assemble one complete topology-neutral fp32 CPU logical field."""

    manifest = _validate_manifest(manifest)
    parameter = _validate_parameter(parameter)
    shards, payload_by_shard = _validate_fragment_set(
        manifest,
        parameter,
        fragments,
    )
    layout = shards[0].layout
    if layout is ParameterLayout.REPLICATED:
        return _assemble_replicated(
            parameter,
            shards,
            payload_by_shard,
            name="replicated",
        )
    if layout is ParameterLayout.FLATTENED_ELEMENT_SHARD:
        return _assemble_flattened(parameter, shards, payload_by_shard)
    if layout is ParameterLayout.WHOLE_PARAMETER_OWNER:
        return _assemble_owner(parameter, shards, payload_by_shard)
    if layout is ParameterLayout.DTENSOR_1D_DEFAULT_WORLD:
        placement_kind = shards[0].placements[0].kind
        if placement_kind is PlacementKind.REPLICATE:
            return _assemble_replicated(
                parameter,
                shards,
                payload_by_shard,
                name="replicated DTensor",
            )
        if placement_kind is PlacementKind.DIMENSION_SHARD:
            return _assemble_dtensor_dimension_shards(
                parameter,
                shards,
                payload_by_shard,
            )
    raise ValueError("unsupported portable field layout")


def _project_dense_logical_field(
    parameter,
    tensor,
    shard,
):
    """Project one complete logical fp32 CPU field to a target shard identity."""

    parameter = _validate_parameter(parameter)
    if not isinstance(shard, ShardIdentity):
        raise TypeError("shard must be a ShardIdentity")
    if shard.parameter != parameter:
        raise ValueError("target shard parameter identity does not match parameter")
    tensor = _validate_dense_tensor(
        tensor,
        name="global logical field",
        shape=parameter.global_shape,
    )

    if shard.layout is ParameterLayout.REPLICATED:
        return _tight_clone(tensor, name="projected replicated field")
    if shard.layout is ParameterLayout.FLATTENED_ELEMENT_SHARD:
        logical_slice = shard.logical_slice
        if logical_slice.length == 0:
            return None
        return _tight_flat_slice(
            tensor,
            flat_offset=logical_slice.flat_offset,
            length=logical_slice.length,
            name="projected flattened field",
        )
    if shard.layout is ParameterLayout.WHOLE_PARAMETER_OWNER:
        if shard.local_member != shard.owner:
            return None
        return _tight_clone(tensor, name="projected whole-parameter field")
    if shard.layout is ParameterLayout.DTENSOR_1D_DEFAULT_WORLD:
        placement = shard.placements[0]
        if placement.kind is PlacementKind.REPLICATE:
            return _tight_clone(tensor, name="projected replicated DTensor field")
        if placement.kind is PlacementKind.DIMENSION_SHARD:
            region = shard.logical_region
            if region.numel == 0:
                return None
            values = tensor[_region_index(region)]
            return _tight_clone(values, name="projected DTensor field")
    raise ValueError("unsupported portable field layout")

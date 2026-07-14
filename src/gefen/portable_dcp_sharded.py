"""DCP-native sharded transport for portable optimizer tensors.

The helpers in this module keep logical payloads partitioned while converting
between optimizer-owned shard geometry and DCP's canonical one-dimensional
DTensor placement. Portable semantic validation and atomic publication remain
owned by :mod:`gefen.portable_runtime`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import torch
import torch.distributed as dist

from gefen.checkpoint import CheckpointProcessGroupBinding
from gefen.contracts import (
    ParameterLayout,
    PlacementKind,
    ShardIdentity,
)


_DCP_SHARDED_FORMAT = "gefen.portable_dcp_sharded"
_DCP_SHARDED_FORMAT_VERSION = 2
_DCP_SHARDED_METADATA_KEY = "__gefen_portable_metadata_v2__"
_DCP_SHARDED_FIELD_PREFIX = "__gefen_portable_field_"
_DCP_SHARDED_FIELD_SUFFIX = "__"
_DCP_SHARDED_FIELD_DIGITS = 16


def _make_phase_voter(
    binding: CheckpointProcessGroupBinding,
    *,
    transaction_id: str,
    context_digest: bytes,
    limits,
    operation_prefix: str,
):
    """Return a monotonic status voter whose digest commits to each phase label."""

    from gefen.portable_collective import _collective_unanimous_status

    ordinal = 0

    def vote(label: str, error: Exception | None) -> None:
        nonlocal ordinal

        ordinal += 1
        if type(label) is not str or not label:
            label_error = ValueError("collective phase label must be a nonempty string")
            if error is None:
                error = label_error
        label_bytes = label.encode("utf-8") if type(label) is str else b""
        phase_digest = hashlib.sha256(
            b"gefen.portable_dcp_sharded.phase.v1\0"
            + context_digest
            + ordinal.to_bytes(8, "big", signed=False)
            + len(label_bytes).to_bytes(8, "big", signed=False)
            + label_bytes
        ).digest()
        _collective_unanimous_status(
            binding,
            error,
            operation="{}_{:08d}".format(operation_prefix, ordinal),
            transaction_id=transaction_id,
            context_digest=phase_digest,
            limits=limits._wire_limits(collective=True),
        )

    return vote


@dataclass(frozen=True)
class _LogicalSegment:
    """One contiguous mapping between global and rank-local flat storage."""

    global_offset: int
    length: int
    local_offset: int

    def __post_init__(self) -> None:
        for name, value in (
            ("global_offset", self.global_offset),
            ("length", self.length),
            ("local_offset", self.local_offset),
        ):
            if type(value) is not int or value < 0:
                raise ValueError("{} must be a nonnegative exact int".format(name))


def _chunk_segment(numel: int, parts: int, coordinate: int) -> _LogicalSegment:
    """Return the standard ``torch.chunk`` interval for one mesh coordinate."""

    if type(numel) is not int or numel < 0:
        raise ValueError("numel must be a nonnegative exact int")
    if type(parts) is not int or parts < 1:
        raise ValueError("parts must be a positive exact int")
    if type(coordinate) is not int or coordinate < 0 or coordinate >= parts:
        raise ValueError("coordinate is outside the chunk partition")
    chunk = (numel + parts - 1) // parts
    offset = min(coordinate * chunk, numel)
    length = max(0, min(numel, offset + chunk) - offset)
    return _LogicalSegment(offset, length, 0)


def _canonical_segments(numel: int, parts: int):
    return tuple((_chunk_segment(numel, parts, coordinate),) for coordinate in range(parts))


def _full_segment(numel: int):
    return (_LogicalSegment(0, numel, 0),) if numel else ()


def _dimension_shard_segments(shard: ShardIdentity):
    placement = shard.placements[0]
    dimension = placement.parameter_dimension
    if placement.kind is not PlacementKind.DIMENSION_SHARD or type(dimension) is not int:
        raise ValueError("dimension-shard segments require one dimension placement")
    shape = shard.parameter.global_shape
    region = shard.logical_region
    if len(shape) == 0:
        raise ValueError("a scalar cannot use a dimension-shard placement")
    if region.numel == 0:
        return ()
    tail = math.prod(shape[dimension + 1 :])
    local_span = region.lengths[dimension] * tail
    global_span = shape[dimension] * tail
    prefix_count = math.prod(shape[:dimension])
    dimension_offset = region.offsets[dimension] * tail
    return tuple(
        _LogicalSegment(
            prefix * global_span + dimension_offset,
            local_span,
            prefix * local_span,
        )
        for prefix in range(prefix_count)
        if local_span
    )


def _dense_segments_for_shard(shard: ShardIdentity, *, unique_replicas: bool):
    """Map one parameter-shaped dense field to the live shard's flat storage."""

    if type(shard) is not ShardIdentity:
        raise TypeError("shard must be an exact ShardIdentity")
    numel = shard.parameter.numel
    placement = shard.placements[0]
    if shard.layout is ParameterLayout.REPLICATED:
        if unique_replicas and placement.coordinate != 0:
            return ()
        return _full_segment(numel)
    if shard.layout is ParameterLayout.FLATTENED_ELEMENT_SHARD:
        logical_slice = shard.logical_slice
        if logical_slice.length == 0:
            return ()
        return (
            _LogicalSegment(
                logical_slice.flat_offset,
                logical_slice.length,
                0,
            ),
        )
    if shard.layout is ParameterLayout.WHOLE_PARAMETER_OWNER:
        if shard.local_member != shard.owner:
            return ()
        return _full_segment(numel)
    if shard.layout is ParameterLayout.DTENSOR_1D_DEFAULT_WORLD:
        if placement.kind is PlacementKind.REPLICATE:
            if unique_replicas and placement.coordinate != 0:
                return ()
            return _full_segment(numel)
        return _dimension_shard_segments(shard)
    raise ValueError("unsupported dense portable shard layout")


def _special_segments_for_shard(
    shard: ShardIdentity,
    *,
    numel: int,
    unique_replicas: bool,
):
    """Map a full-vector field whose ownership follows its parameter shard."""

    placement = shard.placements[0]
    if shard.layout is ParameterLayout.REPLICATED or (
        shard.layout is ParameterLayout.DTENSOR_1D_DEFAULT_WORLD
        and placement.kind is PlacementKind.REPLICATE
    ):
        if unique_replicas and placement.coordinate != 0:
            return ()
        return _full_segment(numel)
    if shard.layout is ParameterLayout.WHOLE_PARAMETER_OWNER:
        if shard.local_member != shard.owner:
            return ()
        return _full_segment(numel)
    raise ValueError("special portable fields require replicas or a whole owner")


def _local_numel(segments) -> int:
    return max((segment.local_offset + segment.length for segment in segments), default=0)


def _validate_segments(segments, *, global_numel: int, name: str):
    try:
        segments = tuple(segments)
    except TypeError as exc:
        raise TypeError("{} segments must be iterable".format(name)) from exc
    for segment in segments:
        if type(segment) is not _LogicalSegment:
            raise TypeError("{} entries must be exact logical segments".format(name))
        if segment.global_offset + segment.length > global_numel:
            raise ValueError("{} segment exceeds the global field".format(name))
    local_intervals = sorted(
        (segment.local_offset, segment.local_offset + segment.length)
        for segment in segments
        if segment.length
    )
    cursor = 0
    for start, stop in local_intervals:
        if start != cursor:
            raise ValueError("{} local segments must cover storage exactly once".format(name))
        cursor = stop
    return segments


def _intersection(source: _LogicalSegment, target: _LogicalSegment):
    start = max(source.global_offset, target.global_offset)
    stop = min(
        source.global_offset + source.length,
        target.global_offset + target.length,
    )
    if stop <= start:
        return None
    return (
        source.local_offset + start - source.global_offset,
        target.local_offset + start - target.global_offset,
        stop - start,
    )


def _canonical_projection_send_numel(
    global_numel: int,
    *,
    target_segments_by_rank,
    parts: int,
):
    """Count exact canonical-source send elements, including target duplication."""

    if len(target_segments_by_rank) != parts:
        raise ValueError("projection target segments must match the checkpoint group")
    targets = tuple(
        _validate_segments(
            segments,
            global_numel=global_numel,
            name="projection target",
        )
        for segments in target_segments_by_rank
    )
    sources = _canonical_segments(global_numel, parts)
    full = _full_segment(global_numel)
    if all(target == full for target in targets):
        return tuple(
            source[0].length * parts if source else 0
            for source in sources
        )
    events = []
    for segments in targets:
        for segment in segments:
            if segment.length:
                events.append((segment.global_offset, 1))
                events.append((segment.global_offset + segment.length, -1))
    events.sort()
    counts = []
    event_index = 0
    multiplicity = 0
    for source_segments in sources:
        count = 0
        for source in source_segments:
            start = source.global_offset
            stop = start + source.length
            while event_index < len(events) and events[event_index][0] <= start:
                position = events[event_index][0]
                while event_index < len(events) and events[event_index][0] == position:
                    multiplicity += events[event_index][1]
                    event_index += 1
            cursor = start
            while cursor < stop:
                next_position = stop
                if event_index < len(events):
                    next_position = min(next_position, events[event_index][0])
                count += (next_position - cursor) * multiplicity
                cursor = next_position
                while event_index < len(events) and events[event_index][0] == cursor:
                    multiplicity += events[event_index][1]
                    event_index += 1
        counts.append(count)
    return tuple(counts)


def _dense_projection_send_numel(global_numel: int, *, target_shards, parts: int):
    if len(target_shards) != parts:
        raise ValueError("dense projection targets must match the checkpoint group")
    replicated = all(
        shard.layout is ParameterLayout.REPLICATED
        or (
            shard.layout is ParameterLayout.DTENSOR_1D_DEFAULT_WORLD
            and shard.placements[0].kind is PlacementKind.REPLICATE
        )
        for shard in target_shards
    )
    unique = all(
        shard.layout in {
            ParameterLayout.FLATTENED_ELEMENT_SHARD,
            ParameterLayout.WHOLE_PARAMETER_OWNER,
        }
        or (
            shard.layout is ParameterLayout.DTENSOR_1D_DEFAULT_WORLD
            and shard.placements[0].kind is PlacementKind.DIMENSION_SHARD
        )
        for shard in target_shards
    )
    if not replicated and not unique:
        raise ValueError("dense projection targets have incompatible layouts")
    multiplier = parts if replicated else 1
    return tuple(
        _chunk_segment(global_numel, parts, coordinate).length * multiplier
        for coordinate in range(parts)
    )


def _redistribute_flat_field(
    local_tensor,
    *,
    source_segments_by_rank,
    target_segments_by_rank,
    global_numel: int,
    binding: CheckpointProcessGroupBinding,
    phase_vote=None,
) -> torch.Tensor:
    """Redistribute one flat logical field without materializing it globally."""

    def finish_phase(label, error):
        if phase_vote is not None:
            phase_vote(label, error)
        if error is not None:
            raise error

    members = None
    parts = None
    coordinate = None
    sources = None
    targets = None
    local_sources = None
    source_numel = None
    dtype = None
    dtype_codes = None
    dtype_code = None
    gathered_codes = None
    error = None
    try:
        if type(binding) is not CheckpointProcessGroupBinding:
            raise TypeError("binding must be a CheckpointProcessGroupBinding")
        if phase_vote is not None and not callable(phase_vote):
            raise TypeError("phase_vote must be callable or None")
        members = binding.identity.ordered_members
        parts = len(members)
        if (
            len(source_segments_by_rank) != parts
            or len(target_segments_by_rank) != parts
        ):
            raise ValueError("segment tables must match the checkpoint process group")
        coordinate = members.index(binding.local_member)
        sources = tuple(
            _validate_segments(
                segments,
                global_numel=global_numel,
                name="source",
            )
            for segments in source_segments_by_rank
        )
        targets = tuple(
            _validate_segments(
                segments,
                global_numel=global_numel,
                name="target",
            )
            for segments in target_segments_by_rank
        )
        local_sources = sources[coordinate]
        source_numel = _local_numel(local_sources)
        if source_numel:
            if type(local_tensor) is not torch.Tensor:
                raise TypeError(
                    "a nonempty source mapping requires an exact torch.Tensor"
                )
            if (
                local_tensor.layout is not torch.strided
                or local_tensor.is_meta
                or local_tensor.is_nested
                or local_tensor.is_quantized
                or local_tensor.requires_grad
                or local_tensor.numel() != source_numel
            ):
                raise ValueError(
                    "source tensor is incompatible with its logical segments"
                )
            dtype = local_tensor.dtype
        else:
            if local_tensor is not None and (
                type(local_tensor) is not torch.Tensor
                or local_tensor.numel() != 0
            ):
                raise ValueError(
                    "an empty source mapping requires no payload or an empty tensor"
                )
            dtype = torch.float32 if local_tensor is None else local_tensor.dtype
        dtype_codes = {
            torch.uint8: 1,
            torch.int8: 2,
            torch.int16: 3,
            torch.int32: 4,
            torch.int64: 5,
            torch.float16: 6,
            torch.bfloat16: 7,
            torch.float32: 8,
            torch.float64: 9,
        }
        if dtype not in dtype_codes:
            raise TypeError(
                "portable sharded redistribution has an unsupported dtype"
            )
        dtype_code = torch.tensor(
            [dtype_codes[dtype]],
            dtype=torch.int64,
            device=binding.collective_device,
        )
        gathered_codes = [torch.empty_like(dtype_code) for _ in range(parts)]
    except Exception as exc:
        error = exc
    finish_phase("flat_pre_dtype", error)

    error = None
    if parts > 1:
        try:
            dist.all_gather(
                gathered_codes,
                dtype_code,
                group=binding.process_group,
            )
            if any(
                int(item.item()) != dtype_codes[dtype]
                for item in gathered_codes
            ):
                raise ValueError(
                    "portable sharded field dtypes disagree across ranks"
                )
        except Exception as exc:
            error = exc
    finish_phase("flat_dtype", error)

    send = None
    input_splits = None
    output_splits = None
    receive = None
    error = None
    try:
        flat = None
        if source_numel:
            flat = local_tensor.detach().reshape(-1).to(
                binding.collective_device
            )
        send_pieces = []
        input_splits = []
        for destination in range(parts):
            count = 0
            for source in local_sources:
                for target in targets[destination]:
                    overlap = _intersection(source, target)
                    if overlap is None:
                        continue
                    source_offset, _target_offset, length = overlap
                    send_pieces.append(
                        flat[source_offset : source_offset + length]
                    )
                    count += length
            input_splits.append(count)
        if send_pieces:
            send = torch.cat(send_pieces)
        else:
            send = torch.empty(
                0,
                dtype=dtype,
                device=binding.collective_device,
            )
        output_splits = []
        for source_rank in range(parts):
            count = 0
            for source in sources[source_rank]:
                for target in targets[coordinate]:
                    overlap = _intersection(source, target)
                    if overlap is not None:
                        count += overlap[2]
            output_splits.append(count)
        receive = torch.empty(
            sum(output_splits),
            dtype=dtype,
            device=binding.collective_device,
        )
    except Exception as exc:
        error = exc
    finish_phase("flat_send_receive_prep", error)

    error = None
    try:
        if parts == 1:
            receive.copy_(send)
        else:
            dist.all_to_all_single(
                receive,
                send,
                output_split_sizes=output_splits,
                input_split_sizes=input_splits,
                group=binding.process_group,
            )
    except Exception as exc:
        error = exc
    finish_phase("flat_exchange", error)

    result = None
    error = None
    try:
        target_numel = _local_numel(targets[coordinate])
        result = torch.empty(
            target_numel,
            dtype=dtype,
            device=binding.collective_device,
        )
        written = torch.zeros(
            target_numel,
            dtype=torch.bool,
            device=binding.collective_device,
        )
        receive_offset = 0
        for source_rank in range(parts):
            for source in sources[source_rank]:
                for target in targets[coordinate]:
                    overlap = _intersection(source, target)
                    if overlap is None:
                        continue
                    _source_offset, target_offset, length = overlap
                    if bool(
                        written[target_offset : target_offset + length].any()
                    ):
                        raise ValueError(
                            "portable sharded source segments overlap"
                        )
                    result[target_offset : target_offset + length].copy_(
                        receive[receive_offset : receive_offset + length]
                    )
                    written[target_offset : target_offset + length] = True
                    receive_offset += length
        if receive_offset != receive.numel() or (
            target_numel and not bool(written.all())
        ):
            raise ValueError(
                "portable sharded source segments do not cover the target"
            )
    except Exception as exc:
        error = exc
    finish_phase("flat_finish", error)
    assert result is not None
    return result


def _dense_local_numel(shard: ShardIdentity, *, unique_replicas: bool) -> int:
    placement = shard.placements[0]
    if shard.layout is ParameterLayout.REPLICATED:
        return 0 if unique_replicas and placement.coordinate != 0 else shard.parameter.numel
    if shard.layout is ParameterLayout.FLATTENED_ELEMENT_SHARD:
        return shard.logical_slice.length
    if shard.layout is ParameterLayout.WHOLE_PARAMETER_OWNER:
        return shard.parameter.numel if shard.local_member == shard.owner else 0
    if shard.layout is ParameterLayout.DTENSOR_1D_DEFAULT_WORLD:
        if placement.kind is PlacementKind.REPLICATE:
            return 0 if unique_replicas and placement.coordinate != 0 else shard.parameter.numel
        if placement.kind is PlacementKind.DIMENSION_SHARD:
            return shard.logical_region.numel
    raise ValueError("unsupported dense portable shard layout")


def _dense_global_indices(
    shard: ShardIdentity,
    *,
    unique_replicas: bool,
    device: torch.device,
) -> torch.Tensor:
    local_numel = _dense_local_numel(
        shard,
        unique_replicas=unique_replicas,
    )
    if local_numel == 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    if shard.layout in {
        ParameterLayout.REPLICATED,
        ParameterLayout.WHOLE_PARAMETER_OWNER,
    } or (
        shard.layout is ParameterLayout.DTENSOR_1D_DEFAULT_WORLD
        and shard.placements[0].kind is PlacementKind.REPLICATE
    ):
        return torch.arange(shard.parameter.numel, dtype=torch.int64, device=device)
    if shard.layout is ParameterLayout.FLATTENED_ELEMENT_SHARD:
        logical_slice = shard.logical_slice
        return torch.arange(
            logical_slice.flat_offset,
            logical_slice.flat_offset + logical_slice.length,
            dtype=torch.int64,
            device=device,
        )
    placement = shard.placements[0]
    dimension = placement.parameter_dimension
    if (
        shard.layout is not ParameterLayout.DTENSOR_1D_DEFAULT_WORLD
        or placement.kind is not PlacementKind.DIMENSION_SHARD
        or type(dimension) is not int
    ):
        raise ValueError("dense index mapping requires a supported shard")
    shape = shard.parameter.global_shape
    region = shard.logical_region
    tail = math.prod(shape[dimension + 1 :])
    local_span = region.lengths[dimension] * tail
    global_span = shape[dimension] * tail
    local = torch.arange(local_numel, dtype=torch.int64, device=device)
    prefix = torch.div(local, local_span, rounding_mode="floor")
    within = torch.remainder(local, local_span)
    return prefix.mul_(global_span).add_(
        region.offsets[dimension] * tail
    ).add_(within)


def _dense_target_offsets(global_indices: torch.Tensor, shard: ShardIdentity):
    placement = shard.placements[0]
    if shard.layout is ParameterLayout.REPLICATED or (
        shard.layout is ParameterLayout.DTENSOR_1D_DEFAULT_WORLD
        and placement.kind is PlacementKind.REPLICATE
    ):
        return torch.ones_like(global_indices, dtype=torch.bool), global_indices
    if shard.layout is ParameterLayout.WHOLE_PARAMETER_OWNER:
        if shard.local_member != shard.owner:
            return (
                torch.zeros_like(global_indices, dtype=torch.bool),
                torch.empty(0, dtype=torch.int64, device=global_indices.device),
            )
        return torch.ones_like(global_indices, dtype=torch.bool), global_indices
    if shard.layout is ParameterLayout.FLATTENED_ELEMENT_SHARD:
        logical_slice = shard.logical_slice
        mask = global_indices.ge(logical_slice.flat_offset) & global_indices.lt(
            logical_slice.flat_offset + logical_slice.length
        )
        return mask, global_indices[mask] - logical_slice.flat_offset
    dimension = placement.parameter_dimension
    if (
        shard.layout is not ParameterLayout.DTENSOR_1D_DEFAULT_WORLD
        or placement.kind is not PlacementKind.DIMENSION_SHARD
        or type(dimension) is not int
    ):
        raise ValueError("dense target mapping requires a supported shard")
    shape = shard.parameter.global_shape
    region = shard.logical_region
    tail = math.prod(shape[dimension + 1 :])
    global_span = shape[dimension] * tail
    within_span = torch.remainder(global_indices, global_span)
    dimension_coordinate = torch.div(
        within_span,
        tail,
        rounding_mode="floor",
    )
    start = region.offsets[dimension]
    length = region.lengths[dimension]
    mask = dimension_coordinate.ge(start) & dimension_coordinate.lt(start + length)
    selected = global_indices[mask]
    selected_within = within_span[mask]
    prefix = torch.div(selected, global_span, rounding_mode="floor")
    local_span = length * tail
    offsets = prefix.mul(local_span).add_(
        selected_within - start * tail
    )
    return mask, offsets


def _redistribute_indexed_dense(
    local_tensor,
    local_global_indices: torch.Tensor,
    *,
    target_shards,
    global_numel: int,
    binding: CheckpointProcessGroupBinding,
    phase_vote=None,
) -> torch.Tensor:
    def finish_phase(label, error):
        if phase_vote is not None:
            phase_vote(label, error)
        if error is not None:
            raise error

    members = None
    parts = None
    coordinate = None
    source_numel = None
    dtype = None
    flat = None
    indices = None
    dtype_codes = None
    dtype_code = None
    gathered_codes = None
    error = None
    try:
        if type(binding) is not CheckpointProcessGroupBinding:
            raise TypeError("binding must be a CheckpointProcessGroupBinding")
        if phase_vote is not None and not callable(phase_vote):
            raise TypeError("phase_vote must be callable or None")
        if type(global_numel) is not int or global_numel < 0:
            raise ValueError("global_numel must be a nonnegative exact int")
        members = binding.identity.ordered_members
        parts = len(members)
        coordinate = members.index(binding.local_member)
        if target_shards is not None and len(target_shards) != parts:
            raise ValueError("dense target shards do not match the checkpoint group")
        if type(local_global_indices) is not torch.Tensor or local_global_indices.dtype is not torch.int64:
            raise TypeError("dense redistribution indices must be int64 tensors")
        source_numel = local_global_indices.numel()
        if source_numel:
            if (
                type(local_tensor) is not torch.Tensor
                or local_tensor.layout is not torch.strided
                or local_tensor.numel() != source_numel
                or local_tensor.requires_grad
            ):
                raise ValueError("dense redistribution source tensor is incompatible")
            dtype = local_tensor.dtype
            flat = local_tensor.detach().reshape(-1).to(binding.collective_device)
        else:
            if local_tensor is not None and (
                type(local_tensor) is not torch.Tensor or local_tensor.numel() != 0
            ):
                raise ValueError("empty dense redistribution source has a payload")
            dtype = torch.float32 if local_tensor is None else local_tensor.dtype
            flat = torch.empty(0, dtype=dtype, device=binding.collective_device)
        indices = local_global_indices.to(binding.collective_device)
        if bool((indices < 0).any()) or bool((indices >= global_numel).any()):
            raise ValueError("dense redistribution indices exceed the logical field")
        dtype_codes = {
            torch.uint8: 1,
            torch.int8: 2,
            torch.int16: 3,
            torch.int32: 4,
            torch.int64: 5,
            torch.float16: 6,
            torch.bfloat16: 7,
            torch.float32: 8,
            torch.float64: 9,
        }
        if dtype not in dtype_codes:
            raise TypeError("portable sharded redistribution has an unsupported dtype")
        dtype_code = torch.tensor(
            [dtype_codes[dtype]],
            dtype=torch.int64,
            device=binding.collective_device,
        )
        gathered_codes = [torch.empty_like(dtype_code) for _ in range(parts)]
    except Exception as exc:
        error = exc
    finish_phase("indexed_pre_dtype", error)

    error = None
    if parts > 1:
        try:
            dist.all_gather(gathered_codes, dtype_code, group=binding.process_group)
            if any(int(item.item()) != dtype_codes[dtype] for item in gathered_codes):
                raise ValueError("portable sharded field dtypes disagree across ranks")
        except Exception as exc:
            error = exc
    finish_phase("indexed_dtype", error)

    input_splits = None
    send_values = None
    send_indices = None
    send_counts = None
    receive_counts = None
    error = None
    try:
        value_pieces = []
        index_pieces = []
        input_splits = []
        for destination in range(parts):
            if target_shards is None:
                segment = _chunk_segment(global_numel, parts, destination)
                mask = indices.ge(segment.global_offset) & indices.lt(segment.global_offset + segment.length)
            else:
                mask, _offsets = _dense_target_offsets(indices, target_shards[destination])
            value_pieces.append(flat[mask])
            index_pieces.append(indices[mask])
            input_splits.append(int(mask.sum().item()))
        send_values = torch.cat(value_pieces) if value_pieces else torch.empty(0, dtype=dtype, device=binding.collective_device)
        send_indices = torch.cat(index_pieces) if index_pieces else torch.empty(0, dtype=torch.int64, device=binding.collective_device)
        send_counts = torch.tensor(input_splits, dtype=torch.int64, device=binding.collective_device)
        receive_counts = torch.empty_like(send_counts)
    except Exception as exc:
        error = exc
    finish_phase("indexed_send_prep", error)

    error = None
    try:
        if parts == 1:
            receive_counts.copy_(send_counts)
        else:
            dist.all_to_all_single(receive_counts, send_counts, group=binding.process_group)
    except Exception as exc:
        error = exc
    finish_phase("indexed_count_exchange", error)

    output_splits = None
    receive_values = None
    receive_indices = None
    error = None
    try:
        output_splits = [int(value.item()) for value in receive_counts]
        if any(value < 0 for value in output_splits):
            raise ValueError("dense redistribution received a negative count")
        receive_values = torch.empty(sum(output_splits), dtype=dtype, device=binding.collective_device)
        receive_indices = torch.empty(sum(output_splits), dtype=torch.int64, device=binding.collective_device)
    except Exception as exc:
        error = exc
    finish_phase("indexed_receive_prep", error)

    error = None
    try:
        if parts == 1:
            receive_values.copy_(send_values)
        else:
            dist.all_to_all_single(
                receive_values,
                send_values,
                output_split_sizes=output_splits,
                input_split_sizes=input_splits,
                group=binding.process_group,
            )
    except Exception as exc:
        error = exc
    finish_phase("indexed_values_exchange", error)

    error = None
    try:
        if parts == 1:
            receive_indices.copy_(send_indices)
        else:
            dist.all_to_all_single(
                receive_indices,
                send_indices,
                output_split_sizes=output_splits,
                input_split_sizes=input_splits,
                group=binding.process_group,
            )
    except Exception as exc:
        error = exc
    finish_phase("indexed_indices_exchange", error)

    result = None
    error = None
    try:
        if target_shards is None:
            local_target = _chunk_segment(global_numel, parts, coordinate)
            target_numel = local_target.length
            offsets = receive_indices - local_target.global_offset
            mask = offsets.ge(0) & offsets.lt(target_numel)
        else:
            target_numel = _dense_local_numel(target_shards[coordinate], unique_replicas=False)
            mask, offsets = _dense_target_offsets(receive_indices, target_shards[coordinate])
        if not bool(mask.all()):
            raise ValueError("dense redistribution received an unowned element")
        result = torch.empty(target_numel, dtype=dtype, device=binding.collective_device)
        written = torch.zeros(target_numel, dtype=torch.bool, device=binding.collective_device)
        if offsets.numel():
            if offsets.unique().numel() != offsets.numel() or bool(written[offsets].any()):
                raise ValueError("dense redistribution source elements overlap")
            result[offsets] = receive_values
            written[offsets] = True
        if target_numel and not bool(written.all()):
            raise ValueError("dense redistribution does not cover the target")
    except Exception as exc:
        error = exc
    finish_phase("indexed_finish", error)
    assert result is not None
    return result


def _redistribute_dense_to_canonical(
    local_tensor,
    *,
    source_shard: ShardIdentity,
    global_numel: int,
    binding: CheckpointProcessGroupBinding,
    phase_vote=None,
):
    indices = None
    error = None
    try:
        indices = _dense_global_indices(
            source_shard,
            unique_replicas=True,
            device=binding.collective_device,
        )
    except Exception as exc:
        error = exc
    if phase_vote is not None:
        phase_vote("dense_source_index_prep", error)
    if error is not None:
        raise error
    assert indices is not None
    return _redistribute_indexed_dense(
        local_tensor,
        indices,
        target_shards=None,
        global_numel=global_numel,
        binding=binding,
        phase_vote=phase_vote,
    )


def _redistribute_canonical_to_dense(
    local_tensor,
    *,
    target_shards,
    global_numel: int,
    binding: CheckpointProcessGroupBinding,
    phase_vote=None,
):
    indices = None
    error = None
    try:
        coordinate = binding.identity.ordered_members.index(binding.local_member)
        source = _chunk_segment(
            global_numel,
            len(binding.identity.ordered_members),
            coordinate,
        )
        indices = torch.arange(
            source.global_offset,
            source.global_offset + source.length,
            dtype=torch.int64,
            device=binding.collective_device,
        )
    except Exception as exc:
        error = exc
    if phase_vote is not None:
        phase_vote("canonical_source_index_prep", error)
    if error is not None:
        raise error
    assert indices is not None
    return _redistribute_indexed_dense(
        local_tensor,
        indices,
        target_shards=target_shards,
        global_numel=global_numel,
        binding=binding,
        phase_vote=phase_vote,
    )


def _device_mesh(binding: CheckpointProcessGroupBinding):
    """Build a 1-D DeviceMesh around the optimizer-owned checkpoint group."""

    from torch.distributed.device_mesh import DeviceMesh

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("sharded portable DCP requires initialized torch.distributed")
    process_group = binding.process_group
    if process_group is None:
        process_group = dist.group.WORLD
    return DeviceMesh.from_group(
        process_group,
        binding.collective_device.type,
        mesh_dim_names=("gefen_portable_dcp",),
    )


def _as_canonical_dtensor(
    local_tensor: torch.Tensor,
    *,
    global_numel: int,
    binding: CheckpointProcessGroupBinding,
):
    """Wrap one standard canonical chunk for DCP without another collective."""

    from torch.distributed.tensor import DTensor, Shard

    parts = len(binding.identity.ordered_members)
    coordinate = binding.identity.ordered_members.index(binding.local_member)
    expected = _chunk_segment(global_numel, parts, coordinate)
    if type(local_tensor) is not torch.Tensor or local_tensor.numel() != expected.length:
        raise ValueError("canonical DCP local tensor has invalid chunk geometry")
    if parts == 1:
        return local_tensor
    return DTensor.from_local(
        local_tensor.contiguous(),
        _device_mesh(binding),
        (Shard(0),),
        run_check=False,
        shape=torch.Size((global_numel,)),
        stride=(1,),
    )


def _allocate_canonical_dtensor(
    *,
    global_numel: int,
    dtype: torch.dtype,
    binding: CheckpointProcessGroupBinding,
):
    parts = len(binding.identity.ordered_members)
    coordinate = binding.identity.ordered_members.index(binding.local_member)
    local = torch.empty(
        _chunk_segment(global_numel, parts, coordinate).length,
        dtype=dtype,
        device=binding.collective_device,
    )
    return _as_canonical_dtensor(
        local,
        global_numel=global_numel,
        binding=binding,
    )


def _canonical_local_tensor(value) -> torch.Tensor:
    from torch.distributed.tensor import DTensor

    if type(value) is DTensor:
        local = value.to_local()
        if hasattr(local, "wait"):
            local = local.wait()
        if type(local) is not torch.Tensor:
            raise TypeError("canonical DTensor produced a non-tensor local value")
        return local
    if type(value) is not torch.Tensor:
        raise TypeError("canonical DCP value must be a tensor or exact DTensor")
    return value


def _tensor_sha256(value: torch.Tensor) -> bytes:
    if type(value) is not torch.Tensor:
        raise TypeError("integrity hashing requires an exact torch.Tensor")
    hasher = hashlib.sha256()
    flat = value.detach().contiguous().reshape(-1).view(torch.uint8)
    chunk_bytes = 1 << 20
    for start in range(0, flat.numel(), chunk_bytes):
        chunk = flat[start : start + chunk_bytes].cpu()
        hasher.update(memoryview(chunk.numpy()))
    return hasher.digest()


def _field_key(index: int) -> str:
    if type(index) is not int or index < 0 or index >= 10**_DCP_SHARDED_FIELD_DIGITS:
        raise ValueError("portable DCP field index is out of range")
    return "{}{:0{}d}{}".format(
        _DCP_SHARDED_FIELD_PREFIX,
        index,
        _DCP_SHARDED_FIELD_DIGITS,
        _DCP_SHARDED_FIELD_SUFFIX,
    )


def _tensor_descriptor(value, *, name: str):
    if type(value) is not torch.Tensor:
        raise TypeError("{} must be an exact torch.Tensor".format(name))
    if (
        value.layout is not torch.strided
        or value.device.type != "cpu"
        or value.is_meta
        or value.is_nested
        or value.is_quantized
        or value.requires_grad
        or value.dtype is not torch.float32
        or not value.is_contiguous()
    ):
        raise TypeError("{} must be a contiguous CPU fp32 tensor".format(name))
    return {
        "dtype": "float32",
        "shape": list(value.shape),
        "sha256": _tensor_sha256(value).hex(),
    }


def _structure_local_fragment(fragment):
    """Remove tensor payloads while retaining bounded semantic descriptors."""

    if type(fragment) is not dict:
        raise TypeError("portable local fragment must be a dictionary")
    common = fragment["common"]
    codebook = common["gefen_codebook"]
    structured_common = {
        "gefen_global_step": common["gefen_global_step"],
        "gefen_codebook": (
            None
            if codebook is None
            else _tensor_descriptor(codebook, name="gefen_codebook")
        ),
        "gefen_deterministic": common["gefen_deterministic"],
    }
    slots = []
    for slot in fragment["logical_slots"]:
        state = {}
        for key, value in slot["state"].items():
            state[key] = (
                _tensor_descriptor(value, name="{} {}".format(slot["compatibility_name"], key))
                if type(value) is torch.Tensor
                else value
            )
        slots.append({**slot, "state": state})
    return {
        "format": fragment["format"],
        "format_version": fragment["format_version"],
        "coverage": fragment["coverage"],
        "implementation": fragment["implementation"],
        "member": fragment["member"],
        "policy": fragment["policy"],
        "common": structured_common,
        "manifest": fragment["manifest"],
        "catalog": fragment["catalog"],
        "logical_slots": slots,
    }


def _consensus(values, *, name: str):
    from gefen.portable_state import _values_equal

    if not values:
        raise ValueError("{} requires at least one value".format(name))
    reference = values[0]
    if any(not _values_equal(reference, value) for value in values[1:]):
        raise ValueError("{} disagree across sharded portable fragments".format(name))
    return reference


def _descriptor_shape(descriptor, *, name: str):
    if type(descriptor) is not dict or set(descriptor) != {"dtype", "shape", "sha256"}:
        raise ValueError("{} has an invalid tensor descriptor".format(name))
    if descriptor["dtype"] != "float32":
        raise TypeError("{} must use fp32".format(name))
    shape = descriptor["shape"]
    digest = descriptor["sha256"]
    if (
        type(shape) is not list
        or any(type(dimension) is not int or dimension < 0 for dimension in shape)
        or type(digest) is not str
        or len(digest) != 64
    ):
        raise ValueError("{} has an invalid tensor descriptor".format(name))
    try:
        bytes.fromhex(digest)
    except ValueError as exc:
        raise ValueError("{} has an invalid tensor digest".format(name)) from exc
    return tuple(shape)


def _add_field(fields, *, fqn, name, shape, kind, nonnegative):
    index = len(fields)
    spec = {
        "index": index,
        "key": _field_key(index),
        "fqn": fqn,
        "name": name,
        "shape": list(shape),
        "numel": math.prod(shape),
        "dtype": "float32",
        "kind": kind,
        "nonnegative": nonnegative,
    }
    fields.append(spec)
    return index


def _validate_field_descriptors(slots, *, key: str, shape_by_shard, replicated: bool):
    descriptors = []
    for slot in slots:
        from gefen.portable_identity import _parse_shard_identity

        shard = _parse_shard_identity(slot["shard"])
        descriptor = slot["state"].get(key)
        expected_shape = shape_by_shard(shard)
        if expected_shape is None:
            if descriptor is not None:
                raise ValueError("a non-payload shard carries {}".format(key))
            continue
        if descriptor is None or _descriptor_shape(descriptor, name=key) != tuple(expected_shape):
            raise ValueError("{} descriptor does not match its shard".format(key))
        descriptors.append(descriptor)
    if not descriptors:
        raise ValueError("{} requires at least one payload descriptor".format(key))
    if replicated:
        _consensus(descriptors, name="{} replicas".format(key))


def _assemble_sharded_schema(structures, *, binding: CheckpointProcessGroupBinding):
    """Assemble bounded payload metadata without assembling any dense field."""

    from gefen.portable_identity import (
        _parse_parameter_identity,
        _parse_shard_identity,
        _parse_sharding_manifest,
        _serialize_process_group_identity,
    )
    from gefen.portable_state import _local_dense_shape

    members = binding.identity.ordered_members
    structures = tuple(structures)
    if len(structures) != len(members):
        raise ValueError("sharded portable metadata requires every checkpoint member")
    if tuple(fragment["member"] for fragment in structures) != members:
        raise ValueError("sharded portable metadata is not in member order")
    implementation = _consensus(
        [fragment["implementation"] for fragment in structures],
        name="implementation",
    )
    policy = _consensus([fragment["policy"] for fragment in structures], name="policy")
    manifest_record = _consensus(
        [fragment["manifest"] for fragment in structures],
        name="manifest",
    )
    catalog = _consensus([fragment["catalog"] for fragment in structures], name="catalog")
    manifest = _parse_sharding_manifest(manifest_record)
    if any(shard.process_group != binding.identity for shard in manifest.shards):
        raise ValueError("portable manifest process groups must match the checkpoint binding")

    common_values = [fragment["common"] for fragment in structures]
    global_step = _consensus(
        [common["gefen_global_step"] for common in common_values],
        name="gefen_global_step",
    )
    deterministic = _consensus(
        [common["gefen_deterministic"] for common in common_values],
        name="gefen_deterministic",
    )
    codebooks = [common["gefen_codebook"] for common in common_values]
    fields = []
    codebook_field = None
    if any(codebook is not None for codebook in codebooks):
        if any(codebook is None for codebook in codebooks):
            raise ValueError("portable codebook presence differs across members")
        _consensus(codebooks, name="gefen_codebook")
        if _descriptor_shape(codebooks[0], name="gefen_codebook") != (256,):
            raise ValueError("portable codebook has invalid geometry")
        codebook_field = _add_field(
            fields,
            fqn=None,
            name="gefen_codebook",
            shape=(256,),
            kind="common",
            nonnegative=False,
        )

    all_slots = [slot for fragment in structures for slot in fragment["logical_slots"]]
    slots_by_shard = {_parse_shard_identity(slot["shard"]): slot for slot in all_slots}
    if len(slots_by_shard) != len(all_slots) or set(slots_by_shard) != set(manifest.shards):
        raise ValueError("sharded portable slots must exactly cover the manifest")
    parameters = {}
    for fqn in sorted(catalog):
        parameter = _parse_parameter_identity(catalog[fqn]["identity"])
        shards = manifest.for_parameter(fqn)
        slots = [slots_by_shard[shard] for shard in shards]
        _consensus(
            [(slot["group_index"], slot["original_slot_index"]) for slot in slots],
            name="logical slot position",
        )
        compatibility_name = _consensus(
            [slot["compatibility_name"] for slot in slots],
            name="compatibility_name",
        )
        required = [
            slot
            for slot, shard in zip(slots, shards)
            if _local_dense_shape(shard) is not None and parameter.numel > 0
        ]
        variants = {slot["state_variant"] for slot in required}
        if not required or variants == {"pristine"}:
            variant = "pristine"
        elif variants == {"period_selected"}:
            variant = "period_selected"
        elif len(variants) == 1 and next(iter(variants)).startswith("initialized_"):
            variant = next(iter(variants))
        else:
            raise ValueError("portable parameter shards disagree on state_variant")
        if any(slot["state_variant"] != "pristine" for slot in slots if slot not in required):
            raise ValueError("non-payload portable slots must remain pristine")
        periods = sorted(
            {
                slot["source_period"]
                for slot in required
                if slot["source_period"] is not None
            }
        )
        state = {}
        source_second = None
        if variant.startswith("initialized_"):
            step = _consensus(
                [slot["state"]["step"] for slot in required],
                name="parameter step",
            )
            if type(step) is not int or step < 1 or step > global_step:
                raise ValueError("portable parameter step is invalid")
            replicated_dense = shards[0].layout is ParameterLayout.REPLICATED or (
                shards[0].layout is ParameterLayout.DTENSOR_1D_DEFAULT_WORLD
                and shards[0].placements[0].kind is PlacementKind.REPLICATE
            )
            _validate_field_descriptors(
                slots,
                key="momentum",
                shape_by_shard=_local_dense_shape,
                replicated=replicated_dense,
            )
            state["step"] = step
            state["momentum_field"] = _add_field(
                fields,
                fqn=fqn,
                name="momentum",
                shape=parameter.global_shape,
                kind="dense",
                nonnegative=False,
            )
            if implementation == "gefen.Gefen" and variant == "initialized_dense":
                source_second = "block"
                second_step = _consensus(
                    [slot["state"]["second_moment_step"] for slot in required],
                    name="second_moment_step",
                )
                if type(second_step) is not int or second_step < 1 or second_step > step:
                    raise ValueError("portable second_moment_step is invalid")
                _validate_field_descriptors(
                    slots,
                    key="second_moment",
                    shape_by_shard=_local_dense_shape,
                    replicated=replicated_dense,
                )
                state.update(
                    {
                        "second_moment_field": _add_field(
                            fields,
                            fqn=fqn,
                            name="second_moment",
                            shape=parameter.global_shape,
                            kind="dense",
                            nonnegative=True,
                        ),
                        "second_moment_step": second_step,
                    }
                )
            elif implementation == "gefen.Gefen" and variant == "initialized_factored":
                source_second = "factored"
                if any(shard.layout is not ParameterLayout.REPLICATED for shard in shards):
                    raise ValueError("factored portable state requires replicated shards")
                factored_step = _consensus(
                    [slot["state"]["factored_step"] for slot in required],
                    name="factored_step",
                )
                if type(factored_step) is not int or factored_step < 1 or factored_step > step:
                    raise ValueError("portable factored_step is invalid")
                rows, columns = parameter.global_shape
                for key, shape in (("v_row", (rows,)), ("v_col", (columns,))):
                    _validate_field_descriptors(
                        slots,
                        key=key,
                        shape_by_shard=lambda _shard, expected=shape: expected,
                        replicated=True,
                    )
                    state["{}_field".format(key)] = _add_field(
                        fields,
                        fqn=fqn,
                        name=key,
                        shape=shape,
                        kind="special",
                        nonnegative=True,
                    )
                state["v_row_denominator_field"] = _add_field(
                    fields,
                    fqn=fqn,
                    name="v_row_denominator",
                    shape=(1,),
                    kind="special",
                    nonnegative=True,
                )
                state["factored_step"] = factored_step
            elif implementation == "gefen.GefenMuon" and variant == "initialized_dense_normuon":
                shape = (parameter.global_shape[0], 1)
                normuon_step = _consensus(
                    [slot["state"]["normuon_step"] for slot in required],
                    name="normuon_step",
                )
                if type(normuon_step) is not int or normuon_step < 1 or normuon_step > step:
                    raise ValueError("portable normuon_step is invalid")
                _validate_field_descriptors(
                    slots,
                    key="normuon_v",
                    shape_by_shard=lambda shard: (
                        shape
                        if shard.layout is ParameterLayout.REPLICATED
                        or (
                            shard.layout is ParameterLayout.WHOLE_PARAMETER_OWNER
                            and shard.local_member == shard.owner
                        )
                        else None
                    ),
                    replicated=shards[0].layout is ParameterLayout.REPLICATED,
                )
                state.update(
                    {
                        "normuon_v_field": _add_field(
                            fields,
                            fqn=fqn,
                            name="normuon_v",
                            shape=shape,
                            kind="special",
                            nonnegative=True,
                        ),
                        "normuon_step": normuon_step,
                    }
                )
        parameters[fqn] = {
            "identity": catalog[fqn]["identity"],
            "algorithm_options": catalog[fqn]["algorithm_options"],
            "group_index": slots[0]["group_index"],
            "original_slot_index": slots[0]["original_slot_index"],
            "compatibility_name": compatibility_name,
            "state_variant": variant,
            "state": state,
            "projection_hints": (
                {
                    "source_periods": periods,
                    "source_second_moment": source_second,
                    "target_period": 1,
                }
                if implementation == "gefen.Gefen"
                else {"source_periods": periods, "target_period": 1}
            ),
        }
    return {
        "format": _DCP_SHARDED_FORMAT,
        "format_version": _DCP_SHARDED_FORMAT_VERSION,
        "implementation": implementation,
        "source_process_group": _serialize_process_group_identity(binding.identity),
        "source_world_size": len(members),
        "policy": policy,
        "common": {
            "gefen_global_step": global_step,
            "gefen_codebook_field": codebook_field,
            "gefen_deterministic": deterministic,
        },
        "source_manifest": manifest_record,
        "parameters": parameters,
        "fields": fields,
    }


def _local_fragment_field(fragment, spec):
    if spec["kind"] == "common":
        return fragment["common"]["gefen_codebook"]
    slot = next(
        item
        for item in fragment["logical_slots"]
        if item["shard"]["parameter"]["fqn"] == spec["fqn"]
    )
    if spec["name"] == "v_row_denominator":
        row = slot["state"].get("v_row")
        if row is None:
            return None
        return row.mean().clamp_(min=torch.finfo(torch.float32).tiny).reshape(1)
    return slot["state"].get(spec["name"])


def _source_shards_in_member_order(manifest, fqn, members):
    shards_by_member = {
        shard.local_member: shard
        for shard in manifest.for_parameter(fqn)
    }
    if set(shards_by_member) != set(members):
        raise ValueError("portable field shards are not in checkpoint member order")
    return tuple(shards_by_member[member] for member in members)


def _source_segments_for_field(spec, manifest, *, members):
    if spec["kind"] == "common":
        return tuple(
            _full_segment(spec["numel"]) if coordinate == 0 else ()
            for coordinate in range(len(members))
        )
    shards = _source_shards_in_member_order(manifest, spec["fqn"], members)
    if spec["kind"] == "dense":
        return tuple(
            _dense_segments_for_shard(shard, unique_replicas=True)
            for shard in shards
        )
    if spec["kind"] == "special":
        return tuple(
            _special_segments_for_shard(
                shard,
                numel=spec["numel"],
                unique_replicas=True,
            )
            for shard in shards
        )
    raise ValueError("portable DCP field has an unknown geometry kind")


def _all_gather_digest(
    digest: bytes,
    binding: CheckpointProcessGroupBinding,
    *,
    phase_vote=None,
):
    def finish_phase(label, error):
        if phase_vote is not None:
            phase_vote(label, error)
        if error is not None:
            raise error

    local = None
    parts = None
    gathered = None
    error = None
    try:
        if type(digest) is not bytes or len(digest) != hashlib.sha256().digest_size:
            raise ValueError("portable field digest must be one SHA-256 value")
        local = torch.tensor(
            list(digest),
            dtype=torch.uint8,
            device=binding.collective_device,
        )
        parts = len(binding.identity.ordered_members)
        gathered = [torch.empty_like(local) for _ in range(parts)]
    except Exception as exc:
        error = exc
    finish_phase("digest_prep", error)

    error = None
    try:
        if parts == 1:
            gathered[0].copy_(local)
        else:
            dist.all_gather(gathered, local, group=binding.process_group)
    except Exception as exc:
        error = exc
    finish_phase("digest_exchange", error)

    result = None
    error = None
    try:
        result = tuple(bytes(item.cpu().tolist()) for item in gathered)
    except Exception as exc:
        error = exc
    finish_phase("digest_finish", error)
    assert result is not None
    return result


def _metadata_bytes(metadata, *, limits):
    from gefen.portable_schema import portable_state_digest
    from gefen.portable_wire import _prepare_canonical_wire_value

    semantic = {**metadata}
    semantic_digest = portable_state_digest(semantic)
    payload_hasher = hashlib.sha256()
    for field in semantic["fields"]:
        payload_hasher.update(field["key"].encode("ascii"))
        for digest in field["source_chunk_sha256"]:
            payload_hasher.update(bytes.fromhex(digest))
    completed = {
        **semantic,
        "completion": {
            "metadata_digest": semantic_digest,
            "payload_digest": payload_hasher.hexdigest(),
        },
    }
    plan = _prepare_canonical_wire_value(completed, limits._wire_limits(collective=True))
    if plan.payload_tensors:
        raise RuntimeError("sharded portable DCP metadata unexpectedly contains tensor payloads")
    return plan.metadata


def _gather_structures(
    binding: CheckpointProcessGroupBinding,
    local_structure,
    *,
    transaction_id: str,
    limits,
):
    from gefen.portable_collective import _collective_visit_canonical_fragments

    structures = []
    _collective_visit_canonical_fragments(
        binding,
        local_structure,
        operation="portable_dcp_sharded_structures",
        transaction_id=transaction_id,
        context_digest=bytes(32),
        limits=limits._wire_limits(collective=True),
        consume=lambda _member, value: structures.append(value),
    )
    return tuple(structures)


def _prepare_sharded_save_state(
    optimizer,
    *,
    binding: CheckpointProcessGroupBinding,
    transaction_id: str,
    limits,
    namespace: str,
):
    """Prepare a DCP state dict whose large fields are sharded DTensors."""

    from gefen.portable_identity import _parse_sharding_manifest
    from gefen import portable_runtime as runtime
    from gefen.portable_collective import _collective_unanimous_status

    implementation = runtime._optimizer_implementation(optimizer)
    live_token = runtime._portable_live_token(optimizer)
    local_fragment = None
    local_structure = None
    error = None
    try:
        prepared = runtime._prepare_local_structure(
            optimizer,
            implementation,
            binding,
            limits,
            include_payload=True,
        )
        runtime._validate_prepared_local_state(
            optimizer,
            implementation,
            prepared,
        )
        local_fragment = runtime._materialize_local_fragment(
            optimizer,
            implementation,
            binding,
            limits,
            prepared,
        )
        local_structure = _structure_local_fragment(local_fragment)
    except Exception as exc:
        error = exc
    _collective_unanimous_status(
        binding,
        error,
        operation="portable_dcp_sharded_save_local",
        transaction_id=transaction_id,
        context_digest=bytes(32),
        limits=limits._wire_limits(collective=True),
    )
    assert local_fragment is not None and local_structure is not None
    structures = _gather_structures(
        binding,
        local_structure,
        transaction_id=transaction_id,
        limits=limits,
    )
    metadata = None
    manifest = None
    source_segments_by_index = {}
    source_shards_by_index = {}
    local_values = {}
    schema_digest = bytes(32)
    error = None
    try:
        metadata = _assemble_sharded_schema(structures, binding=binding)
        expected_chunks = 1 + len(metadata["fields"]) * len(
            binding.identity.ordered_members
        )
        if (
            expected_chunks > limits.max_container_items
            or expected_chunks > limits.max_tree_nodes
        ):
            raise ValueError("sharded portable DCP chunks exceed aggregate limits")
        _validate_field_byte_limits(
            metadata["fields"],
            binding=binding,
            limits=limits,
        )
        manifest = _parse_sharding_manifest(metadata["source_manifest"])
        coordinate = binding.identity.ordered_members.index(binding.local_member)
        for spec in metadata["fields"]:
            local_value = _local_fragment_field(local_fragment, spec)
            if spec["kind"] == "dense":
                source_shards = _source_shards_in_member_order(
                    manifest,
                    spec["fqn"],
                    binding.identity.ordered_members,
                )
                expected_numel = _dense_local_numel(
                    source_shards[coordinate],
                    unique_replicas=True,
                )
                canonical_numel = _chunk_segment(
                    spec["numel"],
                    len(binding.identity.ordered_members),
                    coordinate,
                ).length
                if max(
                    expected_numel * 8,
                    canonical_numel * 8,
                    len(binding.identity.ordered_members) * 8,
                ) > limits.max_fragment_tensor_bytes:
                    raise ValueError(
                        "sharded portable save routing scratch exceeds max_fragment_tensor_bytes"
                    )
                source_shards_by_index[spec["index"]] = source_shards
            else:
                source_segments = _source_segments_for_field(
                    spec,
                    manifest,
                    members=binding.identity.ordered_members,
                )
                expected_numel = _local_numel(source_segments[coordinate])
                source_segments_by_index[spec["index"]] = source_segments
            if expected_numel == 0:
                local_value = None
            elif (
                type(local_value) is not torch.Tensor
                or local_value.numel() != expected_numel
                or local_value.dtype is not torch.float32
            ):
                raise ValueError("sharded portable source field geometry is invalid")
            local_values[spec["index"]] = local_value
        schema_digest = runtime._context_digest(metadata)
    except Exception as exc:
        error = exc
    _collective_unanimous_status(
        binding,
        error,
        operation="portable_dcp_sharded_save_schema",
        transaction_id=transaction_id,
        context_digest=schema_digest,
        limits=limits._wire_limits(collective=True),
    )
    assert metadata is not None and manifest is not None
    phase_vote = _make_phase_voter(
        binding,
        transaction_id=transaction_id,
        context_digest=schema_digest,
        limits=limits,
        operation_prefix="portable_dcp_sharded_save_phase",
    )
    state = {}
    for spec in metadata["fields"]:
        def field_vote(label, error, *, field_index=spec["index"]):
            phase_vote("field.{}.{}".format(field_index, label), error)

        local_value = local_values[spec["index"]]
        canonical = None
        local_digest = None
        error = None
        try:
            if spec["kind"] == "dense":
                coordinate = binding.identity.ordered_members.index(
                    binding.local_member
                )
                canonical = _redistribute_dense_to_canonical(
                    local_value,
                    source_shard=source_shards_by_index[spec["index"]][
                        coordinate
                    ],
                    global_numel=spec["numel"],
                    binding=binding,
                    phase_vote=field_vote,
                )
            else:
                source_segments = source_segments_by_index[spec["index"]]
                canonical = _redistribute_flat_field(
                    local_value,
                    source_segments_by_rank=source_segments,
                    target_segments_by_rank=_canonical_segments(
                        spec["numel"],
                        len(binding.identity.ordered_members),
                    ),
                    global_numel=spec["numel"],
                    binding=binding,
                    phase_vote=field_vote,
                )
            local_digest = _tensor_sha256(canonical)
        except Exception as exc:
            error = exc
        field_vote("projection_hash", error)
        assert canonical is not None and local_digest is not None
        chunk_digests = _all_gather_digest(
            local_digest,
            binding,
            phase_vote=field_vote,
        )
        error = None
        try:
            spec["source_chunk_sha256"] = [digest.hex() for digest in chunk_digests]
            state[spec["key"]] = _as_canonical_dtensor(
                canonical,
                global_numel=spec["numel"],
                binding=binding,
            )
        except Exception as exc:
            error = exc
        field_vote("wrap_finish", error)

    error = None
    try:
        metadata = _metadata_bytes(metadata, limits=limits)
        state[_DCP_SHARDED_METADATA_KEY] = torch.frombuffer(
            bytearray(metadata),
            dtype=torch.uint8,
        )
        runtime._validate_live_readiness(optimizer, implementation, binding)
        if live_token != runtime._portable_live_token(optimizer):
            raise RuntimeError("live optimizer state changed during sharded DCP save preparation")
    except Exception as exc:
        error = exc
    phase_vote("save.finalize", error)
    return {namespace: state}


def _strict_record(value, keys, *, name: str):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError("{} has an invalid schema".format(name))
    return value


def _strict_counter(value, *, name: str, minimum=0, maximum=(1 << 53) - 1):
    if type(value) is not int or value < minimum or value > maximum:
        raise ValueError("{} is not a bounded exact counter".format(name))
    return value


def _validate_loaded_field_spec(
    spec,
    *,
    index: int,
    source_world_size: int,
    limits,
):
    _strict_record(
        spec,
        {
            "index",
            "key",
            "fqn",
            "name",
            "shape",
            "numel",
            "dtype",
            "kind",
            "nonnegative",
            "source_chunk_sha256",
        },
        name="sharded portable field",
    )
    if spec["index"] != index or spec["key"] != _field_key(index):
        raise ValueError("sharded portable field ordering is invalid")
    if spec["fqn"] is not None and (
        type(spec["fqn"]) is not str or not spec["fqn"]
    ):
        raise ValueError("sharded portable field FQN is invalid")
    if type(spec["name"]) is not str or not spec["name"]:
        raise ValueError("sharded portable field name is invalid")
    shape = spec["shape"]
    if (
        type(shape) is not list
        or any(type(dimension) is not int or dimension < 0 for dimension in shape)
        or len(shape) > limits.max_tensor_rank
        or spec["numel"] != math.prod(shape)
        or type(spec["numel"]) is not int
        or spec["numel"] < 1
        or spec["dtype"] != "float32"
        or spec["kind"] not in {"common", "dense", "special"}
        or type(spec["nonnegative"]) is not bool
    ):
        raise ValueError("sharded portable field geometry is invalid")
    digests = spec["source_chunk_sha256"]
    if type(digests) is not list or len(digests) != source_world_size:
        raise ValueError("sharded portable field chunk digests are incomplete")
    for digest in digests:
        if type(digest) is not str or len(digest) != 64:
            raise ValueError("sharded portable field chunk digest is invalid")
        try:
            bytes.fromhex(digest)
        except ValueError as exc:
            raise ValueError("sharded portable field chunk digest is invalid") from exc
    return spec


def _validate_loaded_parameter_record(
    fqn,
    record,
    *,
    implementation,
    policy,
    manifest,
    fields,
    global_step,
    referenced_fields,
):
    from gefen.portable_identity import (
        _normalize_parameter_identity,
        _parse_parameter_identity,
    )
    from gefen.portable_state import _normalize_options

    _strict_record(
        record,
        {
            "identity",
            "algorithm_options",
            "group_index",
            "original_slot_index",
            "compatibility_name",
            "state_variant",
            "state",
            "projection_hints",
        },
        name="sharded portable parameter",
    )
    identity_record = _normalize_parameter_identity(record["identity"])
    identity = _parse_parameter_identity(identity_record)
    if identity.fqn != fqn or any(
        shard.parameter != identity for shard in manifest.for_parameter(fqn)
    ):
        raise ValueError("sharded portable parameter identity conflicts with its manifest")
    options = _normalize_options(record["algorithm_options"], implementation)
    for key in ("group_index", "original_slot_index"):
        _strict_counter(record[key], name=key)
    if (
        type(record["compatibility_name"]) is not str
        or not record["compatibility_name"]
        or record["compatibility_name"] != record["compatibility_name"].lower()
    ):
        raise ValueError("sharded portable compatibility_name is invalid")
    variant = record["state_variant"]
    state = record["state"]
    hints = record["projection_hints"]
    if implementation == "gefen.Gefen":
        _strict_record(
            hints,
            {"source_periods", "source_second_moment", "target_period"},
            name="plain sharded projection hints",
        )
    else:
        _strict_record(
            hints,
            {"source_periods", "target_period"},
            name="Muon sharded projection hints",
        )
    periods = hints["source_periods"]
    if type(periods) is not list or periods not in ([], [1]) or hints["target_period"] != 1:
        raise ValueError("sharded portable source periods are invalid")

    def field_reference(key, *, expected_name, expected_shape, expected_kind, nonnegative):
        index = state[key]
        if type(index) is not int or index < 0 or index >= len(fields):
            raise ValueError("{} has an invalid field reference".format(key))
        if index in referenced_fields:
            raise ValueError("sharded portable fields must be referenced exactly once")
        field = fields[index]
        if (
            field["fqn"] != fqn
            or field["name"] != expected_name
            or tuple(field["shape"]) != tuple(expected_shape)
            or field["kind"] != expected_kind
            or field["nonnegative"] is not nonnegative
        ):
            raise ValueError("{} refers to an incompatible field".format(key))
        referenced_fields.add(index)

    if variant == "pristine":
        _strict_record(state, set(), name="pristine sharded state")
        if periods:
            raise ValueError("pristine sharded state has source periods")
    elif variant == "period_selected":
        _strict_record(state, set(), name="period-selected sharded state")
        if periods != [1]:
            raise ValueError("period-selected sharded state requires period one")
    elif implementation == "gefen.Gefen" and variant == "initialized_dense":
        _strict_record(
            state,
            {"step", "momentum_field", "second_moment_field", "second_moment_step"},
            name="dense sharded state",
        )
        step = _strict_counter(state["step"], name="step", minimum=1)
        second_step = _strict_counter(
            state["second_moment_step"],
            name="second_moment_step",
            minimum=1,
        )
        if step > global_step or second_step > step or periods != [1]:
            raise ValueError("dense sharded state counters are invalid")
        if hints["source_second_moment"] != "block" or options["second_moment_policy"] != "block":
            raise ValueError("dense sharded state conflicts with its second-moment policy")
        field_reference(
            "momentum_field",
            expected_name="momentum",
            expected_shape=identity.global_shape,
            expected_kind="dense",
            nonnegative=False,
        )
        field_reference(
            "second_moment_field",
            expected_name="second_moment",
            expected_shape=identity.global_shape,
            expected_kind="dense",
            nonnegative=True,
        )
    elif implementation == "gefen.Gefen" and variant == "initialized_factored":
        _strict_record(
            state,
            {
                "step",
                "momentum_field",
                "v_row_field",
                "v_col_field",
                "v_row_denominator_field",
                "factored_step",
            },
            name="factored sharded state",
        )
        step = _strict_counter(state["step"], name="step", minimum=1)
        factored_step = _strict_counter(
            state["factored_step"],
            name="factored_step",
            minimum=1,
        )
        if step > global_step or factored_step > step or periods != [1]:
            raise ValueError("factored sharded state counters are invalid")
        if (
            hints["source_second_moment"] != "factored"
            or options["second_moment_policy"] != "factored"
            or len(identity.global_shape) != 2
        ):
            raise ValueError("factored sharded state conflicts with its policy")
        field_reference(
            "momentum_field",
            expected_name="momentum",
            expected_shape=identity.global_shape,
            expected_kind="dense",
            nonnegative=False,
        )
        field_reference(
            "v_row_field",
            expected_name="v_row",
            expected_shape=(identity.global_shape[0],),
            expected_kind="special",
            nonnegative=True,
        )
        field_reference(
            "v_col_field",
            expected_name="v_col",
            expected_shape=(identity.global_shape[1],),
            expected_kind="special",
            nonnegative=True,
        )
        field_reference(
            "v_row_denominator_field",
            expected_name="v_row_denominator",
            expected_shape=(1,),
            expected_kind="special",
            nonnegative=True,
        )
    elif implementation == "gefen.GefenMuon" and variant in {
        "initialized_dense",
        "initialized_dense_normuon",
    }:
        expected = {"step", "momentum_field"}
        if variant == "initialized_dense_normuon":
            expected.update({"normuon_v_field", "normuon_step"})
        _strict_record(state, expected, name="Muon sharded state")
        step = _strict_counter(state["step"], name="step", minimum=1)
        if step > global_step or periods != [1]:
            raise ValueError("Muon sharded state counters are invalid")
        if options["normuon"] is not (variant == "initialized_dense_normuon"):
            raise ValueError("Muon sharded state conflicts with its NorMuon policy")
        field_reference(
            "momentum_field",
            expected_name="momentum",
            expected_shape=identity.global_shape,
            expected_kind="dense",
            nonnegative=False,
        )
        if variant == "initialized_dense_normuon":
            normuon_step = _strict_counter(
                state["normuon_step"],
                name="normuon_step",
                minimum=1,
            )
            if normuon_step > step:
                raise ValueError("normuon_step exceeds parameter step")
            field_reference(
                "normuon_v_field",
                expected_name="normuon_v",
                expected_shape=(identity.global_shape[0], 1),
                expected_kind="special",
                nonnegative=True,
            )
    else:
        raise ValueError("sharded portable parameter state_variant is unsupported")
    return {
        **record,
        "identity": identity_record,
        "algorithm_options": options,
    }


def _normalize_sharded_metadata(value, *, limits):
    from gefen.portable_identity import (
        _normalize_process_group_identity,
        _normalize_sharding_manifest,
        _parse_process_group_identity,
        _parse_sharding_manifest,
    )
    from gefen.portable_schema import portable_state_digest
    from gefen.portable_state import (
        _IMPLEMENTATIONS,
        _normalize_catalog,
        _normalize_policy,
    )

    _strict_record(
        value,
        {
            "format",
            "format_version",
            "implementation",
            "source_process_group",
            "source_world_size",
            "policy",
            "common",
            "source_manifest",
            "parameters",
            "fields",
            "completion",
        },
        name="sharded portable metadata",
    )
    if value["format"] != _DCP_SHARDED_FORMAT or value["format_version"] != _DCP_SHARDED_FORMAT_VERSION:
        raise ValueError("unsupported sharded portable DCP format")
    implementation = value["implementation"]
    if implementation not in _IMPLEMENTATIONS:
        raise ValueError("unsupported sharded portable implementation")
    process_group_record = _normalize_process_group_identity(value["source_process_group"])
    process_group = _parse_process_group_identity(process_group_record)
    source_world_size = value["source_world_size"]
    if (
        type(source_world_size) is not int
        or source_world_size < 1
        or source_world_size > limits.max_members
        or source_world_size != len(process_group.ordered_members)
    ):
        raise ValueError("sharded portable source world size is invalid")
    policy = _normalize_policy(value["policy"], implementation)
    manifest_record = _normalize_sharding_manifest(value["source_manifest"])
    manifest = _parse_sharding_manifest(manifest_record)
    if any(shard.process_group != process_group for shard in manifest.shards):
        raise ValueError("sharded portable source manifest has a different process group")
    common = _strict_record(
        value["common"],
        {"gefen_global_step", "gefen_codebook_field", "gefen_deterministic"},
        name="sharded portable common state",
    )
    global_step = _strict_counter(common["gefen_global_step"], name="gefen_global_step")
    if type(common["gefen_deterministic"]) is not bool:
        raise ValueError("sharded portable deterministic policy must be bool")
    fields_value = value["fields"]
    if (
        type(fields_value) is not list
        or len(fields_value) > limits.max_tensors
        or len(fields_value) > limits.max_container_items
    ):
        raise ValueError("sharded portable field count exceeds limits")
    fields = [
        _validate_loaded_field_spec(
            spec,
            index=index,
            source_world_size=source_world_size,
            limits=limits,
        )
        for index, spec in enumerate(fields_value)
    ]
    codebook_field = common["gefen_codebook_field"]
    referenced_fields = set()
    if codebook_field is not None:
        if type(codebook_field) is not int or codebook_field < 0 or codebook_field >= len(fields):
            raise ValueError("sharded portable codebook field is invalid")
        field = fields[codebook_field]
        if (
            field["fqn"] is not None
            or field["name"] != "gefen_codebook"
            or field["shape"] != [256]
            or field["kind"] != "common"
            or field["nonnegative"]
        ):
            raise ValueError("sharded portable codebook field is incompatible")
        referenced_fields.add(codebook_field)
    parameters_value = value["parameters"]
    manifest_fqns = {shard.parameter.fqn for shard in manifest.shards}
    if type(parameters_value) is not dict or set(parameters_value) != manifest_fqns:
        raise ValueError("sharded portable parameters do not match the manifest")
    parameters = {
        fqn: _validate_loaded_parameter_record(
            fqn,
            parameters_value[fqn],
            implementation=implementation,
            policy=policy,
            manifest=manifest,
            fields=fields,
            global_step=global_step,
            referenced_fields=referenced_fields,
        )
        for fqn in sorted(parameters_value)
    }
    allowed_layouts = (
        {
            ParameterLayout.REPLICATED,
            ParameterLayout.FLATTENED_ELEMENT_SHARD,
            ParameterLayout.DTENSOR_1D_DEFAULT_WORLD,
        }
        if implementation == "gefen.Gefen"
        else {
            ParameterLayout.REPLICATED,
            ParameterLayout.WHOLE_PARAMETER_OWNER,
        }
    )
    if any(shard.layout not in allowed_layouts for shard in manifest.shards):
        raise ValueError("sharded portable source layout is unsupported for its implementation")
    _normalize_catalog(
        {
            fqn: {
                "identity": record["identity"],
                "algorithm_options": record["algorithm_options"],
            }
            for fqn, record in parameters.items()
        },
        implementation,
        manifest,
        policy,
    )
    for fqn, record in parameters.items():
        shards = manifest.for_parameter(fqn)
        if record["state_variant"] == "initialized_factored" and any(
            shard.layout is not ParameterLayout.REPLICATED for shard in shards
        ):
            raise ValueError("factored sharded portable state requires replicated source shards")
        if (
            implementation == "gefen.GefenMuon"
            and any(shard.layout is ParameterLayout.WHOLE_PARAMETER_OWNER for shard in shards)
            and record["algorithm_options"]["sharded_mode"] != "distributed"
        ):
            raise ValueError("whole-parameter Muon state requires sharded_mode='distributed'")
    if referenced_fields != set(range(len(fields))):
        raise ValueError("sharded portable metadata contains unreferenced fields")
    if codebook_field is None and any(
        record["state_variant"].startswith("initialized_")
        for record in parameters.values()
    ):
        raise ValueError("initialized sharded portable state requires a codebook")
    completion = _strict_record(
        value["completion"],
        {"metadata_digest", "payload_digest"},
        name="sharded portable completion",
    )
    for key in ("metadata_digest", "payload_digest"):
        digest = completion[key]
        if type(digest) is not str or len(digest) != 64:
            raise ValueError("sharded portable completion digest is invalid")
        try:
            bytes.fromhex(digest)
        except ValueError as exc:
            raise ValueError("sharded portable completion digest is invalid") from exc
    semantic = {key: value[key] for key in value if key != "completion"}
    if portable_state_digest(semantic) != completion["metadata_digest"]:
        raise ValueError("sharded portable metadata digest mismatch")
    payload_hasher = hashlib.sha256()
    for field in fields:
        payload_hasher.update(field["key"].encode("ascii"))
        for digest in field["source_chunk_sha256"]:
            payload_hasher.update(bytes.fromhex(digest))
    if payload_hasher.hexdigest() != completion["payload_digest"]:
        raise ValueError("sharded portable payload digest mismatch")
    return {
        **value,
        "source_process_group": process_group_record,
        "policy": policy,
        "source_manifest": manifest_record,
        "parameters": parameters,
        "fields": fields,
    }


def _validate_field_byte_limits(
    fields,
    *,
    binding: CheckpointProcessGroupBinding,
    limits,
):
    coordinate = binding.identity.ordered_members.index(binding.local_member)
    parts = len(binding.identity.ordered_members)
    element_size = torch.tensor([], dtype=torch.float32).element_size()
    global_bytes = 0
    local_bytes = 0
    for field in fields:
        field_bytes = field["numel"] * element_size
        if global_bytes > limits.max_collective_tensor_bytes - field_bytes:
            raise ValueError("sharded portable fields exceed max_collective_tensor_bytes")
        global_bytes += field_bytes
        local_chunk_bytes = (
            _chunk_segment(field["numel"], parts, coordinate).length
            * element_size
        )
        if local_bytes > limits.max_fragment_tensor_bytes - local_chunk_bytes:
            raise ValueError("sharded portable local fields exceed max_fragment_tensor_bytes")
        local_bytes += local_chunk_bytes
    return global_bytes, local_bytes


def _validate_source_storage_chunk_limit(length: int, *, limits):
    if type(length) is not int or length < 0:
        raise ValueError("sharded portable DCP source storage chunk has invalid size")
    if length * torch.tensor([], dtype=torch.float32).element_size() > limits.max_fragment_tensor_bytes:
        raise ValueError("sharded portable DCP source storage chunk exceeds max_fragment_tensor_bytes")


def _decode_metadata_tensor(value, *, limits):
    from gefen.portable_wire import (
        _parse_canonical_wire_metadata,
        _reconstruct_canonical_wire_value,
    )

    if (
        type(value) is not torch.Tensor
        or value.dtype is not torch.uint8
        or value.device.type != "cpu"
        or value.ndim != 1
        or not value.is_contiguous()
        or value.numel() < 1
        or value.numel() > limits.max_metadata_bytes
    ):
        raise ValueError("sharded portable metadata tensor is invalid")
    raw = bytes(memoryview(value.numpy()))
    prepared = _parse_canonical_wire_metadata(
        raw,
        limits=limits._wire_limits(collective=True),
    )
    if prepared.tensor_specs:
        raise ValueError("sharded portable metadata must not contain tensor payloads")
    document = _reconstruct_canonical_wire_value(prepared, ())
    return _normalize_sharded_metadata(document, limits=limits)


def _metadata_storage_spec(checkpoint_metadata, *, namespace: str, limits):
    from torch.distributed.checkpoint.metadata import Metadata
    from gefen.portable_dcp import _flat_key, _validate_tensor_metadata

    if type(checkpoint_metadata) is not Metadata:
        raise TypeError("portable DCP requires exact checkpoint Metadata")
    entries = checkpoint_metadata.state_dict_metadata
    if type(entries) is not dict:
        raise TypeError("portable DCP metadata entries must be a dict")
    key = _flat_key(namespace, _DCP_SHARDED_METADATA_KEY)
    if key not in entries:
        return None
    dtype, shape, nbytes = _validate_tensor_metadata(
        entries[key],
        name="sharded portable DCP metadata",
        limits=limits._wire_limits(collective=True),
    )
    if dtype is not torch.uint8 or len(shape) != 1 or nbytes < 1 or nbytes > limits.max_metadata_bytes:
        raise ValueError("sharded portable DCP metadata exceeds limits")
    return shape


def _read_sharded_metadata(
    *,
    storage_reader,
    checkpoint_metadata,
    binding: CheckpointProcessGroupBinding,
    namespace: str,
    limits,
    transaction_id: str,
):
    import torch.distributed.checkpoint as dcp
    from gefen.portable_collective import _collective_unanimous_status

    shape = None
    tensor = None
    state = None
    planner = None
    error = None
    try:
        shape = _metadata_storage_spec(
            checkpoint_metadata,
            namespace=namespace,
            limits=limits,
        )
        if shape is not None:
            tensor = torch.empty(shape, dtype=torch.uint8, device="cpu")
            state = {namespace: {_DCP_SHARDED_METADATA_KEY: tensor}}
            planner = dcp.DefaultLoadPlanner(allow_partial_load=True)
    except Exception as exc:
        error = exc
    shape_bytes = repr(tuple(shape) if shape is not None else None).encode("ascii")
    _collective_unanimous_status(
        binding,
        error,
        operation="portable_dcp_sharded_metadata_prepare",
        transaction_id=transaction_id,
        context_digest=hashlib.sha256(
            b"gefen.portable_dcp_sharded.metadata.v1\0"
            + namespace.encode("utf-8")
            + b"\0"
            + shape_bytes
        ).digest(),
        limits=limits._wire_limits(collective=True),
    )
    if shape is None:
        return None
    assert tensor is not None and state is not None and planner is not None
    dcp.load(
        state,
        storage_reader=storage_reader,
        planner=planner,
        process_group=binding.process_group,
    )
    return _decode_metadata_tensor(tensor, limits=limits)


def _validate_sharded_dcp_entries(
    checkpoint_metadata,
    metadata,
    *,
    namespace: str,
    binding: CheckpointProcessGroupBinding,
    limits,
    checkpoint_keys=None,
):
    from torch.distributed.checkpoint.metadata import (
        ChunkStorageMetadata,
        TensorProperties,
        TensorStorageMetadata,
    )
    from gefen.portable_dcp import _flat_key

    entries = checkpoint_metadata.state_dict_metadata
    metadata_key = _flat_key(namespace, _DCP_SHARDED_METADATA_KEY)
    child_keys = {metadata_key}
    for field in metadata["fields"]:
        child_keys.add(_flat_key(namespace, field["key"]))
    expected = child_keys if checkpoint_keys is None else set(checkpoint_keys)
    if (
        not child_keys.issubset(expected)
        or set(entries) != expected
        or any(type(key) is not str for key in entries)
    ):
        raise ValueError("sharded portable DCP checkpoint has unexpected tensor keys")
    planner_data = checkpoint_metadata.planner_data
    expected_paths = {key: tuple(key.split(".", 1)) for key in expected}
    if type(planner_data) is not dict or planner_data != expected_paths:
        raise ValueError("sharded portable DCP checkpoint has incompatible planner paths")
    _validate_field_byte_limits(
        metadata["fields"],
        binding=binding,
        limits=limits,
    )
    current_parts = len(binding.identity.ordered_members)
    from gefen.portable_dcp import _validate_tensor_metadata

    _validate_tensor_metadata(
        entries[metadata_key],
        name="sharded portable DCP metadata",
        limits=limits._wire_limits(collective=True),
    )
    total_chunks = len(entries[metadata_key].chunks)
    if (
        total_chunks > limits.max_container_items
        or total_chunks > limits.max_tree_nodes
    ):
        raise ValueError("sharded portable DCP chunks exceed aggregate limits")
    for field in metadata["fields"]:
        _validate_integrity_assignment_limits(
            field,
            target_parts=current_parts,
            limits=limits,
        )
        entry = entries[_flat_key(namespace, field["key"])]
        if type(entry) is not TensorStorageMetadata:
            raise TypeError("sharded portable field requires tensor DCP metadata")
        properties = entry.properties
        if (
            type(properties) is not TensorProperties
            or properties.dtype is not torch.float32
            or properties.layout is not torch.strided
            or properties.requires_grad is not False
            or properties.memory_format is not torch.contiguous_format
            or properties.pin_memory is not False
            or type(entry.size) is not torch.Size
            or any(type(dimension) is not int for dimension in entry.size)
            or tuple(entry.size) != (field["numel"],)
            or type(entry.chunks) is not list
            or not entry.chunks
            or len(entry.chunks) > limits.max_container_items
        ):
            raise ValueError("sharded portable DCP field metadata is incompatible")
        total_chunks += len(entry.chunks)
        if (
            total_chunks > limits.max_container_items
            or total_chunks > limits.max_tree_nodes
        ):
            raise ValueError("sharded portable DCP chunks exceed aggregate limits")
        intervals = []
        for chunk in entry.chunks:
            if (
                type(chunk) is not ChunkStorageMetadata
                or type(chunk.offsets) is not torch.Size
                or type(chunk.sizes) is not torch.Size
                or len(chunk.offsets) != 1
                or len(chunk.sizes) != 1
                or type(chunk.offsets[0]) is not int
                or type(chunk.sizes[0]) is not int
            ):
                raise ValueError("sharded portable DCP field chunk is invalid")
            offset = chunk.offsets[0]
            length = chunk.sizes[0]
            if offset < 0 or length < 0 or offset + length > field["numel"]:
                raise ValueError("sharded portable DCP field chunk exceeds its tensor")
            _validate_source_storage_chunk_limit(length, limits=limits)
            intervals.append((offset, offset + length))
        intervals.sort()
        cursor = 0
        for start, stop in intervals:
            if start != cursor:
                raise ValueError("sharded portable DCP chunks must cover each field exactly once")
            cursor = stop
        if cursor != field["numel"]:
            raise ValueError("sharded portable DCP chunks do not cover their field")
        local_chunk = _chunk_segment(
            field["numel"],
            current_parts,
            binding.identity.ordered_members.index(binding.local_member),
        )
        if local_chunk.length * torch.tensor([], dtype=torch.float32).element_size() > limits.max_fragment_tensor_bytes:
            raise ValueError("sharded portable DCP local field exceeds max_fragment_tensor_bytes")


def _allocate_sharded_load_state(
    metadata,
    *,
    metadata_tensor: torch.Tensor,
    namespace: str,
    binding: CheckpointProcessGroupBinding,
):
    envelope = {_DCP_SHARDED_METADATA_KEY: metadata_tensor}
    for field in metadata["fields"]:
        envelope[field["key"]] = _allocate_canonical_dtensor(
            global_numel=field["numel"],
            dtype=torch.float32,
            binding=binding,
        )
    return {namespace: envelope}


def _source_chunk_assignments(numel: int, source_parts: int, target_parts: int):
    assigned = [[] for _ in range(target_parts)]
    local_offsets = [0] * target_parts
    for source_coordinate in range(source_parts):
        target_coordinate = source_coordinate % target_parts
        source_chunk = _chunk_segment(numel, source_parts, source_coordinate)
        assigned[target_coordinate].append(
            (
                source_coordinate,
                _LogicalSegment(
                    source_chunk.global_offset,
                    source_chunk.length,
                    local_offsets[target_coordinate],
                ),
            )
        )
        local_offsets[target_coordinate] += source_chunk.length
    return tuple(tuple(items) for items in assigned)


def _validate_integrity_assignment_limits(field, *, target_parts: int, limits):
    assignments = _source_chunk_assignments(
        field["numel"],
        len(field["source_chunk_sha256"]),
        target_parts,
    )
    element_size = torch.tensor([], dtype=torch.float32).element_size()
    source_parts = len(field["source_chunk_sha256"])
    if max(
        source_parts * hashlib.sha256().digest_size * 4,
        source_parts * 4,
    ) > limits.max_fragment_tensor_bytes:
        raise ValueError("sharded portable integrity digest scratch exceeds max_fragment_tensor_bytes")
    for items in assignments:
        local_numel = max(
            (
                segment.local_offset + segment.length
                for _source_coordinate, segment in items
            ),
            default=0,
        )
        if local_numel * element_size > limits.max_fragment_tensor_bytes:
            raise ValueError("sharded portable integrity scratch exceeds max_fragment_tensor_bytes")


def _verify_field_integrity(
    local_canonical: torch.Tensor,
    field,
    *,
    binding: CheckpointProcessGroupBinding,
    phase_vote=None,
):
    def finish_phase(label, error):
        if phase_vote is not None:
            phase_vote(label, error)
        if error is not None:
            raise error

    target_parts = None
    source_parts = None
    assignments = None
    assigned_values = None
    error = None
    try:
        target_parts = len(binding.identity.ordered_members)
        source_parts = len(field["source_chunk_sha256"])
        assignments = _source_chunk_assignments(field["numel"], source_parts, target_parts)
        target_segments = tuple(tuple(segment for _source_coordinate, segment in items) for items in assignments)
        source_segments = _canonical_segments(field["numel"], target_parts)
        assigned_values = _redistribute_flat_field(
            local_canonical,
            source_segments_by_rank=source_segments,
            target_segments_by_rank=target_segments,
            global_numel=field["numel"],
            binding=binding,
            phase_vote=phase_vote,
        )
    except Exception as exc:
        error = exc
    finish_phase("integrity_projection", error)

    digests = None
    owners = None
    error = None
    try:
        coordinate = binding.identity.ordered_members.index(binding.local_member)
        digests = torch.zeros(
            (source_parts, hashlib.sha256().digest_size),
            dtype=torch.int32,
            device=binding.collective_device,
        )
        owners = torch.zeros(source_parts, dtype=torch.int32, device=binding.collective_device)
        for source_coordinate, segment in assignments[coordinate]:
            value = assigned_values[segment.local_offset : segment.local_offset + segment.length]
            digest = _tensor_sha256(value)
            digests[source_coordinate].copy_(torch.tensor(list(digest), dtype=torch.int32, device=binding.collective_device))
            owners[source_coordinate] = 1
    except Exception as exc:
        error = exc
    finish_phase("integrity_reduce_prep", error)

    error = None
    try:
        if target_parts > 1:
            dist.all_reduce(digests, op=dist.ReduceOp.SUM, group=binding.process_group)
    except Exception as exc:
        error = exc
    finish_phase("integrity_digest_reduce", error)

    error = None
    try:
        if target_parts > 1:
            dist.all_reduce(owners, op=dist.ReduceOp.SUM, group=binding.process_group)
    except Exception as exc:
        error = exc
    finish_phase("integrity_owner_reduce", error)

    error = None
    try:
        if not bool((owners == 1).all()):
            raise RuntimeError("portable DCP integrity chunks do not have one verifier")
        actual = [bytes(row.to(torch.uint8).cpu().tolist()).hex() for row in digests]
        if actual != field["source_chunk_sha256"]:
            raise ValueError("sharded portable DCP field integrity digest mismatch")
    except Exception as exc:
        error = exc
    finish_phase("integrity_finish", error)


def _load_sharded_payloads(
    *,
    storage_reader,
    checkpoint_metadata,
    metadata,
    metadata_tensor: torch.Tensor | None = None,
    metadata_shape=None,
    binding: CheckpointProcessGroupBinding,
    namespace: str,
    limits,
    transaction_id: str,
    context_digest: bytes,
    checkpoint_keys=None,
):
    import torch.distributed.checkpoint as dcp
    from gefen.portable_collective import _collective_unanimous_status

    state = None
    planner = None
    error = None
    try:
        if metadata_tensor is None:
            if metadata_shape is None:
                raise ValueError("sharded portable metadata allocation requires its shape")
            metadata_tensor = torch.empty(metadata_shape, dtype=torch.uint8, device="cpu")
        _validate_sharded_dcp_entries(
            checkpoint_metadata,
            metadata,
            namespace=namespace,
            binding=binding,
            limits=limits,
            checkpoint_keys=checkpoint_keys,
        )
        state = _allocate_sharded_load_state(
            metadata,
            metadata_tensor=metadata_tensor,
            namespace=namespace,
            binding=binding,
        )
        planner = dcp.DefaultLoadPlanner()
    except Exception as exc:
        error = exc
    _collective_unanimous_status(
        binding,
        error,
        operation="portable_dcp_sharded_load_allocate",
        transaction_id=transaction_id,
        context_digest=context_digest,
        limits=limits._wire_limits(collective=True),
    )
    assert state is not None and planner is not None
    error = None
    try:
        dcp.load(
            state,
            storage_reader=storage_reader,
            planner=planner,
            process_group=binding.process_group,
        )
    except Exception as exc:
        error = exc
    _collective_unanimous_status(
        binding,
        error,
        operation="portable_dcp_sharded_load_dcp_complete",
        transaction_id=transaction_id,
        context_digest=context_digest,
        limits=limits._wire_limits(collective=True),
    )
    envelope = state[namespace]
    local_fields = {}
    phase_vote = _make_phase_voter(
        binding,
        transaction_id=transaction_id,
        context_digest=context_digest,
        limits=limits,
        operation_prefix="portable_dcp_sharded_integrity_phase",
    )
    for field in metadata["fields"]:
        def field_vote(label, error, *, field_index=field["index"]):
            phase_vote("field.{}.{}".format(field_index, label), error)

        local = None
        error = None
        try:
            local = _canonical_local_tensor(envelope[field["key"]])
            if local.dtype is not torch.float32 or local.ndim != 1:
                raise TypeError("sharded portable DCP loaded an incompatible field")
            for start in range(0, local.numel(), 1 << 20):
                chunk = local[start : start + (1 << 20)]
                if not bool(torch.isfinite(chunk).all()):
                    raise ValueError("sharded portable DCP field must be finite")
                if field["nonnegative"] and not bool((chunk >= 0).all()):
                    raise ValueError("sharded portable DCP field must be nonnegative")
        except Exception as exc:
            error = exc
        field_digest = hashlib.sha256(
            context_digest
            + field["index"].to_bytes(8, "big", signed=False)
        ).digest()
        _collective_unanimous_status(
            binding,
            error,
            operation="portable_dcp_sharded_load_field",
            transaction_id=transaction_id,
            context_digest=field_digest,
            limits=limits._wire_limits(collective=True),
        )
        assert local is not None
        _verify_field_integrity(
            local,
            field,
            binding=binding,
            phase_vote=field_vote,
        )
        local_fields[field["index"]] = local
    return local_fields


def _target_shards_in_member_order(manifest, fqn, binding):
    by_member = {
        shard.local_member: shard for shard in manifest.for_parameter(fqn)
    }
    if set(by_member) != set(binding.identity.ordered_members):
        raise ValueError("portable target shards do not match checkpoint members")
    return tuple(by_member[member] for member in binding.identity.ordered_members)


def _project_loaded_field(
    local_canonical,
    field,
    *,
    target_shards,
    binding: CheckpointProcessGroupBinding,
    replicate=False,
    phase_vote=None,
):
    mode = None
    target_segments = None
    result = None
    error = None
    try:
        if not replicate and field["kind"] == "dense":
            mode = "dense"
        else:
            mode = "flat"
            if replicate or field["kind"] == "common":
                target_segments = tuple(
                    _full_segment(field["numel"])
                    for _member in binding.identity.ordered_members
                )
            elif field["kind"] == "special":
                target_segments = tuple(
                    _special_segments_for_shard(
                        shard,
                        numel=field["numel"],
                        unique_replicas=False,
                    )
                    for shard in target_shards
                )
            else:
                raise ValueError("portable loaded field has an unknown geometry kind")
    except Exception as exc:
        error = exc
    if phase_vote is not None:
        phase_vote("projection_prep", error)
    if error is not None:
        raise error

    error = None
    try:
        if mode == "dense":
            result = _redistribute_canonical_to_dense(
                local_canonical,
                target_shards=target_shards,
                global_numel=field["numel"],
                binding=binding,
                phase_vote=phase_vote,
            )
        else:
            assert target_segments is not None
            result = _redistribute_flat_field(
                local_canonical,
                source_segments_by_rank=_canonical_segments(
                    field["numel"],
                    len(binding.identity.ordered_members),
                ),
                target_segments_by_rank=target_segments,
                global_numel=field["numel"],
                binding=binding,
                phase_vote=phase_vote,
            )
        result = result.cpu()
    except Exception as exc:
        error = exc
    if phase_vote is not None:
        phase_vote("projection_finish", error)
    if error is not None:
        raise error
    assert result is not None
    return result


def _factored_axis_segments(shard: ShardIdentity, *, axis: int):
    """Return the sorted compact factor slice needed by one target shard."""

    if type(shard) is not ShardIdentity or axis not in (0, 1):
        raise ValueError("factored factor routing requires a target shard and axis")
    shape = shard.parameter.global_shape
    if len(shape) != 2:
        raise ValueError("factored factor routing requires a matrix parameter")
    axis_numel = shape[axis]
    placement = shard.placements[0]
    if shard.layout is ParameterLayout.REPLICATED or (
        shard.layout is ParameterLayout.DTENSOR_1D_DEFAULT_WORLD
        and placement.kind is PlacementKind.REPLICATE
    ):
        return _full_segment(axis_numel)
    if shard.layout is ParameterLayout.WHOLE_PARAMETER_OWNER:
        return _full_segment(axis_numel) if shard.local_member == shard.owner else ()
    if shard.layout is ParameterLayout.DTENSOR_1D_DEFAULT_WORLD:
        region = shard.logical_region
        if region.numel == 0:
            return ()
        start = region.offsets[axis]
        length = region.lengths[axis]
        return (_LogicalSegment(start, length, 0),) if length else ()
    if shard.layout is not ParameterLayout.FLATTENED_ELEMENT_SHARD:
        raise ValueError("factored factor routing has an unsupported target layout")
    logical_slice = shard.logical_slice
    if logical_slice.length == 0:
        return ()
    rows, columns = shape
    start = logical_slice.flat_offset
    stop = start + logical_slice.length
    if axis == 0:
        row_start = start // columns
        row_stop = (stop - 1) // columns + 1
        return (_LogicalSegment(row_start, row_stop - row_start, 0),)
    if logical_slice.length >= columns:
        return _full_segment(columns)
    first_column = start % columns
    final_column = (stop - 1) % columns + 1
    if start // columns == (stop - 1) // columns:
        return (_LogicalSegment(first_column, logical_slice.length, 0),)
    head_length = final_column
    tail_length = columns - first_column
    segments = []
    local_offset = 0
    if head_length:
        segments.append(_LogicalSegment(0, head_length, local_offset))
        local_offset += head_length
    if tail_length:
        segments.append(_LogicalSegment(first_column, tail_length, local_offset))
    return tuple(segments)


def _segment_global_indices(segments):
    pieces = [
        torch.arange(
            segment.global_offset,
            segment.global_offset + segment.length,
            dtype=torch.int64,
        )
        for segment in segments
        if segment.length
    ]
    return torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.int64)


def _project_loaded_factored_axis(
    local_canonical,
    field,
    *,
    target_shards,
    axis: int,
    binding: CheckpointProcessGroupBinding,
    phase_vote=None,
):
    target_segments = None
    result = None
    indices = None
    error = None
    try:
        target_segments = tuple(
            _factored_axis_segments(shard, axis=axis)
            for shard in target_shards
        )
    except Exception as exc:
        error = exc
    if phase_vote is not None:
        phase_vote("factored_axis_prep", error)
    if error is not None:
        raise error

    error = None
    try:
        result = _redistribute_flat_field(
            local_canonical,
            source_segments_by_rank=_canonical_segments(
                field["numel"],
                len(binding.identity.ordered_members),
            ),
            target_segments_by_rank=target_segments,
            global_numel=field["numel"],
            binding=binding,
            phase_vote=phase_vote,
        ).cpu()
        coordinate = binding.identity.ordered_members.index(binding.local_member)
        indices = _segment_global_indices(target_segments[coordinate])
        if result.numel() != indices.numel():
            raise ValueError("factored factor routing produced invalid local geometry")
    except Exception as exc:
        error = exc
    if phase_vote is not None:
        phase_vote("factored_axis_finish", error)
    if error is not None:
        raise error
    assert result is not None and indices is not None
    return result, indices


def _local_factored_projection(
    row,
    row_indices,
    column,
    column_indices,
    denominator,
    shard: ShardIdentity,
):
    if (
        type(row) is not torch.Tensor
        or type(row_indices) is not torch.Tensor
        or type(column) is not torch.Tensor
        or type(column_indices) is not torch.Tensor
        or type(denominator) is not torch.Tensor
        or row.dtype is not torch.float32
        or row_indices.dtype is not torch.int64
        or column.dtype is not torch.float32
        or column_indices.dtype is not torch.int64
        or denominator.dtype is not torch.float32
        or row.ndim != 1
        or row_indices.ndim != 1
        or column.ndim != 1
        or column_indices.ndim != 1
        or denominator.numel() != 1
        or row.numel() != row_indices.numel()
        or column.numel() != column_indices.numel()
        or len(shard.parameter.global_shape) != 2
    ):
        raise ValueError("factored projection inputs do not match the target parameter")
    if row_indices.numel() and not bool(row_indices[1:].gt(row_indices[:-1]).all()):
        raise ValueError("factored row routing indices must be strictly increasing")
    if column_indices.numel() and not bool(column_indices[1:].gt(column_indices[:-1]).all()):
        raise ValueError("factored column routing indices must be strictly increasing")
    if not bool(torch.isfinite(denominator).all()) or not bool((denominator > 0).all()):
        raise ValueError("factored row denominator must be finite and positive")
    local_numel = _dense_local_numel(shard, unique_replicas=False)
    if local_numel == 0:
        if row.numel() or column.numel():
            raise ValueError("empty target shard received factored factor payloads")
        return None
    global_indices = _dense_global_indices(
        shard,
        unique_replicas=False,
        device=torch.device("cpu"),
    )
    columns = shard.parameter.global_shape[1]
    required_rows = torch.div(global_indices, columns, rounding_mode="floor")
    required_columns = torch.remainder(global_indices, columns)
    row_offsets = torch.searchsorted(row_indices, required_rows)
    column_offsets = torch.searchsorted(column_indices, required_columns)
    if (
        bool((row_offsets >= row_indices.numel()).any())
        or bool((column_offsets >= column_indices.numel()).any())
        or not bool(row_indices[row_offsets].eq(required_rows).all())
        or not bool(column_indices[column_offsets].eq(required_columns).all())
    ):
        raise ValueError("factored factor routing does not cover the target shard")
    return row[row_offsets].mul(column[column_offsets]).div_(denominator.reshape(()))


def _parameter_options_compatible(
    source,
    target,
    *,
    implementation: str,
):
    from gefen.portable_state import _PLAIN_OPTION_KEYS, _values_equal

    if implementation != "gefen.Gefen":
        return _values_equal(source, target)
    invariant = _PLAIN_OPTION_KEYS - {"second_moment_policy"}
    if any(not _values_equal(source[key], target[key]) for key in invariant):
        return False
    source_second = source["second_moment_policy"]
    target_second = target["second_moment_policy"]
    return source_second == target_second or (
        source_second == "factored" and target_second == "block"
    )


def _projected_target_byte_plan(
    optimizer,
    metadata,
    *,
    binding: CheckpointProcessGroupBinding,
    limits,
):
    """Validate aggregate target fp32 payloads before DCP reads any field."""

    from gefen import portable_runtime as runtime
    from gefen.portable_state import _validate_portable_projection_policy

    implementation = runtime._optimizer_implementation(optimizer)
    if metadata["implementation"] != implementation:
        raise ValueError("sharded portable implementation does not match the target")
    prepared = runtime._prepare_local_structure(
        optimizer,
        implementation,
        binding,
        limits,
        include_payload=False,
    )
    runtime._validate_prepared_local_state(optimizer, implementation, prepared)
    target_descriptor = prepared["target_descriptor"]
    _validate_portable_projection_policy(
        metadata["policy"],
        target_descriptor["policy"],
        implementation,
    )
    if set(metadata["parameters"]) != set(target_descriptor["catalog"]):
        raise ValueError("sharded portable parameter catalog does not match the target")
    target_manifest = optimizer._gefen_sharding_manifest
    members = binding.identity.ordered_members
    projected_numel = [
        256 if metadata["common"]["gefen_codebook_field"] is not None else 0
        for _member in members
    ]
    largest_factored_route_bytes = [0 for _member in members]
    largest_projection_routing_bytes = [0 for _member in members]

    def include_projection_send(
        global_numel,
        target_segments_by_rank,
        *,
        element_size,
    ):
        send_numel = _canonical_projection_send_numel(
            global_numel,
            target_segments_by_rank=target_segments_by_rank,
            parts=len(members),
        )
        for source_coordinate, value in enumerate(send_numel):
            largest_projection_routing_bytes[source_coordinate] = max(
                largest_projection_routing_bytes[source_coordinate],
                value * element_size,
            )

    if metadata["common"]["gefen_codebook_field"] is not None:
        include_projection_send(
            256,
            tuple(_full_segment(256) for _member in members),
            element_size=4,
        )
    for fqn in sorted(metadata["parameters"]):
        source_record = metadata["parameters"][fqn]
        target_record = target_descriptor["catalog"][fqn]
        if source_record["identity"] != target_record["identity"]:
            raise ValueError("sharded portable parameter identity does not match the target")
        source_options = source_record["algorithm_options"]
        target_options = target_record["algorithm_options"]
        if not _parameter_options_compatible(source_options, target_options, implementation=implementation):
            raise ValueError("sharded portable algorithm options do not match the target")
        source_second = source_options.get("second_moment_policy")
        target_second = target_options.get("second_moment_policy")
        if source_second == "block" and target_second == "factored":
            raise ValueError("portable block-to-factored second-moment migration is unsupported")
        target_shards = _target_shards_in_member_order(target_manifest, fqn, binding)
        variant = source_record["state_variant"]
        if variant.startswith("initialized_"):
            dense_send_numel = _dense_projection_send_numel(
                target_shards[0].parameter.numel,
                target_shards=target_shards,
                parts=len(members),
            )
            for coordinate, send_numel in enumerate(dense_send_numel):
                canonical_numel = _chunk_segment(
                    target_shards[0].parameter.numel,
                    len(members),
                    coordinate,
                ).length
                target_numel = _dense_local_numel(
                    target_shards[coordinate],
                    unique_replicas=False,
                )
                largest_projection_routing_bytes[coordinate] = max(
                    largest_projection_routing_bytes[coordinate],
                    len(members) * 8,
                    canonical_numel * 8,
                    send_numel * 4,
                    send_numel * 8,
                    target_numel * 8,
                )
        for coordinate, target_shard in enumerate(target_shards):
            local_numel = _dense_local_numel(target_shard, unique_replicas=False)
            if variant.startswith("initialized_"):
                projected_numel[coordinate] += local_numel
            if implementation == "gefen.Gefen" and variant == "initialized_dense":
                projected_numel[coordinate] += local_numel
            elif implementation == "gefen.Gefen" and variant == "initialized_factored":
                row_segments = _factored_axis_segments(target_shard, axis=0)
                column_segments = _factored_axis_segments(target_shard, axis=1)
                row_numel = _local_numel(row_segments)
                column_numel = _local_numel(column_segments)
                route_numel = row_numel + column_numel + 1
                if target_second != "factored":
                    route_numel += local_numel
                largest_factored_route_bytes[coordinate] = max(
                    largest_factored_route_bytes[coordinate],
                    route_numel * 4,
                    row_numel * 8,
                    column_numel * 8,
                    local_numel * 8,
                )
                if target_second == "factored":
                    projected_numel[coordinate] += (
                        target_shard.parameter.global_shape[0]
                        + target_shard.parameter.global_shape[1]
                    )
                else:
                    projected_numel[coordinate] += local_numel
            elif implementation == "gefen.GefenMuon" and variant == "initialized_dense_normuon":
                normuon_numel = target_shard.parameter.global_shape[0]
                projected_numel[coordinate] += _local_numel(
                    _special_segments_for_shard(
                        target_shard,
                        numel=normuon_numel,
                        unique_replicas=False,
                    )
                )
        if implementation == "gefen.Gefen" and variant == "initialized_factored":
            rows, columns = target_shards[0].parameter.global_shape
            include_projection_send(
                rows,
                tuple(
                    _factored_axis_segments(target_shard, axis=0)
                    for target_shard in target_shards
                ),
                element_size=4,
            )
            include_projection_send(
                columns,
                tuple(
                    _factored_axis_segments(target_shard, axis=1)
                    for target_shard in target_shards
                ),
                element_size=4,
            )
            include_projection_send(
                1,
                tuple(_full_segment(1) for _member in members),
                element_size=4,
            )
        elif implementation == "gefen.GefenMuon" and variant == "initialized_dense_normuon":
            normuon_numel = target_shards[0].parameter.global_shape[0]
            include_projection_send(
                normuon_numel,
                tuple(
                    _special_segments_for_shard(
                        target_shard,
                        numel=normuon_numel,
                        unique_replicas=False,
                    )
                    for target_shard in target_shards
                ),
                element_size=4,
            )
    element_size = torch.tensor([], dtype=torch.float32).element_size()
    projected_bytes = [numel * element_size for numel in projected_numel]
    factored_route_bytes = largest_factored_route_bytes
    projection_routing_bytes = largest_projection_routing_bytes
    if any(value > limits.max_fragment_tensor_bytes for value in projected_bytes):
        raise ValueError("sharded portable projected target state exceeds max_fragment_tensor_bytes")
    if any(value > limits.max_fragment_tensor_bytes for value in factored_route_bytes):
        raise ValueError("sharded portable factored routing scratch exceeds max_fragment_tensor_bytes")
    if any(value > limits.max_fragment_tensor_bytes for value in projection_routing_bytes):
        raise ValueError("sharded portable projection routing scratch exceeds max_fragment_tensor_bytes")
    digest = runtime._context_digest(
        {
            "target_descriptor": target_descriptor,
            "projected_fp32_bytes": projected_bytes,
            "factored_route_fp32_bytes": factored_route_bytes,
            "projection_routing_bytes": projection_routing_bytes,
        }
    )
    return (
        digest,
        tuple(projected_bytes),
        tuple(factored_route_bytes),
        tuple(projection_routing_bytes),
    )


def _stage_sharded_import(
    optimizer,
    metadata,
    local_fields,
    *,
    binding: CheckpointProcessGroupBinding,
    limits,
    transaction_id: str,
):
    from gefen import portable_runtime as runtime
    from gefen.portable import (
        _recompress_dense_momentum,
        _reduce_block_second_moment,
    )
    from gefen.portable_state import (
        _SECOND_MOMENT_PROJECTION_FACTORED_TO_BLOCK,
        _normalize_common,
        _validate_portable_projection_policy,
    )
    from gefen.portable_collective import _collective_unanimous_status

    implementation = None
    live_token = None
    target_descriptor = None
    target_manifest = None
    local_by_fqn = None
    fields = None
    target_digest = bytes.fromhex(metadata["completion"]["metadata_digest"])
    error = None
    try:
        implementation = runtime._optimizer_implementation(optimizer)
        if metadata["implementation"] != implementation:
            raise ValueError("sharded portable implementation does not match the target")
        live_token = runtime._portable_live_token(optimizer)
        prepared = runtime._prepare_local_structure(
            optimizer,
            implementation,
            binding,
            limits,
            include_payload=False,
        )
        runtime._validate_prepared_local_state(
            optimizer,
            implementation,
            prepared,
        )
        target_descriptor = prepared["target_descriptor"]
        _validate_portable_projection_policy(
            metadata["policy"],
            target_descriptor["policy"],
            implementation,
        )
        if set(metadata["parameters"]) != set(target_descriptor["catalog"]):
            raise ValueError("sharded portable parameter catalog does not match the target")
        target_manifest = optimizer._gefen_sharding_manifest
        local_by_fqn = prepared["local_by_fqn"]
        fields = {field["index"]: field for field in metadata["fields"]}
        if set(local_fields) != set(fields):
            raise ValueError("sharded portable local fields are incomplete")
        coordinate = binding.identity.ordered_members.index(binding.local_member)
        parts = len(binding.identity.ordered_members)
        for index, field in fields.items():
            local = local_fields[index]
            if (
                type(local) is not torch.Tensor
                or local.dtype is not torch.float32
                or local.ndim != 1
                or local.numel()
                != _chunk_segment(field["numel"], parts, coordinate).length
            ):
                raise ValueError("sharded portable local field geometry is invalid")
        for fqn in sorted(local_by_fqn):
            parameter, shard = local_by_fqn[fqn]
            source_record = metadata["parameters"][fqn]
            target_record = target_descriptor["catalog"][fqn]
            if source_record["identity"] != target_record["identity"]:
                raise ValueError(
                    "sharded portable parameter identity does not match the target"
                )
            target_options = target_record["algorithm_options"]
            source_options = source_record["algorithm_options"]
            if not _parameter_options_compatible(
                source_options,
                target_options,
                implementation=implementation,
            ):
                raise ValueError("sharded portable algorithm options do not match the target")
            source_second = source_options.get("second_moment_policy")
            target_second = target_options.get("second_moment_policy")
            if source_second == "block" and target_second == "factored":
                raise ValueError("portable block-to-factored second-moment migration is unsupported")
            if (
                source_second == "factored"
                and target_second == "block"
                and metadata["policy"]["second_moment_projection"]
                != _SECOND_MOMENT_PROJECTION_FACTORED_TO_BLOCK
            ):
                raise ValueError(
                    "sharded portable factored-to-block projection is not authorized by its policy"
                )
            target_shards = _target_shards_in_member_order(
                target_manifest,
                fqn,
                binding,
            )
            variant = source_record["state_variant"]
            if variant.startswith("initialized_"):
                for target_shard in target_shards:
                    _dense_local_numel(
                        target_shard,
                        unique_replicas=False,
                    )
            if variant == "initialized_factored":
                for target_shard in target_shards:
                    _factored_axis_segments(target_shard, axis=0)
                    _factored_axis_segments(target_shard, axis=1)
            if (
                implementation == "gefen.GefenMuon"
                and variant == "initialized_dense_normuon"
            ):
                normuon_field = fields[
                    source_record["state"]["normuon_v_field"]
                ]
                for target_shard in target_shards:
                    _special_segments_for_shard(
                        target_shard,
                        numel=normuon_field["numel"],
                        unique_replicas=False,
                    )
            if parameter is not None:
                runtime._local_dense_shape(shard)
        target_context = runtime._target_context(
            runtime._base_context(binding, implementation),
            target_descriptor,
        )
        target_context["document_digest"] = metadata["completion"][
            "metadata_digest"
        ]
        runtime._preflight_portable_value(target_context, limits)
        target_digest = runtime._context_digest(target_context)
    except Exception as exc:
        error = exc
    _collective_unanimous_status(
        binding,
        error,
        operation="portable_dcp_sharded_load_target",
        transaction_id=transaction_id,
        context_digest=target_digest,
        limits=limits._wire_limits(collective=True),
    )
    assert (
        implementation is not None
        and live_token is not None
        and target_descriptor is not None
        and target_manifest is not None
        and local_by_fqn is not None
        and fields is not None
    )

    phase_vote = _make_phase_voter(
        binding,
        transaction_id=transaction_id,
        context_digest=target_digest,
        limits=limits,
        operation_prefix="portable_dcp_sharded_stage_phase",
    )

    def scoped_vote(prefix):
        return lambda label, error: phase_vote("{}.{}".format(prefix, label), error)

    codebook = None
    common = None
    error = None
    try:
        codebook_index = metadata["common"]["gefen_codebook_field"]
        if codebook_index is not None:
            field = fields[codebook_index]
            codebook = _project_loaded_field(
                local_fields[codebook_index],
                field,
                target_shards=(),
                binding=binding,
                replicate=True,
                phase_vote=scoped_vote("common.codebook"),
            ).reshape(256)
        common = _normalize_common(
            {
                "gefen_global_step": metadata["common"]["gefen_global_step"],
                "gefen_codebook": codebook,
                "gefen_deterministic": metadata["common"]["gefen_deterministic"],
            }
        )
    except Exception as exc:
        error = exc
    phase_vote("common.normalize", error)
    assert common is not None

    canonical_parameters = {}
    for fqn in sorted(local_by_fqn):
        parameter = None
        shard = None
        source_record = None
        target_options = None
        target_second = None
        target_shards = None
        variant = None
        source_state = None
        local_shape = None
        has_local_payload = None
        projected = None
        error = None
        try:
            parameter, shard = local_by_fqn[fqn]
            source_record = metadata["parameters"][fqn]
            target_options = target_descriptor["catalog"][fqn]["algorithm_options"]
            source_options = source_record["algorithm_options"]
            if not _parameter_options_compatible(source_options, target_options, implementation=implementation):
                raise ValueError("sharded portable algorithm options do not match the target")
            source_second = source_options.get("second_moment_policy")
            target_second = target_options.get("second_moment_policy")
            if source_second == "block" and target_second == "factored":
                raise ValueError("portable block-to-factored second-moment migration is unsupported")
            target_shards = _target_shards_in_member_order(target_manifest, fqn, binding)
            variant = source_record["state_variant"]
            source_state = source_record["state"]
            local_shape = runtime._local_dense_shape(shard) if parameter is not None else None
            has_local_payload = local_shape is not None
            projected = {}
        except Exception as exc:
            error = exc
        phase_vote("parameter.{}.setup".format(fqn), error)
        assert shard is not None and source_record is not None and target_shards is not None and variant is not None and source_state is not None and projected is not None

        if variant == "period_selected":
            error = None
            try:
                if has_local_payload:
                    projected = {"automatic_period": 1}
            except Exception as exc:
                error = exc
            phase_vote("parameter.{}.period_selected".format(fqn), error)
        elif variant.startswith("initialized_"):
            error = None
            try:
                if codebook is None:
                    raise ValueError("initialized sharded portable state requires a codebook")
            except Exception as exc:
                error = exc
            phase_vote("parameter.{}.initialized_prep".format(fqn), error)

            momentum = None
            expected_momentum_numel = None
            error = None
            try:
                momentum_index = source_state["momentum_field"]
                momentum = _project_loaded_field(
                    local_fields[momentum_index],
                    fields[momentum_index],
                    target_shards=target_shards,
                    binding=binding,
                    phase_vote=scoped_vote("parameter.{}.momentum".format(fqn)),
                )
                expected_momentum_numel = 0 if local_shape is None else math.prod(local_shape)
                if momentum.numel() != expected_momentum_numel:
                    raise ValueError("sharded portable momentum does not match its target shard")
                if has_local_payload:
                    momentum = momentum.reshape(local_shape)
            except Exception as exc:
                error = exc
            phase_vote("parameter.{}.momentum_geometry".format(fqn), error)
            assert momentum is not None and expected_momentum_numel is not None

            error = None
            try:
                if has_local_payload:
                    indices, magnitudes = _recompress_dense_momentum(
                        momentum,
                        codebook,
                        period=1,
                        step=source_state["step"],
                    )
                    projected = {
                        "automatic_period": 1,
                        "step": source_state["step"],
                        "m_codebook": indices,
                        "m_magnitude": magnitudes,
                    }
                del momentum
            except Exception as exc:
                error = exc
            phase_vote("parameter.{}.momentum_recompress".format(fqn), error)

            if implementation == "gefen.Gefen" and variant == "initialized_dense":
                second = None
                error = None
                try:
                    second_index = source_state["second_moment_field"]
                    second = _project_loaded_field(
                        local_fields[second_index],
                        fields[second_index],
                        target_shards=target_shards,
                        binding=binding,
                        phase_vote=scoped_vote("parameter.{}.second_moment".format(fqn)),
                    )
                    if second.numel() != expected_momentum_numel:
                        raise ValueError("sharded portable second moment does not match its target shard")
                    if has_local_payload:
                        projected.update(
                            {
                                "vmean": _reduce_block_second_moment(
                                    second.reshape(local_shape),
                                    period=1,
                                    step=source_state["second_moment_step"],
                                ),
                                "vmean_step": source_state["second_moment_step"],
                            }
                        )
                    del second
                except Exception as exc:
                    error = exc
                phase_vote("parameter.{}.second_moment_finish".format(fqn), error)
            elif implementation == "gefen.Gefen" and variant == "initialized_factored":
                row = None
                row_indices = None
                error = None
                try:
                    row_index = source_state["v_row_field"]
                    row, row_indices = _project_loaded_factored_axis(
                        local_fields[row_index],
                        fields[row_index],
                        target_shards=target_shards,
                        axis=0,
                        binding=binding,
                        phase_vote=scoped_vote("parameter.{}.factored_row".format(fqn)),
                    )
                except Exception as exc:
                    error = exc
                phase_vote("parameter.{}.factored_row_finish".format(fqn), error)
                assert row is not None and row_indices is not None

                column = None
                column_indices = None
                error = None
                try:
                    column_index = source_state["v_col_field"]
                    column, column_indices = _project_loaded_factored_axis(
                        local_fields[column_index],
                        fields[column_index],
                        target_shards=target_shards,
                        axis=1,
                        binding=binding,
                        phase_vote=scoped_vote("parameter.{}.factored_column".format(fqn)),
                    )
                except Exception as exc:
                    error = exc
                phase_vote("parameter.{}.factored_column_finish".format(fqn), error)
                assert column is not None and column_indices is not None

                denominator = None
                error = None
                try:
                    denominator_index = source_state["v_row_denominator_field"]
                    denominator = _project_loaded_field(
                        local_fields[denominator_index],
                        fields[denominator_index],
                        target_shards=(),
                        binding=binding,
                        replicate=True,
                        phase_vote=scoped_vote("parameter.{}.factored_denominator".format(fqn)),
                    )
                    if denominator.numel() != 1:
                        raise ValueError("sharded portable factored denominator has invalid geometry")
                except Exception as exc:
                    error = exc
                phase_vote("parameter.{}.factored_denominator_finish".format(fqn), error)
                assert denominator is not None

                error = None
                try:
                    if target_second == "factored" and has_local_payload:
                        if shard.layout is not ParameterLayout.REPLICATED:
                            raise ValueError("factored portable state requires a replicated target")
                        projected.update(
                            {
                                "v_row": row,
                                "v_col": column,
                                "factored_step": source_state["factored_step"],
                            }
                        )
                    elif target_second != "factored":
                        second = _local_factored_projection(
                            row,
                            row_indices,
                            column,
                            column_indices,
                            denominator,
                            shard,
                        )
                        del row, row_indices, column, column_indices, denominator
                        if has_local_payload:
                            if second is None:
                                raise ValueError("initialized target shard is missing factored projection state")
                            projected.update(
                                {
                                    "vmean": _reduce_block_second_moment(
                                        second,
                                        period=1,
                                        step=source_state["factored_step"],
                                    ),
                                    "vmean_step": source_state["factored_step"],
                                }
                            )
                        elif second is not None:
                            raise ValueError("empty target shard received factored projection state")
                        del second
                except Exception as exc:
                    error = exc
                phase_vote("parameter.{}.factored_projection_finish".format(fqn), error)
            elif implementation == "gefen.GefenMuon" and variant == "initialized_dense_normuon":
                normuon = None
                error = None
                try:
                    normuon_index = source_state["normuon_v_field"]
                    normuon = _project_loaded_field(
                        local_fields[normuon_index],
                        fields[normuon_index],
                        target_shards=target_shards,
                        binding=binding,
                        phase_vote=scoped_vote("parameter.{}.normuon".format(fqn)),
                    )
                    expected_normuon_numel = shard.parameter.global_shape[0] if has_local_payload else 0
                    if normuon.numel() != expected_normuon_numel:
                        raise ValueError("sharded portable NorMuon state does not match its target shard")
                    if has_local_payload:
                        projected.update(
                            {
                                "normuon_v": normuon.reshape(shard.parameter.global_shape[0], 1),
                                "normuon_step": source_state["normuon_step"],
                            }
                        )
                except Exception as exc:
                    error = exc
                phase_vote("parameter.{}.normuon_finish".format(fqn), error)

        error = None
        try:
            if parameter is not None:
                canonical_parameters[fqn] = {"state": projected}
        except Exception as exc:
            error = exc
        phase_vote("parameter.{}.finish".format(fqn), error)

    staged = None
    error = None
    try:
        canonical_state = {
            "common": {
                "gefen_global_step": common["gefen_global_step"],
                "gefen_codebook": common["gefen_codebook"],
                "gefen_deterministic": common["gefen_deterministic"],
                "gefen_codebook_scope": optimizer._serialized_codebook_scope(),
            },
            "parameters": canonical_parameters,
        }
        native = optimizer._canonical_native_state_dict(canonical_state)
        staging_owner = object.__new__(type(optimizer))
        staging_owner.__dict__ = optimizer.__dict__.copy()
        staging_owner._deterministic = common["gefen_deterministic"]
        has_dtensor_target = any(
            target_shard.layout is ParameterLayout.DTENSOR_1D_DEFAULT_WORLD
            for _parameter, target_shard in local_by_fqn.values()
        )
        if has_dtensor_target:
            native.pop("gefen_native_local_shards", None)
            for group in native["param_groups"]:
                checkpoint_metadata = dict(group[runtime._RANK_LOCAL_GROUP_METADATA_KEY])
                checkpoint_metadata.pop("native_local_shards", None)
                group[runtime._RANK_LOCAL_GROUP_METADATA_KEY] = checkpoint_metadata
            staging_owner._uses_rank_local_sharded_state = lambda: False
            staging_owner._serialized_native_local_shards = lambda: None
        staged = staging_owner._stage_load_state_dict(native)
        if has_dtensor_target:
            staged.__dict__.pop("_uses_rank_local_sharded_state", None)
            staged.__dict__.pop("_serialized_native_local_shards", None)
            staged._install_rank_local_checkpoint_schema()
            runtime._validate_rank_local_transport_schema(staged, binding)
        optimizer._preserve_canonical_target_configuration(staged)
        if (
            type(staged.__dict__) is not dict
            or type(staged.defaults) is not dict
            or set(staged.defaults) != set(optimizer.defaults)
            or staged.param_groups is not optimizer.param_groups
        ):
            raise TypeError("sharded portable import staging produced unsafe publication containers")
        if live_token != runtime._portable_live_token(optimizer):
            raise RuntimeError("live optimizer state changed during sharded portable import preparation")
    except Exception as exc:
        error = exc
    phase_vote("stage.finalize", error)
    assert staged is not None
    return staged, live_token


def _save_sharded_portable_dcp(
    optimizer,
    *,
    binding: CheckpointProcessGroupBinding,
    storage_writer,
    transaction_id: str,
    limits,
    namespace: str,
):
    import torch.distributed.checkpoint as dcp

    from gefen.portable_collective import _collective_unanimous_status

    state = None
    planner = None
    digest = bytes(32)
    error = None
    try:
        state = _prepare_sharded_save_state(
            optimizer,
            binding=binding,
            transaction_id=transaction_id,
            limits=limits,
            namespace=namespace,
        )
        metadata = _decode_metadata_tensor(
            state[namespace][_DCP_SHARDED_METADATA_KEY],
            limits=limits,
        )
        digest = bytes.fromhex(metadata["completion"]["metadata_digest"])
        planner = dcp.DefaultSavePlanner()
    except Exception as exc:
        error = exc
    _collective_unanimous_status(
        binding,
        error,
        operation="portable_dcp_sharded_save_prepare",
        transaction_id=transaction_id,
        context_digest=digest,
        limits=limits._wire_limits(collective=True),
    )
    assert state is not None and planner is not None
    result = None
    error = None
    try:
        result = dcp.save(
            state,
            storage_writer=storage_writer,
            planner=planner,
            process_group=binding.process_group,
        )
    except Exception as exc:
        error = exc
    _collective_unanimous_status(
        binding,
        error,
        operation="portable_dcp_sharded_save_dcp_complete",
        transaction_id=transaction_id,
        context_digest=digest,
        limits=limits._wire_limits(collective=True),
    )
    return result


def _load_sharded_portable_dcp(
    optimizer,
    *,
    binding: CheckpointProcessGroupBinding,
    storage_reader,
    checkpoint_metadata,
    transaction_id: str,
    limits,
    namespace: str,
):
    from gefen.portable_collective import _collective_unanimous_status

    metadata = None
    digest = bytes(32)
    error = None
    try:
        metadata = _read_sharded_metadata(
            storage_reader=storage_reader,
            checkpoint_metadata=checkpoint_metadata,
            binding=binding,
            namespace=namespace,
            limits=limits,
            transaction_id=transaction_id,
        )
        if metadata is None:
            raise ValueError("checkpoint does not contain a sharded portable DCP envelope")
        digest = bytes.fromhex(metadata["completion"]["metadata_digest"])
    except Exception as exc:
        error = exc
    _collective_unanimous_status(
        binding,
        error,
        operation="portable_dcp_sharded_load_metadata",
        transaction_id=transaction_id,
        context_digest=digest,
        limits=limits._wire_limits(collective=True),
    )
    assert checkpoint_metadata is not None and metadata is not None

    projected_digest = digest
    error = None
    try:
        (
            projected_digest,
            _projected_bytes,
            _scratch_bytes,
            _send_bytes,
        ) = _projected_target_byte_plan(
            optimizer,
            metadata,
            binding=binding,
            limits=limits,
        )
    except Exception as exc:
        error = exc
    _collective_unanimous_status(
        binding,
        error,
        operation="portable_dcp_sharded_load_projected_limits",
        transaction_id=transaction_id,
        context_digest=projected_digest,
        limits=limits._wire_limits(collective=True),
    )

    local_fields = None
    error = None
    try:
        metadata_shape = _metadata_storage_spec(
            checkpoint_metadata,
            namespace=namespace,
            limits=limits,
        )
        assert metadata_shape is not None
        local_fields = _load_sharded_payloads(
            storage_reader=storage_reader,
            checkpoint_metadata=checkpoint_metadata,
            metadata=metadata,
            metadata_shape=metadata_shape,
            binding=binding,
            namespace=namespace,
            limits=limits,
            transaction_id=transaction_id,
            context_digest=digest,
        )
    except Exception as exc:
        error = exc
    _collective_unanimous_status(
        binding,
        error,
        operation="portable_dcp_sharded_load_payloads",
        transaction_id=transaction_id,
        context_digest=digest,
        limits=limits._wire_limits(collective=True),
    )
    assert local_fields is not None

    staged = None
    live_token = None
    error = None
    try:
        staged, live_token = _stage_sharded_import(
            optimizer,
            metadata,
            local_fields,
            binding=binding,
            limits=limits,
            transaction_id=transaction_id,
        )
    except Exception as exc:
        error = exc
    _collective_unanimous_status(
        binding,
        error,
        operation="portable_dcp_sharded_load_stage",
        transaction_id=transaction_id,
        context_digest=digest,
        limits=limits._wire_limits(collective=True),
    )
    assert staged is not None and live_token is not None

    error = None
    try:
        from gefen import portable_runtime as runtime

        implementation = runtime._optimizer_implementation(optimizer)
        runtime._validate_live_readiness(optimizer, implementation, binding)
        if live_token != runtime._portable_live_token(optimizer):
            raise RuntimeError("live optimizer state changed after sharded portable import preparation")
    except Exception as exc:
        error = exc
    _collective_unanimous_status(
        binding,
        error,
        operation="portable_dcp_sharded_load_freshness",
        transaction_id=transaction_id,
        context_digest=digest,
        limits=limits._wire_limits(collective=True),
    )

    from gefen.gefen import Gefen

    Gefen._commit_staged_load_state_dict(optimizer, staged)


__all__ = []

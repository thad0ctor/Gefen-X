"""Symmetric tensor-only collectives for bounded canonical-state fragments.

This module deliberately keeps the transport private.  Callers stage values in
callbacks and publish them only after the complete operation returns.
"""

from __future__ import annotations

from dataclasses import fields
import hashlib
import hmac
import struct
import sys

import torch

from gefen.checkpoint import CheckpointProcessGroupBinding
from gefen.portable_wire import (
    _CanonicalWireLimits,
    _CanonicalWirePlan,
    _PreparedCanonicalWireValue,
    _parse_canonical_wire_metadata,
    _prepare_canonical_wire_value,
    _reconstruct_canonical_wire_value,
)


_I64_MAX = (1 << 63) - 1
_HEADER_MAGIC = 0x47464E434F4C4C31  # ``GFNCOLL1``
_VOTE_MAGIC = 0x47464E564F544531  # ``GFNVOTE1``
_PROTOCOL_VERSION = 1
_DIGEST_FIELDS = 4
_DIAGNOSTIC_BYTES = 2048
_LABEL_BYTES = 1024
_LIMIT_NAMES = tuple(field.name for field in fields(_CanonicalWireLimits))

_H_MAGIC = 0
_H_VERSION = 1
_H_STATUS = 2
_H_COORDINATE = 3
_H_MEMBERS = 4
_H_METADATA_BYTES = 5
_H_TENSOR_COUNT = 6
_H_TENSOR_BYTES = 7
_H_TRANSPORT = 8
_H_OPERATION_BYTES = 9
_H_TRANSACTION_BYTES = 10
_H_CALLBACK_FLAGS = 11
_H_OPERATION_DIGEST = 12
_H_TRANSACTION_DIGEST = _H_OPERATION_DIGEST + _DIGEST_FIELDS
_H_CONTEXT_DIGEST = _H_TRANSACTION_DIGEST + _DIGEST_FIELDS
_H_IDENTITY_DIGEST = _H_CONTEXT_DIGEST + _DIGEST_FIELDS
_H_METADATA_DIGEST = _H_IDENTITY_DIGEST + _DIGEST_FIELDS
_H_FRAGMENT_DIGEST = _H_METADATA_DIGEST + _DIGEST_FIELDS
_H_LIMITS = _H_FRAGMENT_DIGEST + _DIGEST_FIELDS
_HEADER_FIELDS = _H_LIMITS + len(_LIMIT_NAMES)

_V_MAGIC = 0
_V_VERSION = 1
_V_PHASE = 2
_V_SOURCE = 3
_V_STATUS = 4
_V_COORDINATE = 5
_V_RESERVED_0 = 6
_V_RESERVED_1 = 7
_VOTE_FIELDS = 8

_PHASE_METADATA_BUFFER = 1
_PHASE_METADATA_PARSE = 2
_PHASE_PAYLOAD_ALLOCATION = 3
_PHASE_CHUNK_PREPARE = 4
_PHASE_CHUNK_STAGE = 5
_PHASE_RECONSTRUCT = 6
_PHASE_CONSUME = 7


class _CanonicalCollectiveError(RuntimeError):
    """A deterministic, live-member-wide canonical transport rejection."""


def _require_signed_header_int(name: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError("{} must be an int".format(name))
    if value < 0 or value > _I64_MAX:
        raise ValueError("{} must fit a nonnegative signed int64 header field".format(name))


def _label_bytes(name: str, value: str) -> bytes:
    if type(value) is not str:
        raise TypeError("{} must be a string".format(name))
    if not value or value != value.strip() or "\x00" in value:
        raise ValueError("{} must be nonempty, trimmed, and contain no NUL".format(name))
    encoded = value.encode("utf-8")
    if len(encoded) > _LABEL_BYTES:
        raise ValueError("{} exceeds {} UTF-8 bytes".format(name, _LABEL_BYTES))
    return encoded


def _digest_fields(value: bytes) -> tuple[int, int, int, int]:
    if type(value) is not bytes or len(value) != hashlib.sha256().digest_size:
        raise ValueError("digest header fields require exactly 32 bytes")
    return tuple(
        int.from_bytes(value[start : start + 8], "big", signed=True)
        for start in range(0, len(value), 8)
    )


def _fields_digest(values: tuple[int, ...] | list[int]) -> bytes:
    if len(values) != _DIGEST_FIELDS:
        raise ValueError("a digest requires four signed int64 fields")
    return b"".join(int(value).to_bytes(8, "big", signed=True) for value in values)


def _identity_digest(binding: CheckpointProcessGroupBinding) -> bytes:
    identity = binding.identity
    hasher = hashlib.sha256()
    hasher.update(b"gefen.canonical_collective.identity.v1\0")
    hasher.update(struct.pack(">Q", identity.schema_version))
    semantic_name = identity.semantic_name.encode("utf-8")
    hasher.update(struct.pack(">Q", len(semantic_name)))
    hasher.update(semantic_name)
    hasher.update(struct.pack(">Q", len(identity.ordered_members)))
    for member in identity.ordered_members:
        encoded = member.encode("utf-8")
        hasher.update(struct.pack(">Q", len(encoded)))
        hasher.update(encoded)
    return hasher.digest()


def _transport_code(binding: CheckpointProcessGroupBinding) -> int:
    if sys.byteorder == "little":
        byteorder = 0
    elif sys.byteorder == "big":
        byteorder = 1
    else:
        raise RuntimeError("unsupported native byte order")
    return (0 if binding.collective_device.type == "cpu" else 2) + byteorder


def _diagnostic(exc: Exception | None, *, limit: int = _DIAGNOSTIC_BYTES) -> bytes:
    if exc is None:
        return bytes(_DIAGNOSTIC_BYTES)
    usable = max(1, min(limit, _DIAGNOSTIC_BYTES))
    message = "{}: {}".format(type(exc).__name__, str(exc)).replace("\x00", "\\0")
    encoded = message.encode("utf-8", errors="replace")[: usable - 1]
    return encoded + bytes(_DIAGNOSTIC_BYTES - len(encoded))


def _decode_diagnostic(value: bytes) -> str:
    return value.split(b"\0", 1)[0].decode("utf-8", errors="replace") or "unspecified failure"


def _runtime_size(binding: CheckpointProcessGroupBinding) -> int:
    if len(binding.identity.ordered_members) == 1:
        return 1
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("a multi-member canonical collective requires initialized torch.distributed")
    return dist.get_world_size(binding.process_group)


def _all_gather_fixed(
    binding: CheckpointProcessGroupBinding,
    values: torch.Tensor,
    *,
    member_count: int,
) -> tuple[torch.Tensor, ...]:
    if member_count == 1:
        return (values.detach().cpu(),)
    import torch.distributed as dist

    collective = values.to(binding.collective_device)
    gathered = [torch.empty_like(collective) for _ in range(member_count)]
    dist.all_gather(gathered, collective, group=binding.process_group)
    return tuple(value.cpu() for value in gathered)


def _exchange_header(
    binding: CheckpointProcessGroupBinding,
    header: tuple[int, ...],
    diagnostic: bytes,
    *,
    member_count: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[bytes, ...]]:
    header_tensor = torch.tensor(header, dtype=torch.int64)
    diagnostic_tensor = torch.frombuffer(bytearray(diagnostic), dtype=torch.uint8)
    gathered_headers = _all_gather_fixed(binding, header_tensor, member_count=member_count)
    gathered_diagnostics = _all_gather_fixed(binding, diagnostic_tensor, member_count=member_count)
    return (
        tuple(tuple(int(item) for item in value.tolist()) for value in gathered_headers),
        tuple(bytes(value.numpy().tobytes()) for value in gathered_diagnostics),
    )


def _error_from_diagnostics(
    phase: str,
    diagnostics: tuple[bytes, ...],
    failed: tuple[int, ...],
) -> _CanonicalCollectiveError:
    details = "; ".join(
        "semantic-coordinate[{}]: {}".format(coordinate, _decode_diagnostic(diagnostics[coordinate]))
        for coordinate in failed
    )
    return _CanonicalCollectiveError("canonical collective {} failed: {}".format(phase, details))


def _phase_vote(
    binding: CheckpointProcessGroupBinding,
    *,
    member_count: int,
    coordinate: int,
    source: int,
    phase: int,
    phase_name: str,
    error: Exception | None,
    diagnostic_limit: int,
) -> None:
    vote = [0] * _VOTE_FIELDS
    vote[_V_MAGIC] = _VOTE_MAGIC
    vote[_V_VERSION] = _PROTOCOL_VERSION
    vote[_V_PHASE] = phase
    vote[_V_SOURCE] = source
    vote[_V_STATUS] = 1 if error is not None else 0
    vote[_V_COORDINATE] = coordinate
    headers, diagnostics = _exchange_header(
        binding,
        tuple(vote),
        _diagnostic(error, limit=diagnostic_limit),
        member_count=member_count,
    )
    for expected_coordinate, header in enumerate(headers):
        if len(header) != _VOTE_FIELDS or header[_V_MAGIC] != _VOTE_MAGIC or header[_V_VERSION] != _PROTOCOL_VERSION:
            raise _CanonicalCollectiveError("canonical collective received an invalid fixed-size phase vote")
        if header[_V_PHASE] != phase or header[_V_SOURCE] != source:
            raise _CanonicalCollectiveError("canonical collective phase/source order diverged")
        if header[_V_COORDINATE] != expected_coordinate or header[_V_RESERVED_0] != 0 or header[_V_RESERVED_1] != 0:
            raise _CanonicalCollectiveError("canonical collective phase vote has an invalid semantic coordinate")
        if header[_V_STATUS] not in {0, 1}:
            raise _CanonicalCollectiveError("canonical collective phase vote has an invalid status")
    failed = tuple(index for index, header in enumerate(headers) if header[_V_STATUS])
    if failed:
        raise _error_from_diagnostics(
            phase_name,
            diagnostics,
            failed,
        )


def _failure_header(coordinate: int, member_count: int) -> tuple[int, ...]:
    header = [0] * _HEADER_FIELDS
    header[_H_MAGIC] = _HEADER_MAGIC
    header[_H_VERSION] = _PROTOCOL_VERSION
    header[_H_STATUS] = 1
    header[_H_COORDINATE] = coordinate
    header[_H_MEMBERS] = member_count
    return tuple(header)


def _success_header(
    binding: CheckpointProcessGroupBinding,
    plan: _CanonicalWirePlan | None,
    limits: _CanonicalWireLimits,
    *,
    coordinate: int,
    member_count: int,
    operation: bytes,
    transaction: bytes,
    context_digest: bytes,
    callback_flags: int,
) -> tuple[int, ...]:
    if plan is not None and type(plan) is not _CanonicalWirePlan:
        raise TypeError("plan must be a _CanonicalWirePlan or None")
    metadata_bytes = len(plan.metadata) if plan is not None else 0
    tensor_count = len(plan.tensor_specs) if plan is not None else 0
    tensor_bytes = plan.total_tensor_bytes if plan is not None else 0
    for name in _LIMIT_NAMES:
        _require_signed_header_int("limits.{}".format(name), getattr(limits, name))
    if limits.diagnostic_bytes > _DIAGNOSTIC_BYTES:
        raise ValueError("limits.diagnostic_bytes exceeds the collective diagnostic capacity")
    for name, value in (
        ("member_count", member_count),
        ("coordinate", coordinate),
        ("metadata bytes", metadata_bytes),
        ("tensor count", tensor_count),
        ("tensor bytes", tensor_bytes),
        ("operation bytes", len(operation)),
        ("transaction bytes", len(transaction)),
        ("callback flags", callback_flags),
    ):
        _require_signed_header_int(name, value)
    if member_count > limits.max_members:
        raise ValueError("checkpoint member count exceeds limits.max_members")
    header = [0] * _HEADER_FIELDS
    header[_H_MAGIC] = _HEADER_MAGIC
    header[_H_VERSION] = _PROTOCOL_VERSION
    header[_H_STATUS] = 0
    header[_H_COORDINATE] = coordinate
    header[_H_MEMBERS] = member_count
    header[_H_METADATA_BYTES] = metadata_bytes
    header[_H_TENSOR_COUNT] = tensor_count
    header[_H_TENSOR_BYTES] = tensor_bytes
    header[_H_TRANSPORT] = _transport_code(binding)
    header[_H_OPERATION_BYTES] = len(operation)
    header[_H_TRANSACTION_BYTES] = len(transaction)
    if callback_flags > 3:
        raise ValueError("callback flags contain unknown bits")
    header[_H_CALLBACK_FLAGS] = callback_flags
    for offset, digest in (
        (_H_OPERATION_DIGEST, hashlib.sha256(b"operation\0" + operation).digest()),
        (_H_TRANSACTION_DIGEST, hashlib.sha256(b"transaction\0" + transaction).digest()),
        (_H_CONTEXT_DIGEST, context_digest),
        (_H_IDENTITY_DIGEST, _identity_digest(binding)),
        (_H_METADATA_DIGEST, plan.metadata_digest if plan is not None else bytes(32)),
        (_H_FRAGMENT_DIGEST, plan.fragment_digest if plan is not None else bytes(32)),
    ):
        header[offset : offset + _DIGEST_FIELDS] = _digest_fields(digest)
    for index, name in enumerate(_LIMIT_NAMES):
        header[_H_LIMITS + index] = getattr(limits, name)
    return tuple(header)


def _validate_initial_exchange(
    binding: CheckpointProcessGroupBinding,
    headers: tuple[tuple[int, ...], ...],
    diagnostics: tuple[bytes, ...],
    *,
    member_count: int,
) -> None:
    for expected_coordinate, header in enumerate(headers):
        if len(header) != _HEADER_FIELDS or header[_H_MAGIC] != _HEADER_MAGIC or header[_H_VERSION] != _PROTOCOL_VERSION:
            raise _CanonicalCollectiveError("canonical collective received an invalid fixed-size header")
        if header[_H_STATUS] not in {0, 1}:
            raise _CanonicalCollectiveError("canonical collective header has an invalid status")
        if header[_H_COORDINATE] != expected_coordinate or header[_H_MEMBERS] != member_count:
            raise _CanonicalCollectiveError("canonical collective header has an invalid semantic coordinate/order")
        if header[_H_TRANSPORT] not in {0, 1, 2, 3} or header[_H_CALLBACK_FLAGS] not in {0, 1, 2, 3}:
            raise _CanonicalCollectiveError("canonical collective header has invalid transport/callback flags")
    failed = tuple(index for index, header in enumerate(headers) if header[_H_STATUS])
    if failed:
        raise _error_from_diagnostics(
            "preparation",
            diagnostics,
            failed,
        )
    consensus_ranges = (
        (_H_OPERATION_BYTES, _H_CONTEXT_DIGEST),
        (_H_CONTEXT_DIGEST, _H_METADATA_DIGEST),
        (_H_LIMITS, _HEADER_FIELDS),
    )
    reference = headers[0]
    if any(
        header[start:stop] != reference[start:stop]
        for header in headers[1:]
        for start, stop in consensus_ranges
    ):
        raise _CanonicalCollectiveError(
            "canonical collective operation, transaction, context, identity, callback configuration, or limits diverged"
        )
    limits = reference[_H_LIMITS:_HEADER_FIELDS]
    limit_by_name = dict(zip(_LIMIT_NAMES, limits))
    total_metadata = 0
    total_tensors = 0
    for header in headers:
        metadata_bytes = header[_H_METADATA_BYTES]
        tensor_count = header[_H_TENSOR_COUNT]
        tensor_bytes = header[_H_TENSOR_BYTES]
        for name, value in (
            ("metadata bytes", metadata_bytes),
            ("tensor count", tensor_count),
            ("tensor bytes", tensor_bytes),
        ):
            _require_signed_header_int(name, value)
        if metadata_bytes > limit_by_name["max_metadata_bytes"]:
            raise _CanonicalCollectiveError("canonical collective fragment metadata exceeds max_metadata_bytes")
        if tensor_count > limit_by_name["max_tensors"]:
            raise _CanonicalCollectiveError("canonical collective fragment exceeds max_tensors")
        if tensor_bytes > limit_by_name["max_fragment_tensor_bytes"]:
            raise _CanonicalCollectiveError("canonical collective fragment exceeds max_fragment_tensor_bytes")
        if total_metadata > limit_by_name["max_collective_metadata_bytes"] - metadata_bytes:
            raise _CanonicalCollectiveError("canonical collective metadata exceeds max_collective_metadata_bytes")
        if total_tensors > limit_by_name["max_collective_tensor_bytes"] - tensor_bytes:
            raise _CanonicalCollectiveError("canonical collective tensors exceed max_collective_tensor_bytes")
        total_metadata += metadata_bytes
        total_tensors += tensor_bytes


def _broadcast_tensor(
    binding: CheckpointProcessGroupBinding,
    value: torch.Tensor,
    *,
    source: int,
    member_count: int,
) -> None:
    if member_count == 1:
        return
    import torch.distributed as dist

    global_source = dist.get_global_rank(binding.process_group, source)
    dist.broadcast(value, src=global_source, group=binding.process_group)


def _metadata_from_tensor(value: torch.Tensor) -> bytes:
    return bytes(value.detach().cpu().numpy().tobytes())


def _canonical_byte_indices(start: int, stop: int, component_bytes: int) -> torch.Tensor:
    canonical = torch.arange(start, stop, dtype=torch.int64)
    base = torch.div(canonical, component_bytes, rounding_mode="floor") * component_bytes
    return base + (component_bytes - 1 - torch.remainder(canonical, component_bytes))


def _flat_native_bytes(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(-1).view(torch.uint8).reshape(-1)


def _canonical_cpu_chunk(value: torch.Tensor, start: int, stop: int) -> torch.Tensor:
    raw = _flat_native_bytes(value)
    if sys.byteorder == "little":
        return raw[start:stop].clone()
    if sys.byteorder != "big":
        raise RuntimeError("unsupported native byte order")
    component_bytes = value.element_size() // 2 if value.is_complex() else value.element_size()
    if component_bytes == 1:
        return raw[start:stop].clone()
    return raw.index_select(0, _canonical_byte_indices(start, stop, component_bytes))


def _store_canonical_cpu_chunk(
    destination: torch.Tensor,
    start: int,
    stop: int,
    canonical: torch.Tensor,
) -> None:
    raw = _flat_native_bytes(destination)
    if sys.byteorder == "little":
        raw[start:stop].copy_(canonical)
        return
    if sys.byteorder != "big":
        raise RuntimeError("unsupported native byte order")
    component_bytes = destination.element_size() // 2 if destination.is_complex() else destination.element_size()
    if component_bytes == 1:
        raw[start:stop].copy_(canonical)
        return
    raw.index_copy_(0, _canonical_byte_indices(start, stop, component_bytes), canonical)


def _header_digest(header: tuple[int, ...], offset: int) -> bytes:
    return _fields_digest(header[offset : offset + _DIGEST_FIELDS])


def _receive_metadata(
    binding: CheckpointProcessGroupBinding,
    local_plan: _CanonicalWirePlan,
    headers: tuple[tuple[int, ...], ...],
    limits: _CanonicalWireLimits,
    *,
    coordinate: int,
    source: int,
    member_count: int,
    validate_plan,
) -> _PreparedCanonicalWireValue:
    header = headers[source]
    metadata_bytes = header[_H_METADATA_BYTES]
    metadata_buffer = None
    error = None
    try:
        if coordinate == source:
            metadata_buffer = torch.frombuffer(bytearray(local_plan.metadata), dtype=torch.uint8).to(
                binding.collective_device
            )
        else:
            metadata_buffer = torch.empty(
                metadata_bytes,
                dtype=torch.uint8,
                device=binding.collective_device,
            )
        if metadata_buffer.numel() != metadata_bytes:
            raise ValueError("canonical collective metadata buffer has an invalid size")
    except Exception as exc:
        error = exc
    _phase_vote(
        binding,
        member_count=member_count,
        coordinate=coordinate,
        source=source,
        phase=_PHASE_METADATA_BUFFER,
        phase_name="metadata-buffer allocation",
        error=error,
        diagnostic_limit=limits.diagnostic_bytes,
    )
    assert metadata_buffer is not None
    _broadcast_tensor(binding, metadata_buffer, source=source, member_count=member_count)

    prepared = None
    error = None
    try:
        metadata = _metadata_from_tensor(metadata_buffer)
        if not hmac.compare_digest(hashlib.sha256(metadata).digest(), _header_digest(header, _H_METADATA_DIGEST)):
            raise ValueError("canonical collective metadata digest does not match its source header")
        prepared = _parse_canonical_wire_metadata(metadata, limits=limits)
        if len(prepared.tensor_specs) != header[_H_TENSOR_COUNT]:
            raise ValueError("canonical collective tensor count does not match its source header")
        if prepared.total_tensor_bytes != header[_H_TENSOR_BYTES]:
            raise ValueError("canonical collective tensor bytes do not match its source header")
        if not hmac.compare_digest(prepared.fragment_digest, _header_digest(header, _H_FRAGMENT_DIGEST)):
            raise ValueError("canonical collective fragment digest does not match its source header")
        if validate_plan is not None:
            validate_plan(binding.identity.ordered_members[source], prepared)
    except Exception as exc:
        error = exc
    _phase_vote(
        binding,
        member_count=member_count,
        coordinate=coordinate,
        source=source,
        phase=_PHASE_METADATA_PARSE,
        phase_name="metadata validation",
        error=error,
        diagnostic_limit=limits.diagnostic_bytes,
    )
    assert prepared is not None
    return prepared


def _allocate_payloads(
    binding: CheckpointProcessGroupBinding,
    prepared: _PreparedCanonicalWireValue,
    limits: _CanonicalWireLimits,
    *,
    coordinate: int,
    source: int,
    member_count: int,
) -> tuple[torch.Tensor, ...]:
    payloads = None
    error = None
    try:
        payloads = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device="cpu")
            for spec in prepared.tensor_specs
        )
    except Exception as exc:
        error = exc
    _phase_vote(
        binding,
        member_count=member_count,
        coordinate=coordinate,
        source=source,
        phase=_PHASE_PAYLOAD_ALLOCATION,
        phase_name="payload allocation",
        error=error,
        diagnostic_limit=limits.diagnostic_bytes,
    )
    assert payloads is not None
    return payloads


def _transfer_payloads(
    binding: CheckpointProcessGroupBinding,
    local_plan: _CanonicalWirePlan,
    prepared: _PreparedCanonicalWireValue,
    payloads: tuple[torch.Tensor, ...],
    limits: _CanonicalWireLimits,
    *,
    coordinate: int,
    source: int,
    member_count: int,
    direct_cpu: bool,
) -> None:
    for tensor_index, (spec, destination) in enumerate(zip(prepared.tensor_specs, payloads)):
        for start in range(0, spec.nbytes, limits.chunk_bytes):
            stop = min(start + limits.chunk_bytes, spec.nbytes)
            wire_buffer = None
            error = None
            try:
                if direct_cpu:
                    wire_buffer = _flat_native_bytes(destination)[start:stop]
                    if coordinate == source:
                        source_bytes = _flat_native_bytes(local_plan.tensors[tensor_index])
                        wire_buffer.copy_(source_bytes[start:stop])
                else:
                    wire_buffer = torch.empty(
                        stop - start,
                        dtype=torch.uint8,
                        device=binding.collective_device,
                    )
                    if coordinate == source:
                        canonical = _canonical_cpu_chunk(local_plan.tensors[tensor_index], start, stop)
                        wire_buffer.copy_(canonical)
            except Exception as exc:
                error = exc
            _phase_vote(
                binding,
                member_count=member_count,
                coordinate=coordinate,
                source=source,
                phase=_PHASE_CHUNK_PREPARE,
                phase_name="payload chunk {}:{} preparation".format(tensor_index, start),
                error=error,
                diagnostic_limit=limits.diagnostic_bytes,
            )
            assert wire_buffer is not None
            _broadcast_tensor(binding, wire_buffer, source=source, member_count=member_count)
            if not direct_cpu:
                error = None
                try:
                    canonical = wire_buffer.detach().cpu()
                    _store_canonical_cpu_chunk(destination, start, stop, canonical)
                except Exception as exc:
                    error = exc
                _phase_vote(
                    binding,
                    member_count=member_count,
                    coordinate=coordinate,
                    source=source,
                    phase=_PHASE_CHUNK_STAGE,
                    phase_name="payload chunk {}:{} staging".format(tensor_index, start),
                    error=error,
                    diagnostic_limit=limits.diagnostic_bytes,
                )


def _collective_unanimous_status(
    binding: CheckpointProcessGroupBinding,
    local_error: Exception | None,
    *,
    operation: str,
    transaction_id: str,
    context_digest: bytes,
    limits: _CanonicalWireLimits,
) -> None:
    """Exchange one fixed-size pre-publication status across all live members.

    This is the lightweight companion to the fragment visitor for import
    readiness and freshness gates. It transfers only fixed-size status/header
    tensors and diagnostics; it never broadcasts canonical metadata or payloads.
    """

    if not isinstance(binding, CheckpointProcessGroupBinding):
        raise TypeError("binding must be a CheckpointProcessGroupBinding")
    member_count = _runtime_size(binding)
    coordinate = binding.identity.ordered_members.index(binding.local_member)
    header = _failure_header(coordinate, member_count)
    error = None
    diagnostic_limit = _DIAGNOSTIC_BYTES
    try:
        binding.validate_runtime()
        if type(limits) is not _CanonicalWireLimits:
            raise TypeError("limits must be _CanonicalWireLimits")
        diagnostic_limit = limits.diagnostic_bytes
        if local_error is not None and not isinstance(local_error, Exception):
            raise TypeError("local_error must be an Exception or None")
        operation_bytes = _label_bytes("operation", operation)
        transaction_bytes = _label_bytes("transaction_id", transaction_id)
        if type(context_digest) is not bytes or len(context_digest) != hashlib.sha256().digest_size:
            raise ValueError("context_digest must be exactly 32 bytes")
        header = _success_header(
            binding,
            None,
            limits,
            coordinate=coordinate,
            member_count=member_count,
            operation=operation_bytes,
            transaction=transaction_bytes,
            context_digest=context_digest,
            callback_flags=0,
        )
        if local_error is not None:
            error = local_error
            header = _failure_header(coordinate, member_count)
    except Exception as exc:
        error = exc
    headers, diagnostics = _exchange_header(
        binding,
        header,
        _diagnostic(error, limit=diagnostic_limit),
        member_count=member_count,
    )
    _validate_initial_exchange(
        binding,
        headers,
        diagnostics,
        member_count=member_count,
    )


def _collective_visit_canonical_fragments(
    binding: CheckpointProcessGroupBinding,
    local_fragment,
    *,
    operation: str,
    transaction_id: str,
    context_digest: bytes,
    limits: _CanonicalWireLimits,
    validate_plan=None,
    consume=None,
) -> None:
    """Visit one staged canonical fragment from every semantic member in order.

    ``validate_plan(member, prepared)`` runs after bounded metadata validation
    and before payload allocation. ``consume(member, value)`` receives an owned,
    digest-verified value. Callbacks must stage only reversible local work; this
    transport does not publish or roll back callback side effects.
    """

    if not isinstance(binding, CheckpointProcessGroupBinding):
        raise TypeError("binding must be a CheckpointProcessGroupBinding")
    member_count = _runtime_size(binding)
    members = binding.identity.ordered_members
    coordinate = members.index(binding.local_member)
    plan = None
    header = _failure_header(coordinate, member_count)
    error = None
    diagnostic_limit = _DIAGNOSTIC_BYTES
    try:
        binding.validate_runtime()
        if type(limits) is not _CanonicalWireLimits:
            raise TypeError("limits must be _CanonicalWireLimits")
        diagnostic_limit = limits.diagnostic_bytes
        if validate_plan is not None and not callable(validate_plan):
            raise TypeError("validate_plan must be callable or None")
        if consume is not None and not callable(consume):
            raise TypeError("consume must be callable or None")
        operation_bytes = _label_bytes("operation", operation)
        transaction_bytes = _label_bytes("transaction_id", transaction_id)
        if type(context_digest) is not bytes or len(context_digest) != hashlib.sha256().digest_size:
            raise ValueError("context_digest must be exactly 32 bytes")
        plan = _prepare_canonical_wire_value(local_fragment, limits)
        header = _success_header(
            binding,
            plan,
            limits,
            coordinate=coordinate,
            member_count=member_count,
            operation=operation_bytes,
            transaction=transaction_bytes,
            context_digest=context_digest,
            callback_flags=(1 if validate_plan is not None else 0) | (2 if consume is not None else 0),
        )
    except Exception as exc:
        error = exc
    headers, diagnostics = _exchange_header(
        binding,
        header,
        _diagnostic(error, limit=diagnostic_limit),
        member_count=member_count,
    )
    _validate_initial_exchange(
        binding,
        headers,
        diagnostics,
        member_count=member_count,
    )
    assert plan is not None
    assert type(limits) is _CanonicalWireLimits
    direct_cpu = all(header[_H_TRANSPORT] == 0 for header in headers)

    for source in range(member_count):
        prepared = _receive_metadata(
            binding,
            plan,
            headers,
            limits,
            coordinate=coordinate,
            source=source,
            member_count=member_count,
            validate_plan=validate_plan,
        )
        payloads = _allocate_payloads(
            binding,
            prepared,
            limits,
            coordinate=coordinate,
            source=source,
            member_count=member_count,
        )
        _transfer_payloads(
            binding,
            plan,
            prepared,
            payloads,
            limits,
            coordinate=coordinate,
            source=source,
            member_count=member_count,
            direct_cpu=direct_cpu,
        )
        value = None
        error = None
        try:
            value = _reconstruct_canonical_wire_value(
                prepared,
                payloads,
                expected_fragment_digest=_header_digest(headers[source], _H_FRAGMENT_DIGEST),
            )
        except Exception as exc:
            error = exc
        _phase_vote(
            binding,
            member_count=member_count,
            coordinate=coordinate,
            source=source,
            phase=_PHASE_RECONSTRUCT,
            phase_name="fragment reconstruction",
            error=error,
            diagnostic_limit=limits.diagnostic_bytes,
        )
        error = None
        if consume is not None:
            try:
                consume(members[source], value)
            except Exception as exc:
                error = exc
        _phase_vote(
            binding,
            member_count=member_count,
            coordinate=coordinate,
            source=source,
            phase=_PHASE_CONSUME,
            phase_name="fragment consumer",
            error=error,
            diagnostic_limit=limits.diagnostic_bytes,
        )


__all__ = []

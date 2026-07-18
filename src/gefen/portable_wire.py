"""Bounded binary wire codec for canonical optimizer-state fragments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import math
import struct
import sys

import torch


_CANONICAL_WIRE_PREAMBLE = b"GFNCV1\0\0"
_CANONICAL_WIRE_CODEC = 1
_CANONICAL_WIRE_TENSOR_DOMAIN = b"gefen.canonical_wire.tensor.v1\0"
_CANONICAL_WIRE_FRAGMENT_DOMAIN = b"gefen.canonical_wire.fragment.v1\0"
_SHA256_BYTES = hashlib.sha256().digest_size
_U16_MAX = (1 << 16) - 1
_I64_MAX = (1 << 63) - 1
_U64_MAX = (1 << 64) - 1
_MAX_SAFE_TREE_DEPTH = 256

_TAG_NONE = 0x00
_TAG_FALSE = 0x01
_TAG_TRUE = 0x02
_TAG_INTEGER = 0x03
_TAG_FLOAT = 0x04
_TAG_STRING = 0x05
_TAG_LIST = 0x06
_TAG_TUPLE = 0x07
_TAG_DICTIONARY = 0x08
_TAG_TENSOR = 0x09

_DTYPE_RECORDS = (
    (1, torch.bool, 1),
    (2, torch.uint8, 1),
    (3, torch.int8, 1),
    (4, torch.int16, 2),
    (5, torch.int32, 4),
    (6, torch.int64, 8),
    (7, torch.float16, 2),
    (8, torch.bfloat16, 2),
    (9, torch.float32, 4),
    (10, torch.float64, 8),
    (11, torch.complex64, 8),
    (12, torch.complex128, 16),
)
_DTYPE_BY_CODE = {code: (dtype, size) for code, dtype, size in _DTYPE_RECORDS}
_DTYPE_BY_VALUE = {dtype: (code, size) for code, dtype, size in _DTYPE_RECORDS}


def _require_positive_limit(name: str, value: int, *, maximum: int = _U64_MAX) -> None:
    if type(value) is not int:
        raise TypeError("{} must be an int".format(name))
    if value <= 0 or value > maximum:
        raise ValueError("{} must be in [1, {}]".format(name, maximum))


@dataclass(frozen=True, slots=True)
class _CanonicalWireLimits:
    max_fragment_tensor_bytes: int
    max_collective_tensor_bytes: int
    max_collective_metadata_bytes: int
    chunk_bytes: int = 8 << 20
    max_members: int = 4096
    max_metadata_bytes: int = 64 << 20
    max_tree_nodes: int = 1_000_000
    max_tree_depth: int = 64
    max_container_items: int = 1_000_000
    max_string_bytes: int = 1 << 20
    max_integer_bytes: int = 4096
    max_tensors: int = 262_144
    max_tensor_rank: int = 64
    diagnostic_bytes: int = 2048

    def __post_init__(self) -> None:
        for name in (
            "max_fragment_tensor_bytes",
            "max_collective_tensor_bytes",
            "max_collective_metadata_bytes",
            "chunk_bytes",
            "max_members",
            "max_metadata_bytes",
            "max_tree_nodes",
            "max_tree_depth",
            "max_container_items",
            "max_string_bytes",
            "max_integer_bytes",
            "max_tensors",
            "diagnostic_bytes",
        ):
            _require_positive_limit(name, getattr(self, name))
        _require_positive_limit("max_tensor_rank", self.max_tensor_rank, maximum=_U16_MAX)
        if self.max_tree_depth > _MAX_SAFE_TREE_DEPTH:
            raise ValueError(
                "max_tree_depth cannot exceed {} for recursive parsing".format(
                    _MAX_SAFE_TREE_DEPTH
                )
            )
        if self.max_fragment_tensor_bytes > self.max_collective_tensor_bytes:
            raise ValueError("max_fragment_tensor_bytes cannot exceed max_collective_tensor_bytes")
        if self.max_metadata_bytes > self.max_collective_metadata_bytes:
            raise ValueError("max_metadata_bytes cannot exceed max_collective_metadata_bytes")


@dataclass(frozen=True, slots=True)
class _CanonicalWireTensorSpec:
    index: int
    dtype_code: int
    dtype: torch.dtype
    shape: tuple[int, ...]
    numel: int
    nbytes: int
    digest: bytes

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0 or self.index > _U64_MAX:
            raise ValueError("tensor index must be a nonnegative int")
        if type(self.dtype_code) is not int:
            raise TypeError("tensor dtype code must be an int")
        record = _DTYPE_BY_CODE.get(self.dtype_code)
        if record is None or record[0] != self.dtype:
            raise ValueError("tensor dtype code and dtype do not agree")
        if (
            type(self.shape) is not tuple
            or len(self.shape) > _U16_MAX
            or any(
                type(dimension) is not int
                or dimension < 0
                or dimension > _I64_MAX
                for dimension in self.shape
            )
        ):
            raise ValueError("tensor shape must be a tuple of nonnegative ints")
        if type(self.numel) is not int or self.numel < 0 or self.numel > _U64_MAX:
            raise ValueError("tensor numel must be a nonnegative int")
        if type(self.nbytes) is not int or self.nbytes < 0 or self.nbytes > _U64_MAX:
            raise ValueError("tensor nbytes must be a nonnegative int")
        if type(self.digest) is not bytes or len(self.digest) != _SHA256_BYTES:
            raise ValueError("tensor digest must be a 32-byte value")
        if _shape_numel(self.shape) != self.numel or self.numel * record[1] != self.nbytes:
            raise ValueError("tensor shape, numel, and nbytes do not agree")


@dataclass(frozen=True, slots=True)
class _CanonicalWirePlan:
    metadata: bytes
    tensors: tuple[torch.Tensor, ...]
    tensor_specs: tuple[_CanonicalWireTensorSpec, ...]
    metadata_digest: bytes
    fragment_digest: bytes
    total_tensor_bytes: int

    def __post_init__(self) -> None:
        if type(self.metadata) is not bytes:
            raise TypeError("canonical wire metadata must be bytes")
        if type(self.tensors) is not tuple or any(type(value) is not torch.Tensor for value in self.tensors):
            raise TypeError("canonical wire tensors must be a tuple of plain tensors")
        if type(self.tensor_specs) is not tuple or any(type(value) is not _CanonicalWireTensorSpec for value in self.tensor_specs):
            raise TypeError("canonical wire tensor specs must be a tuple of tensor specs")
        if len(self.tensors) != len(self.tensor_specs):
            raise ValueError("canonical wire tensors and specs must have equal lengths")
        if any(spec.index != index for index, spec in enumerate(self.tensor_specs)):
            raise ValueError("canonical wire tensor specs must have consecutive indices")
        if any(not _tight_cpu_tensor(tensor, spec) for tensor, spec in zip(self.tensors, self.tensor_specs)):
            raise ValueError("canonical wire tensors must be tight CPU tensors matching their specs")
        _validate_digest_field("metadata_digest", self.metadata_digest)
        _validate_digest_field("fragment_digest", self.fragment_digest)
        if type(self.total_tensor_bytes) is not int or self.total_tensor_bytes < 0:
            raise ValueError("total_tensor_bytes must be a nonnegative int")
        if sum(spec.nbytes for spec in self.tensor_specs) != self.total_tensor_bytes:
            raise ValueError("total_tensor_bytes does not match the tensor specs")
        if not hmac.compare_digest(hashlib.sha256(self.metadata).digest(), self.metadata_digest):
            raise ValueError("metadata_digest does not match canonical wire metadata")
        if not hmac.compare_digest(_canonical_fragment_digest(self.metadata_digest, self.tensor_specs), self.fragment_digest):
            raise ValueError("fragment_digest does not match canonical wire metadata and tensor specs")

    @property
    def payload_tensors(self) -> tuple[torch.Tensor, ...]:
        return self.tensors


@dataclass(frozen=True, slots=True)
class _TensorReference:
    index: int


@dataclass(frozen=True, slots=True)
class _ListValue:
    values: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _TupleValue:
    values: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _DictionaryValue:
    values: tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class _PreparedCanonicalWireValue:
    tensor_specs: tuple[_CanonicalWireTensorSpec, ...]
    metadata_digest: bytes
    fragment_digest: bytes
    total_tensor_bytes: int
    _value: object
    _limits: _CanonicalWireLimits

    def __post_init__(self) -> None:
        if type(self.tensor_specs) is not tuple or any(type(value) is not _CanonicalWireTensorSpec for value in self.tensor_specs):
            raise TypeError("canonical wire tensor specs must be a tuple of tensor specs")
        if any(spec.index != index for index, spec in enumerate(self.tensor_specs)):
            raise ValueError("canonical wire tensor specs must have consecutive indices")
        _validate_digest_field("metadata_digest", self.metadata_digest)
        _validate_digest_field("fragment_digest", self.fragment_digest)
        if type(self.total_tensor_bytes) is not int or self.total_tensor_bytes < 0:
            raise ValueError("total_tensor_bytes must be a nonnegative int")
        if sum(spec.nbytes for spec in self.tensor_specs) != self.total_tensor_bytes:
            raise ValueError("total_tensor_bytes does not match the tensor specs")
        if not hmac.compare_digest(_canonical_fragment_digest(self.metadata_digest, self.tensor_specs), self.fragment_digest):
            raise ValueError("fragment_digest does not match canonical wire metadata and tensor specs")
        if type(self._limits) is not _CanonicalWireLimits:
            raise TypeError("prepared canonical wire values require canonical wire limits")


def _validate_digest_field(name: str, value: bytes) -> None:
    if type(value) is not bytes:
        raise TypeError("{} must be bytes".format(name))
    if len(value) != _SHA256_BYTES:
        raise ValueError("{} must be a 32-byte value".format(name))


def _shape_numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        if dimension != 0 and result > _U64_MAX // dimension:
            raise ValueError("tensor shape product overflows u64")
        result *= dimension
    return result


class _MetadataWriter:
    __slots__ = ("_data", "_limit")

    def __init__(self, limit: int):
        self._data = bytearray()
        self._limit = limit

    def write(self, value: bytes) -> None:
        if len(self._data) + len(value) > self._limit:
            raise ValueError("canonical wire metadata exceeds max_metadata_bytes")
        self._data.extend(value)

    def finish(self) -> bytes:
        return bytes(self._data)


class _PreparationState:
    __slots__ = ("limits", "writer", "nodes", "tensors", "specs", "tensor_bytes")

    def __init__(self, limits: _CanonicalWireLimits):
        self.limits = limits
        self.writer = _MetadataWriter(limits.max_metadata_bytes)
        self.nodes = 0
        self.tensors: list[torch.Tensor] = []
        self.specs: list[_CanonicalWireTensorSpec] = []
        self.tensor_bytes = 0

    def visit(self, value, *, depth: int, path: str) -> None:
        if depth > self.limits.max_tree_depth:
            raise ValueError("canonical wire value exceeds max_tree_depth at {}".format(path))
        self.nodes += 1
        if self.nodes > self.limits.max_tree_nodes:
            raise ValueError("canonical wire value exceeds max_tree_nodes")
        value_type = type(value)
        if value is None:
            self.writer.write(bytes((_TAG_NONE,)))
            return
        if value_type is bool:
            self.writer.write(bytes((_TAG_TRUE if value else _TAG_FALSE,)))
            return
        if value_type is int:
            magnitude = abs(value)
            magnitude_size = (magnitude.bit_length() + 7) // 8
            if magnitude_size > self.limits.max_integer_bytes:
                raise ValueError("integer exceeds max_integer_bytes at {}".format(path))
            magnitude_bytes = magnitude.to_bytes(magnitude_size, "big")
            self.writer.write(bytes((_TAG_INTEGER, 1 if value < 0 else 0)))
            self.writer.write(struct.pack(">Q", len(magnitude_bytes)))
            self.writer.write(magnitude_bytes)
            return
        if value_type is float:
            if not math.isfinite(value):
                raise ValueError("float must be finite at {}".format(path))
            self.writer.write(bytes((_TAG_FLOAT,)))
            self.writer.write(struct.pack(">d", value))
            return
        if value_type is str:
            self._write_string(value, path=path)
            return
        if torch.is_tensor(value):
            self._write_tensor(value, path=path)
            return
        if value_type is list or value_type is tuple:
            if len(value) > self.limits.max_container_items:
                raise ValueError("container exceeds max_container_items at {}".format(path))
            self.writer.write(bytes((_TAG_LIST if value_type is list else _TAG_TUPLE,)))
            self.writer.write(struct.pack(">Q", len(value)))
            for index, item in enumerate(value):
                self.visit(item, depth=depth + 1, path="{}[{}]".format(path, index))
            return
        if value_type is dict:
            if len(value) > self.limits.max_container_items:
                raise ValueError("container exceeds max_container_items at {}".format(path))
            if any(type(key) is not str for key in value):
                raise TypeError("dictionary keys must be strings at {}".format(path))
            self.writer.write(bytes((_TAG_DICTIONARY,)))
            self.writer.write(struct.pack(">Q", len(value)))
            for key in sorted(value):
                self.visit(key, depth=depth + 1, path="{}.<key>".format(path))
                self.visit(value[key], depth=depth + 1, path="{}.{}".format(path, key))
            return
        raise TypeError("{} has unsupported canonical wire type {}".format(path, value_type.__name__))

    def _write_string(self, value: str, *, path: str) -> None:
        encoded = _bounded_utf8(value, limit=self.limits.max_string_bytes, path=path)
        self.writer.write(bytes((_TAG_STRING,)))
        self.writer.write(struct.pack(">Q", len(encoded)))
        self.writer.write(encoded)

    def _write_tensor(self, value: torch.Tensor, *, path: str) -> None:
        if type(value) is not torch.Tensor:
            raise TypeError("tensor must be a plain torch.Tensor at {}".format(path))
        if value.layout is not torch.strided or value.is_meta or value.is_nested or value.is_quantized:
            raise TypeError("tensor must be a materialized strided tensor at {}".format(path))
        dtype_record = _DTYPE_BY_VALUE.get(value.dtype)
        if dtype_record is None:
            raise TypeError("tensor dtype {} is not supported at {}".format(value.dtype, path))
        shape = tuple(value.shape)
        if len(shape) > self.limits.max_tensor_rank:
            raise ValueError("tensor rank exceeds max_tensor_rank at {}".format(path))
        if len(self.tensors) >= self.limits.max_tensors:
            raise ValueError("canonical wire value exceeds max_tensors")
        dtype_code, element_size = dtype_record
        numel = value.numel()
        nbytes = numel * element_size
        if nbytes > _U64_MAX or self.tensor_bytes + nbytes > self.limits.max_fragment_tensor_bytes:
            raise ValueError("canonical wire tensors exceed max_fragment_tensor_bytes")
        index = len(self.tensors)
        cloned = _clone_tensor(value, chunk_bytes=self.limits.chunk_bytes, path=path)
        digest = _canonical_tensor_digest(
            cloned,
            dtype_code=dtype_code,
            shape=shape,
            nbytes=nbytes,
            chunk_bytes=self.limits.chunk_bytes,
        )
        spec = _CanonicalWireTensorSpec(index, dtype_code, value.dtype, shape, numel, nbytes, digest)
        self.tensors.append(cloned)
        self.specs.append(spec)
        self.tensor_bytes += nbytes
        self.writer.write(bytes((_TAG_TENSOR,)))
        self.writer.write(struct.pack(">Q", index))


def _bounded_utf8(value: str, *, limit: int, path: str) -> bytes:
    encoded = bytearray()
    characters_per_chunk = max(1, min(1 << 14, limit // 4))
    for start in range(0, len(value), characters_per_chunk):
        chunk = value[start : start + characters_per_chunk].encode("utf-8")
        if len(encoded) + len(chunk) > limit:
            raise ValueError("string exceeds max_string_bytes at {}".format(path))
        encoded.extend(chunk)
    return bytes(encoded)


def _read_tensor_chunk(value: torch.Tensor, start: int, stop: int) -> torch.Tensor:
    detached = value.detach()
    if detached.is_contiguous():
        return detached.reshape(-1)[start:stop]
    linear = torch.arange(start, stop, dtype=torch.int64, device=detached.device)
    remainder = linear
    reversed_coordinates = []
    for dimension in reversed(detached.shape):
        reversed_coordinates.append(torch.remainder(remainder, dimension))
        remainder = torch.div(remainder, dimension, rounding_mode="floor")
    return detached[tuple(reversed(reversed_coordinates))]


def _clone_tensor(value: torch.Tensor, *, chunk_bytes: int, path: str) -> torch.Tensor:
    cloned = torch.empty(tuple(value.shape), dtype=value.dtype, device="cpu")
    flat = cloned.reshape(-1)
    scratch_bytes_per_element = value.element_size()
    if not value.is_contiguous():
        # Logical indexing retains one int64 coordinate vector per dimension,
        # plus the linear index and current quotient. Keep that scratch within
        # the same caller-supplied chunk budget as the tensor copy.
        scratch_bytes_per_element += 8 * (value.ndim + 2)
    chunk_elements = max(1, chunk_bytes // scratch_bytes_per_element)
    for start in range(0, value.numel(), chunk_elements):
        stop = min(start + chunk_elements, value.numel())
        source = _read_tensor_chunk(value, start, stop).resolve_conj().resolve_neg().reshape(-1)
        destination = flat[start:stop]
        destination.copy_(source)
        if cloned.is_floating_point() or cloned.is_complex():
            try:
                finite = bool(torch.isfinite(destination).all())
            except (NotImplementedError, RuntimeError, TypeError) as exc:
                raise ValueError("tensor dtype does not support finite values at {}".format(path)) from exc
            if not finite:
                raise ValueError("tensor must be finite at {}".format(path))
    return cloned


def _canonical_tensor_digest(
    value: torch.Tensor,
    *,
    dtype_code: int,
    shape: tuple[int, ...],
    nbytes: int,
    chunk_bytes: int,
) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(_CANONICAL_WIRE_TENSOR_DOMAIN)
    hasher.update(struct.pack(">HH", dtype_code, len(shape)))
    for dimension in shape:
        hasher.update(struct.pack(">Q", dimension))
    hasher.update(struct.pack(">Q", nbytes))
    element_size = value.element_size()
    component_size = element_size // 2 if value.is_complex() else element_size
    chunk_elements = max(1, chunk_bytes // element_size)
    flat = value.reshape(-1)
    for start in range(0, value.numel(), chunk_elements):
        stop = min(start + chunk_elements, value.numel())
        raw = flat[start:stop].view(torch.uint8).reshape(-1)
        if sys.byteorder == "big" and component_size > 1:
            raw = raw.reshape(-1, component_size).flip(1).contiguous().reshape(-1)
        elif sys.byteorder != "little":
            raise RuntimeError("unsupported native byte order")
        hasher.update(memoryview(raw.numpy()))
    return hasher.digest()


def _canonical_fragment_digest(metadata_digest: bytes, specs: tuple[_CanonicalWireTensorSpec, ...]) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(_CANONICAL_WIRE_FRAGMENT_DOMAIN)
    hasher.update(metadata_digest)
    hasher.update(struct.pack(">Q", len(specs)))
    for spec in specs:
        hasher.update(struct.pack(">Q", spec.nbytes))
        hasher.update(spec.digest)
    return hasher.digest()


def _prepare_canonical_wire_value(value, limits: _CanonicalWireLimits) -> _CanonicalWirePlan:
    """Clone and encode one canonical value into metadata and tensor payloads."""

    if type(limits) is not _CanonicalWireLimits:
        raise TypeError("limits must be _CanonicalWireLimits")
    state = _PreparationState(limits)
    state.writer.write(_CANONICAL_WIRE_PREAMBLE)
    state.writer.write(struct.pack(">HHI", _CANONICAL_WIRE_CODEC, 0, 0))
    state.visit(value, depth=0, path="value")
    state.writer.write(struct.pack(">Q", len(state.specs)))
    for spec in state.specs:
        state.writer.write(struct.pack(">HHI", spec.dtype_code, len(spec.shape), 0))
        for dimension in spec.shape:
            state.writer.write(struct.pack(">Q", dimension))
        state.writer.write(struct.pack(">QQ", spec.numel, spec.nbytes))
        state.writer.write(spec.digest)
    metadata = state.writer.finish()
    specs = tuple(state.specs)
    metadata_digest = hashlib.sha256(metadata).digest()
    fragment_digest = _canonical_fragment_digest(metadata_digest, specs)
    return _CanonicalWirePlan(
        metadata=metadata,
        tensors=tuple(state.tensors),
        tensor_specs=specs,
        metadata_digest=metadata_digest,
        fragment_digest=fragment_digest,
        total_tensor_bytes=state.tensor_bytes,
    )


class _MetadataReader:
    __slots__ = ("metadata", "limits", "position", "nodes", "next_tensor")

    def __init__(self, metadata: bytes, limits: _CanonicalWireLimits):
        self.metadata = metadata
        self.limits = limits
        self.position = 0
        self.nodes = 0
        self.next_tensor = 0

    def read(self, size: int) -> bytes:
        if size < 0 or size > len(self.metadata) - self.position:
            raise ValueError("canonical wire metadata is truncated")
        start = self.position
        self.position += size
        return self.metadata[start : self.position]

    def u8(self) -> int:
        return self.read(1)[0]

    def u16(self) -> int:
        return struct.unpack(">H", self.read(2))[0]

    def u32(self) -> int:
        return struct.unpack(">I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack(">Q", self.read(8))[0]

    def value(self, *, depth: int, dictionary_key: bool = False):
        if depth > self.limits.max_tree_depth:
            raise ValueError("canonical wire metadata exceeds max_tree_depth")
        self.nodes += 1
        if self.nodes > self.limits.max_tree_nodes:
            raise ValueError("canonical wire metadata exceeds max_tree_nodes")
        tag = self.u8()
        if tag == _TAG_NONE:
            result = None
        elif tag == _TAG_FALSE:
            result = False
        elif tag == _TAG_TRUE:
            result = True
        elif tag == _TAG_INTEGER:
            sign = self.u8()
            if sign not in {0, 1}:
                raise ValueError("canonical wire integer has an invalid sign")
            size = self.u64()
            if size > self.limits.max_integer_bytes:
                raise ValueError("canonical wire integer exceeds max_integer_bytes")
            encoded = self.read(size)
            if encoded[:1] == b"\0":
                raise ValueError("canonical wire integer magnitude is not canonical")
            magnitude = int.from_bytes(encoded, "big")
            if sign == 1 and magnitude == 0:
                raise ValueError("canonical wire integer encodes negative zero")
            result = -magnitude if sign else magnitude
        elif tag == _TAG_FLOAT:
            result = struct.unpack(">d", self.read(8))[0]
            if not math.isfinite(result):
                raise ValueError("canonical wire float must be finite")
        elif tag == _TAG_STRING:
            result = self.string_after_tag()
        elif tag == _TAG_LIST or tag == _TAG_TUPLE:
            count = self.u64()
            if count > self.limits.max_container_items:
                raise ValueError("canonical wire container exceeds max_container_items")
            values = tuple(self.value(depth=depth + 1) for _ in range(count))
            result = _ListValue(values) if tag == _TAG_LIST else _TupleValue(values)
        elif tag == _TAG_DICTIONARY:
            count = self.u64()
            if count > self.limits.max_container_items:
                raise ValueError("canonical wire dictionary exceeds max_container_items")
            values = []
            previous = None
            for _ in range(count):
                key = self.value(depth=depth + 1, dictionary_key=True)
                if type(key) is not str:
                    raise ValueError("canonical wire dictionary key must be a string")
                if previous is not None and key <= previous:
                    raise ValueError("canonical wire dictionary keys are not strictly increasing")
                previous = key
                values.append((key, self.value(depth=depth + 1)))
            result = _DictionaryValue(tuple(values))
        elif tag == _TAG_TENSOR:
            index = self.u64()
            if index != self.next_tensor:
                raise ValueError("canonical wire tensor references are not ordered")
            if index >= self.limits.max_tensors:
                raise ValueError("canonical wire metadata exceeds max_tensors")
            self.next_tensor += 1
            result = _TensorReference(index)
        else:
            raise ValueError("canonical wire metadata has an unknown value tag")
        if dictionary_key and type(result) is not str:
            raise ValueError("canonical wire dictionary key must be a string")
        return result

    def string_after_tag(self) -> str:
        size = self.u64()
        if size > self.limits.max_string_bytes:
            raise ValueError("canonical wire string exceeds max_string_bytes")
        encoded = self.read(size)
        try:
            result = encoded.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("canonical wire string is not valid UTF-8") from exc
        if result.encode("utf-8") != encoded:
            raise ValueError("canonical wire string is not canonically encoded")
        return result


def _parse_canonical_wire_metadata(
    metadata: bytes,
    *,
    limits: _CanonicalWireLimits,
) -> _PreparedCanonicalWireValue:
    """Validate bounded metadata without allocating any tensor payloads."""

    if type(metadata) is not bytes:
        raise TypeError("metadata must be bytes")
    if type(limits) is not _CanonicalWireLimits:
        raise TypeError("limits must be _CanonicalWireLimits")
    if len(metadata) > limits.max_metadata_bytes:
        raise ValueError("canonical wire metadata exceeds max_metadata_bytes")
    reader = _MetadataReader(metadata, limits)
    if reader.read(len(_CANONICAL_WIRE_PREAMBLE)) != _CANONICAL_WIRE_PREAMBLE:
        raise ValueError("canonical wire metadata has an invalid preamble")
    if reader.u16() != _CANONICAL_WIRE_CODEC:
        raise ValueError("canonical wire metadata uses an unsupported codec")
    if reader.u16() != 0 or reader.u32() != 0:
        raise ValueError("canonical wire metadata has nonzero reserved header fields")
    value = reader.value(depth=0)
    descriptor_count = reader.u64()
    if descriptor_count != reader.next_tensor:
        raise ValueError("canonical wire descriptor count does not match tensor references")
    if descriptor_count > limits.max_tensors:
        raise ValueError("canonical wire metadata exceeds max_tensors")
    specs = []
    total_tensor_bytes = 0
    for index in range(descriptor_count):
        dtype_code = reader.u16()
        dtype_record = _DTYPE_BY_CODE.get(dtype_code)
        if dtype_record is None:
            raise ValueError("canonical wire tensor descriptor has an unknown dtype")
        rank = reader.u16()
        if rank > limits.max_tensor_rank:
            raise ValueError("canonical wire tensor rank exceeds max_tensor_rank")
        if reader.u32() != 0:
            raise ValueError("canonical wire tensor descriptor has nonzero flags")
        shape = tuple(reader.u64() for _ in range(rank))
        if any(dimension > _I64_MAX for dimension in shape):
            raise ValueError("canonical wire tensor dimension exceeds signed int64")
        numel = reader.u64()
        nbytes = reader.u64()
        digest = reader.read(_SHA256_BYTES)
        expected_numel = 1
        for dimension in shape:
            if dimension != 0 and expected_numel > _U64_MAX // dimension:
                raise ValueError("canonical wire tensor shape product overflows u64")
            expected_numel *= dimension
        if expected_numel != numel:
            raise ValueError("canonical wire tensor shape does not match numel")
        dtype, element_size = dtype_record
        if numel != 0 and numel > _U64_MAX // element_size:
            raise ValueError("canonical wire tensor byte count overflows u64")
        if numel * element_size != nbytes:
            raise ValueError("canonical wire tensor numel does not match nbytes")
        if total_tensor_bytes + nbytes > limits.max_fragment_tensor_bytes:
            raise ValueError("canonical wire tensors exceed max_fragment_tensor_bytes")
        specs.append(_CanonicalWireTensorSpec(index, dtype_code, dtype, shape, numel, nbytes, digest))
        total_tensor_bytes += nbytes
    if reader.position != len(metadata):
        raise ValueError("canonical wire metadata has trailing bytes")
    specs_tuple = tuple(specs)
    metadata_digest = hashlib.sha256(metadata).digest()
    fragment_digest = _canonical_fragment_digest(metadata_digest, specs_tuple)
    return _PreparedCanonicalWireValue(
        tensor_specs=specs_tuple,
        metadata_digest=metadata_digest,
        fragment_digest=fragment_digest,
        total_tensor_bytes=total_tensor_bytes,
        _value=value,
        _limits=limits,
    )


def _tight_cpu_tensor(value: torch.Tensor, spec: _CanonicalWireTensorSpec) -> bool:
    if (
        type(value) is not torch.Tensor
        or value.layout is not torch.strided
        or value.device.type != "cpu"
        or value.is_meta
        or value.is_nested
        or value.is_quantized
        or value.dtype != spec.dtype
        or tuple(value.shape) != spec.shape
        or value.numel() != spec.numel
        or value.element_size() * value.numel() != spec.nbytes
        or not value.is_contiguous()
        or value.is_conj()
        or value.is_neg()
        or value.storage_offset() != 0
    ):
        return False
    try:
        return value.untyped_storage().nbytes() == spec.nbytes
    except (AttributeError, RuntimeError):
        return False


def _materialize_value(value, tensors: tuple[torch.Tensor, ...]):
    if type(value) is _TensorReference:
        return tensors[value.index]
    if type(value) is _ListValue:
        return [_materialize_value(item, tensors) for item in value.values]
    if type(value) is _TupleValue:
        return tuple(_materialize_value(item, tensors) for item in value.values)
    if type(value) is _DictionaryValue:
        return {key: _materialize_value(item, tensors) for key, item in value.values}
    return value


def _reconstruct_canonical_wire_value(
    prepared: _PreparedCanonicalWireValue,
    payload_tensors,
    *,
    expected_fragment_digest: bytes | None = None,
):
    """Verify separate tensor payloads and reconstruct one owned canonical value."""

    if type(prepared) is not _PreparedCanonicalWireValue:
        raise TypeError("prepared must be _PreparedCanonicalWireValue")
    if expected_fragment_digest is not None:
        _validate_digest_field("expected_fragment_digest", expected_fragment_digest)
        if not hmac.compare_digest(prepared.fragment_digest, expected_fragment_digest):
            raise ValueError("canonical wire fragment digest does not match")
    if type(payload_tensors) not in {tuple, list}:
        raise TypeError("payload_tensors must be a tuple or list")
    if len(payload_tensors) != len(prepared.tensor_specs):
        raise ValueError("canonical wire payload count does not match tensor descriptors")
    payloads = tuple(payload_tensors)
    for index, (payload, spec) in enumerate(zip(payloads, prepared.tensor_specs)):
        if not torch.is_tensor(payload) or not _tight_cpu_tensor(payload, spec):
            raise ValueError("canonical wire tensor payload {} does not match its descriptor".format(index))
    cloned = []
    for index, (payload, spec) in enumerate(zip(payloads, prepared.tensor_specs)):
        owned = _clone_tensor(payload, chunk_bytes=prepared._limits.chunk_bytes, path="payload_tensors[{}]".format(index))
        digest = _canonical_tensor_digest(
            owned,
            dtype_code=spec.dtype_code,
            shape=spec.shape,
            nbytes=spec.nbytes,
            chunk_bytes=prepared._limits.chunk_bytes,
        )
        if not hmac.compare_digest(digest, spec.digest):
            raise ValueError("canonical wire tensor payload {} has an invalid digest".format(index))
        cloned.append(owned)
    return _materialize_value(prepared._value, tuple(cloned))

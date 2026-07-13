"""Strict source-topology-neutral envelope for portable optimizer state."""

import hashlib
import hmac
import math
import struct
import sys

import torch

from gefen.contracts import ParameterIdentity


PORTABLE_STATE_FORMAT = "gefen.portable_state"
PORTABLE_STATE_FORMAT_VERSION = 3
PORTABLE_STATE_COVERAGE = "global_logical_optimizer"
PORTABLE_STATE_DIGEST_ALGORITHM = "sha256"
_PORTABLE_DIGEST_CHUNK_BYTES = 1 << 23
_PORTABLE_CLONE_CHUNK_BYTES = 1 << 23
_PORTABLE_NATIVE_BYTEORDER = sys.byteorder

_PORTABLE_STATE_TOP_LEVEL_KEYS = frozenset(
    {
        "format",
        "format_version",
        "coverage",
        "implementation",
        "policy",
        "common",
        "parameters",
        "provenance",
        "completion",
    }
)
_PORTABLE_PARAMETER_KEYS = frozenset(
    {
        "identity",
        "algorithm_options",
        "state_variant",
        "state",
        "projection_hints",
    }
)
_PORTABLE_IDENTITY_KEYS = frozenset(
    {"schema_version", "fqn", "global_shape"}
)
_PORTABLE_COMPLETION_KEYS = frozenset(
    {"status", "digest_algorithm", "digest"}
)


def _portable_tensor_chunk_elements(value: torch.Tensor) -> int:
    budget = _PORTABLE_CLONE_CHUNK_BYTES
    if type(budget) is not int or budget <= 0:
        raise RuntimeError("portable clone chunk budget must be a positive int")
    scratch_bytes_per_element = value.element_size()
    if not value.is_contiguous():
        # The non-contiguous read path (_read_portable_tensor_chunk) materializes
        # one int64 coordinate vector per dimension plus the int64 linear index
        # and its running quotient. Size the chunk for that scratch too, so a
        # high-rank strided tensor stays within the fixed clone budget instead of
        # oversubscribing it. Mirrors portable_wire._clone_tensor.
        scratch_bytes_per_element += 8 * (value.ndim + 2)
    return max(1, budget // scratch_bytes_per_element)


def _read_portable_tensor_chunk(
    value: torch.Tensor, start: int, stop: int
) -> torch.Tensor:
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


def _clone_portable_value(value, *, path: str):
    if torch.is_tensor(value):
        if (
            type(value) is not torch.Tensor
            or value.layout is not torch.strided
            or value.is_meta
            or value.is_nested
            or value.is_quantized
        ):
            raise TypeError(
                "{} must be a plain materialized strided tensor for portable state".format(
                    path
                )
            )
        cloned = torch.empty(tuple(value.shape), dtype=value.dtype, device="cpu")
        cloned_flat = cloned.reshape(-1)
        chunk_elements = _portable_tensor_chunk_elements(value)
        for start in range(0, value.numel(), chunk_elements):
            stop = min(start + chunk_elements, value.numel())
            source = (
                _read_portable_tensor_chunk(value, start, stop)
                .resolve_conj()
                .resolve_neg()
                .reshape(-1)
            )
            destination = cloned_flat[start:stop]
            destination.copy_(source)
            if cloned.is_floating_point() or cloned.is_complex():
                try:
                    finite = bool(torch.isfinite(destination).all())
                except (NotImplementedError, RuntimeError, TypeError) as exc:
                    raise ValueError(
                        "{} tensor dtype does not support finite portable state".format(
                            path
                        )
                    ) from exc
                if not finite:
                    raise ValueError("{} tensor must be finite".format(path))
        return cloned
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("{} must be finite".format(path))
        return value
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is list:
        return [
            _clone_portable_value(item, path="{}[{}]".format(path, index))
            for index, item in enumerate(value)
        ]
    if type(value) is tuple:
        return tuple(
            _clone_portable_value(item, path="{}[{}]".format(path, index))
            for index, item in enumerate(value)
        )
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError("{} dictionary keys must be strings".format(path))
        return {
            key: _clone_portable_value(
                value[key], path="{}.{}".format(path, key)
            )
            for key in sorted(value)
        }
    raise TypeError(
        "{} has unsupported portable-state type {}".format(
            path, type(value).__name__
        )
    )


def _digest_bytes(hasher, value: bytes) -> None:
    hasher.update(struct.pack(">Q", len(value)))
    hasher.update(value)


def _update_portable_digest(hasher, value) -> None:
    value_type = type(value)
    if value is None:
        hasher.update(b"N")
        return
    if value_type is bool:
        hasher.update(b"B1" if value else b"B0")
        return
    if value_type is int:
        hasher.update(b"I")
        _digest_bytes(hasher, str(value).encode("ascii"))
        return
    if value_type is float:
        hasher.update(b"F")
        hasher.update(struct.pack(">d", value))
        return
    if value_type is str:
        hasher.update(b"S")
        _digest_bytes(hasher, value.encode("utf-8"))
        return
    if value_type is list:
        hasher.update(b"L")
        hasher.update(struct.pack(">Q", len(value)))
        for item in value:
            _update_portable_digest(hasher, item)
        return
    if value_type is tuple:
        hasher.update(b"T")
        hasher.update(struct.pack(">Q", len(value)))
        for item in value:
            _update_portable_digest(hasher, item)
        return
    if value_type is dict:
        hasher.update(b"D")
        hasher.update(struct.pack(">Q", len(value)))
        for key in sorted(value):
            if type(key) is not str:
                raise TypeError("portable state dictionary keys must be strings")
            _update_portable_digest(hasher, key)
            _update_portable_digest(hasher, value[key])
        return
    if value_type is torch.Tensor:
        if value.device.type != "cpu" or not value.is_contiguous():
            raise ValueError("portable digest tensors must be contiguous CPU tensors")
        hasher.update(b"R")
        _digest_bytes(hasher, str(value.dtype).encode("ascii"))
        _update_portable_digest(hasher, list(value.shape))
        element_size = value.element_size()
        hasher.update(struct.pack(">Q", value.numel() * element_size))
        elements_per_chunk = max(
            1, _PORTABLE_DIGEST_CHUNK_BYTES // element_size
        )
        flat = value.reshape(-1)
        for start in range(0, value.numel(), elements_per_chunk):
            stop = min(start + elements_per_chunk, value.numel())
            raw = flat[start:stop].view(torch.uint8).reshape(-1)
            component_size = element_size // 2 if value.is_complex() else element_size
            if _PORTABLE_NATIVE_BYTEORDER not in {"little", "big"}:
                raise RuntimeError("unsupported native byte order")
            if _PORTABLE_NATIVE_BYTEORDER == "big" and component_size > 1:
                raw = (
                    raw.reshape(-1, component_size)
                    .flip(1)
                    .contiguous()
                    .reshape(-1)
                )
            hasher.update(memoryview(raw.numpy()))
        return
    raise TypeError(
        "portable state digest does not support {}".format(value_type.__name__)
    )


def _canonical_portable_state_digest(canonical) -> str:
    hasher = hashlib.sha256()
    _update_portable_digest(hasher, canonical)
    return hasher.hexdigest()


def portable_state_digest(payload) -> str:
    """Return the deterministic SHA-256 digest of one canonical wire value."""

    canonical = _clone_portable_value(payload, path="portable digest payload")
    return _canonical_portable_state_digest(canonical)


def _normalize_portable_parameter(fqn, record):
    if type(record) is not dict or set(record) != _PORTABLE_PARAMETER_KEYS:
        raise ValueError(
            "portable parameter {!r} has an invalid schema".format(fqn)
        )
    identity_record = record["identity"]
    if (
        type(identity_record) is not dict
        or set(identity_record) != _PORTABLE_IDENTITY_KEYS
        or type(identity_record["global_shape"]) is not list
    ):
        raise ValueError(
            "portable parameter {!r} has an invalid identity".format(fqn)
        )
    try:
        identity = ParameterIdentity(
            identity_record["fqn"],
            tuple(identity_record["global_shape"]),
            schema_version=identity_record["schema_version"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "portable parameter {!r} has an invalid identity".format(fqn)
        ) from exc
    if identity.fqn != fqn:
        raise ValueError("portable parameter identity does not match its FQN key")
    for key in ("algorithm_options", "state", "projection_hints"):
        if type(record[key]) is not dict:
            raise ValueError(
                "portable parameter {!r} {} must be a dictionary".format(
                    fqn, key
                )
            )
    if type(record["state_variant"]) is not str or not record["state_variant"]:
        raise ValueError(
            "portable parameter {!r} state_variant must be a non-empty string".format(
                fqn
            )
        )
    return record


def _portable_payload(document):
    return {
        key: document[key]
        for key in sorted(_PORTABLE_STATE_TOP_LEVEL_KEYS - {"completion"})
    }


def _validate_portable_payload(document) -> None:
    if document["format"] != PORTABLE_STATE_FORMAT:
        raise ValueError("unsupported portable state format")
    if (
        type(document["format_version"]) is not int
        or document["format_version"] != PORTABLE_STATE_FORMAT_VERSION
    ):
        raise ValueError(
            "unsupported portable state format_version: {}".format(
                document["format_version"]
            )
        )
    if document["coverage"] != PORTABLE_STATE_COVERAGE:
        raise ValueError("unsupported portable state coverage")
    implementation = document["implementation"]
    if type(implementation) is not str or not implementation:
        raise ValueError("portable state implementation must be a non-empty string")
    if type(document["policy"]) is not dict:
        raise ValueError("portable state policy must be a dictionary")
    if type(document["common"]) is not dict:
        raise ValueError("portable common state must be a dictionary")
    if document["provenance"] is not None and type(document["provenance"]) is not dict:
        raise ValueError("portable state provenance must be a dictionary or None")
    parameters = document["parameters"]
    if type(parameters) is not dict:
        raise ValueError("portable parameters must be an FQN mapping")
    for fqn, record in parameters.items():
        if type(fqn) is not str:
            raise ValueError("portable parameter FQN keys must be strings")
        _normalize_portable_parameter(fqn, record)


def _validate_portable_completion(document) -> None:
    completion = document["completion"]
    if type(completion) is not dict or set(completion) != _PORTABLE_COMPLETION_KEYS:
        raise ValueError("portable state completion marker has an invalid schema")
    if completion["status"] != "complete":
        raise ValueError("portable state is not marked complete")
    if completion["digest_algorithm"] != PORTABLE_STATE_DIGEST_ALGORITHM:
        raise ValueError("unsupported portable state digest algorithm")
    digest = completion["digest"]
    if (
        type(digest) is not str
        or len(digest) != hashlib.sha256().digest_size * 2
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("portable state completion digest is invalid")
    expected_digest = _canonical_portable_state_digest(_portable_payload(document))
    if not hmac.compare_digest(digest, expected_digest):
        raise ValueError("portable state completion digest does not match its payload")


def normalize_portable_state_document(state, *, expected_implementation=None):
    """Clone and validate a complete portable v3 state document."""

    document = _clone_portable_value(state, path="portable state")
    if type(document) is not dict or set(document) != _PORTABLE_STATE_TOP_LEVEL_KEYS:
        raise ValueError("portable state has an invalid top-level schema")
    _validate_portable_payload(document)
    _validate_portable_completion(document)
    if (
        expected_implementation is not None
        and document["implementation"] != expected_implementation
    ):
        raise ValueError("portable state implementation does not match the target")
    return document


def build_portable_state_document(
    *,
    implementation,
    policy,
    common,
    parameters,
    provenance=None,
):
    """Build and validate a complete source-topology-neutral v3 document."""

    document = {
        "format": PORTABLE_STATE_FORMAT,
        "format_version": PORTABLE_STATE_FORMAT_VERSION,
        "coverage": PORTABLE_STATE_COVERAGE,
        "implementation": implementation,
        "policy": policy,
        "common": common,
        "parameters": parameters,
        "provenance": provenance,
    }
    canonical = _clone_portable_value(document, path="portable state")
    if type(canonical) is not dict or set(canonical) != (
        _PORTABLE_STATE_TOP_LEVEL_KEYS - {"completion"}
    ):
        raise ValueError("portable state has an invalid top-level schema")
    _validate_portable_payload(canonical)
    canonical["completion"] = {
        "status": "complete",
        "digest_algorithm": PORTABLE_STATE_DIGEST_ALGORITHM,
        "digest": _canonical_portable_state_digest(canonical),
    }
    return canonical


__all__ = [
    "PORTABLE_STATE_COVERAGE",
    "PORTABLE_STATE_DIGEST_ALGORITHM",
    "PORTABLE_STATE_FORMAT",
    "PORTABLE_STATE_FORMAT_VERSION",
    "build_portable_state_document",
    "normalize_portable_state_document",
    "portable_state_digest",
]

"""Synchronous PyTorch DCP persistence for collective portable Gefen state."""

from __future__ import annotations

import math
import os

import torch

from gefen.checkpoint import CheckpointProcessGroupBinding
from gefen.portable_state import PortableStateLimits
from gefen.portable_wire import (
    _DTYPE_BY_VALUE,
    _CanonicalWireLimits,
    _CanonicalWirePlan,
    _parse_canonical_wire_metadata,
    _reconstruct_canonical_wire_value,
)


_DCP_METADATA_KEY = "__gefen_portable_metadata_v1__"
_DCP_PAYLOAD_PREFIX = "__gefen_portable_tensor_"
_DCP_PAYLOAD_SUFFIX = "__"
_DCP_PAYLOAD_DIGITS = 16
_DCP_SAVE_PREFLIGHT_TRANSACTION = "gefen-portable-dcp-save-preflight-v1"
_DCP_LOAD_PREFLIGHT_TRANSACTION = "gefen-portable-dcp-load-preflight-v1"


def _prepare_canonical_wire_value(value, limits):
    from gefen.portable_wire import _prepare_canonical_wire_value as prepare

    return prepare(value, limits)


def _require_namespace(namespace, limits: PortableStateLimits) -> str:
    if type(namespace) is not str or not namespace or namespace != namespace.strip():
        raise ValueError("namespace must be a non-empty trimmed string")
    if "." in namespace or "\x00" in namespace:
        raise ValueError("namespace must not contain dots or NUL")
    if any(not (character.isascii() and (character.isalnum() or character in {"_", "-"})) for character in namespace):
        raise ValueError("namespace must contain only ASCII letters, digits, underscores, and hyphens")
    from gefen.portable_runtime import _bounded_utf8_length

    _bounded_utf8_length(
        namespace,
        limit=limits.max_string_bytes,
        name="DCP namespace",
    )
    return namespace


def _payload_key(index: int) -> str:
    if type(index) is not int or index < 0 or index >= 10**_DCP_PAYLOAD_DIGITS:
        raise ValueError("portable DCP payload index is out of range")
    return "{}{:0{}d}{}".format(
        _DCP_PAYLOAD_PREFIX,
        index,
        _DCP_PAYLOAD_DIGITS,
        _DCP_PAYLOAD_SUFFIX,
    )


def _flat_key(namespace: str, inner_key: str) -> str:
    return "{}.{}".format(namespace, inner_key)


def _validate_dcp_runtime(binding: CheckpointProcessGroupBinding) -> None:
    import torch.distributed as dist

    members = binding.identity.ordered_members
    if len(members) == 1:
        if not dist.is_available() or not dist.is_initialized():
            return
        if dist.get_world_size() > 1:
            raise RuntimeError("a singleton portable DCP binding cannot run inside a larger initialized default world")
        backend = str(dist.get_backend()).lower()
    else:
        if dist.get_global_rank(binding.process_group, 0) != 0:
            raise RuntimeError("portable DCP requires checkpoint group coordinate zero to be global rank zero")
        backend = str(dist.get_backend(binding.process_group)).lower()
    if "nccl" in backend:
        if binding.collective_device.type != "cuda":
            raise ValueError("portable DCP requires a CUDA checkpoint device with NCCL")
        expected = binding.collective_device.index
        if expected is None or torch.cuda.current_device() != expected:
            raise ValueError("portable DCP requires the current CUDA device to match the checkpoint binding")
    elif "gloo" in backend or "mpi" in backend:
        if binding.collective_device.type != "cpu":
            raise ValueError("portable DCP requires a CPU checkpoint device with this backend")


def _storage_identity(storage, limits: PortableStateLimits):
    storage_class = type(storage)
    module = storage_class.__module__
    qualname = storage_class.__qualname__
    if type(module) is not str or type(qualname) is not str:
        raise TypeError("DCP storage classes require exact string identities")
    checkpoint_id = getattr(storage, "checkpoint_id", None)
    try:
        checkpoint_id = os.fspath(checkpoint_id)
    except TypeError as exc:
        raise TypeError("DCP storage must expose a string or path-like checkpoint_id") from exc
    if type(checkpoint_id) is not str or not checkpoint_id or "\x00" in checkpoint_id:
        raise ValueError("DCP storage checkpoint_id must be a non-empty string without NUL")
    from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter

    if isinstance(storage, (FileSystemReader, FileSystemWriter)):
        checkpoint_id = os.path.abspath(checkpoint_id)
    from gefen.portable_runtime import _bounded_utf8_length

    for name, value in (
        ("DCP storage class module", module),
        ("DCP storage class qualname", qualname),
        ("DCP storage checkpoint_id", checkpoint_id),
    ):
        _bounded_utf8_length(
            value,
            limit=limits.max_string_bytes,
            name=name,
        )
    return {
        "module": module,
        "qualname": qualname,
        "checkpoint_id": checkpoint_id,
    }


def _preflight_dcp_operation(
    optimizer,
    *,
    checkpoint_process_group,
    transaction_id,
    limits,
    namespace,
    storage,
    storage_type,
    operation: str,
):
    from gefen import portable_runtime as runtime
    from gefen.portable_collective import _collective_unanimous_status
    from gefen.hybrid import GefenMuonHybrid

    if type(optimizer) is GefenMuonHybrid:
        from gefen.portable_hybrid import _hybrid_transport_binding

        transport = _hybrid_transport_binding(
            optimizer,
            checkpoint_process_group,
        )
    else:
        transport = runtime._preflight_transport_binding(
            optimizer,
            checkpoint_process_group,
        )
    binding = None
    normalized_limits = None
    normalized_transaction = None
    normalized_namespace = None
    implementation = None
    context_digest = bytes(32)
    wire_limits = runtime._STATUS_FALLBACK_LIMITS
    error = None
    try:
        if type(optimizer) is GefenMuonHybrid:
            from gefen.portable_hybrid import (
                HYBRID_PORTABLE_STATE_IMPLEMENTATION,
                _preflight_hybrid_portable_local,
            )

            (
                binding,
                normalized_transaction,
                normalized_limits,
                _children,
                _routing,
                base_context,
                _base_digest,
                _live_token,
            ) = _preflight_hybrid_portable_local(
                optimizer,
                checkpoint_process_group=checkpoint_process_group,
                transaction_id=transaction_id,
                limits=limits,
            )
            implementation = HYBRID_PORTABLE_STATE_IMPLEMENTATION
        else:
            runtime._validate_supplied_binding(checkpoint_process_group, transport)
            binding = transport
            normalized_limits = runtime._require_limits(limits)
            normalized_transaction = runtime._require_transaction_id(transaction_id)
            implementation = runtime._optimizer_implementation(optimizer)
            runtime._validate_context_identity_bounds(binding, normalized_limits)
            prepared = runtime._prepare_local_structure(
                optimizer,
                implementation,
                binding,
                normalized_limits,
                include_payload=False,
            )
            runtime._validate_prepared_local_state(
                optimizer,
                implementation,
                prepared,
            )
            base_context = runtime._base_context(binding, implementation)
        wire_limits = normalized_limits._wire_limits()
        normalized_namespace = _require_namespace(namespace, normalized_limits)
        if not isinstance(storage, storage_type):
            raise TypeError("storage must be a {}".format(storage_type.__name__))
        storage_identity = _storage_identity(storage, normalized_limits)
        _validate_dcp_runtime(binding)
        context = {
            **base_context,
            "dcp_envelope_version": 2,
            "dcp_namespace": normalized_namespace,
            "dcp_storage": storage_identity,
            "transaction_id": normalized_transaction,
        }
        runtime._preflight_portable_value(context, normalized_limits)
        context_digest = runtime._context_digest(context)
    except Exception as exc:
        error = exc
    _collective_unanimous_status(
        transport,
        error,
        operation="portable_dcp_{}_preflight".format(operation),
        transaction_id=(_DCP_SAVE_PREFLIGHT_TRANSACTION if operation == "save" else _DCP_LOAD_PREFLIGHT_TRANSACTION),
        context_digest=context_digest,
        limits=wire_limits,
    )
    assert (
        binding is not None
        and normalized_limits is not None
        and normalized_transaction is not None
        and normalized_namespace is not None
        and implementation is not None
    )
    return (
        binding,
        normalized_transaction,
        normalized_limits,
        normalized_namespace,
    )


def _metadata_tensor(metadata: bytes) -> torch.Tensor:
    if type(metadata) is not bytes:
        raise TypeError("portable DCP metadata must be bytes")
    return torch.frombuffer(bytearray(metadata), dtype=torch.uint8)


def _state_from_plan(
    namespace: str,
    plan: _CanonicalWirePlan,
    limits: _CanonicalWireLimits,
):
    if type(plan) is not _CanonicalWirePlan:
        raise TypeError("plan must be a canonical wire plan")
    if type(limits) is not _CanonicalWireLimits:
        raise TypeError("limits must be canonical wire limits")
    if len(plan.payload_tensors) + 1 > limits.max_container_items:
        raise ValueError("portable DCP envelope exceeds max_container_items")
    envelope = {_DCP_METADATA_KEY: _metadata_tensor(plan.metadata)}
    envelope.update({_payload_key(index): tensor for index, tensor in enumerate(plan.payload_tensors)})
    return {namespace: envelope}


def _validate_tensor_metadata(value, *, name: str, limits: _CanonicalWireLimits):
    from torch.distributed.checkpoint.metadata import (
        ChunkStorageMetadata,
        TensorProperties,
        TensorStorageMetadata,
    )

    if type(value) is not TensorStorageMetadata:
        raise TypeError("{} must be full-tensor DCP metadata".format(name))
    properties = value.properties
    if (
        type(properties) is not TensorProperties
        or type(value.size) is not torch.Size
        or properties.dtype not in _DTYPE_BY_VALUE
        or properties.layout is not torch.strided
        or properties.requires_grad is not False
        or properties.memory_format is not torch.contiguous_format
        or properties.pin_memory is not False
    ):
        raise TypeError("{} has unsupported tensor properties".format(name))
    shape = tuple(value.size)
    if len(shape) > limits.max_tensor_rank or any(
        type(dimension) is not int or dimension < 0 or dimension > (1 << 63) - 1 for dimension in shape
    ):
        raise ValueError("{} has invalid tensor geometry".format(name))
    if type(value.chunks) is not list or len(value.chunks) != 1:
        raise ValueError("{} must contain exactly one full DCP chunk".format(name))
    chunk = value.chunks[0]
    if (
        type(chunk) is not ChunkStorageMetadata
        or type(chunk.offsets) is not torch.Size
        or type(chunk.sizes) is not torch.Size
        or tuple(chunk.offsets) != (0,) * len(shape)
        or tuple(chunk.sizes) != shape
    ):
        raise ValueError("{} must contain one complete unsharded tensor".format(name))
    numel = math.prod(shape)
    element_size = _DTYPE_BY_VALUE[properties.dtype][1]
    nbytes = numel * element_size
    return properties.dtype, shape, nbytes


def _allocate_dcp_state(
    metadata,
    *,
    namespace: str,
    limits: _CanonicalWireLimits,
):
    from torch.distributed.checkpoint.metadata import Metadata

    if type(metadata) is not Metadata:
        raise TypeError("portable DCP requires exact checkpoint Metadata")
    entries = metadata.state_dict_metadata
    if type(entries) is not dict:
        raise TypeError("portable DCP metadata entries must be a dict")
    if len(entries) < 1 or len(entries) > limits.max_tensors + 1 or len(entries) > limits.max_container_items:
        raise ValueError("portable DCP tensor count exceeds limits")
    metadata_key = _flat_key(namespace, _DCP_METADATA_KEY)
    if metadata_key not in entries:
        raise ValueError("portable DCP metadata tensor is missing")
    payload_count = len(entries) - 1
    expected_payload_keys = tuple(_flat_key(namespace, _payload_key(index)) for index in range(payload_count))
    expected_keys = {metadata_key, *expected_payload_keys}
    if set(entries) != expected_keys or any(type(key) is not str for key in entries):
        raise ValueError("portable DCP checkpoint has unexpected tensor keys")
    planner_data = metadata.planner_data
    expected_planner_data = {key: tuple(key.split(".", 1)) for key in expected_keys}
    if (
        type(planner_data) is not dict
        or planner_data != expected_planner_data
        or any(
            type(key) is not str
            or type(path) is not tuple
            or len(path) != 2
            or any(type(component) is not str for component in path)
            for key, path in planner_data.items()
        )
    ):
        raise ValueError("portable DCP checkpoint has incompatible planner paths")

    metadata_dtype, metadata_shape, metadata_nbytes = _validate_tensor_metadata(
        entries[metadata_key],
        name="portable DCP wire metadata",
        limits=limits,
    )
    if (
        metadata_dtype is not torch.uint8
        or len(metadata_shape) != 1
        or metadata_nbytes == 0
        or metadata_nbytes > limits.max_metadata_bytes
    ):
        raise ValueError("portable DCP wire metadata exceeds its byte limit")

    payload_specs = []
    total_payload_bytes = 0
    for index, key in enumerate(expected_payload_keys):
        dtype, shape, nbytes = _validate_tensor_metadata(
            entries[key],
            name="portable DCP payload {}".format(index),
            limits=limits,
        )
        if total_payload_bytes > limits.max_fragment_tensor_bytes - nbytes:
            raise ValueError("portable DCP payload tensors exceed their byte limit")
        total_payload_bytes += nbytes
        payload_specs.append((_payload_key(index), dtype, shape))

    envelope = {
        _DCP_METADATA_KEY: torch.empty(
            metadata_shape,
            dtype=metadata_dtype,
            device="cpu",
        )
    }
    for key, dtype, shape in payload_specs:
        envelope[key] = torch.empty(shape, dtype=dtype, device="cpu")
    return {namespace: envelope}


def _load_planner(namespace: str, limits: _CanonicalWireLimits):
    import torch.distributed.checkpoint as dcp

    class _PortableDCPDynamicLoadPlanner(dcp.DefaultLoadPlanner):
        def set_up_planner(
            self,
            state_dict,
            metadata=None,
            is_coordinator=False,
        ) -> None:
            if (
                type(state_dict) is not dict
                or set(state_dict) != {namespace}
                or type(state_dict[namespace]) is not dict
                or state_dict[namespace]
            ):
                raise ValueError("portable DCP load planner requires one empty exact namespace")
            allocated = _allocate_dcp_state(
                metadata,
                namespace=namespace,
                limits=limits,
            )
            dict.update(state_dict[namespace], allocated[namespace])
            super().set_up_planner(state_dict, metadata, is_coordinator)

    return _PortableDCPDynamicLoadPlanner()


def _decode_dcp_state(
    state,
    *,
    namespace: str,
    limits: _CanonicalWireLimits,
):
    if type(state) is not dict or set(state) != {namespace}:
        raise ValueError("portable DCP root state has an invalid schema")
    envelope = state[namespace]
    if type(envelope) is not dict or _DCP_METADATA_KEY not in envelope:
        raise ValueError("portable DCP envelope has an invalid schema")
    metadata_tensor = envelope[_DCP_METADATA_KEY]
    if (
        type(metadata_tensor) is not torch.Tensor
        or metadata_tensor.dtype is not torch.uint8
        or metadata_tensor.device.type != "cpu"
        or metadata_tensor.ndim != 1
        or not metadata_tensor.is_contiguous()
        or metadata_tensor.numel() > limits.max_metadata_bytes
    ):
        raise ValueError("portable DCP wire metadata tensor is invalid")
    metadata = bytes(memoryview(metadata_tensor.numpy()))
    prepared = _parse_canonical_wire_metadata(metadata, limits=limits)
    expected_keys = {
        _DCP_METADATA_KEY,
        *(_payload_key(index) for index in range(len(prepared.tensor_specs))),
    }
    if set(envelope) != expected_keys:
        raise ValueError("portable DCP payload key count disagrees with wire metadata")
    payloads = tuple(envelope[_payload_key(index)] for index in range(len(prepared.tensor_specs)))
    document = _reconstruct_canonical_wire_value(prepared, payloads)
    return document, prepared.fragment_digest


def save_portable_dcp(
    optimizer,
    *,
    checkpoint_process_group,
    storage_writer,
    transaction_id,
    limits,
    namespace="optimizer",
):
    """Collectively save one tensor-only portable optimizer document through DCP."""

    from torch.distributed.checkpoint.storage import StorageWriter

    binding, transaction_id, limits, namespace = _preflight_dcp_operation(
        optimizer,
        checkpoint_process_group=checkpoint_process_group,
        transaction_id=transaction_id,
        limits=limits,
        namespace=namespace,
        storage=storage_writer,
        storage_type=StorageWriter,
        operation="save",
    )
    from gefen.hybrid import GefenMuonHybrid

    if type(optimizer) is GefenMuonHybrid:
        from gefen.portable_dcp_hybrid_sharded import (
            _save_sharded_hybrid_dcp,
        )

        return _save_sharded_hybrid_dcp(
            optimizer,
            binding=binding,
            storage_writer=storage_writer,
            transaction_id=transaction_id,
            limits=limits,
            namespace=namespace,
        )
    else:
        from gefen.portable_dcp_sharded import _save_sharded_portable_dcp

        return _save_sharded_portable_dcp(
            optimizer,
            binding=binding,
            storage_writer=storage_writer,
            transaction_id=transaction_id,
            limits=limits,
            namespace=namespace,
        )
    raise AssertionError("unreachable portable DCP optimizer dispatch")


def load_portable_dcp(
    optimizer,
    *,
    checkpoint_process_group,
    storage_reader,
    transaction_id,
    limits,
    namespace="optimizer",
) -> None:
    """Collectively load, verify, and atomically import portable optimizer state."""

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.storage import StorageReader

    binding, transaction_id, limits, namespace = _preflight_dcp_operation(
        optimizer,
        checkpoint_process_group=checkpoint_process_group,
        transaction_id=transaction_id,
        limits=limits,
        namespace=namespace,
        storage=storage_reader,
        storage_type=StorageReader,
        operation="load",
    )
    from gefen.hybrid import GefenMuonHybrid

    modern = False
    checkpoint_metadata = None
    error = None
    try:
        checkpoint_metadata = storage_reader.read_metadata()
        if type(optimizer) is GefenMuonHybrid:
            from gefen.portable_dcp_hybrid_sharded import (
                _metadata_storage_spec,
            )
        else:
            from gefen.portable_dcp_sharded import _metadata_storage_spec

        modern = (
            _metadata_storage_spec(
                checkpoint_metadata,
                namespace=namespace,
                limits=limits,
            )
            is not None
        )
    except Exception as exc:
        error = exc
    from gefen.portable_collective import _collective_unanimous_status

    _collective_unanimous_status(
        binding,
        error,
        operation="portable_dcp_load_envelope",
        transaction_id=transaction_id,
        context_digest=bytes(31) + bytes((2 if modern else 1,)),
        limits=limits._wire_limits(collective=True),
    )
    assert checkpoint_metadata is not None
    if modern:
        if type(optimizer) is GefenMuonHybrid:
            from gefen.portable_dcp_hybrid_sharded import (
                _load_sharded_hybrid_dcp,
            )

            return _load_sharded_hybrid_dcp(
                optimizer,
                binding=binding,
                storage_reader=storage_reader,
                checkpoint_metadata=checkpoint_metadata,
                transaction_id=transaction_id,
                limits=limits,
                namespace=namespace,
            )
        else:
            from gefen.portable_dcp_sharded import _load_sharded_portable_dcp

            return _load_sharded_portable_dcp(
                optimizer,
                binding=binding,
                storage_reader=storage_reader,
                checkpoint_metadata=checkpoint_metadata,
                transaction_id=transaction_id,
                limits=limits,
                namespace=namespace,
            )
    wire_limits = limits._wire_limits(collective=True)
    state = {namespace: {}}
    dcp.load(
        state,
        storage_reader=storage_reader,
        planner=_load_planner(namespace, wire_limits),
        process_group=binding.process_group,
    )
    document = None
    digest = bytes(32)
    error = None
    try:
        document, digest = _decode_dcp_state(
            state,
            namespace=namespace,
            limits=wire_limits,
        )
    except Exception as exc:
        error = exc
    from gefen.portable_collective import _collective_unanimous_status

    _collective_unanimous_status(
        binding,
        error,
        operation="portable_dcp_load_decode",
        transaction_id=transaction_id,
        context_digest=digest,
        limits=wire_limits,
    )
    assert document is not None
    optimizer.import_portable_state(
        document,
        checkpoint_process_group=binding,
        transaction_id=transaction_id,
        limits=limits,
    )


__all__ = ["load_portable_dcp", "save_portable_dcp"]

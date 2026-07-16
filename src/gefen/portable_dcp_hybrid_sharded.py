"""Scalable sharded-DCP envelope for Gefen-backed Hybrid optimizers."""

from __future__ import annotations

import hashlib

import torch

from gefen.checkpoint import CheckpointProcessGroupBinding


_COMPOSITE_FORMAT = "gefen.portable_dcp_sharded_composite"
_COMPOSITE_FORMAT_VERSION = 2
_COMPOSITE_COVERAGE = "global_logical_composite_optimizer"
_COMPOSITE_IMPLEMENTATION = "gefen.GefenMuonHybrid"
_COMPOSITE_METADATA_KEY = "__gefen_portable_composite_metadata_v2__"
_ROLES = ("muon", "backup")
_CHILD_IMPLEMENTATIONS = {
    "muon": "gefen.GefenMuon",
    "backup": "gefen.Gefen",
}


def _child_namespace(namespace: str, role: str, limits) -> str:
    if role not in _ROLES:
        raise ValueError("invalid sharded Hybrid child role")
    from gefen.portable_dcp import _require_namespace

    return _require_namespace(
        "{}-gefen-{}".format(namespace, role),
        limits,
    )


def _encode_metadata(value, *, limits) -> torch.Tensor:
    from gefen.portable_wire import _prepare_canonical_wire_value

    plan = _prepare_canonical_wire_value(
        value,
        limits._wire_limits(collective=True),
    )
    if plan.payload_tensors:
        raise RuntimeError("sharded Hybrid metadata unexpectedly contains tensors")
    return torch.frombuffer(bytearray(plan.metadata), dtype=torch.uint8)


def _completed_metadata(value, *, limits):
    from gefen.portable_schema import portable_state_digest

    semantic = dict(value)
    completed = {
        **semantic,
        "completion": {
            "metadata_digest": portable_state_digest(semantic),
        },
    }
    return completed, _encode_metadata(completed, limits=limits)


def _validate_aggregate_child_limits(
    child_metadata,
    *,
    binding: CheckpointProcessGroupBinding,
    limits,
):
    from gefen.portable_dcp_sharded import _chunk_segment

    fields = [
        field
        for metadata in child_metadata.values()
        for field in metadata["fields"]
    ]
    if (
        len(fields) > limits.max_tensors
        or len(fields) > limits.max_container_items
    ):
        raise ValueError("sharded Hybrid fields exceed count limits")
    coordinate = binding.identity.ordered_members.index(binding.local_member)
    parts = len(binding.identity.ordered_members)
    expected_chunks = 1 + len(child_metadata) + len(fields) * parts
    if (
        expected_chunks > limits.max_container_items
        or expected_chunks > limits.max_tree_nodes
    ):
        raise ValueError("sharded Hybrid DCP chunks exceed aggregate limits")
    global_bytes = 0
    local_bytes = 0
    for field in fields:
        field_bytes = field["numel"] * 4
        if global_bytes > limits.max_collective_tensor_bytes - field_bytes:
            raise ValueError("sharded Hybrid fields exceed max_collective_tensor_bytes")
        global_bytes += field_bytes
        chunk_bytes = _chunk_segment(
            field["numel"],
            parts,
            coordinate,
        ).length * 4
        if local_bytes > limits.max_fragment_tensor_bytes - chunk_bytes:
            raise ValueError("sharded Hybrid local fields exceed max_fragment_tensor_bytes")
        local_bytes += chunk_bytes


def _decode_metadata(value, *, namespace: str, limits):
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
        raise ValueError("sharded Hybrid metadata tensor is invalid")
    prepared = _parse_canonical_wire_metadata(
        bytes(memoryview(value.numpy())),
        limits=limits._wire_limits(collective=True),
    )
    if prepared.tensor_specs:
        raise ValueError("sharded Hybrid metadata must not contain tensor payloads")
    return _normalize_metadata(
        _reconstruct_canonical_wire_value(prepared, ()),
        namespace=namespace,
        limits=limits,
    )


def _normalize_metadata(value, *, namespace: str, limits):
    from gefen.portable_dcp_sharded import _strict_counter, _strict_record
    from gefen.portable_hybrid import _normalize_routing
    from gefen.portable_identity import (
        _normalize_process_group_identity,
        _parse_process_group_identity,
    )
    from gefen.portable_schema import portable_state_digest

    _strict_record(
        value,
        {
            "format",
            "format_version",
            "coverage",
            "implementation",
            "backup_optimizer",
            "source_process_group",
            "routing",
            "common",
            "children",
            "completion",
        },
        name="sharded Hybrid metadata",
    )
    if (
        value["format"] != _COMPOSITE_FORMAT
        or value["format_version"] != _COMPOSITE_FORMAT_VERSION
        or value["coverage"] != _COMPOSITE_COVERAGE
        or value["implementation"] != _COMPOSITE_IMPLEMENTATION
        or value["backup_optimizer"] != "gefen"
    ):
        raise ValueError("unsupported sharded Hybrid metadata identity")
    process_group_record = _normalize_process_group_identity(
        value["source_process_group"]
    )
    process_group = _parse_process_group_identity(process_group_record)
    if len(process_group.ordered_members) > limits.max_members:
        raise ValueError("sharded Hybrid source process group exceeds limits")
    routing = _normalize_routing(value["routing"])
    if len(routing) > limits.max_container_items:
        raise ValueError("sharded Hybrid routing exceeds limits")
    common = _strict_record(
        value["common"],
        {"gefen_global_step", "gefen_deterministic"},
        name="sharded Hybrid common state",
    )
    _strict_counter(common["gefen_global_step"], name="gefen_global_step")
    if type(common["gefen_deterministic"]) is not bool:
        raise ValueError("sharded Hybrid deterministic policy must be bool")
    children = _strict_record(
        value["children"],
        set(_ROLES),
        name="sharded Hybrid children",
    )
    normalized_children = {}
    for role in _ROLES:
        child = children[role]
        if child is None:
            normalized_children[role] = None
            continue
        _strict_record(
            child,
            {"namespace", "implementation", "metadata_digest"},
            name="sharded Hybrid child",
        )
        if (
            child["namespace"] != _child_namespace(namespace, role, limits)
            or child["implementation"] != _CHILD_IMPLEMENTATIONS[role]
            or type(child["metadata_digest"]) is not str
            or len(child["metadata_digest"]) != 64
        ):
            raise ValueError("sharded Hybrid child descriptor is invalid")
        try:
            bytes.fromhex(child["metadata_digest"])
        except ValueError as exc:
            raise ValueError("sharded Hybrid child digest is invalid") from exc
        normalized_children[role] = dict(child)
    if all(child is None for child in normalized_children.values()):
        raise ValueError("sharded Hybrid metadata requires at least one child")
    completion = _strict_record(
        value["completion"],
        {"metadata_digest"},
        name="sharded Hybrid completion",
    )
    digest = completion["metadata_digest"]
    if type(digest) is not str or len(digest) != 64:
        raise ValueError("sharded Hybrid completion digest is invalid")
    try:
        bytes.fromhex(digest)
    except ValueError as exc:
        raise ValueError("sharded Hybrid completion digest is invalid") from exc
    semantic = {key: value[key] for key in value if key != "completion"}
    if portable_state_digest(semantic) != digest:
        raise ValueError("sharded Hybrid metadata digest mismatch")
    return {
        **value,
        "source_process_group": process_group_record,
        "routing": routing,
        "common": dict(common),
        "children": normalized_children,
        "completion": dict(completion),
    }


def _metadata_storage_spec(checkpoint_metadata, *, namespace: str, limits):
    from torch.distributed.checkpoint.metadata import Metadata
    from gefen.portable_dcp import _flat_key, _validate_tensor_metadata

    if type(checkpoint_metadata) is not Metadata:
        raise TypeError("portable DCP requires exact checkpoint Metadata")
    entries = checkpoint_metadata.state_dict_metadata
    if type(entries) is not dict:
        raise TypeError("portable DCP metadata entries must be a dict")
    key = _flat_key(namespace, _COMPOSITE_METADATA_KEY)
    if key not in entries:
        return None
    dtype, shape, nbytes = _validate_tensor_metadata(
        entries[key],
        name="sharded Hybrid DCP metadata",
        limits=limits._wire_limits(collective=True),
    )
    if (
        dtype is not torch.uint8
        or len(shape) != 1
        or nbytes < 1
        or nbytes > limits.max_metadata_bytes
    ):
        raise ValueError("sharded Hybrid DCP metadata exceeds limits")
    return shape


def _read_metadata(
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
            state = {namespace: {_COMPOSITE_METADATA_KEY: tensor}}
            planner = dcp.DefaultLoadPlanner(allow_partial_load=True)
    except Exception as exc:
        error = exc
    shape_bytes = repr(tuple(shape) if shape is not None else None).encode("ascii")
    _collective_unanimous_status(
        binding,
        error,
        operation="portable_dcp_hybrid_metadata_prepare",
        transaction_id=transaction_id,
        context_digest=hashlib.sha256(
            b"gefen.portable_dcp_hybrid.metadata.v1\0"
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
    return _decode_metadata(tensor, namespace=namespace, limits=limits)


def _prepare_save_state(
    optimizer,
    *,
    binding: CheckpointProcessGroupBinding,
    transaction_id: str,
    limits,
    namespace: str,
):
    from gefen.portable_collective import _collective_unanimous_status
    from gefen.portable_dcp_sharded import (
        _DCP_SHARDED_METADATA_KEY,
        _decode_metadata_tensor,
        _prepare_sharded_save_state,
    )
    from gefen.portable_hybrid import (
        _hybrid_child_transaction_id,
        _hybrid_portable_live_token,
        _validate_hybrid_portable_readiness,
    )
    from gefen.portable_identity import _serialize_process_group_identity

    children = None
    routing = None
    live_token = None
    error = None
    try:
        children, routing, _layouts = _validate_hybrid_portable_readiness(
            optimizer,
            binding,
        )
        live_token = _hybrid_portable_live_token(optimizer)
    except Exception as exc:
        error = exc
    _collective_unanimous_status(
        binding,
        error,
        operation="portable_dcp_hybrid_save_local",
        transaction_id=transaction_id,
        context_digest=bytes(32),
        limits=limits._wire_limits(collective=True),
    )
    assert children is not None and routing is not None and live_token is not None

    state = {}
    child_descriptors = {role: None for role in _ROLES}
    child_metadata = {}
    for role, child in children:
        child_transaction = _hybrid_child_transaction_id(
            transaction_id,
            role,
            "export",
        )
        child_namespace = None
        error = None
        try:
            child_namespace = _child_namespace(namespace, role, limits)
        except Exception as exc:
            error = exc
        child_context = hashlib.sha256(
            b"gefen.portable_dcp_hybrid.save_child.v1\0"
            + role.encode("ascii")
        ).digest()
        _collective_unanimous_status(
            binding,
            error,
            operation="portable_dcp_hybrid_save_child_prep_{}".format(role),
            transaction_id=child_transaction,
            context_digest=child_context,
            limits=limits._wire_limits(collective=True),
        )
        assert child_namespace is not None

        child_state = None
        metadata = None
        error = None
        try:
            child_state = _prepare_sharded_save_state(
                child,
                binding=binding,
                transaction_id=child_transaction,
                limits=limits,
                namespace=child_namespace,
            )
            metadata = _decode_metadata_tensor(
                child_state[child_namespace][_DCP_SHARDED_METADATA_KEY],
                limits=limits,
            )
            child_metadata[role] = metadata
            child_descriptors[role] = {
                "namespace": child_namespace,
                "implementation": metadata["implementation"],
                "metadata_digest": metadata["completion"]["metadata_digest"],
            }
            state.update(child_state)
        except Exception as exc:
            error = exc
        _collective_unanimous_status(
            binding,
            error,
            operation="portable_dcp_hybrid_save_child_{}".format(role),
            transaction_id=child_transaction,
            context_digest=child_context,
            limits=limits._wire_limits(collective=True),
        )
        assert child_state is not None and metadata is not None

    common_values = [
        (
            metadata["common"]["gefen_global_step"],
            metadata["common"]["gefen_deterministic"],
        )
        for metadata in child_metadata.values()
    ]
    _validate_aggregate_child_limits(
        child_metadata,
        binding=binding,
        limits=limits,
    )
    if any(value != common_values[0] for value in common_values[1:]):
        raise RuntimeError("sharded Hybrid child common state disagrees")
    semantic = {
        "format": _COMPOSITE_FORMAT,
        "format_version": _COMPOSITE_FORMAT_VERSION,
        "coverage": _COMPOSITE_COVERAGE,
        "implementation": _COMPOSITE_IMPLEMENTATION,
        "backup_optimizer": "gefen",
        "source_process_group": _serialize_process_group_identity(
            binding.identity
        ),
        "routing": routing,
        "common": {
            "gefen_global_step": common_values[0][0],
            "gefen_deterministic": common_values[0][1],
        },
        "children": child_descriptors,
    }
    metadata, metadata_tensor = _completed_metadata(semantic, limits=limits)
    _normalize_metadata(metadata, namespace=namespace, limits=limits)
    if live_token != _hybrid_portable_live_token(optimizer):
        raise RuntimeError("live Hybrid state changed during sharded DCP save preparation")
    state[namespace] = {_COMPOSITE_METADATA_KEY: metadata_tensor}
    metadata_bytes = metadata_tensor.numel()
    for role, descriptor in child_descriptors.items():
        if descriptor is None:
            continue
        from gefen.portable_dcp_sharded import _DCP_SHARDED_METADATA_KEY

        metadata_bytes += state[descriptor["namespace"]][
            _DCP_SHARDED_METADATA_KEY
        ].numel()
    if metadata_bytes > limits.max_collective_metadata_bytes:
        raise ValueError("sharded Hybrid metadata exceeds max_collective_metadata_bytes")
    return state, metadata


def _save_sharded_hybrid_dcp(
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
    metadata = None
    planner = None
    error = None
    try:
        state, metadata = _prepare_save_state(
            optimizer,
            binding=binding,
            transaction_id=transaction_id,
            limits=limits,
            namespace=namespace,
        )
        planner = dcp.DefaultSavePlanner()
    except Exception as exc:
        error = exc
    digest = (
        bytes(32)
        if metadata is None
        else bytes.fromhex(metadata["completion"]["metadata_digest"])
    )
    _collective_unanimous_status(
        binding,
        error,
        operation="portable_dcp_hybrid_save_prepare",
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
        operation="portable_dcp_hybrid_save_dcp_complete",
        transaction_id=transaction_id,
        context_digest=digest,
        limits=limits._wire_limits(collective=True),
    )
    return result


def _validate_children(
    optimizer,
    metadata,
    child_metadata,
    *,
    binding: CheckpointProcessGroupBinding,
):
    from gefen.portable_hybrid import _validate_hybrid_portable_readiness
    from gefen.portable_identity import _parse_process_group_identity

    children, routing, _layouts = _validate_hybrid_portable_readiness(
        optimizer,
        binding,
    )
    child_by_role = dict(children)
    if metadata["routing"] != routing:
        raise ValueError("sharded Hybrid routing does not match the target")
    if {
        role: metadata["children"][role] is not None for role in _ROLES
    } != {role: role in child_by_role for role in _ROLES}:
        raise ValueError("sharded Hybrid child presence does not match the target")
    source_group = _parse_process_group_identity(metadata["source_process_group"])
    expected_routing = {}
    for role, child in child_by_role.items():
        descriptor = metadata["children"][role]
        child_state = child_metadata[role]
        if (
            child_state["implementation"] != _CHILD_IMPLEMENTATIONS[role]
            or child_state["completion"]["metadata_digest"]
            != descriptor["metadata_digest"]
            or _parse_process_group_identity(child_state["source_process_group"])
            != source_group
            or child_state["common"]["gefen_global_step"]
            != metadata["common"]["gefen_global_step"]
            or child_state["common"]["gefen_deterministic"]
            is not metadata["common"]["gefen_deterministic"]
        ):
            raise ValueError("sharded Hybrid child metadata conflicts with its envelope")
        for fqn in child_state["parameters"]:
            if fqn in expected_routing:
                raise ValueError("sharded Hybrid child parameter FQNs overlap")
            expected_routing[fqn] = role
    if metadata["routing"] != {
        fqn: expected_routing[fqn] for fqn in sorted(expected_routing)
    }:
        raise ValueError("sharded Hybrid routing does not match its children")
    return tuple((role, child_by_role[role]) for role in _ROLES if role in child_by_role)


def _expected_checkpoint_keys(metadata, child_metadata, *, namespace: str):
    from gefen.portable_dcp import _flat_key

    keys = {_flat_key(namespace, _COMPOSITE_METADATA_KEY)}
    for role, child in child_metadata.items():
        child_namespace = metadata["children"][role]["namespace"]
        from gefen.portable_dcp_sharded import _DCP_SHARDED_METADATA_KEY

        keys.add(_flat_key(child_namespace, _DCP_SHARDED_METADATA_KEY))
        keys.update(
            _flat_key(child_namespace, field["key"])
            for field in child["fields"]
        )
    return keys


def _validate_checkpoint_entries(
    checkpoint_metadata,
    metadata,
    child_metadata,
    *,
    namespace: str,
    binding: CheckpointProcessGroupBinding,
    limits,
):
    from gefen.portable_dcp import _flat_key
    from gefen.portable_dcp_sharded import _validate_sharded_dcp_entries

    expected = _expected_checkpoint_keys(
        metadata,
        child_metadata,
        namespace=namespace,
    )
    if set(checkpoint_metadata.state_dict_metadata) != expected:
        raise ValueError("sharded Hybrid checkpoint has unexpected tensor keys")
    if _flat_key(namespace, _COMPOSITE_METADATA_KEY) not in expected:
        raise ValueError("sharded Hybrid checkpoint is missing its envelope")
    total_chunks = 0
    metadata_bytes = 0
    metadata_keys = {_flat_key(namespace, _COMPOSITE_METADATA_KEY)}
    from gefen.portable_dcp_sharded import _DCP_SHARDED_METADATA_KEY

    metadata_keys.update(
        _flat_key(metadata["children"][role]["namespace"], _DCP_SHARDED_METADATA_KEY)
        for role in child_metadata
    )
    for key in expected:
        entry = checkpoint_metadata.state_dict_metadata[key]
        chunks = getattr(entry, "chunks", None)
        if type(chunks) is not list:
            raise TypeError("sharded Hybrid DCP entries require chunk lists")
        total_chunks += len(chunks)
        if key in metadata_keys:
            size = getattr(entry, "size", None)
            if type(size) is not torch.Size or len(size) != 1:
                raise ValueError("sharded Hybrid metadata geometry is invalid")
            metadata_bytes += size[0]
    if (
        total_chunks > limits.max_container_items
        or total_chunks > limits.max_tree_nodes
    ):
        raise ValueError("sharded Hybrid DCP chunks exceed aggregate limits")
    if metadata_bytes > limits.max_collective_metadata_bytes:
        raise ValueError("sharded Hybrid metadata exceeds max_collective_metadata_bytes")
    for role, child in child_metadata.items():
        _validate_sharded_dcp_entries(
            checkpoint_metadata,
            child,
            namespace=metadata["children"][role]["namespace"],
            binding=binding,
            limits=limits,
            checkpoint_keys=expected,
        )
    return expected


def _load_sharded_hybrid_dcp(
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
    from gefen.portable_dcp_sharded import (
        _load_sharded_payloads,
        _metadata_storage_spec as _child_metadata_storage_spec,
        _projected_target_byte_plan,
        _read_sharded_metadata,
        _stage_sharded_import,
    )
    from gefen.portable_hybrid import (
        _hybrid_child_transaction_id,
        _hybrid_portable_live_token,
        _validate_hybrid_portable_readiness,
    )

    metadata = None
    error = None
    try:
        metadata = _read_metadata(
            storage_reader=storage_reader,
            checkpoint_metadata=checkpoint_metadata,
            binding=binding,
            namespace=namespace,
            limits=limits,
            transaction_id=transaction_id,
        )
        if metadata is None:
            raise ValueError("checkpoint does not contain a sharded Hybrid envelope")
    except Exception as exc:
        error = exc
    digest = (
        bytes(32)
        if metadata is None
        else bytes.fromhex(metadata["completion"]["metadata_digest"])
    )
    _collective_unanimous_status(
        binding,
        error,
        operation="portable_dcp_hybrid_load_metadata",
        transaction_id=transaction_id,
        context_digest=digest,
        limits=limits._wire_limits(collective=True),
    )
    assert checkpoint_metadata is not None and metadata is not None

    child_metadata = {}
    children = None
    expected_keys = None
    composite_live_token = None
    child_shapes = {}
    error = None
    try:
        composite_live_token = _hybrid_portable_live_token(optimizer)
        root_shape = _metadata_storage_spec(
            checkpoint_metadata,
            namespace=namespace,
            limits=limits,
        )
        assert root_shape is not None
        aggregate_metadata_bytes = root_shape[0]
        for role in _ROLES:
            descriptor = metadata["children"][role]
            if descriptor is None:
                continue
            shape = _child_metadata_storage_spec(
                checkpoint_metadata,
                namespace=descriptor["namespace"],
                limits=limits,
            )
            if shape is None:
                raise ValueError("sharded Hybrid child metadata is missing")
            child_shapes[role] = shape
            aggregate_metadata_bytes += shape[0]
        if aggregate_metadata_bytes > limits.max_collective_metadata_bytes:
            raise ValueError("sharded Hybrid metadata exceeds max_collective_metadata_bytes")
    except Exception as exc:
        error = exc
    _collective_unanimous_status(
        binding,
        error,
        operation="portable_dcp_hybrid_load_child_specs",
        transaction_id=transaction_id,
        context_digest=digest,
        limits=limits._wire_limits(collective=True),
    )
    assert composite_live_token is not None

    for role in _ROLES:
        descriptor = metadata["children"][role]
        if descriptor is None:
            continue
        child_state = None
        error = None
        try:
            child_transaction = _hybrid_child_transaction_id(
                transaction_id,
                role,
                "import",
            )
            child_state = _read_sharded_metadata(
                storage_reader=storage_reader,
                checkpoint_metadata=checkpoint_metadata,
                binding=binding,
                namespace=descriptor["namespace"],
                limits=limits,
                transaction_id=child_transaction,
            )
            if child_state is None:
                raise ValueError("sharded Hybrid child metadata is missing")
        except Exception as exc:
            error = exc
        role_digest = hashlib.sha256(digest + role.encode("ascii")).digest()
        _collective_unanimous_status(
            binding,
            error,
            operation="portable_dcp_hybrid_load_child_metadata",
            transaction_id=transaction_id,
            context_digest=role_digest,
            limits=limits._wire_limits(collective=True),
        )
        assert child_state is not None
        child_metadata[role] = child_state

    error = None
    try:
        children = _validate_children(
            optimizer,
            metadata,
            child_metadata,
            binding=binding,
        )
        _validate_aggregate_child_limits(
            child_metadata,
            binding=binding,
            limits=limits,
        )
        expected_keys = _validate_checkpoint_entries(
            checkpoint_metadata,
            metadata,
            child_metadata,
            namespace=namespace,
            binding=binding,
            limits=limits,
        )
    except Exception as exc:
        error = exc
    _collective_unanimous_status(
        binding,
        error,
        operation="portable_dcp_hybrid_load_children",
        transaction_id=transaction_id,
        context_digest=digest,
        limits=limits._wire_limits(collective=True),
    )
    assert (
        children is not None
        and expected_keys is not None
        and composite_live_token is not None
    )

    projected_digest = digest
    error = None
    try:
        child_plans = []
        for role, child in children:
            (
                child_digest,
                child_projected,
                child_scratch,
                child_send,
            ) = _projected_target_byte_plan(
                child,
                child_metadata[role],
                binding=binding,
                limits=limits,
            )
            child_plans.append(
                (
                    role,
                    child_digest,
                    child_projected,
                    child_scratch,
                    child_send,
                )
            )
        projected_totals = [0 for _member in binding.identity.ordered_members]
        for (
            _role,
            _child_digest,
            child_projected,
            _child_scratch,
            _child_send,
        ) in child_plans:
            for coordinate, value in enumerate(child_projected):
                projected_totals[coordinate] += value
        if any(value > limits.max_fragment_tensor_bytes for value in projected_totals):
            raise ValueError("sharded Hybrid projected target state exceeds max_fragment_tensor_bytes")
        hasher = hashlib.sha256()
        hasher.update(b"gefen.portable_dcp_hybrid.projected.v1\0")
        hasher.update(digest)
        for role, child_digest, child_projected, child_scratch, child_send in child_plans:
            hasher.update(role.encode("ascii"))
            hasher.update(child_digest)
            for values in (child_projected, child_scratch, child_send):
                for value in values:
                    hasher.update(value.to_bytes(16, "big", signed=False))
        projected_digest = hasher.digest()
    except Exception as exc:
        error = exc
    _collective_unanimous_status(
        binding,
        error,
        operation="portable_dcp_hybrid_load_projected_limits",
        transaction_id=transaction_id,
        context_digest=projected_digest,
        limits=limits._wire_limits(collective=True),
    )

    staged_children = []
    for role, child in children:
        child_state = child_metadata[role]
        child_namespace = metadata["children"][role]["namespace"]
        child_transaction = _hybrid_child_transaction_id(
            transaction_id,
            role,
            "import",
        )
        local_fields = None
        staged = None
        live_token = None
        error = None
        try:
            shape = child_shapes[role]
            local_fields = _load_sharded_payloads(
                storage_reader=storage_reader,
                checkpoint_metadata=checkpoint_metadata,
                metadata=child_state,
                metadata_shape=shape,
                binding=binding,
                namespace=child_namespace,
                limits=limits,
                transaction_id=child_transaction,
                context_digest=bytes.fromhex(
                    child_state["completion"]["metadata_digest"]
                ),
                checkpoint_keys=expected_keys,
            )
            staged, live_token = _stage_sharded_import(
                child,
                child_state,
                local_fields,
                binding=binding,
                limits=limits,
                transaction_id=child_transaction,
            )
        except Exception as exc:
            error = exc
        role_digest = hashlib.sha256(digest + role.encode("ascii")).digest()
        _collective_unanimous_status(
            binding,
            error,
            operation="portable_dcp_hybrid_load_stage",
            transaction_id=child_transaction,
            context_digest=role_digest,
            limits=limits._wire_limits(collective=True),
        )
        assert local_fields is not None and staged is not None and live_token is not None
        staged_children.append((child, staged, live_token))

    error = None
    try:
        _validate_hybrid_portable_readiness(optimizer, binding)
        if composite_live_token != _hybrid_portable_live_token(optimizer):
            raise RuntimeError("live Hybrid state changed after sharded DCP staging")
        from gefen import portable_runtime as runtime

        for child, _staged, live_token in staged_children:
            if live_token != runtime._portable_live_token(child):
                raise RuntimeError("live Hybrid child changed after sharded DCP staging")
    except Exception as exc:
        error = exc
    _collective_unanimous_status(
        binding,
        error,
        operation="portable_dcp_hybrid_load_freshness",
        transaction_id=transaction_id,
        context_digest=digest,
        limits=limits._wire_limits(collective=True),
    )

    from gefen.gefen import Gefen

    for child, staged, _live_token in staged_children:
        Gefen._commit_staged_load_state_dict(child, staged)
    dict.__setitem__(
        optimizer.__dict__,
        "_deterministic",
        metadata["common"]["gefen_deterministic"],
    )


__all__ = []

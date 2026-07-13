"""Optimizer-facing collective runtime for exact portable Gefen state."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math

import torch
from torch import nn

from gefen.canonical import canonical_value_supported
from gefen.checkpoint import CheckpointProcessGroupBinding
from gefen.codebook import CodebookProcessGroupBinding
from gefen.contracts import (
    IDENTITY_SCHEMA_VERSION,
    LogicalSlice,
    ParameterIdentity,
    ParameterLayout,
    PlacementKind,
    ProcessGroupIdentity,
    ShardIdentity,
    ShardPlacement,
    ShardingManifest,
)
from gefen.portable import (
    _decode_quantized_momentum,
    _expand_block_second_moment,
    _read_flat_chunk,
)
from gefen.portable_collective import (
    _collective_unanimous_status,
    _collective_visit_canonical_fragments,
)
from gefen.portable_identity import (
    _serialize_parameter_identity,
    _serialize_process_group_identity,
    _serialize_shard_identity,
    _serialize_sharding_manifest,
)
from gefen.portable_schema import portable_state_digest
from gefen.portable_state import (
    PortableStateLimits,
    _MOMENTUM_PROJECTION,
    _SECOND_MOMENT_PROJECTION,
    _assemble_portable_state_fragments,
    _build_portable_state_fragment,
    _derived_role,
    _local_dense_shape,
    _normalize_common,
    _normalize_gefen_portable_state_document,
    _normalize_options,
    _normalize_policy,
    _normalize_portable_state_fragment,
    _project_portable_parameter_state,
    _values_equal,
)
from gefen.rebinding import LogicalSlotBinding


_PLAIN_IMPLEMENTATION = "gefen.Gefen"
_MUON_IMPLEMENTATION = "gefen.GefenMuon"
_MAX_TRANSACTION_BYTES = 1024
_EXPORT_PREFLIGHT_TRANSACTION = "gefen-portable-export-preflight-v1"
_IMPORT_PREFLIGHT_TRANSACTION = "gefen-portable-import-preflight-v1"
_STATUS_FALLBACK_LIMITS = PortableStateLimits(
    max_fragment_tensor_bytes=1,
    max_collective_tensor_bytes=1,
    max_collective_metadata_bytes=1,
    max_metadata_bytes=1,
)._wire_limits()

_PLAIN_GROUP_REQUIRED = frozenset({"params", "param_names", "lr", "beta1", "beta2", "eps", "weight_decay"})
_PLAIN_GROUP_ALLOWED = _PLAIN_GROUP_REQUIRED | {"name"}
_MUON_GROUP_REQUIRED = _PLAIN_GROUP_REQUIRED | frozenset(
    {
        "momentum",
        "nesterov",
        "ns_coefficients",
        "ns_steps",
        "adjust_lr_fn",
        "sharded_mode",
        "fp8_ns",
        "fp8_ns_compile",
        "batched_ns",
        "batched_ns_workspace_bytes",
        "normuon",
        "normuon_beta2",
        "normuon_eps",
        "cautious",
    }
)
_MUON_GROUP_ALLOWED = _MUON_GROUP_REQUIRED | {"name", "ns_schedule"}

_AUTHORITATIVE_STATE_KEYS = frozenset(
    {
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
        "normuon_v",
        "normuon_step",
    }
)
_IGNORED_DERIVED_STATE_KEYS = frozenset({"stepsize", "_h_buf", "m_codebook_shape"})
_PLAIN_BLOCK_KEYS = frozenset(
    {
        "name",
        "automatic_period",
        "step",
        "m_codebook",
        "m_magnitude",
        "vmean",
        "vmean_step",
    }
)
_PLAIN_FACTORED_KEYS = frozenset(
    {
        "name",
        "automatic_period",
        "step",
        "m_codebook",
        "m_magnitude",
        "v_row",
        "v_col",
        "factored_step",
    }
)
_MUON_KEYS = frozenset({"name", "automatic_period", "step", "m_codebook", "m_magnitude"})
_NORMUON_KEYS = _MUON_KEYS | {"normuon_v", "normuon_step"}


def _bounded_utf8_length(value: str, *, limit: int, name: str) -> int:
    total = 0
    for start in range(0, len(value), 4096):
        chunk = value[start : start + 4096].encode("utf-8")
        if total > limit - len(chunk):
            raise ValueError("{} exceeds its UTF-8 byte limit".format(name))
        total += len(chunk)
    return total


class _PreflightTensor:
    __slots__ = ("shape",)

    def __init__(self, shape):
        self.shape = tuple(shape)


def _preflight_portable_value(value, limits: PortableStateLimits) -> None:
    """Apply exact wire structural limits without allocating tensor payloads."""

    wire = limits._wire_limits()
    nodes = 0
    tensor_bytes = 0
    tensor_shapes = []
    metadata_bytes = 16

    def add_metadata(count):
        nonlocal metadata_bytes
        metadata_bytes += count
        if metadata_bytes > wire.max_metadata_bytes:
            raise ValueError("portable fragment exceeds max_metadata_bytes before materialization")

    def visit(item, *, depth):
        nonlocal nodes, tensor_bytes
        if depth > wire.max_tree_depth:
            raise ValueError("portable fragment exceeds max_tree_depth before materialization")
        nodes += 1
        if nodes > wire.max_tree_nodes:
            raise ValueError("portable fragment exceeds max_tree_nodes before materialization")
        item_type = type(item)
        if item is None or item_type is bool:
            add_metadata(1)
            return
        if item_type is int:
            magnitude_bytes = (abs(item).bit_length() + 7) // 8
            if magnitude_bytes > wire.max_integer_bytes:
                raise ValueError("portable fragment integer exceeds max_integer_bytes")
            add_metadata(10 + magnitude_bytes)
            return
        if item_type is float:
            if not math.isfinite(item):
                raise ValueError("portable fragment contains a nonfinite float")
            add_metadata(9)
            return
        if item_type is str:
            encoded_bytes = _bounded_utf8_length(
                item,
                limit=wire.max_string_bytes,
                name="portable fragment string",
            )
            add_metadata(9 + encoded_bytes)
            return
        if item_type is _PreflightTensor:
            shape = item.shape
            if any(type(dimension) is not int or dimension < 0 for dimension in shape):
                raise ValueError("portable fragment tensor has invalid geometry")
            if len(shape) > wire.max_tensor_rank:
                raise ValueError("portable fragment tensor exceeds max_tensor_rank")
            if len(tensor_shapes) >= wire.max_tensors:
                raise ValueError("portable fragment exceeds max_tensors")
            numel = math.prod(shape)
            nbytes = numel * 4
            if tensor_bytes > wire.max_fragment_tensor_bytes - nbytes:
                raise ValueError("portable fragment exceeds max_fragment_tensor_bytes before materialization")
            tensor_bytes += nbytes
            tensor_shapes.append(shape)
            add_metadata(9)
            return
        if item_type in {list, tuple}:
            if len(item) > wire.max_container_items:
                raise ValueError("portable fragment container exceeds max_container_items")
            add_metadata(9)
            for child in item:
                visit(child, depth=depth + 1)
            return
        if item_type is dict:
            if len(item) > wire.max_container_items:
                raise ValueError("portable fragment container exceeds max_container_items")
            if any(type(key) is not str for key in item):
                raise TypeError("portable fragment dictionary keys must be strings")
            add_metadata(9)
            for key in sorted(item):
                visit(key, depth=depth + 1)
                visit(item[key], depth=depth + 1)
            return
        raise TypeError("portable preflight encountered an unsupported value")

    visit(value, depth=0)
    add_metadata(8)
    for shape in tensor_shapes:
        add_metadata(56 + 8 * len(shape))


def _optimizer_implementation(optimizer) -> str:
    # Keep imports lazy so Gefen may delegate to this module without creating a
    # module-import cycle.
    from gefen.gefen import Gefen
    from gefen.gefen_muon import GefenMuon

    if type(optimizer) is Gefen:
        return _PLAIN_IMPLEMENTATION
    if type(optimizer) is GefenMuon:
        return _MUON_IMPLEMENTATION
    raise TypeError("portable state supports exact Gefen and GefenMuon instances only")


def _reject_method_shadows(optimizer) -> None:
    if type(optimizer.__dict__) is not dict:
        raise TypeError("portable state requires an exact optimizer attribute mapping")
    for name in optimizer.__dict__:
        descriptor = None
        for owner in type(optimizer).__mro__:
            if name in owner.__dict__:
                descriptor = owner.__dict__[name]
                break
        if isinstance(descriptor, (staticmethod, classmethod)):
            descriptor = descriptor.__func__
        if callable(descriptor):
            raise TypeError("portable state rejects instance-level method shadows")


def _require_limits(limits) -> PortableStateLimits:
    if type(limits) is not PortableStateLimits:
        raise TypeError("limits must be a PortableStateLimits")
    return limits


def _require_binding(binding) -> CheckpointProcessGroupBinding:
    if type(binding) is not CheckpointProcessGroupBinding:
        raise TypeError("checkpoint_process_group must be a CheckpointProcessGroupBinding")
    return binding


def _validate_exact_process_group_identity(identity) -> ProcessGroupIdentity:
    if (
        type(identity) is not ProcessGroupIdentity
        or type(identity.semantic_name) is not str
        or type(identity.ordered_members) is not tuple
        or any(type(member) is not str for member in identity.ordered_members)
        or type(identity.schema_version) is not int
    ):
        raise TypeError("portable state requires an exact ProcessGroupIdentity")
    validated = ProcessGroupIdentity(
        identity.semantic_name,
        identity.ordered_members,
        schema_version=identity.schema_version,
    )
    if validated != identity:
        raise ValueError("portable process-group identities must be canonical")
    return validated


def _validate_exact_shard_identity(shard) -> ShardIdentity:
    if type(shard) is not ShardIdentity:
        raise TypeError("portable state requires exact ShardIdentity values")
    if type(shard.parameter) is not ParameterIdentity:
        raise TypeError("portable state requires exact ParameterIdentity values")
    if (
        type(shard.parameter.fqn) is not str
        or type(shard.parameter.global_shape) is not tuple
        or any(type(dimension) is not int for dimension in shard.parameter.global_shape)
        or type(shard.parameter.schema_version) is not int
    ):
        raise TypeError("portable parameter identities require exact primitive fields")
    parameter = ParameterIdentity(
        shard.parameter.fqn,
        shard.parameter.global_shape,
        schema_version=shard.parameter.schema_version,
    )
    if type(shard.logical_slice) is not LogicalSlice:
        raise TypeError("portable state requires exact LogicalSlice values")
    if (
        type(shard.logical_slice.flat_offset) is not int
        or type(shard.logical_slice.length) is not int
        or type(shard.local_member) is not str
        or (shard.owner is not None and type(shard.owner) is not str)
        or type(shard.schema_version) is not int
    ):
        raise TypeError("portable shard identities require exact primitive fields")
    logical_slice = LogicalSlice(
        shard.logical_slice.flat_offset,
        shard.logical_slice.length,
    )
    process_group = _validate_exact_process_group_identity(shard.process_group)
    if type(shard.placements) is not tuple or any(
        type(placement) is not ShardPlacement for placement in shard.placements
    ):
        raise TypeError("portable state requires exact ShardPlacement values")
    if any(
        type(placement.mesh_axis) is not str
        or type(placement.kind) is not PlacementKind
        or type(placement.coordinate) is not int
        or type(placement.parts) is not int
        or (placement.parameter_dimension is not None and type(placement.parameter_dimension) is not int)
        for placement in shard.placements
    ):
        raise TypeError("portable shard placements require exact primitive fields")
    placements = tuple(
        ShardPlacement(
            placement.mesh_axis,
            placement.kind,
            placement.coordinate,
            placement.parts,
            placement.parameter_dimension,
        )
        for placement in shard.placements
    )
    if type(shard.layout) is not ParameterLayout:
        raise TypeError("portable shard layouts require exact ParameterLayout values")
    validated = ShardIdentity(
        parameter,
        shard.layout,
        logical_slice,
        placements=placements,
        process_group=process_group,
        local_member=shard.local_member,
        owner=shard.owner,
        schema_version=shard.schema_version,
    )
    if validated != shard:
        raise ValueError("portable shard identities must be canonical")
    return validated


def _validate_exact_manifest(manifest) -> None:
    if (
        type(manifest) is not ShardingManifest
        or type(manifest.shards) is not tuple
        or type(manifest.schema_version) is not int
    ):
        raise TypeError("portable state requires an exact ShardingManifest")
    validated = ShardingManifest(
        tuple(_validate_exact_shard_identity(shard) for shard in manifest.shards),
        schema_version=manifest.schema_version,
    )
    if validated != manifest:
        raise ValueError("portable sharding manifests must be canonical")


def _require_transaction_id(transaction_id) -> str:
    if (
        type(transaction_id) is not str
        or not transaction_id
        or transaction_id != transaction_id.strip()
        or "\x00" in transaction_id
    ):
        raise ValueError("transaction_id must be non-empty, trimmed, and without NUL")
    _bounded_utf8_length(
        transaction_id,
        limit=_MAX_TRANSACTION_BYTES,
        name="transaction_id",
    )
    return transaction_id


def _preflight_transport_binding(optimizer, supplied):
    scope = getattr(optimizer, "_gefen_codebook_process_group", None)
    if type(scope) is CodebookProcessGroupBinding:
        identity = ProcessGroupIdentity(
            str(scope.identity.semantic_name),
            tuple(str(member) for member in scope.identity.ordered_members),
            schema_version=IDENTITY_SCHEMA_VERSION,
        )
        return CheckpointProcessGroupBinding(
            identity,
            str(scope.local_member),
            scope.process_group,
            scope.collective_device,
        )
    if isinstance(supplied, CheckpointProcessGroupBinding):
        return supplied
    raise TypeError("portable preflight requires a transport-usable process group")


def _validate_supplied_binding(supplied, transport) -> None:
    supplied = _require_binding(supplied)
    _validate_exact_process_group_identity(supplied.identity)
    _validate_exact_process_group_identity(transport.identity)
    if (
        type(supplied.local_member) is not str
        or type(supplied.collective_device) is not torch.device
        or type(transport.local_member) is not str
        or type(transport.collective_device) is not torch.device
    ):
        raise TypeError("portable checkpoint bindings require exact primitive fields")
    if supplied.identity != transport.identity or supplied.local_member != transport.local_member:
        raise ValueError("checkpoint and codebook process-group identities must match")
    if (
        supplied.process_group is not transport.process_group
        or supplied.collective_device != transport.collective_device
    ):
        raise ValueError("checkpoint_process_group must exactly match the optimizer-owned runtime transport")


def _validate_context_identity_bounds(
    binding: CheckpointProcessGroupBinding,
    limits: PortableStateLimits,
) -> None:
    identity = binding.identity
    member_count = len(identity.ordered_members)
    if member_count > limits.max_members or member_count > limits.max_container_items:
        raise ValueError("checkpoint process-group identity exceeds member limits")
    _bounded_utf8_length(
        identity.semantic_name,
        limit=limits.max_string_bytes,
        name="process-group semantic name",
    )
    for member in identity.ordered_members:
        _bounded_utf8_length(
            member,
            limit=limits.max_string_bytes,
            name="process-group member",
        )


def _context_digest(value) -> bytes:
    return bytes.fromhex(portable_state_digest(value))


def _base_context(binding: CheckpointProcessGroupBinding, implementation: str):
    return {
        "format": "gefen.portable_runtime_context",
        "format_version": 1,
        "implementation": implementation,
        "checkpoint_process_group": _serialize_process_group_identity(binding.identity),
    }


def _plain_tensor(
    value,
    *,
    name: str,
    dtype=None,
    shape=None,
    nonnegative=False,
    validate_values=True,
):
    if (
        type(value) is not torch.Tensor
        or value.layout is not torch.strided
        or value.is_meta
        or value.is_nested
        or value.is_quantized
        or value.requires_grad
        or value.is_conj()
        or value.is_neg()
    ):
        raise TypeError("{} must be a plain materialized strided tensor".format(name))
    if dtype is not None and value.dtype != dtype:
        raise TypeError("{} has an invalid dtype".format(name))
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError("{} has invalid geometry".format(name))
    if value.is_floating_point() and validate_values:
        for start in range(0, value.numel(), 1 << 20):
            chunk = _read_flat_chunk(
                value,
                start,
                min(start + (1 << 20), value.numel()),
            )
            if not bool(torch.isfinite(chunk).all()):
                raise ValueError("{} must be finite".format(name))
            if nonnegative and not bool((chunk >= 0).all()):
                raise ValueError("{} must be nonnegative".format(name))
    return value


def _tight_cpu_fp32(value, *, name: str, shape=None, nonnegative=False):
    value = _plain_tensor(
        value,
        name=name,
        dtype=torch.float32,
        shape=shape,
        nonnegative=nonnegative,
    )
    result = torch.empty(tuple(value.shape), dtype=torch.float32, device="cpu")
    result.copy_(value.detach())
    return result


def _strict_counter(value, *, name: str, minimum=0):
    if type(value) is not int or value < minimum or value > (1 << 53) - 1:
        raise ValueError("{} must be an exact bounded host int".format(name))
    return value


def _live_policy(optimizer, implementation: str):
    raw = optimizer._canonical_policy()
    if type(raw) is not dict:
        raise ValueError("live portable policy must be a plain dictionary")
    policy = {
        "schema_version": 1,
        "factored_v_2d": raw.get("factored_v_2d"),
        "force_1d_period_one": raw.get("force_1d_period_one"),
        "force_2d_period_one": raw.get("force_2d_period_one"),
        "period_one_substrings": raw.get("period_one_substrings"),
        "codebook_refresh_every": raw.get("codebook_refresh_every"),
        "stochastic_round": raw.get("stochastic_round"),
        "momentum_projection": _MOMENTUM_PROJECTION,
        "second_moment_projection": _SECOND_MOMENT_PROJECTION,
    }
    return _normalize_policy(policy, implementation)


def _normalized_ns_schedule(group):
    from gefen.gefen_muon import _normalize_ns_schedule

    schedule = _normalize_ns_schedule(group["ns_coefficients"], group["ns_steps"])
    return [[float(a), float(b), float(c)] for a, b, c in schedule]


def _group_options(group, implementation: str, *, second_moment_policy=None):
    if type(group) is not dict:
        raise TypeError("portable parameter groups must be plain dictionaries")
    allowed = _PLAIN_GROUP_ALLOWED if implementation == _PLAIN_IMPLEMENTATION else _MUON_GROUP_ALLOWED
    required = _PLAIN_GROUP_REQUIRED if implementation == _PLAIN_IMPLEMENTATION else _MUON_GROUP_REQUIRED
    if not required.issubset(group) or not set(group).issubset(allowed):
        raise ValueError("portable parameter group contains missing or unknown keys")
    if type(group["params"]) is not list or type(group["param_names"]) is not list:
        raise TypeError("portable parameter group params and names must be plain lists")

    if implementation == _PLAIN_IMPLEMENTATION:
        return _normalize_options(
            {
                "lr": group["lr"],
                "beta1": group["beta1"],
                "beta2": group["beta2"],
                "eps": group["eps"],
                "weight_decay": group["weight_decay"],
                "second_moment_policy": second_moment_policy,
            },
            implementation,
        )

    if any(type(group[key]) is not float for key in ("beta1", "beta2", "momentum")):
        raise ValueError("Muon inherited betas and momentum must be exact floats")
    if group["beta1"] != group["momentum"] or group["beta2"] != 0.0:
        raise ValueError("Muon inherited beta values disagree with its momentum policy")
    if "ns_schedule" in group and not canonical_value_supported(group["ns_schedule"], finite_tensors=True):
        raise ValueError("Muon raw ns_schedule must remain weights-only-safe metadata")
    return _normalize_options(
        {
            "lr": group["lr"],
            "weight_decay": group["weight_decay"],
            "momentum": group["momentum"],
            "nesterov": group["nesterov"],
            "ns_schedule": _normalized_ns_schedule(group),
            "ns_eps": group["eps"],
            "adjust_lr_fn": group["adjust_lr_fn"],
            "sharded_mode": group["sharded_mode"],
            "fp8_ns": group["fp8_ns"],
            "fp8_ns_compile": group["fp8_ns_compile"],
            "batched_ns": group["batched_ns"],
            "batched_ns_workspace_bytes": group["batched_ns_workspace_bytes"],
            "normuon": group["normuon"],
            "normuon_beta2": group["normuon_beta2"],
            "normuon_eps": group["normuon_eps"],
            "cautious": group["cautious"],
        },
        implementation,
    )


def _parameter_supported(parameter, shard: ShardIdentity) -> bool:
    if type(parameter) not in {torch.Tensor, nn.Parameter} or not (
        parameter.layout is torch.strided
        and parameter.device.type in {"cpu", "cuda"}
        and parameter.dtype in {torch.float16, torch.bfloat16, torch.float32, torch.float64}
        and not torch.is_complex(parameter)
        and not parameter.is_meta
        and not parameter.is_nested
        and not parameter.is_quantized
        and getattr(parameter, "fake_mode", None) is None
    ):
        return False
    if shard.layout in {
        ParameterLayout.REPLICATED,
        ParameterLayout.WHOLE_PARAMETER_OWNER,
    }:
        return (
            tuple(parameter.shape) == shard.parameter.global_shape and parameter.numel() == shard.logical_slice.length
        )
    if shard.layout is ParameterLayout.FLATTENED_ELEMENT_SHARD:
        return parameter.ndim == 1 and parameter.is_contiguous() and parameter.numel() == shard.logical_slice.length
    return False


def _validate_optimizer_shell(
    optimizer,
    implementation: str,
    binding: CheckpointProcessGroupBinding,
) -> None:
    _reject_method_shadows(optimizer)
    _validate_exact_manifest(optimizer._gefen_sharding_manifest)
    optimizer._assert_finalized_binding_layout()
    if type(optimizer.defaults) is not dict:
        raise TypeError("portable state requires exact built-in optimizer defaults")
    if type(optimizer.param_groups) is not list:
        raise TypeError("portable state requires an exact parameter-group list")
    if optimizer.capturable is not False:
        raise RuntimeError("portable state does not support capturable optimizers")
    if optimizer._stochastic_round is not False:
        raise RuntimeError("portable state does not support stochastic rounding")
    if optimizer._capt_stacks is not None:
        raise RuntimeError("portable state requires inactive capturable stacks")
    if optimizer._gefen_global_step_by_device or optimizer._sr_seed_by_device:
        raise RuntimeError("portable state does not support device-authoritative counters")
    if torch.compiler.is_compiling():
        raise RuntimeError("portable state cannot run while compiling")
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("portable state cannot run during CUDA capture")
    if type(optimizer._gefen_logical_slots) is not tuple or not optimizer._gefen_logical_slots:
        raise ValueError("portable state requires immutable logical slots")
    if type(optimizer._gefen_local_shard_bindings) is not tuple:
        raise TypeError("portable state requires immutable local shard bindings")

    scope = optimizer._gefen_codebook_process_group
    if type(scope) is not CodebookProcessGroupBinding:
        raise RuntimeError("portable state requires an explicit codebook process group")
    _validate_exact_process_group_identity(scope.identity)
    if type(scope.local_member) is not str or type(scope.collective_device) is not torch.device:
        raise TypeError("portable codebook bindings require exact primitive fields")
    if scope.identity != binding.identity or scope.local_member != binding.local_member:
        raise ValueError("checkpoint and codebook process-group identities must match")
    optimizer._validate_codebook_runtime_binding(scope)
    for shard in optimizer._gefen_sharding_manifest.shards:
        if shard.process_group != binding.identity:
            raise ValueError("portable manifest process groups must match the checkpoint binding")
        if implementation == _PLAIN_IMPLEMENTATION:
            if shard.layout not in {
                ParameterLayout.REPLICATED,
                ParameterLayout.FLATTENED_ELEMENT_SHARD,
            }:
                raise ValueError("plain portable state has an unsupported layout")
        elif shard.layout not in {
            ParameterLayout.REPLICATED,
            ParameterLayout.WHOLE_PARAMETER_OWNER,
        }:
            raise ValueError("Muon portable state has an unsupported layout")
    for slot in optimizer._gefen_logical_slots:
        if type(slot) is not LogicalSlotBinding:
            raise TypeError("portable logical slots must be exact LogicalSlotBinding values")
        _validate_exact_shard_identity(slot.shard)
        if slot.shard.local_member != binding.local_member:
            raise ValueError("local logical slots must match the checkpoint member")


def _live_maps(optimizer, binding: CheckpointProcessGroupBinding):
    local_by_fqn = {}
    for item in optimizer._gefen_local_shard_bindings:
        if type(item) is not tuple or len(item) != 2:
            raise TypeError("portable local bindings must be parameter/shard tuples")
        parameter, shard = item
        _validate_exact_shard_identity(shard)
        if shard.process_group != binding.identity:
            raise ValueError("portable local shard process groups must exactly match the checkpoint binding")
        if shard.local_member != binding.local_member:
            raise ValueError("portable local shards must match the checkpoint member")
        if shard.parameter.fqn in local_by_fqn:
            raise ValueError("portable local bindings contain duplicate FQNs")
        if parameter is not None and not _parameter_supported(parameter, shard):
            raise TypeError("portable state requires supported live shard geometry and dtype")
        local_by_fqn[shard.parameter.fqn] = (parameter, shard)

    expected_live = {
        parameter
        for parameter, shard in local_by_fqn.values()
        if not (shard.layout is ParameterLayout.WHOLE_PARAMETER_OWNER and shard.local_member != shard.owner)
    }
    if None in expected_live:
        raise TypeError("portable state requires plain materialized local parameters")
    state_type = type(optimizer.state)
    if not (state_type is dict or (state_type is defaultdict and optimizer.state.default_factory is dict)):
        raise TypeError("portable optimizer state must use a standard mapping")
    if set(optimizer.state) != expected_live:
        raise ValueError("portable optimizer state contains missing or foreign parameter keys")
    if type(optimizer._param_names) is not dict or set(optimizer._param_names) != expected_live:
        raise ValueError("portable parameter-name state does not match live parameters")
    return local_by_fqn


def _parameter_storage_token(parameter):
    try:
        version = parameter._version
    except RuntimeError:
        version = None
    return (
        id(parameter),
        version,
        str(parameter.device),
        str(parameter.dtype),
        str(parameter.layout),
        tuple(parameter.shape),
        tuple(parameter.stride()),
        parameter.storage_offset(),
        parameter.numel(),
        parameter.is_contiguous(),
        parameter.untyped_storage().data_ptr(),
        parameter.untyped_storage().nbytes(),
    )


def _portable_value_token(value):
    if type(value) is torch.Tensor:
        hasher = hashlib.sha256()
        element_size = value.element_size()
        chunk_elements = max(1, (1 << 20) // max(1, element_size))
        for start in range(0, value.numel(), chunk_elements):
            stop = min(start + chunk_elements, value.numel())
            chunk = _read_flat_chunk(value, start, stop).contiguous().cpu()
            hasher.update(memoryview(chunk.view(torch.uint8).numpy()))
        try:
            version = value._version
        except RuntimeError:
            version = None
        return (
            "tensor",
            id(value),
            version,
            hasher.digest(),
            str(value.device),
            str(value.dtype),
            str(value.layout),
            tuple(value.shape),
            tuple(value.stride()),
            value.storage_offset(),
            value.requires_grad,
            value.is_conj(),
            value.is_neg(),
        )
    if type(value) is dict:
        return (
            "dict",
            tuple((key, _portable_value_token(value[key])) for key in sorted(value, key=repr)),
        )
    if type(value) in {list, tuple}:
        return (
            type(value).__name__,
            tuple(_portable_value_token(item) for item in value),
        )
    try:
        hash(value)
        token = value
    except TypeError:
        token = (id(value), repr(value))
    return (type(value).__name__, token)


def _portable_live_token(optimizer):
    _reject_method_shadows(optimizer)

    def live_group_options(group):
        options = optimizer._canonical_group_options_value(group)
        if type(options) is dict:
            options = dict(options)
            options.pop("ns_schedule", None)
        return options

    groups = tuple(
        (
            id(group),
            tuple(id(parameter) for parameter in group["params"]),
            tuple(str(name) for name, _parameter in optimizer._iter_group_params_with_names(group)),
            _portable_value_token(live_group_options(group)),
        )
        for group in optimizer.param_groups
    )
    state_entries = tuple(
        sorted(
            (
                id(parameter),
                type(parameter),
                id(state),
                _portable_value_token(state),
            )
            for parameter, state in optimizer.state.items()
        )
    )
    local_storage = tuple(
        None if parameter is None else _parameter_storage_token(parameter)
        for parameter, _shard in optimizer._gefen_local_shard_bindings
    )
    return (
        id(optimizer.param_groups),
        id(optimizer.state),
        id(optimizer.defaults),
        type(optimizer.defaults),
        type(optimizer.param_groups),
        type(optimizer.state),
        _portable_value_token(optimizer.defaults),
        groups,
        state_entries,
        _portable_value_token(optimizer._param_names),
        local_storage,
        optimizer._gefen_global_step,
        _portable_value_token(optimizer._gefen_codebook),
        _portable_value_token(optimizer._gefen_global_step_by_device),
        _portable_value_token(optimizer._sr_seed_by_device),
        _portable_value_token(optimizer._canonical_policy()),
        optimizer._deterministic,
        optimizer.capturable,
        optimizer.fused,
        optimizer.verbose,
        optimizer._fused_build_ok,
        id(optimizer._gefen_codebook_process_group),
        _portable_value_token(optimizer._serialized_codebook_scope()),
        id(optimizer._gefen_sharding_manifest),
        _portable_value_token(_serialize_sharding_manifest(optimizer._gefen_sharding_manifest)),
        id(optimizer._gefen_logical_slots),
        tuple(
            (
                id(slot),
                slot.group_index,
                slot.original_slot_index,
                slot.compatibility_name,
                _portable_value_token(_serialize_shard_identity(slot.shard)),
            )
            for slot in optimizer._gefen_logical_slots
        ),
        tuple(
            (
                None if parameter is None else id(parameter),
                shard.sort_key,
            )
            for parameter, shard in optimizer._gefen_local_shard_bindings
        ),
    )


def _catalog_and_options(optimizer, implementation: str, policy):
    group_options = []
    for group in optimizer.param_groups:
        # Plain options depend on each logical parameter rank, so validate the
        # group shell here and construct per-slot values below.
        if implementation == _PLAIN_IMPLEMENTATION:
            _group_options(group, implementation, second_moment_policy="block")
            group_options.append(group)
        else:
            group_options.append(_group_options(group, implementation))

    catalog = {}
    options_by_fqn = {}
    for slot in optimizer._gefen_logical_slots:
        identity = slot.shard.parameter
        if slot.group_index >= len(group_options):
            raise ValueError("portable logical slot group index is out of range")
        if implementation == _PLAIN_IMPLEMENTATION:
            second = "factored" if policy["factored_v_2d"] and len(identity.global_shape) == 2 else "block"
            options = _group_options(
                group_options[slot.group_index],
                implementation,
                second_moment_policy=second,
            )
            if second == "factored" and slot.shard.layout is not ParameterLayout.REPLICATED:
                raise ValueError("factored portable state supports replicated matrices only")
        else:
            options = group_options[slot.group_index]
            if slot.shard.layout is ParameterLayout.WHOLE_PARAMETER_OWNER and options["sharded_mode"] != "distributed":
                raise ValueError("Muon whole-owner portable state requires distributed mode")
        fqn = identity.fqn
        if fqn in catalog:
            raise ValueError("portable logical slots contain duplicate FQNs")
        options_by_fqn[fqn] = options
        catalog[fqn] = {
            "identity": _serialize_parameter_identity(identity),
            "algorithm_options": options,
        }
    return catalog, options_by_fqn


def _common_state(optimizer):
    global_step = _strict_counter(optimizer._canonical_common_global_step(), name="gefen_global_step")
    codebook = optimizer._gefen_codebook
    codebook_cpu = None
    if codebook is not None:
        codebook_cpu = _tight_cpu_fp32(
            codebook,
            name="gefen_codebook",
            shape=(256,),
        )
    return _normalize_common(
        {
            "gefen_global_step": global_step,
            "gefen_codebook": codebook_cpu,
            "gefen_deterministic": optimizer._deterministic,
        }
    )


def _preflight_common_state(optimizer, *, validate_values=True):
    global_step = _strict_counter(
        optimizer._canonical_common_global_step(),
        name="gefen_global_step",
    )
    codebook = optimizer._gefen_codebook
    codebook_preview = None
    if codebook is not None:
        codebook = _plain_tensor(
            codebook,
            name="gefen_codebook",
            dtype=torch.float32,
            shape=(256,),
            validate_values=validate_values,
        )
        if validate_values:
            if float(codebook[0].item()) != -1.0 or float(codebook[-1].item()) != 1.0:
                raise ValueError("portable codebook must retain exact endpoints")
            if not bool(torch.all(codebook.detach()[1:] >= codebook.detach()[:-1])):
                raise ValueError("portable codebook must be sorted")
        codebook_preview = _PreflightTensor((256,))
    if type(optimizer._deterministic) is not bool:
        raise TypeError("gefen_deterministic must be a bool")
    return {
        "gefen_global_step": global_step,
        "gefen_codebook": codebook_preview,
        "gefen_deterministic": optimizer._deterministic,
    }


def _parameter_state_core(optimizer, parameter, compatibility_name: str):
    state = optimizer.state[parameter]
    if type(state) is not dict:
        raise TypeError("portable per-parameter state must be a plain dictionary")
    if any(key not in _AUTHORITATIVE_STATE_KEYS and key not in _IGNORED_DERIVED_STATE_KEYS for key in state):
        raise ValueError("portable parameter state contains an unknown key")
    core = {key: value for key, value in state.items() if key not in _IGNORED_DERIVED_STATE_KEYS}
    if core.get("name") != compatibility_name:
        raise ValueError("portable parameter state name does not match its logical slot")
    return core


def _decode_local_momentum(core, shard: ShardIdentity, codebook):
    local_shape = _local_dense_shape(shard)
    if local_shape is None:
        raise ValueError("a non-payload shard cannot carry initialized momentum")
    indices = _plain_tensor(core["m_codebook"], name="m_codebook", dtype=torch.uint8)
    magnitudes = _plain_tensor(
        core["m_magnitude"],
        name="m_magnitude",
        dtype=torch.float32,
        nonnegative=True,
    )
    if indices.device != magnitudes.device:
        raise ValueError("momentum indices and magnitudes must share a device")
    local_codebook = codebook.detach().to(device=indices.device).contiguous()
    dense = _decode_quantized_momentum(
        local_codebook,
        indices,
        magnitudes,
        logical_shape=local_shape,
        period=core["automatic_period"],
        step=core["step"],
    )
    return _tight_cpu_fp32(dense, name="dense momentum", shape=local_shape)


def _slot_record(
    optimizer,
    implementation: str,
    slot: LogicalSlotBinding,
    parameter,
    shard: ShardIdentity,
    options,
    common,
):
    role = _derived_role(shard)
    record = {
        "group_index": slot.group_index,
        "original_slot_index": slot.original_slot_index,
        "compatibility_name": slot.compatibility_name,
        "shard": _serialize_shard_identity(shard),
        "algorithm_options": options,
        "role": role,
        "source_period": None,
        "source_second_moment": None,
        "state_variant": "pristine",
        "state": {},
    }
    payload_role = role in {"live", "whole_owner"} and shard.parameter.numel > 0
    if not payload_role:
        if (
            parameter is not None
            and shard.layout is ParameterLayout.WHOLE_PARAMETER_OWNER
            and shard.local_member != shard.owner
        ):
            raise ValueError("whole-owner nonowners must not retain parameter storage")
        if parameter is not None:
            core = _parameter_state_core(optimizer, parameter, slot.compatibility_name)
            if frozenset(core) != {"name"}:
                raise ValueError("empty portable shards must remain pristine")
        return record
    if parameter is None:
        raise ValueError("a portable payload shard requires local parameter storage")

    core = _parameter_state_core(optimizer, parameter, slot.compatibility_name)
    keys = frozenset(core)
    if keys == {"name"}:
        return record
    if keys == {"name", "automatic_period"}:
        if type(core["automatic_period"]) is not int or core["automatic_period"] != 1:
            raise ValueError("exact portable period-selected state requires period one")
        record["source_period"] = 1
        record["state_variant"] = "period_selected"
        return record
    if common["gefen_codebook"] is None:
        raise ValueError("initialized portable state requires a codebook")
    if type(core.get("automatic_period")) is not int or core["automatic_period"] != 1:
        raise ValueError("exact portable initialized state requires period one")
    step = _strict_counter(core.get("step"), name="step", minimum=1)
    if step > common["gefen_global_step"]:
        raise ValueError("parameter step exceeds optimizer global step")
    dense_momentum = _decode_local_momentum(
        core,
        shard,
        optimizer._gefen_codebook,
    )
    record["source_period"] = 1

    if implementation == _PLAIN_IMPLEMENTATION:
        if options["second_moment_policy"] == "factored":
            if keys != _PLAIN_FACTORED_KEYS:
                raise ValueError("plain factored state is incomplete or contains inactive fields")
            if shard.layout is not ParameterLayout.REPLICATED:
                raise ValueError("factored state requires a replicated shard")
            factored_step = _strict_counter(core["factored_step"], name="factored_step", minimum=1)
            if factored_step > step:
                raise ValueError("factored_step exceeds parameter step")
            record.update(
                {
                    "source_second_moment": "factored",
                    "state_variant": "initialized_factored",
                    "state": {
                        "step": step,
                        "momentum": dense_momentum,
                        "v_row": _tight_cpu_fp32(
                            core["v_row"],
                            name="v_row",
                            shape=(shard.parameter.global_shape[0],),
                            nonnegative=True,
                        ),
                        "v_col": _tight_cpu_fp32(
                            core["v_col"],
                            name="v_col",
                            shape=(shard.parameter.global_shape[1],),
                            nonnegative=True,
                        ),
                        "factored_step": factored_step,
                    },
                }
            )
            return record

        if keys != _PLAIN_BLOCK_KEYS:
            raise ValueError("plain block state is incomplete or contains inactive fields")
        vmean_step = _strict_counter(core["vmean_step"], name="vmean_step", minimum=1)
        if vmean_step > step:
            raise ValueError("vmean_step exceeds parameter step")
        local_shape = _local_dense_shape(shard)
        vmean = _plain_tensor(
            core["vmean"],
            name="vmean",
            dtype=torch.float32,
            nonnegative=True,
        )
        second = _expand_block_second_moment(
            vmean,
            logical_shape=local_shape,
            period=1,
            step=vmean_step,
        )
        record.update(
            {
                "source_second_moment": "block",
                "state_variant": "initialized_dense",
                "state": {
                    "step": step,
                    "momentum": dense_momentum,
                    "second_moment": _tight_cpu_fp32(
                        second,
                        name="dense second moment",
                        shape=local_shape,
                        nonnegative=True,
                    ),
                    "second_moment_step": vmean_step,
                },
            }
        )
        return record

    expected = _NORMUON_KEYS if options["normuon"] else _MUON_KEYS
    if keys != expected:
        raise ValueError("Muon state is incomplete or conflicts with its NorMuon policy")
    state = {"step": step, "momentum": dense_momentum}
    variant = "initialized_dense"
    if options["normuon"]:
        normuon_step = _strict_counter(core["normuon_step"], name="normuon_step", minimum=1)
        if normuon_step > step:
            raise ValueError("normuon_step exceeds parameter step")
        state.update(
            {
                "normuon_v": _tight_cpu_fp32(
                    core["normuon_v"],
                    name="normuon_v",
                    shape=(shard.parameter.global_shape[0], 1),
                    nonnegative=True,
                ),
                "normuon_step": normuon_step,
            }
        )
        variant = "initialized_dense_normuon"
    record["state_variant"] = variant
    record["state"] = state
    return record


def _prepare_local_structure(
    optimizer,
    implementation: str,
    binding: CheckpointProcessGroupBinding,
    limits: PortableStateLimits,
    *,
    include_payload: bool,
):
    for name, value in (
        ("logical slots", optimizer._gefen_logical_slots),
        ("manifest shards", optimizer._gefen_sharding_manifest.shards),
        ("parameter groups", optimizer.param_groups),
    ):
        if type(value) not in {list, tuple}:
            continue
        if len(value) > limits.max_container_items or len(value) > limits.max_tree_nodes:
            raise ValueError("portable {} exceed structural limits".format(name))
    for group in optimizer.param_groups:
        if type(group) is not dict:
            continue
        for key in ("params", "param_names"):
            value = group.get(key)
            if type(value) is list and (len(value) > limits.max_container_items or len(value) > limits.max_tree_nodes):
                raise ValueError("portable group slots exceed structural limits")
    _validate_optimizer_shell(optimizer, implementation, binding)
    local_by_fqn = _live_maps(optimizer, binding)
    policy = _live_policy(optimizer, implementation)
    catalog, options_by_fqn = _catalog_and_options(optimizer, implementation, policy)
    common_preview = _preflight_common_state(
        optimizer,
        validate_values=False,
    )
    preview_slots = []
    for logical_slot in optimizer._gefen_logical_slots:
        fqn = logical_slot.shard.parameter.fqn
        parameter, shard = local_by_fqn[fqn]
        if shard != logical_slot.shard:
            raise ValueError("logical-slot and local shard identities disagree")
        preview_slots.append(
            _preflight_slot_record(
                optimizer,
                implementation,
                logical_slot,
                parameter,
                shard,
                options_by_fqn[fqn],
                common_preview,
            )
        )
    manifest = _serialize_sharding_manifest(optimizer._gefen_sharding_manifest)
    target_slots = [
        {
            "fqn": slot["shard"]["parameter"]["fqn"],
            "group_index": slot["group_index"],
            "original_slot_index": slot["original_slot_index"],
            "compatibility_name": slot["compatibility_name"],
        }
        for slot in preview_slots
    ]
    if include_payload:
        bounded_value = {
            "format": "gefen.portable_state_fragment",
            "format_version": 1,
            "coverage": "local_logical_optimizer_fragment",
            "implementation": implementation,
            "member": binding.local_member,
            "policy": policy,
            "common": common_preview,
            "manifest": manifest,
            "catalog": catalog,
            "logical_slots": preview_slots,
        }
    else:
        bounded_value = {
            "policy": policy,
            "manifest": manifest,
            "catalog": catalog,
            "logical_slots": target_slots,
        }
    _preflight_portable_value(bounded_value, limits)
    return {
        "local_by_fqn": local_by_fqn,
        "policy": policy,
        "catalog": catalog,
        "options_by_fqn": options_by_fqn,
        "target_descriptor": {
            "policy": policy,
            "manifest": manifest,
            "catalog": catalog,
            "logical_slots": target_slots,
        },
    }


def _validate_prepared_local_state(optimizer, implementation: str, prepared) -> None:
    common = _preflight_common_state(optimizer)
    local_by_fqn = prepared["local_by_fqn"]
    options_by_fqn = prepared["options_by_fqn"]
    for slot in optimizer._gefen_logical_slots:
        fqn = slot.shard.parameter.fqn
        parameter, shard = local_by_fqn[fqn]
        if shard != slot.shard:
            raise ValueError("logical-slot and local shard identities disagree")
        _validate_readiness_slot(
            optimizer,
            implementation,
            slot,
            parameter,
            shard,
            options_by_fqn[fqn],
            common,
        )


def _materialize_local_fragment(
    optimizer,
    implementation: str,
    binding: CheckpointProcessGroupBinding,
    limits: PortableStateLimits,
    prepared,
):
    local_by_fqn = prepared["local_by_fqn"]
    policy = prepared["policy"]
    catalog = prepared["catalog"]
    options_by_fqn = prepared["options_by_fqn"]
    common = _common_state(optimizer)
    slots = []
    for logical_slot in optimizer._gefen_logical_slots:
        fqn = logical_slot.shard.parameter.fqn
        parameter, shard = local_by_fqn[fqn]
        if shard != logical_slot.shard:
            raise ValueError("logical-slot and local shard identities disagree")
        slots.append(
            _slot_record(
                optimizer,
                implementation,
                logical_slot,
                parameter,
                shard,
                options_by_fqn[fqn],
                common,
            )
        )
    return _build_portable_state_fragment(
        implementation=implementation,
        member=binding.local_member,
        policy=policy,
        common=common,
        manifest=optimizer._gefen_sharding_manifest,
        catalog=catalog,
        logical_slots=slots,
        limits=limits,
    )


def _target_context(base, fragment):
    return {
        **base,
        "target_policy": fragment["policy"],
        "target_manifest": fragment["manifest"],
        "target_catalog": fragment["catalog"],
        "target_logical_slots": fragment["logical_slots"],
    }


def _export_portable_state(
    optimizer,
    *,
    checkpoint_process_group,
    transaction_id,
    limits,
):
    """Collectively export one complete exact portable v3 optimizer document."""

    transport_binding = _preflight_transport_binding(
        optimizer,
        checkpoint_process_group,
    )
    binding = None
    normalized_limits = None
    normalized_transaction = None
    implementation = None
    context = bytes(32)
    wire_limits = _STATUS_FALLBACK_LIMITS
    local_fragment = None
    live_token = None
    error = None
    try:
        _validate_supplied_binding(checkpoint_process_group, transport_binding)
        binding = transport_binding
        normalized_limits = _require_limits(limits)
        wire_limits = normalized_limits._wire_limits()
        normalized_transaction = _require_transaction_id(transaction_id)
        implementation = _optimizer_implementation(optimizer)
        _validate_context_identity_bounds(binding, normalized_limits)
        base_context = _base_context(binding, implementation)
        _preflight_portable_value(base_context, normalized_limits)
        context = _context_digest(base_context)
        _prepare_local_structure(
            optimizer,
            implementation,
            binding,
            normalized_limits,
            include_payload=True,
        )
        live_token = _portable_live_token(optimizer)
        prepared_local = _prepare_local_structure(
            optimizer,
            implementation,
            binding,
            normalized_limits,
            include_payload=True,
        )
        _validate_prepared_local_state(
            optimizer,
            implementation,
            prepared_local,
        )
        local_fragment = _materialize_local_fragment(
            optimizer,
            implementation,
            binding,
            normalized_limits,
            prepared_local,
        )
        if live_token != _portable_live_token(optimizer):
            raise RuntimeError("live optimizer state changed during portable export preparation")
    except Exception as exc:
        error = exc
    _collective_unanimous_status(
        transport_binding,
        error,
        operation="portable_export_prepare",
        transaction_id=_EXPORT_PREFLIGHT_TRANSACTION,
        context_digest=context,
        limits=wire_limits,
    )
    assert (
        normalized_limits is not None
        and normalized_transaction is not None
        and implementation is not None
        and binding is not None
        and local_fragment is not None
    )

    fragments = []
    _collective_visit_canonical_fragments(
        binding,
        local_fragment,
        operation="portable_export_fragments",
        transaction_id=normalized_transaction,
        context_digest=context,
        limits=wire_limits,
        consume=lambda _member, value: fragments.append(
            _normalize_portable_state_fragment(
                value,
                limits=normalized_limits,
            )
        ),
    )

    document = None
    error = None
    try:
        document = _assemble_portable_state_fragments(
            fragments,
            process_group_identity=binding.identity,
            limits=normalized_limits,
        )
        _validate_live_readiness(optimizer, implementation, binding)
        if live_token != _portable_live_token(optimizer):
            raise RuntimeError("live optimizer state changed during portable export")
    except Exception as exc:
        error = exc
    _collective_unanimous_status(
        binding,
        error,
        operation="portable_export_finalize",
        transaction_id=normalized_transaction,
        context_digest=context,
        limits=wire_limits,
    )
    assert document is not None
    return document


def _stage_portable_import(
    optimizer,
    implementation: str,
    binding: CheckpointProcessGroupBinding,
    limits: PortableStateLimits,
    document,
):
    _prepare_local_structure(
        optimizer,
        implementation,
        binding,
        limits,
        include_payload=False,
    )
    live_token = _portable_live_token(optimizer)
    prepared_local = _prepare_local_structure(
        optimizer,
        implementation,
        binding,
        limits,
        include_payload=False,
    )
    _validate_prepared_local_state(
        optimizer,
        implementation,
        prepared_local,
    )
    target_fragment = prepared_local["target_descriptor"]
    if not _values_equal(document["policy"], target_fragment["policy"]):
        raise ValueError("portable document policy does not match the target")
    common = document["common"]
    if set(document["parameters"]) != set(target_fragment["catalog"]):
        raise ValueError("portable parameter catalog does not match the target")

    local_by_fqn = _live_maps(optimizer, binding)
    canonical_parameters = {}
    for fqn, (parameter, shard) in local_by_fqn.items():
        if parameter is None:
            continue
        target_options = target_fragment["catalog"][fqn]["algorithm_options"]
        target_second = target_options["second_moment_policy"] if implementation == _PLAIN_IMPLEMENTATION else None
        projected = _project_portable_parameter_state(
            document["parameters"][fqn],
            shard,
            implementation=implementation,
            global_step=common["gefen_global_step"],
            codebook=common["gefen_codebook"],
            target_algorithm_options=target_options,
            target_second_moment=target_second,
        )
        canonical_parameters[fqn] = {"state": projected}

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
    # Portable state intentionally carries the source replica-determinism
    # setting. Native loading ordinarily rejects a policy change, so perform
    # the same complete staging path through an isolated owner configured with
    # the document value. The live optimizer remains untouched until commit.
    staging_owner = object.__new__(type(optimizer))
    staging_owner.__dict__ = optimizer.__dict__.copy()
    staging_owner._deterministic = common["gefen_deterministic"]
    staged = staging_owner._stage_load_state_dict(native)
    optimizer._preserve_canonical_target_configuration(staged)
    if (
        type(staged.__dict__) is not dict
        or type(staged.defaults) is not dict
        or set(staged.defaults) != set(optimizer.defaults)
        or staged.param_groups is not optimizer.param_groups
    ):
        raise TypeError("portable import staging produced unsafe publication containers")
    if live_token != _portable_live_token(optimizer):
        raise RuntimeError("live optimizer state changed during portable import preparation")
    return staged, live_token, target_fragment


def _import_portable_state(
    optimizer,
    state,
    *,
    checkpoint_process_group,
    transaction_id,
    limits,
) -> None:
    """Collectively stage, vote, and atomically publish portable v3 state."""

    transport_binding = _preflight_transport_binding(
        optimizer,
        checkpoint_process_group,
    )
    binding = None
    normalized_limits = None
    normalized_transaction = None
    implementation = None
    base = None
    base_digest = bytes(32)
    wire_limits = _STATUS_FALLBACK_LIMITS
    document = None
    error = None
    try:
        _validate_supplied_binding(checkpoint_process_group, transport_binding)
        binding = transport_binding
        normalized_limits = _require_limits(limits)
        wire_limits = normalized_limits._wire_limits()
        normalized_transaction = _require_transaction_id(transaction_id)
        implementation = _optimizer_implementation(optimizer)
        _validate_context_identity_bounds(binding, normalized_limits)
        base = _base_context(binding, implementation)
        _preflight_portable_value(base, normalized_limits)
        base_digest = _context_digest(base)
        document = _normalize_gefen_portable_state_document(
            state,
            limits=normalized_limits,
            expected_implementation=implementation,
        )
    except Exception as exc:
        error = exc
    _collective_unanimous_status(
        transport_binding,
        error,
        operation="portable_import_document",
        transaction_id=_IMPORT_PREFLIGHT_TRANSACTION,
        context_digest=base_digest,
        limits=wire_limits,
    )
    assert (
        normalized_limits is not None
        and normalized_transaction is not None
        and implementation is not None
        and binding is not None
        and base is not None
        and document is not None
    )

    staged = None
    live_token = None
    target_fragment = None
    error = None
    try:
        staged, live_token, target_fragment = _stage_portable_import(
            optimizer,
            implementation,
            binding,
            normalized_limits,
            document,
        )
        target_context = _target_context(base, target_fragment)
        target_context["document_digest"] = document["completion"]["digest"]
        _preflight_portable_value(target_context, normalized_limits)
        import_digest = _context_digest(target_context)
    except Exception as exc:
        error = exc
        import_digest = _context_digest({**base, "document_digest": document["completion"]["digest"]})
    _collective_unanimous_status(
        binding,
        error,
        operation="portable_import_prepare",
        transaction_id=normalized_transaction,
        context_digest=import_digest,
        limits=wire_limits,
    )
    assert staged is not None and live_token is not None and target_fragment is not None

    error = None
    try:
        _validate_live_readiness(optimizer, implementation, binding)
        if live_token != _portable_live_token(optimizer):
            raise RuntimeError("live optimizer state changed after portable import preparation")
    except Exception as exc:
        error = exc
    _collective_unanimous_status(
        binding,
        error,
        operation="portable_import_freshness",
        transaction_id=normalized_transaction,
        context_digest=import_digest,
        limits=wire_limits,
    )

    # This is the established non-throwing publication primitive. No collective
    # or semantic validation is allowed after this point.
    from gefen.gefen import Gefen

    Gefen._commit_staged_load_state_dict(optimizer, staged)


def _validate_readiness_slot(
    optimizer,
    implementation: str,
    slot: LogicalSlotBinding,
    parameter,
    shard: ShardIdentity,
    options,
    common,
    *,
    validate_values=True,
) -> None:
    role = _derived_role(shard)
    payload_role = role in {"live", "whole_owner"} and shard.parameter.numel > 0
    if not payload_role:
        if parameter is None:
            if shard.layout is not ParameterLayout.WHOLE_PARAMETER_OWNER:
                raise ValueError("only whole-parameter nonowners may omit storage")
            return
        core = _parameter_state_core(optimizer, parameter, slot.compatibility_name)
        if frozenset(core) != {"name"}:
            raise ValueError("empty portable shards must remain pristine")
        return
    if parameter is None:
        raise ValueError("a portable payload shard requires local storage")

    core = _parameter_state_core(optimizer, parameter, slot.compatibility_name)
    keys = frozenset(core)
    if keys == {"name"}:
        return
    if keys == {"name", "automatic_period"}:
        if type(core["automatic_period"]) is not int or core["automatic_period"] != 1:
            raise ValueError("exact portable period-selected state requires period one")
        if common["gefen_codebook"] is None:
            raise ValueError("period-selected portable state requires a codebook")
        return
    if common["gefen_codebook"] is None:
        raise ValueError("initialized portable state requires a codebook")
    if type(core.get("automatic_period")) is not int or core["automatic_period"] != 1:
        raise ValueError("exact portable initialized state requires period one")
    step = _strict_counter(core.get("step"), name="step", minimum=1)
    if step > common["gefen_global_step"]:
        raise ValueError("parameter step exceeds optimizer global step")
    local_shape = _local_dense_shape(shard)
    if local_shape is None:
        raise ValueError("initialized portable state requires a local payload")
    blocks = math.prod(local_shape)
    indices = _plain_tensor(
        core.get("m_codebook"),
        name="m_codebook",
        dtype=torch.uint8,
        shape=(blocks, 1),
        validate_values=validate_values,
    )
    magnitudes = _plain_tensor(
        core.get("m_magnitude"),
        name="m_magnitude",
        dtype=torch.float32,
        shape=(blocks, 1),
        nonnegative=True,
        validate_values=validate_values,
    )
    if indices.device != magnitudes.device:
        raise ValueError("momentum indices and magnitudes must share a device")
    if indices.device != parameter.device:
        raise ValueError("portable parameter state must share its parameter device")

    if implementation == _PLAIN_IMPLEMENTATION:
        if options["second_moment_policy"] == "factored":
            if keys != _PLAIN_FACTORED_KEYS:
                raise ValueError("plain factored state is incomplete or contains inactive fields")
            factored_step = _strict_counter(core["factored_step"], name="factored_step", minimum=1)
            if factored_step > step:
                raise ValueError("factored_step exceeds parameter step")
            rows, columns = shard.parameter.global_shape
            v_row = _plain_tensor(
                core["v_row"],
                name="v_row",
                dtype=torch.float32,
                shape=(rows,),
                nonnegative=True,
                validate_values=validate_values,
            )
            v_col = _plain_tensor(
                core["v_col"],
                name="v_col",
                dtype=torch.float32,
                shape=(columns,),
                nonnegative=True,
                validate_values=validate_values,
            )
            if v_row.device != parameter.device or v_col.device != parameter.device:
                raise ValueError("factored portable state must share its parameter device")
            return
        if keys != _PLAIN_BLOCK_KEYS:
            raise ValueError("plain block state is incomplete or contains inactive fields")
        vmean_step = _strict_counter(core["vmean_step"], name="vmean_step", minimum=1)
        if vmean_step > step:
            raise ValueError("vmean_step exceeds parameter step")
        vmean = _plain_tensor(
            core["vmean"],
            name="vmean",
            dtype=torch.float32,
            shape=(blocks, 1),
            nonnegative=True,
            validate_values=validate_values,
        )
        if vmean.device != parameter.device:
            raise ValueError("block portable state must share its parameter device")
        return

    expected = _NORMUON_KEYS if options["normuon"] else _MUON_KEYS
    if keys != expected:
        raise ValueError("Muon state is incomplete or conflicts with its NorMuon policy")
    if options["normuon"]:
        normuon_step = _strict_counter(core["normuon_step"], name="normuon_step", minimum=1)
        if normuon_step > step:
            raise ValueError("normuon_step exceeds parameter step")
        normuon_v = _plain_tensor(
            core["normuon_v"],
            name="normuon_v",
            dtype=torch.float32,
            shape=(shard.parameter.global_shape[0], 1),
            nonnegative=True,
            validate_values=validate_values,
        )
        if normuon_v.device != parameter.device:
            raise ValueError("NorMuon portable state must share its parameter device")


def _preflight_slot_record(
    optimizer,
    implementation: str,
    slot: LogicalSlotBinding,
    parameter,
    shard: ShardIdentity,
    options,
    common,
):
    _validate_readiness_slot(
        optimizer,
        implementation,
        slot,
        parameter,
        shard,
        options,
        common,
        validate_values=False,
    )
    role = _derived_role(shard)
    record = {
        "group_index": slot.group_index,
        "original_slot_index": slot.original_slot_index,
        "compatibility_name": slot.compatibility_name,
        "shard": _serialize_shard_identity(shard),
        "algorithm_options": options,
        "role": role,
        "source_period": None,
        "source_second_moment": None,
        "state_variant": "pristine",
        "state": {},
    }
    payload_role = role in {"live", "whole_owner"} and shard.parameter.numel > 0
    if not payload_role:
        return record
    core = _parameter_state_core(optimizer, parameter, slot.compatibility_name)
    keys = frozenset(core)
    if keys == {"name"}:
        return record
    record["source_period"] = 1
    if keys == {"name", "automatic_period"}:
        record["state_variant"] = "period_selected"
        return record

    local_shape = _local_dense_shape(shard)
    state = {
        "step": core["step"],
        "momentum": _PreflightTensor(local_shape),
    }
    if implementation == _PLAIN_IMPLEMENTATION:
        if options["second_moment_policy"] == "factored":
            rows, columns = shard.parameter.global_shape
            record.update(
                {
                    "source_second_moment": "factored",
                    "state_variant": "initialized_factored",
                    "state": {
                        **state,
                        "v_row": _PreflightTensor((rows,)),
                        "v_col": _PreflightTensor((columns,)),
                        "factored_step": core["factored_step"],
                    },
                }
            )
            return record
        record.update(
            {
                "source_second_moment": "block",
                "state_variant": "initialized_dense",
                "state": {
                    **state,
                    "second_moment": _PreflightTensor(local_shape),
                    "second_moment_step": core["vmean_step"],
                },
            }
        )
        return record

    if options["normuon"]:
        state.update(
            {
                "normuon_v": _PreflightTensor((shard.parameter.global_shape[0], 1)),
                "normuon_step": core["normuon_step"],
            }
        )
        record["state_variant"] = "initialized_dense_normuon"
    else:
        record["state_variant"] = "initialized_dense"
    record["state"] = state
    return record


def _validate_live_readiness(
    optimizer,
    implementation: str,
    binding: CheckpointProcessGroupBinding,
):
    _validate_optimizer_shell(optimizer, implementation, binding)
    local_by_fqn = _live_maps(optimizer, binding)
    policy = _live_policy(optimizer, implementation)
    _, options_by_fqn = _catalog_and_options(optimizer, implementation, policy)
    common = _preflight_common_state(optimizer)
    for slot in optimizer._gefen_logical_slots:
        parameter, shard = local_by_fqn[slot.shard.parameter.fqn]
        if shard != slot.shard:
            raise ValueError("logical-slot and local shard identities disagree")
        _validate_readiness_slot(
            optimizer,
            implementation,
            slot,
            parameter,
            shard,
            options_by_fqn[slot.shard.parameter.fqn],
            common,
        )
    return frozenset(slot.shard.layout for slot in optimizer._gefen_logical_slots)


def _portable_runtime_layouts(optimizer):
    """Return dynamically eligible layouts without executing a collective."""

    try:
        implementation = _optimizer_implementation(optimizer)
        scope = optimizer._gefen_codebook_process_group
        if type(scope) is not CodebookProcessGroupBinding:
            return frozenset()
        binding = CheckpointProcessGroupBinding(
            scope.identity,
            scope.local_member,
            scope.process_group,
            scope.collective_device,
        )
        return _validate_live_readiness(optimizer, implementation, binding)
    except Exception:
        return frozenset()


__all__ = []

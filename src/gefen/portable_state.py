"""Exact semantic normalization, assembly, and projection for portable Gefen state."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct

import torch

from gefen.contracts import (
    ParameterLayout,
    ProcessGroupIdentity,
    ShardIdentity,
    ShardingManifest,
)
from gefen.portable import (
    _expand_factored_second_moment_live_fp32_v1,
    _recompress_dense_momentum,
    _reduce_block_second_moment,
    _validate_codebook,
)
from gefen.portable_fields import _assemble_dense_logical_field, _project_dense_logical_field, _tensor_bits_equal
from gefen.portable_identity import (
    _normalize_parameter_identity,
    _normalize_shard_identity,
    _normalize_sharding_manifest,
    _parse_parameter_identity,
    _parse_shard_identity,
    _parse_sharding_manifest,
    _serialize_parameter_identity,
    _serialize_sharding_manifest,
)
from gefen.portable_schema import build_portable_state_document, normalize_portable_state_document
from gefen.portable_wire import (
    _CanonicalWireLimits,
    _parse_canonical_wire_metadata,
    _prepare_canonical_wire_value,
    _reconstruct_canonical_wire_value,
)


_IMPLEMENTATIONS = frozenset({"gefen.Gefen", "gefen.GefenMuon"})
_FRAGMENT_FORMAT = "gefen.portable_state_fragment"
_FRAGMENT_FORMAT_VERSION = 1
_FRAGMENT_COVERAGE = "local_logical_optimizer_fragment"
_MOMENTUM_PROJECTION = "dense_fp32_target_period_one_v1"
_SECOND_MOMENT_PROJECTION_EXACT = "exact_representation_target_period_one_v1"
_SECOND_MOMENT_PROJECTION_FACTORED_TO_BLOCK = "defined_projection_factored_to_block_live_fp32_target_period_one_v1"
_SECOND_MOMENT_PROJECTIONS = frozenset(
    {
        _SECOND_MOMENT_PROJECTION_EXACT,
        _SECOND_MOMENT_PROJECTION_FACTORED_TO_BLOCK,
    }
)
_MAX_EXACT_COUNTER = (1 << 53) - 1

_FRAGMENT_KEYS = frozenset(
    {
        "format",
        "format_version",
        "coverage",
        "implementation",
        "member",
        "policy",
        "common",
        "manifest",
        "catalog",
        "logical_slots",
    }
)
_CATALOG_KEYS = frozenset({"identity", "algorithm_options"})
_SLOT_KEYS = frozenset(
    {
        "group_index",
        "original_slot_index",
        "compatibility_name",
        "shard",
        "algorithm_options",
        "role",
        "source_period",
        "source_second_moment",
        "state_variant",
        "state",
    }
)
_COMMON_KEYS = frozenset({"gefen_global_step", "gefen_codebook", "gefen_deterministic"})
_POLICY_KEYS = frozenset(
    {
        "schema_version",
        "factored_v_2d",
        "force_1d_period_one",
        "force_2d_period_one",
        "period_one_substrings",
        "codebook_refresh_every",
        "stochastic_round",
        "momentum_projection",
        "second_moment_projection",
    }
)
_PLAIN_OPTION_KEYS = frozenset({"lr", "beta1", "beta2", "eps", "weight_decay", "second_moment_policy"})
_MUON_OPTION_KEYS = frozenset(
    {
        "lr",
        "weight_decay",
        "momentum",
        "nesterov",
        "ns_schedule",
        "ns_eps",
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
_PLAIN_HINT_KEYS = frozenset({"source_periods", "source_second_moment", "target_period"})
_MUON_HINT_KEYS = frozenset({"source_periods", "target_period"})
_PARAMETER_KEYS = frozenset({"identity", "algorithm_options", "state_variant", "state", "projection_hints"})


@dataclass(frozen=True, slots=True)
class PortableStateLimits:
    """Public resource limits shared by portable semantic and collective I/O."""

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
        self._wire_limits()

    def _wire_limits(self, *, collective: bool = False) -> _CanonicalWireLimits:
        return _CanonicalWireLimits(
            max_fragment_tensor_bytes=(
                self.max_collective_tensor_bytes if collective else self.max_fragment_tensor_bytes
            ),
            max_collective_tensor_bytes=self.max_collective_tensor_bytes,
            max_collective_metadata_bytes=self.max_collective_metadata_bytes,
            chunk_bytes=self.chunk_bytes,
            max_members=self.max_members,
            max_metadata_bytes=(self.max_collective_metadata_bytes if collective else self.max_metadata_bytes),
            max_tree_nodes=self.max_tree_nodes,
            max_tree_depth=self.max_tree_depth,
            max_container_items=self.max_container_items,
            max_string_bytes=self.max_string_bytes,
            max_integer_bytes=self.max_integer_bytes,
            max_tensors=self.max_tensors,
            max_tensor_rank=self.max_tensor_rank,
            diagnostic_bytes=self.diagnostic_bytes,
        )


def _require_limits(limits) -> PortableStateLimits:
    if type(limits) is not PortableStateLimits:
        raise TypeError("limits must be a PortableStateLimits")
    return limits


def _bounded_clone(value, limits: PortableStateLimits, *, collective: bool = False):
    wire_limits = limits._wire_limits(collective=collective)
    plan = _prepare_canonical_wire_value(value, wire_limits)
    prepared = _parse_canonical_wire_metadata(plan.metadata, limits=wire_limits)
    return _reconstruct_canonical_wire_value(
        prepared,
        plan.payload_tensors,
        expected_fragment_digest=plan.fragment_digest,
    )


def _exact_record(value, keys, *, name: str):
    if type(value) is not dict or set(value) != keys:
        raise ValueError("{} has an invalid schema".format(name))
    return value


def _strict_int(value, *, name: str, minimum: int = 0, maximum=None) -> int:
    if type(value) is not int:
        raise ValueError("{} must be an int".format(name))
    if value < minimum:
        raise ValueError("{} must be at least {}".format(name, minimum))
    if maximum is not None and value > maximum:
        raise ValueError("{} must be at most {}".format(name, maximum))
    return value


def _strict_float(value, *, name: str, minimum=None, maximum=None, maximum_open=False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError("{} must be a finite float".format(name))
    if minimum is not None and value < minimum:
        raise ValueError("{} is below its minimum".format(name))
    if maximum is not None and (value > maximum or (maximum_open and value == maximum)):
        raise ValueError("{} is above its maximum".format(name))
    return value


def _strict_name(value, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip() or "\x00" in value:
        raise ValueError("{} must be a non-empty canonical string".format(name))
    return value


def _float_bits_equal(left: float, right: float) -> bool:
    return struct.pack(">d", left) == struct.pack(">d", right)


def _values_equal(left, right) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is torch.Tensor:
        return left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape) and _tensor_bits_equal(left, right)
    if type(left) is float:
        return _float_bits_equal(left, right)
    if type(left) is dict:
        return set(left) == set(right) and all(_values_equal(left[key], right[key]) for key in left)
    if type(left) in {list, tuple}:
        return len(left) == len(right) and all(_values_equal(a, b) for a, b in zip(left, right))
    return left == right


def _tight_fp32(value, *, name: str, shape=None, nonnegative: bool = False) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or value.layout is not torch.strided
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or value.is_meta
        or value.is_nested
        or value.is_quantized
        or value.requires_grad
        or not value.is_contiguous()
        or value.storage_offset() != 0
        or value.untyped_storage().nbytes() != value.numel() * value.element_size()
    ):
        raise ValueError("{} must be a tight detached CPU fp32 tensor".format(name))
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError("{} has invalid shape".format(name))
    if not bool(torch.isfinite(value).all()):
        raise ValueError("{} must be finite".format(name))
    if nonnegative and not bool((value >= 0).all()):
        raise ValueError("{} must be nonnegative".format(name))
    return value


def _tight_clone(value: torch.Tensor) -> torch.Tensor:
    result = torch.empty(tuple(value.shape), dtype=torch.float32, device="cpu")
    result.copy_(value)
    return result


def _normalize_codebook(value, *, global_step: int):
    if value is None:
        return None
    value = _tight_fp32(value, name="common.gefen_codebook", shape=(256,))
    _validate_codebook(value)
    if float(value[0].item()) != -1.0 or float(value[-1].item()) != 1.0:
        raise ValueError("portable codebook must retain exact -1 and +1 endpoints")
    return value


def _normalize_common(value):
    value = _exact_record(value, _COMMON_KEYS, name="portable common state")
    global_step = _strict_int(value["gefen_global_step"], name="gefen_global_step", maximum=_MAX_EXACT_COUNTER)
    if type(value["gefen_deterministic"]) is not bool:
        raise ValueError("gefen_deterministic must be a bool")
    return {
        "gefen_global_step": global_step,
        "gefen_codebook": _normalize_codebook(value["gefen_codebook"], global_step=global_step),
        "gefen_deterministic": value["gefen_deterministic"],
    }


def _normalize_policy(value, implementation: str):
    value = _exact_record(value, _POLICY_KEYS, name="portable policy")
    if value["schema_version"] != 1 or type(value["schema_version"]) is not int:
        raise ValueError("unsupported portable semantic policy schema_version")
    _strict_int(value["codebook_refresh_every"], name="codebook_refresh_every")
    if value["stochastic_round"] is not False:
        raise ValueError("portable semantic state requires stochastic_round=False")
    if value["momentum_projection"] != _MOMENTUM_PROJECTION:
        raise ValueError("unsupported portable momentum projection")
    if value["second_moment_projection"] not in _SECOND_MOMENT_PROJECTIONS:
        raise ValueError("unsupported portable second-moment projection")
    if (
        implementation == "gefen.GefenMuon"
        and value["second_moment_projection"] != _SECOND_MOMENT_PROJECTION_EXACT
    ):
        raise ValueError("Muon portable policy requires exact second-moment projection")
    for key in ("factored_v_2d", "force_1d_period_one", "force_2d_period_one"):
        if type(value[key]) is not bool:
            raise ValueError("{} must be a bool".format(key))
    if implementation == "gefen.GefenMuon" and value["factored_v_2d"]:
        raise ValueError("Muon portable policy requires factored_v_2d=False")
    substrings = value["period_one_substrings"]
    if type(substrings) is not list or any(type(item) is not str or item != item.lower() for item in substrings):
        raise ValueError("period_one_substrings must be a canonical lowercase string list")
    return {**value, "period_one_substrings": list(substrings)}


def _validate_portable_projection_policy(source_policy, target_policy, implementation: str):
    """Validate exact policy compatibility plus the defined one-way migration."""

    source = _normalize_policy(source_policy, implementation)
    target = _normalize_policy(target_policy, implementation)
    invariant_keys = _POLICY_KEYS - {"factored_v_2d", "second_moment_projection"}
    if any(not _values_equal(source[key], target[key]) for key in invariant_keys):
        raise ValueError("portable document policy does not match the target")
    source_factored = source["factored_v_2d"]
    target_factored = target["factored_v_2d"]
    if source_factored == target_factored:
        return
    if implementation != "gefen.Gefen":
        raise ValueError("portable document policy does not match the target")
    if not source_factored and target_factored:
        raise ValueError("portable block-to-factored second-moment migration is unsupported")
    if source["second_moment_projection"] != _SECOND_MOMENT_PROJECTION_FACTORED_TO_BLOCK:
        raise ValueError("portable source policy does not authorize factored-to-block second-moment projection")


def _normalize_ns_schedule(value):
    if type(value) is not list or not value or len(value) >= 100:
        raise ValueError("ns_schedule must contain between 1 and 99 entries")
    if any(type(item) is not list or len(item) != 3 for item in value):
        raise ValueError("ns_schedule must contain three-float lists")
    return [[_strict_float(component, name="ns_schedule") for component in item] for item in value]


def _normalize_options(value, implementation: str):
    keys = _PLAIN_OPTION_KEYS if implementation == "gefen.Gefen" else _MUON_OPTION_KEYS
    value = _exact_record(value, keys, name="portable algorithm options")
    result = dict(value)
    result["lr"] = _strict_float(value["lr"], name="lr", minimum=0.0)
    result["weight_decay"] = _strict_float(value["weight_decay"], name="weight_decay", minimum=0.0)
    if implementation == "gefen.Gefen":
        result["beta1"] = _strict_float(value["beta1"], name="beta1", minimum=0.0, maximum=1.0, maximum_open=True)
        result["beta2"] = _strict_float(value["beta2"], name="beta2", minimum=0.0, maximum=1.0, maximum_open=True)
        result["eps"] = _strict_float(value["eps"], name="eps", minimum=0.0)
        if result["eps"] == 0.0:
            raise ValueError("eps must be positive")
        if value["second_moment_policy"] not in {"block", "factored"}:
            raise ValueError("second_moment_policy must be 'block' or 'factored'")
        return result

    result["momentum"] = _strict_float(value["momentum"], name="momentum", minimum=0.0, maximum=1.0, maximum_open=True)
    for key in ("nesterov", "fp8_ns", "fp8_ns_compile", "batched_ns", "normuon", "cautious"):
        if type(value[key]) is not bool:
            raise ValueError("{} must be a bool".format(key))
    result["ns_schedule"] = _normalize_ns_schedule(value["ns_schedule"])
    result["ns_eps"] = _strict_float(value["ns_eps"], name="ns_eps", minimum=0.0)
    if result["ns_eps"] == 0.0:
        raise ValueError("ns_eps must be positive")
    if value["adjust_lr_fn"] not in {None, "original", "match_rms_adamw"}:
        raise ValueError("adjust_lr_fn is unsupported")
    if value["sharded_mode"] not in {"exact", "approx", "distributed"}:
        raise ValueError("sharded_mode is unsupported")
    result["batched_ns_workspace_bytes"] = _strict_int(
        value["batched_ns_workspace_bytes"], name="batched_ns_workspace_bytes", minimum=1
    )
    result["normuon_beta2"] = _strict_float(
        value["normuon_beta2"], name="normuon_beta2", minimum=0.0, maximum=1.0, maximum_open=True
    )
    result["normuon_eps"] = _strict_float(value["normuon_eps"], name="normuon_eps", minimum=0.0)
    if result["normuon_eps"] == 0.0:
        raise ValueError("normuon_eps must be positive")
    return result


def _normalize_periods(value, *, name: str):
    if type(value) is not list or any(type(period) is not int or period <= 0 for period in value):
        raise ValueError("{} must be a list of positive ints".format(name))
    if value != sorted(set(value)):
        raise ValueError("{} must be sorted and unique".format(name))
    if any(period != 1 for period in value):
        raise ValueError("exact portable v3 state supports only source period one")
    return list(value)


def _normalize_complete_parameter_record(fqn, value, implementation: str, global_step: int, policy):
    value = _exact_record(value, _PARAMETER_KEYS, name="portable parameter {!r}".format(fqn))
    identity_record = _normalize_parameter_identity(value["identity"])
    identity = _parse_parameter_identity(identity_record)
    if identity.fqn != fqn:
        raise ValueError("portable parameter identity does not match its FQN key")
    options = _normalize_options(value["algorithm_options"], implementation)
    variant = value["state_variant"]
    if type(variant) is not str:
        raise ValueError("portable state_variant must be a string")
    state = value["state"]
    hints = value["projection_hints"]
    if type(state) is not dict or type(hints) is not dict:
        raise ValueError("portable parameter state and projection_hints must be dictionaries")
    shape = identity.global_shape
    if identity.numel == 0 and variant != "pristine":
        raise ValueError("empty logical parameters must remain pristine")

    if implementation == "gefen.Gefen":
        _exact_record(hints, _PLAIN_HINT_KEYS, name="plain projection_hints")
        periods = _normalize_periods(hints["source_periods"], name="source_periods")
        if hints["target_period"] != 1 or type(hints["target_period"]) is not int:
            raise ValueError("portable target_period must be exactly one")
        if hints["source_second_moment"] not in {None, "block", "factored"}:
            raise ValueError("portable source_second_moment is invalid")
        configured_factored = policy["factored_v_2d"] and len(shape) == 2
        representation = options["second_moment_policy"]
        if representation == "factored" and not configured_factored:
            raise ValueError("parameter second_moment_policy conflicts with the optimizer policy")
        if variant == "pristine":
            expected_keys = frozenset()
            if periods or hints["source_second_moment"] is not None:
                raise ValueError("pristine portable state has invalid projection hints")
        elif variant == "period_selected":
            expected_keys = frozenset()
            if not periods or hints["source_second_moment"] is not None:
                raise ValueError("period-selected portable state has invalid projection hints")
        elif variant == "initialized_dense":
            expected_keys = frozenset({"step", "momentum", "second_moment", "second_moment_step"})
            if not periods or hints["source_second_moment"] != "block" or representation != "block":
                raise ValueError("dense block state conflicts with its policy or projection hints")
        elif variant == "initialized_factored":
            expected_keys = frozenset({"step", "momentum", "v_row", "v_col", "factored_step"})
            if not periods or hints["source_second_moment"] != "factored" or representation != "factored":
                raise ValueError("factored state conflicts with its policy or projection hints")
        else:
            raise ValueError("unsupported plain portable state_variant")
        _exact_record(state, expected_keys, name="plain portable parameter state")
        normalized_state = dict(state)
        if variant.startswith("initialized_"):
            if identity.numel == 0:
                raise ValueError("empty logical parameters cannot carry initialized state")
            step = _strict_int(state["step"], name="step", minimum=1, maximum=_MAX_EXACT_COUNTER)
            if step > global_step:
                raise ValueError("parameter step exceeds optimizer global step")
            normalized_state["momentum"] = _tight_fp32(state["momentum"], name="momentum", shape=shape)
            if variant == "initialized_dense":
                second_step = _strict_int(
                    state["second_moment_step"], name="second_moment_step", minimum=1, maximum=_MAX_EXACT_COUNTER
                )
                normalized_state["second_moment"] = _tight_fp32(
                    state["second_moment"], name="second_moment", shape=shape, nonnegative=True
                )
            else:
                if len(shape) != 2:
                    raise ValueError("factored portable state requires a logical matrix")
                second_step = _strict_int(
                    state["factored_step"], name="factored_step", minimum=1, maximum=_MAX_EXACT_COUNTER
                )
                normalized_state["v_row"] = _tight_fp32(
                    state["v_row"], name="v_row", shape=(shape[0],), nonnegative=True
                )
                normalized_state["v_col"] = _tight_fp32(
                    state["v_col"], name="v_col", shape=(shape[1],), nonnegative=True
                )
            if second_step > step:
                raise ValueError("secondary parameter counter exceeds step")
        return {
            "identity": identity_record,
            "algorithm_options": options,
            "state_variant": variant,
            "state": normalized_state,
            "projection_hints": {
                "source_periods": periods,
                "source_second_moment": hints["source_second_moment"],
                "target_period": 1,
            },
        }

    _exact_record(hints, _MUON_HINT_KEYS, name="Muon projection_hints")
    periods = _normalize_periods(hints["source_periods"], name="source_periods")
    if hints["target_period"] != 1 or type(hints["target_period"]) is not int:
        raise ValueError("portable target_period must be exactly one")
    if len(shape) != 2:
        raise ValueError("portable Muon parameters must be logical matrices")
    if variant == "pristine":
        expected_keys = frozenset()
        if periods:
            raise ValueError("pristine Muon state must not contain source periods")
    elif variant == "period_selected":
        expected_keys = frozenset()
        if not periods:
            raise ValueError("period-selected Muon state requires source periods")
    elif variant == "initialized_dense":
        expected_keys = frozenset({"step", "momentum"})
        if options["normuon"] or not periods:
            raise ValueError("initialized Muon state conflicts with its options or periods")
    elif variant == "initialized_dense_normuon":
        expected_keys = frozenset({"step", "momentum", "normuon_v", "normuon_step"})
        if not options["normuon"] or not periods:
            raise ValueError("initialized NorMuon state conflicts with its options or periods")
    else:
        raise ValueError("unsupported Muon portable state_variant")
    _exact_record(state, expected_keys, name="Muon portable parameter state")
    normalized_state = dict(state)
    if variant.startswith("initialized_"):
        if identity.numel == 0:
            raise ValueError("empty logical parameters cannot carry initialized state")
        step = _strict_int(state["step"], name="step", minimum=1, maximum=_MAX_EXACT_COUNTER)
        if step > global_step:
            raise ValueError("parameter step exceeds optimizer global step")
        normalized_state["momentum"] = _tight_fp32(state["momentum"], name="momentum", shape=shape)
        if variant == "initialized_dense_normuon":
            normuon_step = _strict_int(
                state["normuon_step"], name="normuon_step", minimum=1, maximum=_MAX_EXACT_COUNTER
            )
            if normuon_step > step:
                raise ValueError("normuon_step exceeds parameter step")
            normalized_state["normuon_v"] = _tight_fp32(
                state["normuon_v"], name="normuon_v", shape=(shape[0], 1), nonnegative=True
            )
    return {
        "identity": identity_record,
        "algorithm_options": options,
        "state_variant": variant,
        "state": normalized_state,
        "projection_hints": {"source_periods": periods, "target_period": 1},
    }


def _normalize_gefen_portable_state_document(state, *, limits, expected_implementation=None):
    """Bound, clone, and semantically validate one complete portable v3 document."""

    limits = _require_limits(limits)
    if expected_implementation is not None and expected_implementation not in _IMPLEMENTATIONS:
        raise ValueError("unsupported expected portable implementation")
    bounded = _bounded_clone(state, limits, collective=True)
    document = normalize_portable_state_document(bounded, expected_implementation=expected_implementation)
    implementation = document["implementation"]
    if implementation not in _IMPLEMENTATIONS:
        raise ValueError("unsupported portable Gefen implementation")
    if document["provenance"] is not None:
        raise ValueError("Gefen portable v3 provenance must be None")
    policy = _normalize_policy(document["policy"], implementation)
    common = _normalize_common(document["common"])
    parameters = document["parameters"]
    if type(parameters) is not dict or not parameters:
        raise ValueError("portable parameters must be a non-empty FQN mapping")
    normalized_parameters = {}
    for fqn in sorted(parameters):
        _strict_name(fqn, name="parameter FQN")
        normalized_parameters[fqn] = _normalize_complete_parameter_record(
            fqn,
            parameters[fqn],
            implementation,
            common["gefen_global_step"],
            policy,
        )
    if common["gefen_codebook"] is None and any(
        record["state_variant"] != "pristine" for record in normalized_parameters.values()
    ):
        raise ValueError("non-pristine portable parameter state requires a codebook")
    normalized = build_portable_state_document(
        implementation=implementation,
        policy=policy,
        common=common,
        parameters=normalized_parameters,
        provenance=None,
    )
    if not _values_equal(normalized, document):
        raise ValueError("portable state is not in canonical semantic form")
    return normalized


def _derived_role(shard: ShardIdentity) -> str:
    if shard.layout is ParameterLayout.REPLICATED:
        return "live" if shard.parameter.numel else "empty_replicated"
    if shard.layout is ParameterLayout.FLATTENED_ELEMENT_SHARD:
        return "live" if shard.logical_slice.length else "empty_flat"
    if shard.layout is ParameterLayout.WHOLE_PARAMETER_OWNER:
        return "whole_owner" if shard.local_member == shard.owner else "whole_nonowner"
    if shard.layout is ParameterLayout.DTENSOR_1D_DEFAULT_WORLD:
        return "live" if shard.logical_region.numel else "empty_dtensor"
    raise ValueError("unsupported portable fragment layout")


def _local_dense_shape(shard: ShardIdentity):
    role = _derived_role(shard)
    if role not in {"live", "whole_owner"} or shard.parameter.numel == 0:
        return None
    if shard.layout in {ParameterLayout.REPLICATED, ParameterLayout.WHOLE_PARAMETER_OWNER}:
        return shard.parameter.global_shape
    if shard.layout is ParameterLayout.FLATTENED_ELEMENT_SHARD:
        return (shard.logical_slice.length,)
    if shard.layout is ParameterLayout.DTENSOR_1D_DEFAULT_WORLD:
        return shard.logical_region.lengths
    raise ValueError("unsupported portable fragment layout")


def _normalize_catalog(value, implementation: str, manifest: ShardingManifest, policy):
    if type(value) is not dict:
        raise ValueError("portable fragment catalog must be an FQN mapping")
    manifest_fqns = {shard.parameter.fqn for shard in manifest.shards}
    if set(value) != manifest_fqns:
        raise ValueError("portable fragment catalog does not match the manifest")
    result = {}
    for fqn in sorted(value):
        record = _exact_record(value[fqn], _CATALOG_KEYS, name="portable catalog entry")
        identity_record = _normalize_parameter_identity(record["identity"])
        identity = _parse_parameter_identity(identity_record)
        if identity.fqn != fqn or any(shard.parameter != identity for shard in manifest.for_parameter(fqn)):
            raise ValueError("portable catalog identity does not match the manifest")
        options = _normalize_options(record["algorithm_options"], implementation)
        if implementation == "gefen.Gefen":
            configured_factored = policy["factored_v_2d"] and len(identity.global_shape) == 2
            parameter_shards = manifest.for_parameter(fqn)
            layouts = {shard.layout for shard in parameter_shards}
            if configured_factored and layouts == {ParameterLayout.REPLICATED}:
                expected = "factored"
            elif configured_factored and layouts == {ParameterLayout.DTENSOR_1D_DEFAULT_WORLD}:
                expected = "block"
            elif configured_factored:
                raise ValueError("factored logical matrices require replicated storage or the DTensor block fallback")
            else:
                expected = "block"
            if options["second_moment_policy"] != expected:
                raise ValueError("catalog second_moment_policy conflicts with the optimizer policy")
            if expected == "factored" and any(
                shard.layout is not ParameterLayout.REPLICATED for shard in manifest.for_parameter(fqn)
            ):
                raise ValueError("factored logical matrices require replicated portable shards")
        elif len(identity.global_shape) != 2:
            raise ValueError("portable Muon catalog parameters must be logical matrices")
        result[fqn] = {
            "identity": identity_record,
            "algorithm_options": options,
        }
    return result


def _normalize_fragment_slot(value, implementation: str, common, catalog, member: str):
    value = _exact_record(value, _SLOT_KEYS, name="portable logical slot")
    group_index = _strict_int(value["group_index"], name="group_index")
    slot_index = _strict_int(value["original_slot_index"], name="original_slot_index")
    compatibility_name = _strict_name(value["compatibility_name"], name="compatibility_name")
    if compatibility_name != compatibility_name.lower():
        raise ValueError("compatibility_name must be lowercase")
    shard_record = _normalize_shard_identity(value["shard"])
    shard = _parse_shard_identity(shard_record)
    allowed_layouts = (
        {
            ParameterLayout.REPLICATED,
            ParameterLayout.FLATTENED_ELEMENT_SHARD,
            ParameterLayout.DTENSOR_1D_DEFAULT_WORLD,
        }
        if implementation == "gefen.Gefen"
        else {ParameterLayout.REPLICATED, ParameterLayout.WHOLE_PARAMETER_OWNER}
    )
    if shard.layout not in allowed_layouts:
        raise ValueError("portable fragment layout is unsupported for its implementation")
    if shard.process_group is None or shard.local_member != member:
        raise ValueError("portable fragment member does not match its shard")
    role = _derived_role(shard)
    if value["role"] != role:
        raise ValueError("portable fragment role does not match its shard")
    options = _normalize_options(value["algorithm_options"], implementation)
    catalog_entry = catalog.get(shard.parameter.fqn)
    if catalog_entry is None or not _values_equal(options, catalog_entry["algorithm_options"]):
        raise ValueError("portable slot options do not match the catalog")
    if (
        implementation == "gefen.GefenMuon"
        and shard.layout is ParameterLayout.WHOLE_PARAMETER_OWNER
        and options["sharded_mode"] != "distributed"
    ):
        raise ValueError("whole-parameter Muon state requires sharded_mode='distributed'")
    variant = value["state_variant"]
    if variant not in {
        "pristine",
        "period_selected",
        "initialized_dense",
        "initialized_factored",
        "initialized_dense_normuon",
    }:
        raise ValueError("portable fragment state_variant is unsupported")
    period = value["source_period"]
    if period is not None:
        _strict_int(period, name="source_period", minimum=1)
        if period != 1:
            raise ValueError("exact portable v3 fragments require source_period=1")
        local_shape = _local_dense_shape(shard)
        if local_shape is None or math.prod(local_shape) % period != 0:
            raise ValueError("source_period does not divide the local logical payload")
    source_second = value["source_second_moment"]
    if source_second not in {None, "block", "factored"}:
        raise ValueError("portable source_second_moment is invalid")
    state = value["state"]
    if type(state) is not dict:
        raise ValueError("portable fragment slot state must be a dictionary")
    payload_shape = _local_dense_shape(shard)
    payload_role = role in {"live", "whole_owner"} and shard.parameter.numel > 0
    if not payload_role and (variant != "pristine" or period is not None or state or source_second is not None):
        raise ValueError("empty and whole-nonowner slots must remain pristine")
    if variant == "pristine":
        _exact_record(state, frozenset(), name="pristine fragment state")
        if period is not None or source_second is not None:
            raise ValueError("pristine fragment state has invalid metadata")
    elif variant == "period_selected":
        _exact_record(state, frozenset(), name="period-selected fragment state")
        if period is None or source_second is not None:
            raise ValueError("period-selected fragment state has invalid metadata")
    elif implementation == "gefen.Gefen" and variant == "initialized_dense":
        _exact_record(
            state, frozenset({"step", "momentum", "second_moment", "second_moment_step"}), name="dense fragment state"
        )
        if period is None or source_second != "block":
            raise ValueError("initialized block fragment has invalid metadata")
        if options["second_moment_policy"] != "block":
            raise ValueError("initialized block fragment conflicts with its algorithm options")
    elif implementation == "gefen.Gefen" and variant == "initialized_factored":
        _exact_record(
            state, frozenset({"step", "momentum", "v_row", "v_col", "factored_step"}), name="factored fragment state"
        )
        if period is None or source_second != "factored" or shard.layout is not ParameterLayout.REPLICATED:
            raise ValueError("factored fragments require replicated storage and factored metadata")
        if options["second_moment_policy"] != "factored":
            raise ValueError("initialized factored fragment conflicts with its algorithm options")
    elif implementation == "gefen.GefenMuon" and variant in {"initialized_dense", "initialized_dense_normuon"}:
        keys = {"step", "momentum"}
        if variant == "initialized_dense_normuon":
            keys.update({"normuon_v", "normuon_step"})
        _exact_record(state, frozenset(keys), name="Muon fragment state")
        if period is None or source_second is not None:
            raise ValueError("initialized Muon fragment has invalid metadata")
        if (variant == "initialized_dense_normuon") != options["normuon"]:
            raise ValueError("initialized Muon fragment conflicts with its NorMuon option")
    else:
        raise ValueError("fragment state_variant does not match its implementation")
    normalized_state = dict(state)
    if variant.startswith("initialized_"):
        step = _strict_int(state["step"], name="step", minimum=1, maximum=_MAX_EXACT_COUNTER)
        if step > common["gefen_global_step"]:
            raise ValueError("fragment parameter step exceeds optimizer global step")
        normalized_state["momentum"] = _tight_fp32(state["momentum"], name="local momentum", shape=payload_shape)
        if variant == "initialized_dense" and implementation == "gefen.Gefen":
            second_step = _strict_int(
                state["second_moment_step"], name="second_moment_step", minimum=1, maximum=_MAX_EXACT_COUNTER
            )
            normalized_state["second_moment"] = _tight_fp32(
                state["second_moment"], name="local second_moment", shape=payload_shape, nonnegative=True
            )
            if second_step > step:
                raise ValueError("fragment second_moment_step exceeds step")
        elif variant == "initialized_factored":
            rows, columns = shard.parameter.global_shape
            factored_step = _strict_int(
                state["factored_step"], name="factored_step", minimum=1, maximum=_MAX_EXACT_COUNTER
            )
            normalized_state["v_row"] = _tight_fp32(state["v_row"], name="v_row", shape=(rows,), nonnegative=True)
            normalized_state["v_col"] = _tight_fp32(state["v_col"], name="v_col", shape=(columns,), nonnegative=True)
            if factored_step > step:
                raise ValueError("fragment factored_step exceeds step")
        elif variant == "initialized_dense_normuon":
            normuon_step = _strict_int(
                state["normuon_step"], name="normuon_step", minimum=1, maximum=_MAX_EXACT_COUNTER
            )
            normalized_state["normuon_v"] = _tight_fp32(
                state["normuon_v"], name="normuon_v", shape=(shard.parameter.global_shape[0], 1), nonnegative=True
            )
            if normuon_step > step:
                raise ValueError("fragment normuon_step exceeds step")
    return {
        "group_index": group_index,
        "original_slot_index": slot_index,
        "compatibility_name": compatibility_name,
        "shard": shard_record,
        "algorithm_options": options,
        "role": role,
        "source_period": period,
        "source_second_moment": source_second,
        "state_variant": variant,
        "state": normalized_state,
    }


def _normalize_portable_state_fragment(fragment, *, limits):
    """Bound, clone, and validate one reversible local portable-state fragment."""

    limits = _require_limits(limits)
    fragment = _bounded_clone(fragment, limits)
    _exact_record(fragment, _FRAGMENT_KEYS, name="portable state fragment")
    if fragment["format"] != _FRAGMENT_FORMAT or fragment["format_version"] != _FRAGMENT_FORMAT_VERSION:
        raise ValueError("unsupported portable state fragment format")
    if type(fragment["format_version"]) is not int or fragment["coverage"] != _FRAGMENT_COVERAGE:
        raise ValueError("unsupported portable state fragment coverage")
    implementation = fragment["implementation"]
    if implementation not in _IMPLEMENTATIONS:
        raise ValueError("unsupported portable fragment implementation")
    member = _strict_name(fragment["member"], name="portable fragment member")
    policy = _normalize_policy(fragment["policy"], implementation)
    common = _normalize_common(fragment["common"])
    manifest_record = _normalize_sharding_manifest(fragment["manifest"])
    manifest = _parse_sharding_manifest(manifest_record)
    allowed_layouts = (
        {
            ParameterLayout.REPLICATED,
            ParameterLayout.FLATTENED_ELEMENT_SHARD,
            ParameterLayout.DTENSOR_1D_DEFAULT_WORLD,
        }
        if implementation == "gefen.Gefen"
        else {ParameterLayout.REPLICATED, ParameterLayout.WHOLE_PARAMETER_OWNER}
    )
    if any(shard.layout not in allowed_layouts or shard.process_group is None for shard in manifest.shards):
        raise ValueError("portable fragment manifest contains an unsupported or ungrouped shard")
    catalog = _normalize_catalog(fragment["catalog"], implementation, manifest, policy)
    slots_value = fragment["logical_slots"]
    if type(slots_value) is not list or not slots_value:
        raise ValueError("portable fragment logical_slots must be a non-empty list")
    slots = [_normalize_fragment_slot(slot, implementation, common, catalog, member) for slot in slots_value]
    if common["gefen_codebook"] is None and any(slot["state_variant"] != "pristine" for slot in slots):
        raise ValueError("non-pristine portable fragment state requires a codebook")
    positions = [(slot["group_index"], slot["original_slot_index"]) for slot in slots]
    if positions[0] != (0, 0) or positions != sorted(positions) or len(set(positions)) != len(positions):
        raise ValueError("portable logical slot positions must start at (0, 0) and be strictly ordered")
    expected_group = 0
    expected_slot = 0
    for group_index, slot_index in positions:
        if group_index == expected_group and slot_index == expected_slot:
            expected_slot += 1
        elif group_index == expected_group + 1 and slot_index == 0:
            expected_group += 1
            expected_slot = 1
        else:
            raise ValueError("portable logical slot positions must be contiguous")
    identities = [_parse_shard_identity(slot["shard"]) for slot in slots]
    if len(set(identities)) != len(identities):
        raise ValueError("portable fragment contains duplicate shard identities")
    if any(shard not in manifest.shards for shard in identities):
        raise ValueError("portable fragment slot is absent from the manifest")
    return {
        "format": _FRAGMENT_FORMAT,
        "format_version": _FRAGMENT_FORMAT_VERSION,
        "coverage": _FRAGMENT_COVERAGE,
        "implementation": implementation,
        "member": member,
        "policy": policy,
        "common": common,
        "manifest": manifest_record,
        "catalog": catalog,
        "logical_slots": slots,
    }


def _build_portable_state_fragment(*, implementation, member, policy, common, manifest, catalog, logical_slots, limits):
    """Build one strict local fragment from serialized primitives and local dense fields."""

    if isinstance(manifest, ShardingManifest):
        manifest = _serialize_sharding_manifest(manifest)
    fragment = {
        "format": _FRAGMENT_FORMAT,
        "format_version": _FRAGMENT_FORMAT_VERSION,
        "coverage": _FRAGMENT_COVERAGE,
        "implementation": implementation,
        "member": member,
        "policy": policy,
        "common": common,
        "manifest": manifest,
        "catalog": catalog,
        "logical_slots": logical_slots,
    }
    return _normalize_portable_state_fragment(fragment, limits=limits)


def _required_payload_slots(slots):
    return [
        slot
        for slot in slots
        if slot["role"] in {"live", "whole_owner"} and _parse_shard_identity(slot["shard"]).parameter.numel > 0
    ]


def _consensus(values, *, name: str):
    if not values:
        raise ValueError("{} requires at least one value".format(name))
    reference = values[0]
    if any(not _values_equal(reference, value) for value in values[1:]):
        raise ValueError("{} disagree across portable fragments".format(name))
    return reference


def _assemble_special_field(slots, *, key: str, shape, name: str) -> torch.Tensor:
    values = []
    for slot in slots:
        shard = _parse_shard_identity(slot["shard"])
        payload = slot["state"].get(key)
        if shard.layout is ParameterLayout.REPLICATED:
            values.append(_tight_fp32(payload, name=name, shape=shape, nonnegative=True))
        elif shard.layout is ParameterLayout.WHOLE_PARAMETER_OWNER:
            if shard.local_member == shard.owner:
                values.append(_tight_fp32(payload, name=name, shape=shape, nonnegative=True))
            elif payload is not None:
                raise ValueError("whole-parameter nonowner carries {}".format(name))
        else:
            raise ValueError("{} supports only exact replicas or a sole whole owner".format(name))
    return _tight_clone(_consensus(values, name=name))


def _assemble_parameter(fqn, slots, manifest: ShardingManifest, catalog_entry, implementation, policy, global_step):
    parameter = _parse_parameter_identity(catalog_entry["identity"])
    required = _required_payload_slots(slots)
    variants = {slot["state_variant"] for slot in required}
    if not required:
        variant = "pristine"
    elif variants == {"pristine"}:
        variant = "pristine"
    elif variants == {"period_selected"}:
        variant = "period_selected"
    elif len(variants) == 1 and next(iter(variants)).startswith("initialized_"):
        variant = next(iter(variants))
    else:
        raise ValueError("portable parameter shards disagree on initialization state")
    if any(slot["state_variant"] != "pristine" for slot in slots if slot not in required):
        raise ValueError("non-payload portable slots must remain pristine")
    periods = sorted({slot["source_period"] for slot in required if slot["source_period"] is not None})
    if variant == "pristine" and periods:
        raise ValueError("pristine portable parameter has source periods")
    if variant != "pristine" and not periods:
        raise ValueError("non-pristine portable parameter requires source periods")
    state = {}
    source_second = None
    if variant.startswith("initialized_"):
        step = _consensus([slot["state"]["step"] for slot in required], name="parameter step")
        if step > global_step:
            raise ValueError("parameter step exceeds optimizer global step")
        momentum = _assemble_dense_logical_field(
            manifest,
            parameter,
            [(_parse_shard_identity(slot["shard"]), slot["state"].get("momentum")) for slot in slots],
        )
        state = {"step": step, "momentum": momentum}
        if implementation == "gefen.Gefen" and variant == "initialized_dense":
            source_second = "block"
            second_step = _consensus(
                [slot["state"]["second_moment_step"] for slot in required], name="second_moment_step"
            )
            state.update(
                {
                    "second_moment": _assemble_dense_logical_field(
                        manifest,
                        parameter,
                        [(_parse_shard_identity(slot["shard"]), slot["state"].get("second_moment")) for slot in slots],
                    ),
                    "second_moment_step": second_step,
                }
            )
        elif implementation == "gefen.Gefen" and variant == "initialized_factored":
            source_second = "factored"
            if any(_parse_shard_identity(slot["shard"]).layout is not ParameterLayout.REPLICATED for slot in slots):
                raise ValueError("factored portable state must be assembled from exact replicas")
            factored_step = _consensus([slot["state"]["factored_step"] for slot in required], name="factored_step")
            state.update(
                {
                    "v_row": _assemble_special_field(
                        slots, key="v_row", shape=(parameter.global_shape[0],), name="v_row"
                    ),
                    "v_col": _assemble_special_field(
                        slots, key="v_col", shape=(parameter.global_shape[1],), name="v_col"
                    ),
                    "factored_step": factored_step,
                }
            )
        elif implementation == "gefen.GefenMuon" and variant == "initialized_dense_normuon":
            state.update(
                {
                    "normuon_v": _assemble_special_field(
                        slots,
                        key="normuon_v",
                        shape=(parameter.global_shape[0], 1),
                        name="normuon_v",
                    ),
                    "normuon_step": _consensus(
                        [slot["state"]["normuon_step"] for slot in required], name="normuon_step"
                    ),
                }
            )
    if implementation == "gefen.Gefen":
        representation = catalog_entry["algorithm_options"]["second_moment_policy"]
        if variant == "initialized_dense" and representation != "block":
            raise ValueError("block portable state conflicts with factored_v_2d policy")
        if variant == "initialized_factored" and representation != "factored":
            raise ValueError("factored portable state conflicts with factored_v_2d policy")
        hints = {"source_periods": periods, "source_second_moment": source_second, "target_period": 1}
    else:
        hints = {"source_periods": periods, "target_period": 1}
    return {
        "identity": _serialize_parameter_identity(parameter),
        "algorithm_options": catalog_entry["algorithm_options"],
        "state_variant": variant,
        "state": state,
        "projection_hints": hints,
    }


def _assemble_portable_state_fragments(fragments, *, process_group_identity, limits):
    """Assemble member-ordered fragments into one complete topology-neutral document."""

    limits = _require_limits(limits)
    if not isinstance(process_group_identity, ProcessGroupIdentity):
        raise TypeError("process_group_identity must be a ProcessGroupIdentity")
    process_group_identity = ProcessGroupIdentity(
        process_group_identity.semantic_name,
        process_group_identity.ordered_members,
        schema_version=process_group_identity.schema_version,
    )
    members = tuple(_strict_name(member, name="ordered member") for member in process_group_identity.ordered_members)
    if len(set(members)) != len(members) or len(members) > limits.max_members:
        raise ValueError("process-group members must be unique and within max_members")
    if isinstance(fragments, (str, bytes, bytearray)):
        raise TypeError("fragments must be a sequence")
    try:
        iterator = iter(fragments)
    except TypeError as exc:
        raise TypeError("fragments must be a sequence") from exc
    bounded_fragments = []
    for fragment in iterator:
        bounded_fragments.append(fragment)
        if len(bounded_fragments) > limits.max_members:
            raise ValueError("portable fragments exceed max_members")
    fragments = tuple(bounded_fragments)
    if len(fragments) != len(members):
        raise ValueError("portable fragments must contain every process-group member exactly once")

    normalized = []
    total_tensor_bytes = 0
    total_metadata_bytes = 0
    for fragment in fragments:
        normalized_fragment = _normalize_portable_state_fragment(fragment, limits=limits)
        plan = _prepare_canonical_wire_value(normalized_fragment, limits._wire_limits())
        if total_tensor_bytes > limits.max_collective_tensor_bytes - plan.total_tensor_bytes:
            raise ValueError("portable fragments exceed max_collective_tensor_bytes")
        if total_metadata_bytes > limits.max_collective_metadata_bytes - len(plan.metadata):
            raise ValueError("portable fragments exceed max_collective_metadata_bytes")
        total_tensor_bytes += plan.total_tensor_bytes
        total_metadata_bytes += len(plan.metadata)
        normalized.append(normalized_fragment)
    if tuple(fragment["member"] for fragment in normalized) != members:
        raise ValueError("portable fragments are not in process-group member order")

    implementation = _consensus([fragment["implementation"] for fragment in normalized], name="implementation")
    policy = _consensus([fragment["policy"] for fragment in normalized], name="policy")
    common = _consensus([fragment["common"] for fragment in normalized], name="common")
    manifest_record = _consensus([fragment["manifest"] for fragment in normalized], name="manifest")
    catalog = _consensus([fragment["catalog"] for fragment in normalized], name="catalog")
    manifest = _parse_sharding_manifest(manifest_record)
    if any(shard.process_group != process_group_identity for shard in manifest.shards):
        raise ValueError("portable manifest process groups must exactly match process_group_identity")
    all_slots = [slot for fragment in normalized for slot in fragment["logical_slots"]]
    shard_slots = {_parse_shard_identity(slot["shard"]): slot for slot in all_slots}
    if len(shard_slots) != len(all_slots) or set(shard_slots) != set(manifest.shards):
        raise ValueError("portable fragments must exactly cover the complete manifest")

    parameters = {}
    parameter_positions = {}
    for fqn in sorted(catalog):
        slots = [shard_slots[shard] for shard in manifest.for_parameter(fqn)]
        parameter_positions[fqn] = _consensus(
            [(slot["group_index"], slot["original_slot_index"]) for slot in slots],
            name="logical slot position",
        )
        _consensus([slot["compatibility_name"] for slot in slots], name="compatibility_name")
        parameters[fqn] = _assemble_parameter(
            fqn,
            slots,
            manifest,
            catalog[fqn],
            implementation,
            policy,
            common["gefen_global_step"],
        )
    if len(set(parameter_positions.values())) != len(parameter_positions):
        raise ValueError("portable parameters must have unique logical slot positions")
    document = build_portable_state_document(
        implementation=implementation,
        policy=policy,
        common=common,
        parameters=parameters,
        provenance=None,
    )
    return _normalize_gefen_portable_state_document(
        document,
        limits=limits,
        expected_implementation=implementation,
    )


def _project_portable_parameter_state(
    document_record,
    target_shard,
    *,
    implementation,
    global_step,
    codebook,
    source_second_moment_projection,
    target_algorithm_options,
    target_second_moment=None,
):
    """Project one normalized global record to a target shard's native state fields."""

    if implementation not in _IMPLEMENTATIONS:
        raise ValueError("unsupported portable projection implementation")
    global_step = _strict_int(global_step, name="gefen_global_step", maximum=_MAX_EXACT_COUNTER)
    common_codebook = _normalize_codebook(codebook, global_step=global_step)
    if not isinstance(target_shard, ShardIdentity):
        raise TypeError("target_shard must be a ShardIdentity")
    target_layouts = (
        {
            ParameterLayout.REPLICATED,
            ParameterLayout.FLATTENED_ELEMENT_SHARD,
            ParameterLayout.DTENSOR_1D_DEFAULT_WORLD,
        }
        if implementation == "gefen.Gefen"
        else {ParameterLayout.REPLICATED, ParameterLayout.WHOLE_PARAMETER_OWNER}
    )
    if target_shard.layout not in target_layouts:
        raise ValueError("target shard layout is unsupported for the portable implementation")
    if target_shard.process_group is None:
        raise ValueError("portable projection requires a grouped target shard")
    source_factored = (
        implementation == "gefen.Gefen"
        and type(document_record) is dict
        and type(document_record.get("algorithm_options")) is dict
        and document_record["algorithm_options"].get("second_moment_policy") == "factored"
    )
    policy = {
        "schema_version": 1,
        "factored_v_2d": source_factored,
        "force_1d_period_one": False,
        "force_2d_period_one": False,
        "period_one_substrings": [],
        "codebook_refresh_every": 0,
        "stochastic_round": False,
        "momentum_projection": _MOMENTUM_PROJECTION,
        "second_moment_projection": source_second_moment_projection,
    }
    fqn = target_shard.parameter.fqn
    record = _normalize_complete_parameter_record(fqn, document_record, implementation, global_step, policy)
    if _parse_parameter_identity(record["identity"]) != target_shard.parameter:
        raise ValueError("portable parameter identity does not match the target shard")
    target_options = _normalize_options(target_algorithm_options, implementation)
    if implementation == "gefen.Gefen":
        if target_second_moment not in {"block", "factored"}:
            raise ValueError("plain Gefen projection requires target_second_moment")
        if target_second_moment != target_options["second_moment_policy"]:
            raise ValueError("target_second_moment conflicts with target algorithm options")
        source_options = record["algorithm_options"]
        invariant_option_keys = _PLAIN_OPTION_KEYS - {"second_moment_policy"}
        if any(not _values_equal(source_options[key], target_options[key]) for key in invariant_option_keys):
            raise ValueError("portable algorithm options do not match the target")
        source_representation = source_options["second_moment_policy"]
        if source_representation == "block" and target_second_moment == "factored":
            raise ValueError("portable block-to-factored second-moment migration is unsupported")
        if (
            source_representation == "factored"
            and target_second_moment == "block"
            and source_second_moment_projection != _SECOND_MOMENT_PROJECTION_FACTORED_TO_BLOCK
        ):
            raise ValueError("portable source policy does not authorize factored-to-block second-moment projection")
        if (
            target_second_moment == "factored"
            and len(target_shard.parameter.global_shape) == 2
            and target_shard.layout is not ParameterLayout.REPLICATED
        ):
            raise ValueError("factored logical matrices require replicated target shards")
    else:
        if not _values_equal(record["algorithm_options"], target_options):
            raise ValueError("portable algorithm options do not match the target")
        if target_second_moment is not None:
            raise ValueError("Muon projection does not accept target_second_moment")
        if (
            target_shard.layout is ParameterLayout.WHOLE_PARAMETER_OWNER
            and target_options["sharded_mode"] != "distributed"
        ):
            raise ValueError("whole-parameter Muon targets require sharded_mode='distributed'")

    variant = record["state_variant"]
    if variant == "pristine":
        return {}
    if common_codebook is None:
        raise ValueError("non-pristine portable projection requires a codebook")
    if variant == "period_selected":
        return {"automatic_period": 1} if _local_dense_shape(target_shard) is not None else {}
    local_momentum = _project_dense_logical_field(
        target_shard.parameter,
        record["state"]["momentum"],
        target_shard,
    )
    if local_momentum is None:
        return {}
    step = record["state"]["step"]
    indices, magnitudes = _recompress_dense_momentum(local_momentum, common_codebook, period=1, step=step)
    result = {
        "automatic_period": 1,
        "step": step,
        "m_codebook": indices,
        "m_magnitude": magnitudes,
    }
    if implementation == "gefen.Gefen" and variant == "initialized_dense":
        local_second = _project_dense_logical_field(
            target_shard.parameter,
            record["state"]["second_moment"],
            target_shard,
        )
        if local_second is None:
            return {}
        second_step = record["state"]["second_moment_step"]
        result.update(
            {
                "vmean": _reduce_block_second_moment(local_second, period=1, step=second_step),
                "vmean_step": second_step,
            }
        )
    elif implementation == "gefen.Gefen" and variant == "initialized_factored":
        second_step = record["state"]["factored_step"]
        if target_second_moment == "factored":
            if target_shard.layout is not ParameterLayout.REPLICATED:
                raise ValueError("factored portable state projects only to replicated targets")
            result.update(
                {
                    "v_row": _tight_clone(record["state"]["v_row"]),
                    "v_col": _tight_clone(record["state"]["v_col"]),
                    "factored_step": second_step,
                }
            )
        else:
            dense_second = _expand_factored_second_moment_live_fp32_v1(
                record["state"]["v_row"],
                record["state"]["v_col"],
                logical_shape=target_shard.parameter.global_shape,
                step=second_step,
            )
            local_second = _project_dense_logical_field(
                target_shard.parameter,
                dense_second,
                target_shard,
            )
            if local_second is None:
                return {}
            result.update(
                {
                    "vmean": _reduce_block_second_moment(local_second, period=1, step=second_step),
                    "vmean_step": second_step,
                }
            )
    elif implementation == "gefen.GefenMuon" and variant == "initialized_dense_normuon":
        if target_shard.layout not in {ParameterLayout.REPLICATED, ParameterLayout.WHOLE_PARAMETER_OWNER}:
            raise ValueError("NorMuon state projects only to replicas or a whole-parameter owner")
        result.update(
            {
                "normuon_v": _tight_clone(record["state"]["normuon_v"]),
                "normuon_step": record["state"]["normuon_step"],
            }
        )
    return result


__all__ = ["PortableStateLimits"]

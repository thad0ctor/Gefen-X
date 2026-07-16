"""Private stable-rebinding and atomic-load adapter for an exact AdamW child."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
import math

import torch

from gefen.contracts import ParameterLayout, ShardingManifest
from gefen.dtensor import (
    is_exact_dtensor,
    resolve_local_tensor,
    validate_dtensor_rebinding_plan,
)


def _require_exact_adamw(optimizer) -> None:
    if type(optimizer) is not torch.optim.AdamW:
        raise TypeError("stable Hybrid AdamW integration requires an exact torch.optim.AdamW child")
    if type(optimizer.__dict__) is not dict:
        raise TypeError("stable Hybrid AdamW integration requires an exact attribute dictionary")


def _target_has_internal_overlap(target) -> bool:
    required_span = 1
    dimensions = sorted(
        (stride, size)
        for size, stride in zip(target.shape, target.stride())
        if size > 1
    )
    for stride, size in dimensions:
        if stride < required_span:
            return True
        required_span += (size - 1) * stride
    return False


def _validate_storage(target) -> torch.Tensor:
    local = resolve_local_tensor(target) if is_exact_dtensor(target) else target
    if not isinstance(local, torch.Tensor):
        raise TypeError("AdamW rebound targets must be tensors")
    if torch.is_complex(local):
        raise ValueError("Hybrid AdamW rebinding does not support complex targets")
    if local.layout is not torch.strided or local.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise ValueError("Hybrid AdamW rebound targets require strided floating-point storage")
    if local.is_meta:
        raise ValueError("Hybrid AdamW rebound targets require materialized storage")
    if _target_has_internal_overlap(local):
        raise ValueError("Hybrid AdamW rebound targets must be free of internal storage overlap")
    return local


def _validate_rebinding_target(rebinding) -> None:
    shard = rebinding.shard
    target = rebinding.new_parameter
    if rebinding.old_parameter.grad is not None or (
        target is not None and target.grad is not None
    ):
        raise RuntimeError("Hybrid AdamW post_sharding must run before source or target gradients exist")
    owns_whole = (
        shard.layout is ParameterLayout.WHOLE_PARAMETER_OWNER
        and shard.local_member == shard.owner
    )
    if shard.layout is ParameterLayout.WHOLE_PARAMETER_OWNER and not owns_whole:
        if target is not None:
            raise ValueError("Hybrid AdamW whole-parameter nonowners must bind no storage")
        return
    if target is None:
        raise ValueError("only a whole-parameter nonowner may bind no AdamW storage")
    if not isinstance(target, torch.Tensor):
        raise TypeError("Hybrid AdamW rebinding targets must be tensors or None")
    local = _validate_storage(target)
    if not target.is_leaf and not target.retains_grad:
        raise ValueError("can't optimize a non-leaf rebound AdamW Tensor")

    if shard.layout is ParameterLayout.REPLICATED:
        if tuple(target.shape) != shard.parameter.global_shape:
            raise ValueError("replicated Hybrid AdamW rebinding requires complete parameter storage")
        logical_numel = shard.logical_slice.length
        physical_numel = target.numel()
    elif shard.layout is ParameterLayout.FLATTENED_ELEMENT_SHARD:
        if target.ndim != 1 or not target.is_contiguous():
            raise ValueError("flattened Hybrid AdamW rebinding requires a contiguous 1-D tensor shard")
        logical_numel = shard.logical_slice.length
        physical_numel = target.numel()
    elif shard.layout is ParameterLayout.WHOLE_PARAMETER_OWNER:
        if tuple(target.shape) != shard.parameter.global_shape:
            raise ValueError("Hybrid AdamW whole-parameter owners require complete parameter storage")
        logical_numel = shard.logical_slice.length
        physical_numel = target.numel()
    elif shard.layout is ParameterLayout.DTENSOR_1D_DEFAULT_WORLD:
        if tuple(target.shape) != shard.parameter.global_shape:
            raise ValueError("Hybrid AdamW DTensor global shape differs from its identity")
        logical_numel = shard.logical_region.numel
        physical_numel = local.numel()
    else:
        raise ValueError("Hybrid AdamW rebinding does not support this parameter layout")
    if physical_numel != logical_numel:
        raise ValueError("Hybrid AdamW rebound storage size differs from its logical shard")


def stage_adamw_post_sharding(optimizer, rebindings, manifest: ShardingManifest):
    """Return an isolated pristine AdamW shadow with rebound parameter slots."""

    _require_exact_adamw(optimizer)
    rebindings = tuple(rebindings)
    if not rebindings:
        raise ValueError("Hybrid AdamW child has no routed rebinding")
    validate_dtensor_rebinding_plan(rebindings, manifest)

    live_slots = []
    for group_index, group in enumerate(optimizer.param_groups):
        if type(group) is not dict or type(group.get("params")) is not list:
            raise TypeError("Hybrid AdamW rebinding requires exact parameter-group containers")
        names = group.get("param_names")
        if type(names) is not list or len(names) != len(group["params"]):
            raise ValueError("Hybrid AdamW rebinding requires one stable name per parameter slot")
        if any(type(name) is not str for name in names):
            raise TypeError("Hybrid AdamW parameter names must be strings")
        for parameter_index, (parameter, name) in enumerate(zip(group["params"], names)):
            live_slots.append((group_index, parameter_index, parameter, name))

    if len(rebindings) != len(live_slots):
        raise ValueError("Hybrid AdamW post_sharding requires one rebinding per original slot")
    recognized = [slot[2] for slot in live_slots] + [item.old_parameter for item in rebindings]
    for parameter, state in optimizer.state.items():
        if not any(parameter is candidate for candidate in recognized):
            raise RuntimeError("Hybrid AdamW rebinding found orphan optimizer state")
        if type(state) is not dict or state:
            raise RuntimeError("Hybrid AdamW rebinding is allowed only before optimizer state initialization")

    assigned_positions = {}
    seen_sources = []
    seen_targets = []
    for binding_index, rebinding in enumerate(rebindings):
        if not isinstance(rebinding.old_parameter, torch.Tensor):
            raise TypeError("Hybrid AdamW rebinding sources must be tensors")
        if any(rebinding.old_parameter is source for source in seen_sources):
            raise ValueError("Hybrid AdamW rebinding sources must be unique")
        seen_sources.append(rebinding.old_parameter)
        if rebinding.shard not in manifest.shards:
            raise ValueError("Hybrid AdamW local shard identity is absent from its child manifest")
        _validate_rebinding_target(rebinding)
        target = rebinding.new_parameter
        if target is not None:
            if any(target is previous for previous in seen_targets):
                raise ValueError("Hybrid AdamW rebound targets must be unique")
            seen_targets.append(target)

        source_positions = [
            index for index, slot in enumerate(live_slots) if slot[2] is rebinding.old_parameter
        ]
        target_positions = (
            []
            if target is None
            else [index for index, slot in enumerate(live_slots) if slot[2] is target]
        )
        if len(source_positions) == 1:
            position = source_positions[0]
            if target_positions and target_positions != [position]:
                raise ValueError("Hybrid AdamW rebound target occupies another optimizer slot")
        elif not source_positions and len(target_positions) == 1:
            position = target_positions[0]
        else:
            raise ValueError("Hybrid AdamW source must identify exactly one optimizer slot")
        if position in assigned_positions:
            raise ValueError("Hybrid AdamW rebindings target the same optimizer slot")
        assigned_positions[position] = binding_index
    if len(assigned_positions) != len(live_slots):
        raise ValueError("Hybrid AdamW post_sharding did not bind every optimizer slot")

    staged_groups = []
    slot_cursor = 0
    for group in optimizer.param_groups:
        staged_group = dict(group)
        staged_params = []
        staged_names = []
        for _ in group["params"]:
            rebinding = rebindings[assigned_positions[slot_cursor]]
            compatibility_name = live_slots[slot_cursor][3]
            slot_cursor += 1
            if rebinding.new_parameter is not None:
                staged_params.append(rebinding.new_parameter)
                staged_names.append(compatibility_name)
        staged_group["params"] = staged_params
        staged_group["param_names"] = staged_names
        staged_groups.append(staged_group)

    staged = object.__new__(type(optimizer))
    staged.__dict__ = optimizer.__dict__.copy()
    staged.defaults = optimizer.defaults.copy()
    staged.param_groups = staged_groups
    staged.state = defaultdict(dict)
    validate_adamw_state(staged)
    return staged


def _scalar_value(name, value, *, minimum=0.0, strict_maximum=None) -> float:
    if torch.is_tensor(value):
        if value.numel() != 1 or value.dtype == torch.bool:
            raise ValueError("AdamW {} must be a scalar".format(name))
        scalar = float(value.detach().cpu().item())
    elif type(value) in (int, float):
        scalar = float(value)
    else:
        raise TypeError("AdamW {} must be numeric".format(name))
    if not math.isfinite(scalar) or scalar < minimum:
        raise ValueError("AdamW {} is outside its supported range".format(name))
    if strict_maximum is not None and scalar >= strict_maximum:
        raise ValueError("AdamW {} is outside its supported range".format(name))
    return scalar


def _validate_group(group, defaults, expected_names=None) -> None:
    allowed = set(defaults) | {"params", "param_names"}
    if type(group) is not dict or set(group) - {"initial_lr"} != allowed:
        raise ValueError("AdamW parameter group has an invalid exact schema")
    if type(group["params"]) is not list:
        raise TypeError("AdamW parameter group params must be a list")
    names = group["param_names"]
    if type(names) is not list or len(names) != len(group["params"]):
        raise ValueError("AdamW parameter group requires one name per parameter")
    if any(type(name) is not str for name in names):
        raise TypeError("AdamW parameter names must be strings")
    if expected_names is not None and tuple(names) != tuple(expected_names):
        raise ValueError("AdamW checkpoint parameter names differ from the finalized routing")
    _scalar_value("lr", group["lr"])
    if "initial_lr" in group:
        _scalar_value("initial_lr", group["initial_lr"])
    betas = group["betas"]
    if type(betas) is not tuple or len(betas) != 2:
        raise ValueError("AdamW betas must be an exact pair")
    _scalar_value("beta1", betas[0], strict_maximum=1.0)
    _scalar_value("beta2", betas[1], strict_maximum=1.0)
    _scalar_value("eps", group["eps"])
    _scalar_value("weight_decay", group["weight_decay"])
    for key in ("amsgrad", "maximize", "capturable", "differentiable"):
        if type(group[key]) is not bool:
            raise TypeError("AdamW {} must be a bool".format(key))
    for key in ("foreach", "fused"):
        if group[key] is not None and type(group[key]) is not bool:
            raise TypeError("AdamW {} must be a bool or None".format(key))
    if "decoupled_weight_decay" in defaults and group["decoupled_weight_decay"] is not True:
        raise ValueError("AdamW requires decoupled_weight_decay=True")


def _moment_local(moment, parameter, name):
    if is_exact_dtensor(parameter):
        if not is_exact_dtensor(moment):
            raise TypeError("AdamW {} must be a DTensor for a DTensor parameter".format(name))
        if tuple(moment.shape) != tuple(parameter.shape):
            raise ValueError("AdamW {} global shape differs from its parameter".format(name))
        moment_mesh = moment.device_mesh
        parameter_mesh = parameter.device_mesh
        moment_mesh_signature = (
            str(moment_mesh.device_type),
            tuple(moment_mesh.shape),
            tuple(moment_mesh.mesh_dim_names or ()),
            tuple(int(rank) for rank in moment_mesh.mesh.detach().cpu().reshape(-1).tolist()),
        )
        parameter_mesh_signature = (
            str(parameter_mesh.device_type),
            tuple(parameter_mesh.shape),
            tuple(parameter_mesh.mesh_dim_names or ()),
            tuple(int(rank) for rank in parameter_mesh.mesh.detach().cpu().reshape(-1).tolist()),
        )
        if moment_mesh_signature != parameter_mesh_signature or tuple(moment.placements) != tuple(parameter.placements):
            raise ValueError("AdamW {} DTensor layout differs from its parameter".format(name))
        local = resolve_local_tensor(moment)
        parameter_local = resolve_local_tensor(parameter)
        if tuple(local.shape) != tuple(parameter_local.shape):
            raise ValueError("AdamW {} local shape differs from its parameter".format(name))
    else:
        if type(moment) is not torch.Tensor:
            raise TypeError("AdamW {} must be an ordinary tensor".format(name))
        local = moment
        parameter_local = parameter
        if tuple(local.shape) != tuple(parameter_local.shape):
            raise ValueError("AdamW {} shape differs from its parameter".format(name))
    if local.layout is not torch.strided or local.dtype != parameter_local.dtype or local.device != parameter_local.device:
        raise ValueError("AdamW {} storage semantics differ from its parameter".format(name))
    if not bool(torch.isfinite(local.detach()).all()):
        raise ValueError("AdamW {} must contain only finite values".format(name))
    return local


def _validate_parameter_state(parameter, state, group) -> None:
    if type(state) is not dict:
        raise TypeError("AdamW parameter state must be an exact dictionary")
    if not state:
        return
    expected = {"step", "exp_avg", "exp_avg_sq"}
    if group["amsgrad"]:
        expected.add("max_exp_avg_sq")
    if set(state) != expected:
        raise ValueError("AdamW parameter state has an invalid exact schema")
    step = state["step"]
    if (
        type(step) is not torch.Tensor
        or step.ndim != 0
        or step.dtype not in {torch.float32, torch.float64}
        or step.layout is not torch.strided
    ):
        raise ValueError("AdamW step must be a floating-point scalar tensor")
    if group["capturable"] or group["fused"]:
        if step.device != parameter.device:
            raise ValueError("capturable or fused AdamW step must live with its parameter")
    elif step.device.type != "cpu":
        raise ValueError("ordinary AdamW step must use CPU scalar storage")
    step_value = float(step.detach().cpu().item())
    if not math.isfinite(step_value) or step_value <= 0 or step_value != math.floor(step_value):
        raise ValueError("AdamW step must be a positive finite integer-valued scalar")
    _moment_local(state["exp_avg"], parameter, "exp_avg")
    exp_avg_sq = _moment_local(state["exp_avg_sq"], parameter, "exp_avg_sq")
    if bool((exp_avg_sq.detach() < 0).any()):
        raise ValueError("AdamW exp_avg_sq must be nonnegative")
    if group["amsgrad"]:
        maximum = _moment_local(state["max_exp_avg_sq"], parameter, "max_exp_avg_sq")
        if bool((maximum.detach() < exp_avg_sq.detach()).any()):
            raise ValueError("AdamW max_exp_avg_sq must dominate exp_avg_sq")


def validate_adamw_state(optimizer, *, expected_names=None) -> None:
    """Validate exact live AdamW group and per-parameter state semantics."""

    _require_exact_adamw(optimizer)
    if type(optimizer.defaults) is not dict or type(optimizer.param_groups) is not list:
        raise TypeError("AdamW defaults and parameter groups must use exact containers")
    if expected_names is not None and len(expected_names) != len(optimizer.param_groups):
        raise ValueError("AdamW finalized group count differs from its routing")
    live_parameters = []
    group_by_parameter = {}
    for index, group in enumerate(optimizer.param_groups):
        _validate_group(
            group,
            optimizer.defaults,
            None if expected_names is None else expected_names[index],
        )
        for parameter in group["params"]:
            if not isinstance(parameter, torch.Tensor):
                raise TypeError("AdamW parameter groups must contain tensors")
            parameter_id = id(parameter)
            if parameter_id in group_by_parameter:
                raise ValueError("AdamW parameter groups contain a duplicate parameter")
            live_parameters.append(parameter)
            group_by_parameter[parameter_id] = (parameter, group)
    for parameter, state in optimizer.state.items():
        entry = group_by_parameter.get(id(parameter))
        if entry is None or entry[0] is not parameter:
            raise ValueError("AdamW state contains an orphan parameter entry")
        _validate_parameter_state(parameter, state, entry[1])


def prepare_adamw_load_state_dict(optimizer, state_dict):
    """Run AdamW pre-hooks and stage a validated load on an isolated shadow."""

    _require_exact_adamw(optimizer)
    expected_names = tuple(tuple(group["param_names"]) for group in optimizer.param_groups)
    prepared = state_dict.copy()
    for pre_hook in optimizer._optimizer_load_state_dict_pre_hooks.values():
        hook_result = pre_hook(optimizer, prepared)
        if hook_result is not None:
            prepared = hook_result

    staged = object.__new__(type(optimizer))
    staged.__dict__ = optimizer.__dict__.copy()
    staged.defaults = optimizer.defaults.copy()
    staged.param_groups = [
        {
            **group,
            "params": list(group["params"]),
            "param_names": list(group["param_names"]),
        }
        for group in optimizer.param_groups
    ]
    staged.state = defaultdict(dict)
    staged._optimizer_load_state_dict_pre_hooks = OrderedDict()
    staged._optimizer_load_state_dict_post_hooks = OrderedDict()
    torch.optim.Optimizer.load_state_dict(staged, prepared)
    staged._optimizer_load_state_dict_pre_hooks = optimizer._optimizer_load_state_dict_pre_hooks
    staged._optimizer_load_state_dict_post_hooks = optimizer._optimizer_load_state_dict_post_hooks
    validate_adamw_state(staged, expected_names=expected_names)
    return staged


def commit_adamw_stage(optimizer, staged) -> None:
    """Publish a prepared AdamW rebinding/load through callback-free swaps."""

    _require_exact_adamw(optimizer)
    _require_exact_adamw(staged)
    live_defaults = optimizer.defaults
    dict.update(live_defaults, staged.defaults)
    staged.defaults = live_defaults
    dict.update(optimizer.__dict__, staged.__dict__)


def run_adamw_load_post_hooks(optimizer) -> None:
    """Run AdamW load post-hooks after the composite core is fully published."""

    for post_hook in optimizer._optimizer_load_state_dict_post_hooks.values():
        post_hook(optimizer)


__all__ = [
    "commit_adamw_stage",
    "prepare_adamw_load_state_dict",
    "run_adamw_load_post_hooks",
    "stage_adamw_post_sharding",
    "validate_adamw_state",
]

"""Standalone DCP adapter for reshardable plain-Gefen FSDP2 state."""

from __future__ import annotations

from dataclasses import dataclass

import torch


_FORMAT_VERSION = 1
_CODEBOOK = torch.linspace(-1.0, 1.0, 256, dtype=torch.float32)


def _local(value):
    if hasattr(value, "to_local"):
        value = value.to_local()
    if hasattr(value, "wait"):
        value = value.wait()
    return value


def _is_dtensor(value) -> bool:
    return (
        hasattr(value, "to_local")
        and hasattr(value, "placements")
        and hasattr(value, "device_mesh")
    )


def _counter(value, name: str) -> int:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("{} must be scalar".format(name))
        value = value.item()
    if type(value) not in (int, float) or int(value) != value or value < 0:
        raise ValueError("{} must be a nonnegative integer".format(name))
    return int(value)


def _dtensor_from_local(local: torch.Tensor, parameter):
    from torch.distributed.tensor import DTensor

    return DTensor.from_local(
        local,
        device_mesh=parameter.device_mesh,
        placements=parameter.placements,
        shape=parameter.shape,
        stride=parameter.stride(),
        run_check=False,
    )


def _dense_momentum(optimizer, parameter, state) -> torch.Tensor:
    indices = state["m_codebook"].reshape(-1).long()
    magnitude = state["m_magnitude"].reshape(-1).float()
    codebook = optimizer._gefen_codebook
    if codebook is None:
        raise RuntimeError("Gefen DCP save requires an initialized codebook")
    codebook = codebook.detach().to(device=indices.device, dtype=torch.float32)
    period = _counter(state["automatic_period"], "automatic_period")
    if period < 1 or indices.numel() != magnitude.numel() * period:
        raise ValueError("Gefen momentum block geometry is invalid")
    return (
        codebook[indices]
        .reshape(-1, period)
        .mul(magnitude.reshape(-1, 1))
        .reshape(_local(parameter).shape)
    )


def _dense_second_moment(parameter, state) -> torch.Tensor:
    period = _counter(state["automatic_period"], "automatic_period")
    vmean = state["vmean"].reshape(-1).float()
    local_numel = _local(parameter).numel()
    if period < 1 or vmean.numel() * period != local_numel:
        raise ValueError("Gefen second-moment block geometry is invalid")
    return (
        vmean.reshape(-1, 1)
        .expand(-1, period)
        .reshape(_local(parameter).shape)
        .clone()
    )


@dataclass(frozen=True)
class _Slot:
    index: int
    name: str
    parameter: torch.Tensor


class GefenDCPState:
    """DCP ``Stateful`` wrapper for reshardable plain-Gefen FSDP2 state.

    Use this object as the optimizer value passed to
    :func:`torch.distributed.checkpoint.save` and ``load``. The adapter is
    intentionally limited to plain :class:`gefen.Gefen`, one-dimensional
    default-world DTensors, and one ``Shard(0)`` placement. Native
    ``state_dict`` checkpoints remain unchanged.
    """

    def __init__(self, optimizer):
        from gefen.gefen import Gefen

        if type(optimizer) is not Gefen:
            raise TypeError("GefenDCPState supports plain Gefen only")
        self.optimizer = optimizer
        self._slots = self._validate_layout()

    def _validate_layout(self):
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError(
                "GefenDCPState requires an initialized distributed process group"
            )
        import torch.distributed as dist

        slots = []
        index = 0
        for group in self.optimizer.param_groups:
            if group.get("sharded_mode") is not None:
                raise RuntimeError("GefenDCPState does not support Muon sharded modes")
            names = group.get("param_names", ())
            if len(names) != len(group["params"]):
                names = tuple("param_{}".format(index + offset) for offset in range(len(group["params"])))
            for name, parameter in zip(names, group["params"]):
                if not _is_dtensor(parameter):
                    raise RuntimeError(
                        "GefenDCPState requires every parameter to be a DTensor"
                    )
                mesh = parameter.device_mesh
                if len(mesh.shape) != 1 or mesh.size() != dist.get_world_size():
                    raise RuntimeError(
                        "GefenDCPState supports only a one-dimensional default-world DeviceMesh"
                    )
                if len(parameter.placements) != 1:
                    raise RuntimeError(
                        "GefenDCPState supports exactly one DTensor placement"
                    )
                placement = parameter.placements[0]
                if type(placement).__name__ != "Shard" or placement.dim != 0:
                    raise RuntimeError(
                        "GefenDCPState supports only one-dimensional Shard(0) parameters"
                    )
                slots.append(_Slot(index, str(name), parameter))
                index += 1
        if not slots:
            raise RuntimeError("GefenDCPState requires at least one parameter")
        return tuple(slots)

    @staticmethod
    def _key(slot: _Slot, field: str) -> str:
        return "slot_{:08d}.{}".format(slot.index, field)

    def state_dict(self):
        result = {
            "format_version": torch.tensor(_FORMAT_VERSION, dtype=torch.int64),
            "global_step": torch.tensor(
                _counter(self.optimizer._gefen_global_step, "global_step"),
                dtype=torch.int64,
            ),
            "deterministic": torch.tensor(
                int(bool(self.optimizer._deterministic)), dtype=torch.uint8
            ),
            "slot_count": torch.tensor(len(self._slots), dtype=torch.int64),
        }
        for slot in self._slots:
            state = self.optimizer.state.get(slot.parameter)
            initialized = bool(
                state
                and all(
                    key in state
                    for key in (
                        "automatic_period",
                        "step",
                        "m_codebook",
                        "m_magnitude",
                        "vmean",
                    )
                )
            )
            result[self._key(slot, "initialized")] = torch.tensor(
                int(initialized), dtype=torch.uint8
            )
            result[self._key(slot, "step")] = torch.tensor(
                _counter(state.get("step", 0), "step") if state else 0,
                dtype=torch.int64,
            )
            result[self._key(slot, "vmean_step")] = torch.tensor(
                _counter(state.get("vmean_step", state.get("step", 0)), "vmean_step")
                if state
                else 0,
                dtype=torch.int64,
            )
            local = _local(slot.parameter)
            momentum = (
                _dense_momentum(self.optimizer, slot.parameter, state)
                if initialized
                else torch.zeros_like(local, dtype=torch.float32)
            )
            second = (
                _dense_second_moment(slot.parameter, state)
                if initialized
                else torch.zeros_like(local, dtype=torch.float32)
            )
            result[self._key(slot, "momentum")] = _dtensor_from_local(
                momentum, slot.parameter
            )
            result[self._key(slot, "second_moment")] = _dtensor_from_local(
                second, slot.parameter
            )
        return result

    def load_state_dict(self, state_dict):
        required = {"format_version", "global_step", "deterministic", "slot_count"}
        if not required.issubset(state_dict):
            raise ValueError("Gefen DCP checkpoint metadata is incomplete")
        version = _counter(state_dict["format_version"], "format_version")
        if version != _FORMAT_VERSION:
            raise ValueError(
                "Unsupported Gefen DCP checkpoint format version {}".format(version)
            )
        if _counter(state_dict["slot_count"], "slot_count") != len(self._slots):
            raise ValueError("Gefen DCP checkpoint parameter count differs")

        staged = []
        for slot in self._slots:
            keys = {
                field: self._key(slot, field)
                for field in (
                    "initialized",
                    "step",
                    "vmean_step",
                    "momentum",
                    "second_moment",
                )
            }
            if any(key not in state_dict for key in keys.values()):
                raise ValueError(
                    "Gefen DCP checkpoint is missing state for slot {}".format(
                        slot.index
                    )
                )
            initialized = bool(
                _counter(state_dict[keys["initialized"]], "initialized")
            )
            step = _counter(state_dict[keys["step"]], "step")
            vmean_step = _counter(
                state_dict[keys["vmean_step"]], "vmean_step"
            )
            momentum = _local(state_dict[keys["momentum"]]).detach().float()
            second = _local(state_dict[keys["second_moment"]]).detach().float()
            local = _local(slot.parameter)
            if momentum.shape != local.shape or second.shape != local.shape:
                raise ValueError(
                    "Gefen DCP checkpoint local shape differs for slot {}".format(
                        slot.index
                    )
                )
            if not torch.isfinite(momentum).all() or not torch.isfinite(second).all():
                raise ValueError("Gefen DCP checkpoint contains non-finite state")
            if (second < 0).any():
                raise ValueError("Gefen DCP second moment must be nonnegative")
            staged.append((slot, initialized, step, vmean_step, momentum, second))

        global_step = _counter(state_dict["global_step"], "global_step")
        deterministic = bool(
            _counter(state_dict["deterministic"], "deterministic")
        )
        new_state = {}
        codebook = _CODEBOOK.to(
            device=_local(self._slots[0].parameter).device,
            dtype=torch.float32,
        )
        for slot, initialized, step, vmean_step, momentum, second in staged:
            if not initialized:
                new_state[slot.parameter] = {"name": slot.name}
                continue
            flat = momentum.reshape(-1)
            indices = torch.where(
                torch.signbit(flat),
                torch.zeros(flat.numel(), dtype=torch.uint8, device=flat.device),
                torch.full(
                    (flat.numel(),), 255, dtype=torch.uint8, device=flat.device
                ),
            )
            new_state[slot.parameter] = {
                "name": slot.name,
                "automatic_period": 1,
                "step": step,
                "m_codebook": indices.reshape(-1, 1),
                "m_magnitude": flat.abs().reshape(-1, 1),
                "vmean": second.reshape(-1, 1).clone(),
                "vmean_step": vmean_step,
            }

        self.optimizer.state.clear()
        self.optimizer.state.update(new_state)
        self.optimizer._gefen_global_step = global_step
        self.optimizer._deterministic = deterministic
        self.optimizer._gefen_codebook = codebook
        self.optimizer._gefen_codebook_by_device.clear()
        self.optimizer._gefen_codebook_lut_by_device.clear()
        self.optimizer._sr_seed_by_device.clear()
        self.optimizer._reset_gefen_global_step_devices()
        self.optimizer._static_mark_sig = None

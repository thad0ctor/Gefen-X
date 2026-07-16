"""Atomic optimizer-state movement and synchronous CPU offload for Gefen."""

from collections import defaultdict, deque

import torch
import torch.nn as nn


STATE_TENSOR_KEYS = frozenset(
    {
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
COUNTER_KEYS = frozenset({"step", "vmean_step", "factored_step", "normuon_step"})
SCRATCH_KEYS = frozenset({"stepsize", "_h_buf"})
CAPTURABLE_KEYS = frozenset({"_capt_scalars", "_capt_consts", "_capt_consts_key", "_capt_stack", "_capt_row"})
ALLOWED_STATE_KEYS = (
    STATE_TENSOR_KEYS | SCRATCH_KEYS | CAPTURABLE_KEYS | frozenset({"name", "automatic_period", "m_codebook_shape"})
)
_DEVICE_TYPES = frozenset({"cpu", "cuda"})
_METADATA_LEAVES = (
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    torch.device,
    torch.dtype,
    torch.layout,
    torch.memory_format,
)
_METADATA_CONTAINERS = (dict, list, tuple, set, frozenset, deque, torch.Size)


class StateOffloadMixin:
    """State movement surface shared by Gefen and its Muon subclass.

    CPU offload itself is intentionally limited to plain ``Gefen`` because the
    Muon and hybrid stepping paths require a separate paging policy. Atomic
    ``move_state_`` remains available to both native optimizer classes.
    """

    @property
    def state_offload_active(self) -> bool:
        return getattr(self, "_gefen_state_offload_device", None) is not None

    @property
    def state_offload_device(self):
        return getattr(self, "_gefen_state_offload_device", None)

    @property
    def state_offload_poisoned(self) -> bool:
        return bool(getattr(self, "_gefen_state_offload_poisoned", False))

    def _assert_state_export_safe(self) -> None:
        if self.state_offload_poisoned:
            raise RuntimeError(
                "Gefen cannot export optimizer state after a failed state-offload copyback; load a known-good checkpoint first"
            )

    @staticmethod
    def _state_value_is_movement_safe_metadata(value, seen=None) -> bool:
        if value is None or type(value) in _METADATA_LEAVES:
            return True
        if type(value) not in _METADATA_CONTAINERS:
            return False
        if seen is None:
            seen = set()
        if id(value) in seen:
            return False
        seen.add(id(value))
        values = value.items() if type(value) is dict else value
        if type(value) is dict:
            return all(
                StateOffloadMixin._state_value_is_movement_safe_metadata(key, seen)
                and StateOffloadMixin._state_value_is_movement_safe_metadata(item, seen)
                for key, item in values
            )
        return all(StateOffloadMixin._state_value_is_movement_safe_metadata(item, seen) for item in values)

    @staticmethod
    def _state_movement_tensor_supported(value) -> bool:
        return (
            type(value) is torch.Tensor
            and not (hasattr(value, "to_local") and hasattr(value, "placements"))
            and value.layout is torch.strided
            and not value.is_nested
            and not value.is_quantized
            and getattr(value, "fake_mode", None) is None
            and value.device.type in _DEVICE_TYPES
        )

    @staticmethod
    def _parameter_device(parameter) -> torch.device:
        local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
        if hasattr(local, "wait"):
            local = local.wait()
        if not torch.is_tensor(local) or local.device.type not in _DEVICE_TYPES:
            raise RuntimeError("parameters must use CPU or CUDA local storage")
        return torch.device("cpu") if local.device.type == "cpu" else local.device

    def _parameters(self):
        return [parameter for group in self.param_groups for parameter in group["params"]]

    @staticmethod
    def _capturing_on_parameter_device(parameters) -> bool:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return True
        devices = sorted(
            {p.device for p in parameters if torch.is_tensor(p) and p.device.type == "cuda"},
            key=lambda device: -1 if device.index is None else device.index,
        )
        for device in devices:
            with torch.cuda.device(device):
                if torch.cuda.is_current_stream_capturing():
                    return True
        return False

    def _state_movement_rejection_reason(self):
        if self.capturable:
            return "capturable optimizers have device-authoritative replay state"
        if self._capt_stacks is not None:
            return "capturable state stacks are active"
        if self._gefen_global_step_by_device or self._sr_seed_by_device:
            return "capturable device counters or stochastic-rounding seeds are active"
        try:
            parameters = self._parameters()
        except (KeyError, TypeError):
            return "parameter groups have an invalid structure"
        for parameter in parameters:
            try:
                self._parameter_device(parameter)
            except Exception:
                return "parameters must use CPU or CUDA local storage"
        if not (type(self.state) is dict or (type(self.state) is defaultdict and self.state.default_factory is dict)):
            return "optimizer state must use a supported standard mapping"
        for parameter, parameter_state in self.state.items():
            try:
                self._parameter_device(parameter)
            except Exception:
                return "optimizer state is keyed by a parameter without CPU or CUDA storage"
            if type(parameter_state) is not dict:
                return "per-parameter optimizer state must use a plain dictionary"
            for key, value in parameter_state.items():
                if not self._state_value_is_movement_safe_metadata(key):
                    return "optimizer state contains an unsupported key"
                if key in SCRATCH_KEYS:
                    continue
                if key in CAPTURABLE_KEYS:
                    return "capturable per-parameter state is active"
                if key in STATE_TENSOR_KEYS:
                    if torch.is_tensor(value):
                        if not self._state_movement_tensor_supported(value):
                            return "authoritative tensor state has an unsupported representation"
                    elif key not in COUNTER_KEYS or type(value) is not int:
                        return "authoritative tensor state has an invalid value"
                elif not self._state_value_is_movement_safe_metadata(value):
                    return "undeclared optimizer state is not provably tensor-free metadata"
        if self._gefen_codebook is not None and not self._state_movement_tensor_supported(self._gefen_codebook):
            return "the codebook has an unsupported tensor representation"
        return None

    def _atomic_state_movement_supported(self) -> bool:
        try:
            if self._state_movement_rejection_reason() is not None:
                return False
            if torch.compiler.is_compiling():
                return False
            return not (torch.cuda.is_available() and self._capturing_on_parameter_device(self._parameters()))
        except Exception:
            return False

    @staticmethod
    def _normalize_state_move_target(device, live_devices) -> torch.device:
        try:
            target = torch.device(device)
        except (TypeError, RuntimeError) as exc:
            raise TypeError("device must identify a CPU or CUDA device") from exc
        if target.type not in _DEVICE_TYPES:
            raise ValueError("Gefen state movement supports only CPU and CUDA devices")
        if target.type == "cpu":
            target = torch.device("cpu")
        elif not torch.cuda.is_available():
            raise ValueError("CUDA state movement requires an available CUDA device")
        elif target.index is None:
            if len(set(live_devices)) != 1 or live_devices[0].type != "cuda":
                raise ValueError("an unindexed CUDA target requires one CUDA parameter device")
            target = live_devices[0]
        elif target.index < 0 or target.index >= torch.cuda.device_count():
            raise ValueError("CUDA state movement target is not an available device")
        if any(parameter_device != target for parameter_device in live_devices):
            raise ValueError(
                "explicit state movement requires every live parameter to already be co-located on {}".format(target)
            )
        return target

    @staticmethod
    def _copy_state_tensor_for_move(tensor, device):
        return tensor.to(
            device=device,
            dtype=tensor.dtype,
            non_blocking=False,
            copy=True,
            memory_format=torch.contiguous_format,
        ).detach()

    @classmethod
    def _validate_staged_state_tensor(cls, source, staged, device) -> None:
        if (
            not cls._state_movement_tensor_supported(staged)
            or staged is source
            or staged.device != device
            or staged.dtype != source.dtype
            or tuple(staged.shape) != tuple(source.shape)
            or staged.requires_grad
            or not staged.is_contiguous()
            or staged.storage_offset() != 0
            or staged.untyped_storage().nbytes() != staged.numel() * staged.element_size()
        ):
            raise RuntimeError("state movement produced an invalid staged tensor")

    def _stage_state_move(self, device):
        parameters = self._parameters()
        live_devices = [self._parameter_device(parameter) for parameter in parameters]
        explicit_target = None if device is None else self._normalize_state_move_target(device, live_devices)
        codebook_target = explicit_target or (live_devices[0] if live_devices else torch.device("cpu"))
        live_ids = {id(parameter) for parameter in parameters}
        staged_state = defaultdict(dict)
        cuda_devices = set()

        def stage(value, target):
            result = self._copy_state_tensor_for_move(value, target)
            self._validate_staged_state_tensor(value, result, target)
            cuda_devices.update(item for item in (value.device, target) if item.type == "cuda")
            return result

        codebook = None if self._gefen_codebook is None else stage(self._gefen_codebook, codebook_target)
        for parameter, parameter_state in self.state.items():
            target = explicit_target if id(parameter) in live_ids else None
            result = {}
            for key, value in parameter_state.items():
                if key in SCRATCH_KEYS:
                    continue
                if key in CAPTURABLE_KEYS:
                    raise RuntimeError("Gefen state movement cannot migrate capturable scratch state")
                if key in STATE_TENSOR_KEYS and torch.is_tensor(value):
                    target = target or self._parameter_device(parameter)
                    result[key] = stage(value, target)
                else:
                    result[key] = value
            staged_state[parameter] = result
        for cuda_device in sorted(cuda_devices, key=lambda item: item.index or -1):
            torch.cuda.synchronize(cuda_device)
        return staged_state, codebook

    @torch.no_grad()
    def move_state_(self, device=None) -> None:
        """Atomically move authoritative state to live parameter devices or ``device``."""

        reason = self._state_movement_rejection_reason()
        if reason is not None:
            raise RuntimeError("Gefen atomic state movement is unavailable: {}".format(reason))
        if torch.compiler.is_compiling():
            raise RuntimeError("Gefen state movement cannot run during torch.compile")
        if torch.cuda.is_available() and self._capturing_on_parameter_device(self._parameters()):
            raise RuntimeError("Gefen state movement cannot run during CUDA graph capture")
        state, codebook = self._stage_state_move(device)
        self.__dict__.update(
            state=state,
            _gefen_codebook=codebook,
            _gefen_codebook_by_device={},
            _gefen_codebook_lut_by_device={},
            _static_mark_sig=None,
            _lr_scalar_cache=None,
            _gefen_state_offload_device=None,
        )

    @staticmethod
    def _offload_parameter_supported(parameter) -> bool:
        return (
            type(parameter) in {torch.Tensor, nn.Parameter}
            and parameter.layout is torch.strided
            and parameter.device.type == "cuda"
            and parameter.dtype in {torch.float16, torch.bfloat16, torch.float32, torch.float64}
            and not parameter.is_meta
            and not parameter.is_nested
            and not parameter.is_quantized
            and not torch.is_complex(parameter)
            and not hasattr(parameter, "placements")
        )

    @classmethod
    def _tight_cpu_tensor(cls, value) -> bool:
        return (
            cls._state_movement_tensor_supported(value)
            and value.device.type == "cpu"
            and not value.requires_grad
            and value.is_contiguous()
            and value.storage_offset() == 0
            and value.untyped_storage().nbytes() == value.numel() * value.element_size()
        )

    def _state_offload_rejection_reason(self, *, require_cpu_state, allow_poisoned=False):
        if type(self).__name__ != "Gefen":
            return "native state offload is implemented only by plain Gefen"
        if self.state_offload_active and self.state_offload_device != torch.device("cpu"):
            return "the active state-offload policy has an invalid device"
        if self.state_offload_poisoned and not allow_poisoned:
            return "a previous state copyback failed; load a known-good checkpoint"
        if self.capturable:
            return "capturable optimizers have device-authoritative replay state"
        parameters = self._parameters()
        if torch.compiler.is_compiling():
            return "state offload cannot run during torch.compile"
        if torch.cuda.is_available() and self._capturing_on_parameter_device(parameters):
            return "state offload cannot run during CUDA graph capture"
        reason = self._state_movement_rejection_reason()
        if reason is not None:
            return reason
        if not parameters or any(not self._offload_parameter_supported(p) for p in parameters):
            return "state offload requires ordinary replicated CUDA parameters"
        if set(self.state) != set(parameters):
            return "state offload requires exactly one state entry per live parameter"
        for parameter in parameters:
            parameter_state = self.state.get(parameter)
            if type(parameter_state) is not dict or any(key not in ALLOWED_STATE_KEYS for key in parameter_state):
                return "state offload does not support custom per-parameter state"
            if require_cpu_state and any(key in SCRATCH_KEYS for key in parameter_state):
                return "offloaded state contains device-side runtime scratch"
            for key, value in parameter_state.items():
                if key in COUNTER_KEYS and type(value) is not int:
                    return "noncapturable offloaded counters must be Python integers"
                if key in STATE_TENSOR_KEYS and torch.is_tensor(value):
                    if require_cpu_state and not self._tight_cpu_tensor(value):
                        return "authoritative offloaded tensors must be tight CPU tensors"
                    if not require_cpu_state and not self._state_movement_tensor_supported(value):
                        return "authoritative tensor state has an unsupported representation"
        persistent_tensors = list(parameters)
        if torch.is_tensor(self._gefen_codebook):
            persistent_tensors.append(self._gefen_codebook)
        persistent_tensors.extend(
            value
            for parameter_state in self.state.values()
            for key, value in parameter_state.items()
            if key in STATE_TENSOR_KEYS and torch.is_tensor(value)
        )
        try:
            storage_ids = [
                (str(tensor.device), tensor.untyped_storage().data_ptr())
                for tensor in persistent_tensors
                if tensor.numel()
            ]
        except Exception:
            return "persistent optimizer-state storage could not be inspected"
        if len(storage_ids) != len(set(storage_ids)):
            return "persistent optimizer-state storage aliases another live tensor"
        try:
            self._validate_loaded_native_state()
        except Exception:
            return "optimizer state does not match Gefen's declared native schema"
        if require_cpu_state and self._gefen_codebook is not None and self._gefen_codebook.device.type != "cuda":
            return "the optimizer-common codebook must remain CUDA-resident"
        return None

    def _copy_state_tensor_to_offload_cpu(self, tensor):
        return self._copy_state_tensor_for_move(tensor, torch.device("cpu"))

    def _prepare_cpu_parameter_state(self, parameter_state):
        result = {}
        cuda_devices = set()
        for key, value in parameter_state.items():
            if key in SCRATCH_KEYS:
                continue
            if key in CAPTURABLE_KEYS:
                raise RuntimeError("offloaded stepping produced capturable runtime state")
            if key in STATE_TENSOR_KEYS and torch.is_tensor(value):
                staged = self._copy_state_tensor_to_offload_cpu(value)
                self._validate_staged_state_tensor(value, staged, torch.device("cpu"))
                result[key] = staged
                if value.device.type == "cuda":
                    cuda_devices.add(value.device)
            else:
                result[key] = value
        for device in sorted(cuda_devices, key=lambda item: item.index or -1):
            torch.cuda.synchronize(device)
        return result

    def _stage_all_parameter_state_to_cpu(self):
        result = defaultdict(dict)
        for parameter, state in self.state.items():
            result[parameter] = self._prepare_cpu_parameter_state(state)
        return result

    def _stage_offloaded_parameter_state(self, parameter):
        result = {}
        for key, value in self.state[parameter].items():
            if key in STATE_TENSOR_KEYS and torch.is_tensor(value):
                if not self._tight_cpu_tensor(value):
                    raise RuntimeError("offloaded parameter tensors must remain tight CPU tensors")
                staged = self._copy_state_tensor_for_move(value, parameter.device)
                self._validate_staged_state_tensor(value, staged, parameter.device)
                result[key] = staged
            else:
                result[key] = value
        torch.cuda.synchronize(parameter.device)
        return result

    def _step_with_offloaded_parameter_state(self, update, group, name, parameter, grad):
        runtime_state = self._stage_offloaded_parameter_state(parameter)
        try:
            update(group, name, parameter, grad, state=runtime_state)
        except BaseException as operation_error:
            try:
                self.state[parameter] = self._prepare_cpu_parameter_state(runtime_state)
            except BaseException as copyback_error:
                self._gefen_state_offload_poisoned = True
                raise operation_error from copyback_error
            raise
        try:
            self.state[parameter] = self._prepare_cpu_parameter_state(runtime_state)
        except BaseException as exc:
            self._gefen_state_offload_poisoned = True
            raise RuntimeError(
                "Gefen state copyback failed after a parameter update; load a known-good checkpoint before stepping again"
            ) from exc

    def _assert_state_offload_step_ready(self) -> None:
        if self.state_offload_poisoned:
            raise RuntimeError(
                "Gefen state offload is poisoned after a failed copyback; load a known-good checkpoint before stepping again"
            )
        if self.state_offload_active:
            reason = self._state_offload_rejection_reason(require_cpu_state=True)
            if reason is not None:
                raise RuntimeError("Gefen state offload cannot step: {}".format(reason))

    @torch.no_grad()
    def offload_state_(self, device="cpu") -> None:
        """Atomically enable synchronous CPU-authoritative parameter state."""

        if torch.device(device).type != "cpu":
            raise ValueError("Gefen native state offload currently supports only CPU")
        reason = self._state_offload_rejection_reason(require_cpu_state=False)
        if reason is not None:
            raise RuntimeError("Gefen state offload is unavailable: {}".format(reason))
        state = self._stage_all_parameter_state_to_cpu()
        self.__dict__.update(
            state=state,
            _gefen_state_offload_device=torch.device("cpu"),
            _gefen_state_offload_poisoned=False,
            _static_mark_sig=None,
            _lr_scalar_cache=None,
        )

    @torch.no_grad()
    def restore_state_(self) -> None:
        """Atomically co-locate state with parameters and disable CPU offload."""

        if self.state_offload_active:
            self.move_state_()

    def _colocate_offload_codebook_with_parameters(self) -> None:
        """Restore the CUDA-resident codebook invariant on a staged load shadow.

        Offload keeps the small shared codebook co-located with the parameters
        while per-parameter state lives on CPU. A checkpoint restored with
        ``map_location='cpu'`` can deposit the codebook on CPU; move it back onto
        the parameter device and drop the now-stale per-device caches so the next
        step does not trip the "codebook must remain CUDA-resident" guard.
        """

        if not torch.is_tensor(self._gefen_codebook):
            return
        device = self._gefen_codebook_device()
        if self._gefen_codebook.device == device:
            return
        self._gefen_codebook = self._gefen_codebook.to(device)
        self._gefen_codebook_by_device = {}
        self._gefen_codebook_lut_by_device = {}

    def _stage_loaded_state_for_offload(self, staged) -> None:
        """Reconcile a staged load shadow with offload and poison state.

        A committed full state-dict load replaces the optimizer state outright,
        so it clears any prior copyback poison unconditionally: loading a
        known-good checkpoint is the documented recovery path and must not depend
        on whether offload happens to still be active at load time.
        """

        if staged.state_offload_active:
            reason = staged._state_offload_rejection_reason(require_cpu_state=False, allow_poisoned=True)
            if reason is not None:
                raise RuntimeError("Gefen could not preserve active state offload while loading: {}".format(reason))
            # The codebook must stay on the parameter device under active
            # offload; a map_location='cpu' load can land it on CPU, so restore
            # the invariant before staging per-parameter state back to CPU.
            staged._colocate_offload_codebook_with_parameters()
            staged.state = staged._stage_all_parameter_state_to_cpu()
        staged._gefen_state_offload_poisoned = False

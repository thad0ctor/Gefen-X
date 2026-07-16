"""Standalone DCP adapter for reshardable plain-Gefen FSDP2 state."""

from __future__ import annotations

from dataclasses import dataclass

import torch


# Schema history:
#   v1 -- counters + dense momentum/second-moment slots only.
#   v2 -- adds per-slot parameter identities (name + group + global shape) and
#         per-group hyperparameters (lr/betas/eps/weight_decay). Load is
#         fail-closed on version, identity, topology, and deterministic mismatch.
_FORMAT_VERSION = 2

_GROUP_HYPER_KEYS = ("lr", "beta1", "beta2", "eps", "weight_decay")


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


def _choose_period(optimizer, name: str, parameter, second: torch.Tensor) -> int:
    """Re-derive a compact block period for the resharded local shard.

    Runs Gefen's block-variance period search on the dense per-element second
    moment (a grad^2 proxy -- ``vmean`` is itself the EMA of block-mean grad^2),
    exactly as the native first step runs it on grad^2. The result is a divisor
    of the *new* local numel, so the restored state keeps ~1 byte/param instead
    of collapsing to per-element (period one).

    The optimizer's explicit period-one routing gates (``force_1d_period_one``,
    ``force_2d_period_one``, ``period_one_substrings``) are honored first, exactly
    as :meth:`Gefen._resolve_automatic_period` applies them before the raw search.
    Otherwise a checkpoint could restore period>1 into an optimizer configured for
    period-one, and the frozen restored codebook would keep the next step from
    correcting it.
    """
    from gefen.partitioning import find_period_by_block_variance

    if getattr(optimizer, "_force_1d_period_one", False) and parameter.ndim == 1:
        return 1
    if getattr(optimizer, "_force_2d_period_one", False) and parameter.ndim == 2:
        return 1
    substrings = getattr(optimizer, "_period_one_substrings", ())
    if substrings:
        lname = str(name).lower()
        if any(sub in lname for sub in substrings):
            return 1

    flat = second.reshape(-1)
    if flat.numel() < 8:
        return 1
    if flat.device.type == "cuda":
        return find_period_by_block_variance(
            flat.detach().float(),
            print_results=False,
            parameter_name=name,
            parameter_shape=tuple(parameter.shape),
            backend="gpu",
            input_is_squared=True,
        )
    return find_period_by_block_variance(
        flat.detach().float().cpu().numpy(),
        print_results=False,
        parameter_name=name,
        parameter_shape=tuple(parameter.shape),
        backend="cpu",
        input_is_squared=True,
    )


def _learn_codebook(name_flat_period, device):
    """Re-learn one exact codebook per rank from the resharded local momentum.

    Mirrors the native per-rank codebook (one histogram across all of the rank's
    parameters); the resharded momentum is the distribution being quantized, so
    learning from it directly minimizes the re-quantization error on this shard.
    Returns ``None`` when there is no initialized momentum to learn from.
    """
    from gefen.gefen import learn_gefen_exact_codebook_from_grad_periods

    grad_periods = [
        (name, flat, period, flat) for name, flat, period in name_flat_period
    ]
    if not grad_periods:
        return None
    return learn_gefen_exact_codebook_from_grad_periods(
        grad_periods=grad_periods,
        codebook_device=device,
        num_codebooks=256,
        force_endpoints=True,
        verbose=False,
        compute_mse_logging=False,
        use_fused_histogram=False,
    )


def _reblock(codebook, momentum: torch.Tensor, second: torch.Tensor, period: int):
    """Re-quantize a dense fp32 momentum shard into compact Gefen block state.

    Reproduces the native quantize arithmetic (``_automatic_momentum_update``):
    per-block max-abs magnitude, normalize into the codebook domain, nearest
    codeword indices. ``vmean`` is the per-block mean of the dense second moment
    (the block-mean grad^2 for the new blocking). Returns
    ``(m_codebook, m_magnitude, vmean)``.
    """
    from gefen.gefen import (
        automatic_partition_reduce,
        automatic_partition_view,
        gefen_nearest_codebook_indices,
    )

    flat = momentum.reshape(-1)
    blocks = automatic_partition_view(flat, period)
    magnitude = automatic_partition_reduce(flat.abs(), period, reduce_op="max")
    nonzero = magnitude > 0
    normalized = torch.where(nonzero, blocks / magnitude, torch.zeros_like(blocks))
    indices = gefen_nearest_codebook_indices(codebook, normalized)
    vmean = automatic_partition_reduce(second.reshape(-1), period, reduce_op="mean")
    return indices, magnitude.float(), vmean.float()


@dataclass(frozen=True)
class _Slot:
    index: int
    group_index: int
    name: str
    parameter: torch.Tensor


class GefenDCPState:
    """DCP ``Stateful`` wrapper for reshardable plain-Gefen FSDP2 state.

    Use this object as the optimizer value passed to
    :func:`torch.distributed.checkpoint.save` and ``load``. Save dequantizes the
    quantized momentum against the learned per-rank codebook into dense
    ``Shard(0)`` DTensors; load reshards those dense tensors and re-blocks them
    back into Gefen's compact per-block state (re-running the period search,
    relearning the codebook, and re-quantizing on each new local shard), so the
    restored optimizer keeps its ~1 byte/param footprint. Because it goes through
    a dense reshard this path is for *changing* topology; same-topology resumes
    should use the native bit-exact ``state_dict`` path (left unchanged).

    The adapter is intentionally limited to plain :class:`gefen.Gefen` with
    ``factored_v_2d=False`` and ``capturable=False``, one-dimensional
    default-world DTensors, and one ``Shard(0)`` placement.
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

        if getattr(self.optimizer, "capturable", False):
            raise RuntimeError(
                "GefenDCPState does not support capturable=True: the step and "
                "global-step counters are per-device tensors under capturable "
                "(and compiled) execution, so their serialized host-side "
                "semantics differ. Checkpoint with capturable=False."
            )
        if getattr(self.optimizer, "_factored_v_2d", False):
            raise RuntimeError(
                "GefenDCPState does not support factored_v_2d=True: the factored "
                "row/column second moment is not shard-addressable. Construct the "
                "optimizer with factored_v_2d=False for DCP resharding."
            )

        slots = []
        index = 0
        for group_index, group in enumerate(self.optimizer.param_groups):
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
                slots.append(_Slot(index, group_index, str(name), parameter))
                index += 1
        if not slots:
            raise RuntimeError("GefenDCPState requires at least one parameter")
        return tuple(slots)

    @staticmethod
    def _key(slot: _Slot, field: str) -> str:
        return "slot_{:08d}.{}".format(slot.index, field)

    # Required per-parameter Gefen state fields. A slot that carries all of them
    # is materialized; a slot that carries none is fresh (name-only, pre-first
    # step); a slot with only some is partial/corrupted and rejected on save.
    _REQUIRED_STATE_FIELDS = (
        "automatic_period",
        "step",
        "m_codebook",
        "m_magnitude",
        "vmean",
    )

    def _identities(self):
        """Stable per-slot identity list (replicated, in slot order).

        Persists the Gefen parameter name, its group membership, and the global
        parameter shape so a load can reject a target whose parameters were
        registered in a different order (index-only addressing would otherwise
        cross-assign each same-shaped parameter the other's momentum).
        """
        return [
            {
                "name": slot.name,
                "group": slot.group_index,
                "shape": [int(dim) for dim in slot.parameter.shape],
            }
            for slot in self._slots
        ]

    def _group_hypers(self):
        """Per-group hyperparameters (replicated, in group order).

        The compact per-block state is only half of an optimizer resume; the
        native full-state path restores the param-group hyperparameters too. A
        freshly constructed optimizer resumed after an LR-schedule or hyper change
        would otherwise silently keep its constructor values, not the checkpoint's.
        """
        hypers = []
        for group in self.optimizer.param_groups:
            entry = {}
            for key in _GROUP_HYPER_KEYS:
                value = group[key]
                if torch.is_tensor(value):
                    value = value.item()
                entry[key] = float(value)
            hypers.append(entry)
        return hypers

    def _slot_initialized(self, slot: _Slot) -> bool:
        """Classify a slot's live state; reject a partial/malformed materialize.

        Returns True for a fully materialized slot and False for a fresh
        name-only slot. A slot that carries some but not all required fields is a
        corrupted materialize whose missing history must not be silently
        zero-filled, so it raises instead.
        """
        state = self.optimizer.state.get(slot.parameter)
        if not state:
            return False
        present = [key for key in self._REQUIRED_STATE_FIELDS if key in state]
        if not present:
            return False
        if len(present) != len(self._REQUIRED_STATE_FIELDS):
            missing = [
                key for key in self._REQUIRED_STATE_FIELDS if key not in state
            ]
            raise ValueError(
                "Gefen DCP slot {} ({}) has partial optimizer state; missing "
                "{}. Refusing to save it as uninitialized (which would zero the "
                "momentum/second-moment history).".format(
                    slot.index, slot.name, ", ".join(missing)
                )
            )
        return True

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
            "group_count": torch.tensor(
                len(self.optimizer.param_groups), dtype=torch.int64
            ),
            "param_identities": self._identities(),
            "param_group_hypers": self._group_hypers(),
        }
        for slot in self._slots:
            state = self.optimizer.state.get(slot.parameter)
            initialized = self._slot_initialized(slot)
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
        required = {
            "format_version",
            "global_step",
            "deterministic",
            "slot_count",
            "group_count",
            "param_identities",
            "param_group_hypers",
        }
        if not required.issubset(state_dict):
            raise ValueError("Gefen DCP checkpoint metadata is incomplete")
        version = _counter(state_dict["format_version"], "format_version")
        if version != _FORMAT_VERSION:
            raise ValueError(
                "Unsupported Gefen DCP checkpoint format version {}".format(version)
            )
        if _counter(state_dict["slot_count"], "slot_count") != len(self._slots):
            raise ValueError("Gefen DCP checkpoint parameter count differs")

        # Fail closed on a parameter-identity or group-topology mismatch BEFORE
        # touching live state. Index-only slot addressing would otherwise let a
        # target that registered its parameters in a different order load
        # "successfully" while cross-assigning each same-shaped parameter the
        # other's momentum/second-moment.
        if _counter(state_dict["group_count"], "group_count") != len(
            self.optimizer.param_groups
        ):
            raise ValueError("Gefen DCP checkpoint parameter-group count differs")
        if list(state_dict["param_identities"]) != self._identities():
            raise ValueError(
                "Gefen DCP checkpoint parameter identities (name/group/shape) do "
                "not match this optimizer; refusing to load index-addressed state "
                "that could cross-assign momentum between parameters"
            )
        saved_hypers = list(state_dict["param_group_hypers"])
        if len(saved_hypers) != len(self.optimizer.param_groups):
            raise ValueError(
                "Gefen DCP checkpoint parameter-group hyperparameters differ in count"
            )

        # Reject a deterministic-policy mismatch instead of silently overwriting
        # it, matching native Gefen.load_state_dict: the flag changes fused-routing
        # / replica-determinism semantics, so resuming under a different policy
        # requires an intentional migration, not a checkpoint that flips it.
        checkpoint_deterministic = bool(
            _counter(state_dict["deterministic"], "deterministic")
        )
        if checkpoint_deterministic != bool(self.optimizer._deterministic):
            raise ValueError(
                "Gefen DCP checkpoint deterministic={!r}, but this optimizer was "
                "constructed with deterministic={!r}. Resuming under a different "
                "replica-determinism policy requires an intentional state "
                "migration or a fresh optimizer; refusing to change it "
                "silently.".format(
                    checkpoint_deterministic, bool(self.optimizer._deterministic)
                )
            )

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

        # Re-block the resharded dense momentum back into Gefen's compact
        # per-block representation instead of collapsing it to period one. Going
        # through a dense DCP reshard is inherently a new blocking (so this is
        # NOT bit-exact to a native same-topology resume), but it restores the
        # ~1 byte/param memory profile and is a correct, finite continuation:
        # per new local shard we re-run the block-variance period search, relearn
        # the exact codebook, and re-quantize. All of this happens BEFORE the
        # optimizer state is cleared, preserving the fail-atomic load contract.
        #
        # A slot that is initialized in the (replicated) checkpoint metadata but
        # reshards to an EMPTY local shard on this rank -- N->M where dim-0 < the
        # target world -- carries no local momentum to quantize. Native Gefen
        # never materializes state for an empty local shard (the step returns
        # early on an empty grad), so it is restored as name-only here and left
        # out of the codebook learning; feeding an empty shard through the learner
        # would return a None codebook and crash the re-quantize.
        def _materialized(initialized, momentum):
            return initialized and momentum.numel() > 0

        device = _local(self._slots[0].parameter).device
        periods = {}
        for slot, initialized, step, vmean_step, momentum, second in staged:
            if _materialized(initialized, momentum):
                periods[slot.index] = _choose_period(
                    self.optimizer, slot.name, slot.parameter, second
                )
        codebook = _learn_codebook(
            [
                (slot.name, momentum.reshape(-1), periods[slot.index])
                for slot, initialized, _, _, momentum, _ in staged
                if _materialized(initialized, momentum)
            ],
            device,
        )

        new_state = {}
        for slot, initialized, step, vmean_step, momentum, second in staged:
            if not _materialized(initialized, momentum):
                new_state[slot.parameter] = {"name": slot.name}
                continue
            period = periods[slot.index]
            m_codebook, m_magnitude, vmean = _reblock(
                codebook, momentum, second, period
            )
            new_state[slot.parameter] = {
                "name": slot.name,
                "automatic_period": period,
                "step": step,
                "m_codebook": m_codebook,
                "m_magnitude": m_magnitude,
                "vmean": vmean,
                "vmean_step": vmean_step,
            }

        self.optimizer.state.clear()
        self.optimizer.state.update(new_state)
        # Restore the checkpoint's per-group hyperparameters so an LR-schedule or
        # hyperparameter change survives the resume, matching the native full-state
        # path. A live tensor lr is updated in place to preserve its identity/device
        # for any fused kernel holding a reference.
        for group, entry in zip(self.optimizer.param_groups, saved_hypers):
            for key in _GROUP_HYPER_KEYS:
                value = float(entry[key])
                current = group.get(key)
                if torch.is_tensor(current):
                    current.fill_(value)
                else:
                    group[key] = value
        self.optimizer._gefen_global_step = global_step
        # A relearned codebook reflects this rank's resharded momentum and is kept
        # by _maybe_refresh_gefen_codebook on the next step (which then also skips
        # re-predicting periods, keeping the restored block geometry consistent).
        # With no initialized momentum there is nothing to learn from; leave the
        # codebook unset so the next real step learns it cold.
        self.optimizer._gefen_codebook = codebook
        self.optimizer._gefen_codebook_by_device.clear()
        self.optimizer._gefen_codebook_lut_by_device.clear()
        self.optimizer._sr_seed_by_device.clear()
        self.optimizer._reset_gefen_global_step_devices()
        self.optimizer._static_mark_sig = None

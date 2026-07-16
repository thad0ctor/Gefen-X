"""Standalone DCP adapter for reshardable plain-Gefen FSDP2 state."""

from __future__ import annotations

from dataclasses import dataclass
import math

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
    """Dequantize one shard's block state into dense fp32 momentum.

    Uses Gefen's chunked dequantize rather than ``codebook[indices.long()]``.
    Advanced indexing makes a full-size int64 copy of the uint8 indices (8N
    bytes) plus a full-size fp32 gather (4N) on top of the 4N result, so the
    transient peaked at ~16 bytes/param -- ~16 GiB on a 1B-param local shard,
    enough to OOM an otherwise-viable save in the one optimizer whose premise is
    memory efficiency. ``gefen_dequantize_unpacked_indices`` gathers bounded
    element chunks straight into the preallocated output, and the magnitude
    scaling is applied in place, so only the 4N result plus one bounded chunk is
    live. The gathered values, the broadcast multiply, and therefore the saved
    dense momentum are bit-for-bit what advanced indexing produced.
    """
    from gefen.gefen import gefen_dequantize_unpacked_indices

    stored = state["m_codebook"].reshape(-1)
    magnitude = state["m_magnitude"].reshape(-1).float()
    codebook = optimizer._gefen_codebook
    if codebook is None:
        raise RuntimeError("Gefen DCP save requires an initialized codebook")
    codebook = codebook.detach().to(device=stored.device, dtype=torch.float32)
    period = _counter(state["automatic_period"], "automatic_period")
    if period < 1 or stored.numel() != magnitude.numel() * period:
        raise ValueError("Gefen momentum block geometry is invalid")
    # An empty fp32 exemplar: the helper takes the output dtype/device from it
    # (and stays a plain tensor, so no DTensor is reconstructed here).
    like = torch.empty(0, dtype=torch.float32, device=stored.device)
    dense = gefen_dequantize_unpacked_indices(codebook, stored, like)
    return (
        dense.reshape(-1, period)
        .mul_(magnitude.reshape(-1, 1))
        .reshape(_local(parameter).shape)
    )


def _dense_second_moment(parameter, state) -> torch.Tensor:
    """Expand one shard's per-block ``vmean`` into a dense per-element tensor.

    This is a *repeat*, not a reconstruction: Gefen keeps one second-moment value
    per block, so the per-element history this dense form implies never existed.
    Load averages these repeated values into the target blocks, which is exact
    (to fp32 round-off) only while each target block stays inside one source
    block -- i.e. when the target blocking matches or refines the source's. A
    coarser target block averages several source blocks' distinct vmeans
    together, and that loss is irreversible. See the resume-error note in
    COMPATIBILITY.md.
    """
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
    # Co-locate the learned codebook with this shard's operand device. When the
    # default-world DTensors span more than one device (e.g. mixed CPU/CUDA
    # shards) the codebook was learned on the first slot's device, but
    # gefen_nearest_codebook_indices rejects a codebook whose device differs from
    # the operand -- native keeps the codebook on the compute device.
    codebook = codebook.to(device=flat.device, dtype=torch.float32)
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
        seen_identities = {}
        for group_index, group in enumerate(self.optimizer.param_groups):
            if group.get("sharded_mode") is not None:
                raise RuntimeError("GefenDCPState does not support Muon sharded modes")
            names = group.get("param_names")
            # Index-addressed slots are only safe to reshard if each carries a
            # caller-stable, unique identity. A missing/short param_names list or
            # a name Gefen synthesized for an unnamed parameter encodes
            # registration order, not identity, so refuse it rather than silently
            # cross-assigning momentum on a reordered load.
            if names is None or len(names) != len(group["params"]):
                raise RuntimeError(
                    "GefenDCPState requires caller-provided stable param_names on "
                    "every group (group {} has {} name(s) for {} parameter(s)); "
                    "pass named parameters so resharded momentum is assigned by "
                    "identity, not by registration order".format(
                        group_index,
                        0 if names is None else len(names),
                        len(group["params"]),
                    )
                )
            for name, parameter in zip(names, group["params"]):
                # Ask whether Gefen *generated* this name, rather than matching
                # its spelling: the synthesized forms are ordinary identifiers a
                # model may legitimately use for a real parameter, and rejecting
                # a caller's own "param_0" locked such an optimizer out of
                # GefenDCPState entirely. Provenance is recorded at registration.
                if self.optimizer._param_name_is_synthesized(parameter):
                    raise RuntimeError(
                        "GefenDCPState requires caller-provided stable parameter "
                        "names, but slot {} carries the positional name {!r} that "
                        "Gefen synthesized for an unnamed parameter. Construct the "
                        "optimizer with explicit parameter names so resharded "
                        "state is addressed by identity, not registration "
                        "order.".format(index, str(name))
                    )
                if not _is_dtensor(parameter):
                    raise RuntimeError(
                        "GefenDCPState requires every parameter to be a DTensor"
                    )
                mesh = parameter.device_mesh
                if len(mesh.shape) != 1 or mesh.size() != dist.get_world_size():
                    raise RuntimeError(
                        "GefenDCPState supports only a one-dimensional default-world DeviceMesh"
                    )
                # A reordered full-world mesh (e.g. ranks [1, 0]) passes the size
                # check but permutes shard->rank ownership, so the replicated
                # initialized/counter metadata can mis-align with the sharded
                # momentum. The documented scope is the canonical default world;
                # reject anything else.
                if mesh.mesh.flatten().tolist() != list(range(dist.get_world_size())):
                    raise RuntimeError(
                        "GefenDCPState supports only the canonical default-world "
                        "rank order (0..N-1); a reordered mesh is rejected"
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
                identity = (
                    str(name),
                    group_index,
                    tuple(int(dim) for dim in parameter.shape),
                )
                if identity in seen_identities:
                    raise RuntimeError(
                        "GefenDCPState requires unique parameter identities, but "
                        "slots {} and {} share identity (name/group/shape) {!r}; "
                        "resharding cannot disambiguate their momentum.".format(
                            seen_identities[identity], index, identity
                        )
                    )
                seen_identities[identity] = index
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

    # Fields whose presence means the slot was materialized (or partially so).
    # ``vmean_step`` is deliberately NOT *required*: a checkpoint written before
    # the counter existed carries vmean without it, and native Gefen backfills it
    # from ``step`` at step time (the save below mirrors that backfill). But its
    # presence still marks a materialize, so a slot carrying only a name and an
    # orphaned vmean_step is partial -- not fresh. Classifying it as fresh wrote
    # an "uninitialized" slot with a nonzero counter, which every later load
    # rejects as incoherent: a silently dead checkpoint.
    _MATERIALIZED_STATE_FIELDS = _REQUIRED_STATE_FIELDS + ("vmean_step",)

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

    @staticmethod
    def _validate_hyper_entry(group_index, entry):
        """Parse and range-validate one group's saved hyperparameters.

        Mirrors native ``Gefen._validate_group_options`` (lr>=0, 0<=betas<1,
        weight_decay>=0, finite eps>0) and additionally rejects NaN/inf so a
        corrupted checkpoint cannot commit values that silently poison the next
        update. Returns a fresh ``{key: float}`` dict; raises before any mutation.
        """
        if not isinstance(entry, dict):
            raise ValueError(
                "Gefen DCP checkpoint group {} hyperparameters must be a mapping".format(
                    group_index
                )
            )
        parsed = {}
        for key in _GROUP_HYPER_KEYS:
            if key not in entry:
                raise ValueError(
                    "Gefen DCP checkpoint group {} is missing hyperparameter {!r}".format(
                        group_index, key
                    )
                )
            value = entry[key]
            if torch.is_tensor(value):
                if value.numel() != 1:
                    raise ValueError(
                        "Gefen DCP checkpoint group {} hyperparameter {!r} must be "
                        "scalar".format(group_index, key)
                    )
                value = value.item()
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    "Gefen DCP checkpoint group {} hyperparameter {!r} is "
                    "non-numeric".format(group_index, key)
                )
            if not math.isfinite(value):
                raise ValueError(
                    "Gefen DCP checkpoint group {} hyperparameter {!r} is not "
                    "finite ({})".format(group_index, key, value)
                )
            parsed[key] = value
        if parsed["lr"] < 0.0:
            raise ValueError(
                "Gefen DCP checkpoint group {} has invalid lr {}".format(
                    group_index, parsed["lr"]
                )
            )
        if not 0.0 <= parsed["beta1"] < 1.0:
            raise ValueError(
                "Gefen DCP checkpoint group {} has invalid beta1 {}".format(
                    group_index, parsed["beta1"]
                )
            )
        if not 0.0 <= parsed["beta2"] < 1.0:
            raise ValueError(
                "Gefen DCP checkpoint group {} has invalid beta2 {}".format(
                    group_index, parsed["beta2"]
                )
            )
        if parsed["eps"] <= 0.0:
            raise ValueError(
                "Gefen DCP checkpoint group {} has invalid eps {} (must be > 0)".format(
                    group_index, parsed["eps"]
                )
            )
        if parsed["weight_decay"] < 0.0:
            raise ValueError(
                "Gefen DCP checkpoint group {} has invalid weight_decay {}".format(
                    group_index, parsed["weight_decay"]
                )
            )
        return parsed

    @staticmethod
    def _validate_hyper_destinations(group_index, group, entry):
        """Reject a live tensor hyperparameter that cannot represent its value.

        The commit below fills a tensor hyperparameter in place to preserve its
        identity/device for any fused kernel holding a reference. ``fill_`` casts
        to the destination dtype, so restoring a fractional checkpoint lr (1e-3)
        into a one-element *integral* lr tensor truncates it to 0 and silently
        freezes every subsequent update. Checking representability here -- during
        staging, before anything is committed -- keeps that failure fail-atomic
        instead of discovering it after the state was already replaced.
        Floating-point destinations are accepted: rounding an fp64 python float
        into an fp32/bf16 lr tensor is ordinary, pre-existing behavior that
        matches the native path.
        """
        for key in _GROUP_HYPER_KEYS:
            current = group.get(key)
            if not torch.is_tensor(current) or current.is_floating_point():
                continue
            value = entry[key]
            if float(value) != float(int(value)):
                raise ValueError(
                    "Gefen DCP checkpoint group {} restores {}={!r}, but this "
                    "optimizer holds it in a {} tensor that cannot represent it "
                    "(the in-place fill would truncate to {}). Rebuild the "
                    "optimizer with a floating-point {} tensor (or a plain "
                    "float) before resuming; refusing to commit a silently "
                    "truncated value.".format(
                        group_index,
                        key,
                        value,
                        current.dtype,
                        int(value),
                        key,
                    )
                )

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
        present = [key for key in self._MATERIALIZED_STATE_FIELDS if key in state]
        if not present:
            return False
        missing = [key for key in self._REQUIRED_STATE_FIELDS if key not in state]
        if missing:
            raise ValueError(
                "Gefen DCP slot {} ({}) has partial optimizer state; missing "
                "{}. Refusing to save it as uninitialized (which would zero the "
                "momentum/second-moment history).".format(
                    slot.index, slot.name, ", ".join(missing)
                )
            )
        return True

    def state_dict(self):
        # Re-derive the slot layout (and re-run the fail-closed layout gate) so a
        # retained wrapper whose optimizer gained parameters via add_param_group
        # after construction saves the current set, not the stale snapshot.
        self._slots = self._validate_layout()
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
        # Re-derive the slot layout (and re-run the fail-closed layout gate) so a
        # load reflects any add_param_group that ran after construction instead of
        # clearing+repopulating from a stale construction-time snapshot.
        self._slots = self._validate_layout()
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
        # Parse and range-validate every hyperparameter BEFORE any mutation so a
        # missing/non-numeric key or an out-of-range value (lr=NaN, beta1=1,
        # negative eps/weight_decay) is rejected fail-atomically instead of being
        # committed and silently corrupting the next update.
        staged_hypers = [
            self._validate_hyper_entry(group_index, entry)
            for group_index, entry in enumerate(saved_hypers)
        ]
        # Also verify every live destination can hold the value it is about to be
        # given, while the commit is still ahead of us (see the helper).
        for group_index, (group, entry) in enumerate(
            zip(self.optimizer.param_groups, staged_hypers)
        ):
            self._validate_hyper_destinations(group_index, group, entry)

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

        # Stage, validate, and re-block everything locally BEFORE mutating any
        # live optimizer state. The whole preparation runs inside this helper so a
        # per-rank failure (a non-finite resharded slice, an allocation failure,
        # ...) is caught and synchronized across the process group below: either
        # every rank commits or every rank raises. Without the cross-rank
        # agreement a slice that only fails on one target rank would leave some
        # live optimizers restored and others untouched.
        def _stage():
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
                if (
                    not torch.isfinite(momentum).all()
                    or not torch.isfinite(second).all()
                ):
                    raise ValueError("Gefen DCP checkpoint contains non-finite state")
                if (second < 0).any():
                    raise ValueError("Gefen DCP second moment must be nonnegative")
                # Reject incoherent initialization metadata instead of loading it
                # or silently discarding history, matching native Gefen validation
                # (an initialized momentum + its second moment must have positive
                # ages; an uninitialized slot must carry no counters or history).
                # An initialized slot that reshards to an EMPTY local shard keeps
                # its replicated positive counters and is materialized as name-only
                # later, exactly as native never materializes an empty local shard.
                if initialized:
                    if step < 1:
                        raise ValueError(
                            "Gefen DCP slot {} is initialized but carries step={}; "
                            "initialized momentum requires step >= 1".format(
                                slot.index, step
                            )
                        )
                    if vmean_step < 1:
                        raise ValueError(
                            "Gefen DCP slot {} is initialized but carries vmean_step"
                            "={}; initialized second moment requires vmean_step >= "
                            "1".format(slot.index, vmean_step)
                        )
                else:
                    if step != 0 or vmean_step != 0:
                        raise ValueError(
                            "Gefen DCP slot {} is uninitialized but carries nonzero "
                            "counters (step={}, vmean_step={})".format(
                                slot.index, step, vmean_step
                            )
                        )
                    if momentum.numel() and (
                        bool(momentum.any()) or bool(second.any())
                    ):
                        raise ValueError(
                            "Gefen DCP slot {} is uninitialized but carries nonzero "
                            "momentum/second-moment history that would be silently "
                            "discarded".format(slot.index)
                        )
                staged.append((slot, initialized, step, vmean_step, momentum, second))

            global_step = _counter(state_dict["global_step"], "global_step")

            # Re-block the resharded dense momentum back into Gefen's compact
            # per-block representation instead of collapsing it to period one.
            # Going through a dense DCP reshard is inherently a new blocking (so
            # this is NOT bit-exact to a native same-topology resume), but it
            # restores the ~1 byte/param memory profile and is a correct, finite
            # continuation: per new local shard we re-run the block-variance
            # period search, relearn the exact codebook, and re-quantize. All of
            # this happens BEFORE the optimizer state is cleared, preserving the
            # fail-atomic load contract.
            #
            # A slot that is initialized in the (replicated) checkpoint metadata
            # but reshards to an EMPTY local shard on this rank -- N->M where dim-0
            # < the target world -- carries no local momentum to quantize. Native
            # Gefen never materializes state for an empty local shard (the step
            # returns early on an empty grad), so it is restored as name-only here
            # and left out of the codebook learning; feeding an empty shard through
            # the learner would return a None codebook and crash the re-quantize.
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
            return new_state, global_step, codebook

        import torch.distributed as dist
        from gefen.gefen import _synchronize_step_failure

        local_error = None
        prepared = None
        try:
            prepared = _stage()
        except Exception as exc:  # resynchronized across the group below
            local_error = exc
        # Agree on success across the process group before publishing so a
        # one-rank failure aborts every rank rather than committing a partial,
        # cross-rank-inconsistent restore. The plain-Gefen DTensors live on the
        # default world (validated in _validate_layout), so the WORLD group is the
        # right scope; _synchronize_step_failure keeps the flag on CPU for
        # gloo/CPU-resident state and no-ops for a single-rank group.
        if _synchronize_step_failure(local_error is not None, dist.group.WORLD):
            if local_error is not None:
                raise local_error
            raise RuntimeError(
                "Gefen DCP load aborted before commit: another rank failed "
                "staging/validation/re-block, so no rank commits its restore."
            )
        new_state, global_step, codebook = prepared

        self.optimizer.state.clear()
        self.optimizer.state.update(new_state)
        # Restore the checkpoint's per-group hyperparameters so an LR-schedule or
        # hyperparameter change survives the resume, matching the native full-state
        # path. A live tensor lr is updated in place to preserve its identity/device
        # for any fused kernel holding a reference.
        for group, entry in zip(self.optimizer.param_groups, staged_hypers):
            for key in _GROUP_HYPER_KEYS:
                value = entry[key]
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

"""Standalone DCP adapter for reshardable plain-Gefen FSDP2 state."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading

import torch

# Module scope (unlike this file's other deferred imports): the planner/writer
# below subclass these, so they must resolve at class-creation time. This module
# is itself imported lazily by gefen/__init__.py.
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.filesystem import FileSystemWriter


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


def _dense_momentum(optimizer, parameter, state) -> torch.Tensor:
    """Dequantize one shard's block state into dense fp32 momentum.

    Uses Gefen's chunked dequantize rather than ``codebook[indices.long()]``:
    advanced indexing materializes a full-size int64 index copy plus a full-size
    gather on top of the result (~16 bytes/param transient), enough to OOM the
    save this path exists to bound. Bit-identical to what indexing produced.
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
    # Empty fp32 exemplar: the helper takes the output dtype/device from it, and
    # it stays a plain tensor, so no DTensor is reconstructed here.
    like = torch.empty(0, dtype=torch.float32, device=stored.device)
    dense = gefen_dequantize_unpacked_indices(codebook, stored, like)
    return (
        dense.reshape(-1, period)
        .mul_(magnitude.reshape(-1, 1))
        .reshape(_local(parameter).shape)
    )


def _dense_second_moment(parameter, state) -> torch.Tensor:
    """Expand one shard's per-block ``vmean`` into a dense per-element tensor.

    A *repeat*, not a reconstruction: Gefen keeps one value per block, so the
    per-element history this implies never existed. Load re-averages onto the
    target blocks -- coarsening onto source boundaries is exact, refining is not
    (see the resume-error note in COMPATIBILITY.md). Fills a preallocated buffer
    rather than ``expand().reshape().clone()``, which copied twice; the clone
    could not be dropped, because at ``period == 1`` the reshape returns a view
    aliasing the live ``vmean``.
    """
    period = _counter(state["automatic_period"], "automatic_period")
    vmean = state["vmean"].reshape(-1).float()
    local = _local(parameter)
    local_numel = local.numel()
    if period < 1 or vmean.numel() * period != local_numel:
        raise ValueError("Gefen second-moment block geometry is invalid")
    dense = torch.empty(local_numel, dtype=torch.float32, device=vmean.device)
    dense.view(-1, period).copy_(vmean.reshape(-1, 1))
    return dense.reshape(local.shape)


def _validate_dense_geometry(optimizer, parameter, state) -> None:
    """Re-check the dense-expansion invariants without allocating anything.

    The dense forms are materialized lazily at write time, so their own guards
    would fire mid-write, with part of the checkpoint already on disk. These are
    pure scalar/metadata checks, so running them up front costs nothing and keeps
    a malformed optimizer fail-closed before the first byte. The write path still
    enforces every one of them -- this is only an earlier report.
    """
    codebook = optimizer._gefen_codebook
    if codebook is None:
        raise RuntimeError("Gefen DCP save requires an initialized codebook")
    # Whether the indices ADDRESS the codebook, which the size checks say nothing
    # about. uint8 bounds every index to [0, 255], so a 1-D codebook with >= 256
    # rows covers any shard -- O(1), where a max() costs a pass plus a sync.
    if codebook.dim() != 1 or codebook.numel() < 256:
        raise ValueError("Gefen momentum codebook geometry is invalid")
    if state["m_codebook"].dtype != torch.uint8:
        raise ValueError("Gefen momentum index dtype is invalid")
    # A slot straddling devices raises from inside the write anyway, so this only
    # moves the failure earlier and cannot reject a save that would otherwise
    # work. Device identity, not type: cuda:0 and cuda:1 are mismatched too.
    local = _local(parameter)
    if any(
        state[key].device != local.device
        for key in ("m_codebook", "m_magnitude", "vmean")
    ):
        raise ValueError("Gefen optimizer state device is inconsistent")
    period = _counter(state["automatic_period"], "automatic_period")
    local_numel = local.numel()
    codebook_numel = state["m_codebook"].reshape(-1).numel()
    if (
        period < 1
        or codebook_numel != state["m_magnitude"].reshape(-1).numel() * period
        # The index/magnitude ratio says nothing about whether the blocks tile
        # the shard: a self-consistent but short block count still fails in
        # _dense_momentum's reshape, mid-write.
        or codebook_numel != local_numel
    ):
        raise ValueError("Gefen momentum block geometry is invalid")
    if period < 1 or state["vmean"].reshape(-1).numel() * period != local_numel:
        raise ValueError("Gefen second-moment block geometry is invalid")


class _LazyDenseShard(torch.Tensor):
    """A zero-storage stand-in for one slot's dense fp32 shard.

    Resharding must expand the block state to dense fp32 (the codebook is learned
    per rank, so indices are meaningless off their owning rank), and expanding
    every slot up front held 8 bytes/param for the whole write. DCP plans from
    *metadata* and asks for one write item at a time, so this materializes only
    when the writer asks. A ``torch.Tensor`` subclass via
    ``_make_wrapper_subclass`` (real metadata, no storage), because DCP routes to
    ``__get_tensor_shard__`` only for values passing ``isinstance(obj,
    torch.Tensor)``; the three dunders are torch's ``_Checkpointable`` protocol.
    Three details are load-bearing:

    * The shape is the parameter's *global* shape -- DCP's load planner compares
      ``obj.size()`` against the checkpoint's recorded global size.
    * The device must NOT be ``meta`` -- DCP's ``_init_state_dict`` rebuilds meta
      tensors with ``empty_like``, which drops the attributes below.
    * ``__get_tensor_shard__`` caches, because ``state_dict`` also builds DCP's
      ``load`` destination: a fresh tensor per call would be a silent no-op
      restore. Only :class:`GefenSavePlanner` bypasses the cache, which is what
      makes the save bound opt-in.
    """

    @staticmethod
    def __new__(cls, parameter, materialize, device=None):
        """Build the stand-in for ``parameter``'s dense fp32 shard.

        ``materialize`` returns the dense *local* tensor, mirroring
        ``DTensor.__get_tensor_shard__`` -> ``to_local()``: the reported shape is
        global, the resolved data is this rank's shard.
        """
        shard = torch.Tensor._make_wrapper_subclass(
            cls,
            parameter.shape,
            dtype=torch.float32,
            device=_local(parameter).device if device is None else device,
            requires_grad=False,
        )
        shard._gefen_parameter = parameter
        shard._gefen_materialize = materialize
        shard._gefen_cache = None
        return shard

    def __init__(self, parameter, materialize, device=None):
        # torch.Tensor.__init__ takes no arguments; __new__ did the work.
        pass

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # The ONLY layer that can refuse this: `.to(cpu)` on a CPU-resident
        # stand-in short-circuits inside torch's Python binding without
        # dispatching, so __torch_dispatch__ never sees it. Torch 2.5's async
        # staging used it, staging stand-ins by reference for the writer to
        # expand against live state.
        if func in (torch.Tensor.to, torch.Tensor.cpu, torch.Tensor.cuda):
            raise RuntimeError(
                "Gefen DCP stand-in cannot be moved between devices: it carries "
                "no data of its own, so copying it copies nothing and leaves a "
                "reference to live optimizer state behind. This usually means "
                "torch.distributed.checkpoint.async_save was called without "
                "GefenFileSystemWriter -- async staging must expand each slot's "
                "dense form as it copies it to the CPU, which only that writer "
                "does. Pass storage_writer=GefenFileSystemWriter(path) to "
                "async_save, or use the synchronous "
                "torch.distributed.checkpoint.save."
            )
        return super().__torch_function__(func, types, args, kwargs or {})

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # is_pinned is pure metadata and the one operator DCP genuinely needs
        # (TensorProperties.create_from_tensor reads it while planning); the dense
        # form this stands in for is never pinned.
        if func is torch.ops.aten.is_pinned.default:
            return False
        # Names both ways an operator can reach here, which `func` alone cannot
        # distinguish: a stand-in escaping into ordinary tensor code, and an
        # async_save staging with the wrong writer (torch >= 2.6 builds its CPU
        # copy with zeros_like, which lands here).
        raise RuntimeError(
            "Gefen DCP stand-in cannot execute {}: it carries no data of its "
            "own. It stands in for one slot's dense fp32 shard, which is "
            "expanded only through torch.distributed.checkpoint's save/load "
            "protocol, so an operator reaching here means a GefenDCPState "
            "state_dict() is being used as a dict of ordinary tensors. In "
            "particular, torch.distributed.checkpoint.async_save must be given "
            "storage_writer=GefenFileSystemWriter(path): the default writer's "
            "CPU staging copies the whole state dict up front, which is the "
            "materialization the stand-in exists to avoid. To hold real tensors "
            "outside DCP, use the optimizer's own state_dict().".format(func)
        )

    def __repr__(self, *, tensor_contents=None):
        # A repr must never raise, and the default formats by reading elements
        # (aten.select), which __torch_dispatch__ refuses. getattr defends the one
        # path that builds this without __new__ (_init_state_dict -> empty_like).
        return (
            "{}({} Gefen DCP stand-in for dense fp32 optimizer state, "
            "shape={}, dtype={}, device={})".format(
                type(self).__name__,
                "materialized"
                if getattr(self, "_gefen_cache", None) is not None
                else "unmaterialized",
                tuple(self.shape),
                self.dtype,
                self.device,
            )
        )

    def __reduce_ex__(self, protocol):
        # Pickle (torch.save, copy.copy) would otherwise fail inside pickle on the
        # materialize closure, naming nothing the caller can act on.
        raise TypeError(
            "Gefen DCP stand-in cannot be pickled: it carries no data of its "
            "own and is resolved only through torch.distributed.checkpoint's "
            "save/load protocol. Use torch.distributed.checkpoint.save with a "
            "GefenDCPState to write it, or the optimizer's own state_dict() "
            "for an ordinary serializable dict."
        )

    def _gefen_chunk(self):
        """This shard's global offsets/sizes, from the parameter's own chunk list.

        Reuses DTensor's own helper rather than recomputing, so the stand-in's
        geometry cannot drift from the parameter's.
        """
        return self._gefen_parameter.__create_chunk_list__()[0]

    def _gefen_dense_nbytes(self) -> int:
        """Size of this slot's dense shard, from metadata alone.

        Reads the LOCAL shard's element count, not this stand-in's own numel: the
        stand-in reports the parameter's global shape (see ``__new__``).
        """
        return _local(self._gefen_parameter).numel() * self.element_size()

    def _gefen_expand_to_cpu(
        self, staging: "_PinnedStaging | None" = None
    ) -> torch.Tensor:
        """Expand this slot's dense shard onto the CPU, dropping the device copy.

        ``staging`` routes the copy through a reusable page-locked buffer, which
        roughly doubles its bandwidth; what comes back is pageable either way, so
        the caller may hold it at no cost to the host's page-locked supply.
        """
        dense = self._gefen_materialize()
        if dense.device.type == "cpu":
            # Already where the caller wants it (a CPU-resident optimizer, or a
            # slot GefenFileSystemWriter.stage() has snapshotted). Still a private
            # snapshot: materialize expands into a fresh tensor rather than
            # viewing the optimizer's own.
            return dense
        if staging is not None and dense.device.type == "cuda":
            staged = staging.expand(dense)
        else:
            # empty_like, not empty: it preserves the source's memory format, as
            # `.to("cpu")` would. The writer serializes strides, so allocating
            # contiguous would change the saved bytes for a channels_last slot.
            staged = torch.empty_like(dense, device="cpu")
            staged.copy_(dense)
        # Explicit: the device-side expansion must be unreachable before the
        # next one begins, or the bound is one slot per concurrent expansion.
        del dense
        return staged

    def _gefen_resolve_uncached(
        self, staging: "_PinnedStaging | None" = None
    ) -> torch.Tensor:
        """Expand the dense shard for the writer, without retaining it.

        Returns the shard *on the CPU*, which is what lets
        :meth:`GefenSavePlanner.resolve_data`'s lock bound anything: the writer
        holds the result until its write finishes, so a device-side return would
        outlive the lock and N writer threads would hold N of them.
        """
        return self._gefen_expand_to_cpu(staging)

    def _gefen_stage_to_cpu(self) -> "_LazyDenseShard":
        """Snapshot this slot's dense form to the CPU (the bounded async path).

        Drops the device-side tensor before returning, so staging a whole state
        dict holds one slot's dense form at a time. Returns another stand-in, not
        the plain CPU tensor, because the staged dict is what DCP plans from: it
        must keep publishing this slot's *global* shape and chunk offsets.

        Copied into pageable rather than page-locked memory: staging keeps every
        slot's snapshot until the background write drains it, so they must be
        memory the host can swap.
        """
        staged = self._gefen_expand_to_cpu()
        return _LazyDenseShard(
            self._gefen_parameter, lambda: staged, device=staged.device
        )

    def __create_write_items__(self, fqn, obj):
        from torch.distributed.checkpoint.metadata import (
            MetadataIndex,
            TensorProperties,
        )
        from torch.distributed.checkpoint.planner import (
            TensorWriteData,
            WriteItem,
            WriteItemType,
        )

        chunk = self._gefen_chunk()
        return [
            WriteItem(
                index=MetadataIndex(fqn, chunk.offsets),
                type=WriteItemType.SHARD,
                tensor_data=TensorWriteData(
                    chunk=chunk,
                    # Reads dtype/layout/requires_grad/pin_memory, none of which
                    # depend on shape or storage; the stand-in mirrors the dense
                    # tensor in all of them.
                    properties=TensorProperties.create_from_tensor(self),
                    size=self._gefen_parameter.size(),
                ),
            )
        ]

    def __create_chunk_list__(self):
        return [self._gefen_chunk()]

    def __get_tensor_shard__(self, index):
        # Retained on purpose -- see the class docstring: this is the load
        # destination, and it must stay the same buffer across calls.
        if self._gefen_cache is None:
            self._gefen_cache = self._gefen_materialize()
        return self._gefen_cache


def _lazy_momentum(optimizer, parameter, state, initialized):
    """Defer one slot's dense momentum expansion to write time."""

    def materialize():
        if initialized:
            return _dense_momentum(optimizer, parameter, state)
        return torch.zeros_like(_local(parameter), dtype=torch.float32)

    return _LazyDenseShard(parameter, materialize)


def _lazy_second_moment(parameter, state, initialized):
    """Defer one slot's dense second-moment expansion to write time."""

    def materialize():
        if initialized:
            return _dense_second_moment(parameter, state)
        return torch.zeros_like(_local(parameter), dtype=torch.float32)

    return _LazyDenseShard(parameter, materialize)


def _dense_from_checkpoint(value) -> torch.Tensor:
    """Read one dense slot out of a checkpoint value.

    A DCP load fills the stand-ins ``state_dict`` returned and hands the same dict
    back, so this is the stand-in whose retained buffer holds the checkpoint. A
    caller may also feed a ``state_dict()`` straight back into ``load_state_dict``
    with no save/load between; then the buffer holds the current state's dense
    form, which is what the eager save produced there too.
    """
    if isinstance(value, _LazyDenseShard):
        return value.__get_tensor_shard__(None)
    return _local(value)


def _choose_period(optimizer, name: str, parameter, second: torch.Tensor) -> int:
    """Re-derive a compact block period for the resharded local shard.

    Runs Gefen's block-variance period search on the dense second moment (a grad^2
    proxy -- ``vmean`` is the EMA of block-mean grad^2), as the native first step
    runs it on grad^2, so the restored state keeps ~1 byte/param rather than
    collapsing to period one. The optimizer's period-one routing gates are honored
    first, as :meth:`Gefen._resolve_automatic_period` applies them before the raw
    search: otherwise a checkpoint could restore period>1 into a period-one
    optimizer, and the frozen codebook would keep the next step from correcting it.
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
    parameters). Returns ``None`` when there is no initialized momentum.
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
    # Co-locate the codebook with the operand: it was learned on the first slot's
    # device, and gefen_nearest_codebook_indices rejects a device mismatch.
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
    momentum into dense fp32 ``Shard(0)`` shards, one slot at a time as the writer
    asks (see :class:`_LazyDenseShard`); load reshards that dense form and
    re-blocks it into Gefen's compact per-block state, so the restored optimizer
    keeps its ~1 byte/param footprint. Being a dense reshard, this path is for
    *changing* topology; same-topology resumes should use the native bit-exact
    ``state_dict`` path.

    Limited to plain :class:`gefen.Gefen` with ``factored_v_2d=False`` and
    ``capturable=False``, one-dimensional default-world DTensors, and one
    ``Shard(0)`` placement. To bound the peak to a single slot's dense form,
    ``save`` takes ``planner=GefenSavePlanner()`` and ``async_save`` takes
    ``storage_writer=GefenFileSystemWriter(path)``.
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
        seen_names = set()
        seen_identities = {}
        for group_index, group in enumerate(self.optimizer.param_groups):
            if group.get("sharded_mode") is not None:
                raise RuntimeError("GefenDCPState does not support Muon sharded modes")
            names = group.get("param_names")
            # Index-addressed slots are only safe to reshard if each carries a
            # caller-stable, unique identity: a missing/short param_names list
            # encodes registration order, not identity, so a reordered load would
            # silently cross-assign momentum.
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
                # Ask whether Gefen *generated* the name, not whether it looks
                # generated: the synthesized forms are ordinary identifiers a
                # model may legitimately use. Provenance is recorded at
                # registration.
                if self.optimizer._param_name_is_synthesized(parameter):
                    raise RuntimeError(
                        "GefenDCPState requires caller-provided stable parameter "
                        "names, but slot {} carries the positional name {!r} that "
                        "Gefen synthesized for an unnamed parameter. Construct the "
                        "optimizer with explicit parameter names so resharded "
                        "state is addressed by identity, not registration "
                        "order.".format(index, str(name))
                    )
                # The identity is (name, group, shape) with a positional group
                # index, so two same-shaped parameters sharing a name survive a
                # group reorder unchanged and each slot's state lands on the other.
                if name in seen_names:
                    raise RuntimeError(
                        "GefenDCPState requires parameter names to be unique "
                        "across every group, but {!r} names more than one "
                        "parameter. A repeated name is not an identity: it is "
                        "told apart only by the positional group index, so "
                        "rebuilding the groups in a different order would "
                        "validate and still assign each parameter the other's "
                        "state.".format(str(name))
                    )
                seen_names.add(name)
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
                # counter metadata can mis-align with the sharded momentum.
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

    # Presence means materialized (or partially so). ``vmean_step`` is NOT
    # *required* -- a pre-counter checkpoint carries vmean without it and native
    # backfills from ``step`` -- but its presence still marks a materialize, so a
    # slot holding only a name and an orphaned vmean_step is partial, not fresh.
    _MATERIALIZED_STATE_FIELDS = _REQUIRED_STATE_FIELDS + ("vmean_step",)

    def _identities(self):
        """Stable per-slot identity list (replicated, in slot order).

        Persists name, group membership, and global shape so a load can reject a
        target whose parameters were registered in a different order (index-only
        addressing would cross-assign same-shaped parameters' momentum).
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

        The native full-state path restores these too; without them a fresh
        optimizer resumed after an LR-schedule change would silently keep its
        constructor values rather than the checkpoint's.
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
        weight_decay>=0, finite eps>0), and additionally rejects NaN/inf. Returns
        a fresh ``{key: float}`` dict; raises before any mutation.
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

        The commit fills a tensor hyperparameter in place (preserving identity for
        any fused kernel holding a reference), and ``fill_`` casts to the
        destination dtype: a fractional lr into an *integral* lr tensor truncates
        to 0 and freezes every update, while overflow makes ``fill_`` raise only
        after the state was swapped in. Both are rejected during staging.
        """
        for key in _GROUP_HYPER_KEYS:
            current = group.get(key)
            if (
                not torch.is_tensor(current)
                or current.is_floating_point()
                or current.is_complex()
            ):
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
            if current.dtype is torch.bool:
                # torch.iinfo has no bool entry, and bool needs the range check
                # most: it is the one integral dtype whose fill_ never raises, so
                # an out-of-range value coerces to True and commits a wrong lr.
                low, high = 0, 1
            else:
                info = torch.iinfo(current.dtype)
                low, high = info.min, info.max
            if not low <= int(value) <= high:
                raise ValueError(
                    "Gefen DCP checkpoint group {} restores {}={!r}, but this "
                    "optimizer holds it in a {} tensor that cannot represent it "
                    "(outside the representable range [{}, {}]). Rebuild the "
                    "optimizer with a floating-point {} tensor (or a plain "
                    "float) before resuming; refusing to commit a partially "
                    "applied restore.".format(
                        group_index,
                        key,
                        value,
                        current.dtype,
                        low,
                        high,
                        key,
                    )
                )

    def _validate_rank_local(self, state_dict, staged_hypers):
        """Validate the checkpoint against THIS rank's live optimizer.

        Every check here reads rank-local state and so may fail on one rank while
        passing on another. Callers must run it inside the synchronized staging
        block, where a raise becomes a group-wide abort; raising it outside would
        deadlock the peers in the collective. All checks precede the commit.
        """
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

    def _slot_initialized(self, slot: _Slot) -> bool:
        """Classify a slot's live state; reject a partial/malformed materialize.

        True for a fully materialized slot, False for a fresh name-only one. Some
        but not all required fields is a corrupted materialize whose missing
        history must not be silently zero-filled, so it raises.
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
        """Build the checkpoint dict for this rank. **Collective.**

        Every rank must call this together, like ``load_state_dict``; ``dcp.save``
        and ``dcp.load`` both do, and a lone rank-0 call blocks below. The checks
        below read rank-local state, so they can fail on ONE rank while the others
        pass, and ``dcp.save`` runs this BEFORE entering its own planning
        collectives -- a bare raise would drop the failing rank out while its peers
        walk into collectives it never joins. Making the collective conditional is
        not possible: detecting whether peers participate is itself a collective.
        """
        # Re-derive the slot layout so a retained wrapper whose optimizer gained
        # parameters via add_param_group saves the current set. OUTSIDE the
        # synchronized region below: every check it makes is replicated, so it
        # fails on every rank together or none, and it is what establishes that a
        # process group exists at all.
        self._slots = self._validate_layout()

        import torch.distributed as dist

        from gefen.gefen import _synchronize_step_failure

        local_error = None
        result = None
        try:
            result = self._build_state_dict()
        except Exception as exc:  # resynchronized across the group below
            local_error = exc
        # Agree across the group before any rank returns, so a one-rank failure
        # aborts every rank instead of stranding the healthy ones in the planning
        # collectives that follow. The plain-Gefen DTensors live on the default
        # world (validated above), and _synchronize_step_failure no-ops for a
        # single-rank group.
        if _synchronize_step_failure(local_error is not None, dist.group.WORLD):
            if local_error is not None:
                raise local_error
            # Neutral about save vs load: dcp.load calls state_dict() too, to
            # build its destination.
            raise RuntimeError(
                "Gefen DCP state_dict aborted before any rank proceeds: another "
                "rank's optimizer state failed validation, so no rank continues "
                "into the save or load."
            )
        return result

    def _build_state_dict(self):
        """Rank-local half of :meth:`state_dict`; every raise here is synchronized."""
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
            # Check the dense-expansion invariants now, while nothing has been
            # written, rather than letting them fire from inside a write.
            if initialized:
                _validate_dense_geometry(self.optimizer, slot.parameter, state)
            # Stand-ins, not tensors: expanding every slot here held 8 bytes/param
            # live for the whole write. The stand-ins carry the planning metadata
            # and expand one at a time as the writer asks for them.
            result[self._key(slot, "momentum")] = _lazy_momentum(
                self.optimizer, slot.parameter, state, initialized
            )
            result[self._key(slot, "second_moment")] = _lazy_second_moment(
                slot.parameter, state, initialized
            )
        return result

    def load_state_dict(self, state_dict):
        # Re-derive the slot layout so a load reflects any add_param_group that
        # ran after construction, rather than a stale construction-time snapshot.
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

        # Fail closed on an identity or group-topology mismatch BEFORE touching
        # live state: index-only addressing would otherwise let a reordered target
        # load "successfully" while cross-assigning same-shaped parameters.
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
        # Parse and range-validate every hyperparameter BEFORE any mutation, so a
        # missing/non-numeric key or an out-of-range value is rejected
        # fail-atomically rather than committed.
        staged_hypers = [
            self._validate_hyper_entry(group_index, entry)
            for group_index, entry in enumerate(saved_hypers)
        ]
        # Validation that reads the LIVE optimizer is deliberately NOT done here:
        # it is rank-local, so raising it outside the synchronized block below
        # would let the failing rank exit while its peers block forever in the
        # collective. It runs inside that block (_validate_rank_local).

        # Stage, validate, and re-block everything locally BEFORE mutating any
        # live optimizer state, inside this helper so a per-rank failure is caught
        # and synchronized below: either every rank commits or every rank raises.
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
                momentum = (
                    _dense_from_checkpoint(state_dict[keys["momentum"]])
                    .detach()
                    .float()
                )
                second = (
                    _dense_from_checkpoint(state_dict[keys["second_moment"]])
                    .detach()
                    .float()
                )
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
                # Reject incoherent initialization metadata rather than silently
                # discarding history, matching native Gefen validation. An
                # initialized slot that reshards to an EMPTY local shard keeps its
                # replicated positive counters and is materialized as name-only
                # below, as native never materializes an empty local shard.
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

            # Re-block rather than collapse to period one: a dense reshard is a
            # new blocking (so NOT bit-exact to a native same-topology resume) but
            # restores the ~1 byte/param profile. Runs BEFORE the state is
            # cleared, preserving fail-atomic load.
            #
            # A slot initialized in the replicated metadata but resharding to an
            # EMPTY local shard has no momentum to quantize. Native never
            # materializes an empty local shard, so restore it name-only and keep
            # it out of the codebook learning -- an empty shard returns a None
            # codebook and crashes the re-quantize.
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
            self._validate_rank_local(state_dict, staged_hypers)
            prepared = _stage()
        except Exception as exc:  # resynchronized across the group below
            local_error = exc
        # Agree on success across the group before publishing, so a one-rank
        # failure aborts every rank rather than committing a partial,
        # cross-rank-inconsistent restore. The plain-Gefen DTensors live on the
        # default world (validated in _validate_layout), so WORLD is the scope.
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
        # Restore the checkpoint's per-group hyperparameters, matching the native
        # full-state path. A live tensor lr is filled in place to preserve its
        # identity/device for any fused kernel holding a reference.
        for group, entry in zip(self.optimizer.param_groups, staged_hypers):
            for key in _GROUP_HYPER_KEYS:
                value = entry[key]
                current = group.get(key)
                if torch.is_tensor(current):
                    current.fill_(value)
                else:
                    group[key] = value
        self.optimizer._gefen_global_step = global_step
        # A relearned codebook is kept by _maybe_refresh_gefen_codebook on the
        # next step, which then also skips re-predicting periods, keeping the
        # restored block geometry consistent. With no initialized momentum there
        # is nothing to learn from; leave it unset so the next step learns cold.
        self.optimizer._gefen_codebook = codebook
        self.optimizer._gefen_codebook_by_device.clear()
        self.optimizer._gefen_codebook_lut_by_device.clear()
        self.optimizer._sr_seed_by_device.clear()
        self.optimizer._reset_gefen_global_step_devices()
        self.optimizer._static_mark_sig = None


class _PinnedStaging:
    """One reusable page-locked buffer that every synchronous D2H copy passes through.

    A page-locked destination roughly doubles a D2H copy's bandwidth, but handing
    the page-locked tensor *itself* to the writer would leave the whole dense state
    dict page-locked -- ``FileSystemWriter`` holds every tensor until the file
    closes, and the caching host allocator never returns blocks to the OS. Copying
    through one buffer and handing back pageable memory keeps the bandwidth at a
    one-slot page-locked footprint. Sized ONCE -- see :meth:`reserve`.
    """

    def __init__(self) -> None:
        self._buffer: torch.Tensor | None = None
        self._capacity = 0

    def reserve(self, nbytes: int) -> None:
        """Declare the largest slot this save will stage, before it starts.

        Growing the buffer on demand would not hold the one-slot bound: torch's
        CachingHostAllocator never returns freed pinned blocks to the OS, so each
        new maximum ADDS a page-locked block. ``FileSystemWriter`` resolves write
        items smallest first, so a save with N distinct slot sizes walks the maxima
        in exactly the order that allocates every one of them, pinning their sum.
        """
        self._capacity = nbytes

    def expand(self, dense: torch.Tensor) -> torch.Tensor:
        """Copy ``dense`` to pageable CPU memory via the page-locked buffer."""
        # empty_like preserves the source's memory format (the writer serializes
        # strides, so contiguous would change a channels_last slot's saved bytes)
        # and fixes the strides the staging view below must match.
        out = torch.empty_like(dense, device="cpu")
        nbytes = out.numel() * out.element_size()
        if self._buffer is None or self._buffer.numel() < nbytes:
            # Allocated once, at the reserved size: cudaHostAlloc runs
            # milliseconds, so re-pinning per slot would dominate the copy it
            # accelerates. max(): reserve() covers every slot this planner
            # stages, so the capacity normally wins; falling back to the request
            # keeps a stand-in reserve() never saw correct, merely unbounded.
            self._buffer = torch.empty(
                max(nbytes, self._capacity), dtype=torch.uint8, pin_memory=True
            )
        pinned = torch.empty(0, dtype=out.dtype, device="cpu")
        pinned.set_(self._buffer.untyped_storage(), 0, out.shape, out.stride())
        pinned.copy_(dense)
        out.copy_(pinned)
        return out


class GefenSavePlanner(DefaultSavePlanner):
    """``SavePlanner`` that keeps a :class:`GefenDCPState` save memory-bounded.

    Optional; pass it to ``dcp.save`` to opt a :class:`GefenDCPState` into the
    bound::

        dcp.save(
            {"model": model, "optimizer": GefenDCPState(optimizer)},
            storage_writer=dcp.FileSystemWriter(path),
            planner=GefenSavePlanner(),
        )

    Expands one stand-in at a time, when the writer asks for that write item, and
    does not keep it, so the save's *device* peak is the resident state plus a
    single slot's dense form. The bound holds at any ``thread_count``. Host memory
    stays the writer's to bound (torch's holds every tensor until it closes the
    file); what is bounded here is the *page-locked* part. Omitting the planner is
    still *correct*, at the same 8 bytes/param the eager implementation cost.
    ``flatten_state_dict=False`` is rejected -- see :meth:`__init__`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gefen_expand_lock = threading.Lock()
        # Lazily allocates, so a CPU-resident optimizer page-locks nothing.
        self._gefen_staging = _PinnedStaging()
        # A GefenDCPState constraint, not just this planner's: DCP descends into a
        # nested mapping only when it flattens it, so unflattened the whole
        # {"optimizer": ...} mapping becomes one BYTE_IO item pickled wholesale
        # and the per-slot write items never exist for resolve_data to intercept.
        if not self.flatten_state_dict:
            raise ValueError(
                "GefenSavePlanner requires flatten_state_dict=True (the default), "
                "but got flatten_state_dict=False. A GefenDCPState hands DCP "
                "zero-storage stand-ins that are addressed by their flattened "
                "dotted FQN; without flattening DCP never creates their write "
                "items and pickles the whole optimizer mapping instead, which a "
                "stand-in cannot satisfy. Save a GefenDCPState with a flattened "
                "state dict, or use the optimizer's native state_dict() if you "
                "need an unflattened layout."
            )

    def set_up_planner(self, state_dict, storage_meta=None, is_coordinator=False):
        """Override of ``SavePlanner.set_up_planner``: size the staging buffer.

        The arguments are named out in full, not taken as ``*args``: ``dcp.save``
        inspects this signature and, not finding ``storage_meta``, falls back to
        the pre-2.3 call ``set_up_planner(state_dict, is_coordinator)`` -- leaving
        ``is_coordinator`` False on every rank, including the one that builds the
        global plan. A ``*args`` passthrough is invisible to that check.
        """
        super().set_up_planner(state_dict, storage_meta, is_coordinator)
        # Every slot is known here, before the writer starts, so the buffer can be
        # sized to the largest in one pass over metadata (see reserve). Only CUDA
        # slots stage, so a CPU-resident optimizer -- and an async save, whose
        # slots arrive already snapshotted -- page-locks nothing.
        largest = 0
        for entry in self.state_dict.values():
            if isinstance(entry, _LazyDenseShard) and entry.device.type == "cuda":
                largest = max(largest, entry._gefen_dense_nbytes())
        self._gefen_staging.reserve(largest)

    def resolve_data(self, write_item):
        # Deliberately NOT via lookup_object(): that resolves through
        # find_tensor_shard -> __get_tensor_shard__, which retains the buffer for
        # the load path and would defeat the bound. Read the entry straight out of
        # the (already flattened) state dict instead.
        entry = self.state_dict.get(write_item.index.fqn)
        if isinstance(entry, _LazyDenseShard):
            # FileSystemWriter(thread_count=N) runs N of these concurrently, so
            # without the lock N slots expand at once and the bound becomes N
            # slots' dense form. It also guards the staging buffer: what comes
            # back is a private pageable tensor, so the next expansion may
            # overwrite the buffer as soon as this one releases.
            with self._gefen_expand_lock:
                return entry._gefen_resolve_uncached(self._gefen_staging)
        return super().resolve_data(write_item)


def _stage_lazy_shards(state_dict, staged, prefix=()):
    """Copy ``state_dict`` without its stand-ins, snapshotting each to the CPU.

    Records ``path -> CPU-resident stand-in`` in ``staged`` and returns a copy of
    the tree with those entries removed, so torch's own staging never sees them.
    Copies rather than mutates: the dict belongs to the ``async_save`` caller.
    """
    import copy

    stripped = copy.copy(state_dict)
    for key, value in state_dict.items():
        path = prefix + (key,)
        if isinstance(value, _LazyDenseShard):
            staged[path] = value._gefen_stage_to_cpu()
            del stripped[key]
        elif isinstance(value, dict):
            stripped[key] = _stage_lazy_shards(value, staged, path)
    return stripped


def _restore_staged_shards(state_dict, staged):
    """Put the CPU-resident stand-ins back at their paths in the staged tree.

    Rebuilds the dicts along those paths instead of mutating them, because what
    torch's staging returns may be a buffer it caches and reuses across saves.
    """
    import copy

    result = copy.copy(state_dict)
    rebuilt = {(): result}
    for path, shard in staged.items():
        node = result
        for depth, key in enumerate(path[:-1]):
            branch = path[: depth + 1]
            if branch not in rebuilt:
                rebuilt[branch] = copy.copy(node[key])
                node[key] = rebuilt[branch]
            node = rebuilt[branch]
        node[path[-1]] = shard
    return result


class GefenFileSystemWriter(FileSystemWriter):
    """``FileSystemWriter`` that stages a :class:`GefenDCPState` for async save.

    Required by ``async_save`` whenever the state dict contains a
    :class:`GefenDCPState`::

        response = dcp.async_save(
            {"model": model, "optimizer": GefenDCPState(optimizer)},
            storage_writer=GefenFileSystemWriter(path),
            planner=GefenSavePlanner(),
        )
        # torch 2.5 returns a bare Future, newer torch an AsyncSaveResponse;
        # waiting is what surfaces a background write failure.
        getattr(response, "upload_completion", response).result()

    ``async_save`` returns once the state dict has been *staged* -- copied to the
    CPU -- and writes from a background thread. The default writer stages by
    asking each entry for a CPU copy of itself, which a stand-in cannot satisfy;
    this one expands the slots itself, one at a time. The synchronous save does
    not stage and does not need it.
    """

    def stage(self, state_dict):
        """Override of ``AsyncStager.stage``."""
        staged_shards: dict = {}
        stripped = _stage_lazy_shards(state_dict, staged_shards)
        return _restore_staged_shards(super().stage(stripped), staged_shards)

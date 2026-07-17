"""Standalone DCP adapter for reshardable plain-Gefen FSDP2 state."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading

import torch

# Imported at module scope (unlike the rest of this file's deferred imports)
# because GefenSavePlanner and GefenFileSystemWriter subclass them, so they must
# resolve at class-creation time. This module is the DCP adapter and is itself
# only imported on demand (gefen/__init__.py exposes it lazily), so nothing pays
# for it unless DCP is actually being used.
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
    Load re-averages these repeated values onto the target blocks. Because vmean
    is an EMA of the block's mean grad^2 and both are linear, a target block that
    is a union of whole source blocks recovers exactly what a natively
    target-blocked run holds (to fp32 round-off) -- coarsening onto source
    boundaries is not the lossy direction. The loss is the other way: a target
    block inside a source block (or straddling one) can only be handed the single
    stored vmean, where a native run would hold a distinct value per sub-block.
    See the resume-error note in COMPATIBILITY.md.

    Broadcasting into one preallocated output rather than
    ``expand(...).reshape(...).clone()``: reshaping a stride-0 expand cannot
    return a view, so it already materialized a full 4 byte/param copy that the
    clone then copied a *second* time -- 8 bytes/param live for a 4 byte/param
    result, which is the whole transient this save path is trying to bound. The
    clone could not simply be dropped, because at ``period == 1`` the expand is a
    no-op and the reshape *does* return a view aliasing the live ``vmean`` state.
    Filling an owned buffer is single-copy at every period and never aliases, and
    copying values is bit-identical to repeating them.
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

    The dense momentum/second moment are materialized lazily, one at a time, at
    DCP write time (see :class:`_LazyDenseShard`). Their guards would therefore
    fire in the middle of a write, once part of the checkpoint is already on
    disk. These are pure scalar/metadata checks -- a missing codebook, one the
    indices cannot address, a nonpositive period, a block count that does not
    divide the local shard -- so running them up front in ``state_dict`` costs
    nothing and keeps a malformed optimizer fail-closed *before* the first byte
    is written, matching the fail-before-mutation contract the load path holds
    itself to. The write path enforces every one of these conditions again --
    the materializers re-check the geometry, and ``index_select`` bounds-checks
    the gather itself -- so those remain the real correctness barrier; this is
    only an early, cheaper report of the same conditions, before a partial
    checkpoint exists.
    """
    codebook = optimizer._gefen_codebook
    if codebook is None:
        raise RuntimeError("Gefen DCP save requires an initialized codebook")
    # Whether the indices ADDRESS that codebook, which the size checks below say
    # nothing about: _dense_momentum gathers with index_select, so a codebook too
    # short for an index present in the shard raises IndexError from inside the
    # write. Checking the dtype and the row count settles it in O(1), without
    # reducing over the indices: they are uint8 (_init_gefen_state allocates them
    # so), which bounds every index to [0, 255] by construction, and a 1-D
    # codebook with at least 256 rows covers that range whatever the shard holds.
    # A max() here would instead cost a full pass plus a device sync per slot on
    # every save, and would still pass a truncated codebook whose live indices
    # happened to be small -- leaving the same failure latent for the next step.
    if codebook.dim() != 1 or codebook.numel() < 256:
        raise ValueError("Gefen momentum codebook geometry is invalid")
    if state["m_codebook"].dtype != torch.uint8:
        raise ValueError("Gefen momentum index dtype is invalid")
    # Whether the block state agrees with itself about where it lives, which the
    # counts say nothing about either. _dense_momentum scales the gathered values
    # by m_magnitude in place, and _dense_second_moment fills a buffer on vmean's
    # device that is then wrapped against the parameter's mesh, so a slot whose
    # tensors straddle devices raises from inside the write. This cannot reject a
    # save that would otherwise have worked: those same operations would fail on
    # any mismatch it catches, so the check only moves the failure ahead of the
    # first byte. Device identity, not just type -- cuda:0 and cuda:1 are as
    # mismatched here as cuda and cpu.
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
        # The index/magnitude ratio holding says nothing about whether the blocks
        # tile the shard: _dense_momentum reshapes the dequantized indices to the
        # local shape, so a block count that is self-consistent but short (both
        # arrays scaled down together) still fails there, mid-write.
        or codebook_numel != local_numel
    ):
        raise ValueError("Gefen momentum block geometry is invalid")
    if period < 1 or state["vmean"].reshape(-1).numel() * period != local_numel:
        raise ValueError("Gefen second-moment block geometry is invalid")


class _LazyDenseShard(torch.Tensor):
    """A zero-storage stand-in for one slot's dense fp32 shard.

    Resharding forces the save to expand Gefen's ~1 byte/param block state into a
    dense 4 byte/param momentum plus a dense 4 byte/param second moment, because
    the momentum codebook is learned *per rank* (rank 3's index 47 is not rank
    5's index 47), so the indices are meaningless off their owning rank. Building
    every slot's dense pair up front and handing DCP a finished dict held 8
    bytes/param for the whole write -- eight times the optimizer's resident state,
    and a higher peak than a training step -- which defeats the one optimizer
    whose premise is memory efficiency.

    DCP never needs those tensors simultaneously. It plans from *metadata* and
    then asks for one write item's data at a time, so this stands in for the
    dense tensor during planning and materializes it only when the writer asks.
    :meth:`GefenSavePlanner.resolve_data` expands one at a time and hands the
    writer the result already on the CPU, so no device-side dense form outlives
    the call that built it: exactly one is live at a time, whatever the writer's
    thread count, and the save's device peak falls to the resident state plus a
    single slot's dense form.

    The instance is a ``torch.Tensor`` subclass rather than a plain descriptor
    object because DCP resolves data through ``find_state_dict_object`` ->
    ``find_tensor_shard``, which routes to the ``__get_tensor_shard__`` hook
    below only for values that pass ``isinstance(obj, torch.Tensor)``; a
    non-tensor descriptor is rejected there instead. The three dunders implement
    torch's ``_Checkpointable`` protocol -- the same interface DTensor itself
    hooks into DCP with -- so planning reads this stand-in's metadata natively and
    composes with the model and any other Stateful items in the caller's state
    dict.

    It is built with ``_make_wrapper_subclass``, which yields a tensor that
    reports a real shape/dtype/device but owns no allocated storage (its data
    pointer is null), so a slot costs nothing until it is expanded. Three
    properties of that construction are load-bearing:

    * The shape is the parameter's *global* shape, matching what the eager
      DTensor reported. DCP's load planner compares ``obj.size()` against the
      checkpoint's recorded global size and rejects a mismatch, so a stand-in
      that described only its local shard would fail every load.
    * The device must NOT be ``meta``. DCP's load planner runs
      ``_init_state_dict`` over the destination and rebuilds every meta tensor
      with ``empty_like``, which preserves this subclass but drops the attributes
      below, replacing a stand-in with a plain non-DTensor buffer whose geometry
      no longer describes a shard.
    * ``__torch_dispatch__`` refuses every real operation. The stand-in has no
      data, so anything that tries to compute with it is a bug; failing loudly
      there is what keeps a bypassed expansion from quietly writing uninitialized
      bytes into a checkpoint.

    ``state_dict`` is the *same* method DCP calls to build the destination of a
    ``load``, so a stand-in must also work as a load destination. That is why
    ``__get_tensor_shard__`` allocates once and keeps the buffer: DCP reads the
    checkpoint into whatever that hook returns, so handing back a fresh tensor
    per call would let the load copy into a temporary that is dropped on the
    floor -- a silent no-op restore. Caching makes the default path behave
    exactly like the eager dense tensor the pre-fix save handed DCP: correct for
    load, and correct (if still 8 bytes/param) for a save that does not pass
    :class:`GefenSavePlanner`. Only that planner bypasses the cache, which is
    what makes the save bound opt-in rather than automatic.
    """

    @staticmethod
    def __new__(cls, parameter, materialize, device=None):
        """Build the stand-in for ``parameter``'s dense fp32 shard.

        ``materialize`` is a zero-argument callable returning the dense *local*
        tensor, mirroring ``DTensor.__get_tensor_shard__`` -> ``to_local()``:
        the reported shape is global, the resolved data is this rank's shard.

        ``device`` defaults to the parameter's own device -- where the dense form
        would be expanded. :meth:`_gefen_stage_to_cpu` overrides it to build the
        CPU-resident stand-in that async staging leaves in the staged dict, whose
        dense form is already on the CPU.
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
        # Refuses the device movement that a stand-in must never undergo, and is
        # the ONLY layer that can: `t.to(cpu)` on a stand-in already on the CPU
        # short-circuits to `self` inside torch's Python binding and never
        # dispatches an operator, so __torch_dispatch__ below cannot see it.
        #
        # That silent pass-through is not hypothetical. Torch 2.5's async staging
        # copies with `.to(cpu_device)`, so CPU-resident state staged that way
        # left the stand-ins in the "staged" dict by reference: nothing was
        # copied, and the writer thread then expanded them against LIVE optimizer
        # state while training continued -- a torn snapshot racing step(), which
        # is precisely the guarantee async_save exists to provide. It only looked
        # like it worked because the state happened to be quiescent. Refusing
        # here converts that into the same loud failure every other torch/device
        # combination already gives, and points at the writer that stages these
        # correctly.
        #
        # Everything else falls through to the default, which routes real
        # operators to __torch_dispatch__ (is_pinned is answered there, and the
        # rest are refused with the message below).
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
        # is_pinned is pure metadata and is the one operator DCP genuinely needs
        # (TensorProperties.create_from_tensor reads it while planning); the
        # dense form this stands in for is never pinned.
        if func is torch.ops.aten.is_pinned.default:
            return False
        # Names both ways an operator can reach here, because the two are not
        # distinguishable from `func` alone: a stand-in escaping into ordinary
        # tensor code, and an async_save staging the dict with the wrong writer
        # (torch >= 2.6 builds its CPU copy with zeros_like, which lands here).
        # Deliberately does NOT suggest passing a planner: the planner bounds the
        # save's peak, it is not what resolves the stand-in, and a caller who
        # already passed one is told to do again the thing that did not help.
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
        # torch.Tensor.__repr__ formats by reading elements (aten.select), which
        # __torch_dispatch__ refuses -- so printing a state dict, or logging one
        # from an exception handler, raised a secondary error that buried the
        # real one. Report the metadata the stand-in genuinely has instead; a
        # repr must never raise. getattr defends the one path that builds this
        # subclass without __new__ (DCP's _init_state_dict -> empty_like), where
        # the attributes below are absent.
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
        # Pickle (torch.save, copy.copy) would otherwise fail inside pickle on
        # the materialize closure -- "Can't get local object
        # '_lazy_momentum.<locals>.materialize'" -- which names nothing the
        # caller can act on. The real constraint is that this holds a reference
        # to live optimizer state, not data, so it is not serializable at all
        # outside DCP's protocol.
        raise TypeError(
            "Gefen DCP stand-in cannot be pickled: it carries no data of its "
            "own and is resolved only through torch.distributed.checkpoint's "
            "save/load protocol. Use torch.distributed.checkpoint.save with a "
            "GefenDCPState to write it, or the optimizer's own state_dict() "
            "for an ordinary serializable dict."
        )

    def _gefen_chunk(self):
        """This shard's global offsets/sizes.

        Taken from the *parameter*'s own chunk list rather than recomputed: the
        eager path wrapped the dense tensor in a DTensor built from the
        parameter's mesh/placements/shape, so the parameter's chunk geometry is
        by construction the geometry the dense slot had, and reusing DTensor's
        own helper keeps the two from drifting apart.
        """
        return self._gefen_parameter.__create_chunk_list__()[0]

    def _gefen_dense_nbytes(self) -> int:
        """Size of this slot's dense shard, from metadata alone.

        Reads the LOCAL shard's element count, not this stand-in's own numel:
        the stand-in reports the parameter's global shape (see ``__new__``), so
        its numel is the whole tensor's, not this rank's share of it.
        """
        return _local(self._gefen_parameter).numel() * self.element_size()

    def _gefen_expand_to_cpu(
        self, staging: "_PinnedStaging | None" = None
    ) -> torch.Tensor:
        """Expand this slot's dense shard onto the CPU, dropping the device copy.

        ``staging`` routes the device-to-host copy through a reusable page-locked
        buffer, which roughly doubles the copy's bandwidth without page-locking
        what comes back: the result is ordinary pageable memory, so the caller
        may hold it for as long as it likes at no cost to the host's supply of
        page-locked pages. Without it the copy lands straight in pageable memory,
        which is slower but allocates nothing that outlives the shard.
        """
        dense = self._gefen_materialize()
        if dense.device.type == "cpu":
            # Already where the caller wants it -- a CPU-resident optimizer, or
            # a slot GefenFileSystemWriter.stage() has already snapshotted.
            # Copying again would double the CPU cost to no purpose. It is still
            # a private snapshot: materialize expands the block state into a
            # fresh tensor rather than viewing the optimizer's own.
            return dense
        if staging is not None and dense.device.type == "cuda":
            staged = staging.expand(dense)
        else:
            # empty_like rather than empty: it preserves the source's memory
            # format, as `.to("cpu")` would. An uninitialized slot expands with
            # zeros_like, which keeps a channels_last parameter channels_last,
            # and the writer serializes strides -- so allocating contiguous here
            # would quietly change the saved bytes for those slots.
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

        Returns the shard *on the CPU*, which is where the writer needs it and
        the only way :meth:`GefenSavePlanner.resolve_data`'s lock can bound
        anything: the writer holds what this returns until it has finished
        writing that item, so a dense form returned on the device outlives any
        lock held across this call, and N writer threads would hold N of them.
        Landing on the CPU here means the device-side expansion is dead before
        the lock is released.

        The copy is necessarily synchronous -- the writer may write this tensor
        the moment it is returned -- which gives up the copy/expand overlap
        torch's single-threaded loader gets from its own non_blocking copy.
        Staging through a page-locked buffer buys that bandwidth back, and
        returning the shard in pageable memory keeps the buffer's page-locked
        cost at one slot's dense form however long the writer holds the result.
        """
        return self._gefen_expand_to_cpu(staging)

    def _gefen_stage_to_cpu(self) -> "_LazyDenseShard":
        """Snapshot this slot's dense form to the CPU (the bounded async path).

        Expands the dense shard, copies it to the CPU, and drops the device-side
        tensor before returning, so staging a whole state dict holds one slot's
        dense form on the device at a time rather than every slot's at once --
        the same bound the synchronous save gets from
        :meth:`GefenSavePlanner.resolve_data`, applied to the copy that makes the
        write asynchronous.

        The result is another stand-in rather than the plain CPU tensor because
        the staged dict is what DCP then plans and writes from: it has to keep
        publishing this slot's *global* shape and chunk offsets, which a plain
        local tensor no longer describes. Its dense form is already resolved, so
        it holds real data and expands nothing further.

        The staged shard reads its geometry from the same live parameter as this
        one, which is metadata only (shape/mesh/placements) and does not change
        under a concurrent step. The snapshot itself -- the dense values -- is
        fully detached from the optimizer once this returns, because
        :meth:`_gefen_expand_to_cpu` builds it by expanding the block state into
        a fresh tensor rather than by viewing the optimizer's own.

        Copied straight into pageable memory rather than through the synchronous
        path's page-locked staging buffer: staging keeps every slot's snapshot
        until the background write drains it, so the snapshots must be memory
        the host can swap.
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
                    # Reads dtype/layout/requires_grad/pin_memory -- none of which
                    # depend on shape or storage -- off this stand-in, which
                    # mirrors the dense tensor in all of them, so they match what
                    # the eager DTensor's local shard reported.
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

    A DCP load fills the stand-ins returned by ``state_dict`` and hands the same
    dict back, so the value here is the stand-in whose retained buffer DCP just
    read the checkpoint into -- ``__get_tensor_shard__`` returns exactly that
    buffer. A caller can also feed a ``state_dict()`` straight back into
    ``load_state_dict`` with no save/load in between (an in-memory reset that
    several tests exercise); then the buffer has not been filled and holds the
    dense form of the optimizer's current state, which is what the eager save
    produced there too. Both cases are one call.
    """
    if isinstance(value, _LazyDenseShard):
        return value.__get_tensor_shard__(None)
    return _local(value)


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
    quantized momentum against the learned per-rank codebook into dense fp32
    ``Shard(0)`` shards -- expanded one slot at a time as the writer asks for
    them (see :class:`_LazyDenseShard`) rather than handed to DCP as finished
    tensors, which is what keeps the save's peak bounded; load reshards that
    dense form and re-blocks it back into Gefen's compact per-block state
    (re-running the period search, relearning the codebook, and re-quantizing on
    each new local shard), so the restored optimizer keeps its ~1 byte/param
    footprint. Because it goes through
    a dense reshard this path is for *changing* topology; same-topology resumes
    should use the native bit-exact ``state_dict`` path (left unchanged).

    The adapter is intentionally limited to plain :class:`gefen.Gefen` with
    ``factored_v_2d=False`` and ``capturable=False``, one-dimensional
    default-world DTensors, and one ``Shard(0)`` placement.

    Both :func:`torch.distributed.checkpoint.save` and
    :func:`torch.distributed.checkpoint.async_save` write this state, and each
    takes one Gefen-specific argument to keep its peak bounded to a single slot's
    dense form: the synchronous save takes ``planner=GefenSavePlanner()``, and
    ``async_save`` takes ``storage_writer=GefenFileSystemWriter(path)`` (see
    :class:`GefenFileSystemWriter`, which stages the dense slots to the CPU one
    at a time so training can resume against a snapshot taken at the moment of
    the call). Given anything else, ``async_save`` raises rather than stage a
    dict it cannot copy.
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
                # A name has to identify a parameter on its own. The saved
                # identity is (name, group, shape), and the group is a positional
                # index -- so two same-shaped parameters sharing a name across
                # groups produce an identity list that is unchanged when those
                # groups are rebuilt in the opposite order. Validation would pass
                # and each slot's state would land on the other parameter, which
                # is the silent cross-assignment identities exist to prevent. The
                # index cannot disambiguate them precisely because it is the thing
                # a reorder changes.
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
        Floating-point (and complex) destinations are accepted: rounding an fp64
        python float into an fp32/bf16 lr tensor is ordinary, pre-existing
        behavior that matches the native path.

        Range is checked alongside truncation because the two failures differ
        only in which side of the commit they land on. An integral value that
        overflows the destination (lr=128.0 into int8) makes ``fill_`` itself
        raise -- but that raise happens after the state was swapped in, leaving
        exactly the half-applied restore this check exists to prevent. Both are
        rejected here, during staging, so neither can.
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
                # A bool destination represents exactly {0, 1}, and torch.iinfo
                # has no entry for it. It needs the range check most: bool is the
                # one integral dtype whose fill_ never raises, so an out-of-range
                # value would coerce to True and commit a silently wrong lr.
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

        Split out from the replicated checkpoint-only validation because every
        check here reads rank-local state -- the live hyperparameter destinations
        and the constructor's deterministic flag -- and so may fail on one rank
        while passing on another. Callers must run this inside the synchronized
        staging block, where a raise becomes a group-wide abort; raising it
        outside would deadlock the peers in the _synchronize_step_failure
        collective. All checks stay ahead of the commit, preserving fail-atomic.
        """
        # Verify every live destination can hold the value it is about to be
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
        """Build the checkpoint dict for this rank. **Collective.**

        Every rank in the default process group must call this together, and no
        rank may call it alone -- this is a collective, like ``load_state_dict``.
        ``dcp.save`` and ``dcp.load`` both satisfy that (each calls it on every
        rank when converting Stateful objects), so the documented usage needs no
        care; a lone rank-0 call with a multi-rank group initialized will block in
        the agreement collective below.

        That contract is the price of the alternative. The checks below read
        rank-local state -- the live codebook, each slot's block geometry, its
        counters -- so they can fail on ONE rank while the others pass, and
        ``dcp.save`` runs this while converting Stateful objects, BEFORE it enters
        its own planning collectives. A bare raise here therefore drops the
        failing rank out while its peers walk into collectives it will never join:
        a silent, indefinite hang on the main save path. Agreeing on failure
        across the group first converts that into a clean group-wide abort, the
        same trade load_state_dict already makes.

        The cost is narrow: a lone rank-0 call is not a use this object supports
        anyway. The dict it returns holds _LazyDenseShard stand-ins that carry no
        data, cannot be pickled, and resolve only through DCP's save/load
        protocol, so inspecting it outside a collective save/load yields nothing
        usable. Use the optimizer's own state_dict() for an ordinary dict.

        Making the collective conditional was considered and rejected: a rank
        cannot detect locally whether its peers are also calling (any probe for
        that is itself a collective), so there is no safe no-op to fall back to.
        """
        # Re-derive the slot layout (and re-run the fail-closed layout gate) so a
        # retained wrapper whose optimizer gained parameters via add_param_group
        # after construction saves the current set, not the stale snapshot.
        #
        # Deliberately OUTSIDE the synchronized region below, mirroring
        # load_state_dict: every check it makes is replicated (group topology,
        # parameter names, DTensor placements, global shapes, constructor flags),
        # so it fails on every rank together or none. It also has to run first --
        # it is what establishes that a process group exists at all, without which
        # the agreement collective could not run.
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
        # collectives that follow. Same scope and rationale as the load path: the
        # plain-Gefen DTensors live on the default world (validated above), and
        # _synchronize_step_failure keeps the flag on CPU for gloo/CPU-resident
        # state and no-ops for a single-rank group.
        if _synchronize_step_failure(local_error is not None, dist.group.WORLD):
            if local_error is not None:
                raise local_error
            # Deliberately neutral about save vs load: dcp.load calls state_dict()
            # too, to build its destination.
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
            # Hand DCP stand-ins, not tensors: expanding every slot here held the
            # dense momentum AND second moment for every parameter live at once
            # (8 bytes/param, eight times the optimizer's resident state) for the
            # whole write. The stand-ins carry the planning metadata and expand
            # one at a time as the writer asks for them.
            result[self._key(slot, "momentum")] = _lazy_momentum(
                self.optimizer, slot.parameter, state, initialized
            )
            result[self._key(slot, "second_moment")] = _lazy_second_moment(
                slot.parameter, state, initialized
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
        # NOTE: validation that reads the LIVE optimizer -- hyperparameter
        # destinations and the deterministic flag -- is deliberately NOT done
        # here. It is rank-local, so it can fail on one rank and pass on another;
        # raising it outside the synchronized block below would let the failing
        # rank exit while its peers block forever in the _synchronize_step_failure
        # collective, turning a validation error into a distributed hang. It runs
        # inside that block instead (_validate_rank_local).

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
            self._validate_rank_local(state_dict, staged_hypers)
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


class _PinnedStaging:
    """One reusable page-locked buffer that every synchronous D2H copy passes through.

    A page-locked destination roughly doubles a device-to-host copy's bandwidth,
    which is worth having on a copy that cannot overlap with anything. Handing
    the page-locked tensor *itself* to the writer is not, because the writer
    decides how long to keep it: torch's ``FileSystemWriter`` holds every tensor
    it has written until the file is closed, so returning page-locked memory
    leaves the whole dense state dict page-locked for the save -- 8 bytes/param
    the host cannot swap, and cannot reclaim afterwards either, since the caching
    host allocator does not return blocks to the OS.

    Copying through one buffer and handing back pageable memory keeps both: the
    copy is still page-locked, while the page-locked footprint stays at a single
    slot's dense form no matter how many shards the writer accumulates or how
    many threads it runs. The cost is a host-to-host copy per slot.

    The buffer is sized ONCE, from the largest slot the save will stage, rather
    than grown as bigger slots arrive -- see :meth:`reserve`.
    """

    def __init__(self) -> None:
        self._buffer: torch.Tensor | None = None
        self._capacity = 0

    def reserve(self, nbytes: int) -> None:
        """Declare the largest slot this save will stage, before it starts.

        Growing the buffer on demand would not hold the one-slot bound, because
        replacing ``self._buffer`` does not give the old block back: torch's
        CachingHostAllocator keeps freed pinned blocks in its own cache and never
        returns them to the OS (a free does not even reach cudaHostFree). Each
        new maximum therefore ADDS a page-locked block, and the footprint climbs
        toward the sum of the successive maxima rather than settling at the
        largest one.

        That is not a corner case for this writer: ``FileSystemWriter`` sorts its
        write items by size and resolves them smallest first, so a save with N
        distinct slot sizes walks the maxima in exactly the order that allocates
        every one of them -- pinning the sum of all N. Sizing up front costs one
        pass over the stand-ins' metadata and makes the first expansion the only
        allocation.
        """
        self._capacity = nbytes

    def expand(self, dense: torch.Tensor) -> torch.Tensor:
        """Copy ``dense`` to pageable CPU memory via the page-locked buffer."""
        # empty_like rather than empty: it preserves the source's memory format,
        # as `.to("cpu")` would. An uninitialized slot expands with zeros_like,
        # which keeps a channels_last parameter channels_last, and the writer
        # serializes strides -- so allocating contiguous here would quietly
        # change the saved bytes for those slots. Allocating it first also fixes
        # the strides the staging view below has to match, and guarantees they
        # span exactly numel elements, which is what makes that view safe.
        out = torch.empty_like(dense, device="cpu")
        nbytes = out.numel() * out.element_size()
        if self._buffer is None or self._buffer.numel() < nbytes:
            # Allocated once, at the reserved size: re-pinning per slot is the
            # cost this exists to avoid (cudaHostAlloc runs milliseconds, so
            # paying it per slot dominates the copy it is meant to accelerate),
            # and re-pinning per new maximum silently accumulates page-locked
            # blocks, which is what reserve() exists to prevent.
            #
            # max(): reserve() covers every slot this planner will stage, so the
            # capacity normally wins and this runs exactly once. Falling back to
            # the request keeps a stand-in that reserve() never saw correct --
            # merely unbounded, as it was before -- rather than overflowing.
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

    Pass this to ``torch.distributed.checkpoint.save`` whenever the state dict
    contains a :class:`GefenDCPState`::

        dcp.save(
            {"model": model, "optimizer": GefenDCPState(optimizer)},
            storage_writer=dcp.FileSystemWriter(path),
            planner=GefenSavePlanner(),
        )

    Resharding forces the save to expand Gefen's ~1 byte/param block state into a
    dense 4 byte/param momentum plus a dense 4 byte/param second moment (the
    momentum codebook is learned per rank, so the indices mean nothing off their
    owning rank). :class:`GefenDCPState` therefore hands DCP zero-storage
    stand-ins instead of finished tensors, and this planner expands exactly one
    of them at a time, at the moment the writer asks for that write item, without
    keeping it afterwards. So the save's *device* memory peaks at the optimizer's
    resident state plus a single slot's dense form rather than holding 8
    bytes/param for every parameter across the whole write.

    That bound holds at any ``FileSystemWriter(thread_count=N)``. N writer
    threads ask for N write items at once, so expansion is serialized, and the
    expanded shard is copied to the CPU before the next expansion begins --
    returning it on the device instead would leave it live until the writer had
    finished writing it, which is past the point where serializing could bound
    anything. Slot expansion therefore does not run in parallel across writer
    threads; their file writes still do.

    Host memory is the writer's to bound, and torch's does not: ``FileSystemWriter``
    keeps every tensor it has written until it closes the file, so the host holds
    the whole dense state dict by the end of a save whatever this planner returns
    -- as it does for an eager state dict, and as it does without this planner.
    What is bounded here is the *page-locked* part of that. Copies go through
    :class:`_PinnedStaging`'s single reusable buffer and come back pageable, so a
    save page-locks one slot's dense form rather than all of it, and the host can
    swap the shards the writer accumulates.

    Without this planner the save is still *correct* -- the stand-ins fall back to
    allocating and retaining their dense form, which is what the load path needs
    them to do anyway -- but it costs the same 8 bytes/param the eager
    implementation did. Everything that is not a Gefen stand-in (the model, other
    Stateful items) is handled entirely by ``DefaultSavePlanner``.

    ``flatten_state_dict=False`` (inherited from ``DefaultSavePlanner``) is
    rejected -- see :meth:`__init__`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gefen_expand_lock = threading.Lock()
        # Lazily allocates, so a CPU-resident optimizer page-locks nothing.
        self._gefen_staging = _PinnedStaging()
        # Rejected up front rather than at write time, where it surfaces as an
        # opaque "stand-in cannot be pickled" TypeError from deep inside the
        # writer, after the save has already begun.
        #
        # This is a GefenDCPState constraint, not merely this planner's: DCP only
        # descends into a nested mapping when it flattens it. Unflattened, the
        # whole {"optimizer": ...} mapping becomes a SINGLE BYTE_IO write item
        # that DCP pickles wholesale -- the per-slot write items the stand-ins
        # publish via __create_write_items__ are never created, so there is
        # nothing for resolve_data to intercept and the stand-ins are pickled
        # instead (they carry no data of their own and refuse). Teaching this
        # planner to walk the nested mapping could not fix that: the write items
        # do not exist to be resolved. DefaultSavePlanner(flatten_state_dict=
        # False) fails the same way for the same reason; this constructor can only
        # police the planner Gefen owns.
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

        The arguments are named out in full, and forwarded positionally, rather
        than taken as ``*args``: ``dcp.save`` inspects this signature and, when it
        does not find ``storage_meta`` in it, falls back to calling the planner
        the pre-2.3 way -- ``set_up_planner(state_dict, is_coordinator)`` -- which
        would bind this rank's ``is_coordinator`` to ``storage_meta`` and silently
        leave ``is_coordinator`` False on every rank, including the one that has
        to build the global plan. A ``*args`` passthrough is invisible to that
        check and takes the legacy path.
        """
        super().set_up_planner(state_dict, storage_meta, is_coordinator)
        # The whole save's slots are known here, before the writer starts, so the
        # page-locked buffer can be sized to the largest of them in one pass over
        # metadata -- no expansion, no allocation. Growing it as bigger slots
        # arrive would pin the sum of the successive maxima instead; see
        # _PinnedStaging.reserve.
        #
        # Only CUDA slots stage: _gefen_expand_to_cpu hands back a CPU-resident
        # dense form untouched, so a CPU-resident optimizer (and an async save,
        # whose slots reach the writer already snapshotted to the CPU) reserves
        # nothing and page-locks nothing, as before.
        largest = 0
        for entry in self.state_dict.values():
            if isinstance(entry, _LazyDenseShard) and entry.device.type == "cuda":
                largest = max(largest, entry._gefen_dense_nbytes())
        self._gefen_staging.reserve(largest)

    def resolve_data(self, write_item):
        # Deliberately NOT via lookup_object(): that resolves through
        # find_tensor_shard -> __get_tensor_shard__, which retains the buffer for
        # the load path and would defeat the bound. Read the entry straight out
        # of the (already flattened) state dict instead, and fall through to the
        # default for anything that is not a Gefen stand-in.
        entry = self.state_dict.get(write_item.index.fqn)
        if isinstance(entry, _LazyDenseShard):
            # FileSystemWriter(thread_count=N) runs N of these concurrently, one
            # per file bucket, so without this lock N slots expand at once and
            # the bound below becomes N slots' dense form rather than one.
            # _gefen_resolve_uncached returns the shard already on the CPU, which
            # is what lets a lock held only across this call bound anything at
            # all: the writer keeps the returned tensor until its write finishes,
            # so a device-side return would outlive the lock and serializing
            # would buy nothing. The lock also guards the staging buffer, which
            # every expansion copies through and none may hold: what comes back
            # is a private pageable tensor, so the next expansion is free to
            # overwrite the buffer the moment this one releases the lock.
            with self._gefen_expand_lock:
                return entry._gefen_resolve_uncached(self._gefen_staging)
        return super().resolve_data(write_item)


def _stage_lazy_shards(state_dict, staged, prefix=()):
    """Copy ``state_dict`` without its stand-ins, snapshotting each to the CPU.

    Records ``path -> CPU-resident stand-in`` in ``staged`` and returns a copy of
    the tree with those entries removed, so torch's own staging never sees them.
    Copies rather than mutates: the dict belongs to the caller of ``async_save``.
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

    Pass this to ``torch.distributed.checkpoint.async_save`` whenever the state
    dict contains a :class:`GefenDCPState`::

        response = dcp.async_save(
            {"model": model, "optimizer": GefenDCPState(optimizer)},
            storage_writer=GefenFileSystemWriter(path),
            planner=GefenSavePlanner(),
        )
        # ... training continues ...
        # torch 2.5 returns a bare Future; newer torch wraps it in an
        # AsyncSaveResponse. Waiting is what surfaces a background write failure.
        getattr(response, "upload_completion", response).result()

    ``async_save`` returns once the state dict has been *staged* -- copied to the
    CPU -- and writes it from a background thread, so training may resume
    immediately and the checkpoint still reflects the state at the moment of the
    call. Staging is what makes that true, and it is the whole reason the default
    writer cannot be used here: it copies the state dict by asking each entry for
    a CPU copy of itself, and a :class:`GefenDCPState` hands DCP zero-storage
    stand-ins for the dense fp32 shards resharding requires (the momentum
    codebook is learned per rank, so the compact block state cannot be written as
    is). A stand-in has nothing to copy; it refuses, and the save fails.

    This writer stages those slots itself, one at a time -- expanding a slot's
    dense form, copying it to the CPU, and releasing it before expanding the next
    -- so the device holds a single slot's dense form during staging rather than
    every slot's at once, the same bound :class:`GefenSavePlanner` gives the
    synchronous save. The CPU then holds the full dense snapshot, which is what
    async staging costs for any optimizer. Everything that is not a Gefen stand-in
    is staged by ``FileSystemWriter`` unchanged.

    The synchronous :func:`torch.distributed.checkpoint.save` does not stage and
    does not need this writer; use ``FileSystemWriter`` with
    :class:`GefenSavePlanner` there.
    """

    def stage(self, state_dict):
        """Override of ``AsyncStager.stage``."""
        staged_shards: dict = {}
        stripped = _stage_lazy_shards(state_dict, staged_shards)
        return _restore_staged_shards(super().stage(stripped), staged_shards)

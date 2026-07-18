"""Gefen: a memory-efficient Adam-family optimizer with 8-bit quantized momentum.

Gefen keeps its first moment as uint8 codebook indices plus one fp32 magnitude
per block (the codebook itself is learned once from the first step's gradients
via an exact dynamic program) and shares/factors the second moment below
per-element resolution, cutting optimizer state to roughly 1 byte per parameter.
Fused CUDA kernels drive the whole update in a few launches per parameter; a
decomposed pure-PyTorch path covers CPU and fused-disabled runs. This module
contains the ``Gefen`` optimizer and the block-partitioning / quantization
helpers it steps through; ``gefen_muon``/``hybrid`` build on it.
"""

import hashlib
import io
import logging
import math
import os
import re
import warnings
from collections import defaultdict, OrderedDict
from itertools import chain
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn

from gefen.canonical import (
    CANONICAL_STATE_FORMAT_VERSION,
    PreparedCanonicalStateImport,
    canonical_value_supported,
    canonical_values_equal,
    clone_canonical_value,
    make_prepared_canonical_state_import,
)
from gefen.codebook import CodebookProcessGroupBinding
from gefen.contracts import (
    LogicalSlice,
    OptimizerContract,
    ParameterIdentity,
    ParameterLayout,
    ParameterStateRole,
    PlacementKind,
    ProcessGroupIdentity,
    ShardIdentity,
    ShardPlacement,
    ShardingManifest,
    TopologyChange,
    _gefen_contract,
)
from gefen.partitioning import find_period_by_block_variance
from gefen.portable_identity import (
    _parse_shard_identity,
    _serialize_shard_identity,
)
from gefen.rebinding import LogicalSlotBinding, ParameterRebinding
import gefen.quantization as quantization_module
from gefen.kernels.automatic_vmean import (
    automatic_vmean_update_cuda as _automatic_vmean_update_cuda,
)
from gefen.kernels.automatic_gefen_fused import (
    automatic_gefen_fused_update_cuda as _automatic_gefen_fused_update_cuda,
)
from gefen.kernels.automatic_gefen_fused import (
    automatic_gefen_fused_full_update_cuda as _automatic_gefen_fused_full_update_cuda,
)
from gefen.kernels.automatic_gefen_fused import (
    automatic_gefen_fused_update_v2_full_cuda as _automatic_gefen_fused_update_v2_full_cuda,
)
from gefen.kernels.automatic_gefen_fused import (
    gefen_quantized_momentum_update_cuda as _gefen_quantized_momentum_update_cuda,
)
from gefen.kernels.automatic_gefen_fused import (
    gefen_factored_update_cuda as _gefen_factored_update_cuda,
)
from gefen.kernels.automatic_gefen_fused import _should_use_v2_full
from gefen.kernels.automatic_gefen_fused import _CAPT_V1_FULL_TINY_NUMEL
from gefen.kernels.automatic_gefen_fused import _codebook_search_lut
from gefen.kernels.automatic_gefen_fused import (
    ensure_extension_loaded as _ensure_gefen_fused_extension_loaded,
)

logger = logging.getLogger(__name__)

_RANK_LOCAL_PAYLOAD_KEY_PREFIX = "_gefen_rank_local_payload_"
_RANK_LOCAL_MEMBER_KEY = "_gefen_rank_local_member"
_RANK_LOCAL_FORMAT = "rank_local_dtensor_v2"
_RANK_LOCAL_METADATA_VERSION = 3
_CODEBOOK_SCOPE_FORMAT_VERSION = 1
_SCOPED_NATIVE_METADATA_VERSION = 4
_NATIVE_LOCAL_SHARDS_LEGACY_FORMAT_VERSION = 1
_NATIVE_LOCAL_SHARDS_FORMAT_VERSION = 2
_CANONICAL_PARAMETER_STATE_KEYS = frozenset(
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
_CANONICAL_DERIVED_PARAMETER_STATE_KEYS = frozenset(
    {
        "stepsize",
        "_h_buf",
        "_capt_scalars",
        "_capt_consts",
        "_capt_consts_key",
        "_capt_stack",
        "_capt_row",
        "m_codebook_shape",
    }
)

# Per-param optimizer-state tensors that the non-fused step migrates onto the
# current compute device when a framework relocates a parameter between
# iterations (CPU-offload paging). Covers quantized momentum (m_codebook /
# m_magnitude), the block second moment (vmean), the factored second moment
# (v_row / v_col), the tensor step counters, and the reused scratch buffers.
# See Gefen._colocate_param_state_.
_COLOCATABLE_STATE_KEYS = (
    "m_codebook",
    "m_magnitude",
    "vmean",
    "vmean_step",
    "v_row",
    "v_col",
    "factored_step",
    "step",
    "stepsize",
    "_h_buf",
)


def _rank_local_payload_key(global_rank: int) -> str:
    return "{}{}".format(_RANK_LOCAL_PAYLOAD_KEY_PREFIX, int(global_rank))


# Optional explicit override for the period-search backend ("cuda_kernel" / "cpu"
# / "gpu"). None (default) means resolve per call from each tensor's own device
# -- see _resolve_find_period_backend. NOT used as an auto-populated cache.
FIND_PERIOD_BACKEND = None
FUSE_AUTOMATIC_VMEAN_UPDATE = True
FUSE_GEFEN_AUTOMATIC_STEP = True
FUSE_HISTOGRAM_FOR_EXACT = True

# Dispatch is shape-driven: the legacy partial-update pair uses its original
# occupancy predicate, while the production fully-fused pair uses a separately
# calibrated predicate because later multi-row/vectorized kernels moved the
# crossover. ``GEFEN_UPDATE_V2`` still hard-forces either pair for diagnostics.


def automatic_partition_view(flat_tensor: torch.Tensor, period: int) -> torch.Tensor:
    if flat_tensor.dim() != 1:
        raise ValueError(
            "Automatic partition view expects a 1D tensor, got dim={}".format(
                flat_tensor.dim()
            )
        )
    if period <= 0:
        raise ValueError(
            "Automatic partition period must be positive, got {}".format(period)
        )
    if flat_tensor.numel() % period != 0:
        raise ValueError(
            "Automatic partition expects numel {} to be divisible by period {}".format(
                flat_tensor.numel(),
                period,
            )
        )
    return flat_tensor.view(-1, period)


def automatic_partition_reduce(
    flat_tensor: torch.Tensor, period: int, reduce_op: str
) -> torch.Tensor:
    flat_view = automatic_partition_view(flat_tensor, period)
    if reduce_op == "mean":
        return flat_view.mean(dim=1, keepdim=True)
    if reduce_op == "max":
        return flat_view.max(dim=1, keepdim=True).values
    raise ValueError("Unexpected reduce_op: {}".format(reduce_op))


def automatic_vmean_update(
    vmean: torch.Tensor,
    grad_view: torch.Tensor,
    beta2: float,
    *,
    use_fused: bool,
) -> None:
    if use_fused:
        if grad_view.device.type != "cuda":
            raise ValueError(
                "FUSE_AUTOMATIC_VMEAN_UPDATE requires CUDA grad_view, got device {}".format(
                    grad_view.device
                )
            )
        if vmean.device.type != "cuda":
            raise ValueError(
                "FUSE_AUTOMATIC_VMEAN_UPDATE requires CUDA vmean, got device {}".format(
                    vmean.device
                )
            )
        _automatic_vmean_update_cuda(vmean, grad_view, beta2)
        return

    # Square in the model dtype (e.g. bf16) rather than promoting to float32
    # first: this avoids a full-size float32 temporary (~3-4% faster on the
    # non-fused step for large params) at the cost of ~1e-3 relative error in
    # the second moment, which is below the bf16 training noise floor. The
    # float32 vmean buffer still keeps the running estimate stable; add_ upcasts
    # the bf16 increment. (Diverges ~1e-3 from the fused kernel, which squares in
    # float32 — there is no cross-path parity requirement.)
    #
    # fp16 is the exception: its max finite value is 65504, so |g| >= 256
    # squares to inf and poisons vmean permanently (stepsize -> 0, the block
    # silently freezes). Promote ONLY fp16 to fp32 before squaring; bf16 keeps
    # the model-dtype square above bit-for-bit.
    promote_fp16 = grad_view.dtype == torch.float16
    #
    # Stream the per-block mean-square reduction in row chunks so the full-size
    # `grad_view * grad_view` temporary (the last param-proportional transient on
    # the non-fused step -- ~594 MiB of bf16 on the 311M-element tied embedding) is
    # never materialized whole. Each block's mean is a row-local reduction over
    # dim=1, so chunking whole rows is bit-for-bit identical to the whole-tensor
    # form. vmean.mul_(beta2) is applied once up front (it is the tiny [blocks,1]
    # state, not param-proportional); the scaled increment is then added per chunk.
    num_blocks = grad_view.shape[0]
    period = grad_view.shape[1]
    rows_per_chunk = max(1, GEFEN_MOMENTUM_UPDATE_CHUNK // period)
    if rows_per_chunk >= num_blocks:
        # Whole tensor fits one chunk (every param except the largest, e.g. the
        # tied embedding): run the original two-op form directly. Avoids the
        # per-param Python loop / slice overhead while staying the exact same math.
        g = grad_view.float() if promote_fp16 else grad_view
        block_mean_grad_sq = torch.mean(g * g, dim=1, keepdim=True)
        vmean.mul_(beta2).add_(block_mean_grad_sq, alpha=1 - beta2)
        return
    vmean.mul_(beta2)
    for r0 in range(0, num_blocks, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, num_blocks)
        g = grad_view[r0:r1]
        if promote_fp16:
            g = g.float()
        block_mean_grad_sq = torch.mean(g * g, dim=1, keepdim=True)
        vmean[r0:r1].add_(block_mean_grad_sq, alpha=1 - beta2)


# Process the nearest-codebook search in element chunks of this size. The search
# is the dominant transient allocator on the non-fused step: the naive form
# materialized ~10 full-size fp32/int64 temporaries (~40N bytes vs the 2N-byte
# bf16 param), which drove the full-FT OOM. Bounding the work to a fixed element
# count caps the scratch independent of param size while keeping each kernel
# launch large enough to stay GPU-bound (smaller chunks regress on launch
# overhead). The exact gather/abs/where keeps several int64/fp32 temporaries live
# per chunk (~40 bytes/elem), so 8M elems -> ~320 MB scratch ceiling (vs the
# param-proportional ~40N of the old whole-tensor form -- multiple GiB on a large
# embedding). The search runs ~10% slower than that whole-tensor form at this
# chunk size; it is not the step bottleneck (Newton-Schulz dominates), and the
# bounded scratch is what turns the full-FT OOM into a fit.
GEFEN_CODEBOOK_SEARCH_CHUNK = 1 << 23


# Shape of the names Gefen synthesizes positionally for an unnamed parameter.
# The spelling decides nothing on its own -- a caller may declare a real
# parameter called "param_0", and a collision-suffixed "weight_1" is positional
# without matching -- so this is only a last-resort guess for a parameter with no
# registration record; see _param_name_is_synthesized.
_SYNTHESIZED_PARAM_NAME_RE = re.compile(r"^(group_\d+_param_\d+|param_\d+)$")

# Serialized mirror of the registration-time provenance flags, parallel to a
# group's "param_names". Provenance cannot be recovered from a name's spelling,
# so it travels WITH the name through a checkpoint. Additive: a checkpoint
# written before this key existed lacks it, and load treats that as unknowable.
_SYNTHESIZED_PARAM_NAMES_KEY = "param_names_synthesized"


# Dequantize the stored uint8 indices into codebook coefficients in element
# chunks of this size. The naive `codebook[stored.long()]` makes a full-size
# int64 copy of the indices (8N bytes) on top of the full-size fp32 gather
# (4N bytes), so the transient peaks at ~12N before the final cast even though
# only the 2N/4N output is kept -- and this runs every step on both the fused and
# non-fused paths. Chunked index_select keeps the int64 index temporary and the
# fp32 gather bounded to one chunk while writing straight into the preallocated
# output. Same constant as the search chunk for the same launch-overhead reason.
GEFEN_DEQUANT_GATHER_CHUNK = 1 << 23


# Feed the one-time exact-codebook histogram in element chunks of this size
# (rounded down to a block-period multiple). Per-block histogram counts are
# independent, so chunked accumulation is bit-exact, but the upcast is the
# whole-tensor `flat.float()` (4N bytes at once) without it -- a >2G-element
# grad (Gemma 4's per-layer embedding table) allocates 8.75 GiB of fp32 and
# OOMs a 24 GB card during the very first step. 256M elems -> 1 GiB ceiling;
# runs once per codebook (re)build, so launch overhead is irrelevant here.
GEFEN_EXACT_HISTOGRAM_CHUNK = 1 << 28


# Stream the whole non-fused momentum update + parameter apply (dequant -> lerp ->
# per-block magnitude -> normalize -> re-quantize -> reconstruct -> scale -> p.add_)
# in row (block) chunks of about this many elements. The stock non-fused path ran
# every stage on the full tensor, so up to ~5 full-size fp32 temporaries
# (dequantized current_m, the promoted fp32 grad, the lerp output updated_m, its
# abs() reduction temp, the reconstructed shared_m) plus the fp32 update/-update
# were simultaneously live -- on the 311M-element tied embedding that is multiple
# GiB of transient fp32 scratch. Each block's update depends only on that block's
# own row (the per-block max magnitude, normalize, and codebook round-trip are all
# row-local), so processing whole rows one chunk at a time is bit-for-bit identical
# to the whole-tensor form while capping every fp32 temporary to one chunk. Rows
# are kept whole (never split mid-block) so the per-block max sees the full block.
# Same element budget as the gather/search chunks for the same launch-overhead
# reason; an exact multiple keeps each kernel launch GPU-bound.
#
# This constant also defines the "large param" routing threshold for the combined
# non-fused step: a param whose local numel exceeds it actually triggers the
# row-chunk loop (numel <= it runs whole-tensor in one iteration), so such params
# stay on the per-param streaming path instead of the foreach-merged path -- i.e.
# when a param is both "large" and would otherwise batch, streaming wins.
GEFEN_MOMENTUM_UPDATE_CHUNK = 1 << 23


# Non-fused step batching budget (in elements). The non-fused step runs a Python
# loop over every parameter, each issuing ~48 small elementwise CUDA ops, so the
# step is dominated by kernel-launch / CPU-dispatch overhead (~15k launches/step
# on Qwen3-1.7B, ~half pure dispatch on the 113 tiny RMSNorm params that move
# ~zero data). To cut that overhead we group parameters that share block geometry
# -- identical (local numel, automatic_period) and identical hyperparameters --
# and process each group as ONE merged (K*nblocks, period) tensor: the codebook
# search, dequant, vmean update, normalization and param update then run as a
# single set of ops over K params instead of K separate loop iterations. Because
# every op in the non-fused path is per-block or per-element independent, merging
# K params (a concatenation of their blocks) is bit-for-bit identical to looping
# over them one at a time. The merged tensors are bounded to this many elements so
# the extra transient stays small (the codebook-search scratch is independently
# chunk-bounded); K = max(1, budget // numel), so tiny params batch wide (the big
# launch win) while large params batch only a few at a time (their launch cost is
# already amortized by real GPU work). Env-overridable for tuning / OOM headroom.
GEFEN_NONFUSED_FOREACH_BUDGET = int(
    os.environ.get("GEFEN_NONFUSED_FOREACH_BUDGET", str(1 << 25))
)


def gefen_nearest_codebook_indices(
    codebook: torch.Tensor, normalized_vals: torch.Tensor
) -> torch.Tensor:

    if codebook.device != normalized_vals.device:
        raise ValueError(
            "Gefen codebook device {} does not match normalized_vals device {}.".format(
                codebook.device,
                normalized_vals.device,
            )
        )
    k = codebook.numel()
    out = torch.empty(
        normalized_vals.shape, dtype=torch.uint8, device=normalized_vals.device
    )

    # k == 1: the only codeword is index 0 for every value.
    if k <= 1:
        out.zero_()
        return out

    # The original assignment -- searchsorted into the sorted codebook, then a
    # gather/abs/where tie-break (nearest codeword; equal distance resolves to the
    # lower index via `<=`) -- materialized ~10 full-size fp32/int64 temporaries
    # (~40N bytes vs the 2N-byte bf16 param), which drove the full-FT OOM. We run
    # that *exact* arithmetic one element-chunk at a time, so the result is
    # bit-for-bit identical to the old whole-tensor form for every input dtype
    # (including the plain-Gefen fp32 momentum path, whose quantized indices feed
    # the next optimizer step -- so the trajectory is unchanged), while the
    # int64/fp32 scratch is bounded to a single chunk instead of the whole tensor.
    flat_out = out.view(-1)
    flat_in = normalized_vals.reshape(-1)
    n = flat_in.numel()
    chunk = GEFEN_CODEBOOK_SEARCH_CHUNK
    for start in range(0, n, chunk):
        stop = start + chunk
        seg = flat_in[start:stop].float()
        idx = torch.searchsorted(codebook, seg)
        left = (idx - 1).clamp_(0, k - 1)
        right = idx.clamp_(0, k - 1)
        assign = torch.where(
            (seg - codebook[left]).abs() <= (seg - codebook[right]).abs(),
            left,
            right,
        )
        flat_out[start:stop].copy_(assign.to(torch.uint8))
    return out


def gefen_dequantize_unpacked_indices(
    codebook: torch.Tensor,
    stored_indices: torch.Tensor,
    like_tensor: torch.Tensor,
) -> torch.Tensor:
    if codebook.device != stored_indices.device:
        raise ValueError(
            "Gefen codebook device {} does not match stored index device {}.".format(
                codebook.device,
                stored_indices.device,
            )
        )

    # Equivalent to `codebook[stored_indices.long()].to(device, dtype)` but
    # without the full-size int64 index copy + full-size fp32 gather: chunked
    # index_select gathers one bounded slice at a time and copy_ writes it into
    # the preallocated output, casting fp32 -> output dtype (and crossing devices
    # if needed) per chunk. The gathered values and the fp32->dtype rounding are
    # identical to advanced indexing, so the result is bit-for-bit the same.
    coeff = torch.empty(
        stored_indices.shape, dtype=like_tensor.dtype, device=like_tensor.device
    )
    coeff_flat = coeff.view(-1)
    idx_flat = stored_indices.reshape(-1)
    n = idx_flat.numel()
    chunk = GEFEN_DEQUANT_GATHER_CHUNK
    for start in range(0, n, chunk):
        stop = start + chunk
        seg = idx_flat[start:stop].long()
        coeff_flat[start:stop].copy_(codebook.index_select(0, seg))
    if hasattr(like_tensor, "device_mesh") and hasattr(like_tensor, "placements"):
        coeff_cls = type(like_tensor)
        if hasattr(coeff_cls, "from_local"):
            return coeff_cls.from_local(
                coeff, like_tensor.device_mesh, like_tensor.placements
            )
    return coeff


def gefen_set_unpacked_indices(
    stored_indices: torch.Tensor, indices: torch.Tensor
) -> None:
    stored_indices.copy_(indices.to(torch.uint8))


def gefen_automatic_fused_update(
    *,
    p: torch.Tensor,
    grad_view: torch.Tensor,
    m_codebook: torch.Tensor,
    m_magnitude: torch.Tensor,
    stepsize: torch.Tensor,
    codebook: torch.Tensor,
    packed_indices: bool,
    beta1: float,
    lr: float,
) -> None:
    if grad_view.device.type != "cuda":
        raise ValueError("Fused automatic Gefen update requires CUDA tensors.")
    if codebook.device != grad_view.device:
        raise ValueError(
            "Gefen codebook device {} does not match grad_view device {}.".format(
                codebook.device,
                grad_view.device,
            )
        )

    _automatic_gefen_fused_update_cuda(
        p,
        grad_view,
        m_codebook,
        m_magnitude,
        stepsize,
        codebook,
        packed_indices,
        beta1,
        lr,
    )


def _gefen_exact_histogram_from_grad_periods(
    *, grad_periods, num_codebooks: int, use_fused_histogram: bool
) -> torch.Tensor:
    histogram_bins = num_codebooks * 16
    bin_width = 2.0 / float(histogram_bins)
    # Accumulate counts as int64 and cast to float32 once at assembly: repeated
    # fp32 adds round for bins past 2^24 counts, so chunked accumulation could
    # otherwise drift from the whole-tensor form's single cast.
    bin_counts_cpu = torch.zeros(histogram_bins, dtype=torch.int64, device="cpu")
    prev_mode = torch.get_deterministic_debug_mode()
    torch.set_deterministic_debug_mode(0)
    try:
        for _, flat, period, _ in grad_periods:
            flat_1d = flat.reshape(-1)
            chunk_elems = max(1, GEFEN_EXACT_HISTOGRAM_CHUNK // period) * period
            for start in range(0, flat_1d.numel(), chunk_elems):
                flat_float = flat_1d[start : start + chunk_elems].float()
                if use_fused_histogram and flat_float.device.type == "cuda":
                    from gefen.kernels.exact_histogram_fused import (
                        gefen_exact_histogram_cuda,
                    )

                    local_counts_cuda = torch.zeros(
                        histogram_bins, dtype=torch.int64, device=flat_float.device
                    )
                    gefen_exact_histogram_cuda(flat_float, period, local_counts_cuda)
                    bin_counts_cpu.add_(local_counts_cuda.cpu())
                else:
                    blocks = automatic_partition_view(flat_float, period)
                    absmax = blocks.abs().amax(dim=1, keepdim=True)
                    normalized = torch.where(
                        absmax > 0, blocks / absmax, torch.zeros_like(blocks)
                    )
                    normalized_flat = normalized.reshape(-1)
                    bin_indices = torch.floor((normalized_flat + 1.0) / bin_width).to(
                        torch.long
                    )
                    bin_indices = bin_indices.clamp(0, histogram_bins - 1)
                    local_counts = torch.bincount(bin_indices, minlength=histogram_bins)
                    bin_counts_cpu.add_(local_counts.cpu())
    finally:
        torch.set_deterministic_debug_mode(prev_mode)
    return bin_counts_cpu


def _learn_gefen_exact_codebook_from_histogram(
    *,
    bin_counts_cpu: torch.Tensor,
    codebook_device: torch.device,
    num_codebooks: int,
    force_endpoints: bool,
    verbose: bool,
    compute_mse_logging: bool,
) -> Optional[torch.Tensor]:
    histogram_bins = num_codebooks * 16
    if (
        not torch.is_tensor(bin_counts_cpu)
        or bin_counts_cpu.device.type != "cpu"
        or bin_counts_cpu.dtype != torch.int64
        or tuple(bin_counts_cpu.shape) != (histogram_bins,)
    ):
        raise ValueError(
            "Gefen exact-codebook histogram must be a CPU int64 vector with {} bins".format(
                histogram_bins
            )
        )
    total_numel = int(bin_counts_cpu.sum().item())

    if total_numel == 0:
        return None

    bin_edges_cpu = torch.linspace(
        -1.0, 1.0, steps=histogram_bins + 1, dtype=torch.float32
    )
    bin_centers_cpu = 0.5 * (bin_edges_cpu[:-1] + bin_edges_cpu[1:])
    bin_counts_cpu = bin_counts_cpu.to(torch.float32)
    active_bins_cpu = bin_counts_cpu > 0
    histogram_stats = {
        "bin_edges": bin_edges_cpu,
        "bin_centers": bin_centers_cpu,
        "bin_counts": bin_counts_cpu,
        "active_bins": active_bins_cpu,
        "active_centers": bin_centers_cpu[active_bins_cpu],
        "active_counts": bin_counts_cpu[active_bins_cpu],
        "active_sums": bin_centers_cpu[active_bins_cpu]
        * bin_counts_cpu[active_bins_cpu],
    }

    codebook = quantization_module.exact_dp(
        histogram_stats=histogram_stats,
        num_codebooks=num_codebooks,
        force_endpoints=force_endpoints,
    ).to(codebook_device)

    if verbose and compute_mse_logging:
        codebook_cpu = codebook.cpu()
        active_centers = histogram_stats["active_centers"]
        idx = torch.searchsorted(codebook_cpu, active_centers)
        left_idx = (idx - 1).clamp(0, codebook_cpu.numel() - 1)
        right_idx = idx.clamp(0, codebook_cpu.numel() - 1)
        assignments = torch.where(
            (active_centers - codebook_cpu[left_idx]).abs()
            <= (active_centers - codebook_cpu[right_idx]).abs(),
            left_idx,
            right_idx,
        )
        errors = active_centers - codebook_cpu[assignments]
        mse_normalized = (
            errors.square() * histogram_stats["active_counts"]
        ).sum().item() / total_numel
        print("  exact DP: MSE_normalized={:.6e}".format(mse_normalized))

    return codebook


def learn_gefen_exact_codebook_from_grad_periods(
    *,
    grad_periods,
    codebook_device: torch.device,
    num_codebooks: int,
    force_endpoints: bool,
    verbose: bool,
    compute_mse_logging: bool,
    use_fused_histogram: bool = FUSE_HISTOGRAM_FOR_EXACT,
) -> Optional[torch.Tensor]:
    """Learn one exact-DP codebook from an unscoped local gradient stream."""

    histogram = _gefen_exact_histogram_from_grad_periods(
        grad_periods=grad_periods,
        num_codebooks=num_codebooks,
        use_fused_histogram=use_fused_histogram,
    )
    return _learn_gefen_exact_codebook_from_histogram(
        bin_counts_cpu=histogram,
        codebook_device=codebook_device,
        num_codebooks=num_codebooks,
        force_endpoints=force_endpoints,
        verbose=verbose,
        compute_mse_logging=compute_mse_logging,
    )


def _resolve_find_period_backend(grad: torch.Tensor) -> str:
    # An explicit FIND_PERIOD_BACKEND (set by config/tests) is an override and
    # wins. Otherwise resolve PER CALL from the tensor's own device. Do NOT cache
    # a device-derived value in the module global: the first tensor's device
    # would leak into every later call, so a CUDA step poisons a subsequent CPU
    # parameter ("cuda_kernel" backend on a CPU tensor -> ValueError). The
    # resolution is a trivial device-type check, so per-call costs nothing.
    if FIND_PERIOD_BACKEND is not None:
        return FIND_PERIOD_BACKEND

    grad_work = grad.to_local() if hasattr(grad, "to_local") else grad
    return "cuda_kernel" if grad_work.device.type == "cuda" else "cpu"


def _step_failure_collective_device(
    process_group, *, collective_device=None
) -> torch.device:
    """Choose a backend-compatible device for a process-group control flag."""
    import torch.distributed as dist

    if collective_device is not None:
        return torch.device(collective_device)

    group = process_group if process_group is not None else dist.group.WORLD
    bound_device = getattr(group, "bound_device_id", None)
    if bound_device is not None:
        return torch.device(bound_device)

    backend = str(dist.get_backend(process_group)).lower()
    if "nccl" in backend:
        return torch.device("cuda", torch.cuda.current_device())
    if "xccl" in backend:
        return torch.device("xpu", torch.xpu.current_device())
    # Gloo, MPI, UCC, and PyTorch's multi-backend default process group all
    # accept CPU control tensors. Keeping their flag on CPU also avoids an
    # unrelated accelerator initialization in spawned Gloo workers.
    return torch.device("cpu")


@torch.no_grad()
@torch._dynamo.disable
def _synchronize_step_failure(
    local_failed, process_group, *, collective_device=None
) -> bool:
    """Return whether any process-group member reported a step failure."""
    failed = bool(local_failed)
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return failed

    import torch.distributed as dist

    if dist.get_world_size(process_group) < 2:
        return failed
    flag = torch.tensor(
        int(failed),
        dtype=torch.int32,
        device=_step_failure_collective_device(
            process_group, collective_device=collective_device
        ),
    )
    dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=process_group)
    return bool(flag.item())


@torch.no_grad()
@torch._dynamo.disable
def _synchronize_step_control_range(
    local_minimum, local_maximum, process_group, *, collective_device=None
):
    """Expand control-value bounds across one process group."""
    minimum = tuple(float(item) for item in local_minimum)
    maximum = tuple(float(item) for item in local_maximum)
    if len(minimum) != len(maximum):
        raise ValueError("step-control bounds must have equal lengths")
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return minimum, maximum

    import torch.distributed as dist

    if dist.get_world_size(process_group) < 2:
        return minimum, maximum
    bounds = torch.tensor(
        minimum + tuple(-item for item in maximum),
        dtype=torch.float64,
        device=_step_failure_collective_device(
            process_group, collective_device=collective_device
        ),
    )
    dist.all_reduce(bounds, op=dist.ReduceOp.MIN, group=process_group)
    values = bounds.cpu().tolist()
    width = len(minimum)
    return tuple(values[:width]), tuple(-item for item in values[width:])


def _amp_optimizer_step_controls(optimizer):
    """Parse GradScaler's temporary controls without mutating gradients."""
    found_inf = getattr(optimizer, "found_inf", None)
    if found_inf is not None:
        if torch.is_tensor(found_inf):
            if found_inf.numel() != 1:
                raise RuntimeError(
                    "GradScaler supplied a non-scalar found_inf tensor with "
                    "shape {}".format(tuple(found_inf.shape))
                )
            overflow = bool(found_inf.detach().item())
        else:
            overflow = bool(found_inf)
    else:
        overflow = False

    grad_scale = getattr(optimizer, "grad_scale", None)
    if torch.is_tensor(grad_scale):
        if grad_scale.numel() != 1:
            raise RuntimeError(
                "GradScaler supplied a non-scalar grad_scale tensor with shape "
                "{}".format(tuple(grad_scale.shape))
            )
        scale_value = float(grad_scale.detach().item())
    elif grad_scale is not None:
        scale_value = float(grad_scale)
    else:
        scale_value = 0.0
    if grad_scale is not None and (
        not math.isfinite(scale_value) or scale_value <= 0.0
    ):
        raise RuntimeError(
            "GradScaler supplied a non-finite or non-positive grad_scale"
        )
    return overflow, grad_scale, scale_value


@torch.no_grad()
def _amp_prepare_optimizer_step(optimizer) -> bool:
    """Honor PyTorch's native ``GradScaler`` optimizer-step protocol.

    ``GradScaler.step`` gives AMP-aware optimizers two temporary attributes:
    ``found_inf`` contains the result of the non-finite scan and ``grad_scale``
    contains the scale that still needs to be removed. When users explicitly
    call ``scaler.unscale_(optimizer)`` first, ``grad_scale`` is ``None`` and
    the gradients must not be divided a second time.

    Return ``False`` on overflow before callers learn a codebook or mutate any
    optimizer/parameter state. On a finite automatic-unscale step, multiply the
    gradients in place by the same fp32 reciprocal used by ``GradScaler``. This
    keeps every existing fused/unfused update path operating on ordinary
    unscaled gradients and also works for local DTensor shards.
    """
    overflow, grad_scale, _ = _amp_optimizer_step_controls(optimizer)
    if overflow:
        return False
    if grad_scale is None:
        # Explicit scaler.unscale_(optimizer), or an ordinary non-AMP step.
        return True

    if torch.is_tensor(grad_scale):
        scale = grad_scale.detach()
        # Match torch.amp.GradScaler.unscale_: computing the reciprocal in
        # fp64 avoids compile-option-dependent fp32 division differences.
        inv_scale = scale.double().reciprocal().float()
    else:
        inv_scale = torch.tensor(float(grad_scale), dtype=torch.float64)
        inv_scale = inv_scale.reciprocal().float()

    grads_by_device = OrderedDict()
    for group in optimizer.param_groups:
        for param in group["params"]:
            grad = param.grad
            if grad is None:
                continue
            # A DTensor's local shard is a writable view of its local storage;
            # scaling that view avoids introducing a DTensor collective.
            if hasattr(grad, "placements") and hasattr(grad, "to_local"):
                grad = grad.to_local()
            if grad.is_sparse:
                grad = grad._values()
            grads_by_device.setdefault(grad.device, []).append(grad)

    inv_scale_by_device = {}
    for device, grads in grads_by_device.items():
        device_inv_scale = inv_scale_by_device.get(device)
        if device_inv_scale is None:
            device_inv_scale = inv_scale.to(device=device, non_blocking=True)
            inv_scale_by_device[device] = device_inv_scale
        # One foreach launch per parameter device. Mixed floating dtypes are
        # supported; each tensor rounds the fp32 reciprocal in its own dtype's
        # elementwise multiply, just as GradScaler's regular unscale path does.
        torch._foreach_mul_(grads, device_inv_scale)
    return True


@torch._dynamo.disable
def _amp_dtensor_protocol_preflight(optimizer) -> bool:
    """Synchronize FP16-DTensor AMP protocol selection before GradScaler scans.

    ``GradScaler.step`` reads ``_step_supports_amp_scaling`` before it checks
    gradients for non-finite values. DTensor dispatch may make that check
    collective, so choosing the protocol from rank-local active gradients can
    leave one rank in the ordinary path while another enters a DTensor
    collective. When an optimizer contains any multi-rank DTensor, non-FSDP1
    FP16 parameter storage makes the native protocol a topology property
    instead. Before returning it, compare a fixed gradient-presence vector for
    every DTensor parameter on the same mesh so every rank either proceeds with
    the same protocol or raises before GradScaler touches any gradient.

    Return ``True`` when a distributed optimizer containing DTensors also has
    any non-FSDP1 FP16 parameter storage, even if that parameter is local or
    inactive on this step. Every DTensor in the optimizer is included because
    GradScaler scans the complete active set after selecting the native
    protocol. Single-rank and DTensor-free optimizers retain the active-gradient
    protocol and do not pay a collective here.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return False

    import torch.distributed as dist

    params = [param for group in optimizer.param_groups for param in group["params"]]
    has_fp16_storage = any(
        param.dtype == torch.float16 and not getattr(param, "_is_flat_param", False)
        for param in params
    )
    if not has_fp16_storage:
        return False

    by_mesh = OrderedDict()
    for group in optimizer.param_groups:
        names = group.get("param_names")
        for index, param in enumerate(group["params"]):
            if not (
                hasattr(param, "to_local")
                and hasattr(param, "placements")
                and hasattr(param, "device_mesh")
            ):
                continue
            mesh = param.device_mesh
            if mesh.get_coordinate() is None or mesh.size() < 2:
                continue
            process_groups = tuple(mesh.get_all_groups())
            key = (
                str(mesh.device_type),
                tuple(int(item) for item in mesh.shape),
                tuple(
                    int(item) for item in mesh.mesh.detach().cpu().reshape(-1).tolist()
                ),
                tuple(str(group.group_name) for group in process_groups),
            )
            entry = by_mesh.get(key)
            if entry is None:
                entry = {
                    "mesh": mesh,
                    "process_groups": process_groups,
                    "items": [],
                }
                by_mesh[key] = entry
            name = (
                names[index]
                if isinstance(names, (list, tuple)) and index < len(names)
                else getattr(optimizer, "_param_name", lambda _: "parameter")(param)
            )
            entry["items"].append((str(name), param, param.grad is not None))

    if not by_mesh:
        return False
    # Do not initialize or query CUDA for a CPU-only DTensor optimizer. Spawned
    # Gloo workers may run alongside a CUDA-heavy parent release gate, and an
    # unrelated current-stream query can otherwise block before the CPU
    # collective preflight. CUDA-backed optimizers retain the capture guard.
    if any(param.device.type == "cuda" for param in params) and (
        torch.cuda.is_current_stream_capturing()
    ):
        return False

    mismatches = []
    # Parameter-group order may legitimately differ between ranks while two
    # equivalent DeviceMesh objects still refer to distinct c10d groups. Sort
    # by each mesh's process-group names so every rank enters those collectives
    # in creation order instead of whichever parameter happened to appear
    # first locally.
    for mesh_key in sorted(by_mesh):
        entry = by_mesh[mesh_key]
        mesh = entry["mesh"]
        items = sorted(
            entry["items"],
            key=lambda item: (
                item[0],
                tuple(item[1].shape),
                str(item[1].dtype),
            ),
        )
        local = items[0][1].to_local()
        if hasattr(local, "wait"):
            local = local.wait()
        active_counts = torch.tensor(
            [int(active) for _, _, active in items],
            dtype=torch.int32,
            device=local.device,
        )
        for process_group in entry["process_groups"]:
            if dist.get_world_size(process_group) > 1:
                dist.all_reduce(
                    active_counts, op=dist.ReduceOp.SUM, group=process_group
                )

        mesh_size = mesh.size()
        inconsistent = torch.nonzero(
            (active_counts != 0) & (active_counts != mesh_size),
            as_tuple=False,
        ).flatten()
        if inconsistent.numel() == 0:
            continue
        counts_cpu = active_counts.cpu()
        for index in inconsistent.cpu().tolist():
            mismatches.append(
                "{} ({}/{} mesh ranks have gradients)".format(
                    items[index][0], int(counts_cpu[index]), mesh_size
                )
            )

    if mismatches:
        raise RuntimeError(
            "Gefen GradScaler integration requires identical DTensor gradient "
            "presence on every mesh rank before its distributed non-finite "
            "scan. Mismatched parameters: {}. Ensure conditional or unused "
            "parameters produce the same `.grad is None` pattern on every "
            "mesh rank.".format(", ".join(mismatches))
        )
    return True


def _amp_native_scaling_required(optimizer) -> bool:
    """Use native scaling only when generic GradScaler cannot unscale safely.

    PyTorch's ordinary GradScaler path has the best integration semantics: it
    does not call ``optimizer.step()`` on overflow, so Accelerate/Trainer can
    observe the skip and keep schedulers aligned. Its one unsupported case is a
    true FP16 gradient tensor. Opt into the optimizer-side protocol only for
    that case.

    FSDP1 FlatParameter FP16 needs ``ShardedGradScaler``'s scaler-owned,
    cross-rank non-finite reduction, so native handling is disabled for that
    representation. DTensor/FSDP2 is different: torch's DTensor handler for the
    AMP non-finite op MAX-reduces the scaler-owned flag over every mesh
    dimension even in the native check path, so all ranks skip and update their
    scales identically.
    """
    has_fp16_grad = _amp_dtensor_protocol_preflight(optimizer)
    has_fsdp1_flat_param = False
    for group in optimizer.param_groups:
        for param in group["params"]:
            if getattr(param, "_is_flat_param", False):
                has_fsdp1_flat_param = True
            grad = param.grad
            if grad is not None and grad.dtype == torch.float16:
                has_fp16_grad = True

    if has_fp16_grad and has_fsdp1_flat_param:
        if not getattr(optimizer, "_gefen_warned_sharded_fp16_amp", False):
            warnings.warn(
                "Gefen detected FP16 gradients on an FSDP1 FlatParameter and "
                "disabled optimizer-side AMP unscaling. Use "
                "torch.distributed.fsdp.ShardedGradScaler (or explicitly call "
                "its unscale_ before step) so non-finite detection is reduced "
                "across ranks. Base GradScaler will fail fast while unscaling "
                "FP16 gradients instead of risking rank-divergent optimizer "
                "steps.",
                RuntimeWarning,
                stacklevel=3,
            )
            optimizer._gefen_warned_sharded_fp16_amp = True
        return False
    return has_fp16_grad


def _assert_optimizer_gradients_structurally_valid(
    optimizer, *, require_2d_params: bool = False
) -> None:
    """Validate every active gradient before any optimizer mutation.

    Optimizer implementations normally discover an invalid later gradient only
    after earlier parameters have already stepped. Gefen also learns/refreshes a
    shared codebook before its per-parameter loop, and native AMP may unscale
    gradients in place. Scan the complete parameter set first so sparse,
    compressed-sparse, MKLDNN, complex, or malformed gradients fail atomically.
    """
    for group in optimizer.param_groups:
        names = group.get("param_names")
        for index, param in enumerate(group["params"]):
            name = (
                names[index]
                if isinstance(names, (list, tuple)) and index < len(names)
                else getattr(optimizer, "_param_name", lambda _: "parameter")(param)
            )
            if require_2d_params and param.ndim != 2:
                error_factory = getattr(optimizer, "_step_non_2d_parameter_error", None)
                if callable(error_factory):
                    raise ValueError(error_factory(param))
                raise ValueError(
                    "GefenMuon requires every parameter to remain a 2D matrix "
                    "before step, but {!r} has {} dimensions.".format(
                        str(name), param.ndim
                    )
                )
            grad = param.grad
            if grad is None:
                continue
            layout = getattr(grad, "layout", None)
            if getattr(grad, "is_sparse", False) or layout != torch.strided:
                raise RuntimeError(
                    "Gefen does not support sparse gradients or other "
                    "non-strided layouts; parameter {!r} has gradient layout "
                    "{}.".format(str(name), layout)
                )
            if torch.is_complex(grad):
                raise RuntimeError(
                    "Gefen optimizers do not support complex gradients, but parameter {!r} has dtype {}.".format(
                        str(name), grad.dtype
                    )
                )
            if tuple(grad.shape) != tuple(param.shape):
                raise RuntimeError(
                    "Gefen gradient shape {} for parameter {!r} does not match parameter shape {}.".format(
                        tuple(grad.shape), str(name), tuple(param.shape)
                    )
                )
            # The update is applied in place into the parameter while all of the
            # update math runs on the gradient's device; the parameter, its
            # gradient, and its optimizer state must therefore be co-located (the
            # CPU-offload contract). A genuine param/grad device split cannot
            # produce a correct in-place write and otherwise fails opaquely deep
            # in an add_/kernel. Reject it here in the pre-mutation preflight --
            # before shared-codebook learning/refresh and the per-parameter loop
            # -- so a mismatch fails atomically without leaving a codebook or
            # per-param state learned from the rejected gradient. Normal autograd
            # produces the gradient on the parameter's own device, so this never
            # fires for a same-device run. DTensor / async gradients are sharded
            # wrappers whose local-shard device is resolved by the
            # .to_local()/.wait() unwrap downstream and can legitimately differ
            # from the wrapper's reported device, so skip the check for them and
            # only enforce co-location for plain tensors (the CPU-offload case).
            if (
                not hasattr(grad, "to_local")
                and not hasattr(grad, "wait")
                and not hasattr(param, "placements")
                and grad.device != param.device
            ):
                raise RuntimeError(
                    "Gefen requires each parameter and its gradient to live on "
                    "the same device (param {!r} is on {} but its gradient is on "
                    "{}). For CPU-offloaded training keep the parameter, its "
                    "gradient, and its optimizer state co-located on the same "
                    "device for the step.".format(
                        str(name), param.device, grad.device
                    )
                )


class Gefen(torch.optim.Optimizer):
    """Adam-family optimizer with 8-bit quantized momentum and block-shared or
    factored second moments, at roughly 1 byte of optimizer state per parameter.

    ``params`` accepts plain tensors, ``(name, tensor)`` tuples, or param-group
    dicts (each optionally overriding ``lr`` / ``betas`` / ``eps`` /
    ``weight_decay``). Public ``param_groups`` preserve the caller's group
    boundaries; each parameter's stable lowercase name is stored in its
    per-param state and mirrored in the containing group's ``param_names`` list.

    Args:
        params: iterable of parameters, ``(name, parameter)`` tuples, or
            param-group dicts with a ``"params"`` key.
        lr (float or Tensor, default 1e-3): learning rate. May be a 1-element
            tensor (update it in place to drive an LR schedule; see
            ``capturable`` for the on-device semantics).
        betas (tuple, default (0.9, 0.999)): EMA coefficients of the quantized
            first moment and the second moment.
        eps (float, default 1e-8): term added to the bias-corrected
            second-moment root in the stepsize denominator.
        weight_decay (float, default 0.0): decoupled (AdamW-style) weight
            decay, applied as ``p *= 1 - lr * weight_decay``.
        fused (bool, default True): step through the fused CUDA kernels.
            Downgraded to False with a warning when CUDA is unavailable; the
            decomposed pure-PyTorch path then runs.
        force_1d_period_one (bool, default False): 1D parameters
            (RMSNorm/LayerNorm weights, biases) skip the block-period search
            and use period==1 -> per-element 2nd moment + momentum magnitude,
            i.e. AdamW-like fidelity. Without it the memory-safe fallback can
            lump a whole 1D norm tensor into a single shared block, which
            over-steps on these high-leverage tensors (the norm weights set
            the LR stability ceiling). 1D tensors are tiny, so the per-element
            state is a negligible memory cost.
        force_2d_period_one (bool, default False): 2D parameters skip the
            block search and use period==1. In the hybrid backup these are the
            token embedding / LM head -- the most AdamW-divergent backup
            tensors under a block-shared vmean. The memory cost is real
            (per-element fp32 state on a vocab x hidden tensor), so this is an
            opt-in loss/memory trade, not free like ``force_1d_period_one``.
        period_one_substrings (tuple of str, default ()): case-insensitive
            name substrings routed to period==1 for SELECTED params only --
            the surgical form of the ``force_*_period_one`` flags. E.g.
            ``("embed", "lm_head")`` gives the vocab projections per-element
            state without paying per-element memory on every hidden matrix.
        factored_v_2d (bool, default True): Adafactor-style factored second
            moment for 2D params (the AdamW-parity configuration): the
            quantized momentum machinery is unchanged, but row/col fp32 EMAs
            of grad^2 reconstruct a per-element second moment at
            ``rows + cols`` state instead of ``numel``. Its fair LR is ~0.6x
            the AdamW LR. ``False`` selects the legacy block-shared second
            moment (fair LR ~0.4x). Sharded (DTensor/FSDP2) 2D params fall
            back to the block-vmean path regardless (local-shard row/col
            statistics would be wrong).
        codebook_refresh_every (int, default 0): re-learn the exact-DP
            codebook from the CURRENT gradients every N steps (0 = learn once
            on step 1 and freeze). Stored momentum indices are re-quantized
            onto the new codebook, so the momentum survives the swap with only
            nearest-codeword rounding error.
        stochastic_round (bool, default False): momentum->codebook
            quantization uses unbiased stochastic rounding (seeded per step)
            instead of deterministic nearest, cancelling the systematic bias
            nearest-rounding accumulates in the EMA momentum over a long
            horizon. Only the fused CUDA automatic-step kernels honor it; a
            warning fires when the fused path is off.
        deterministic (bool, default False): require replica-exact update
            arithmetic on identical devices and inputs. Automatic block periods
            use fixed-order GPU reductions instead of the atomic CUDA search.
            Fused block-vmean parameters stay fused but use the fixed-order v1
            reduction instead of the atomic v2-full path. Fused factored-v
            parameters use the deterministic decomposed factored update because
            the fast fused stats kernel accumulates column sums with unordered
            floating-point atomics. This is an explicit throughput-for-
            reproducibility mode; the default keeps the existing performance
            routing unchanged.
        capturable (bool, default False): CUDA-graph capturability (mirrors
            torch.optim's argument). Everything that varies across steps --
            step counters, bias corrections, kernel scalars -- lives in device
            tensors and the step avoids host<->device syncs, so ``step()`` can
            be captured in a ``torch.cuda.CUDAGraph`` and replayed. A tensor
            ``lr`` flows on device (update it in place between replays); a
            float ``lr`` is baked at capture time. Codebook learning and the
            period search are host-driven and run on the FIRST step, so do the
            standard warmup steps before capturing. Incompatible with
            ``codebook_refresh_every > 0``. Every captured parameter must live
            on the current CUDA capture device; distributed jobs should use
            one optimizer per rank/device.
        verbose (bool, default False): extra diagnostics during codebook
            learning and period prediction.
    """

    # Pure rebuildable memoization for the layout-forensics guards. These live
    # in ``__slots__`` rather than the instance ``__dict__`` so that staged
    # ``__dict__`` copies, staged commits, and fail-before-mutation attribute
    # snapshots treat them as what they are — caches, never optimizer state.
    # Every accessor reads them through ``getattr`` with a cold default, so an
    # object whose slots were never populated simply revalidates fully.
    __slots__ = (
        "_gefen_layout_version",
        "_gefen_layout_forensics_verdict",
        "_gefen_manifest_forensics_cache",
    )

    def __init__(
        self,
        params: Iterable[Union[nn.Parameter, Tuple[str, nn.Parameter]]],
        lr: Union[float, torch.Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        fused: bool = True,
        force_1d_period_one: bool = False,
        force_2d_period_one: bool = False,
        period_one_substrings: Tuple[str, ...] = (),
        factored_v_2d: bool = True,
        codebook_refresh_every: int = 0,
        stochastic_round: bool = False,
        deterministic: bool = False,
        capturable: bool = False,
        verbose: bool = False,
    ):
        if not isinstance(deterministic, bool):
            raise TypeError("deterministic must be a bool")
        if deterministic and factored_v_2d and stochastic_round:
            raise ValueError(
                "deterministic=True with factored_v_2d=True cannot honor "
                "stochastic_round=True: the replica-exact factored path uses "
                "deterministic nearest-codeword quantization. Disable stochastic "
                "rounding or factored_v_2d."
            )
        if fused and not torch.cuda.is_available():
            warnings.warn(
                "Gefen optimizer got fused=True, but CUDA is not available. Changing fused to False.",
                UserWarning,
                stacklevel=2,
            )
            fused = False
        logger.info("Initializing Gefen optimizer (fused=%s).", fused)

        self._validate_group_options(lr, betas, eps, weight_decay)

        self.fused = fused
        # Fused-toolchain probe cache (M8 graceful fallback): None = not yet
        # probed, True/False = the JIT extension built or permanently failed on
        # this box. Set lazily on the first step by _gefen_fused_toolchain_ok so
        # a CUDA host without a working nvcc downgrades to the pure-torch path
        # with one warning instead of crashing mid-step.
        self._fused_build_ok = None
        # Period-one routing knobs: see the class docstring and
        # partitioning.memory_safe_fallback_period.
        self._force_1d_period_one = force_1d_period_one
        self._force_2d_period_one = force_2d_period_one
        self._period_one_substrings = tuple(s.lower() for s in period_one_substrings)
        self._factored_v_2d = factored_v_2d
        self._deterministic = deterministic
        if type(codebook_refresh_every) is not int:
            raise TypeError("codebook_refresh_every must be an integer")
        if codebook_refresh_every < 0:
            raise ValueError(
                "codebook_refresh_every must be >= 0 but is: {}".format(
                    codebook_refresh_every
                )
            )
        self._codebook_refresh_every = codebook_refresh_every
        # Under capturable the fused update kernels stay engaged: they read
        # their per-step-varying scalars (lr, bias-correction reciprocals,
        # weight-decay factor) from a small per-param fp32 device buffer that
        # is refreshed in place with tensor ops each step, instead of from
        # host kernel arguments a graph would freeze at capture time.
        self.capturable = capturable
        if capturable and codebook_refresh_every:
            raise ValueError(
                "capturable=True is incompatible with codebook_refresh_every>0: "
                "the periodic codebook re-learn is host-driven and cannot run "
                "inside a replayed CUDA graph."
            )
        # Default False is bit-identical to the prior behavior (the kernels
        # take the same parity-preserving fast path). Under capturable=True the
        # per-step seed lives in a per-device 0-dim int64 tensor advanced on
        # device once per step (see _sr_seed_on), so a captured/replayed CUDA
        # graph dithers with a fresh seed every replay; eager capturable steps
        # read the SAME tensor, so a replay at step k bit-matches the eager
        # capturable step k. Warn (do not hard-error -- mirroring fp8_ns's
        # portable-config behavior) so stochastic_round=True is not a silent
        # no-op when the fused path is off.
        self._stochastic_round = stochastic_round
        # Probe-free check: report the CONFIGURED fused intent here, not the
        # toolchain-probed availability -- construction must never trigger the
        # JIT build (that is deferred to the first step's lazy probe).
        if stochastic_round and not self._fused_configured():
            warnings.warn(
                "stochastic_round=True is honored only by the fused Gefen "
                "automatic-step CUDA kernels; the non-fused / CPU path uses "
                "deterministic nearest rounding, so the flag is a no-op here.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.verbose = verbose

        self._gefen_codebook = None
        self._gefen_codebook_by_device = {}
        # Per-device codebook-search LUTs for the COMPILED capturable step:
        # populated (and marked static) by _mark_state_static_for_compile, so
        # traced frames can hand the fused custom ops a prebuilt LUT instead
        # of building one lazily inside the op body (an in-op allocation that
        # cudagraph-trees recording rejects whenever the codebook input gets
        # copied into the graph pool). Cleared with _gefen_codebook_by_device.
        self._gefen_codebook_lut_by_device = {}
        # An explicit learned-codebook scope is installed only as part of the
        # same full post_sharding transaction that publishes canonical shard
        # identities. It is live runtime configuration, never inferred from a
        # default process group or from tensor storage.
        self._gefen_codebook_process_group = None
        self._gefen_codebook_scope_validated = False
        # Capturable stochastic rounding: ONE 0-dim int64 device tensor per
        # device holding the per-step rounding seed (a device-side mirror of
        # _gefen_global_step). Created lazily by _sr_seed_on and advanced on
        # device once per step by _advance_sr_seeds, so a captured CUDA graph
        # replays fresh seeds. Not serialized: load_state_dict clears the dict
        # and the seeds rebuild from the restored gefen_global_step.
        self._sr_seed_by_device = {}
        # Batched capturable refresh (capturable=True only): each device maps
        # to an ordered tuple of scalar-HYPERPARAMETER cohorts. Every cohort
        # holds its registered params' step counters ([2, N] fp32; row 0 = the
        # bc2 counter, row 1 = step) and kernel-scalar rows ([N, 4] fp32) as the
        # backing store of the per-param state views. One short vectorized op
        # chain per distinct lr/weight-decay cohort replaces the ~10
        # microscopic launches per param while still allowing ordinary
        # decay/no-decay and layerwise-lr parameter groups. None until the
        # first capturable prologue builds the registry; see
        # _capt_build_stacks / _capt_batched_refresh.
        self._capt_stacks = None
        # Identity fingerprint of the last _mark_state_static_for_compile
        # pass; lets the per-step call skip the O(#tensors) re-marking when
        # nothing it would mark has changed. See _static_mark_signature.
        self._static_mark_sig = None
        self._gefen_global_step = 0
        # Manual CUDA-graph replay does not execute Python, so the host counter
        # above cannot be the checkpoint authority under capturable=True. Keep
        # one device counter per parameter device and advance it in the captured
        # step tail; state_dict synchronizes the host mirror before serializing.
        self._gefen_global_step_by_device = {}
        defaults = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            weight_decay=weight_decay,
        )
        self._param_names = {}
        # Exact canonical identity remains separate from compatibility
        # ``param_names``. A full post_sharding transaction publishes these
        # registries only after validating every optimizer slot and manifest.
        self._gefen_shard_bindings = {}
        self._gefen_local_shard_bindings = ()
        self._gefen_sharding_manifest = None
        self._gefen_post_sharding_finalized = False
        self._gefen_finalized_slots = ()
        self._gefen_logical_slots = ()
        # Layout-forensics fast path. After one full structural pass succeeds,
        # the exact container identities it validated are remembered together
        # with a version counter that every legitimate mutating API bumps
        # (post_sharding and staged checkpoint load commits). Per-step guards
        # accept only that unchanged token set; boundary operations (checkpoint
        # prepare/commit, rebinding, collective codebook initialize/refresh,
        # and contract readiness) always rerun the complete forensic rebuild.
        self._gefen_layout_version = 0
        self._gefen_layout_forensics_verdict = None
        # (manifest, frozenset(manifest.shards), sha256 digest) computed once
        # per finalized manifest object instead of rehashing every identity on
        # every scoped step header.
        self._gefen_manifest_forensics_cache = None
        # Provenance for those names: True where Gefen generated the positional
        # name itself, False where the caller supplied it. Only the generated ones
        # encode registration order rather than stable identity, and the spelling
        # cannot tell them apart. Consumers needing caller-stable identity
        # (GefenDCPState) ask _param_name_is_synthesized.
        self._synthesized_param_names = {}
        # ``set_optimizer_state_dict(flatten_optimizer_state_dict=True)`` uses
        # the *live* optimizer state/group keys as its unflattening schema before
        # it calls our loader. Publish the private rank-local transport keys only
        # after the base constructor has finished registering every group.
        self._gefen_checkpoint_schema_ready = False
        super().__init__(self._normalize_param_groups(params), defaults)
        self._gefen_checkpoint_schema_ready = True
        self._install_rank_local_checkpoint_schema()

    @property
    def _step_supports_amp_scaling(self) -> bool:
        # Keep standard GradScaler/Accelerate skip semantics for ordinary
        # FP32-master AMP. Native handling is needed for active local FP16
        # gradients, or statically for distributed optimizers that combine any
        # true-FP16 storage with DTensors after a collective presence preflight.
        # An explicit codebook scope is also a topology property: whole owners
        # and empty nonowners must enter the same GradScaler protocol before
        # Gefen can compare found_inf/grad_scale collectively inside step().
        if self._gefen_codebook_process_group is not None:
            return True
        return _amp_native_scaling_required(self)

    def optimizer_contract(self) -> OptimizerContract:
        """Return the immutable state-layout and integration capability contract."""

        identity_ready = self._canonical_identity_ready()
        canonical_state_layouts = self._canonical_state_layouts()
        try:
            from gefen.portable_runtime import _portable_runtime_layouts

            canonical_global_same_topology = _portable_runtime_layouts(self)
        except Exception:
            canonical_global_same_topology = frozenset()
        has_factored_matrix = (
            canonical_global_same_topology
            and self._factored_v_2d
            and any(
                len(slot.shard.parameter.global_shape) == 2
                for slot in self._gefen_logical_slots
            )
        )
        if canonical_global_same_topology and not has_factored_matrix:
            canonical_global_topology_changing = frozenset(
                {
                    ParameterLayout.REPLICATED,
                    ParameterLayout.FLATTENED_ELEMENT_SHARD,
                }
            )
            canonical_global_topology_change_kinds = frozenset(
                {TopologyChange.PLACEMENT_RESHARD}
            )
        else:
            canonical_global_topology_changing = frozenset()
            canonical_global_topology_change_kinds = frozenset()
        return _gefen_contract(
            factored_v_2d=self._factored_v_2d,
            canonical_parameter_fqns=identity_ready,
            stable_shard_identity=identity_ready,
            explicit_process_group_codebook_scope=True,
            canonical_state_layouts=canonical_state_layouts,
            canonical_global_same_topology=canonical_global_same_topology,
            canonical_global_topology_changing=canonical_global_topology_changing,
            canonical_global_topology_change_kinds=canonical_global_topology_change_kinds,
            native_flattened_checkpoint=(
                self._codebook_scope_ready()
                and any(
                    shard.layout is ParameterLayout.FLATTENED_ELEMENT_SHARD
                    for _, shard in self._gefen_local_shard_bindings
                )
            ),
        )

    def _canonical_state_variant_layout(self):
        return _gefen_contract(
            factored_v_2d=self._factored_v_2d,
        ).state_layout

    def _canonical_state_layout_supported(self, layout) -> bool:
        return layout in {
            ParameterLayout.REPLICATED,
            ParameterLayout.FLATTENED_ELEMENT_SHARD,
        }

    @staticmethod
    def _canonical_group_options_value(group):
        return {
            key: value
            for key, value in group.items()
            if key
            not in {
                "params",
                "param_names",
                "name",
                "_gefen_checkpoint_metadata",
            }
        }

    def _canonical_state_layouts(self):
        if not self._canonical_identity_ready():
            return frozenset()
        if self._stochastic_round:
            return frozenset()
        if any(
            parameter is None
            or not self._canonical_state_layout_supported(shard.layout)
            for parameter, shard in self._gefen_local_shard_bindings
        ):
            return frozenset()
        if any(
            not canonical_value_supported(
                self._canonical_group_options_value(group),
                finite_tensors=True,
            )
            for group in self.param_groups
        ):
            return frozenset()
        if not canonical_value_supported(
            self._canonical_policy(),
            finite_tensors=True,
        ):
            return frozenset()
        if not canonical_value_supported(
            self._gefen_codebook,
            finite_tensors=True,
        ):
            return frozenset()
        try:
            global_step = self._canonical_common_global_step()
            if type(global_step) is not int or global_step < 0:
                return frozenset()
            self._validate_loaded_native_state()
        except Exception:
            return frozenset()
        state_layout = self._canonical_state_variant_layout()
        for group in self.param_groups:
            group_options = self._canonical_group_options_value(group)
            for parameter in group["params"]:
                state = self.state.get(parameter, {})
                if any(
                    key not in _CANONICAL_PARAMETER_STATE_KEYS
                    and key not in _CANONICAL_DERIVED_PARAMETER_STATE_KEYS
                    for key in state
                ):
                    return frozenset()
                if any(
                    not canonical_value_supported(
                        value,
                        finite_tensors=True,
                    )
                    for key, value in state.items()
                    if key in _CANONICAL_PARAMETER_STATE_KEYS
                ):
                    return frozenset()
                canonical_state = {
                    key: value
                    for key, value in state.items()
                    if key in _CANONICAL_PARAMETER_STATE_KEYS and key != "name"
                }
                shard = self._gefen_shard_bindings[parameter]
                try:
                    self._validate_canonical_parameter_semantics(
                        parameter,
                        shard,
                        group_options,
                        canonical_state,
                        state_layout,
                        global_step,
                    )
                except (TypeError, ValueError, RuntimeError):
                    return frozenset()
        return frozenset(shard.layout for _, shard in self._gefen_local_shard_bindings)

    def _codebook_scope_ready(self) -> bool:
        return (
            self._gefen_codebook_process_group is not None
            and self._canonical_identity_ready()
        )

    def _canonical_identity_ready(self) -> bool:
        if (
            not self._gefen_post_sharding_finalized
            or self._gefen_sharding_manifest is None
        ):
            return False
        # Contract readiness is an honest external claim, so it never trusts
        # the per-step fast-path verdict.
        return self._finalized_binding_layout_matches(full=True)

    def _invalidate_layout_forensics_caches(self) -> None:
        self._gefen_layout_version = getattr(self, "_gefen_layout_version", 0) + 1
        self._gefen_layout_forensics_verdict = None

    @staticmethod
    def _forensics_tokens_match(cached, live) -> bool:
        # Containers, tensors, and strings compare by object identity — the
        # cached tuple holds strong references, so an unchanged attribute is
        # the same object. Plain ints (version counter, lengths) compare by
        # value because CPython interns only small integers.
        return len(cached) == len(live) and all(
            cached_item is live_item
            or (
                type(cached_item) is int
                and type(live_item) is int
                and cached_item == live_item
            )
            for cached_item, live_item in zip(cached, live)
        )

    def _layout_forensics_fast_tokens(self):
        # O(local params) identity snapshot of everything user code can reach
        # between two guard calls: the finalized registries by object identity
        # (they are replaced, never mutated, by legitimate APIs) plus every
        # live group container, parameter, and compatibility name element, so
        # in-place mutation of the public groups — e.g. a closure swapping one
        # ``group["params"]`` slot — still falls back to the full forensic
        # pass before any state mutation.
        live = [
            getattr(self, "_gefen_layout_version", 0),
            self._gefen_post_sharding_finalized,
            self._gefen_sharding_manifest,
            self._gefen_logical_slots,
            self._gefen_finalized_slots,
            self._gefen_local_shard_bindings,
            self._gefen_shard_bindings,
            len(self._gefen_shard_bindings),
            self._param_names,
            len(self._param_names),
            self.param_groups,
            len(self.param_groups),
            self.state,
        ]
        live.extend(self._param_names.keys())
        live.extend(self._param_names.values())
        for group in self.param_groups:
            params = group.get("params") if type(group) is dict else None
            names = group.get("param_names") if type(group) is dict else None
            live.append(group)
            live.append(params)
            live.append(names)
            if isinstance(params, (list, tuple)):
                live.append(len(params))
                live.extend(params)
                for parameter in params:
                    parameter_state = self.state.get(parameter)
                    live.append(type(parameter_state))
                    live.append(
                        parameter_state.get("name")
                        if type(parameter_state) is dict
                        else None
                    )
            if isinstance(names, (list, tuple)):
                live.append(len(names))
                live.extend(names)
        return tuple(live)

    def _manifest_layout_forensics(self, *, refresh: bool = False):
        """Return the cached ``(shard set, digest)`` for the live manifest.

        Both values are pure functions of one immutable ``ShardingManifest``,
        so they are computed once per manifest object (at post_sharding
        finalization) instead of rehashing every ``ShardIdentity`` — including
        its world-size ``ordered_members`` tuple — on every step. Full
        forensic passes recompute with ``refresh=True`` so boundary checks
        never trust a stale cache.
        """

        manifest = self._gefen_sharding_manifest
        cache = getattr(self, "_gefen_manifest_forensics_cache", None)
        if not refresh and cache is not None and cache[0] is manifest:
            return cache[1], cache[2]
        shards = frozenset(manifest.shards)
        digest = self._compute_codebook_manifest_fingerprint(manifest)
        self._gefen_manifest_forensics_cache = (manifest, shards, digest)
        return shards, digest

    def _finalized_binding_layout_matches(self, *, full: bool = False) -> bool:
        try:
            verdict = getattr(self, "_gefen_layout_forensics_verdict", None)
            if not full and verdict is not None:
                if self._forensics_tokens_match(
                    verdict, self._layout_forensics_fast_tokens()
                ):
                    return True
            self._gefen_layout_forensics_verdict = None
            matches = self._finalized_binding_layout_matches_full()
            if matches:
                self._gefen_layout_forensics_verdict = (
                    self._layout_forensics_fast_tokens()
                )
            return matches
        except AttributeError:
            return False

    def _finalized_binding_layout_matches_full(self) -> bool:
        try:
            if (
                type(self._gefen_logical_slots) is not tuple
                or not self._gefen_logical_slots
                or type(self._gefen_finalized_slots) is not tuple
                or type(self._gefen_local_shard_bindings) is not tuple
                or type(self._gefen_shard_bindings) is not dict
                or type(self._param_names) is not dict
                or not isinstance(self._gefen_sharding_manifest, ShardingManifest)
                or len(self.param_groups) != len(self._gefen_finalized_slots)
            ):
                return False

            logical_groups = [[] for _ in self.param_groups]
            logical_fqns = set()
            previous_position = None
            for logical_slot in self._gefen_logical_slots:
                if (
                    type(logical_slot) is not LogicalSlotBinding
                    or type(logical_slot.group_index) is not int
                    or type(logical_slot.original_slot_index) is not int
                    or type(logical_slot.compatibility_name) is not str
                    or logical_slot.compatibility_name
                    != logical_slot.compatibility_name.lower()
                    or not isinstance(logical_slot.shard, ShardIdentity)
                ):
                    return False
                position = (
                    logical_slot.group_index,
                    logical_slot.original_slot_index,
                )
                if (
                    logical_slot.group_index < 0
                    or logical_slot.group_index >= len(logical_groups)
                    or logical_slot.original_slot_index
                    != len(logical_groups[logical_slot.group_index])
                ):
                    return False
                if previous_position is None:
                    if position != (0, 0):
                        return False
                elif logical_slot.group_index == previous_position[0]:
                    if logical_slot.original_slot_index != previous_position[1] + 1:
                        return False
                elif position != (previous_position[0] + 1, 0):
                    return False
                fqn = logical_slot.shard.parameter.fqn
                if fqn in logical_fqns:
                    return False
                logical_fqns.add(fqn)
                logical_groups[logical_slot.group_index].append(logical_slot)
                previous_position = position

            if any(not group for group in logical_groups):
                return False

            manifest_shards = self._manifest_layout_forensics(refresh=True)[0]
            if {
                shard.parameter.fqn for shard in manifest_shards
            } != logical_fqns or any(
                logical_slot.shard not in manifest_shards
                for logical_slot in self._gefen_logical_slots
            ):
                return False

            local_bindings = {}
            previous_sort_key = None
            for item in self._gefen_local_shard_bindings:
                if type(item) is not tuple or len(item) != 2:
                    return False
                parameter, shard = item
                if parameter is not None and not isinstance(parameter, torch.Tensor):
                    return False
                if not isinstance(shard, ShardIdentity):
                    return False
                if previous_sort_key is not None and shard.sort_key < previous_sort_key:
                    return False
                previous_sort_key = shard.sort_key
                fqn = shard.parameter.fqn
                if fqn in local_bindings:
                    return False
                local_bindings[fqn] = (parameter, shard)
            if set(local_bindings) != logical_fqns:
                return False

            expected_live_groups = []
            expected_name_groups = []
            expected_live_bindings = []
            for logical_group in logical_groups:
                live_group = []
                name_group = []
                for logical_slot in logical_group:
                    parameter, shard = local_bindings[logical_slot.shard.parameter.fqn]
                    if shard != logical_slot.shard:
                        return False
                    pruned = (
                        shard.layout is ParameterLayout.WHOLE_PARAMETER_OWNER
                        and shard.local_member != shard.owner
                    )
                    if pruned:
                        if parameter is not None:
                            return False
                        continue
                    if parameter is None:
                        return False
                    live_group.append(parameter)
                    name_group.append(logical_slot.compatibility_name)
                    expected_live_bindings.append((parameter, shard))
                expected_live_groups.append(tuple(live_group))
                expected_name_groups.append(tuple(name_group))

            if len(self._gefen_shard_bindings) != len(expected_live_bindings):
                return False
            expected_live_ids = {
                id(parameter) for parameter, _ in expected_live_bindings
            }
            if len(expected_live_ids) != len(expected_live_bindings):
                return False
            if {
                id(parameter) for parameter in self._gefen_shard_bindings
            } != expected_live_ids:
                return False
            for parameter, shard in expected_live_bindings:
                if self._gefen_shard_bindings.get(parameter) != shard:
                    return False

            if len(self._param_names) != len(expected_live_bindings):
                return False
            if {id(parameter) for parameter in self._param_names} != expected_live_ids:
                return False
            for group_index, (group, expected_params, expected_names) in enumerate(
                zip(self.param_groups, expected_live_groups, expected_name_groups)
            ):
                params = group.get("params")
                names = group.get("param_names")
                finalized = self._gefen_finalized_slots[group_index]
                if (
                    not isinstance(params, (list, tuple))
                    or not isinstance(names, (list, tuple))
                    or type(finalized) is not tuple
                    or len(params) != len(expected_params)
                    or len(names) != len(expected_names)
                    or len(finalized) != len(expected_params)
                ):
                    return False
                if any(
                    live is not expected
                    for live, expected in zip(params, expected_params)
                ) or any(
                    bound is not expected
                    for bound, expected in zip(finalized, expected_params)
                ):
                    return False
                if tuple(names) != expected_names:
                    return False
                for parameter, expected_name in zip(expected_params, expected_names):
                    if self._param_names.get(parameter) != expected_name:
                        return False
                    parameter_state = self.state.get(parameter)
                    if (
                        type(parameter_state) is not dict
                        or parameter_state.get("name") != expected_name
                    ):
                        return False
            return True
        except (
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            return False

    def _assert_finalized_binding_layout(self, *, full: bool = False) -> None:
        if (
            self._gefen_post_sharding_finalized
            and not self._finalized_binding_layout_matches(full=full)
        ):
            raise RuntimeError(
                "Gefen finalized parameter layout changed outside post_sharding"
            )

    def parameter_identity(self, parameter) -> ParameterIdentity:
        """Return the exact canonical identity bound to one live parameter."""

        return self.shard_identity(parameter).parameter

    def shard_identity(self, parameter) -> ShardIdentity:
        """Return the stable shard identity bound to one live parameter."""

        self._assert_finalized_binding_layout()
        try:
            return self._gefen_shard_bindings[parameter]
        except KeyError:
            raise KeyError("parameter has no finalized Gefen shard identity") from None

    def shard_bindings(self):
        """Return local tensor/identity pairs in canonical structural order."""

        self._assert_finalized_binding_layout()
        return self._gefen_local_shard_bindings

    def sharding_manifest(self):
        """Return the finalized global identity manifest, or ``None``."""

        self._assert_finalized_binding_layout()
        return self._gefen_sharding_manifest

    def codebook_process_group_binding(self):
        """Return the finalized explicit learned-codebook binding, or ``None``."""

        self._assert_finalized_binding_layout()
        return self._gefen_codebook_process_group

    def _serialized_codebook_scope(self):
        binding = self._gefen_codebook_process_group
        if binding is None:
            return None
        return {
            "format_version": _CODEBOOK_SCOPE_FORMAT_VERSION,
            "semantic_name": binding.identity.semantic_name,
            "ordered_members": list(binding.identity.ordered_members),
            "refresh_every": self._codebook_refresh_every,
        }

    @staticmethod
    def _normalize_serialized_codebook_scope(value):
        if value is None:
            return None
        if not isinstance(value, dict) or set(value) != {
            "format_version",
            "semantic_name",
            "ordered_members",
            "refresh_every",
        }:
            raise ValueError("Gefen checkpoint codebook scope has an invalid schema")
        if (
            type(value["format_version"]) is not int
            or value["format_version"] != _CODEBOOK_SCOPE_FORMAT_VERSION
        ):
            raise ValueError(
                "Gefen checkpoint codebook scope has an unsupported format version"
            )
        if not isinstance(value["ordered_members"], list):
            raise ValueError(
                "Gefen checkpoint codebook scope ordered_members must be a list"
            )
        from gefen.contracts import ProcessGroupIdentity

        try:
            identity = ProcessGroupIdentity(
                value["semantic_name"], tuple(value["ordered_members"])
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Gefen checkpoint codebook scope identity is invalid"
            ) from exc
        refresh_every = value["refresh_every"]
        if type(refresh_every) is not int or refresh_every < 0:
            raise ValueError(
                "Gefen checkpoint codebook scope refresh_every must be a nonnegative integer"
            )
        return {
            "format_version": _CODEBOOK_SCOPE_FORMAT_VERSION,
            "semantic_name": identity.semantic_name,
            "ordered_members": list(identity.ordered_members),
            "refresh_every": refresh_every,
        }

    @staticmethod
    def _serialized_native_local_shard(shard):
        return {
            "fqn": shard.parameter.fqn,
            "global_shape": list(shard.parameter.global_shape),
            "layout": shard.layout.value,
            "flat_offset": shard.logical_slice.flat_offset,
            "length": shard.logical_slice.length,
            "process_group": {
                "semantic_name": shard.process_group.semantic_name,
                "ordered_members": list(shard.process_group.ordered_members),
            },
            "local_member": shard.local_member,
            "owner": shard.owner,
            "placements": [
                {
                    "mesh_axis": placement.mesh_axis,
                    "kind": placement.kind.value,
                    "coordinate": placement.coordinate,
                    "parts": placement.parts,
                    "parameter_dimension": placement.parameter_dimension,
                }
                for placement in shard.placements
            ],
        }

    @staticmethod
    def _serialized_canonical_shard(shard):
        process_group = None
        if shard.process_group is not None:
            process_group = {
                "schema_version": shard.process_group.schema_version,
                "semantic_name": shard.process_group.semantic_name,
                "ordered_members": list(shard.process_group.ordered_members),
            }
        return {
            "schema_version": shard.schema_version,
            "parameter": {
                "schema_version": shard.parameter.schema_version,
                "fqn": shard.parameter.fqn,
                "global_shape": list(shard.parameter.global_shape),
            },
            "layout": shard.layout.value,
            "logical_slice": {
                "flat_offset": shard.logical_slice.flat_offset,
                "length": shard.logical_slice.length,
            },
            "process_group": process_group,
            "local_member": shard.local_member,
            "owner": shard.owner,
            "placements": [
                {
                    "mesh_axis": placement.mesh_axis,
                    "kind": placement.kind.value,
                    "coordinate": placement.coordinate,
                    "parts": placement.parts,
                    "parameter_dimension": placement.parameter_dimension,
                }
                for placement in shard.placements
            ],
        }

    @classmethod
    def _normalize_serialized_canonical_shard(cls, record):
        if not isinstance(record, dict) or set(record) != {
            "schema_version",
            "parameter",
            "layout",
            "logical_slice",
            "process_group",
            "local_member",
            "owner",
            "placements",
        }:
            raise ValueError("Gefen canonical shard has an invalid schema")
        parameter_record = record["parameter"]
        if not isinstance(parameter_record, dict) or set(parameter_record) != {
            "schema_version",
            "fqn",
            "global_shape",
        }:
            raise ValueError("Gefen canonical parameter identity is invalid")
        if not isinstance(parameter_record["global_shape"], list):
            raise ValueError("Gefen canonical global_shape must be a list")
        slice_record = record["logical_slice"]
        if not isinstance(slice_record, dict) or set(slice_record) != {
            "flat_offset",
            "length",
        }:
            raise ValueError("Gefen canonical logical slice is invalid")
        group_record = record["process_group"]
        if group_record is not None and (
            not isinstance(group_record, dict)
            or set(group_record)
            != {"schema_version", "semantic_name", "ordered_members"}
            or not isinstance(group_record["ordered_members"], list)
        ):
            raise ValueError("Gefen canonical process-group identity is invalid")
        if not isinstance(record["placements"], list):
            raise ValueError("Gefen canonical placements must be a list")
        try:
            parameter = ParameterIdentity(
                parameter_record["fqn"],
                tuple(parameter_record["global_shape"]),
                schema_version=parameter_record["schema_version"],
            )
            process_group = None
            if group_record is not None:
                process_group = ProcessGroupIdentity(
                    group_record["semantic_name"],
                    tuple(group_record["ordered_members"]),
                    schema_version=group_record["schema_version"],
                )
            placements = []
            for placement_record in record["placements"]:
                if not isinstance(placement_record, dict) or set(placement_record) != {
                    "mesh_axis",
                    "kind",
                    "coordinate",
                    "parts",
                    "parameter_dimension",
                }:
                    raise ValueError("invalid placement schema")
                placements.append(
                    ShardPlacement(
                        placement_record["mesh_axis"],
                        PlacementKind(placement_record["kind"]),
                        placement_record["coordinate"],
                        placement_record["parts"],
                        placement_record["parameter_dimension"],
                    )
                )
            shard = ShardIdentity(
                parameter,
                ParameterLayout(record["layout"]),
                LogicalSlice(slice_record["flat_offset"], slice_record["length"]),
                process_group=process_group,
                local_member=record["local_member"],
                owner=record["owner"],
                placements=tuple(placements),
                schema_version=record["schema_version"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Gefen canonical shard identity is invalid") from exc
        return cls._serialized_canonical_shard(shard)

    def _serialized_native_local_shards_v1(self):
        if self._gefen_codebook_process_group is None:
            return None
        if not any(
            shard.layout is not ParameterLayout.REPLICATED
            for _, shard in self._gefen_local_shard_bindings
        ):
            return None

        # Native Optimizer.load_state_dict maps per-parameter state by
        # parameter-group and slot position, not by canonical FQN. Bind every
        # serialized live slot to its shard identity (including replicated
        # slots in a mixed optimizer), and retain pruned whole-owner records
        # separately. A canonical shard set alone would not distinguish A/B
        # from B/A when equal-shaped parameters exchange positions.
        param_groups = []
        for group in self.param_groups:
            param_groups.append(
                [
                    self._serialized_native_local_shard(
                        self._gefen_shard_bindings[parameter]
                    )
                    for parameter in group["params"]
                ]
            )
        pruned_shards = [
            self._serialized_native_local_shard(shard)
            for parameter, shard in self._gefen_local_shard_bindings
            if parameter is None
        ]
        return {
            "format_version": _NATIVE_LOCAL_SHARDS_LEGACY_FORMAT_VERSION,
            "param_groups": param_groups,
            "pruned_shards": pruned_shards,
        }

    def _serialized_native_local_shards(self):
        if self._gefen_codebook_process_group is None:
            return None
        if not any(
            shard.layout is not ParameterLayout.REPLICATED
            for _, shard in self._gefen_local_shard_bindings
        ):
            return None
        return {
            "format_version": _NATIVE_LOCAL_SHARDS_FORMAT_VERSION,
            "logical_slots": [
                {
                    "group_index": slot.group_index,
                    "original_slot_index": slot.original_slot_index,
                    "compatibility_name": slot.compatibility_name,
                    "shard": _serialize_shard_identity(slot.shard),
                }
                for slot in self._gefen_logical_slots
            ],
        }

    @classmethod
    def _normalize_serialized_native_local_shard(cls, record):
        expected = {
            "fqn",
            "global_shape",
            "layout",
            "flat_offset",
            "length",
            "process_group",
            "local_member",
            "owner",
            "placements",
        }
        if not isinstance(record, dict) or set(record) != expected:
            raise ValueError("Gefen native local-shard metadata has an invalid schema")
        group_record = record["process_group"]
        if not isinstance(group_record, dict) or set(group_record) != {
            "semantic_name",
            "ordered_members",
        }:
            raise ValueError(
                "Gefen native local-shard process-group metadata is invalid"
            )
        if not isinstance(group_record["ordered_members"], list):
            raise ValueError("Gefen native local-shard ordered_members must be a list")
        if not isinstance(record["global_shape"], list) or not isinstance(
            record["placements"], list
        ):
            raise ValueError(
                "Gefen native local-shard shapes and placements must be lists"
            )
        try:
            parameter = ParameterIdentity(record["fqn"], tuple(record["global_shape"]))
            group = ProcessGroupIdentity(
                group_record["semantic_name"],
                tuple(group_record["ordered_members"]),
            )
            placements = []
            for placement_record in record["placements"]:
                if not isinstance(placement_record, dict) or set(placement_record) != {
                    "mesh_axis",
                    "kind",
                    "coordinate",
                    "parts",
                    "parameter_dimension",
                }:
                    raise ValueError("invalid placement schema")
                placements.append(
                    ShardPlacement(
                        placement_record["mesh_axis"],
                        PlacementKind(placement_record["kind"]),
                        placement_record["coordinate"],
                        placement_record["parts"],
                        placement_record["parameter_dimension"],
                    )
                )
            shard = ShardIdentity(
                parameter,
                ParameterLayout(record["layout"]),
                LogicalSlice(record["flat_offset"], record["length"]),
                process_group=group,
                local_member=record["local_member"],
                owner=record["owner"],
                placements=tuple(placements),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Gefen native local-shard metadata is invalid") from exc
        if shard.layout not in {
            ParameterLayout.REPLICATED,
            ParameterLayout.FLATTENED_ELEMENT_SHARD,
            ParameterLayout.WHOLE_PARAMETER_OWNER,
        }:
            raise ValueError(
                "Gefen native local-shard metadata has an unsupported layout"
            )
        return cls._serialized_native_local_shard(shard)

    @classmethod
    def _normalize_serialized_native_local_shards_v1(cls, value):
        if not isinstance(value, dict) or set(value) != {
            "format_version",
            "param_groups",
            "pruned_shards",
        }:
            raise ValueError("Gefen native local-shard metadata has an invalid schema")
        format_version = value["format_version"]
        if (
            type(format_version) is not int
            or format_version != _NATIVE_LOCAL_SHARDS_LEGACY_FORMAT_VERSION
        ):
            raise ValueError(
                "Unsupported Gefen native local-shard format_version: {}".format(
                    format_version
                )
            )
        param_groups = value["param_groups"]
        pruned_shards = value["pruned_shards"]
        if not isinstance(param_groups, list) or not isinstance(pruned_shards, list):
            raise ValueError(
                "Gefen native local-shard parameter groups and pruned shards must be lists"
            )
        normalized_groups = []
        for group in param_groups:
            if not isinstance(group, list):
                raise ValueError(
                    "Gefen native local-shard parameter groups must contain lists"
                )
            normalized_groups.append(
                [
                    cls._normalize_serialized_native_local_shard(record)
                    for record in group
                ]
            )
        normalized_pruned = [
            cls._normalize_serialized_native_local_shard(record)
            for record in pruned_shards
        ]
        if any(
            record["layout"] != ParameterLayout.WHOLE_PARAMETER_OWNER.value
            for record in normalized_pruned
        ):
            raise ValueError(
                "Gefen native pruned-shard metadata must describe whole-parameter nonowners"
            )
        return {
            "format_version": _NATIVE_LOCAL_SHARDS_LEGACY_FORMAT_VERSION,
            "param_groups": normalized_groups,
            "pruned_shards": normalized_pruned,
        }

    @classmethod
    def _normalize_serialized_native_local_shards_v2(cls, value):
        if type(value) is not dict or set(value) != {
            "format_version",
            "logical_slots",
        }:
            raise ValueError("Gefen native local-shard metadata has an invalid schema")
        format_version = value["format_version"]
        if (
            type(format_version) is not int
            or format_version != _NATIVE_LOCAL_SHARDS_FORMAT_VERSION
        ):
            raise ValueError(
                "Unsupported Gefen native local-shard format_version: {}".format(
                    format_version
                )
            )
        records = value["logical_slots"]
        if type(records) is not list or not records:
            raise ValueError(
                "Gefen native local-shard logical_slots must be a nonempty list"
            )

        normalized_records = []
        seen_fqns = set()
        previous_position = None
        supported_layouts = {
            ParameterLayout.REPLICATED,
            ParameterLayout.FLATTENED_ELEMENT_SHARD,
            ParameterLayout.WHOLE_PARAMETER_OWNER,
        }
        for record in records:
            if type(record) is not dict or set(record) != {
                "group_index",
                "original_slot_index",
                "compatibility_name",
                "shard",
            }:
                raise ValueError(
                    "Gefen native local-shard logical slot has an invalid schema"
                )
            if (
                type(record["shard"]) is dict
                and "process_group" in record["shard"]
                and record["shard"]["process_group"] is None
            ):
                raise ValueError(
                    "Gefen native local-shard metadata requires a process-group identity"
                )
            try:
                shard = _parse_shard_identity(record["shard"])
                slot = LogicalSlotBinding(
                    record["group_index"],
                    record["original_slot_index"],
                    record["compatibility_name"],
                    shard,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Gefen native local-shard logical slot is invalid"
                ) from exc
            position = (slot.group_index, slot.original_slot_index)
            if previous_position is None:
                valid_position = position == (0, 0)
            elif position[0] == previous_position[0]:
                valid_position = position[1] == previous_position[1] + 1
            else:
                valid_position = position == (previous_position[0] + 1, 0)
            if not valid_position:
                raise ValueError(
                    "Gefen native local-shard logical slot positions must be contiguous"
                )
            previous_position = position
            if slot.shard.layout not in supported_layouts:
                raise ValueError(
                    "Gefen native local-shard metadata has an unsupported layout"
                )
            if slot.shard.process_group is None:
                raise ValueError(
                    "Gefen native local-shard metadata requires a process-group identity"
                )
            fqn = slot.shard.parameter.fqn
            if fqn in seen_fqns:
                raise ValueError(
                    "Gefen native local-shard logical slot FQNs must be unique"
                )
            seen_fqns.add(fqn)
            normalized_records.append(
                {
                    "group_index": slot.group_index,
                    "original_slot_index": slot.original_slot_index,
                    "compatibility_name": slot.compatibility_name,
                    "shard": _serialize_shard_identity(slot.shard),
                }
            )
        return {
            "format_version": _NATIVE_LOCAL_SHARDS_FORMAT_VERSION,
            "logical_slots": normalized_records,
        }

    @classmethod
    def _normalize_serialized_native_local_shards(cls, value):
        if value is None:
            return None
        if type(value) is not dict or "format_version" not in value:
            raise ValueError("Gefen native local-shard metadata has an invalid schema")
        format_version = value["format_version"]
        if type(format_version) is not int:
            raise ValueError(
                "Unsupported Gefen native local-shard format_version: {}".format(
                    format_version
                )
            )
        if format_version == _NATIVE_LOCAL_SHARDS_LEGACY_FORMAT_VERSION:
            return cls._normalize_serialized_native_local_shards_v1(value)
        if format_version == _NATIVE_LOCAL_SHARDS_FORMAT_VERSION:
            return cls._normalize_serialized_native_local_shards_v2(value)
        raise ValueError(
            "Unsupported Gefen native local-shard format_version: {}".format(
                format_version
            )
        )

    def _native_local_shards_matches_live(self, value):
        if value is None:
            return self._serialized_native_local_shards() is None
        if value["format_version"] == _NATIVE_LOCAL_SHARDS_LEGACY_FORMAT_VERSION:
            return value == self._serialized_native_local_shards_v1()
        if value["format_version"] == _NATIVE_LOCAL_SHARDS_FORMAT_VERSION:
            return value == self._serialized_native_local_shards()
        return False

    @staticmethod
    def _parameter_in(parameters, candidate) -> bool:
        return any(item is candidate for item in parameters)

    @staticmethod
    def _assert_rebound_storage_disjoint(parameters) -> None:
        storage_ranges = []
        for parameter in parameters:
            if parameter.numel() == 0:
                continue
            storage = parameter.untyped_storage()
            storage_id = (str(parameter.device), storage.data_ptr())
            if not parameter.is_contiguous():
                for other_id, _, _ in storage_ranges:
                    if other_id == storage_id:
                        raise ValueError(
                            "rebound target tensors must not share noncontiguous storage"
                        )
                storage_ranges.append((storage_id, None, None))
                continue
            start = parameter.storage_offset() * parameter.element_size()
            end = start + parameter.numel() * parameter.element_size()
            for other_id, other_start, other_end in storage_ranges:
                if other_id != storage_id:
                    continue
                if other_start is None or max(start, other_start) < min(end, other_end):
                    raise ValueError(
                        "rebound target tensor storage ranges must not overlap"
                    )
            storage_ranges.append((storage_id, start, end))

    @staticmethod
    def _target_may_have_internal_storage_overlap(parameter) -> bool:
        """Conservatively prove disjoint positive-stride tensor elements."""

        required_span = 1
        dimensions = sorted(
            (stride, size)
            for size, stride in zip(parameter.shape, parameter.stride())
            if size > 1
        )
        for stride, size in dimensions:
            if stride < required_span:
                return True
            required_span += (size - 1) * stride
        return False

    def _assert_rebinding_pristine(self, rebindings) -> None:
        if self._gefen_post_sharding_finalized:
            raise RuntimeError("Gefen post-sharding identity is already finalized")
        if (
            self._gefen_shard_bindings
            or self._gefen_local_shard_bindings
            or self._gefen_sharding_manifest is not None
            or self._gefen_finalized_slots
            or self._gefen_logical_slots
        ):
            raise RuntimeError(
                "Gefen parameter rebinding found an incomplete prior identity plan"
            )
        global_step = self._validate_rank_local_counter(
            "gefen_global_step", self._gefen_global_step
        )
        if global_step != 0 or self._gefen_codebook is not None:
            raise RuntimeError(
                "Gefen parameter rebinding is allowed only before optimizer mutation"
            )
        if self._capt_stacks is not None:
            raise RuntimeError(
                "Gefen parameter rebinding cannot discard active capturable stacks"
            )
        if self._gefen_codebook_by_device or self._gefen_codebook_lut_by_device:
            raise RuntimeError(
                "Gefen parameter rebinding cannot discard initialized codebook caches"
            )
        for cache_name in (
            "_gefen_global_step_by_device",
            "_sr_seed_by_device",
        ):
            cache = getattr(self, cache_name)
            for value in cache.values():
                if (
                    not torch.is_tensor(value)
                    or value.numel() != 1
                    or value.dtype == torch.bool
                    or not bool(torch.isfinite(value.detach()).all())
                    or float(value.detach().cpu().item()) != 0.0
                ):
                    raise RuntimeError(
                        "Gefen parameter rebinding found non-pristine device counters"
                    )

        live = [param for group in self.param_groups for param in group["params"]]
        sources = [item.old_parameter for item in rebindings]
        allowed_state_keys = tuple(live) + tuple(sources)
        for parameter in self._param_names:
            if not self._parameter_in(allowed_state_keys, parameter):
                raise RuntimeError(
                    "Gefen parameter rebinding found orphan parameter-name state"
                )
            compatibility_name = self._param_names[parameter]
            if (
                not isinstance(compatibility_name, str)
                or compatibility_name != compatibility_name.lower()
            ):
                raise RuntimeError(
                    "Gefen parameter rebinding found an invalid compatibility name"
                )
        allowed_names = {"name", _RANK_LOCAL_MEMBER_KEY}
        for parameter, pstate in self.state.items():
            if not self._parameter_in(allowed_state_keys, parameter):
                raise RuntimeError(
                    "Gefen parameter rebinding found orphan optimizer state"
                )
            if not isinstance(pstate, dict):
                raise RuntimeError(
                    "Gefen parameter rebinding found invalid constructor state"
                )
            for key in pstate:
                if key in allowed_names:
                    continue
                if isinstance(key, str) and key.startswith(
                    _RANK_LOCAL_PAYLOAD_KEY_PREFIX
                ):
                    continue
                raise RuntimeError(
                    "Gefen parameter rebinding found initialized optimizer state"
                )
            state_name = pstate.get("name")
            if state_name is not None and (
                not isinstance(state_name, str) or state_name != state_name.lower()
            ):
                raise RuntimeError(
                    "Gefen parameter rebinding found an invalid state name"
                )

    def _validate_rebinding_layout(self, rebinding: ParameterRebinding) -> None:
        shard = rebinding.shard
        target = rebinding.new_parameter
        if shard.layout is ParameterLayout.REPLICATED:
            if target is None or tuple(target.shape) != shard.parameter.global_shape:
                raise ValueError(
                    "replicated Gefen rebinding requires complete parameter storage"
                )
        elif shard.layout is ParameterLayout.FLATTENED_ELEMENT_SHARD:
            if target is None or target.ndim != 1:
                raise ValueError(
                    "flattened Gefen rebinding requires a live 1-D tensor shard"
                )
            if not target.is_contiguous():
                raise ValueError(
                    "flattened Gefen rebinding requires contiguous physical storage"
                )
            if self._factored_v_2d and len(shard.parameter.global_shape) == 2:
                raise ValueError(
                    "flattened logical matrices require factored_v_2d=False until "
                    "matrix-aware factored-state projection is implemented"
                )
        else:
            raise ValueError(
                "plain Gefen rebinding supports replicated or flattened element shards only"
            )
        if target.numel() != shard.logical_slice.length:
            raise ValueError(
                "Gefen rebound tensor numel does not match its logical slice"
            )

    @staticmethod
    def _codebook_scope_backend_device_is_valid(backend: str, device) -> bool:
        backend = str(backend).lower()
        if "nccl" in backend:
            return device.type == "cuda" and device.index is not None
        if "gloo" in backend or "mpi" in backend:
            return device.type == "cpu"
        return device.type in {"cpu", "cuda"}

    def _validate_codebook_process_group_binding(
        self, binding, rebindings, manifest
    ) -> None:
        if not isinstance(binding, CodebookProcessGroupBinding):
            raise TypeError(
                "codebook_process_group must be a CodebookProcessGroupBinding"
            )
        identity = binding.identity
        if any(shard.process_group != identity for shard in manifest.shards):
            raise ValueError(
                "every scoped manifest shard must use the codebook process-group identity"
            )
        for rebinding in rebindings:
            shard = rebinding.shard
            if shard.local_member != binding.local_member:
                raise ValueError(
                    "local shard members must match the codebook binding member"
                )
            if shard.layout not in {
                ParameterLayout.REPLICATED,
                ParameterLayout.FLATTENED_ELEMENT_SHARD,
                ParameterLayout.WHOLE_PARAMETER_OWNER,
            }:
                raise ValueError(
                    "explicit codebook scope supports replicated, flattened, or whole-parameter owner identities"
                )

        self._validate_codebook_runtime_binding(binding)

    def _validate_codebook_runtime_binding(self, binding) -> None:
        members = binding.identity.ordered_members
        if len(members) == 1:
            if binding.process_group is not None:
                raise ValueError(
                    "a one-member codebook scope must use process_group=None"
                )
            return

        if self.capturable:
            raise ValueError(
                "capturable=True does not support a multi-member explicit codebook scope"
            )

        if binding.process_group is None:
            raise ValueError(
                "a multi-member codebook scope requires an explicit runtime process group"
            )
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            raise RuntimeError(
                "a multi-member codebook scope requires initialized torch.distributed"
            )
        import torch.distributed as dist

        try:
            world = dist.get_world_size(binding.process_group)
            group_rank = dist.get_group_rank(binding.process_group, dist.get_rank())
            backend = dist.get_backend(binding.process_group)
        except Exception as exc:
            raise ValueError(
                "the current rank must belong to the explicit codebook process group"
            ) from exc
        if world != len(members):
            raise ValueError(
                "codebook process-group world size does not match its stable identity"
            )
        if group_rank < 0 or members[group_rank] != binding.local_member:
            raise ValueError(
                "runtime codebook group order does not match ordered semantic members"
            )
        if not self._codebook_scope_backend_device_is_valid(
            backend, binding.collective_device
        ):
            raise ValueError(
                "codebook collective device is incompatible with the runtime backend"
            )
        if binding.collective_device.type == "cuda":
            index = binding.collective_device.index
            if (
                not torch.cuda.is_available()
                or index is None
                or index < 0
                or index >= torch.cuda.device_count()
            ):
                raise ValueError("codebook collective CUDA device is unavailable")

    @torch._dynamo.disable
    def _assert_runtime_codebook_process_group(self, *, full: bool = False) -> None:
        binding = self._gefen_codebook_process_group
        if binding is None:
            return
        self._assert_finalized_binding_layout(full=full)
        self._validate_codebook_runtime_binding(binding)

    def _capture_codebook_scope_binding_for_step(self):
        """Return a usable failure-vote binding before inspecting live layout."""

        binding = self._gefen_codebook_process_group
        if binding is None:
            return None
        if not isinstance(binding, CodebookProcessGroupBinding):
            raise TypeError("the runtime codebook process-group binding is invalid")
        self._validate_codebook_runtime_binding(binding)
        return binding

    def _codebook_parameter_contributes(self, parameter) -> bool:
        binding = self._gefen_codebook_process_group
        if binding is None:
            return True
        shard = self._gefen_shard_bindings.get(parameter)
        if shard is None:
            raise RuntimeError(
                "scoped codebook parameter has no finalized shard identity"
            )
        if shard.layout is ParameterLayout.REPLICATED:
            return shard.local_member == binding.identity.ordered_members[0]
        return True

    def _stage_post_sharding(self, rebindings, manifest, codebook_process_group=None):
        self._assert_rebinding_pristine(rebindings)
        live_slots = []
        for group_index, group in enumerate(self.param_groups):
            params = list(group["params"])
            names = list(group.get("param_names", ()))
            if len(names) != len(params):
                names = [self._param_name(param) for param in params]
            for parameter_index, (parameter, name) in enumerate(zip(params, names)):
                live_slots.append((group_index, parameter_index, parameter, str(name)))
        if len(rebindings) != len(live_slots):
            raise ValueError(
                "post_sharding requires exactly one rebinding for every optimizer slot"
            )

        manifest_fqns = {shard.parameter.fqn for shard in manifest.shards}
        local_fqns = [item.shard.parameter.fqn for item in rebindings]
        if len(set(local_fqns)) != len(local_fqns):
            raise ValueError("local canonical parameter FQNs must be unique")
        if set(local_fqns) != manifest_fqns:
            raise ValueError(
                "post_sharding manifest FQNs must exactly match optimizer slots"
            )

        assigned_positions = {}
        binding_names = {}
        final_targets = []
        seen_sources = []
        local_member_by_group = {}
        for binding_index, rebinding in enumerate(rebindings):
            if not isinstance(rebinding.old_parameter, torch.Tensor):
                raise TypeError("rebinding source must be a Tensor")
            if self._parameter_in(seen_sources, rebinding.old_parameter):
                raise ValueError("rebinding source tensors must be unique")
            seen_sources.append(rebinding.old_parameter)
            process_group = rebinding.shard.process_group
            if process_group is not None:
                previous_member = local_member_by_group.setdefault(
                    process_group, rebinding.shard.local_member
                )
                if previous_member != rebinding.shard.local_member:
                    raise ValueError(
                        "local shard bindings must use one member per process group"
                    )
            target = rebinding.new_parameter
            if target is not None:
                if not isinstance(target, torch.Tensor):
                    raise TypeError("rebinding target must be a Tensor or None")
                if self._is_dtensor_parameter(target):
                    raise ValueError(
                        "stable DTensor rebinding requires the deferred logical-region identity schema"
                    )
                if torch.is_complex(target):
                    raise ValueError(
                        "Gefen does not support complex rebound parameters"
                    )
                if target.layout is not torch.strided or target.dtype not in (
                    torch.float16,
                    torch.bfloat16,
                    torch.float32,
                    torch.float64,
                ):
                    raise ValueError(
                        "rebound parameters require strided floating-point storage"
                    )
                if target.is_meta:
                    raise ValueError("rebound parameters require materialized storage")
                if (
                    rebinding.shard.layout is ParameterLayout.FLATTENED_ELEMENT_SHARD
                    and not target.is_contiguous()
                ):
                    raise ValueError(
                        "flattened Gefen rebinding requires contiguous physical storage"
                    )
                if self._target_may_have_internal_storage_overlap(target):
                    raise ValueError(
                        "rebound target storage must be provably free of internal storage overlap"
                    )
                if not target.is_leaf and not target.retains_grad:
                    raise ValueError("can't optimize a non-leaf rebound Tensor")
            if rebinding.old_parameter.grad is not None or (
                target is not None and target.grad is not None
            ):
                raise RuntimeError(
                    "post_sharding must run before source or target gradients exist"
                )
            if rebinding.shard not in manifest.shards:
                raise ValueError("local shard identity is absent from the manifest")
            whole_owner = (
                rebinding.shard.layout is ParameterLayout.WHOLE_PARAMETER_OWNER
            )
            local_owns = rebinding.shard.local_member == rebinding.shard.owner
            if whole_owner and local_owns != (target is not None):
                raise ValueError(
                    "whole-parameter owner bindings require storage only on the owner"
                )
            if not whole_owner and target is None:
                raise ValueError("only a whole-parameter non-owner may bind None")

            source_positions = [
                index
                for index, slot in enumerate(live_slots)
                if slot[2] is rebinding.old_parameter
            ]
            target_positions = []
            if target is not None:
                target_positions = [
                    index for index, slot in enumerate(live_slots) if slot[2] is target
                ]
            if len(source_positions) == 1:
                position = source_positions[0]
                if target_positions and target_positions != [position]:
                    raise ValueError(
                        "rebinding target already occupies another optimizer slot"
                    )
            elif len(source_positions) == 0 and len(target_positions) == 1:
                position = target_positions[0]
                source_known = (
                    rebinding.old_parameter in self.state
                    or rebinding.old_parameter in self._param_names
                )
                if not source_known:
                    raise ValueError("rebinding source is not registered with Gefen")
            else:
                raise ValueError(
                    "rebinding source must identify exactly one optimizer slot"
                )
            if position in assigned_positions:
                raise ValueError("multiple rebindings target one optimizer slot")
            assigned_positions[position] = binding_index
            compatibility_name = self._param_names.get(rebinding.old_parameter)
            source_state = self.state.get(rebinding.old_parameter)
            if compatibility_name is None and isinstance(source_state, dict):
                compatibility_name = source_state.get("name")
            if compatibility_name is None:
                compatibility_name = live_slots[position][3]
            if (
                not isinstance(compatibility_name, str)
                or compatibility_name != compatibility_name.lower()
            ):
                raise ValueError(
                    "rebinding source compatibility name must be a lowercase string"
                )
            binding_names[binding_index] = compatibility_name
            if target is not None:
                if self._parameter_in(final_targets, target):
                    raise ValueError("rebound target tensors must be unique")
                final_targets.append(target)
            self._validate_rebinding_layout(rebinding)

        if len(assigned_positions) != len(live_slots):
            raise ValueError("post_sharding did not bind every optimizer slot")
        self._assert_rebound_storage_disjoint(final_targets)
        if codebook_process_group is not None:
            self._validate_codebook_process_group_binding(
                codebook_process_group, rebindings, manifest
            )

        staged = object.__new__(type(self))
        staged.__dict__ = self.__dict__.copy()
        staged.param_groups = []
        staged.state = defaultdict(dict)
        staged._param_names = {}
        staged._gefen_shard_bindings = {}
        local_bindings = []
        logical_slots = []
        slot_cursor = 0
        for group_index, group in enumerate(self.param_groups):
            staged_group = dict(group)
            staged_group.pop("_gefen_checkpoint_metadata", None)
            staged_params = []
            staged_names = []
            names = list(group.get("param_names", ()))
            if len(names) != len(group["params"]):
                names = [self._param_name(param) for param in group["params"]]
            for original_slot_index, _ in enumerate(names):
                binding_index = assigned_positions[slot_cursor]
                rebinding = rebindings[binding_index]
                compatibility_name = binding_names[binding_index]
                slot_cursor += 1
                target = rebinding.new_parameter
                local_bindings.append((target, rebinding.shard))
                logical_slots.append(
                    LogicalSlotBinding(
                        group_index,
                        original_slot_index,
                        compatibility_name,
                        rebinding.shard,
                    )
                )
                if target is None:
                    continue
                staged_params.append(target)
                staged_names.append(compatibility_name)
                staged._param_names[target] = compatibility_name
                staged.state[target]["name"] = compatibility_name
                staged._gefen_shard_bindings[target] = rebinding.shard
            staged_group["params"] = staged_params
            staged_group["param_names"] = staged_names
            staged.param_groups.append(staged_group)

        staged._gefen_local_shard_bindings = tuple(
            sorted(local_bindings, key=lambda item: item[1].sort_key)
        )
        staged._gefen_logical_slots = tuple(logical_slots)
        staged._gefen_sharding_manifest = manifest
        staged._gefen_post_sharding_finalized = True
        staged._gefen_codebook_process_group = codebook_process_group
        staged._gefen_codebook_scope_validated = False
        staged._gefen_codebook_by_device = {}
        staged._gefen_codebook_lut_by_device = {}
        staged._sr_seed_by_device = {}
        staged._gefen_global_step_by_device = {}
        staged._capt_stacks = None
        staged._static_mark_sig = None
        staged._lr_scalar_cache = None
        staged._ensure_gefen_global_step_devices()
        staged._install_rank_local_checkpoint_schema()
        staged._gefen_finalized_slots = tuple(
            tuple(group["params"]) for group in staged.param_groups
        )
        return staged

    def post_sharding(
        self,
        rebindings,
        *,
        manifest: ShardingManifest,
        codebook_process_group=None,
    ) -> None:
        """Atomically finalize every local optimizer slot after sharding."""

        if not isinstance(manifest, ShardingManifest):
            raise TypeError("manifest must be a ShardingManifest")
        if isinstance(rebindings, (str, bytes)):
            raise TypeError("rebindings must be a sequence")
        rebindings = tuple(rebindings)
        if not rebindings or any(
            not isinstance(item, ParameterRebinding) for item in rebindings
        ):
            raise TypeError(
                "rebindings must contain one or more ParameterRebinding values"
            )
        sources = []
        for rebinding in rebindings:
            if self._parameter_in(sources, rebinding.old_parameter):
                raise ValueError("rebinding source tensors must be unique")
            sources.append(rebinding.old_parameter)
        staged = self._stage_post_sharding(rebindings, manifest, codebook_process_group)
        self.__dict__.update(staged.__dict__)
        self._invalidate_layout_forensics_caches()
        # Compute the manifest shard set and digest exactly once at
        # finalization; every later step guard and scoped operation header
        # reuses this cache instead of rehashing every shard identity.
        self._manifest_layout_forensics(refresh=True)

    def rebind_shard(
        self,
        old_parameter,
        new_parameter,
        *,
        shard: ShardIdentity,
        manifest: ShardingManifest,
    ) -> None:
        """Apply a complete one-slot shard plan through ``post_sharding``."""

        self.post_sharding(
            (ParameterRebinding(old_parameter, new_parameter, shard),),
            manifest=manifest,
        )

    def rebind_parameter(
        self,
        old_parameter,
        new_parameter,
        *,
        identity: ParameterIdentity,
    ) -> None:
        """Bind one complete replicated parameter with a canonical identity."""

        shard = ShardIdentity(
            identity,
            ParameterLayout.REPLICATED,
            LogicalSlice.full(identity),
        )
        self.rebind_shard(
            old_parameter,
            new_parameter,
            shard=shard,
            manifest=ShardingManifest((shard,)),
        )

    @staticmethod
    def _normalize_param_groups(params):
        if isinstance(params, torch.Tensor):
            param_groups = [params]
        else:
            param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if isinstance(param_groups[0], dict):
            return param_groups
        return [{"params": param_groups, "_gefen_implicit_group": True}]

    @staticmethod
    def _validate_group_options(lr, betas, eps, weight_decay):
        if torch.is_tensor(lr) and lr.numel() != 1:
            raise ValueError(
                "lr as a Tensor must be 1-element, got numel={}".format(lr.numel())
            )
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        elif not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        elif not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        elif not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        elif not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(
                "Invalid epsilon value: {}; eps must be finite and > 0".format(eps)
            )

    @staticmethod
    def _iter_params_with_names(group_params):
        if isinstance(group_params, torch.Tensor):
            group_params = (group_params,)
        for param_or_named_param in group_params:
            if isinstance(param_or_named_param, tuple):
                param_name, param = param_or_named_param
            else:
                param_name = None
                param = param_or_named_param
            if not isinstance(param, torch.Tensor):
                raise TypeError(
                    "Gefen can only optimize Tensors, but one of the params is {}".format(
                        type(param).__name__,
                    )
                )
            if torch.is_complex(param):
                raise ValueError(
                    "Gefen does not support complex parameters, but got a parameter with dtype {}".format(
                        param.dtype
                    )
                )
            yield param_name, param

    def _param_name(self, param) -> str:
        state = self.state.get(param)
        if state is not None and "name" in state:
            return state["name"]
        return self._param_names.get(param, "none")

    def _param_name_is_synthesized(self, param) -> bool:
        """True when Gefen generated this parameter's name positionally.

        Recorded at registration (``add_param_group``), where the distinction is
        known for certain, rather than inferred from spelling: the synthesized
        forms are ordinary identifiers a caller may legitimately use. A native load
        re-establishes the flag from the checkpoint's own provenance, so a rename
        cannot launder a positional name into a caller-provided one. The spelling
        fallback covers only a parameter with no registration record at all.
        """
        if param in self._synthesized_param_names:
            return self._synthesized_param_names[param]
        return bool(_SYNTHESIZED_PARAM_NAME_RE.match(str(self._param_name(param))))

    def _iter_group_params_with_names(self, group):
        names = group.get("param_names")
        for idx, param in enumerate(group["params"]):
            if names is not None and idx < len(names):
                yield names[idx], param
            else:
                yield self._param_name(param), param

    def _iter_codebook_params_with_names(self):
        """Yield codebook inputs in canonical order when a scope is bound."""

        if self._gefen_codebook_process_group is None:
            for group in self.param_groups:
                for name, parameter in self._iter_group_params_with_names(group):
                    yield group, name, parameter
            return
        by_identity = {}
        for group in self.param_groups:
            for name, parameter in self._iter_group_params_with_names(group):
                by_identity[id(parameter)] = (group, name, parameter)
        for parameter, _ in self._gefen_local_shard_bindings:
            if parameter is not None:
                yield by_identity[id(parameter)]

    def _has_local_codebook_gradients(self) -> bool:
        return any(
            parameter.grad is not None
            for _, _, parameter in self._iter_codebook_params_with_names()
        )

    @staticmethod
    def _unique_name(base: str, existing_names) -> str:
        name = str(base).lower()
        if name not in existing_names:
            return name
        suffix = 1
        while "{}_{}".format(name, suffix) in existing_names:
            suffix += 1
        return "{}_{}".format(name, suffix)

    def _sync_param_names_to_state(self) -> None:
        """Re-derive per-parameter names and their provenance after a native load.

        Provenance decides whether GefenDCPState will reshard against the name, so
        a load that CHANGES a name must re-establish it rather than inherit the
        live optimizer's stale flag: keep the registration-time flag for an
        unchanged name, take the checkpoint's for a changed one, and fail closed
        (mark it synthesized) when a legacy checkpoint carried none.

        That last case must NOT fall back to spelling. Two unnamed groups both
        named "weight" produce the positional pair "weight"/"weight_1", which no
        regex separates from a caller's legitimate "layer_1" -- so a spelling
        check accepts it and lets DCP reshard against a registration-order
        identity, cross-assigning momentum. Refusing a renamed legacy checkpoint
        until it is re-saved is recoverable; cross-assigned momentum is not.
        """
        previous_names = self._param_names
        previous_flags = self._synthesized_param_names
        self._param_names = {}
        logical_groups = None
        if self._gefen_post_sharding_finalized:
            logical_groups = [[] for _ in self.param_groups]
            for logical_slot in self._gefen_logical_slots:
                if (
                    logical_slot.shard.layout is ParameterLayout.WHOLE_PARAMETER_OWNER
                    and logical_slot.shard.local_member != logical_slot.shard.owner
                ):
                    continue
                logical_groups[logical_slot.group_index].append(
                    logical_slot.compatibility_name
                )
        synthesized_flags = {}
        for group_index, group in enumerate(self.param_groups):
            params = group["params"]
            if logical_groups is None:
                names = list(group.get("param_names", ()))
                if len(names) != len(params):
                    names = [self._param_name(p) for p in params]
            else:
                names = logical_groups[group_index]
                if len(names) != len(params):
                    raise RuntimeError(
                        "Gefen finalized logical-slot registry does not match the loaded layout"
                    )
            # A short/absent/malformed list is unusable rather than partially
            # trusted: provenance is positional against param_names, so a length
            # mismatch means we cannot say which flag describes which name.
            checkpoint_flags = group.get(_SYNTHESIZED_PARAM_NAMES_KEY)
            if not isinstance(checkpoint_flags, (list, tuple)) or len(
                checkpoint_flags
            ) != len(params):
                checkpoint_flags = None
            normalized_names = []
            for index, (param, name) in enumerate(zip(params, names)):
                name = str(name).lower()
                normalized_names.append(name)
                self._param_names[param] = name
                if param in previous_flags and previous_names.get(param) == name:
                    synthesized_flags[param] = previous_flags[param]
                elif checkpoint_flags is not None:
                    synthesized_flags[param] = bool(checkpoint_flags[index])
                else:
                    synthesized_flags[param] = True
                self.state[param]["name"] = name
            group["param_names"] = normalized_names
            group[_SYNTHESIZED_PARAM_NAMES_KEY] = [
                synthesized_flags[param] for param in params
            ]
        self._synthesized_param_names = synthesized_flags

    def add_param_group(self, param_group):
        """Add a param group while preserving the caller-visible group boundary.

        Accepts the same group dict shape as the constructor (``params`` as
        tensors or ``(name, tensor)`` tuples; optional ``lr`` / ``betas`` (or
        ``beta1``/``beta2``) / ``eps`` / ``weight_decay``, defaulting to the
        constructor values). The public group is preserved; each parameter's
        stable lowercase name is stored in its per-param state and in the
        group's ``param_names`` list for introspection.
        """
        if getattr(self, "_gefen_post_sharding_finalized", False):
            raise RuntimeError(
                "Gefen cannot add parameter groups after post_sharding finalization"
            )
        if not isinstance(param_group, dict):
            raise TypeError(
                "param_group must be a dict, got {}".format(type(param_group).__name__)
            )
        if "params" not in param_group:
            raise ValueError("Gefen parameter group is missing the 'params' key.")
        defaults = self.defaults
        group_lr = param_group.get("lr", defaults["lr"])
        if "betas" in param_group:
            group_betas = param_group["betas"]
        else:
            group_betas = (
                param_group.get("beta1", defaults["beta1"]),
                param_group.get("beta2", defaults["beta2"]),
            )
        group_eps = param_group.get("eps", defaults["eps"])
        group_weight_decay = param_group.get("weight_decay", defaults["weight_decay"])
        self._validate_group_options(
            group_lr, group_betas, group_eps, group_weight_decay
        )
        named_params = list(self._iter_params_with_names(param_group["params"]))
        if len(named_params) == 0:
            raise ValueError("optimizer got an empty parameter list")
        # Keys beyond the ones Gefen manages (e.g. a subclass's extras) are
        # carried onto the preserved group, mirroring torch's behavior of
        # preserving unknown group keys.
        consumed = (
            "params",
            "name",
            "lr",
            "betas",
            "beta1",
            "beta2",
            "eps",
            "weight_decay",
            "param_names",
            _SYNTHESIZED_PARAM_NAMES_KEY,
            "_gefen_implicit_group",
        )
        extra = {k: v for k, v in param_group.items() if k not in consumed}
        existing_names = set(self._param_names.values())
        group_index = len(self.param_groups)
        # Build (and fully validate) the preserved group up front so a failure
        # partway through -- a complex/non-tensor param from
        # _iter_params_with_names, or a duplicate param -- leaves
        # self.param_groups untouched. add_param_group is all-or-nothing: only
        # once the whole group is materialized do we register any of it.
        existing_params = set()
        for g in self.param_groups:
            existing_params.update(g["params"])
        batch_params = set()
        group_name = param_group.get("name")
        implicit_group = bool(param_group.get("_gefen_implicit_group", False))
        params = []
        param_names = []
        # Parallel to param_names: True where Gefen generated the name below
        # because the caller supplied none. Recorded here, the only place the
        # distinction is knowable.
        synthesized_flags = []
        for param_index, (param_name, param) in enumerate(named_params):
            # Reject duplicates before registering any group, so the base class's
            # own (post-registration) duplicate check can't leave us partial.
            if param in existing_params or param in batch_params:
                raise ValueError(
                    "some parameters appear in more than one parameter group"
                )
            batch_params.add(param)
            params.append(param)
            synthesized = False
            if param_name is None:
                if group_name is not None and len(named_params) == 1:
                    # The caller named the group, and that name identifies the
                    # single parameter it holds: caller-provided, not positional.
                    base_name = str(group_name).lower()
                    param_name = self._unique_name(base_name, existing_names)
                    if param_name != base_name:
                        # ...unless a collision forced a suffix, which
                        # registration order alone decides -- build the same groups
                        # in a different order and the names swap. Marking it
                        # synthesized keeps DCP from resharding against an
                        # identity it cannot rely on.
                        synthesized = True
                elif implicit_group:
                    param_name = self._unique_name(
                        "param_{}".format(len(self._param_names) + param_index),
                        existing_names,
                    )
                    synthesized = True
                else:
                    param_name = self._unique_name(
                        "group_{}_param_{}".format(group_index, param_index),
                        existing_names,
                    )
                    synthesized = True
            else:
                param_name = str(param_name).lower()
            param_names.append(param_name)
            synthesized_flags.append(synthesized)
            existing_names.add(param_name)

        new_group = dict(extra)
        if group_name is not None:
            new_group["name"] = str(group_name).lower()
        elif len(param_names) == 1:
            new_group["name"] = param_names[0]
        new_group.update(
            {
                "params": params,
                "param_names": param_names,
                # Mirrored into the public group (like param_names) so
                # Optimizer.state_dict carries provenance to the checkpoint.
                _SYNTHESIZED_PARAM_NAMES_KEY: synthesized_flags,
                "lr": group_lr,
                "beta1": group_betas[0],
                "beta2": group_betas[1],
                "eps": group_eps,
                "weight_decay": group_weight_decay,
            }
        )
        super().add_param_group(new_group)
        for param, param_name, synthesized in zip(
            params, param_names, synthesized_flags
        ):
            self._param_names[param] = param_name
            self._synthesized_param_names[param] = synthesized
            self.state[param]["name"] = param_name
        self._ensure_gefen_global_step_devices()
        if getattr(self, "_gefen_checkpoint_schema_ready", False):
            self._install_rank_local_checkpoint_schema()

    def _lr_scalar(self, group) -> float:
        """Resolve ``group["lr"]`` to a python float, caching a tensor lr's ``.item()``.

        A float lr is returned as-is (no sync). A tensor lr is ``.item()``'d (a
        D2H sync) only when its identity or ``_version`` changes -- an in-place
        scheduler update (e.g. ``lr.mul_``) bumps ``_version``, a replacement
        swaps identity. The cache is a SINGLE slot (the last ``(tensor, version,
        value)`` triple), shared across all param groups. The tensor-lr idiom is
        a single on-device lr, so one shared slot already costs just one
        ``.item()`` per step (O(steps), per version). One slot
        rather than an id-keyed dict so a loop that assigns a FRESH lr tensor
        each step can't grow the cache (and its strong refs) unboundedly. The
        cached value equals ``lr.item()``, so callers stay bit-identical.
        Inherited by GefenMuon. Kept as a plain instance attribute so it is not
        serialized into ``param_groups`` / the optimizer ``state_dict``.
        """
        lr = group["lr"]
        if not isinstance(lr, torch.Tensor):
            return lr
        ver = lr._version
        cache = getattr(self, "_lr_scalar_cache", None)
        if cache is not None and cache[0] is lr and cache[1] == ver:
            return cache[2]
        val = lr.item()
        self._lr_scalar_cache = (lr, ver, val)
        return val

    def _use_fused_automatic_vmean(self) -> bool:
        return self.fused and FUSE_AUTOMATIC_VMEAN_UPDATE

    def _fused_configured(self) -> bool:
        # The CONFIGURED fused intent: fused=True AND the module fuse flag, with
        # NO toolchain probe (never triggers the JIT build). Used at construction
        # and anywhere a build side effect would be wrong.
        return self.fused and FUSE_GEFEN_AUTOMATIC_STEP

    def _gefen_fused_toolchain_ok(self) -> bool:
        # Lazily probe whether the fused CUDA extension can build/load on this
        # box. On failure (missing nvcc / toolchain mismatch / unwritable build
        # dir) warn ONCE, permanently downgrade this instance to fused=False, and
        # cache the verdict so the probe and the warning happen at most once. A
        # working pure-torch training run continues instead of a mid-step crash.
        if not self.fused:
            return False
        if self._fused_build_ok is not None:
            return self._fused_build_ok
        try:
            _ensure_gefen_fused_extension_loaded()
            self._fused_build_ok = True
        except Exception as exc:  # noqa: BLE001 -- any build/load failure must
            # degrade to the pure-torch path, never crash the training run.
            self._fused_build_ok = False
            self.fused = False
            warnings.warn(
                "Gefen could not build/load its fused CUDA kernels ({}: {}); "
                "continuing with fused=False on the pure-PyTorch path. Set "
                "GEFEN_VERBOSE_BUILD=1 for the full build diagnostic.".format(
                    type(exc).__name__, exc
                ),
                UserWarning,
                stacklevel=2,
            )
        return self._fused_build_ok

    def _fused_kernels_available(self) -> bool:
        # Whether the fused automatic-step CUDA kernels can run at all,
        # REGARDLESS of capturable. GefenMuon uses this for its momentum kernel,
        # whose only scalar argument (the constant momentum beta) is
        # capture-safe to bake into a graph. Lazily probes the toolchain and
        # downgrades to pure-torch on a build failure (see
        # _gefen_fused_toolchain_ok).
        if not self._fused_configured():
            return False
        return self._gefen_fused_toolchain_ok()

    def _use_fused_gefen_automatic_step(self) -> bool:
        # The fused update kernels stay on under capturable=True: the per-step-
        # varying scalars (lr / bias-correction reciprocals / weight-decay
        # factor) are then delivered through a small per-param fp32 DEVICE
        # buffer (state["_capt_scalars"], refreshed in place with tensor ops
        # each step) instead of host kernel arguments, so a captured CUDA graph
        # replays fresh values rather than freezing the capture-time ones.
        return self._fused_kernels_available()

    def _assert_capturable_if_capturing(self) -> None:
        # Mirror torch.optim's capturable contract: capturing step() in a CUDA
        # graph without capturable=True would silently freeze the host-side
        # step counters / bias corrections at their capture-time values, so
        # fail loudly instead.
        capturing = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        if capturing and not self.capturable:
            raise RuntimeError(
                "Attempting CUDA graph capture of {}.step() but capturable=False. "
                "Construct the optimizer with capturable=True to make step() "
                "graph-safe.".format(type(self).__name__)
            )
        if capturing:
            devices = {
                param.device for group in self.param_groups for param in group["params"]
            }
            capture_device = torch.device("cuda", torch.cuda.current_device())
            if devices != {capture_device}:
                raise RuntimeError(
                    "CUDA graph capture requires every Gefen parameter on the "
                    "current capture device {}; found {}".format(
                        capture_device, sorted(map(str, devices))
                    )
                )

    def _static_mark_signature(self):
        # Cheap identity fingerprint of every object the static-marking pass
        # would touch (C-level sum/map/chain loops: ~155us per steady-state
        # call on a ~250-param model, vs ~380us for the marking pass it
        # replaces -- measured on the 386M bench census). Every event that
        # creates or replaces a marked tensor changes it: lazy state init /
        # scratch buffers (entry counts + id sums), a batched-refresh stack
        # rebuild (new backing buffers AND a new stacks dict), a codebook
        # refresh / per-device copy (all assignment sites replace the tensor,
        # never mutate in place, and clear the by-device dicts), a checkpoint
        # load (wholesale-new state tensors + cleared LUT dict), and grads
        # rebound to fresh tensors by the training loop. Marking itself only
        # sets idempotent per-tensor attributes, so skipping it while the
        # fingerprint is unchanged is behavior-preserving.
        acc = 0
        n_params = 0
        for group in self.param_groups:
            params = group["params"]
            n_params += len(params)
            acc += sum(map(id, params))
            acc ^= sum(id(p.grad) for p in params)
        state = self.state
        acc += sum(map(id, chain.from_iterable(map(dict.values, state.values()))))
        stacks = self._capt_stacks
        return (
            acc,
            n_params,
            len(state),
            sum(map(len, state.values())),
            id(self._gefen_codebook),
            tuple(map(id, self._gefen_codebook_by_device.values())),
            tuple(map(id, self._gefen_codebook_lut_by_device.values())),
            # Capturable SR seeds: created lazily (first SR kernel call on a
            # device), so their appearance must re-run the marking pass.
            tuple(map(id, self._sr_seed_by_device.values())),
            id(stacks),
        )

    @torch._dynamo.disable
    def _mark_state_static_for_compile(self) -> None:
        # torch.compile(mode="reduce-overhead") wraps the step in CUDA graphs
        # (cudagraph trees). The params and every persistent state tensor are
        # mutated in place each step; unless their ADDRESSES are marked static
        # -- exactly what torch.optim does for its compiled optimizers -- the
        # runtime treats them as rotating graph-pool outputs and the second
        # compiled call fails with "accessing tensor output of CUDAGraphs that
        # has been overwritten by a subsequent run". Called at the end of every
        # EAGER capturable step (marking is idempotent), so the supported
        # pattern -- run the first step(s) eagerly (codebook learning is
        # host-driven anyway), then compile -- sees every state tensor marked,
        # including lazily created scratch buffers. In steady state the
        # identity-fingerprint guard below skips the O(#tensors) re-walk;
        # any tensor creation/replacement changes the fingerprint and re-runs
        # the full pass. @torch._dynamo.disable:
        # mark_static_address is a dynamo API (a "forbidden callable" inside a
        # traced region), so this helper must stay out of any compiled frame.
        mark = getattr(torch._dynamo, "mark_static_address", None)
        if mark is None:  # very old torch: compile support is best-effort
            return
        if self._static_mark_signature() == self._static_mark_sig:
            return
        # The codebook (and its search LUT) feed every fused-kernel custom
        # op. Prebuild the per-device LUTs HERE, outside any compiled frame or
        # cudagraph recording, and mark both static: the compiled step then
        # passes the LUT to the ops as an explicit argument
        # (_gefen_codebook_lut_on), so the op bodies never allocate. (Building
        # the LUT lazily inside an op is not capture-safe: whenever cudagraph
        # trees decides to copy the codebook input into the graph pool, the
        # in-op data_ptr cache misses and the rebuilt LUT -- kept alive by the
        # cache -- trips recording's "allocations not tracked as outputs"
        # check.)
        if self._gefen_codebook is not None:
            mark(self._gefen_codebook)
            lut = _codebook_search_lut(self._gefen_codebook)
            mark(lut)
            self._gefen_codebook_lut_by_device[self._gefen_codebook.device] = lut
        for device, cached in self._gefen_codebook_by_device.items():
            mark(cached)
            lut = _codebook_search_lut(cached)
            mark(lut)
            self._gefen_codebook_lut_by_device[device] = lut
        # Capturable SR seeds: read by the fused custom ops every step and
        # mutated in place (device add_) at the step tail, so their addresses
        # must be static for cudagraph trees like every other state tensor.
        for seed in self._sr_seed_by_device.values():
            mark(seed)
        for group in self.param_groups:
            for p in group["params"]:
                mark(p)
                if p.grad is not None:
                    mark(p.grad)
                for value in self.state.get(p, {}).values():
                    if torch.is_tensor(value):
                        mark(value)
        # The batched-refresh stacked buffers are mutated in place every step
        # (the per-param state views above alias them); mark the backing
        # tensors static too so cudagraph trees treats their addresses as
        # stable across compiled steps.
        if self._capt_stacks:
            for device_stacks in self._capt_stacks.values():
                for stack in device_stacks:
                    for key in ("steps2d", "consts2d", "scal2d", "sqrt_mask"):
                        value = stack.get(key)
                        if torch.is_tensor(value):
                            mark(value)
        # Recompute AFTER the pass: the LUT prebuild above may have populated
        # _gefen_codebook_lut_by_device, which the fingerprint includes.
        self._static_mark_sig = self._static_mark_signature()

    def _kernel_rng_seed(self) -> int:
        # HOST seed for the kernels' stochastic-rounding hash. Under
        # capturable the rounding seed flows through the per-device 0-dim
        # int64 tensor instead (_sr_seed_on; the kernels ignore this host arg
        # whenever that tensor is passed), so return a CONSTANT rather than
        # the per-step host counter. This matters for torch.compile(step):
        # dynamo specializes on python int arguments, so a seed that changes
        # every step recompiles the kernel-call frames each step until the
        # recompile limit, then runs them through the guard-miss slow path
        # forever (measured: the whole compiled-slower-than-eager anomaly).
        # capturable=False keeps the exact historical per-step host seed
        # (bit-identical).
        return 0 if self.capturable else self._gefen_global_step

    def _sr_seed_on(self, device: torch.device):
        # Device-side stochastic-rounding seed (capturable only): the per-step
        # seed the SR-capable fused kernels read from device memory, so
        # captured CUDA graphs replay fresh seeds. One 0-dim int64 tensor per
        # device, initialized from the CURRENT _gefen_global_step (so
        # eager-then-capture transitions keep advancing monotonically -- the
        # value during step k is exactly k, the same seed capturable=False
        # passes as the host rng_seed) and advanced on device once per step by
        # _advance_sr_seeds. Returns None on every non-capturable or non-SR
        # configuration, which keeps the kernels on the bit-identical legacy
        # host-seed path.
        if not (self.capturable and self._stochastic_round):
            return None
        seed = self._sr_seed_by_device.get(device)
        if seed is None:
            self._ensure_gefen_global_step_devices()
            seed = self._sr_seed_by_device.get(device)
        if seed is None:
            global_counter = self._gefen_global_step_by_device.get(device)
            initial_step = (
                self._gefen_global_step
                if global_counter is None
                else int(global_counter.item())
            )
            seed = torch.full(
                (),
                initial_step,
                dtype=torch.int64,
                device=device,
            )
            self._sr_seed_by_device[device] = seed
        return seed

    def _gefen_cuda_param_devices(self):
        return sorted(
            {
                param.device
                for group in self.param_groups
                for param in group["params"]
                if param.device.type == "cuda"
            },
            key=lambda device: -1 if device.index is None else device.index,
        )

    def _device_gefen_global_step(self):
        if not self._gefen_global_step_by_device:
            return None
        steps = [
            int(counter.item())
            for counter in self._gefen_global_step_by_device.values()
        ]
        if any(step != steps[0] for step in steps[1:]):
            raise RuntimeError(
                "Gefen capturable global-step counters disagree across devices: {}".format(
                    steps
                )
            )
        return steps[0]

    def _ensure_gefen_global_step_devices(self) -> None:
        """Create capturable counters before capture, including no-grad devices."""
        if not self.capturable:
            return
        devices = self._gefen_cuda_param_devices()
        missing = [
            device
            for device in devices
            if device not in self._gefen_global_step_by_device
        ]
        if missing:
            device_step = self._device_gefen_global_step()
            initial_step = (
                self._gefen_global_step if device_step is None else device_step
            )
            self._gefen_global_step = initial_step
            for device in missing:
                self._gefen_global_step_by_device[device] = torch.full(
                    (), initial_step, dtype=torch.int64, device=device
                )
        if self._stochastic_round:
            for device in devices:
                if device not in self._sr_seed_by_device:
                    self._sr_seed_by_device[device] = (
                        self._gefen_global_step_by_device[device].detach().clone()
                    )

    def _reset_gefen_global_step_devices(self) -> None:
        self._gefen_global_step_by_device.clear()
        self._ensure_gefen_global_step_devices()

    def _synchronize_gefen_global_step_for_checkpoint(self) -> None:
        """Refresh the serialized host mirror after manual graph replays."""
        self._ensure_gefen_global_step_devices()
        device_step = self._device_gefen_global_step()
        if device_step is not None:
            self._gefen_global_step = device_step
        if self.capturable and self._stochastic_round:
            seed_steps = [int(seed.item()) for seed in self._sr_seed_by_device.values()]
            if any(seed_step != self._gefen_global_step for seed_step in seed_steps):
                raise RuntimeError(
                    "Gefen capturable stochastic-round seeds disagree with the optimizer global step: {} != {}".format(
                        seed_steps, self._gefen_global_step
                    )
                )

    @torch._dynamo.disable
    def _advance_sr_seeds(self) -> None:
        # Advance every per-device rounding seed by one, ON DEVICE, at the
        # tail of step() -- the device mirror of _advance_gefen_global_step.
        # Under manual CUDA-graph capture the add_ is recorded on-stream, so
        # every replay dithers with a fresh seed; under torch.compile the
        # @torch._dynamo.disable keeps it out of the traced graph (the graph
        # only READS the static-marked seed tensors) and it runs eagerly right
        # after the compiled/recorded step, exactly like the batched scalar
        # refresh runs right before it. Eager capturable behavior is the same
        # +1 per step.
        for seed in self._sr_seed_by_device.values():
            seed.add_(1)

    def _new_step_counter(self, device: torch.device):
        # Step counters start at 0 as python ints (host math, bit-identical to
        # the historical behavior). Under capturable they are 0-dim float32
        # device tensors -- like torch.optim's capturable step -- so the
        # ``beta ** step`` bias corrections are computed on device and stay
        # correct across CUDA-graph replays.
        if self.capturable:
            return torch.zeros((), dtype=torch.float32, device=device)
        return 0

    def _colocate_param_state_(self, state, device: torch.device) -> None:
        # CPU-offload / device-mobility support. The per-param state tensors
        # (quantized momentum, block/factored second moments, tensor step
        # counters, reused scratch) are allocated once on the first step's
        # device and then reused in place. When a framework relocates a
        # parameter between iterations -- e.g. CPU offload paging a shard
        # CPU->CUDA->CPU -- the state is left stranded on the previous device
        # and the next non-fused update trips the codebook/state device-equality
        # guards. Migrate the stranded tensors onto the current compute device
        # (grad_view.device) so the whole update stays co-located.
        #
        # Cheap gate: one device compare on a sentinel (m_codebook, always
        # created by _init_gefen_state on both the block and factored paths). On
        # the common same-device step -- every CUDA-resident run -- this returns
        # immediately with no allocation and no .to(), so the update is
        # byte-for-byte unchanged. Capturable stepping pins static state
        # addresses for CUDA-graph replay, so reassigning a state tensor would
        # break capture; capture is CUDA-graph only (not a CPU-offload path), so
        # this is an unconditional no-op there.
        if self.capturable:
            return
        sentinel = state.get("m_codebook")
        if sentinel is None or sentinel.device == device:
            return
        for key in _COLOCATABLE_STATE_KEYS:
            value = state.get(key)
            if torch.is_tensor(value) and value.device != device:
                # .to() on the uint8 index / fp32 magnitude / fp32 second-moment
                # tensors is a pure device copy (no dtype change), so a
                # round-trip preserves momentum indices and magnitudes exactly.
                state[key] = value.to(device)

    def _refresh_capturable_step_scalars(
        self, state, group, bc2_step_key: str, sqrt_bc2: bool
    ) -> torch.Tensor:
        # Capturable fused stepping: refresh the persistent per-param fp32
        # device buffer the fused kernels read their PER-STEP-VARYING scalars
        # from. Layout: [lr, 1/sqrt(bc2) (or 1/bc2 when sqrt_bc2=False, the
        # factored path), 1/bc1, weight_decay_factor]. Everything is computed
        # with tensor ops IN PLACE on the persistent buffer -- no new
        # persistent allocations inside a captured step (the 0-dim/2-elem
        # intermediates come from the capture pool) -- so a captured CUDA
        # graph replays fresh values every step.
        #
        # Numerics: the bias corrections are computed in float64 on device
        # (pow -> 1-x -> sqrt -> reciprocal, the exact op chain the host path
        # runs in python doubles) and only then cast to fp32 by the buffer
        # copy_, mirroring the host path's double->float cast at kernel launch.
        #
        # lr semantics match the rest of capturable: a TENSOR lr flows on
        # device (update it in place between replays to drive a schedule) and
        # the weight-decay factor is derived from it on device; a FLOAT lr is
        # written with a host fill_, whose value a captured graph bakes --
        # the same baking semantics float lr always had under capture.
        if "_capt_row" in state:
            # Batched capturable stepping: this param's [lr, 1/bcX, 1/bc1,
            # wd_factor] row was already refreshed (elementwise-identically)
            # by the stacked prologue chain this step -- hand the kernels its
            # view directly instead of re-running the per-param chain.
            return state["_capt_scalars"]
        buf = state.get("_capt_scalars")
        consts_key = (group["beta2"], group["beta1"])
        if buf is None or "_capt_consts" not in state:
            # Also rebuilds after a stack teardown MID-CAPTURE (validation
            # failure inside a torch.cuda.graph): the consts must be populated
            # with host-scalar fill_s -- torch.tensor(..., device=cuda) issues
            # a pageable H2D copy, which is illegal while capturing (the
            # allocations themselves draw from the capture pool and are fine).
            device = state["step"].device
            buf = torch.ones(4, dtype=torch.float32, device=device)
            state["_capt_scalars"] = buf
            # [beta2, beta1] in float64, ordered to line up with the
            # [bc2_step, step] stack below.
            consts = torch.empty(2, dtype=torch.float64, device=device)
            consts[0].fill_(consts_key[0])
            consts[1].fill_(consts_key[1])
            state["_capt_consts"] = consts
            state["_capt_consts_key"] = consts_key
        elif state.get("_capt_consts_key") != consts_key:
            # Mid-run group beta changes must flow into the cached consts so
            # the per-param path honors live betas exactly like the batched
            # and non-capturable paths do (one tuple compare per step;
            # capture-safe host-scalar fills on mismatch only).
            consts = state["_capt_consts"]
            consts[0].fill_(consts_key[0])
            consts[1].fill_(consts_key[1])
            state["_capt_consts_key"] = consts_key
        # bc = [1 - beta2**bc2_step, 1 - beta1**step] in float64.
        bc = 1.0 - torch.pow(
            state["_capt_consts"],
            torch.stack((state[bc2_step_key], state["step"])),
        )
        if sqrt_bc2:
            bc[0:1].sqrt_()
        # buf[1:3] = [1/(sqrt-or-plain bc2), 1/bc1], cast float64 -> fp32.
        buf[1:3].copy_(bc.reciprocal_())
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        if torch.is_tensor(lr):
            buf[0].copy_(lr)
            if weight_decay:
                buf[3].copy_(1.0 - lr.double() * weight_decay)
            else:
                buf[3].fill_(1.0)
        else:
            buf[0].fill_(float(lr))
            buf[3].fill_(1.0 - float(lr) * weight_decay)
        return buf

    # ------------------------------------------------------------------
    # Batched capturable step-scalar refresh
    #
    # Under capturable=True every param keeps 0-dim device step counters and a
    # 4-elem fp32 device buffer the fused kernels read their per-step scalars
    # from, refreshed with a small tensor-op chain EVERY step. Per param that
    # is ~8 microscopic launches for the refresh plus 2 counter increments; on
    # a ~250-param model those ~2500 tiny kernels dominate the capturable
    # step's GPU time (they are what made captured/compiled stepping slower
    # than the plain eager step). The machinery below instead keeps each
    # registered param's counters and scalar buffer as VIEWS into per-cohort
    # stacked buffers (steps2d [2, N] fp32, scal2d [N, 4] fp32) and advances /
    # refreshes ALL rows with one short vectorized chain per step (~10
    # launches per distinct lr/weight-decay cohort, independent of N). The
    # chain is elementwise identical to _refresh_capturable_step_scalars (same
    # float64 pow -> 1-x -> sqrt -> reciprocal ops, fp32 cast only at the
    # buffer write), so eager,
    # manually-captured and compiled capturable stepping keep their exact
    # numerics -- the anti-freeze tests assert the buffer values exactly.
    #
    # State-dict semantics are preserved: counter views serialize through
    # state_dict's _compact pass (non-owning views are cloned into independent
    # tensors), "_capt_scalars"/"_capt_stack"/"_capt_row" are scratch and
    # stripped from checkpoints, and load_state_dict tears the stacks down
    # (counters come back as independent 0-dim tensors); the stacks lazily
    # rebuild on the next eager capturable step. Registration is validated
    # against the live stepping set every step and torn down on any mismatch,
    # so grad-set changes, param device moves and state surgery all fall back
    # to the bit-identical per-param path instead of desyncing.
    # ------------------------------------------------------------------

    def _capt_row_info(self, p: torch.Tensor):
        # Registration/validation predicate for the batched capturable
        # refresh: (bc2_step_key, sqrt_bc2) when this param steps through a
        # path whose counters/scalars the batch can own this step, else None.
        # Mirrors step()'s routing exactly: the factored branch corrects bc2
        # with factored_step (plain 1/bc2), everything else with vmean_step
        # (1/sqrt(bc2)).
        grad = p.grad
        if grad is None:
            return None
        # DTensor/FSDP2 grads and params keep the per-param path (local-shard
        # unwrap, waits and empty-shard early-outs live in _step_automatic*).
        if hasattr(grad, "to_local") or hasattr(grad, "wait"):
            return None
        if hasattr(p, "placements"):
            return None
        if not p.is_cuda or grad.numel() == 0:
            return None
        state = self.state.get(p)
        if not state or "step" not in state:
            return None  # lazy state init runs this step; register next step
        if self._factored_v_2d and p.ndim == 2:
            info = ("factored_step", False)
        else:
            info = ("vmean_step", True)
        bc2 = state.get(info[0])
        for t in (state["step"], bc2):
            if (
                not torch.is_tensor(t)
                or t.dim() != 0
                or t.dtype != torch.float32
                or t.device != p.device
            ):
                return None
        return info

    def _capt_invalidate(self) -> None:
        # Host-only and idempotent: forget the stacked buffers and unmark
        # every registered row so per-param increments/refresh resume.
        stacks = self._capt_stacks
        self._capt_stacks = None
        if not stacks:
            return
        for device_stacks in stacks.values():
            for stack in device_stacks:
                for p in stack["rows"]:
                    state = self.state.get(p)
                    if state is not None:
                        state.pop("_capt_stack", None)
                        state.pop("_capt_row", None)
                        # Drop the stack-row scalar view too: the per-param
                        # refresh must not write through into a dead stack,
                        # and (with the consts popped at build time) leaving
                        # it would make the refresh skip the consts rebuild
                        # and KeyError on state["_capt_consts"].
                        state.pop("_capt_scalars", None)

    @torch._dynamo.disable
    def _capt_build_stacks(self) -> bool:
        # (Re)build the per-device stacked buffers from the CURRENT stepping
        # set. Eager-only (allocations + state rebinding are neither capture-
        # nor trace-safe); callers skip the rebuild under capture/compile and
        # step per-param instead.
        self._capt_invalidate()
        # Python dict insertion order gives deterministic device/cohort/row
        # order from the public optimizer group order. Tensor lrs are keyed by
        # OBJECT identity (and retained strongly by the stack), because two
        # equal-valued tensors can be scheduled independently. Float lrs are
        # keyed by their current value when cohorts are first formed; a cohort
        # may subsequently move together to a new float without rebuilding,
        # while divergence is caught by _capt_batched_ready and repartitioned.
        per_dev = {}
        for group in self.param_groups:
            for p in group["params"]:
                info = self._capt_row_info(p)
                if info is None:
                    continue
                lr = group["lr"]
                if torch.is_tensor(lr):
                    lr_key = ("tensor", id(lr))
                else:
                    lr_key = ("float", float(lr))
                cohort_key = (lr_key, group["weight_decay"])
                cohorts = per_dev.setdefault(p.device, {})
                d = cohorts.get(cohort_key)
                if d is None:
                    d = {
                        "cohort_key": cohort_key,
                        "rows": [],
                        "groups": [],
                        "keys": [],
                        "sqrt": [],
                        "betas": [],
                        # Strong ref for tensor-lr identity/lifetime. For float
                        # cohorts this is only the build-time value; refresh
                        # reads the live value from row_groups[0].
                        "lr0": lr,
                        "weight_decay0": group["weight_decay"],
                    }
                    cohorts[cohort_key] = d
                d["rows"].append(p)
                d["groups"].append(group)
                d["keys"].append(info[0])
                d["sqrt"].append(info[1])
                d["betas"].append((group["beta1"], group["beta2"]))
        single_cohort_registry = bool(per_dev) and all(
            len(cohorts) == 1 for cohorts in per_dev.values()
        )
        stacks = {}
        for device, cohorts in per_dev.items():
            device_stacks = []
            for stack_index, d in enumerate(cohorts.values()):
                groups = d["groups"]
                rows = d["rows"]
                n = len(rows)
                steps2d = torch.empty((2, n), dtype=torch.float32, device=device)
                steps2d[0].copy_(
                    torch.stack(
                        [self.state[p][key].detach() for p, key in zip(rows, d["keys"])]
                    )
                )
                steps2d[1].copy_(
                    torch.stack([self.state[p]["step"].detach() for p in rows])
                )
                consts2d = torch.tensor(
                    [
                        [g["beta2"] for g in groups],
                        [g["beta1"] for g in groups],
                    ],
                    dtype=torch.float64,
                    device=device,
                )
                flags = d["sqrt"]
                if all(flags):
                    sqrt_mode, sqrt_mask = "all", None
                elif not any(flags):
                    sqrt_mode, sqrt_mask = "none", None
                else:
                    sqrt_mode = "mixed"
                    sqrt_mask = torch.tensor(flags, dtype=torch.bool, device=device)
                scal2d = torch.ones((n, 4), dtype=torch.float32, device=device)
                step_views = []
                bc2_views = []
                scalar_views = []
                for i, p in enumerate(rows):
                    state = self.state[p]
                    bc2_view = steps2d[0, i]
                    step_view = steps2d[1, i]
                    scalar_view = scal2d[i]
                    state[d["keys"][i]] = bc2_view
                    state["step"] = step_view
                    state["_capt_scalars"] = scalar_view
                    state.pop("_capt_consts", None)
                    state.pop("_capt_consts_key", None)
                    if single_cohort_registry:
                        # The fast validator addresses the sole stack directly.
                        # Avoiding 0 in every row also keeps the per-step static-
                        # address fingerprint at its old uniform-stack cost.
                        state.pop("_capt_stack", None)
                    else:
                        state["_capt_stack"] = stack_index
                    state["_capt_row"] = i
                    step_views.append(step_view)
                    bc2_views.append(bc2_view)
                    scalar_views.append(scalar_view)
                device_stacks.append(
                    {
                        "cohort_key": d["cohort_key"],
                        "rows": rows,
                        "row_groups": groups,
                        "keys": d["keys"],
                        "betas": d["betas"],
                        "weight_decay0": d["weight_decay0"],
                        "steps2d": steps2d,
                        "consts2d": consts2d,
                        "sqrt_mode": sqrt_mode,
                        "sqrt_mask": sqrt_mask,
                        "scal2d": scal2d,
                        "step_views": step_views,
                        "bc2_views": bc2_views,
                        "scalar_views": scalar_views,
                        "tensor_lr": torch.is_tensor(d["lr0"]),
                        "lr0": d["lr0"],
                    }
                )
            stacks[device] = tuple(device_stacks)
        # Keep an EMPTY registry as a stable "currently no eligible rows"
        # sentinel. If lazy init or a grad-set change later makes rows eligible,
        # validation detects the missing device/stack and rebuilds once; a
        # permanently empty stepping set no longer rebuild-thrashes every step.
        self._capt_stacks = stacks
        return bool(stacks)

    @torch._dynamo.disable
    def _capt_batched_prologue(self) -> None:
        # Validate (or lazily build) the stacked buffers, then run the batched
        # refresh chain. @torch._dynamo.disable is LOAD-BEARING, for two
        # reasons:
        #   1. The refresh chain mutates steps2d / scal2d, whose row VIEWS are
        #      also inputs of the traced step (the fused kernels read
        #      state["_capt_scalars"]; the decomposed fallback reads the
        #      counters). Tracing an in-graph mutation of a base that aliases
        #      other graph inputs sends AOTAutograd down its synthetic-base
        #      path, which drops fw_metadata.static_input_indices for the
        #      WHOLE graph -- cudagraph trees then skips the entire step
        #      ("skipping cudagraphs due to mutated inputs"; measured, torch
        #      2.12). Kept outside the trace, a compiled step runs this chain
        #      eagerly right before the recorded graph replays (fresh values
        #      every call, ~10 small launches), while the graph itself only
        #      READS the static-marked row views.
        #   2. The per-step validation walks every param; traced, each of its
        #      identity/dict checks becomes a dynamo guard evaluated on every
        #      compiled call (measured at a large fraction of a millisecond on
        #      a ~250-param model). Disabled, it is plain eager python whose
        #      cost hides under the enqueued GPU work.
        # A manual torch.cuda.graph capture records the refresh kernels
        # on-stream exactly like the rest of the step (the validation is
        # host-only; the rebuild is skipped while capturing).
        if self._capt_batched_ready():
            self._capt_batched_refresh()

    def _capt_batched_ready(self) -> bool:
        # Per-step host validation that the stacks still exactly describe this
        # step's stepping set (params in order, routing keys, counter/scalar
        # aliasing, lr/wd spec). Any mismatch tears down and -- outside a CUDA
        # graph capture -- rebuilds; during capture it falls back to the
        # bit-identical per-param path for this step (a rebuild allocates and
        # copies, which a capture would bake). Only ever called from the
        # dynamo-disabled prologue, so none of this shows up as compiled-step
        # guards.
        stacks = self._capt_stacks
        capturing = torch.cuda.is_available() and (
            torch.cuda.is_current_stream_capturing()
        )
        if stacks is None:
            if capturing:
                return False
            return self._capt_build_stacks()
        # Preserve the old uniform-stack host cost: the overwhelmingly common
        # case has one scalar cohort per device, so stack_index is always zero
        # and counts/live specs can be keyed by device directly. Keep every
        # per-row alias/routing guard, but validate group-owned hypers once per
        # contiguous device run instead of once per parameter. Heterogeneous
        # registries take the fully general path below.
        if stacks and all(len(device_stacks) == 1 for device_stacks in stacks.values()):
            if self._capt_single_cohort_ready(stacks):
                return True
            if capturing:
                self._capt_invalidate()
                return False
            return self._capt_build_stacks()
        counts = {}
        live_specs = {}
        ok = True
        for group in self.param_groups:
            if not ok:
                break
            for p in group["params"]:
                info = self._capt_row_info(p)
                if info is None:
                    continue
                device_stacks = stacks.get(p.device)
                state = self.state[p]
                stack_index = state.get("_capt_stack")
                if (
                    device_stacks is None
                    or not isinstance(stack_index, int)
                    or stack_index < 0
                    or stack_index >= len(device_stacks)
                ):
                    ok = False
                    break
                stack = device_stacks[stack_index]
                count_key = (p.device, stack_index)
                i = counts.get(count_key, 0)
                rows = stack["rows"]
                if (
                    i >= len(rows)
                    or rows[i] is not p
                    or stack["row_groups"][i] is not group
                    or stack["keys"][i] != info[0]
                    or state.get("_capt_row") != i
                    or state["step"] is not stack["step_views"][i]
                    or state.get(info[0]) is not stack["bc2_views"][i]
                    or state.get("_capt_scalars") is not stack["scalar_views"][i]
                    or stack["betas"][i] != (group["beta1"], group["beta2"])
                    or group["weight_decay"] != stack["weight_decay0"]
                ):
                    ok = False
                    break
                lr = group["lr"]
                if stack["tensor_lr"]:
                    if lr is not stack["lr0"]:
                        ok = False
                        break
                elif torch.is_tensor(lr):
                    ok = False
                    break
                live = live_specs.get(count_key)
                if live is None:
                    live_specs[count_key] = lr
                elif not stack["tensor_lr"] and lr != live:
                    ok = False
                    break
                counts[count_key] = i + 1
        if ok:
            for device, device_stacks in stacks.items():
                for stack_index, stack in enumerate(device_stacks):
                    if counts.get((device, stack_index), 0) != len(stack["rows"]):
                        ok = False
                        break
                if not ok:
                    break
        if ok:
            return True
        if capturing:
            self._capt_invalidate()
            return False
        return self._capt_build_stacks()

    def _capt_single_cohort_ready(self, stacks) -> bool:
        """Fast validation for one scalar cohort per device.

        Group hyperparameters cannot vary between parameters inside a group,
        so validate them once per contiguous device run. Row identity and all
        state aliases remain guarded per parameter, including the scalar view
        whose replacement must force a safe rebuild.
        """
        counts = {}
        live_lrs = {}
        for group in self.param_groups:
            spec_device = None
            for p in group["params"]:
                info = self._capt_row_info(p)
                if info is None:
                    continue
                device = p.device
                device_stacks = stacks.get(device)
                state = self.state[p]
                if device_stacks is None or len(device_stacks) != 1:
                    return False
                stack = device_stacks[0]
                i = counts.get(device, 0)
                rows = stack["rows"]
                if (
                    i >= len(rows)
                    or rows[i] is not p
                    or stack["row_groups"][i] is not group
                    or stack["keys"][i] != info[0]
                    or state.get("_capt_row") != i
                    or state["step"] is not stack["step_views"][i]
                    or state.get(info[0]) is not stack["bc2_views"][i]
                    or state.get("_capt_scalars") is not stack["scalar_views"][i]
                ):
                    return False
                if device != spec_device:
                    if (
                        stack["betas"][i] != (group["beta1"], group["beta2"])
                        or group["weight_decay"] != stack["weight_decay0"]
                    ):
                        return False
                    lr = group["lr"]
                    if stack["tensor_lr"]:
                        if lr is not stack["lr0"]:
                            return False
                    elif torch.is_tensor(lr):
                        return False
                    live = live_lrs.get(device)
                    if live is None:
                        live_lrs[device] = lr
                    elif not stack["tensor_lr"] and lr != live:
                        return False
                    spec_device = device
                counts[device] = i + 1
        return all(
            counts.get(device, 0) == len(device_stacks[0]["rows"])
            for device, device_stacks in stacks.items()
        )

    def _capt_batched_refresh(self) -> None:
        # One short vectorized op chain per device/cohort stack replaces every
        # registered param's counter increments + scalar-refresh chain. Only
        # ever called from the dynamo-disabled prologue (see
        # _capt_batched_prologue for why it must stay out of traced frames).
        #
        # Elementwise identical to the per-param path: the += 1 is the same
        # in-place fp32 add the per-param counters ran, and the bias
        # corrections run the exact float64 pow -> 1-x -> sqrt -> reciprocal
        # chain of _refresh_capturable_step_scalars over the stacked counters,
        # cast to fp32 only by the buffer column writes. Rows that skip the
        # sqrt (the factored path corrects with plain 1/bc2) are selected with
        # a where() over the same float64 values, which changes nothing
        # elementwise. lr semantics are also preserved per row: a shared
        # TENSOR lr is copied (and its weight-decay factor derived) on device
        # every step, so captured graphs replay fresh values; a python-float
        # lr is written with host fill_s whose values a captured graph bakes,
        # exactly as the per-param path baked them.
        for device_stacks in self._capt_stacks.values():
            for stack in device_stacks:
                steps2d = stack["steps2d"]
                steps2d += 1
                bc = 1.0 - torch.pow(stack["consts2d"], steps2d)
                bc2 = bc[0]
                sqrt_mode = stack["sqrt_mode"]
                if sqrt_mode == "all":
                    bc2 = bc2.sqrt()
                elif sqrt_mode == "mixed":
                    bc2 = torch.where(stack["sqrt_mask"], bc2.sqrt(), bc2)
                scal2d = stack["scal2d"]
                scal2d[:, 1].copy_(bc2.reciprocal())
                scal2d[:, 2].copy_(bc[1].reciprocal())
                group0 = stack["row_groups"][0]
                lr = stack["lr0"] if stack["tensor_lr"] else group0["lr"]
                weight_decay = stack["weight_decay0"]
                if torch.is_tensor(lr):
                    scal2d[:, 0].copy_(lr)
                    if weight_decay:
                        scal2d[:, 3].copy_(1.0 - lr.double() * weight_decay)
                    else:
                        scal2d[:, 3].fill_(1.0)
                else:
                    scal2d[:, 0].fill_(float(lr))
                    scal2d[:, 3].fill_(1.0 - float(lr) * weight_decay)

    def _automatic_vmean_update(
        self,
        vmean: torch.Tensor,
        grad_view: torch.Tensor,
        beta2: float,
    ) -> None:
        # The fused vmean CUDA kernel only runs on co-located CUDA tensors and
        # squares in float32; a CPU-resident param (device_map spill / CPU
        # offload) or an fp64 param must take the pure-torch reduction instead,
        # which handles both natively. Gate the fused path per-call so those
        # params never reach the CUDA binding.
        use_fused = (
            self._use_fused_automatic_vmean()
            and grad_view.is_cuda
            and vmean.is_cuda
            and grad_view.dtype != torch.float64
        )
        automatic_vmean_update(vmean, grad_view, beta2, use_fused=use_fused)

    def _gefen_codebook_device(self) -> torch.device:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    return p.grad.device
                return p.device
        # A whole-parameter non-owner deliberately has no local parameter
        # storage but must still materialize the group-agreed codebook so its
        # optimizer-common checkpoint state remains complete.
        if self._gefen_codebook_process_group is not None:
            return torch.device("cpu")
        raise ValueError(
            "Expected at least one parameter when choosing the Gefen codebook device."
        )

    def _gefen_codebook_on(self, device: torch.device) -> Optional[torch.Tensor]:
        # Params can live on several devices (HF `device_map` sharding puts
        # whole layers on different GPUs). The codebook is learned once on one
        # device; hand each step a cached per-device copy instead of tripping
        # the kernels' same-device guards. Assignment sites must clear the
        # cache (_gefen_codebook_by_device) so stale copies never survive a
        # codebook refresh or checkpoint load.
        codebook = self._gefen_codebook
        if codebook is None or codebook.device == device:
            return codebook
        cached = self._gefen_codebook_by_device.get(device)
        if cached is None:
            cached = codebook.to(device)
            self._gefen_codebook_by_device[device] = cached
        return cached

    def _gefen_codebook_lut_on(self, device: torch.device) -> Optional[torch.Tensor]:
        # Resolve once per device and retain the tensor on the optimizer.  The
        # eager Muon momentum path calls this per matrix; returning a stable
        # cached tensor avoids rebuilding the data_ptr/version cache key in the
        # public kernel wrapper hundreds of times per step.  Capturable compile
        # additionally marks this same tensor static in
        # _mark_state_static_for_compile, so no allocation enters the graph.
        cached = self._gefen_codebook_lut_by_device.get(device)
        if cached is not None:
            return cached
        codebook = self._gefen_codebook_on(device)
        if codebook is None:
            return None
        cached = _codebook_search_lut(codebook)
        self._gefen_codebook_lut_by_device[device] = cached
        return cached

    def _gefen_nearest_indices(self, normalized_vals: torch.Tensor) -> torch.Tensor:

        codebook = self._gefen_codebook_on(normalized_vals.device)
        if codebook is None:
            raise ValueError(
                "Expected Gefen codebook to be initialized before nearest-index lookup."
            )
        return gefen_nearest_codebook_indices(codebook, normalized_vals)

    def _gefen_dequantize_m_coefficients(
        self, state, like_tensor: torch.Tensor
    ) -> torch.Tensor:

        stored = state["m_codebook"]
        codebook = self._gefen_codebook_on(like_tensor.device)
        if codebook is None:
            raise ValueError(
                "Expected Gefen codebook to be initialized before dequantizing m coefficients."
            )
        return gefen_dequantize_unpacked_indices(codebook, stored, like_tensor)

    def _gefen_set_indices(self, state, indices: torch.Tensor) -> None:
        gefen_set_unpacked_indices(state["m_codebook"], indices)

    def _automatic_view(self, flat_tensor: torch.Tensor, period: int) -> torch.Tensor:
        return automatic_partition_view(flat_tensor, period)

    def _automatic_reduce(
        self, flat_tensor: torch.Tensor, period: int, reduce_op: str
    ) -> torch.Tensor:
        return automatic_partition_reduce(flat_tensor, period, reduce_op)

    def _resolve_automatic_period(
        self, param_name: str, param: torch.Tensor, grad: torch.Tensor
    ) -> int:
        # Single decision point for a parameter's automatic period so the
        # codebook-learning pass and the step agree. force_1d_period_one short-
        # circuits 1D params to per-element (period==1) before the block search.
        routing_name = param_name
        shard = self._gefen_shard_bindings.get(param)
        if shard is not None:
            routing_name = shard.parameter.fqn
        if self._force_1d_period_one and param.ndim == 1:
            return 1
        if self._force_2d_period_one and param.ndim == 2:
            return 1
        if self._period_one_substrings:
            lname = str(routing_name).lower()
            if any(sub in lname for sub in self._period_one_substrings):
                return 1
        return self._predict_period_from_grad_sq(routing_name, param, grad)

    def _predict_period_from_grad_sq(
        self, param_name: str, param: torch.Tensor, grad: torch.Tensor
    ) -> int:
        backend = _resolve_find_period_backend(grad)
        # The dedicated CUDA period-search kernel finishes each candidate's
        # cross-block variance reduction with a floating-point atomicAdd. Its
        # scheduling-dependent last bits normally do not affect the selected
        # period, but a near tie can route otherwise-identical replicas to
        # different state geometries before their first update. Deterministic
        # mode therefore uses PyTorch's fixed-order CUDA reductions, including
        # when a process-wide FIND_PERIOD_BACKEND override requested the faster
        # atomic kernel.
        if self._deterministic and backend == "cuda_kernel":
            backend = "gpu"
        # ``fused=False`` is an explicit request for a pure-PyTorch optimizer
        # step, so it must not cross a different lazy-JIT boundary merely to
        # choose the block period. A failed fused-kernel probe has the same
        # requirement. Outside deterministic mode, keep honoring an explicit
        # FIND_PERIOD_BACKEND override; callers that deliberately select
        # ``cuda_kernel`` accept its JIT.
        grad_work = grad.to_local() if hasattr(grad, "to_local") else grad
        if (
            backend == "cuda_kernel"
            and FIND_PERIOD_BACKEND is None
            and grad_work.device.type == "cuda"
            and (not self.fused or self._fused_build_ok is False)
        ):
            backend = "gpu"
        if backend == "cuda_kernel":
            grad_flat = grad_work.detach().reshape(-1)
            try:
                return find_period_by_block_variance(
                    grad_flat,
                    print_results=False,
                    parameter_name=param_name,
                    parameter_shape=tuple(param.shape),
                    backend=backend,
                    input_is_squared=False,
                )
            except RuntimeError as exc:
                msg = str(exc)
                needs_materialization = (
                    "data is not allocated yet" in msg
                    or "invalid python storage" in msg
                )
                if not needs_materialization:
                    raise

                grad_work.add_(0)
                grad_flat = grad_work.detach().reshape(-1)
                return find_period_by_block_variance(
                    grad_flat,
                    print_results=False,
                    parameter_name=param_name,
                    parameter_shape=tuple(param.shape),
                    backend=backend,
                    input_is_squared=False,
                )

        if backend == "cpu":
            period_input = grad_work.detach().float().square().reshape(-1).cpu().numpy()
        elif backend == "gpu":
            if grad_work.device.type != "cuda":
                raise ValueError("FIND_PERIOD_BACKEND='gpu' requires a CUDA tensor")
            period_input = grad_work.detach().float().square().reshape(-1)
        else:
            raise ValueError("Unexpected FIND_PERIOD_BACKEND: {}".format(backend))
        return find_period_by_block_variance(
            period_input,
            print_results=False,
            parameter_name=param_name,
            parameter_shape=tuple(param.shape),
            backend=backend,
            input_is_squared=True,
        )

    def _print_v_period(
        self, param_name: str, param: torch.Tensor, shared_v: torch.Tensor
    ) -> None:
        if not self.verbose:
            return
        shared_v_numel = shared_v.numel()
        param_numel = param.numel()
        grad = param.grad
        if shared_v_numel == 0:
            raise ValueError(
                "shared_v must have at least one element for {}".format(param_name)
            )
        if grad is None:
            raise ValueError("Expected .grad to exist for {}".format(param_name))
        if grad.shape != param.shape:
            raise ValueError(
                ".grad shape {} does not match parameter shape {} for {}".format(
                    tuple(grad.shape),
                    tuple(param.shape),
                    param_name,
                )
            )
        if param_numel % shared_v_numel != 0:
            raise ValueError(
                "Cannot compute an integer shared-v period for {}: param_numel={} shared_v_numel={}".format(
                    param_name,
                    param_numel,
                    shared_v_numel,
                )
            )

    def _iter_gefen_grad_periods(
        self, reuse_existing_periods: bool = False, staged_periods=None
    ):
        for _, param_name, p in self._iter_codebook_params_with_names():
            if p.grad is None:
                continue
            grad = p.grad
            # Codebook learning runs before the step loop's own guard, so
            # reject sparse grads here too with the same clear error.
            if getattr(grad, "is_sparse", False):
                raise RuntimeError("Gefen does not support sparse gradients")
            if torch.is_complex(grad):
                raise RuntimeError("Gefen does not support complex gradients")
            if hasattr(grad, "to_local"):
                grad = grad.to_local()
            if hasattr(grad, "wait"):
                grad = grad.wait()
            grad = grad.detach()
            flat = grad.reshape(-1)
            if flat.numel() == 0:
                continue

            if reuse_existing_periods:
                state = self.state[p]
                if "automatic_period" not in state:
                    raise ValueError(
                        "Expected automatic_period to exist for {} before refreshing Gefen codebook at optimizer step {}".format(
                            param_name,
                            self._gefen_global_step,
                        )
                    )
                period = state["automatic_period"]
            elif flat.numel() == 1:
                period = 1
            else:
                period = self._resolve_automatic_period(param_name, p, grad)

            if flat.numel() % period != 0:
                raise ValueError(
                    "Automatic partition period {} does not divide parameter {} with numel {} while learning Gefen codebook".format(
                        period,
                        param_name,
                        flat.numel(),
                    )
                )

            if staged_periods is None:
                self.state[p]["automatic_period"] = period
            elif not reuse_existing_periods:
                staged_periods.append((p, period))
            if not self._codebook_parameter_contributes(p):
                continue

            yield param_name, flat, period, grad

    @torch._dynamo.disable
    def _synchronize_codebook_scope_failure(self, error, phase: str) -> None:
        binding = self._gefen_codebook_process_group
        if binding is None or len(binding.identity.ordered_members) == 1:
            if error is not None:
                raise error
            return
        self._assert_runtime_codebook_process_group()
        self._synchronize_prevalidated_codebook_scope_failure(error, phase, binding)

    @staticmethod
    @torch._dynamo.disable
    def _synchronize_prevalidated_codebook_scope_failure(
        error, phase: str, binding
    ) -> None:
        """Synchronize through a binding validated before rank-local user code."""

        if binding is None or len(binding.identity.ordered_members) == 1:
            if error is not None:
                raise error
            return
        failed = _synchronize_step_failure(
            error is not None,
            binding.process_group,
            collective_device=binding.collective_device,
        )
        if not failed:
            return
        if error is not None:
            raise RuntimeError(
                "scoped Gefen codebook {} failed on local member {}: {}".format(
                    phase, binding.local_member, error
                )
            ) from error
        raise RuntimeError(
            "scoped Gefen codebook {} failed on another process-group member".format(
                phase
            )
        )

    @torch._dynamo.disable
    def _prepare_scoped_amp_optimizer_step(self) -> bool:
        binding = self._gefen_codebook_process_group
        if binding is None:
            return _amp_prepare_optimizer_step(self)
        found_inf = getattr(self, "found_inf", None)
        local_present = hasattr(self, "found_inf") or hasattr(
            self, "grad_scale"
        )
        try:
            local_overflow, grad_scale, local_scale = (
                _amp_optimizer_step_controls(self)
            )
            if (
                found_inf is not None
                and not torch.is_tensor(found_inf)
                and len(binding.identity.ordered_members) > 1
            ):
                raise RuntimeError(
                    "a multi-member scoped optimizer requires a group-aware "
                    "gradient scaler to provide tensor found_inf on every member"
                )
            scale_present = grad_scale is not None
            local_error = None
        except Exception as exc:
            local_overflow = False
            scale_present = False
            local_scale = 0.0
            local_error = exc
        self._synchronize_codebook_scope_failure(local_error, "AMP overflow preflight")
        if len(binding.identity.ordered_members) > 1:
            import torch.distributed as dist

            amp_control = torch.tensor(
                [
                    int(local_present),
                    int(local_overflow),
                    int(scale_present),
                    local_scale,
                ],
                dtype=torch.float64,
                device=binding.collective_device,
            )
            controls = [
                torch.empty_like(amp_control) for _ in binding.identity.ordered_members
            ]
            dist.all_gather(controls, amp_control, group=binding.process_group)
            if any(not torch.equal(item, controls[0]) for item in controls[1:]):
                raise RuntimeError(
                    "scoped Gefen AMP requires identical protocol presence, "
                    "found_inf, and grad_scale on every process-group member; "
                    "use a group-aware gradient scaler"
                )
            if local_overflow:
                return False
        elif local_overflow:
            return False
        return _amp_prepare_optimizer_step(self)

    def _reduce_codebook_scope_histogram(self, histogram: torch.Tensor) -> torch.Tensor:
        binding = self._gefen_codebook_process_group
        if binding is None or len(binding.identity.ordered_members) == 1:
            return histogram
        self._assert_runtime_codebook_process_group()
        import torch.distributed as dist

        reduced = histogram.to(binding.collective_device).clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=binding.process_group)
        return reduced.cpu()

    def _codebook_manifest_fingerprint(self):
        return self._manifest_layout_forensics()[1]

    def _compute_codebook_manifest_fingerprint(self, manifest):
        # Scoped manifests always carry contiguous slices and one process-group
        # identity, so their payload (and digest) is unchanged; the ungrouped
        # and logical-region forms only appear for unscoped finalized layouts,
        # whose digest is cached but never exchanged in a scope header.
        payload = tuple(
            (
                shard.parameter.fqn,
                shard.parameter.global_shape,
                shard.layout.value,
                shard.logical_slice.flat_offset
                if isinstance(shard.logical_slice, LogicalSlice)
                else tuple(shard.logical_slice.offsets),
                shard.logical_slice.length
                if isinstance(shard.logical_slice, LogicalSlice)
                else tuple(shard.logical_slice.lengths),
                None
                if shard.process_group is None
                else shard.process_group.semantic_name,
                None
                if shard.process_group is None
                else shard.process_group.ordered_members,
                shard.local_member,
                shard.owner,
                tuple(
                    (
                        placement.mesh_axis,
                        placement.kind.value,
                        placement.coordinate,
                        placement.parts,
                        placement.parameter_dimension,
                    )
                    for placement in shard.placements
                ),
            )
            for shard in manifest.shards
        )
        return self._sha256_int64(repr(payload).encode("utf-8"))

    @staticmethod
    def _sha256_int64(payload):
        digest = hashlib.sha256(payload).digest()
        return tuple(
            int.from_bytes(digest[index : index + 8], "big", signed=True)
            for index in range(0, len(digest), 8)
        )

    def _codebook_scope_fingerprint(self):
        return self._sha256_int64(
            repr(self._serialized_codebook_scope()).encode("utf-8")
        )

    def _codebook_value_fingerprint(self):
        if self._gefen_codebook is None:
            return (0, 0, 0, 0)
        raw = bytes(
            self._gefen_codebook.detach().cpu().contiguous().view(torch.uint8).tolist()
        )
        return self._sha256_int64(raw)

    def _validate_codebook_scope_operation_header(self, operation: str) -> None:
        binding = self._gefen_codebook_process_group
        if binding is None or len(binding.identity.ordered_members) == 1:
            return
        operation_codes = {
            "initialize": 1,
            "refresh": 2,
            "periodic_step": 3,
            "step": 4,
        }
        if operation not in operation_codes:
            raise ValueError("unknown scoped codebook operation")
        self._assert_runtime_codebook_process_group()
        import torch.distributed as dist

        header = torch.tensor(
            [
                operation_codes[operation],
                int(self._gefen_global_step),
                int(self._gefen_codebook is not None),
                int(self._deterministic),
                int(self._codebook_refresh_every),
                int(self.capturable),
                int(hasattr(self, "found_inf") or hasattr(self, "grad_scale")),
                *self._codebook_scope_fingerprint(),
                *self._codebook_manifest_fingerprint(),
                *self._codebook_value_fingerprint(),
                # Trailing decision bit, deliberately excluded from the equality
                # check below: collective-free rank-local operations
                # (post-sharding, staged/native checkpoint loads)
                # legitimately reset _gefen_codebook_scope_validated on a subset
                # of members without changing any fingerprinted value. The group
                # resolves "does any member need re-validation" here so
                # _ensure_codebook_scope_agreement re-validates on every member
                # together instead of diverging in front of its collectives.
                int(self._gefen_codebook_scope_validated),
            ],
            dtype=torch.int64,
            device=binding.collective_device,
        )
        headers = [torch.empty_like(header) for _ in binding.identity.ordered_members]
        dist.all_gather(headers, header, group=binding.process_group)
        if any(not torch.equal(item[:-1], headers[0][:-1]) for item in headers[1:]):
            raise RuntimeError(
                "scoped Gefen codebook operation, step, scope, manifest, policy, "
                "or old codebook differs across process-group members"
            )
        if any(int(item[-1].item()) == 0 for item in headers):
            self._gefen_codebook_scope_validated = False
        if operation != "step" and self._gefen_codebook is not None:
            self._verify_codebook_scope_agreement(self._gefen_codebook)

    def _validate_codebook_scope_contribution_controls(
        self, staged_periods, *, reuse_existing_periods: bool
    ) -> None:
        binding = self._gefen_codebook_process_group
        if binding is None or len(binding.identity.ordered_members) == 1:
            return
        self._assert_runtime_codebook_process_group()
        import torch.distributed as dist

        staged_by_parameter = {
            id(parameter): int(period) for parameter, period in staged_periods
        }
        local_records = []
        for parameter, shard in self._gefen_local_shard_bindings:
            active = int(
                parameter is not None
                and parameter.grad is not None
                and parameter.numel() > 0
            )
            period = -1
            if active:
                if reuse_existing_periods:
                    period = int(self.state[parameter]["automatic_period"])
                else:
                    period = staged_by_parameter[id(parameter)]
            layout_code = {
                ParameterLayout.REPLICATED: 1,
                ParameterLayout.FLATTENED_ELEMENT_SHARD: 2,
                ParameterLayout.WHOLE_PARAMETER_OWNER: 3,
            }[shard.layout]
            local_records.append(
                (layout_code, active, period, shard.logical_slice.length)
            )

        header = torch.tensor(
            list(self._codebook_manifest_fingerprint()) + [len(local_records)],
            dtype=torch.int64,
            device=binding.collective_device,
        )
        headers = [torch.empty_like(header) for _ in binding.identity.ordered_members]
        dist.all_gather(headers, header, group=binding.process_group)
        if any(not torch.equal(item, headers[0]) for item in headers[1:]):
            raise RuntimeError(
                "scoped Gefen codebook manifests differ across process-group members"
            )

        flattened = []
        for record in local_records:
            flattened.extend(record)
        local = torch.tensor(
            flattened, dtype=torch.int64, device=binding.collective_device
        )
        gathered = [torch.empty_like(local) for _ in binding.identity.ordered_members]
        dist.all_gather(gathered, local, group=binding.process_group)
        for record_index, (layout_code, _, _, _) in enumerate(local_records):
            offset = record_index * 4
            if any(int(item[offset].item()) != layout_code for item in gathered):
                raise RuntimeError(
                    "scoped Gefen codebook local layouts differ across members"
                )
            if layout_code != 1:
                if layout_code == 2:
                    nonempty_activity = {
                        int(item[offset + 1].item())
                        for item in gathered
                        if int(item[offset + 3].item()) > 0
                    }
                    if len(nonempty_activity) > 1:
                        raise RuntimeError(
                            "scoped flattened parameters require every nonempty shard to agree on gradient presence"
                        )
                continue
            replicated_controls = {
                (int(item[offset + 1].item()), int(item[offset + 2].item()))
                for item in gathered
            }
            if len(replicated_controls) != 1:
                raise RuntimeError(
                    "scoped replicated parameters require identical gradient "
                    "presence and automatic periods on every member"
                )

    def _verify_codebook_scope_agreement(self, codebook: torch.Tensor) -> None:
        binding = self._gefen_codebook_process_group
        if binding is None or len(binding.identity.ordered_members) == 1:
            return
        self._assert_runtime_codebook_process_group()
        import torch.distributed as dist

        local = (
            codebook.detach()
            .to(device=binding.collective_device, dtype=torch.float32)
            .contiguous()
        )
        gathered = [torch.empty_like(local) for _ in binding.identity.ordered_members]
        dist.all_gather(gathered, local, group=binding.process_group)
        if any(not torch.equal(item, gathered[0]) for item in gathered[1:]):
            raise RuntimeError(
                "scoped Gefen codebook values differ across process-group members"
            )

    def _ensure_codebook_scope_agreement(self) -> None:
        binding = self._gefen_codebook_process_group
        if binding is None or self._gefen_codebook_scope_validated:
            return
        # Scope re-validation runs only after a mutating API reset the flag,
        # so its collectives already amortize one full forensic rebuild.
        self._assert_runtime_codebook_process_group(full=True)
        if len(binding.identity.ordered_members) == 1:
            self._gefen_codebook_scope_validated = self._gefen_codebook is not None
            return
        import torch.distributed as dist

        control = torch.tensor(
            [
                int(self._gefen_global_step),
                int(self._gefen_codebook is not None),
                int(self._deterministic),
                int(self._codebook_refresh_every),
                int(hasattr(self, "found_inf") or hasattr(self, "grad_scale")),
                int(self.capturable),
                *self._codebook_scope_fingerprint(),
                *self._codebook_manifest_fingerprint(),
            ],
            dtype=torch.int64,
            device=binding.collective_device,
        )
        controls = [torch.empty_like(control) for _ in binding.identity.ordered_members]
        dist.all_gather(controls, control, group=binding.process_group)
        if any(not torch.equal(item, controls[0]) for item in controls[1:]):
            raise RuntimeError(
                "scoped Gefen codebook step, presence, policy, scope, manifest, "
                "or AMP protocol differs across process-group members"
            )
        if self._gefen_codebook is not None:
            self._verify_codebook_scope_agreement(self._gefen_codebook)
            self._gefen_codebook_scope_validated = True

    def _codebook_requires_2d_parameters(self) -> bool:
        return False

    def _assert_codebook_capture_ready(self) -> None:
        if (
            torch.cuda.is_available()
            and torch.cuda.is_current_stream_capturing()
            and self._gefen_codebook is None
            and self._has_local_codebook_gradients()
        ):
            raise RuntimeError(
                "Gefen codebook initialization is host-driven and must complete "
                "during eager warmup before CUDA graph capture"
            )

    def _learn_gefen_exact_codebook(
        self,
        reuse_existing_periods: bool = False,
        compute_mse_logging: bool = True,
    ):
        fused_build_ok = self._fused_build_ok
        try:
            return self._prepare_gefen_exact_codebook(
                reuse_existing_periods=reuse_existing_periods,
                compute_mse_logging=compute_mse_logging,
            )
        except Exception:
            self._fused_build_ok = fused_build_ok
            raise

    def _prepare_gefen_exact_codebook(
        self,
        reuse_existing_periods: bool = False,
        compute_mse_logging: bool = True,
    ):
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Gefen codebook initialization is host-driven and cannot run "
                "inside CUDA graph capture; initialize during eager warmup"
            )
        # Fire the fused-toolchain probe ONCE here, at the first fused CUDA
        # extension boundary (step-1 codebook learning), so a broken toolchain
        # is discovered and downgraded before the histogram / period-finding /
        # update paths run. On a healthy box this just builds the extension a
        # little earlier; for fused=False the probe is a no-op (never builds).
        if not reuse_existing_periods:
            self._validate_codebook_scope_operation_header("initialize")
        staged_periods = []
        try:
            if self.fused:
                self._gefen_fused_toolchain_ok()
            histogram = _gefen_exact_histogram_from_grad_periods(
                grad_periods=self._iter_gefen_grad_periods(
                    reuse_existing_periods=reuse_existing_periods,
                    staged_periods=staged_periods,
                ),
                num_codebooks=256,
                # ``fused=False`` promises a JIT-free pure-PyTorch path. Do not
                # load the separate histogram extension in that mode (or after
                # the fused toolchain probe failed).
                use_fused_histogram=(
                    FUSE_HISTOGRAM_FOR_EXACT
                    and self.fused
                    and self._fused_build_ok is not False
                ),
            )
            local_error = None
        except Exception as exc:
            histogram = None
            local_error = exc
        self._synchronize_codebook_scope_failure(
            local_error, "local histogram preparation"
        )
        self._validate_codebook_scope_contribution_controls(
            staged_periods,
            reuse_existing_periods=reuse_existing_periods,
        )
        histogram = self._reduce_codebook_scope_histogram(histogram)

        try:
            codebook = _learn_gefen_exact_codebook_from_histogram(
                bin_counts_cpu=histogram,
                codebook_device=self._gefen_codebook_device(),
                num_codebooks=256,
                force_endpoints=True,
                verbose=self.verbose,
                compute_mse_logging=compute_mse_logging,
            )
            local_error = None
        except Exception as exc:
            codebook = None
            local_error = exc
        self._synchronize_codebook_scope_failure(
            local_error, "exact-DP solve and placement"
        )
        if codebook is not None:
            self._verify_codebook_scope_agreement(codebook)
        return codebook, staged_periods

    def _ensure_gefen_codebook(self, reuse_existing_periods: bool = False) -> None:
        codebook, staged_periods = self._learn_gefen_exact_codebook(
            reuse_existing_periods=reuse_existing_periods,
            compute_mse_logging=False,
        )
        if codebook is not None:
            for parameter, period in staged_periods:
                self.state[parameter]["automatic_period"] = period
            self._gefen_codebook = codebook
            self._gefen_codebook_by_device.clear()
            self._gefen_codebook_lut_by_device.clear()
            self._gefen_codebook_scope_validated = (
                self._gefen_codebook_process_group is not None
            )

    def _step_automatic_factored(
        self,
        group,
        param_name: str,
        p: torch.Tensor,
        grad: torch.Tensor,
        *,
        state=None,
    ) -> None:
        # Adafactor-style factored second moment for a 2D param (opt-in via
        # factored_v_2d). The quantized-momentum machinery is byte-identical to
        # the merged non-fused path (_automatic_momentum_update_merged); only
        # the second moment differs: instead of the block-shared vmean EMA, we
        # keep fp32 row/col EMAs of grad^2 and reconstruct a per-element
        # V ~= outer(R, C) / mean(R) each step (Shazeer & Stern 2018), giving
        # per-element stepsize fidelity at (rows + cols) state. On the fused
        # CUDA path the stepsize is computed in-kernel (no full-size
        # temporaries); elsewhere a decomposed whole-tensor fallback applies.
        if hasattr(grad, "to_local"):
            grad = grad.to_local()
        if hasattr(grad, "wait"):
            grad = grad.wait()
        state = self.state[p] if state is None else state
        beta1 = group["beta1"]
        beta2 = group["beta2"]
        lr = group["lr"]
        eps = group["eps"]

        flat_grad = grad.reshape(-1)
        if flat_grad.numel() == 0:
            return

        if "step" not in state:
            if "automatic_period" in state:
                automatic_period = state["automatic_period"]
            elif flat_grad.numel() == 1:
                automatic_period = 1
            else:
                automatic_period = self._resolve_automatic_period(param_name, p, grad)
            if flat_grad.numel() % automatic_period != 0:
                raise ValueError(
                    "Automatic partition period {} does not divide parameter {} with numel {}".format(
                        automatic_period, param_name, flat_grad.numel()
                    )
                )
            state["automatic_period"] = automatic_period
            state["step"] = self._new_step_counter(p.device)
            grad_view = self._automatic_view(flat_grad, automatic_period)
            self._init_gefen_state(state, grad_view)
            state["v_row"] = torch.zeros(
                p.shape[0], dtype=torch.float32, device=p.device
            )
            state["v_col"] = torch.zeros(
                p.shape[1], dtype=torch.float32, device=p.device
            )
            state["factored_step"] = self._new_step_counter(p.device)

        automatic_period = state["automatic_period"]
        grad_view = self._automatic_view(flat_grad, automatic_period)

        # Resume compatibility: a checkpoint saved with factored_v_2d off (or
        # from before the flag existed) restores "step"/momentum state but no
        # row/col EMAs. Initialize them lazily; the EMA bias correction uses
        # the param's own step count, so the first post-resume steps behave
        # like a fresh EMA warm-up on those tensors (convergence-neutral).
        if "v_row" not in state:
            state["v_row"] = torch.zeros(
                p.shape[0], dtype=torch.float32, device=p.device
            )
            state["v_col"] = torch.zeros(
                p.shape[1], dtype=torch.float32, device=p.device
            )
            state["factored_step"] = self._new_step_counter(p.device)

        # Migrate any state stranded on a previous device (CPU-offload paging)
        # onto the current compute device before the counter bump and second-
        # moment reads below. No-op on the common same-device step.
        self._colocate_param_state_(state, grad_view.device)

        rows, cols = p.shape
        # Rows registered with the batched capturable prologue had BOTH
        # counters advanced by its single stacked add_ already this step.
        if "_capt_row" not in state:
            state["step"] += 1
            state["factored_step"] += 1

        if (
            self._use_fused_gefen_automatic_step()
            and p.is_cuda
            and p.dtype != torch.float64
            and not self._deterministic
        ):
            # Fused path. Phase 1 of the kernel computes the per-block absmax
            # AND the raw row/col grad^2 sums in ONE pass over grad + m_sign;
            # the v_row/v_col EMAs advance in place device-side and the
            # per-element stepsize V_ij ~= r_i * c_j / mean(r) is computed in
            # registers in phase 2 -- no host-side reduction pass, no host
            # sync, no full-size temporaries anywhere in the step.
            codebook = self._gefen_codebook_on(p.device)
            if codebook is None:
                raise ValueError(
                    "Expected Gefen codebook to be initialized before the factored update."
                )
            # The raw kernel flat-indexes a contiguous matrix.  A strided leaf
            # Parameter is uncommon but valid (and can arise from model surgery
            # or a view-backed weight); falling all the way to the decomposed
            # PyTorch path is an order-of-magnitude regression.  Mirror the
            # block-vmean fused path: update a contiguous logical copy, then
            # copy it back into the original storage.  State/grad geometry is
            # unchanged and the direct contiguous case remains allocation-free.
            p_kernel = p if p.is_contiguous() else p.contiguous()
            if self.capturable:
                # Capturable: the per-step-varying scalars flow through the
                # device buffer ([lr, 1/bc2, 1/bc1, wd_factor]; the kernel
                # ignores the matching host args, passed as inert values), so
                # a captured graph replays fresh values and never syncs on a
                # tensor lr.
                step_scalars = self._refresh_capturable_step_scalars(
                    state, group, "factored_step", sqrt_bc2=False
                )
                _gefen_factored_update_cuda(
                    p_kernel,
                    grad_view,
                    state["m_codebook"],
                    state["m_magnitude"],
                    state["v_row"],
                    state["v_col"],
                    codebook,
                    cols,
                    beta1,
                    beta2,
                    0.0,  # lr: read from step_scalars[0]
                    eps,
                    1.0,  # 1/bc2: read from step_scalars[1]
                    1.0,  # 1/bc1: read from step_scalars[2]
                    1.0,  # wd factor: read from step_scalars[3]
                    self._stochastic_round,
                    self._kernel_rng_seed(),
                    step_scalars=step_scalars,
                    codebook_lut=self._gefen_codebook_lut_on(p.device),
                    seed_dev=self._sr_seed_on(p.device),
                )
                if p_kernel is not p:
                    p.copy_(p_kernel)
                return
            # bc1 corrects the (possibly checkpoint-restored, older) momentum
            # EMA; bc2 must correct the factored v EMA's own age.
            bias_correction_1 = 1 - beta1 ** state["step"]
            bias_correction_2 = 1 - beta2 ** state["factored_step"]
            if torch.is_tensor(lr):
                # The kernel binding takes lr / wd_factor as host doubles;
                # scalarize a tensor lr through the cached no-sync .item()
                # hoist (_lr_scalar) instead of passing the tensor through.
                lr = self._lr_scalar(group)
            weight_decay_factor = 1.0 - lr * group["weight_decay"]
            _gefen_factored_update_cuda(
                p_kernel,
                grad_view,
                state["m_codebook"],
                state["m_magnitude"],
                state["v_row"],
                state["v_col"],
                codebook,
                cols,
                beta1,
                beta2,
                lr,
                eps,
                1.0 / bias_correction_2,
                1.0 / bias_correction_1,
                weight_decay_factor,
                self._stochastic_round,
                self._kernel_rng_seed(),
            )
            if p_kernel is not p:
                p.copy_(p_kernel)
            return

        # bc1 corrects the (possibly checkpoint-restored, older) momentum EMA;
        # bc2 must correct the factored v EMA's own age. Under capturable these
        # are 0-dim device tensors (tensor step counters) consumed by the
        # decomposed tensor-op math below.
        bias_correction_1 = 1 - beta1 ** state["step"]
        bias_correction_2 = 1 - beta2 ** state["factored_step"]

        # Decomposed fallback (CPU / fused disabled / deterministic factored-v).
        # The fast factored CUDA stats kernel uses unordered floating-point
        # atomicAdd operations for row/column partials, so deterministic mode
        # deliberately reaches this fixed-order reduction path while block-vmean
        # parameters continue using the fused v1 kernel. Row/col mean-square
        # EMAs run over bounded fp32 row-chunks (one chunk serves both reductions,
        # so the transient stays ~a few chunk sizes), then whole-tensor torch ops.
        grad2d = grad.detach()
        row_ms = torch.empty(rows, dtype=torch.float32, device=p.device)
        col_sq = torch.zeros(cols, dtype=torch.float32, device=p.device)
        rows_per_chunk = max(1, (1 << 20) // max(1, cols))
        for r0 in range(0, rows, rows_per_chunk):
            r1 = min(r0 + rows_per_chunk, rows)
            chunk = grad2d[r0:r1]
            # .float() on an fp32 grad returns an ALIAS, so squaring in place
            # would corrupt p.grad (which grad_view below also reads); square
            # out-of-place there. Half-precision grads get a fresh fp32 copy
            # from .float(), which square_ can safely reuse.
            if chunk.dtype == torch.float32:
                chunk_sq = chunk.square()
            else:
                chunk_sq = chunk.float().square_()
            row_ms[r0:r1] = chunk_sq.sum(dim=1)
            col_sq += chunk_sq.sum(dim=0)
        row_ms.div_(cols)
        col_sq.div_(rows)
        state["v_row"].mul_(beta2).add_(row_ms, alpha=1 - beta2)
        state["v_col"].mul_(beta2).add_(col_sq, alpha=1 - beta2)

        # Decomposed fallback: whole-tensor torch ops.
        # The kernel above is the canonical numerics; this matches it within
        # float tolerance (different op association), not bit-for-bit.
        v_hat = torch.outer(state["v_row"], state["v_col"]).div_(
            state["v_row"].mean().clamp_(min=torch.finfo(torch.float32).tiny)
        )
        denom = v_hat.div_(bias_correction_2).sqrt_().add_(eps)
        stepsize = denom.reciprocal_().mul_(1.0 / bias_correction_1)

        # Quantized momentum advance, identical math to the merged non-fused
        # path; returns the reconstructed fp32 momentum (codebook * magnitude).
        shared_m = self._automatic_momentum_update_merged(state, grad_view, beta1)

        # The weight-decay and update multiplies below run on the compute
        # device; a tensor lr stranded on another device (a CUDA lr after the
        # parameter was paged to CPU) would break them. Scalarize when not
        # capturable -- bit-identical to the 0-dim tensor (value == lr.item()).
        # Capturable keeps the on-device tensor lr for its device-tensor math.
        if torch.is_tensor(lr) and not self.capturable:
            lr = self._lr_scalar(group)
        if group["weight_decay"] > 0.0:
            p.mul_(1 - lr * group["weight_decay"])
        update = shared_m.view(p.shape).mul_(stepsize).mul_(lr)
        p.add_(-update)

    def _refresh_codebook_with_requant(self) -> None:
        # Re-learn the exact-DP codebook from the CURRENT step's gradients
        # (periods reused -- block geometry is baked into the per-param state)
        # and re-quantize every stored momentum index from the old codebook
        # onto the new one. The normalized coefficients are dequantized with
        # the codebook that WROTE them, so the momentum survives the swap with
        # only nearest-codeword rounding error; m_magnitude is unchanged
        # (coefficients stay normalized to [-1, 1]).
        if self.capturable:
            raise RuntimeError(
                "capturable=True does not support replacing the learned codebook"
            )
        self._validate_codebook_scope_operation_header("refresh")
        old_codebook = self._gefen_codebook
        if old_codebook is None:
            return
        new_codebook, staged_periods = self._learn_gefen_exact_codebook(
            reuse_existing_periods=True, compute_mse_logging=False
        )
        if staged_periods:
            raise RuntimeError(
                "codebook refresh unexpectedly attempted to replace block periods"
            )
        if new_codebook is None:
            return
        staged_indices = []
        if self._gefen_codebook_process_group is None:
            parameters = [p for pgroup in self.param_groups for p in pgroup["params"]]
        else:
            parameters = [
                p for p, _ in self._gefen_local_shard_bindings if p is not None
            ]
        try:
            for p in parameters:
                pstate = self.state.get(p)
                if not pstate or "m_codebook" not in pstate:
                    continue
                stored_device = pstate["m_codebook"].device
                # Do not populate/mutate the per-device cache until the whole
                # refresh transaction succeeds. A failure while staging a later
                # parameter must leave both indices and cache identity untouched.
                old_codebook_local = old_codebook.to(stored_device)
                new_codebook_local = new_codebook.to(stored_device)
                coeffs = gefen_dequantize_unpacked_indices(
                    old_codebook_local, pstate["m_codebook"], old_codebook_local
                )
                indices = gefen_nearest_codebook_indices(new_codebook_local, coeffs)
                staged_indices.append((pstate["m_codebook"], indices))
            local_error = None
        except Exception as exc:
            local_error = exc
        self._synchronize_codebook_scope_failure(
            local_error, "momentum requantization staging"
        )

        # Commit only after every member prepared every local replacement.
        # The staged tensors already have the destination shape/device, so this
        # tail is a deterministic sequence of infallible in-place copies.
        for stored_indices, indices in staged_indices:
            gefen_set_unpacked_indices(stored_indices, indices)
        self._gefen_codebook = new_codebook
        self._gefen_codebook_by_device.clear()
        self._gefen_codebook_lut_by_device.clear()

    def _resuming_from_checkpoint(self) -> bool:
        # On resume every param's state has a restored automatic_period; on a
        # fresh run the state is still empty when the first step starts.
        return any(
            "automatic_period" in self.state.get(p, {})
            for group in self.param_groups
            for p in group["params"]
        )

    def _scope_agreed_resuming_from_checkpoint(self) -> bool:
        # Per-parameter state presence is legitimately rank-asymmetric under an
        # explicit scope (empty non-owner slots and zero-length shards never
        # create state), so the rank-local _resuming_from_checkpoint() predicate
        # must not gate collectives directly: resolve one group-wide decision
        # before any member branches on it. Any member that restored periods
        # makes the whole scope reuse them.
        resuming = self._resuming_from_checkpoint()
        binding = self._gefen_codebook_process_group
        if binding is None or len(binding.identity.ordered_members) == 1:
            return resuming
        self._assert_runtime_codebook_process_group()
        import torch.distributed as dist

        control = torch.tensor(
            int(resuming),
            dtype=torch.int32,
            device=binding.collective_device,
        )
        dist.all_reduce(control, op=dist.ReduceOp.MAX, group=binding.process_group)
        return bool(int(control.item()))

    def _maybe_refresh_gefen_codebook(self) -> None:
        if self._gefen_codebook is not None:
            # A codebook restored from a checkpoint may land on CPU (depending on
            # the load map_location); move it onto the gradient device once.
            device = self._gefen_codebook_device()
            if self._gefen_codebook.device != device:
                self._gefen_codebook = self._gefen_codebook.to(device)
                self._gefen_codebook_by_device.clear()
                self._gefen_codebook_lut_by_device.clear()
            return

        binding = self._gefen_codebook_process_group
        if (
            binding is None or len(binding.identity.ordered_members) == 1
        ) and not self._has_local_codebook_gradients():
            # No local histogram exists, so learning would return None. This
            # fast no-op is also what keeps a no-gradient capturable step free
            # of the host-driven exact-DP preparation path. Multi-member scopes
            # still enter collectively because another member (for example a
            # whole-parameter owner) may hold the only active gradient.
            return

        # No codebook yet. On a fresh run this learns it and predicts periods.
        # On resume the persisted codebook is normally restored in
        # load_state_dict, but FSDP optim-state consolidation strips custom
        # top-level keys, so the codebook can be missing even though the
        # per-param state (incl. automatic_period) was restored. In that case we
        # must reuse the restored periods rather than re-predict them, otherwise
        # the refreshed periods desync from the restored vmean/m_codebook block
        # geometry and the vmean kernel aborts on a block-count mismatch.
        self._ensure_gefen_codebook(
            reuse_existing_periods=self._scope_agreed_resuming_from_checkpoint()
        )

    @torch.no_grad()
    def initialize_codebook(self) -> bool:
        """Collectively initialize the learned codebook without taking a step."""

        # Capture a usable failure-vote binding before inspecting live layout so
        # a one-sided preamble failure -- for example capture-readiness raising
        # only on the gradient-owning member while non-owners proceed -- is
        # reported through the scope instead of stranding peers inside the
        # scoped operation-header collective below.
        scope_binding = self._capture_codebook_scope_binding_for_step()
        try:
            self._assert_finalized_binding_layout(full=True)
            self._assert_runtime_codebook_process_group()
            self._assert_codebook_capture_ready()
            local_preamble_error = None
        except Exception as exc:
            local_preamble_error = exc
        if scope_binding is not None:
            self._synchronize_prevalidated_codebook_scope_failure(
                local_preamble_error, "initialize preamble", scope_binding
            )
        elif local_preamble_error is not None:
            raise local_preamble_error
        self._validate_codebook_scope_operation_header("initialize")
        try:
            _assert_optimizer_gradients_structurally_valid(
                self, require_2d_params=self._codebook_requires_2d_parameters()
            )
            local_error = None
        except Exception as exc:
            local_error = exc
        if self._gefen_codebook_process_group is not None:
            self._synchronize_codebook_scope_failure(local_error, "gradient preflight")
        elif local_error is not None:
            raise local_error
        self._ensure_codebook_scope_agreement()
        if self._gefen_codebook is not None:
            return False
        self._ensure_gefen_codebook(
            reuse_existing_periods=self._scope_agreed_resuming_from_checkpoint()
        )
        return self._gefen_codebook is not None

    @torch.no_grad()
    def refresh_codebook(self) -> bool:
        """Collectively relearn and atomically requantize the current codebook."""

        self._assert_finalized_binding_layout(full=True)
        self._assert_runtime_codebook_process_group()
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "capturable=True does not support replacing the learned codebook"
            )
        self._validate_codebook_scope_operation_header("refresh")
        if self.capturable:
            raise RuntimeError(
                "capturable=True does not support replacing the learned codebook"
            )
        if self._gefen_codebook is None:
            raise RuntimeError("Gefen codebook must be initialized before refresh")
        try:
            _assert_optimizer_gradients_structurally_valid(
                self, require_2d_params=self._codebook_requires_2d_parameters()
            )
            local_error = None
        except Exception as exc:
            local_error = exc
        if self._gefen_codebook_process_group is not None:
            self._synchronize_codebook_scope_failure(local_error, "gradient preflight")
        elif local_error is not None:
            raise local_error
        self._ensure_codebook_scope_agreement()
        before = self._gefen_codebook
        self._refresh_codebook_with_requant()
        return self._gefen_codebook is not before

    def _maybe_save_gefen_grad_histogram(self) -> None:
        if not hasattr(quantization_module, "LIST_STEPS_SAVE_HIST_GRAD"):
            return
        requested_steps = quantization_module.LIST_STEPS_SAVE_HIST_GRAD
        if requested_steps is None:
            return
        if self.capturable and requested_steps:
            raise ValueError(
                "capturable=True is incompatible with LIST_STEPS_SAVE_HIST_GRAD: "
                "histogram collection and file output are host-driven"
            )
        if self._gefen_global_step not in requested_steps:
            return
        if not hasattr(quantization_module, "SAVE_CODEBOOK_PREFIX"):
            raise ValueError(
                "LIST_STEPS_SAVE_HIST_GRAD requires SAVE_CODEBOOK_PREFIX to be defined and set."
            )
        save_codebook_prefix = quantization_module.SAVE_CODEBOOK_PREFIX
        if save_codebook_prefix is None:
            raise ValueError(
                "LIST_STEPS_SAVE_HIST_GRAD requires SAVE_CODEBOOK_PREFIX to be set."
            )
        if self._gefen_codebook is None:
            raise ValueError(
                "Expected Gefen codebook to be initialized before saving gradient histogram."
            )

        histogram_bins = self._gefen_codebook.numel() * 16
        bin_width = 2.0 / float(histogram_bins)
        bin_counts_cpu = torch.zeros(histogram_bins, dtype=torch.float32, device="cpu")
        total_numel = 0

        for _, flat, period, _ in self._iter_gefen_grad_periods(
            reuse_existing_periods=True
        ):
            flat_float = flat.float()
            blocks = self._automatic_view(flat_float, period)
            absmax = blocks.abs().amax(dim=1, keepdim=True)
            normalized = torch.where(
                absmax > 0, blocks / absmax, torch.zeros_like(blocks)
            )
            normalized_flat = normalized.reshape(-1)
            bin_indices = torch.floor((normalized_flat + 1.0) / bin_width).to(
                torch.long
            )
            bin_indices = bin_indices.clamp(0, histogram_bins - 1)
            local_counts = torch.bincount(bin_indices, minlength=histogram_bins).float()
            bin_counts_cpu.add_(local_counts.cpu())
            total_numel += normalized_flat.numel()

        if total_numel == 0:
            raise ValueError(
                "Cannot save Gefen gradient histogram because no gradients were available."
            )

        quantization_module.save_distribution_histogram_pt_from_counts(
            hist=bin_counts_cpu,
            prefix=save_codebook_prefix,
            suffix="grad_hist_step_{}".format(self._gefen_global_step),
            num_samples=total_numel,
        )

    def _init_gefen_state(self, state, grad_view: torch.Tensor) -> None:
        state["m_codebook"] = torch.zeros_like(
            grad_view, dtype=torch.uint8, memory_format=torch.preserve_format
        )

        state["m_magnitude"] = torch.zeros_like(
            grad_view[:, 0:1],
            dtype=torch.float32,
            memory_format=torch.preserve_format,
        )

    def _gefen_quantized_momentum_update(
        self, state, grad_view, beta1, *, nesterov=False
    ) -> torch.Tensor:
        # Muon-specific quantized-momentum update. Advances the quantized momentum
        # state (m_codebook / m_magnitude) by one EMA step and returns the DENSE
        # quantized momentum for Newton-Schulz in a single kernel pass -- no lr==0
        # dummy-stepsize parameter write, and no second full-size codebook gather.
        # Optional Nesterov changes only that dense output; persistent state stays
        # the underlying EMA, matching the decomposed optimizer path.
        codebook = self._gefen_codebook_on(grad_view.device)
        if codebook is None:
            raise ValueError(
                "Expected Gefen codebook to be initialized before quantized momentum update."
            )
        # Allocate contiguous explicitly (not empty_like, which would inherit a
        # non-contiguous memory format and trip the kernel's contiguity guard);
        # the kernel wrapper also forces grad_view contiguous.
        momentum_out = torch.empty(
            grad_view.shape, dtype=grad_view.dtype, device=grad_view.device
        )
        _gefen_quantized_momentum_update_cuda(
            grad_view,
            state["m_codebook"],
            state["m_magnitude"],
            codebook,
            momentum_out,
            beta1,
            self._stochastic_round,
            self._kernel_rng_seed(),
            seed_dev=self._sr_seed_on(grad_view.device),
            codebook_lut=self._gefen_codebook_lut_on(grad_view.device),
            nesterov=nesterov,
        )
        return momentum_out

    def _automatic_gefen_fused_full_update(
        self,
        p,
        state,
        grad_view,
        beta1,
        beta2,
        lr,
        eps,
        bias_correction_1,
        bias_correction_2,
        weight_decay_factor,
        step_scalars=None,
    ) -> None:
        # Tier-1 fully-fused v1 path: the kernel computes the vmean EMA and the
        # per-block stepsize in-kernel, so the separate vmean kernel and the host
        # sqrt/div/reciprocal/mul launches are skipped. The scalar reciprocals are
        # precomputed here as Python doubles and cast to float32 in the kernel
        # exactly as PyTorch casts the scalar operands of the host div_/mul_.
        # PyTorch lowers `h.div_(sqrt(bc2))` to a multiply by float32(1/sqrt(bc2))
        # (a true divide differs by up to ~2 ULP), so pass the reciprocal directly.
        # Under capturable (step_scalars is a device tensor) the kernel reads
        # [lr, 1/sqrt(bc2), 1/bc1, wd_factor] from device memory instead and the
        # host scalars are inert placeholders (never .item()-sync a tensor lr).
        codebook = self._gefen_codebook_on(p.device)
        if codebook is None:
            raise ValueError(
                "Expected Gefen codebook to be initialized before fused update."
            )
        if step_scalars is None:
            lr_arg = lr
            inv_sqrt_bc2 = 1.0 / math.sqrt(bias_correction_2)
            inv_bc1 = 1.0 / bias_correction_1
        else:
            lr_arg = 0.0
            inv_sqrt_bc2 = 1.0
            inv_bc1 = 1.0
        _automatic_gefen_fused_full_update_cuda(
            p,
            grad_view,
            state["m_codebook"],
            state["m_magnitude"],
            state["vmean"],
            codebook,
            False,
            beta1,
            beta2,
            lr_arg,
            eps,
            inv_sqrt_bc2,
            inv_bc1,
            weight_decay_factor,
            self._stochastic_round,
            self._kernel_rng_seed(),
            step_scalars=step_scalars,
            codebook_lut=(
                self._gefen_codebook_lut_on(p.device)
                if step_scalars is not None
                else None
            ),
            seed_dev=self._sr_seed_on(p.device),
        )

    def _automatic_gefen_fused_update_v2_full(
        self,
        p,
        state,
        grad_view,
        beta1,
        beta2,
        lr,
        eps,
        bias_correction_1,
        bias_correction_2,
        weight_decay_factor,
        step_scalars=None,
    ) -> None:
        # Tier-1 fully-fused v2 (two-phase) path for occupancy-flexible params:
        # the magnitude phase also accumulates Sum(grad^2), a tiny finalize forms
        # the vmean EMA + per-block stepsize, and weight decay is folded into the
        # update write -- so the separate vmean kernel and the host
        # stepsize/weight-decay passes are skipped. The scalar reciprocals are
        # precomputed as Python doubles and cast in-kernel exactly as PyTorch
        # casts the operands of the host div_/mul_ (see the v1-full path).
        # Under capturable (step_scalars is a device tensor) the kernels read
        # [lr, 1/sqrt(bc2), 1/bc1, wd_factor] from device memory instead and the
        # host scalars are inert placeholders (never .item()-sync a tensor lr).
        codebook = self._gefen_codebook_on(p.device)
        if codebook is None:
            raise ValueError(
                "Expected Gefen codebook to be initialized before fused update."
            )
        if step_scalars is None:
            lr_arg = lr
            inv_sqrt_bc2 = 1.0 / math.sqrt(bias_correction_2)
            inv_bc1 = 1.0 / bias_correction_1
        else:
            lr_arg = 0.0
            inv_sqrt_bc2 = 1.0
            inv_bc1 = 1.0
        _automatic_gefen_fused_update_v2_full_cuda(
            p,
            grad_view,
            state["m_codebook"],
            state["m_magnitude"],
            state["vmean"],
            codebook,
            False,
            beta1,
            beta2,
            lr_arg,
            eps,
            inv_sqrt_bc2,
            inv_bc1,
            weight_decay_factor,
            self._stochastic_round,
            self._kernel_rng_seed(),
            step_scalars=step_scalars,
            codebook_lut=(
                self._gefen_codebook_lut_on(p.device)
                if step_scalars is not None
                else None
            ),
            seed_dev=self._sr_seed_on(p.device),
        )

    def _automatic_momentum_update(
        self,
        state,
        grad_view: torch.Tensor,
        beta1: float,
        stepsize: torch.Tensor,
        lr: float,
        p_local: torch.Tensor,
    ) -> None:
        # Streamed (row-chunked) momentum update *and* parameter apply. This is a
        # numerics-preserving, low-transient-memory rewrite of the old
        #     shared_m = <whole-tensor momentum update returning fp32 [blocks,period]>
        #     update  = (shared_m * stepsize).view(grad.size()); update.mul_(lr)
        #     p_local.add_(-update)
        # sequence. The old form kept up to ~5 full-size fp32 temporaries live at
        # once (current_m, the promoted fp32 grad, updated_m, its abs() temp, the
        # reconstructed shared_m) plus the fp32 update/-update -- multiple GiB on
        # the tied embedding. Every block's update is row-local (per-block max,
        # normalize, codebook round-trip, and the elementwise scale/add all stay
        # within one row), so we run the exact same arithmetic one whole-row chunk
        # at a time, capping every fp32 temporary to one chunk while writing the
        # result straight into p. Bit-for-bit identical to the whole-tensor form.
        #
        # Used by the per-param (incl. the large/streamed) non-fused path. The
        # foreach-merged path uses the sibling _automatic_momentum_update_merged
        # (bounded merged transient, scatter-back instead of in-place apply).
        if grad_view.dim() != 2:
            raise ValueError(
                "Automatic shared momentum expects a 2D tensor, got dim={}".format(
                    grad_view.dim()
                )
            )

        codebook = self._gefen_codebook_on(grad_view.device)
        if codebook is None:
            raise ValueError(
                "Expected Gefen codebook to be initialized before the momentum update."
            )
        if codebook.device != grad_view.device:
            raise ValueError(
                "Gefen codebook device {} does not match grad_view device {}.".format(
                    codebook.device,
                    grad_view.device,
                )
            )

        period = state["automatic_period"]
        m_codebook = state["m_codebook"]
        m_magnitude = state["m_magnitude"]
        num_blocks = grad_view.shape[0]

        # The apply writes p through a flat view so each row chunk maps to a
        # contiguous element range. A contiguous shard is written in place; a
        # non-contiguous shard is updated on a contiguous copy and copied back so
        # the in-place semantics match the stock p_local.add_(-update).
        p_contig = p_local.is_contiguous()
        p_work = p_local if p_contig else p_local.contiguous()
        p_flat = p_work.view(-1)

        rows_per_chunk = max(1, GEFEN_MOMENTUM_UPDATE_CHUNK // period)
        for r0 in range(0, num_blocks, rows_per_chunk):
            r1 = min(r0 + rows_per_chunk, num_blocks)

            # Dequantize the stored (old) momentum coefficients for these rows and
            # rescale by their old magnitude. Dequant returns grad_view.dtype
            # (e.g. bf16); the * fp32 magnitude promotes current_m to fp32 -- same
            # as the stock dequant(...) * m_magnitude.
            current_m = (
                gefen_dequantize_unpacked_indices(
                    codebook, m_codebook[r0:r1], grad_view
                )
                * m_magnitude[r0:r1]
            )
            # lerp() needs matching dtypes, so promote the (bf16) grad rows to fp32.
            updated_m = current_m.lerp(grad_view[r0:r1].to(current_m.dtype), 1 - beta1)

            # New per-block magnitude = max |updated_m| over each row, written back
            # into the persistent fp32 m_magnitude state in place.
            mag_slot = m_magnitude[r0:r1]
            mag_slot.copy_(
                automatic_partition_reduce(
                    updated_m.abs().reshape(-1), period, reduce_op="max"
                )
            )

            nonzero_mask = mag_slot > 0
            updated_m.div_(mag_slot)
            updated_m.masked_fill_(~nonzero_mask, 0.0)

            # Re-quantize the normalized coefficients and store the fresh indices.
            indices = gefen_nearest_codebook_indices(codebook, updated_m)
            gefen_set_unpacked_indices(m_codebook[r0:r1], indices)

            # Reconstruct the quantized momentum, scale by the new magnitude, the
            # per-block stepsize, and lr, then apply to p in place. Folding the
            # scale/negate into this loop avoids the full-size fp32 shared_m and
            # update/-update temporaries entirely.
            shared_m = gefen_dequantize_unpacked_indices(codebook, indices, grad_view)
            update = shared_m * mag_slot
            update.mul_(stepsize[r0:r1]).mul_(lr)
            # p_flat slice (bf16) += -update (fp32); alpha=-1 negates exactly
            # (sign flip), so this matches the stock p_local.add_(-update).
            p_flat[r0 * period : r1 * period].add_(update.reshape(-1), alpha=-1)

        if not p_contig:
            p_local.copy_(p_work)

    def _automatic_momentum_update_merged(
        self, state, grad_view: torch.Tensor, beta1: float
    ) -> torch.Tensor:
        # Whole-tensor (non-streamed) momentum reconstruct for the foreach-merged
        # batch. `state` here is the synthetic merged dict assembled in
        # _step_automatic_merged (automatic_period + the stacked merged m_magnitude
        # / m_codebook buffers), and grad_view is the merged (K*nblocks, period)
        # grid. The merged tensor is bounded by GEFEN_NONFUSED_FOREACH_BUDGET, so
        # its fp32 temporaries are already small -- streaming it would only add
        # kernel launches back and defeat the batching. This is the original
        # whole-tensor momentum update with the chunked-gather fix (identical math
        # to the per-param streamed sibling above), returning the reconstructed
        # shared momentum (fp32 [K*nblocks, period]); the caller scatters it back
        # to the K params via torch._foreach_add_.
        if grad_view.dim() != 2:
            raise ValueError(
                "Automatic shared momentum expects a 2D tensor, got dim={}".format(
                    grad_view.dim()
                )
            )

        codebook = self._gefen_codebook_on(grad_view.device)
        if codebook is None:
            raise ValueError(
                "Expected Gefen codebook to be initialized before reconstructing quantized m."
            )

        period = state["automatic_period"]
        current_m = (
            gefen_dequantize_unpacked_indices(codebook, state["m_codebook"], grad_view)
            * state["m_magnitude"]
        )
        updated_m = current_m.lerp(grad_view.to(current_m.dtype), 1 - beta1)

        state["m_magnitude"].copy_(
            automatic_partition_reduce(
                updated_m.abs().reshape(-1), period, reduce_op="max"
            )
        )

        nonzero_mask = state["m_magnitude"] > 0
        updated_m.div_(state["m_magnitude"])
        updated_m.masked_fill_(~nonzero_mask, 0.0)

        indices = gefen_nearest_codebook_indices(codebook, updated_m)
        gefen_set_unpacked_indices(state["m_codebook"], indices)

        quantized_m = gefen_dequantize_unpacked_indices(codebook, indices, grad_view)
        return quantized_m * state["m_magnitude"]

    def _step_automatic(
        self,
        group,
        param_name: str,
        p: torch.Tensor,
        grad: torch.Tensor,
        *,
        state=None,
    ) -> None:
        # Unwrap the gradient to its local shard for both the fused and
        # non-fused paths. Under FSDP2 the gradient arrives as a sharded DTensor;
        # the raw CUDA kernels (vmean + fused Gefen update) need a materialized,
        # contiguous local tensor, otherwise they hit "data is not allocated
        # yet". The non-fused path already relied on this unwrap.
        if hasattr(grad, "to_local"):
            grad = grad.to_local()
        if hasattr(grad, "wait"):
            grad = grad.wait()

        state = self.state[p] if state is None else state
        beta1 = group["beta1"]
        beta2 = group["beta2"]
        lr = group["lr"]
        eps = group["eps"]

        flat_grad = grad.reshape(-1)

        # Empty local shard (e.g. a tiny or uneven DTensor/FSDP2 rank that owns
        # zero elements): there is nothing to update and no period to predict.
        # _iter_gefen_grad_periods already skips these during codebook learning,
        # so state["automatic_period"] is never set for them; mirror that here
        # and no-op instead of raising in the state-init branch below. The guard
        # precedes every state access, so empty shards also no-op on later steps
        # rather than KeyError-ing on the missing vmean.
        if flat_grad.numel() == 0:
            return

        if "step" not in state:
            # grad (hence flat_grad) is already unwrapped to the local shard, and
            # _predict_period_from_grad_sq derives the period from that local
            # tensor. Size the first-step branch and divisibility check off the
            # local shard too (flat_grad.numel()), matching
            # _iter_gefen_grad_periods; mixing in the global DTensor p.numel()
            # breaks on uneven or tiny shards.
            local_numel = flat_grad.numel()
            if "automatic_period" in state:
                automatic_period = state["automatic_period"]
            elif local_numel == 1:
                automatic_period = 1
            elif local_numel > 1:
                automatic_period = self._resolve_automatic_period(param_name, p, grad)
            else:
                raise ValueError(
                    "Automatic partition received an empty parameter {}".format(
                        param_name
                    )
                )

            if local_numel % automatic_period != 0:
                raise ValueError(
                    "Automatic partition period {} does not divide parameter {} with numel {}".format(
                        automatic_period,
                        param_name,
                        local_numel,
                    )
                )

            state["automatic_period"] = automatic_period
            state["step"] = self._new_step_counter(p.device)
            grad_view = self._automatic_view(flat_grad, automatic_period)
            self._init_gefen_state(state, grad_view)

            state["vmean"] = torch.zeros_like(
                grad_view[:, 0:1],
                dtype=torch.float32,
                memory_format=torch.preserve_format,
            )
            state["vmean_step"] = self._new_step_counter(p.device)
            self._print_v_period(param_name, p, state["vmean"])

        automatic_period = state["automatic_period"]
        grad_view = self._automatic_view(flat_grad, automatic_period)

        # Resume compatibility: a checkpoint written with factored_v_2d on has
        # no vmean for 2D params; recreate it if the flag was turned off across
        # the resume, with its own fresh age so the bias correction below
        # treats it as a warm-up EMA rather than a mature one.
        if "vmean" not in state:
            state["vmean"] = torch.zeros_like(
                grad_view[:, 0:1],
                dtype=torch.float32,
                memory_format=torch.preserve_format,
            )
            state["vmean_step"] = self._new_step_counter(p.device)
        elif "vmean_step" not in state:
            # Checkpoint from before the counter existed: its vmean is exactly
            # as old as the param step, so backfill (keeps bc2 bit-identical).
            # A capturable (tensor) step must be CLONED -- assigning the same
            # 0-dim tensor would alias the two counters and double-increment.
            step = state["step"]
            state["vmean_step"] = step.clone() if torch.is_tensor(step) else step

        # If a framework relocated this parameter since its last step (CPU
        # offload paging the shard), the state tensors are stranded on the old
        # device. Migrate them onto the current compute device before any state
        # read below. No-op (single device compare) on the common same-device
        # step, so the CUDA path is unchanged.
        self._colocate_param_state_(state, grad_view.device)

        # Tier-1 uses a separately calibrated predicate for the current full
        # v1/v2 kernels. Both fold the vmean EMA, per-block stepsize math, and
        # weight decay into the update path; v1 is one kernel while v2 splits
        # the work into several launches. Their grad^2 reductions have different
        # orders, so rerouted tensors may differ at sub-ULP scale in vmean/p;
        # momentum indices and magnitudes remain bit-identical. The older
        # partial-update pair keeps its own historical crossover.
        # Per-tensor device/dtype gating: the fused update kernels take a raw
        # CUDA pointer launch, so a CPU-resident param (device_map spill / CPU
        # offload) would crash in the binding, and an fp64 param would be
        # silently downcast to fp32 (the reject-Double dispatch now errors).
        # Route both to the decomposed pure-torch path per-param -- it handles
        # CPU tensors and fp64 natively. grad_view is the unwrapped local shard
        # actually fed to the kernel, so gate on its device/dtype (mirrors the
        # factored path's p.is_cuda guard for the co-located shard).
        fused_step = (
            self._use_fused_gefen_automatic_step()
            and grad_view.is_cuda
            and grad_view.dtype != torch.float64
        )
        # v2-full forms each block's grad^2 sum through floating-point atomics.
        # The result is convergence-equivalent but not replica-exact because
        # independent launches may observe a different accumulation order.
        # Deterministic mode keeps the fused update and selects v1-full, whose
        # fixed thread tree produces bit-identical vmean/parameter writes for
        # identical inputs on a homogeneous architecture.
        route_v2 = (
            False
            if self._deterministic
            else _should_use_v2_full(
                grad_view.shape[0], automatic_period, grad_view.device
            )
        )
        if (
            route_v2
            and self.capturable
            and flat_grad.numel() <= _CAPT_V1_FULL_TINY_NUMEL
        ):
            # Capturable-specific extension: tiny params are launch-bound, so
            # the ONE-kernel v1-full path beats the 4-5-launch v2-full path
            # inside a captured/replayed graph regardless of occupancy. Eager
            # execution already uses the general full-kernel routing above.
            # See _CAPT_V1_FULL_TINY_NUMEL.
            route_v2 = False
        use_full_fused = fused_step and not route_v2
        use_v2_full = fused_step and route_v2

        # Only the non-fused fallback still needs the standalone vmean kernel; the
        # v1-full and v2-full fused paths both compute the vmean EMA in-kernel.
        if not (use_full_fused or use_v2_full):
            self._automatic_vmean_update(state["vmean"], grad_view, beta2)

        # Rows registered with the batched capturable prologue had BOTH
        # counters advanced by its single stacked add_ already this step.
        if "_capt_row" not in state:
            state["step"] += 1
            state["vmean_step"] += 1

        # Under FSDP2 the parameter is a sharded DTensor; every in-place write
        # below (weight decay, fused kernel, non-fused add_) must land on the
        # local shard's storage. DTensor.to_local() returns a view of that
        # storage, so in-place ops propagate back to the param.
        p_local = (
            p.to_local() if (hasattr(p, "to_local") and hasattr(p, "placements")) else p
        )

        if self.capturable and (use_full_fused or use_v2_full):
            # Capturable fused stepping: the per-step-varying scalars flow
            # through the persistent device buffer (refreshed in place with
            # tensor ops -- capture-safe), the kernels ignore the matching
            # host args, and the host bias corrections are skipped entirely
            # (they would either sync on the tensor step counters or freeze
            # in a captured graph).
            step_scalars = self._refresh_capturable_step_scalars(
                state, group, "vmean_step", sqrt_bc2=True
            )
            bias_correction_1 = None
            bias_correction_2 = None
        else:
            step_scalars = None
            bias_correction_1 = 1 - beta1 ** state["step"]
            # bc2 corrects the vmean EMA by its OWN age: on fresh runs and
            # normal resumes vmean_step == step (bit-identical); it differs
            # only when vmean was lazily recreated after a factored->legacy
            # toggle.
            bias_correction_2 = 1 - beta2 ** state["vmean_step"]
            if torch.is_tensor(lr) and not self.capturable:
                # The fused kernel bindings take lr / wd_factor as host doubles,
                # and the non-fused decomposed path multiplies the update by lr
                # on the compute device -- a tensor lr stranded on another device
                # (e.g. a CUDA lr after the parameter was paged to CPU) would
                # break that in-place math. Scalarize through the cached no-sync
                # .item() hoist (_lr_scalar); the value equals lr.item(), so this
                # is bit-identical to passing the 0-dim tensor. Capturable keeps
                # the on-device tensor lr for CUDA-graph replay (never a
                # CPU-offload path).
                lr = self._lr_scalar(group)

        if use_full_fused:
            # K1+K2+K3: vmean EMA, per-block stepsize, and weight decay are all
            # computed inside the fused kernel. weight_decay_factor == 1 - lr*wd
            # mirrors the host p.mul_(1 - lr*wd) pass (an exact identity when
            # wd == 0); under capturable it is computed on device into the
            # scalar buffer instead. The kernel flat-indexes p and requires it
            # contiguous; a non-contiguous shard is updated on a contiguous
            # copy that is copied back to preserve the in-place semantics.
            weight_decay_factor = (
                1.0 if step_scalars is not None else 1.0 - lr * group["weight_decay"]
            )
            if p_local.is_contiguous():
                self._automatic_gefen_fused_full_update(
                    p_local,
                    state,
                    grad_view,
                    beta1,
                    beta2,
                    lr,
                    eps,
                    bias_correction_1,
                    bias_correction_2,
                    weight_decay_factor,
                    step_scalars=step_scalars,
                )
            else:
                p_contig = p_local.contiguous()
                self._automatic_gefen_fused_full_update(
                    p_contig,
                    state,
                    grad_view,
                    beta1,
                    beta2,
                    lr,
                    eps,
                    bias_correction_1,
                    bias_correction_2,
                    weight_decay_factor,
                    step_scalars=step_scalars,
                )
                p_local.copy_(p_contig)
            return

        if use_v2_full:
            # K1+K2+K3 on the occupancy-flexible two-phase path: vmean EMA,
            # per-block stepsize and weight decay are folded into the v2 kernels
            # (see _automatic_gefen_fused_update_v2_full). Same contiguity
            # handling as the v1-full branch.
            weight_decay_factor = (
                1.0 if step_scalars is not None else 1.0 - lr * group["weight_decay"]
            )
            if p_local.is_contiguous():
                self._automatic_gefen_fused_update_v2_full(
                    p_local,
                    state,
                    grad_view,
                    beta1,
                    beta2,
                    lr,
                    eps,
                    bias_correction_1,
                    bias_correction_2,
                    weight_decay_factor,
                    step_scalars=step_scalars,
                )
            else:
                p_contig = p_local.contiguous()
                self._automatic_gefen_fused_update_v2_full(
                    p_contig,
                    state,
                    grad_view,
                    beta1,
                    beta2,
                    lr,
                    eps,
                    bias_correction_1,
                    bias_correction_2,
                    weight_decay_factor,
                    step_scalars=step_scalars,
                )
                p_local.copy_(p_contig)
            return

        if group["weight_decay"] > 0.0:
            p_local.mul_(1 - lr * group["weight_decay"])

        # Reuse two cached [num_blocks, 1] scratch buffers instead of allocating
        # three fresh tensors (sqrt, /scale, final stepsize) every step per param.
        # The arithmetic is preserved bit-for-bit:
        #   h        = sqrt(vmean) / sqrt(bc2) + eps   (sqrt out=, in-place div/add)
        #   stepsize = (1/bc1) / h
        # `scalar / tensor` is implemented by PyTorch as reciprocal(tensor) * scalar
        # (NOT div(scalar, tensor) -- they round differently), so reproduce that
        # exact pair of ops in place; verified bit-identical over random inputs.
        vmean = state["vmean"]
        stepsize = state.get("stepsize")
        h = state.get("_h_buf")
        if (
            stepsize is None
            or stepsize.shape != vmean.shape
            or stepsize.dtype != torch.float32
            or h is None
            or h.shape != vmean.shape
            or h.dtype != torch.float32
        ):
            stepsize = torch.empty_like(vmean)
            h = torch.empty_like(vmean)
            state["stepsize"] = stepsize
            state["_h_buf"] = h
        torch.sqrt(vmean, out=h)
        if torch.is_tensor(bias_correction_2):
            # Capturable: bc2 is a 0-dim device tensor, so take its sqrt on
            # device (math.sqrt would force a host sync).
            h.div_(bias_correction_2.sqrt()).add_(eps)
        else:
            h.div_(math.sqrt(bias_correction_2)).add_(eps)
        torch.reciprocal(h, out=stepsize)
        stepsize.mul_(1.0 / bias_correction_1)

        # Streamed momentum update + parameter apply: row-chunked, bit-for-bit
        # identical to the old whole-tensor form, but it never materializes the
        # full-size fp32 shared_m / update / -update temporaries. This per-param
        # path carries the memory win (it is where the few LARGE params -- esp. the
        # tied embedding -- land; the many small same-shape params are routed to the
        # foreach-merged path in step()).
        self._automatic_momentum_update(state, grad_view, beta1, stepsize, lr, p_local)

    # ------------------------------------------------------------------
    # Batched (foreach / merged-tensor) non-fused step
    #
    # The per-param non-fused path above is correct but launch-bound. The methods
    # below batch parameters that share block geometry into a single merged
    # tensor and run the *same* arithmetic once over all of them, collapsing the
    # per-param kernel launches. They are only used when fused=False, and only for
    # params that are NOT "large" (numel <= GEFEN_MOMENTUM_UPDATE_CHUNK): a large
    # param actually triggers the per-param streaming loop, so it stays on the
    # per-param path (streaming wins over batching for memory/correctness).
    # ------------------------------------------------------------------

    def _nonfused_batch_eligible(self, p: torch.Tensor, grad: torch.Tensor) -> bool:
        # Only the plain (non-DTensor / non-FSDP2), contiguous, CUDA path is
        # batched. DTensor / async-collective gradients and non-contiguous shards
        # keep the per-param path, which already handles their unwrap/copy-back.
        if hasattr(grad, "to_local") or hasattr(grad, "wait"):
            return False
        if hasattr(p, "to_local") and hasattr(p, "placements"):
            return False
        if not grad.is_cuda:
            return False
        if not p.is_contiguous():
            return False
        # A parameter just paged onto this device (CPU-offload) may still carry
        # optimizer state stranded on its previous device; the merged path
        # stacks that state against the gradient and cannot bridge devices.
        # Route it to the per-param path, which re-colocates the state before
        # stepping. Steady-state runs keep their state co-located, so this never
        # rejects a normal CUDA-resident param.
        state = self.state.get(p)
        sentinel = None if state is None else state.get("m_codebook")
        if sentinel is not None and sentinel.device != grad.device:
            return False
        # Large params (those that would actually stream in the per-param momentum
        # update) stay per-param so they get the transient-VRAM win; only the
        # smaller, launch-bound params are batched. This is the routing rule:
        # large -> chunked-streaming path; small/batchable -> merged foreach path.
        if grad.reshape(-1).numel() > GEFEN_MOMENTUM_UPDATE_CHUNK:
            return False
        return True

    def _nonfused_batch_key(self, group, p: torch.Tensor):
        # Params batch together only if every per-block op lines up bit-for-bit:
        # identical local numel + automatic_period (=> identical (nblocks,period)
        # block grid) and identical hyperparameters. lr may be a Tensor (LR
        # schedule); that is unhashable and also implies per-step-varying scalars,
        # so fall back to per-param for it.
        lr = group["lr"]
        if isinstance(lr, torch.Tensor):
            return None
        state = self.state[p]
        period = state.get("automatic_period")
        if period is None:
            return None
        grad = p.grad
        local_numel = grad.reshape(-1).numel()
        # Every merge invariant must be in the key. step: _step_automatic_merged
        # uses one state's bias correction for the whole batch, so params at
        # different step counts (e.g. one whose grad was None on a prior step)
        # must not group. device/dtype: merged buffers only combine compatible
        # tensors.
        return (
            p.device,
            p.dtype,
            grad.device,
            grad.dtype,
            local_numel,
            period,
            state["step"],
            state.get("vmean_step", state["step"]),
            group["beta1"],
            group["beta2"],
            float(lr),
            group["eps"],
            group["weight_decay"],
        )

    def _nonfused_state_ready(self, p: torch.Tensor) -> bool:
        state = self.state.get(p)
        if state is None:
            return False
        return (
            "step" in state
            and "vmean" in state
            and "m_magnitude" in state
            and "m_codebook" in state
            and "automatic_period" in state
        )

    def _dispatch_nonfused_batch(self, items) -> None:
        # items: list of (group, name, p, grad) sharing block geometry + hypers.
        # Sub-batch by an element budget so the merged transient stays bounded:
        # K wide for tiny params (big launch win), narrow for larger params.
        local_numel = items[0][2].grad.reshape(-1).numel()
        budget = GEFEN_NONFUSED_FOREACH_BUDGET
        k = max(1, budget // max(local_numel, 1))
        for start in range(0, len(items), k):
            chunk = items[start : start + k]
            if len(chunk) == 1:
                group, name, p, grad = chunk[0]
                self._step_automatic(group, name, p, grad)
            else:
                self._step_automatic_merged(chunk)

    def _step_automatic_merged(self, items) -> None:
        # All items share (local numel, automatic_period, beta1, beta2, lr, eps,
        # weight_decay). Stack their per-block tensors into one merged
        # (K*nblocks, period) problem and run the existing non-fused arithmetic on
        # it. Each op in that arithmetic is per-block (vmean/magnitude reductions,
        # stepsize, weight decay) or per-element (lerp, codebook search, dequant,
        # the final update), so the merged result is bit-for-bit identical to
        # looping over the params one at a time. The stepsize op sequence here is
        # the algebraic equal of the per-param cached-buffer form in
        # _step_automatic (scalar/tensor == reciprocal(tensor)*scalar), verified
        # bit-identical by the K-series scratch-buffer change.
        group0 = items[0][0]
        beta1 = group0["beta1"]
        beta2 = group0["beta2"]
        lr = group0["lr"]
        eps = group0["eps"]
        weight_decay = group0["weight_decay"]

        first_state = self.state[items[0][2]]
        period = first_state["automatic_period"]
        k = len(items)

        grad_views = []
        vmeans = []
        mags = []
        codebooks = []
        params = []
        for _, _, p, grad in items:
            flat = grad.detach().reshape(-1)
            grad_views.append(flat.view(-1, period))
            state = self.state[p]
            vmeans.append(state["vmean"])
            mags.append(state["m_magnitude"])
            codebooks.append(state["m_codebook"])
            params.append(p)
        nblocks = grad_views[0].shape[0]

        # Merged views. torch.stack copies into fresh contiguous buffers, so the
        # per-param state tensors are untouched until we scatter the results back.
        merged_grad = torch.stack(grad_views).reshape(k * nblocks, period)
        merged_vmean = torch.stack(vmeans).reshape(k * nblocks, 1)
        merged_mag = torch.stack(mags).reshape(k * nblocks, 1)
        merged_codebook = torch.stack(codebooks).reshape(k * nblocks, period)

        # vmean update (same non-fused arithmetic, on the merged block grid).
        automatic_vmean_update(merged_vmean, merged_grad, beta2, use_fused=False)

        for _, _, p, _ in items:
            pstate = self.state[p]
            pstate["step"] += 1
            # Backfill for pre-counter checkpoints (vmean as old as step),
            # then advance the vmean age; the batch key groups by it, so one
            # state's correction is valid for the whole batch.
            if "vmean_step" not in pstate:
                pstate["vmean_step"] = pstate["step"] - 1
            pstate["vmean_step"] += 1
        step_count = first_state["step"]

        if weight_decay > 0.0:
            torch._foreach_mul_(params, 1 - lr * weight_decay)

        bias_correction_1 = 1 - beta1**step_count
        bias_correction_2 = 1 - beta2 ** first_state["vmean_step"]
        h = (merged_vmean.sqrt() / math.sqrt(bias_correction_2)).add_(eps)
        stepsize = (1 / bias_correction_1) / h

        merged_state = {
            "automatic_period": period,
            "m_magnitude": merged_mag,
            "m_codebook": merged_codebook,
        }
        shared_m = self._automatic_momentum_update_merged(
            merged_state, merged_grad, beta1
        )
        update = shared_m * stepsize
        update.mul_(lr)

        # Scatter results back. The momentum update wrote the new magnitude and
        # codebook indices into the merged buffers in place; copy them, the new
        # vmean, and the parameter delta into each param/state with one foreach
        # call apiece.
        update_views = update.reshape(k, nblocks, period)
        param_updates = [update_views[i].reshape(params[i].shape) for i in range(k)]
        torch._foreach_add_(params, param_updates, alpha=-1.0)
        torch._foreach_copy_(
            vmeans, list(merged_vmean.reshape(k, nblocks, 1).unbind(0))
        )
        torch._foreach_copy_(mags, list(merged_mag.reshape(k, nblocks, 1).unbind(0)))
        torch._foreach_copy_(
            codebooks, list(merged_codebook.reshape(k, nblocks, period).unbind(0))
        )

    def _canonical_policy(self):
        return {
            "factored_v_2d": self._factored_v_2d,
            "force_1d_period_one": self._force_1d_period_one,
            "force_2d_period_one": self._force_2d_period_one,
            "period_one_substrings": list(self._period_one_substrings),
            "stochastic_round": self._stochastic_round,
            "codebook_refresh_every": self._codebook_refresh_every,
        }

    def _canonical_common_global_step(self) -> int:
        device_step = self._device_gefen_global_step()
        step = self._gefen_global_step if device_step is None else device_step
        if self.capturable and self._stochastic_round:
            seed_steps = [int(seed.item()) for seed in self._sr_seed_by_device.values()]
            if any(seed_step != step for seed_step in seed_steps):
                raise RuntimeError(
                    "Gefen capturable stochastic-round seeds disagree with the canonical optimizer global step"
                )
        return step

    def _serialized_sharding_manifest(self):
        return [
            self._serialized_canonical_shard(shard)
            for shard in self._gefen_sharding_manifest.shards
        ]

    def _canonical_live_entries(self):
        entries = {}
        for group in self.param_groups:
            options = clone_canonical_value(
                self._canonical_group_options_value(group),
                path="parameter group options",
            )
            for compatibility_name, parameter in self._iter_group_params_with_names(
                group
            ):
                shard = self._gefen_shard_bindings[parameter]
                entries[shard.parameter.fqn] = (
                    parameter,
                    shard,
                    str(compatibility_name),
                    options,
                )
        return entries

    @staticmethod
    def _canonical_value_token(value):
        if torch.is_tensor(value):
            raw = (
                value.detach()
                .to(device="cpu")
                .resolve_conj()
                .resolve_neg()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
            try:
                version = value._version
            except RuntimeError:
                version = None
            return (
                "tensor",
                id(value),
                version,
                hashlib.sha256(raw).digest(),
                str(value.device),
                str(value.dtype),
                tuple(value.shape),
            )
        if type(value) is dict:
            return (
                "dict",
                tuple(
                    (key, Gefen._canonical_value_token(value[key]))
                    for key in sorted(value, key=repr)
                ),
            )
        if type(value) in {list, tuple}:
            return (
                type(value).__name__,
                tuple(Gefen._canonical_value_token(item) for item in value),
            )
        try:
            hash(value)
            token = value
        except TypeError:
            token = (id(value), repr(value))
        return (type(value).__name__, token)

    def _canonical_import_live_token(self):
        from gefen.portable_runtime import _parameter_storage_token

        groups = tuple(
            (
                id(group),
                tuple(id(parameter) for parameter in group["params"]),
                tuple(
                    str(name) for name, _ in self._iter_group_params_with_names(group)
                ),
                self._canonical_value_token(self._canonical_group_options_value(group)),
            )
            for group in self.param_groups
        )
        states = tuple(
            (
                id(parameter),
                id(self.state.get(parameter)),
                self._canonical_value_token(self.state.get(parameter, {})),
            )
            for group in self.param_groups
            for parameter in group["params"]
        )
        return (
            id(self.param_groups),
            id(self.state),
            id(self.defaults),
            self._canonical_value_token(self.defaults),
            groups,
            states,
            self._gefen_global_step,
            self._device_gefen_global_step(),
            self._canonical_value_token(self._gefen_codebook),
            self._canonical_value_token(self._gefen_global_step_by_device),
            self._canonical_value_token(self._sr_seed_by_device),
            self._canonical_value_token(self._canonical_policy()),
            self._deterministic,
            self.capturable,
            self.fused,
            self.verbose,
            self._fused_build_ok,
            id(self._gefen_codebook_process_group),
            self._canonical_value_token(self._serialized_codebook_scope()),
            id(self._gefen_sharding_manifest),
            id(self._gefen_logical_slots),
            tuple(
                (
                    id(logical_slot),
                    logical_slot.group_index,
                    logical_slot.original_slot_index,
                    logical_slot.compatibility_name,
                    self._canonical_value_token(
                        self._serialized_canonical_shard(logical_slot.shard)
                    ),
                )
                for logical_slot in self._gefen_logical_slots
            ),
            tuple(
                (
                    id(parameter),
                    shard.sort_key,
                    # Mirror the portable live token: a prepared import must go
                    # stale when a locally bound parameter's storage is
                    # retargeted or mutated between prepare and commit.
                    None
                    if parameter is None
                    else _parameter_storage_token(parameter),
                )
                for parameter, shard in self._gefen_local_shard_bindings
            ),
        )

    def _canonical_parameter_state_matches_variant(
        self, shard, group_options, parameter_state, state_layout
    ) -> bool:
        carries_momentum = any(
            key in parameter_state for key in ("m_codebook", "m_magnitude")
        )
        carries_normuon = any(
            key in parameter_state for key in ("normuon_v", "normuon_step")
        )
        if bool(group_options.get("normuon", False)):
            if carries_momentum and not carries_normuon:
                return False
        elif carries_normuon:
            return False
        fields = frozenset({"name", *parameter_state})
        parameter_rank = len(shard.parameter.global_shape)
        sharded_mode = group_options.get("sharded_mode")
        for variant in state_layout.parameter_variants:
            if frozenset(variant.fields) != fields:
                continue
            if shard.layout not in variant.layouts:
                continue
            if variant.role is not ParameterStateRole.ANY:
                continue
            if variant.sharded_mode != sharded_mode:
                continue
            if (
                variant.parameter_ranks is not None
                and parameter_rank not in variant.parameter_ranks
            ):
                continue
            if parameter_rank in variant.excluded_parameter_ranks:
                continue
            return True
        return False

    def _canonical_period_must_be_one(self, parameter, shard) -> bool:
        if self._force_1d_period_one and parameter.ndim == 1:
            return True
        if self._force_2d_period_one and parameter.ndim == 2:
            return True
        fqn = shard.parameter.fqn.lower()
        return any(substring in fqn for substring in self._period_one_substrings)

    def _validate_canonical_parameter_semantics(
        self,
        parameter,
        shard,
        group_options,
        parameter_state,
        state_layout,
        global_step,
    ) -> None:
        carries_momentum = any(
            key in parameter_state for key in ("m_codebook", "m_magnitude")
        )
        carries_normuon = any(
            key in parameter_state for key in ("normuon_v", "normuon_step")
        )
        counters = {}
        for counter_key in (
            "step",
            "vmean_step",
            "factored_step",
            "normuon_step",
        ):
            if counter_key not in parameter_state:
                continue
            counter = self._validate_rank_local_counter(
                counter_key, parameter_state[counter_key]
            )
            counters[counter_key] = counter
            if counter > global_step:
                raise ValueError(
                    "Gefen canonical parameter counter {} exceeds the optimizer global step".format(
                        counter_key
                    )
                )
        if "step" in counters and any(
            counters[counter_key] > counters["step"]
            for counter_key in (
                "vmean_step",
                "factored_step",
                "normuon_step",
            )
            if counter_key in counters
        ):
            raise ValueError(
                "Gefen canonical secondary counter exceeds the parameter step"
            )
        period = parameter_state.get("automatic_period")
        if (
            period is not None
            and self._canonical_period_must_be_one(parameter, shard)
            and (type(period) is not int or period != 1)
        ):
            raise ValueError(
                "Gefen canonical automatic period violates period-one policy"
            )
        if bool(group_options.get("normuon", False)):
            if carries_momentum and not carries_normuon:
                raise ValueError(
                    "Gefen canonical initialized NorMuon state is incomplete"
                )
        elif carries_normuon:
            raise ValueError(
                "Gefen canonical NorMuon state is invalid for the target policy"
            )
        if not self._canonical_parameter_state_matches_variant(
            shard, group_options, parameter_state, state_layout
        ):
            raise ValueError(
                "Gefen canonical parameter state does not match a declared state variant"
            )

    @staticmethod
    def _assert_canonical_state_outside_cuda_capture(operation) -> None:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "canonical state {} cannot run during CUDA capture".format(operation)
            )

    def _assert_canonical_import_target_safe(self) -> None:
        if not self.capturable:
            return
        device_step = self._device_gefen_global_step()
        initialized_state = any(
            any(key != "name" for key in self.state.get(parameter, {}))
            for group in self.param_groups
            for parameter in group["params"]
        )
        if (
            self._capt_stacks is not None
            or self._gefen_codebook is not None
            or self._gefen_global_step != 0
            or (device_step is not None and device_step != 0)
            or initialized_state
        ):
            raise RuntimeError(
                "canonical import into a capturable optimizer requires a fresh "
                "target before device state or CUDA graphs are initialized"
            )

    def export_canonical_state(self):
        """Export an exact-binding, device-neutral local state fragment."""

        self._assert_finalized_binding_layout(full=True)
        self._assert_canonical_state_outside_cuda_capture("export")
        if not self._canonical_state_layouts():
            raise RuntimeError(
                "canonical state export requires a supported finalized identity layout and primitive algorithm policy"
            )
        entries = self._canonical_live_entries()
        parameters = {}
        for fqn in sorted(entries):
            parameter, shard, compatibility_name, options = entries[fqn]
            state = {}
            for key, value in self.state.get(parameter, {}).items():
                if key in _CANONICAL_DERIVED_PARAMETER_STATE_KEYS:
                    continue
                if key == "name":
                    continue
                if key not in _CANONICAL_PARAMETER_STATE_KEYS:
                    raise RuntimeError(
                        "canonical state export found undeclared parameter state key {!r}".format(
                            key
                        )
                    )
                state[key] = clone_canonical_value(
                    value, path="parameters.{}.state.{}".format(fqn, key)
                )
            parameters[fqn] = {
                "compatibility_name": compatibility_name,
                "shard": self._serialized_canonical_shard(shard),
                "group_options": clone_canonical_value(
                    options, path="parameters.{}.group_options".format(fqn)
                ),
                "state": state,
            }
        return {
            "format": "gefen.bound_state",
            "format_version": CANONICAL_STATE_FORMAT_VERSION,
            "coverage": "local_optimizer_fragment",
            "implementation": self.optimizer_contract().implementation,
            "policy": clone_canonical_value(self._canonical_policy(), path="policy"),
            "common": {
                "gefen_global_step": self._canonical_common_global_step(),
                "gefen_codebook": clone_canonical_value(
                    self._gefen_codebook, path="common.gefen_codebook"
                ),
                "gefen_deterministic": self._deterministic,
                "gefen_codebook_scope": clone_canonical_value(
                    self._serialized_codebook_scope(),
                    path="common.gefen_codebook_scope",
                ),
            },
            "manifest": clone_canonical_value(
                self._serialized_sharding_manifest(), path="manifest"
            ),
            "parameters": parameters,
        }

    def _normalize_canonical_state_import(self, state):
        state = clone_canonical_value(state, path="canonical state")
        expected_top = {
            "format",
            "format_version",
            "coverage",
            "implementation",
            "policy",
            "common",
            "manifest",
            "parameters",
        }
        if type(state) is not dict or set(state) != expected_top:
            raise ValueError("Gefen canonical state has an invalid top-level schema")
        if state["format"] != "gefen.bound_state":
            raise ValueError("Unsupported Gefen canonical state format")
        if (
            type(state["format_version"]) is not int
            or state["format_version"] != CANONICAL_STATE_FORMAT_VERSION
        ):
            raise ValueError(
                "Unsupported Gefen canonical state format_version: {}".format(
                    state["format_version"]
                )
            )
        if state["coverage"] != "local_optimizer_fragment":
            raise ValueError("Unsupported Gefen canonical state coverage")
        implementation = self.optimizer_contract().implementation
        if state["implementation"] != implementation:
            raise ValueError(
                "Gefen canonical state implementation does not match the target"
            )
        live_policy = clone_canonical_value(
            self._canonical_policy(), path="live policy"
        )
        if not canonical_values_equal(state["policy"], live_policy):
            raise ValueError(
                "Gefen canonical state algorithm policy does not match the target"
            )

        common = state["common"]
        if type(common) is not dict or set(common) != {
            "gefen_global_step",
            "gefen_codebook",
            "gefen_deterministic",
            "gefen_codebook_scope",
        }:
            raise ValueError("Gefen canonical common state has an invalid schema")
        if (
            type(common["gefen_global_step"]) is not int
            or common["gefen_global_step"] < 0
        ):
            raise ValueError(
                "Gefen canonical global step must be a nonnegative integer"
            )
        if type(common["gefen_deterministic"]) is not bool:
            raise ValueError("Gefen canonical deterministic policy must be a bool")
        normalized_scope = self._normalize_serialized_codebook_scope(
            common["gefen_codebook_scope"]
        )
        if normalized_scope != self._serialized_codebook_scope():
            raise ValueError(
                "Gefen canonical codebook scope does not match the live binding"
            )
        common["gefen_codebook_scope"] = normalized_scope

        manifest = state["manifest"]
        if type(manifest) is not list:
            raise ValueError("Gefen canonical manifest must be a list")
        normalized_manifest = [
            self._normalize_serialized_canonical_shard(record) for record in manifest
        ]
        if normalized_manifest != self._serialized_sharding_manifest():
            raise ValueError(
                "Gefen canonical manifest does not match the finalized target"
            )
        state["manifest"] = normalized_manifest

        parameters = state["parameters"]
        if type(parameters) is not dict:
            raise ValueError("Gefen canonical parameters must be an FQN mapping")
        live_entries = self._canonical_live_entries()
        state_layout = self._canonical_state_variant_layout()
        if set(parameters) != set(live_entries):
            raise ValueError(
                "Gefen canonical parameter FQNs do not match the finalized target"
            )
        for fqn, record in parameters.items():
            if type(record) is not dict or set(record) != {
                "compatibility_name",
                "shard",
                "group_options",
                "state",
            }:
                raise ValueError(
                    "Gefen canonical parameter {!r} has an invalid schema".format(fqn)
                )
            if type(record["compatibility_name"]) is not str:
                raise ValueError("Gefen canonical compatibility names must be strings")
            parameter, shard, _, live_options = live_entries[fqn]
            normalized_shard = self._normalize_serialized_canonical_shard(
                record["shard"]
            )
            if normalized_shard != self._serialized_canonical_shard(shard):
                raise ValueError(
                    "Gefen canonical parameter shard does not match the live binding"
                )
            record["shard"] = normalized_shard
            if not canonical_values_equal(record["group_options"], live_options):
                raise ValueError(
                    "Gefen canonical parameter-group options do not match the target"
                )
            parameter_state = record["state"]
            if type(parameter_state) is not dict or any(
                key == "name" or key not in _CANONICAL_PARAMETER_STATE_KEYS
                for key in parameter_state
            ):
                raise ValueError(
                    "Gefen canonical parameter state has undeclared fields"
                )
            self._validate_canonical_parameter_semantics(
                parameter,
                shard,
                live_options,
                parameter_state,
                state_layout,
                common["gefen_global_step"],
            )
            record["state"] = clone_canonical_value(
                parameter_state, path="parameters.{}.state".format(fqn)
            )
            del parameter
        return state

    def _canonical_native_state_dict(self, canonical_state):
        native = self._base_state_dict_without_hooks()
        native["state"] = {}
        live_entries = self._canonical_live_entries()
        for saved_group, live_group in zip(native["param_groups"], self.param_groups):
            for parameter_id, parameter in zip(
                saved_group["params"], live_group["params"]
            ):
                shard = self._gefen_shard_bindings[parameter]
                record = canonical_state["parameters"][shard.parameter.fqn]
                parameter_state = {
                    "name": self._param_name(parameter),
                    **clone_canonical_value(
                        record["state"],
                        path="parameters.{}.state".format(shard.parameter.fqn),
                    ),
                }
                native["state"][parameter_id] = parameter_state

        common = canonical_state["common"]
        native["gefen_global_step"] = common["gefen_global_step"]
        native["gefen_codebook"] = common["gefen_codebook"]
        native["gefen_deterministic"] = common["gefen_deterministic"]
        scope = common["gefen_codebook_scope"]
        if scope is not None:
            native["gefen_codebook_scope"] = scope
        else:
            native.pop("gefen_codebook_scope", None)
        local_shards = self._serialized_native_local_shards()
        if local_shards is not None:
            native["gefen_native_local_shards"] = local_shards
        else:
            native.pop("gefen_native_local_shards", None)
        metadata = {
            "format_version": (
                _SCOPED_NATIVE_METADATA_VERSION if scope is not None else 1
            ),
            "global_step": common["gefen_global_step"],
            "codebook": common["gefen_codebook"],
            "deterministic": common["gefen_deterministic"],
            "device_anchor": self._checkpoint_device_anchor(),
        }
        if scope is not None:
            metadata["codebook_scope"] = scope
        if local_shards is not None:
            metadata["native_local_shards"] = local_shards
        for group in native["param_groups"]:
            group["_gefen_checkpoint_metadata"] = metadata
        del live_entries
        return native

    def _preserve_canonical_target_configuration(self, staged) -> None:
        staged.defaults = self.defaults.copy()
        staged.param_groups = self.param_groups

    def prepare_canonical_state_import(self, state):
        """Validate and stage a canonical local import without live mutation."""

        self._assert_finalized_binding_layout(full=True)
        self._assert_canonical_state_outside_cuda_capture("import preparation")
        self._assert_canonical_import_target_safe()
        if not self._canonical_state_layouts():
            raise RuntimeError(
                "canonical state import requires a supported finalized identity layout"
            )
        normalized = self._normalize_canonical_state_import(state)
        live_token = self._canonical_import_live_token()
        staged = self._stage_load_state_dict(
            self._canonical_native_state_dict(normalized)
        )
        self._preserve_canonical_target_configuration(staged)
        return make_prepared_canonical_state_import(self, live_token, staged)

    def commit_canonical_state_import(self, prepared) -> None:
        """Commit one still-current prepared canonical import exactly once."""

        if not isinstance(prepared, PreparedCanonicalStateImport):
            raise TypeError(
                "prepared must be a PreparedCanonicalStateImport from this optimizer"
            )
        if prepared._optimizer is not self:
            raise ValueError("prepared canonical state belongs to another optimizer")
        if prepared._consumed:
            raise RuntimeError("prepared canonical state import was already consumed")
        self._assert_finalized_binding_layout(full=True)
        self._assert_canonical_state_outside_cuda_capture("import commit")
        self._assert_canonical_import_target_safe()
        if prepared._live_token != self._canonical_import_live_token():
            raise RuntimeError(
                "live optimizer state changed after canonical import preparation"
            )
        prepared._consumed = True
        self._commit_staged_load_state_dict(prepared._staged)

    def import_canonical_state(self, state) -> None:
        """Atomically prepare and commit an exact-binding canonical fragment."""

        self.commit_canonical_state_import(self.prepare_canonical_state_import(state))

    def export_portable_state(
        self,
        *,
        checkpoint_process_group,
        transaction_id,
        limits,
    ):
        """Collectively export exact period-one state by logical identity."""

        from gefen.portable_runtime import _export_portable_state

        return _export_portable_state(
            self,
            checkpoint_process_group=checkpoint_process_group,
            transaction_id=transaction_id,
            limits=limits,
        )

    def import_portable_state(
        self,
        state,
        *,
        checkpoint_process_group,
        transaction_id,
        limits,
    ) -> None:
        """Collectively stage and atomically publish exact portable state."""

        from gefen.portable_runtime import _import_portable_state

        _import_portable_state(
            self,
            state,
            checkpoint_process_group=checkpoint_process_group,
            transaction_id=transaction_id,
            limits=limits,
        )

    canonical_state_dict = export_canonical_state
    load_canonical_state_dict = import_canonical_state

    def state_dict(self):
        """Run optimizer state-dict hooks around Gefen's complete schema."""

        self._assert_finalized_binding_layout(full=True)
        for pre_hook in self._optimizer_state_dict_pre_hooks.values():
            pre_hook(self)
        self._assert_finalized_binding_layout(full=True)
        state_dict = self._state_dict_impl()
        for post_hook in self._optimizer_state_dict_post_hooks.values():
            hook_result = post_hook(self, state_dict)
            if hook_result is not None:
                state_dict = hook_result
        return state_dict

    def _base_state_dict_without_hooks(self):
        """Call ``Optimizer.state_dict`` without double-firing public hooks."""

        pre_hooks = self._optimizer_state_dict_pre_hooks
        post_hooks = self._optimizer_state_dict_post_hooks
        self._optimizer_state_dict_pre_hooks = OrderedDict()
        self._optimizer_state_dict_post_hooks = OrderedDict()
        try:
            return super().state_dict()
        finally:
            self._optimizer_state_dict_pre_hooks = pre_hooks
            self._optimizer_state_dict_post_hooks = post_hooks

    def _uses_rank_local_sharded_state(self) -> bool:
        """Whether this topology stores optimizer state in rank-local geometry."""

        for group in self.param_groups:
            mode = group.get("sharded_mode")
            if mode not in (None, "approx"):
                continue
            for p in group["params"]:
                if (
                    hasattr(p, "to_local")
                    and hasattr(p, "placements")
                    and hasattr(p, "device_mesh")
                ):
                    return True
        return False

    @staticmethod
    def _is_dtensor_parameter(param) -> bool:
        return (
            hasattr(param, "to_local")
            and hasattr(param, "placements")
            and hasattr(param, "device_mesh")
        )

    @staticmethod
    def _placement_checkpoint_signature(placement):
        """Encode a DTensor placement structurally rather than through repr()."""

        encoded = {
            "type": type(placement).__name__,
            "module": type(placement).__module__,
        }
        for attribute in ("dim", "split_factor"):
            value = getattr(placement, attribute, None)
            if value is not None:
                encoded[attribute] = int(value)
        reduce_op = getattr(placement, "reduce_op", None)
        if reduce_op is not None:
            encoded["reduce_op"] = str(reduce_op)
        return encoded

    @staticmethod
    def _device_mesh_checkpoint_signature(mesh):
        mesh_ranks = mesh.mesh.detach().cpu().reshape(-1).tolist()
        return {
            "device_type": str(mesh.device_type),
            "shape": list(mesh.shape),
            "dim_names": list(mesh.mesh_dim_names or ()),
            "ranks": [int(item) for item in mesh_ranks],
        }

    def _rank_local_checkpoint_context(self, *, fail: bool = True):
        """Return the deliberately narrow process-group contract for the adapter.

        Rank-local full-state encoding currently supports exactly one 1-D
        DeviceMesh spanning the default process group. Subgroups, pipeline-local
        optimizers, and multidimensional meshes need a native sharded DCP format;
        silently routing them through the global collectives below can deadlock.
        """

        if not self._uses_rank_local_sharded_state():
            return None

        def reject(message):
            if fail:
                raise RuntimeError(message)
            return None

        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return reject(
                "Gefen rank-local DTensor checkpointing requires an initialized default distributed process group"
            )

        import torch.distributed as dist

        dtensor_params = [
            p
            for group in self.param_groups
            for p in group["params"]
            if self._is_dtensor_parameter(p)
        ]
        if not dtensor_params:
            return None

        first_mesh = dtensor_params[0].device_mesh
        first_mesh_signature = self._device_mesh_checkpoint_signature(first_mesh)
        if len(first_mesh_signature["shape"]) != 1:
            return reject(
                "Gefen rank-local optimizer checkpoints support only one 1-D "
                "DeviceMesh spanning the default process group"
            )
        for param in dtensor_params[1:]:
            mesh_signature = self._device_mesh_checkpoint_signature(param.device_mesh)
            if mesh_signature != first_mesh_signature:
                return reject(
                    "Gefen rank-local optimizer checkpoints require every DTensor "
                    "parameter to use the same 1-D DeviceMesh"
                )

        world = dist.get_world_size()
        world_ranks = list(range(world))
        if (
            first_mesh_signature["shape"] != [world]
            or sorted(first_mesh_signature["ranks"]) != world_ranks
        ):
            return reject(
                "Gefen rank-local optimizer checkpoints do not support DeviceMesh "
                "subgroups; the one-dimensional mesh must span the default world"
            )

        try:
            mesh_group = first_mesh.get_group()
            mesh_group_ranks = [
                dist.get_global_rank(mesh_group, group_rank)
                for group_rank in range(dist.get_world_size(mesh_group))
            ]
        except Exception as exc:
            if fail:
                raise RuntimeError(
                    "Gefen could not resolve the rank-local DeviceMesh process group"
                ) from exc
            return None
        if sorted(mesh_group_ranks) != world_ranks:
            return reject(
                "Gefen rank-local optimizer checkpoints require the DeviceMesh "
                "process group to contain exactly the default-world ranks"
            )

        coordinate = first_mesh.get_coordinate()
        if coordinate is None:
            return reject(
                "The current rank is not a member of Gefen's checkpoint DeviceMesh"
            )
        return {
            "world_size": world,
            "global_rank": dist.get_rank(),
            "group_rank": dist.get_group_rank(mesh_group, dist.get_rank()),
            "world_ranks": world_ranks,
            "mesh": first_mesh_signature,
            "coordinate": [int(item) for item in coordinate],
        }

    def _install_rank_local_checkpoint_schema(self) -> None:
        """Expose private live keys required by PyTorch's flat OSD unflattener."""

        for group in self.param_groups:
            group.pop("_gefen_checkpoint_metadata", None)
            for param in group["params"]:
                pstate = self.state.get(param)
                if pstate is not None:
                    for key in tuple(pstate):
                        if key.startswith(_RANK_LOCAL_PAYLOAD_KEY_PREFIX):
                            pstate.pop(key)
                    pstate.pop(_RANK_LOCAL_MEMBER_KEY, None)

        context = self._rank_local_checkpoint_context(fail=False)
        if context is None:
            return
        params = [param for group in self.param_groups for param in group["params"]]
        if not params:
            return
        for global_rank in context["world_ranks"]:
            self.state[params[0]][_rank_local_payload_key(global_rank)] = torch.zeros(
                1, dtype=torch.uint8
            )
        for param in params[1:]:
            self.state[param][_RANK_LOCAL_MEMBER_KEY] = True
        placeholder = {
            "format_version": _RANK_LOCAL_METADATA_VERSION,
            "global_step": self._gefen_global_step,
            "codebook": None,
            "deterministic": self._deterministic,
            "device_anchor": self._checkpoint_device_anchor(),
            "rank_local_sharded_state": {"format": _RANK_LOCAL_FORMAT},
        }
        for group in self.param_groups:
            group["_gefen_checkpoint_metadata"] = placeholder

    def _rank_local_sharded_signature(self, context=None):
        if context is None:
            context = self._rank_local_checkpoint_context()
        signature = []
        for group_index, group in enumerate(self.param_groups):
            for param_index, (name, p) in enumerate(
                self._iter_group_params_with_names(group)
            ):
                identifier = "group_{}_param_{}_{}".format(
                    group_index, param_index, str(name).lower()
                )
                if not self._is_dtensor_parameter(p):
                    signature.append(
                        {
                            "identifier": identifier,
                            "group": group_index,
                            "param": param_index,
                            "name": str(name).lower(),
                            "sharded": False,
                            "shape": list(p.shape),
                            "local_shape": list(p.shape),
                            "dtype": str(p.dtype),
                            "requires_grad": bool(p.requires_grad),
                            "sharded_mode": group.get("sharded_mode"),
                        }
                    )
                    continue
                local = p.to_local()
                if hasattr(local, "wait"):
                    local = local.wait()
                mesh = p.device_mesh
                coordinate = mesh.get_coordinate()
                signature.append(
                    {
                        "identifier": identifier,
                        "group": group_index,
                        "param": param_index,
                        "name": str(name).lower(),
                        "sharded": True,
                        "shape": list(p.shape),
                        "local_shape": list(local.shape),
                        "dtype": str(p.dtype),
                        "local_dtype": str(local.dtype),
                        "requires_grad": bool(p.requires_grad),
                        "placements": [
                            self._placement_checkpoint_signature(item)
                            for item in p.placements
                        ],
                        "mesh": self._device_mesh_checkpoint_signature(mesh),
                        "coordinate": None if coordinate is None else list(coordinate),
                        "sharded_mode": group.get("sharded_mode"),
                    }
                )
        return signature

    @staticmethod
    def _rank_local_parameter_manifest(signature):
        keys = (
            "identifier",
            "group",
            "param",
            "name",
            "sharded",
            "shape",
            "dtype",
            "requires_grad",
            "sharded_mode",
        )
        return [{key: item.get(key) for key in keys} for item in signature]

    def _checkpoint_device_anchor(self) -> torch.Tensor:
        """One byte that lets DCP locate a fresh lazy optimizer's device."""

        if self._uses_rank_local_sharded_state():
            # The collective rank-indexed adapter deliberately stages every
            # payload on CPU to avoid multiplying live VRAM by world size at
            # checkpoint time. DCP requires every tensor in the fresh local
            # schema to name one device, so its anchor must be CPU as well.
            return torch.zeros(1, dtype=torch.uint8)
        for group in self.param_groups:
            for p in group["params"]:
                local = p.to_local() if hasattr(p, "to_local") else p
                if hasattr(local, "wait"):
                    local = local.wait()
                return torch.zeros(1, dtype=torch.uint8, device=local.device)
        return torch.zeros(1, dtype=torch.uint8)

    @classmethod
    def _pack_checkpoint_payload(cls, value, tensors):
        if torch.is_tensor(value):
            tensor = value.to_local() if hasattr(value, "to_local") else value
            if hasattr(tensor, "wait"):
                tensor = tensor.wait()
            tensor = tensor.detach().contiguous()
            index = len(tensors)
            tensors.append(tensor)
            return ("tensor", index)
        if isinstance(value, dict):
            return (
                "dict",
                [
                    (key, cls._pack_checkpoint_payload(item, tensors))
                    for key, item in value.items()
                ],
            )
        if isinstance(value, list):
            return (
                "list",
                [cls._pack_checkpoint_payload(item, tensors) for item in value],
            )
        if isinstance(value, tuple):
            return (
                "tuple",
                [cls._pack_checkpoint_payload(item, tensors) for item in value],
            )
        return ("object", value)

    @classmethod
    def _unpack_checkpoint_payload(cls, packed, tensors):
        kind, value = packed
        if kind == "tensor":
            return tensors[value]
        if kind == "dict":
            return {
                key: cls._unpack_checkpoint_payload(item, tensors)
                for key, item in value
            }
        if kind == "list":
            return [cls._unpack_checkpoint_payload(item, tensors) for item in value]
        if kind == "tuple":
            return tuple(
                cls._unpack_checkpoint_payload(item, tensors) for item in value
            )
        if kind == "object":
            return value
        raise ValueError("Unknown Gefen checkpoint payload kind: {}".format(kind))

    @staticmethod
    def _serialize_rank_local_payload(payload) -> torch.Tensor:
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        return torch.frombuffer(bytearray(buffer.getvalue()), dtype=torch.uint8).clone()

    @staticmethod
    def _deserialize_rank_local_payload(payload: torch.Tensor):
        if (
            not torch.is_tensor(payload)
            or payload.dtype != torch.uint8
            or payload.dim() != 1
        ):
            raise ValueError(
                "Gefen rank-local checkpoint payload must be a 1-D uint8 tensor"
            )
        buffer = io.BytesIO(payload.detach().cpu().contiguous().numpy().tobytes())
        try:
            return torch.load(buffer, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError("Gefen rank-local checkpoint payload is invalid") from exc

    def _broadcast_checkpoint_payload(self, payload, src, group=None):
        import torch.distributed as dist

        rank = dist.get_rank(group)
        tensors = []
        if rank == src:
            packed = self._pack_checkpoint_payload(payload, tensors)
            specs = [(tuple(tensor.shape), tensor.dtype) for tensor in tensors]
            header = [(packed, specs)]
        else:
            header = [None]
        dist.broadcast_object_list(header, src=src, group=group)
        packed, specs = header[0]

        backend = str(dist.get_backend(group)).lower()
        comm_device = (
            torch.device("cuda", torch.cuda.current_device())
            if "nccl" in backend
            else torch.device("cpu")
        )
        received = []
        for index, (shape, dtype) in enumerate(specs):
            if rank == src:
                tensor = tensors[index].to(device=comm_device).contiguous()
            else:
                tensor = torch.empty(shape, dtype=dtype, device=comm_device)
            dist.broadcast(tensor, src=src, group=group)
            received.append(tensor.detach().cpu().clone())
        return self._unpack_checkpoint_payload(packed, received)

    def _consolidate_rank_local_sharded_state(
        self, state_dict, checkpoint_metadata
    ) -> None:
        """Collect rank-local DTensor state for exact same-topology resume.

        Plain Gefen and ``GefenMuon(sharded_mode='approx')`` learn rank-local
        codebooks and block geometry. PyTorch DCP treats their ordinary local
        tensors as replicated and otherwise keeps rank 0 silently. Make every
        rank return the same rank-indexed payload so generic DCP preserves all
        shards; loading selects the payload for the current rank and rejects a
        topology/world-size change.
        """

        if not self._uses_rank_local_sharded_state():
            return

        import torch.distributed as dist

        context = self._rank_local_checkpoint_context()
        world = context["world_size"]
        rank = context["global_rank"]
        saved_ids = list(state_dict["state"])
        local_signature = self._rank_local_sharded_signature(context)
        local_manifest = self._rank_local_parameter_manifest(local_signature)
        local_control = {
            "saved_ids": saved_ids,
            "manifest": local_manifest,
            "signature": local_signature,
            "global_step": self._gefen_global_step,
            "deterministic": self._deterministic,
        }
        controls = [None] * world
        dist.all_gather_object(controls, local_control)
        if any(control["saved_ids"] != saved_ids for control in controls):
            raise RuntimeError(
                "Gefen DTensor checkpoint requires identical optimizer parameter "
                "ordering on every rank; got state keys {}".format(
                    [control["saved_ids"] for control in controls]
                )
            )
        if any(control["manifest"] != local_manifest for control in controls):
            raise RuntimeError(
                "Gefen DTensor checkpoint parameter identifiers, names, dtypes, or global shapes differ across ranks"
            )
        global_steps = [control["global_step"] for control in controls]
        if (
            any(type(step) is not int for step in global_steps)
            or len(set(global_steps)) != 1
        ):
            raise RuntimeError(
                "Gefen rank-local checkpoint global_step differs across ranks: {}".format(
                    global_steps
                )
            )
        deterministic_values = [control["deterministic"] for control in controls]
        if (
            any(type(value) is not bool for value in deterministic_values)
            or len(set(deterministic_values)) != 1
        ):
            raise RuntimeError(
                "Gefen rank-local checkpoint deterministic policy differs across ranks: {}".format(
                    deterministic_values
                )
            )

        signatures = {
            str(global_rank): controls[global_rank]["signature"]
            for global_rank in context["world_ranks"]
        }
        local_payload = {
            "format": _RANK_LOCAL_FORMAT,
            "global_rank": rank,
            "group_rank": context["group_rank"],
            "world_size": world,
            "world_ranks": context["world_ranks"],
            "mesh": context["mesh"],
            "signature": local_signature,
            "parameter_manifest": local_manifest,
            "global_step": global_steps[0],
            "deterministic": deterministic_values[0],
            "states": [state_dict["state"][saved_id] for saved_id in saved_ids],
            "codebook": self._gefen_codebook,
        }
        # Serialize locally before transfer. The prior tensor-by-tensor gather
        # retained every rank's decoded state and then a second serialized copy
        # on every process. Exchanging one opaque byte tensor per rank keeps CPU
        # peak near one global serialized optimizer state plus this rank's live
        # local state. Share serialization status first so a deterministic local
        # schema/serialization failure makes every rank raise before broadcasts.
        try:
            local_serialized = self._serialize_rank_local_payload(local_payload)
            serialization_status = {"ok": True, "error": None}
        except Exception as exc:
            local_serialized = None
            serialization_status = {
                "ok": False,
                "error": "{}: {}".format(type(exc).__name__, exc),
            }
        serialization_statuses = [None] * world
        dist.all_gather_object(serialization_statuses, serialization_status)
        if any(not item.get("ok", False) for item in serialization_statuses):
            raise RuntimeError(
                "Gefen rank-local checkpoint payload serialization failed across ranks: {}".format(
                    serialization_statuses
                )
            )
        serialized_payloads = {
            str(src): self._broadcast_checkpoint_payload(
                local_serialized if rank == src else None, src
            )
            for src in context["world_ranks"]
        }
        state_dict["state"] = {}
        carrier_state = {
            # PyTorch 2.5's flat optimizer-state loader requires every key
            # present in the fresh live state to have a direct tensor/int/float
            # leaf in the flattened checkpoint. The real stable name remains
            # inside the validated payload; this integer is only an outer
            # transport-schema sentinel.
            "name": 0,
        }
        carrier_state.update(
            {
                _rank_local_payload_key(global_rank): serialized_payloads[
                    str(global_rank)
                ]
                for global_rank in context["world_ranks"]
            }
        )
        for index, saved_id in enumerate(saved_ids):
            state_dict["state"][saved_id] = (
                dict(carrier_state)
                if index == 0
                else {"name": 0, _RANK_LOCAL_MEMBER_KEY: True}
            )
        marker = {
            "format": _RANK_LOCAL_FORMAT,
            "world_size": world,
            "world_ranks": context["world_ranks"],
            "mesh": context["mesh"],
            "signatures": signatures,
            "parameter_manifest": local_manifest,
            "global_step": global_steps[0],
            "deterministic": deterministic_values[0],
        }
        checkpoint_metadata["format_version"] = _RANK_LOCAL_METADATA_VERSION
        checkpoint_metadata["global_step"] = global_steps[0]
        checkpoint_metadata["deterministic"] = deterministic_values[0]
        checkpoint_metadata["codebook"] = None
        checkpoint_metadata["rank_local_sharded_state"] = marker
        # DCP drops custom top-level keys, and a raw state_dict must not carry
        # rank 0's codebook as if it applied to every payload.
        state_dict["gefen_codebook"] = None

    def _state_dict_impl(self, *, consolidate_rank_local: bool = True):
        """Serialize optimizer state, plus Gefen's run-level extras.

        Beyond the base ``torch.optim.Optimizer.state_dict`` contents (per-param
        state keyed by index and caller-preserved ``param_groups``), the dict
        carries each parameter's stable ``name`` in its state, plus
        ``gefen_global_step``, the frozen exact-DP ``gefen_codebook`` (without it
        a resume would re-learn the codebook and desync the restored block
        periods), and the replica-determinism policy. Per-step scratch buffers
        are stripped, and non-owning state views are compacted to tight clones
        so checkpoints stay small.

        Only state entries whose parameter is still reachable from
        ``param_groups`` are serialized. Wrappers like DeepSpeed ZeRO-1/2
        replace each group's ``params`` with flat fp32 partition tensors
        without touching ``self.state``, which orphans the ``name`` entries
        Gefen pre-registers for the original model params at construction;
        the base ``state_dict`` indexes state keys through ``param_groups``
        and raises ``KeyError`` on any orphan. The orphaned entries stay in
        live ``self.state`` untouched (``_param_name`` still reads them);
        they are only withheld from the serialized copy.
        """
        self._synchronize_gefen_global_step_for_checkpoint()

        # Withhold state entries orphaned from param_groups around the base
        # call (see docstring). Tensor hashing is identity-based, so set
        # membership matches the id()-keyed param_mappings the base class
        # builds. The no-orphan path (every non-wrapped run) takes the plain
        # super() call, bit-for-bit unchanged.
        reachable = {param for group in self.param_groups for param in group["params"]}
        orphaned = [param for param in self.state if param not in reachable]
        if orphaned:
            original_order = list(self.state)
            withheld = [(param, self.state.pop(param)) for param in orphaned]
            try:
                state_dict = self._base_state_dict_without_hooks()
            finally:
                # Re-add only the withheld orphans, without clobbering
                # anything a registered state_dict pre-hook wrote to live
                # state during the super() call (torch semantics: hook
                # changes persist). Restore the original key order only when
                # the key set is unchanged; hook additions/removals keep the
                # hook's own ordering.
                for param, entry in withheld:
                    self.state.setdefault(param, entry)
                if set(self.state) == set(original_order):
                    current = dict(self.state)
                    self.state.clear()
                    for key in original_order:
                        self.state[key] = current[key]
        else:
            state_dict = self._base_state_dict_without_hooks()
        # stepsize/_h_buf are per-step scratch buffers (recomputed from vmean every
        # step); they live in self.state only to be reused across steps, so strip
        # them from the serialized dict instead of bloating every checkpoint. Build
        # fresh per-param dicts so live self.state is left untouched. The
        # capturable device-scalar buffer and its beta constants are likewise
        # scratch (fully refreshed from step counters + group hypers every step)
        # and must not survive a checkpoint across a capturable/hyper toggle.
        scratch_keys = (
            "stepsize",
            "_h_buf",
            "_capt_scalars",
            "_capt_consts",
            "_capt_consts_key",
            # Batched-refresh registration marker (a host int): scratch like
            # the buffers it indexes into; leaking it into a checkpoint would
            # falsely mark restored params as batch-covered.
            "_capt_stack",
            "_capt_row",
            # Live-only schema hints consumed by PyTorch's flattened optimizer
            # state unflattener. The collective adapter rebuilds their serialized
            # values after ordinary state has been compacted.
            _RANK_LOCAL_MEMBER_KEY,
        )

        def _is_scratch_key(key):
            return key in scratch_keys or (
                isinstance(key, str) and key.startswith(_RANK_LOCAL_PAYLOAD_KEY_PREFIX)
            )

        def _compact(value):
            # K4 rebinds m_magnitude onto a [2, num_blocks, 1] scratch buffer via
            # set_() (it shares storage with the transient sumsq/stepsize row).
            # Serializing such a non-owning view would persist the whole oversized
            # storage -- the scratch tail included -- into every checkpoint. Clone
            # any state tensor that doesn't own a tight storage so only its own
            # values are saved. Operates on the serialized copy only; live
            # self.state keeps the aliased fast-path buffer.
            if torch.is_tensor(value) and (
                value.storage_offset() != 0
                or value.untyped_storage().nbytes()
                != value.numel() * value.element_size()
            ):
                return value.detach().clone()
            return value

        state_dict["state"] = {
            pid: (
                {k: _compact(v) for k, v in pstate.items() if not _is_scratch_key(k)}
                if isinstance(pstate, dict)
                else pstate
            )
            for pid, pstate in state_dict["state"].items()
        }
        state_dict["gefen_global_step"] = self._gefen_global_step
        state_dict["gefen_deterministic"] = self._deterministic
        codebook_scope = self._serialized_codebook_scope()
        if codebook_scope is not None:
            state_dict["gefen_codebook_scope"] = codebook_scope
        native_local_shards = self._serialized_native_local_shards()
        if native_local_shards is not None:
            state_dict["gefen_native_local_shards"] = native_local_shards
        # The exact-DP codebook is learned once on the first step and then frozen
        # for the rest of the run. It is not per-param state, so persist it
        # explicitly; without it resume would reinterpret the restored uint8
        # m_codebook indices against a freshly learned codebook. (Note: FSDP/DCP
        # optim-state consolidation strips custom top-level keys; the per-group
        # _gefen_checkpoint_metadata mirror below carries the codebook through
        # that round-trip, and load_state_dict refuses quantized-momentum
        # checkpoints that lost both copies rather than relearning.)
        state_dict["gefen_codebook"] = self._gefen_codebook
        # PyTorch Distributed Checkpoint's ``get_optimizer_state_dict`` keeps
        # only the conventional ``state`` and ``param_groups`` top-level keys.
        # Mirror the optimizer-level values inside every serialized group so a
        # DCP/FSDP normalization round-trip cannot silently discard the frozen
        # codebook and reinterpret restored uint8 momentum indices with a newly
        # learned one. The loader removes this private transport metadata before
        # handing groups to torch, so it never leaks into live scheduler groups.
        checkpoint_metadata = {
            "format_version": (
                _SCOPED_NATIVE_METADATA_VERSION if codebook_scope is not None else 1
            ),
            "global_step": self._gefen_global_step,
            "codebook": self._gefen_codebook,
            "deterministic": self._deterministic,
            # PyTorch DCP's full-state loader first inspects a freshly-created
            # optimizer's local state to choose a broadcast/distribution device.
            # Gefen initializes momentum lazily, so before step 1 it otherwise
            # contains only parameter names and DCP raises "Expected device to
            # be set". This one-byte transport value is removed with the rest
            # of the private group metadata during load.
            "device_anchor": self._checkpoint_device_anchor(),
        }
        if codebook_scope is not None:
            checkpoint_metadata["codebook_scope"] = self._serialized_codebook_scope()
        if native_local_shards is not None:
            checkpoint_metadata["native_local_shards"] = native_local_shards
        if consolidate_rank_local:
            self._consolidate_rank_local_sharded_state(state_dict, checkpoint_metadata)
        for group in state_dict["param_groups"]:
            group["_gefen_checkpoint_metadata"] = checkpoint_metadata
        return state_dict

    @staticmethod
    def _same_serialized_group_value(a, b) -> bool:
        if torch.is_tensor(a) or torch.is_tensor(b):
            return torch.is_tensor(a) and torch.is_tensor(b) and torch.equal(a, b)
        return a == b

    def _pack_legacy_param_groups_for_load(self, state_dict):
        """Pack legacy one-param groups to the current caller-preserved layout.

        Older Gefen checkpoints serialized one param group per tensor. The
        underlying parameter order is the same flattening PyTorch uses for
        state_dict ids, so a legacy checkpoint can load into the new grouped
        optimizer when the total parameter count still matches and each run of
        legacy groups that maps to one current group has representable shared
        hyperparameters.
        """
        saved_groups = state_dict.get("param_groups")
        if not saved_groups or len(saved_groups) == len(self.param_groups):
            return state_dict
        if not all(len(group.get("params", ())) == 1 for group in saved_groups):
            return state_dict

        live_sizes = [len(group["params"]) for group in self.param_groups]
        if sum(live_sizes) != len(saved_groups):
            return state_dict

        packed_groups = []
        cursor = 0
        # Provenance is dropped with the names it describes: packing rewrites the
        # names positionally, so a flag saved against the OLD one-param-per-group
        # layout does not describe the packed name. Absent means unknowable, which
        # _sync_param_names_to_state fails closed on.
        ignored = {"params", "name", "param_names", _SYNTHESIZED_PARAM_NAMES_KEY}
        for live_group, size in zip(self.param_groups, live_sizes):
            chunk = saved_groups[cursor : cursor + size]
            cursor += size
            first_meta = {k: v for k, v in chunk[0].items() if k not in ignored}
            for legacy_group in chunk[1:]:
                meta = {k: v for k, v in legacy_group.items() if k not in ignored}
                if set(meta) != set(first_meta) or any(
                    not self._same_serialized_group_value(meta[k], first_meta[k])
                    for k in first_meta
                ):
                    raise ValueError(
                        "Cannot load a legacy Gefen checkpoint with per-parameter "
                        "hyperparameters into a preserved multi-parameter group. "
                        "Recreate the optimizer with matching param groups or "
                        "load before changing the grouping layout."
                    )
            names = [
                str(group.get("name", "param_{}".format(i))).lower()
                for i, group in enumerate(chunk, start=cursor - size)
            ]
            packed = dict(first_meta)
            if "name" in live_group:
                packed["name"] = live_group["name"]
            elif len(names) == 1:
                packed["name"] = names[0]
            packed["params"] = [group["params"][0] for group in chunk]
            packed["param_names"] = names
            packed_groups.append(packed)

        migrated = dict(state_dict)
        migrated["param_groups"] = packed_groups
        return migrated

    def load_state_dict(self, state_dict):
        """Atomically restore Gefen state between the public load hooks."""

        staged = self._prepare_load_state_dict(state_dict)
        self._publish_load_state_dict(staged)

    def _prepare_load_state_dict(self, state_dict):
        """Validate and stage a restore without mutating live state (phase one).

        Runs the load pre-hooks and stages the complete restore, returning an
        isolated shadow. Nothing on the live optimizer is mutated, so a rejection
        here (schema/layout mismatch, foreign backend, corrupted state) is a true
        no-op. Composite owners (GefenMuonHybrid) call this to validate every
        child before publishing any of them; ``load_state_dict`` pairs it with
        ``_publish_load_state_dict`` for the standalone atomic load.
        """

        self._assert_finalized_binding_layout(full=True)
        state_dict = state_dict.copy()
        for pre_hook in self._optimizer_load_state_dict_pre_hooks.values():
            hook_result = pre_hook(self, state_dict)
            if hook_result is not None:
                state_dict = hook_result
        self._assert_finalized_binding_layout(full=True)
        return self._stage_load_state_dict(state_dict)

    def _publish_load_state_dict(self, staged):
        """Commit a prepared restore through non-throwing swaps (phase two)."""

        self._commit_staged_load_state_dict(staged)
        self._run_load_state_dict_post_hooks()

    def _run_load_state_dict_post_hooks(self) -> None:
        """Dispatch the load post-hooks (may raise; separate from the commit).

        Kept apart from ``_commit_staged_load_state_dict`` so a composite owner
        (GefenMuonHybrid) can commit every child's raw state through the
        non-throwing swaps first and only then run any child's post-hooks -- a
        throwing post-hook must not fire while a sibling child is still
        uncommitted, which would strand the hybrid half-loaded.
        """

        for post_hook in self._optimizer_load_state_dict_post_hooks.values():
            post_hook(self)

    def _stage_load_state_dict(self, state_dict):
        """Prepare a complete restore without mutating the live optimizer.

        ``Optimizer.load_state_dict`` commits before Gefen restores auxiliary
        dtypes, normalizes counters, rebuilds names, and prepares checkpoint
        schema. Run that complete dynamically dispatched path on an isolated
        shadow first so a failure in any of those operations is a true no-op.
        ``copy.copy`` cannot be used here because ``Optimizer.__getstate__``
        intentionally omits Gefen's private runtime attributes.
        """

        staged = object.__new__(type(self))
        staged.__dict__ = self.__dict__.copy()

        # ``Optimizer.__setstate__`` mutates defaults, while the remaining
        # containers are cleared or invalidated by Gefen before/after the base
        # load. Isolate them so even a late staging failure cannot disturb live
        # caches, capturable views, or device counters.
        staged.defaults = self.defaults.copy()
        staged._gefen_codebook_by_device = {}
        staged._gefen_codebook_lut_by_device = {}
        staged._sr_seed_by_device = {}
        staged._gefen_global_step_by_device = {}
        staged._capt_stacks = None

        staged._load_state_dict_impl(state_dict)
        staged._validate_loaded_native_state()
        return staged

    def _commit_staged_load_state_dict(self, staged) -> None:
        """Publish an already prepared restore through non-throwing swaps."""

        # The base loader mutates the existing defaults mapping via setdefault;
        # preserve that public object identity while publishing the staged
        # value. Both mappings are ordinary built-in dicts created by Optimizer.
        live_defaults = self.defaults
        dict.update(live_defaults, staged.defaults)
        staged.defaults = live_defaults
        dict.update(self.__dict__, staged.__dict__)
        self._invalidate_layout_forensics_caches()

    def _validate_loaded_native_state(self) -> None:
        """Validate the complete prepared native state before publication."""

        self._validate_rank_local_counter("gefen_global_step", self._gefen_global_step)
        signature = self._rank_local_sharded_signature(context={})
        states = [
            self.state[param]
            for group in self.param_groups
            for param in group["params"]
        ]
        self._validate_rank_local_states(
            states,
            signature,
            self._gefen_codebook,
            allow_legacy_vmean_counter=True,
            allow_preinitialized_periods=(
                self._gefen_global_step == 0 and self._gefen_codebook is not None
            ),
        )
        for group in self.param_groups:
            self._validate_group_options(
                group["lr"],
                (group["beta1"], group["beta2"]),
                group["eps"],
                group["weight_decay"],
            )

    def _base_load_state_dict_without_hooks(self, state_dict):
        pre_hooks = self._optimizer_load_state_dict_pre_hooks
        post_hooks = self._optimizer_load_state_dict_post_hooks
        self._optimizer_load_state_dict_pre_hooks = OrderedDict()
        self._optimizer_load_state_dict_post_hooks = OrderedDict()
        try:
            return super().load_state_dict(state_dict)
        finally:
            self._optimizer_load_state_dict_pre_hooks = pre_hooks
            self._optimizer_load_state_dict_post_hooks = post_hooks

    @staticmethod
    def _validate_rank_local_codebook(codebook, *, required: bool) -> None:
        if codebook is None:
            if required:
                raise ValueError(
                    "Gefen rank-local checkpoint has quantized momentum but no codebook"
                )
            return
        if (
            not torch.is_tensor(codebook)
            or codebook.dtype != torch.float32
            or codebook.dim() != 1
            or codebook.numel() != 256
        ):
            raise ValueError(
                "Gefen rank-local checkpoint codebook must be a 256-element fp32 tensor"
            )
        codebook_cpu = codebook.detach().cpu()
        if not bool(torch.isfinite(codebook_cpu).all()):
            raise ValueError("Gefen rank-local checkpoint codebook must be finite")
        if not bool(torch.all(codebook_cpu[1:] >= codebook_cpu[:-1])):
            raise ValueError("Gefen rank-local checkpoint codebook must be sorted")
        if codebook_cpu[0].item() != -1.0 or codebook_cpu[-1].item() != 1.0:
            raise ValueError(
                "Gefen rank-local checkpoint codebook must retain endpoints -1 and 1"
            )

    @staticmethod
    def _validate_rank_local_counter(name, value) -> float:
        if torch.is_tensor(value):
            if (
                value.numel() != 1
                or value.dtype == torch.bool
                or not bool(torch.isfinite(value.detach()).all())
            ):
                raise ValueError(
                    "Gefen rank-local checkpoint counter {} must be a finite scalar".format(
                        name
                    )
                )
            scalar = float(value.detach().cpu().item())
        elif type(value) is int:
            scalar = float(value)
        else:
            raise ValueError(
                "Gefen rank-local checkpoint counter {} has invalid type {}".format(
                    name, type(value).__name__
                )
            )
        if scalar < 0 or not scalar.is_integer():
            raise ValueError(
                "Gefen rank-local checkpoint counter {} must be a nonnegative integer".format(
                    name
                )
            )
        return scalar

    def _validate_rank_local_states(
        self,
        states,
        signature,
        codebook,
        *,
        allow_legacy_vmean_counter: bool = False,
        allow_preinitialized_periods: bool = False,
    ) -> None:
        if not isinstance(states, list) or len(states) != len(signature):
            raise ValueError(
                "Gefen rank-local checkpoint payload has the wrong parameter count"
            )
        has_quantized_momentum = False
        counter_keys = ("step", "vmean_step", "factored_step", "normuon_step")
        for pstate, param_signature in zip(states, signature):
            if not isinstance(pstate, dict):
                raise ValueError(
                    "Gefen rank-local checkpoint parameter state must be a dict"
                )
            expected_name = param_signature["name"]
            if pstate.get("name") != expected_name:
                raise ValueError(
                    "Gefen rank-local checkpoint state name/order differs: checkpoint={!r} expected={!r}".format(
                        pstate.get("name"), expected_name
                    )
                )
            counters = {}
            for key in counter_keys:
                if key in pstate:
                    counters[key] = self._validate_rank_local_counter(key, pstate[key])

            local_shape = param_signature["local_shape"]
            state_shape = (
                param_signature["shape"]
                if param_signature.get("sharded_mode") in ("exact", "distributed")
                else local_shape
            )
            state_numel = math.prod(state_shape)
            period = pstate.get("automatic_period")
            if period is not None:
                if type(period) is not int or period <= 0:
                    raise ValueError(
                        "Gefen rank-local checkpoint automatic_period must be a positive int"
                    )
                if state_numel == 0 or state_numel % period != 0:
                    raise ValueError(
                        "Gefen rank-local checkpoint automatic_period does not divide the parameter state geometry"
                    )

            momentum_keys = ("m_codebook", "m_magnitude")
            carries_momentum = any(key in pstate for key in momentum_keys)
            initialized_keys = (
                "step",
                "vmean",
                "vmean_step",
                "v_row",
                "v_col",
                "factored_step",
                "normuon_v",
                "normuon_step",
            )
            if any(key in pstate for key in initialized_keys) and not carries_momentum:
                raise ValueError(
                    "Gefen rank-local checkpoint initialized state is missing quantized momentum"
                )
            if (
                period is not None
                and not carries_momentum
                and not (
                    param_signature.get("sharded")
                    and param_signature.get("sharded_mode") == "distributed"
                )
                and not (
                    allow_preinitialized_periods
                    and not any(key in pstate for key in initialized_keys)
                )
            ):
                raise ValueError(
                    "Gefen rank-local checkpoint automatic_period is invalid without initialized momentum"
                )
            if carries_momentum:
                has_quantized_momentum = True
                if not all(key in pstate for key in momentum_keys) or period is None:
                    raise ValueError(
                        "Gefen rank-local checkpoint momentum state is incomplete"
                    )
                if "step" not in pstate:
                    raise ValueError(
                        "Gefen rank-local checkpoint momentum state is missing step"
                    )
                if counters["step"] < 1:
                    raise ValueError(
                        "Gefen rank-local checkpoint initialized momentum requires step >= 1"
                    )
                blocks = state_numel // period
                indices = pstate["m_codebook"]
                magnitude = pstate["m_magnitude"]
                if (
                    not torch.is_tensor(indices)
                    or indices.dtype != torch.uint8
                    or tuple(indices.shape) != (blocks, period)
                ):
                    raise ValueError(
                        "Gefen rank-local checkpoint m_codebook geometry/dtype is invalid"
                    )
                if (
                    not torch.is_tensor(magnitude)
                    or magnitude.dtype != torch.float32
                    or tuple(magnitude.shape) != (blocks, 1)
                    or not bool(torch.isfinite(magnitude).all())
                    or not bool((magnitude >= 0).all())
                ):
                    raise ValueError(
                        "Gefen rank-local checkpoint m_magnitude geometry/dtype/values are invalid"
                    )
                vmean = pstate.get("vmean")
                if vmean is not None and (
                    not torch.is_tensor(vmean)
                    or vmean.dtype != torch.float32
                    or tuple(vmean.shape) != (blocks, 1)
                    or not bool(torch.isfinite(vmean).all())
                    or not bool((vmean >= 0).all())
                ):
                    raise ValueError(
                        "Gefen rank-local checkpoint vmean geometry/dtype/values are invalid"
                    )

                has_vmean = "vmean" in pstate
                has_vmean_step = "vmean_step" in pstate
                if has_vmean_step and not has_vmean:
                    raise ValueError(
                        "Gefen rank-local checkpoint block second moment is incomplete"
                    )
                if has_vmean and not has_vmean_step and not allow_legacy_vmean_counter:
                    raise ValueError(
                        "Gefen rank-local checkpoint block second moment is missing vmean_step"
                    )

                # GefenMuon groups carry a sharded_mode and intentionally use
                # quantized momentum without Adam's second moment. Plain Gefen
                # must carry one complete second-moment representation.
                if param_signature.get("sharded_mode") is None:
                    carries_factored = "v_row" in pstate or "v_col" in pstate
                    if carries_factored:
                        if "factored_step" not in pstate:
                            raise ValueError(
                                "Gefen rank-local checkpoint factored state is missing factored_step"
                            )
                        if counters["factored_step"] < 1:
                            raise ValueError(
                                "Gefen rank-local checkpoint initialized factored state requires factored_step >= 1"
                            )
                    elif "vmean" not in pstate:
                        raise ValueError(
                            "Gefen rank-local checkpoint plain momentum is missing a second moment"
                        )
                    elif "vmean_step" in counters and counters["vmean_step"] < 1:
                        raise ValueError(
                            "Gefen rank-local checkpoint initialized vmean requires vmean_step >= 1"
                        )

            factored = (pstate.get("v_row"), pstate.get("v_col"))
            if (factored[0] is None) != (factored[1] is None):
                raise ValueError(
                    "Gefen rank-local checkpoint factored second moment is incomplete"
                )
            has_factored_step = "factored_step" in pstate
            if has_factored_step != (factored[0] is not None):
                raise ValueError(
                    "Gefen rank-local checkpoint factored second moment is incomplete"
                )
            if factored[0] is not None:
                global_shape = param_signature["shape"]
                if len(global_shape) != 2:
                    raise ValueError(
                        "Gefen rank-local checkpoint factored state requires a 2-D parameter"
                    )
                for key, tensor, expected in (
                    ("v_row", factored[0], global_shape[0]),
                    ("v_col", factored[1], global_shape[1]),
                ):
                    if (
                        not torch.is_tensor(tensor)
                        or tensor.dtype != torch.float32
                        or tuple(tensor.shape) != (expected,)
                        or not bool(torch.isfinite(tensor).all())
                        or not bool((tensor >= 0).all())
                    ):
                        raise ValueError(
                            "Gefen rank-local checkpoint {} geometry/dtype/values are invalid".format(
                                key
                            )
                        )

            has_normuon_v = "normuon_v" in pstate
            has_normuon_step = "normuon_step" in pstate
            if has_normuon_v != has_normuon_step:
                raise ValueError(
                    "Gefen rank-local checkpoint NorMuon state is incomplete"
                )
            if has_normuon_v:
                if param_signature.get("sharded_mode") is None:
                    raise ValueError(
                        "Gefen rank-local checkpoint NorMuon state is invalid for a plain Gefen parameter"
                    )
                if not carries_momentum:
                    raise ValueError(
                        "Gefen rank-local checkpoint NorMuon state is missing initialized momentum"
                    )
                if len(state_shape) != 2:
                    raise ValueError(
                        "Gefen rank-local checkpoint NorMuon state requires a 2-D parameter"
                    )
                normuon_v = pstate["normuon_v"]
                if (
                    not torch.is_tensor(normuon_v)
                    or normuon_v.dtype != torch.float32
                    or tuple(normuon_v.shape) != (state_shape[0], 1)
                    or not bool(torch.isfinite(normuon_v).all())
                    or not bool((normuon_v >= 0).all())
                ):
                    raise ValueError(
                        "Gefen rank-local checkpoint normuon_v geometry/dtype/values are invalid"
                    )
                if counters["normuon_step"] < 1:
                    raise ValueError(
                        "Gefen rank-local checkpoint initialized NorMuon state requires normuon_step >= 1"
                    )

        self._validate_rank_local_codebook(codebook, required=has_quantized_momentum)

    def _unwrap_rank_local_sharded_checkpoint(self, state_dict) -> None:
        groups = state_dict.get("param_groups", ())
        if not isinstance(groups, (list, tuple)):
            raise ValueError("Gefen checkpoint param_groups must be a sequence")
        metadata = []
        for group in groups:
            if not isinstance(group, dict):
                raise ValueError("Gefen checkpoint parameter groups must be dicts")
            metadata.append(group.get("_gefen_checkpoint_metadata"))
        markers = [
            item.get("rank_local_sharded_state")
            for item in metadata
            if isinstance(item, dict)
            and item.get("rank_local_sharded_state") is not None
        ]
        wrapped_state = any(
            isinstance(pstate, dict)
            and (
                any(
                    isinstance(key, str)
                    and key.startswith(_RANK_LOCAL_PAYLOAD_KEY_PREFIX)
                    for key in pstate
                )
                or _RANK_LOCAL_MEMBER_KEY in pstate
            )
            for pstate in (state_dict.get("state", {}) or {}).values()
        )
        if not markers:
            if wrapped_state:
                raise ValueError(
                    "Gefen checkpoint carries rank-local DTensor payloads but is missing their topology metadata"
                )
            has_quantized_momentum = any(
                isinstance(pstate, dict) and "m_codebook" in pstate
                for pstate in (state_dict.get("state", {}) or {}).values()
            )
            if self._uses_rank_local_sharded_state() and has_quantized_momentum:
                raise ValueError(
                    "Cannot safely load an untagged/full Gefen checkpoint into "
                    "rank-local DTensor optimizer state. Older generic FSDP/DCP "
                    "full optimizer checkpoints kept rank 0's codebook and block "
                    "state for every rank. Re-save from the original run with a "
                    "Gefen version that writes {} payloads.".format(_RANK_LOCAL_FORMAT)
                )
            return
        if len(markers) != len(groups) or any(item is None for item in metadata):
            raise ValueError(
                "Gefen rank-local DTensor checkpoint metadata is present on only some parameter groups"
            )
        marker = markers[0]
        if not isinstance(marker, dict):
            raise ValueError("Gefen rank-local checkpoint marker must be a dict")
        if any(not isinstance(item, dict) or item != marker for item in markers[1:]):
            raise ValueError(
                "Gefen rank-local checkpoint parameter groups carry inconsistent markers"
            )
        if marker.get("format") != _RANK_LOCAL_FORMAT:
            raise ValueError(
                "Unsupported Gefen rank-local checkpoint format: {!r}".format(
                    marker.get("format")
                )
            )
        if not self._uses_rank_local_sharded_state():
            raise ValueError(
                "A rank-local DTensor Gefen checkpoint can only load into the same sharded optimizer topology"
            )
        try:
            context = self._rank_local_checkpoint_context()
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
        world = context["world_size"]
        rank = context["global_rank"]
        if marker.get("world_size") != world:
            raise ValueError(
                "Gefen rank-local DTensor checkpoints require the same world size; checkpoint={} current={}".format(
                    marker.get("world_size"), world
                )
            )
        if marker.get("world_ranks") != context["world_ranks"]:
            raise ValueError(
                "Gefen rank-local checkpoint default-world rank membership differs"
            )
        if marker.get("mesh") != context["mesh"]:
            raise ValueError("Gefen rank-local checkpoint DeviceMesh differs")
        signatures = marker.get("signatures")
        expected_rank_keys = {str(item) for item in context["world_ranks"]}
        if not isinstance(signatures, dict) or set(signatures) != expected_rank_keys:
            raise ValueError("Gefen rank-local checkpoint has invalid signatures")
        current_signature = self._rank_local_sharded_signature(context)
        if signatures[str(rank)] != current_signature:
            raise ValueError(
                "Gefen rank-local DTensor checkpoint topology differs on rank {}: checkpoint={!r} current={!r}".format(
                    rank, signatures[str(rank)], current_signature
                )
            )
        current_manifest = self._rank_local_parameter_manifest(current_signature)
        if marker.get("parameter_manifest") != current_manifest:
            raise ValueError(
                "Gefen rank-local checkpoint ordered parameter manifest differs"
            )
        marker_step = marker.get("global_step")
        if type(marker_step) is not int or marker_step < 0:
            raise ValueError("Gefen rank-local checkpoint has invalid global_step")
        marker_deterministic = marker.get("deterministic")
        if type(marker_deterministic) is not bool:
            raise ValueError(
                "Gefen rank-local checkpoint has invalid deterministic policy"
            )
        saved_ids = list(
            chain.from_iterable(group.get("params", ()) for group in groups)
        )
        try:
            unique_saved_ids = set(saved_ids)
        except TypeError as exc:
            raise ValueError(
                "Gefen rank-local checkpoint parameter IDs must be hashable"
            ) from exc
        if len(unique_saved_ids) != len(saved_ids):
            raise ValueError(
                "Gefen rank-local checkpoint contains duplicate outer parameter IDs"
            )
        expected_carrier_keys = {
            _rank_local_payload_key(item) for item in context["world_ranks"]
        }
        serialized_payloads = None
        for param_id in saved_ids:
            pstate = (state_dict.get("state", {}) or {}).get(param_id)
            if not isinstance(pstate, dict):
                raise ValueError(
                    "Gefen rank-local checkpoint is missing parameter state {!r}".format(
                        param_id
                    )
                )
            carrier_keys = {
                key
                for key in pstate
                if isinstance(key, str)
                and key.startswith(_RANK_LOCAL_PAYLOAD_KEY_PREFIX)
            }
            extras = (
                set(pstate)
                - carrier_keys
                - {
                    _RANK_LOCAL_MEMBER_KEY,
                    "name",
                }
            )
            if extras:
                raise ValueError(
                    "Gefen rank-local checkpoint parameter state has an invalid schema: {!r}".format(
                        pstate
                    )
                )
            if pstate.get("name") != 0:
                raise ValueError(
                    "Gefen rank-local checkpoint outer name sentinel is invalid"
                )
            has_carrier = bool(carrier_keys)
            has_member = _RANK_LOCAL_MEMBER_KEY in pstate
            if has_carrier and has_member:
                raise ValueError(
                    "Gefen rank-local checkpoint state cannot be both carrier and member"
                )
            if has_carrier:
                if carrier_keys != expected_carrier_keys:
                    raise ValueError(
                        "Gefen rank-local checkpoint carrier has missing or unexpected per-rank payload keys"
                    )
                if serialized_payloads is not None:
                    raise ValueError(
                        "Gefen rank-local checkpoint repeats its payload transport"
                    )
                serialized_payloads = {
                    str(global_rank): pstate[_rank_local_payload_key(global_rank)]
                    for global_rank in context["world_ranks"]
                }
            elif pstate.get(_RANK_LOCAL_MEMBER_KEY) is not True:
                raise ValueError(
                    "Gefen rank-local checkpoint has an invalid member marker"
                )
        if (
            not isinstance(serialized_payloads, dict)
            or set(serialized_payloads) != expected_rank_keys
            or any(not torch.is_tensor(item) for item in serialized_payloads.values())
        ):
            raise ValueError("Gefen rank-local checkpoint has invalid rank payloads")
        selected_payload = self._deserialize_rank_local_payload(
            serialized_payloads[str(rank)]
        )
        expected_payload_keys = {
            "format",
            "global_rank",
            "group_rank",
            "world_size",
            "world_ranks",
            "mesh",
            "signature",
            "parameter_manifest",
            "global_step",
            "deterministic",
            "states",
            "codebook",
        }
        if (
            not isinstance(selected_payload, dict)
            or set(selected_payload) != expected_payload_keys
        ):
            raise ValueError(
                "Gefen rank-local checkpoint payload has an invalid schema"
            )
        payload_identity = (
            selected_payload["format"] == _RANK_LOCAL_FORMAT
            and selected_payload["global_rank"] == rank
            and selected_payload["group_rank"] == context["group_rank"]
            and selected_payload["world_size"] == world
            and selected_payload["world_ranks"] == context["world_ranks"]
            and selected_payload["mesh"] == context["mesh"]
            and selected_payload["signature"] == current_signature
            and selected_payload["parameter_manifest"] == current_manifest
            and selected_payload["global_step"] == marker_step
            and selected_payload["deterministic"] == marker_deterministic
        )
        if not payload_identity:
            raise ValueError(
                "Gefen rank-local checkpoint payload rank/topology/parameter binding differs"
            )
        selected_states = selected_payload["states"]
        selected_codebook = selected_payload["codebook"]
        # Match the native load path's legacy tolerance: an older rank-local
        # checkpoint can carry vmean without vmean_step (a pre-counter state, as
        # old as step), and the step-time resume path backfills vmean_step from
        # step. Rejecting it here (the default is strict) would block that
        # backfill and break otherwise valid legacy DTensor/FSDP resumes, while
        # the native path already accepts them.
        self._validate_rank_local_states(
            selected_states,
            current_signature,
            selected_codebook,
            allow_legacy_vmean_counter=True,
        )
        state_dict["state"] = dict(zip(saved_ids, selected_states))
        state_dict["gefen_global_step"] = marker_step
        state_dict["gefen_codebook"] = selected_codebook
        state_dict["gefen_deterministic"] = marker_deterministic
        for group in groups:
            group_metadata = dict(group["_gefen_checkpoint_metadata"])
            group_metadata["global_step"] = marker_step
            group_metadata["codebook"] = selected_codebook
            group_metadata["deterministic"] = marker_deterministic
            group["_gefen_checkpoint_metadata"] = group_metadata

    def _load_state_dict_impl(self, state_dict):
        """Restore optimizer state saved by :meth:`state_dict`.

        On top of the base load this restores ``gefen_global_step`` and the
        frozen codebook, verifies that an explicitly tagged checkpoint uses the
        same replica-determinism policy as this optimizer, and undoes the base
        class's dtype coercion of aux state: ``torch.optim`` casts every floating
        state tensor to the owning param's dtype, which would corrupt Gefen's
        fp32 moments and uint8 momentum indices on bf16 models, so each aux
        tensor is written back in its ORIGINAL saved dtype. Step counters are
        normalized to this optimizer's ``capturable`` mode (host ints vs 0-dim
        device tensors), so checkpoints are portable across that toggle in
        either direction. Legacy checkpoints with no deterministic tag remain
        loadable and retain the live optimizer's configured policy.
        """
        # Shallow-copy so the pop()s below don't mutate the caller's checkpoint
        # dict (torch.optim.Optimizer.load_state_dict leaves its input intact);
        # otherwise a second load of the same dict loses the gefen_* keys.
        state_dict = dict(state_dict)
        # Copy the group dictionaries before removing private transport
        # metadata so repeated loads leave the caller's checkpoint untouched.
        state_dict["param_groups"] = [
            dict(group) for group in state_dict.get("param_groups", ())
        ]
        self._unwrap_rank_local_sharded_checkpoint(state_dict)
        group_metadata = []
        for group in state_dict["param_groups"]:
            metadata = group.pop("_gefen_checkpoint_metadata", None)
            if metadata is not None:
                group_metadata.append(metadata)

        gefen_global_step = state_dict.pop("gefen_global_step", None)
        gefen_codebook = state_dict.pop("gefen_codebook", None)
        has_top_level_codebook_scope = "gefen_codebook_scope" in state_dict
        gefen_codebook_scope = state_dict.pop("gefen_codebook_scope", None)
        if has_top_level_codebook_scope:
            gefen_codebook_scope = self._normalize_serialized_codebook_scope(
                gefen_codebook_scope
            )
        has_top_level_native_local_shards = "gefen_native_local_shards" in state_dict
        native_local_shards = state_dict.pop("gefen_native_local_shards", None)
        if has_top_level_native_local_shards:
            native_local_shards = self._normalize_serialized_native_local_shards(
                native_local_shards
            )
        has_top_level_deterministic = "gefen_deterministic" in state_dict
        gefen_deterministic = state_dict.pop("gefen_deterministic", None)
        if has_top_level_deterministic and type(gefen_deterministic) is not bool:
            raise ValueError(
                "Gefen checkpoint top-level deterministic policy must be a bool, got {!r}".format(
                    gefen_deterministic
                )
            )
        if group_metadata:
            if len(group_metadata) != len(state_dict["param_groups"]):
                raise ValueError(
                    "Gefen checkpoint metadata is present on only some parameter groups"
                )
            first_metadata = group_metadata[0]
            normalized_metadata_native_local_shards = []
            for metadata in group_metadata:
                metadata_version = metadata.get("format_version")
                if metadata_version not in (
                    1,
                    2,
                    _RANK_LOCAL_METADATA_VERSION,
                    _SCOPED_NATIVE_METADATA_VERSION,
                ):
                    raise ValueError(
                        "Unsupported Gefen checkpoint metadata format_version: {}".format(
                            metadata_version
                        )
                    )
                if ("codebook_scope" in metadata) != (
                    metadata_version == _SCOPED_NATIVE_METADATA_VERSION
                ):
                    raise ValueError(
                        "Gefen scoped checkpoint metadata must use format_version {}".format(
                            _SCOPED_NATIVE_METADATA_VERSION
                        )
                    )
                if (
                    "deterministic" in metadata
                    and type(metadata["deterministic"]) is not bool
                ):
                    raise ValueError(
                        "Gefen checkpoint parameter-group deterministic policy must be a bool, got {!r}".format(
                            metadata["deterministic"]
                        )
                    )
                normalized_metadata_native_local_shards.append(
                    self._normalize_serialized_native_local_shards(
                        metadata.get("native_local_shards")
                    )
                )
            for metadata_index, metadata in enumerate(group_metadata[1:], start=1):
                same_version = metadata.get("format_version") == first_metadata.get(
                    "format_version"
                )
                same_step = metadata.get("global_step") == first_metadata.get(
                    "global_step"
                )
                left_codebook = metadata.get("codebook")
                right_codebook = first_metadata.get("codebook")
                same_codebook = left_codebook is right_codebook or (
                    torch.is_tensor(left_codebook)
                    and torch.is_tensor(right_codebook)
                    and torch.equal(left_codebook, right_codebook)
                )
                same_deterministic = metadata.get(
                    "deterministic"
                ) == first_metadata.get("deterministic")
                same_codebook_scope = metadata.get(
                    "codebook_scope"
                ) == first_metadata.get("codebook_scope")
                same_native_local_shards = (
                    normalized_metadata_native_local_shards[metadata_index]
                    == normalized_metadata_native_local_shards[0]
                )
                if (
                    not same_version
                    or not same_step
                    or not same_codebook
                    or not same_deterministic
                    or not same_codebook_scope
                    or not same_native_local_shards
                ):
                    raise ValueError(
                        "Gefen checkpoint parameter groups carry inconsistent optimizer metadata"
                    )
            metadata_step = first_metadata.get("global_step", 0)
            metadata_codebook = first_metadata.get("codebook")
            metadata_deterministic = first_metadata.get("deterministic")
            metadata_codebook_scope = self._normalize_serialized_codebook_scope(
                first_metadata.get("codebook_scope")
            )
            metadata_native_local_shards = normalized_metadata_native_local_shards[0]
            if gefen_global_step is None:
                gefen_global_step = metadata_step
            elif gefen_global_step != metadata_step:
                raise ValueError(
                    "Gefen checkpoint top-level and parameter-group global steps disagree"
                )
            if gefen_codebook is None:
                gefen_codebook = metadata_codebook
            elif not (
                gefen_codebook is metadata_codebook
                or (
                    torch.is_tensor(gefen_codebook)
                    and torch.is_tensor(metadata_codebook)
                    and torch.equal(gefen_codebook, metadata_codebook)
                )
            ):
                raise ValueError(
                    "Gefen checkpoint top-level and parameter-group codebooks disagree"
                )
            if gefen_deterministic is None:
                gefen_deterministic = metadata_deterministic
            elif (
                metadata_deterministic is not None
                and gefen_deterministic != metadata_deterministic
            ):
                raise ValueError(
                    "Gefen checkpoint top-level and parameter-group deterministic policies disagree"
                )
            if not has_top_level_codebook_scope:
                gefen_codebook_scope = metadata_codebook_scope
            elif gefen_codebook_scope != metadata_codebook_scope:
                raise ValueError(
                    "Gefen checkpoint top-level and parameter-group codebook scopes disagree"
                )
            if not has_top_level_native_local_shards:
                native_local_shards = metadata_native_local_shards
            elif native_local_shards != metadata_native_local_shards:
                raise ValueError(
                    "Gefen checkpoint top-level and parameter-group local shards disagree"
                )
        live_codebook_scope = self._serialized_codebook_scope()
        if gefen_codebook_scope != live_codebook_scope:
            raise ValueError(
                "Gefen checkpoint codebook scope does not match the live explicit binding"
            )
        if not self._native_local_shards_matches_live(native_local_shards):
            raise ValueError(
                "Gefen checkpoint native local-shard identity does not match the live binding"
            )
        if gefen_deterministic is not None:
            if type(gefen_deterministic) is not bool:
                raise ValueError(
                    "Gefen checkpoint deterministic policy must be a bool, got {!r}".format(
                        gefen_deterministic
                    )
                )
            if gefen_deterministic != self._deterministic:
                raise ValueError(
                    "Gefen checkpoint deterministic={!r}, but this optimizer was "
                    "constructed with deterministic={!r}. Resuming under a "
                    "different replica-determinism policy requires an intentional "
                    "state migration or a fresh optimizer restart; refusing to "
                    "change the policy silently.".format(
                        gefen_deterministic, self._deterministic
                    )
                )
        if gefen_global_step is None:
            gefen_global_step = 0
        has_quantized_momentum = any(
            isinstance(param_state, dict) and "m_codebook" in param_state
            for param_state in (state_dict.get("state", {}) or {}).values()
        )
        if has_quantized_momentum and gefen_codebook is None:
            raise ValueError(
                "Gefen checkpoint contains quantized momentum indices but no frozen "
                "codebook. Loading it would reinterpret the restored indices with a "
                "different codebook and silently corrupt optimizer state. Resume from "
                "an unmodified Gefen checkpoint or a DCP checkpoint written by a "
                "version that preserves _gefen_checkpoint_metadata."
            )
        state_dict = self._pack_legacy_param_groups_for_load(state_dict)

        # torch.optim.Optimizer.load_state_dict casts *every* per-param state
        # tensor to the owning parameter's dtype whenever that parameter is
        # floating point. For a bf16 model that silently downcasts Gefen's
        # float32 aux moments (vmean, m_magnitude) and its uint8 quantized
        # momentum codebook (m_codebook) to bf16, after which the CUDA kernels
        # reject the wrong dtypes and the first post-resume step crashes. Stash
        # the pristine saved tensors and write them back after the base load so
        # every aux tensor round-trips losslessly (a bf16 round-trip would also
        # corrupt the float32 moments). Keyed off the saved state, not hard-coded
        # names, so any future aux tensor is preserved too.
        saved_state = state_dict.get("state", {}) or {}

        # The base load replaces self.state wholesale, so every counter/scalar
        # view the batched capturable refresh installed is gone; drop the
        # stacked buffers with them (the next eager capturable step rebuilds
        # and re-aliases lazily).
        self._capt_invalidate()

        self._base_load_state_dict_without_hooks(state_dict)
        self._gefen_global_step = gefen_global_step
        # Capturable SR seeds are optimizer-level scratch (a device mirror of
        # gefen_global_step): drop them so the first post-load SR kernel call
        # uses a scalar rebuilt from the restored, replay-synchronized counter.
        self._sr_seed_by_device.clear()
        self._reset_gefen_global_step_devices()

        # Restoring the frozen codebook keeps _maybe_refresh_gefen_codebook a
        # no-op on the first resume step, so the restored automatic_period values
        # stay consistent with the restored vmean/m_codebook block geometry.
        # Assign unconditionally: when FSDP strips the codebook (gefen_codebook is
        # None) any stale codebook from a prior train/load must be cleared, else
        # _maybe_refresh_gefen_codebook returns early and skips the restored-period
        # fallback.
        self._gefen_codebook = gefen_codebook
        self._gefen_codebook_by_device.clear()
        self._gefen_codebook_lut_by_device.clear()
        self._gefen_codebook_scope_validated = False
        # The identity fingerprint would catch the wholesale state swap anyway;
        # reset it explicitly so the first post-load capturable step re-marks.
        self._static_mark_sig = None

        # Map saved param ids -> live param objects exactly as the base class
        # does, then restore each aux tensor from its pristine saved copy.
        id_map = dict(
            zip(
                chain.from_iterable(
                    group["params"] for group in state_dict["param_groups"]
                ),
                chain.from_iterable(group["params"] for group in self.param_groups),
            )
        )
        for param_id, saved_param_state in saved_state.items():
            param = id_map.get(param_id)
            if param is None:
                continue
            live_state = self.state.get(param)
            if live_state is None:
                continue
            for key, saved_value in saved_param_state.items():
                if not torch.is_tensor(saved_value):
                    continue
                live_value = live_state.get(key)
                if (
                    torch.is_tensor(live_value)
                    and live_value.dtype == saved_value.dtype
                ):
                    continue
                device = (
                    live_value.device if torch.is_tensor(live_value) else param.device
                )
                live_state[key] = saved_value.to(device=device, dtype=saved_value.dtype)

        # Normalize the step counters to this optimizer's capturable mode, so
        # checkpoints are portable across the toggle in either direction (torch
        # .optim casts its `step` state the same way): python ints when
        # capturable=False (the bit-identical legacy host math), 0-dim float32
        # tensors on the param's device when capturable=True.
        counter_keys = ("step", "vmean_step", "factored_step", "normuon_step")
        for group in self.param_groups:
            for p in group["params"]:
                pstate = self.state.get(p)
                if not pstate:
                    continue
                for key in counter_keys:
                    if key not in pstate:
                        continue
                    value = pstate[key]
                    self._validate_rank_local_counter(key, value)
                    if self.capturable:
                        if not torch.is_tensor(value):
                            pstate[key] = torch.full(
                                (), float(value), dtype=torch.float32, device=p.device
                            )
                        elif (
                            value.device != p.device
                            or value.dtype != torch.float32
                            or value.dim() != 0
                        ):
                            pstate[key] = value.to(
                                device=p.device, dtype=torch.float32
                            ).reshape(())
                    elif torch.is_tensor(value):
                        pstate[key] = int(value.item())
        self._sync_param_names_to_state()
        self._install_rank_local_checkpoint_schema()

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (callable, optional): re-evaluates the model and returns
                the loss; invoked under ``torch.enable_grad()`` BEFORE any
                optimizer work (matching :meth:`GefenMuon.step`), so
                closure-driven loops that produce the gradients inside the
                closure feed the first step's codebook learning correctly. The
                returned loss is passed through.
        """
        # Capture and validate the control scope independently of public
        # parameter containers. A rank-local mutation made between steps can
        # then be reported through the last known-good binding instead of
        # stranding peers after a one-sided entry-guard failure.
        scope_binding = self._capture_codebook_scope_binding_for_step()

        loss = None
        try:
            self._assert_finalized_binding_layout()
            self._assert_runtime_codebook_process_group()
            self._assert_capturable_if_capturing()
            self._assert_codebook_capture_ready()
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            local_preamble_error = None
        except Exception as exc:
            loss = None
            local_preamble_error = exc
        if scope_binding is not None:
            self._synchronize_prevalidated_codebook_scope_failure(
                local_preamble_error, "step preamble", scope_binding
            )
        elif local_preamble_error is not None:
            raise local_preamble_error

        # The closure can replace a finalized parameter or otherwise invalidate
        # the runtime binding. It can also rebind (or clear) the runtime
        # process-group between capture and the operation header; a rank that
        # silently swapped to None or a different binding would enter a
        # different header collective than its peers and deadlock. Recheck
        # against the captured binding and synchronize structural failures
        # before any peer enters a scoped codebook collective.
        try:
            if self._gefen_codebook_process_group is not scope_binding:
                raise RuntimeError(
                    "Gefen codebook process-group binding changed during the step preamble"
                )
            self._assert_finalized_binding_layout()
            self._assert_runtime_codebook_process_group()
            _assert_optimizer_gradients_structurally_valid(self)
            local_preflight_error = None
        except Exception as exc:
            local_preflight_error = exc
        if scope_binding is not None:
            self._synchronize_prevalidated_codebook_scope_failure(
                local_preflight_error, "gradient preflight", scope_binding
            )
        elif local_preflight_error is not None:
            raise local_preflight_error
        self._validate_codebook_scope_operation_header("step")
        self._ensure_codebook_scope_agreement()

        # GradScaler invokes native-AMP optimizers even on overflow. Decide
        # before codebook learning, periodic refresh, capturable counters, or
        # parameter/state mutation. A multi-member codebook scope must run the
        # AMP protocol agreement on EVERY member -- including one with no local
        # GradScaler attributes -- otherwise the presence gather inside
        # _prepare_scoped_amp_optimizer_step is entered by only some members and
        # deadlocks (mirrors the unconditional Muon/Hybrid AMP preflight). Off a
        # multi-member scope the attribute gate still compiles away on the
        # ordinary BF16/FP32 path where GradScaler attaches nothing.
        scope = self._gefen_codebook_process_group
        if scope is not None and len(scope.identity.ordered_members) > 1:
            should_step = self._prepare_scoped_amp_optimizer_step()
        elif hasattr(self, "found_inf") or hasattr(self, "grad_scale"):
            should_step = self._prepare_scoped_amp_optimizer_step()
        else:
            should_step = True
        if not should_step:
            return loss

        if (
            self._gefen_codebook_process_group is not None
            and self._codebook_refresh_every
        ):
            self._validate_codebook_scope_operation_header("periodic_step")

        self._maybe_refresh_gefen_codebook()
        self._maybe_save_gefen_grad_histogram()
        # Periodic codebook re-learn (opt-in): every N steps, refit the exact-DP
        # codebook to the current gradient distribution and re-quantize the
        # stored momentum indices onto it (see _refresh_codebook_with_requant).
        if (
            self._codebook_refresh_every
            and self._gefen_global_step > 0
            and self._gefen_global_step % self._codebook_refresh_every == 0
        ):
            self._refresh_codebook_with_requant()

        # Batched capturable refresh: advance every registered param's device
        # step counters and refresh every param's [lr, 1/bcX, 1/bc1, wd]
        # kernel-scalar row with one short vectorized op chain over per-device
        # stacked buffers, instead of ~10 microscopic launches per param.
        # Elementwise-identical math (see _capt_batched_refresh); runs after
        # the closure so the stepping set (which params carry grads) is final.
        if self.capturable:
            self._capt_batched_prologue()

        # The fused path has its own batched CUDA kernel per param; only the
        # non-fused Python-loop path benefits from foreach/merged batching.
        # Gate on BOTH fused flags: the merged path hard-codes use_fused=False for
        # the vmean update, so it must not engage while fused vmean is still on
        # (e.g. FUSE_GEFEN_AUTOMATIC_STEP off but FUSE_AUTOMATIC_VMEAN_UPDATE on),
        # which would otherwise change the vmean math vs the per-param path.
        # The merged path additionally computes ONE shared bias correction from
        # the first param's host-int step (math.sqrt on it under the hood), so
        # it is skipped under capturable: each param then steps through the
        # per-param tensor-op path with its own device step counter, and the
        # extra launches are exactly what a captured graph amortizes.
        batch_nonfused = (
            not self._use_fused_gefen_automatic_step()
            and not self._use_fused_automatic_vmean()
            and not self.capturable
        )

        # Collect parameters that share block geometry + hyperparameters so they
        # can be processed as one merged tensor (see _step_automatic_merged).
        # Anything not eligible (fused path, first step before state exists,
        # DTensor/FSDP2 shards, LR-Tensor schedules, or LARGE params that should
        # stream) keeps the per-param path.
        nonfused_groups = {}
        for group in self.param_groups:
            for name, p in self._iter_group_params_with_names(group):
                grad = p.grad
                if grad is None:
                    continue
                if getattr(grad, "is_sparse", False):
                    raise RuntimeError("Gefen does not support sparse gradients")
                if torch.is_complex(grad):
                    raise RuntimeError("Gefen does not support complex gradients")
                # Parameter / gradient device co-location is enforced up front in
                # _assert_optimizer_gradients_structurally_valid (before any
                # codebook or per-param state mutation), so by here a plain
                # tensor's grad and param share a device.
                # Factored-v routing (opt-in): plain 2D params keep their
                # quantized momentum but step through the factored-second-
                # moment path (in-kernel elementwise stepsize on the fused
                # path) instead of any vmean path. Sharded (DTensor) params
                # fall through to the standard path: local-shard row/col
                # statistics would be wrong under sharding.
                if self._factored_v_2d and p.ndim == 2 and not hasattr(p, "placements"):
                    self._step_automatic_factored(group, name, p, grad)
                    continue
                if (
                    batch_nonfused
                    and self._nonfused_state_ready(p)
                    and self._nonfused_batch_eligible(p, grad)
                ):
                    key = self._nonfused_batch_key(group, p)
                    if key is not None:
                        nonfused_groups.setdefault(key, []).append(
                            (group, name, p, grad)
                        )
                        continue
                self._step_automatic(group, name, p, grad)

        for items in nonfused_groups.values():
            if len(items) == 1:
                group, name, p, grad = items[0]
                self._step_automatic(group, name, p, grad)
            else:
                self._dispatch_nonfused_batch(items)

        # Capturable stochastic rounding: advance the per-device seed tensors
        # ON DEVICE at the step tail (the device mirror of the host counter
        # bump below). Recorded on-stream by a manual capture -- so replays
        # dither with fresh seeds -- and kept out of traced frames by the
        # helper's @torch._dynamo.disable.
        if self.capturable and self._stochastic_round:
            self._advance_sr_seeds()
        self._advance_gefen_global_step()
        if self.capturable and not torch.compiler.is_compiling():
            self._mark_state_static_for_compile()
        return loss

    @torch._dynamo.disable
    def _advance_gefen_global_step(self) -> None:
        # Kept out of compiled frames on purpose: dynamo guards on the exact
        # value of a python int it reads, so an in-trace `+= 1` on this
        # per-step host counter would recompile the whole compiled step every
        # step. @torch._dynamo.disable makes the increment a single opaque
        # call at the very tail of step() instead (one cheap graph break
        # after all device work). Eager behavior is unchanged.
        self._ensure_gefen_global_step_devices()
        for counter in self._gefen_global_step_by_device.values():
            counter.add_(1)
        self._gefen_global_step += 1

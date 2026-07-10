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

import logging
import math
import os
import warnings
from itertools import chain
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn

from gefen.partitioning import find_period_by_block_variance
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
    histogram_bins = num_codebooks * 16
    bin_width = 2.0 / float(histogram_bins)
    # Accumulate counts as int64 and cast to float32 once at assembly: repeated
    # fp32 adds round for bins past 2^24 counts, so chunked accumulation could
    # otherwise drift from the whole-tensor form's single cast.
    bin_counts_cpu = torch.zeros(histogram_bins, dtype=torch.int64, device="cpu")
    total_numel = 0

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
                    total_numel += flat_float.numel()
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
                    local_counts = torch.bincount(
                        bin_indices, minlength=histogram_bins
                    )
                    bin_counts_cpu.add_(local_counts.cpu())
                    total_numel += normalized_flat.numel()
    finally:
        torch.set_deterministic_debug_mode(prev_mode)

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
        capturable (bool, default False): CUDA-graph capturability (mirrors
            torch.optim's argument). Everything that varies across steps --
            step counters, bias corrections, kernel scalars -- lives in device
            tensors and the step avoids host<->device syncs, so ``step()`` can
            be captured in a ``torch.cuda.CUDAGraph`` and replayed. A tensor
            ``lr`` flows on device (update it in place between replays); a
            float ``lr`` is baked at capture time. Codebook learning and the
            period search are host-driven and run on the FIRST step, so do the
            standard warmup steps before capturing. Incompatible with
            ``codebook_refresh_every > 0``.
        verbose (bool, default False): extra diagnostics during codebook
            learning and period prediction.
    """

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
        capturable: bool = False,
        verbose: bool = False,
    ):
        if fused and not torch.cuda.is_available():
            warnings.warn(
                "Gefen optimizer got fused=True, but CUDA is not available. "
                "Changing fused to False.",
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
        self._period_one_substrings = tuple(
            s.lower() for s in period_one_substrings
        )
        self._factored_v_2d = factored_v_2d
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

        defaults = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            weight_decay=weight_decay,
        )
        self._param_names = {}
        super().__init__(self._normalize_param_groups(params), defaults)

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
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0])
            )
        elif not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1])
            )
        elif not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        elif not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

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
                    "Gefen does not support complex parameters, but got a "
                    "parameter with dtype {}".format(param.dtype)
                )
            yield param_name, param

    def _param_name(self, param) -> str:
        state = self.state.get(param)
        if state is not None and "name" in state:
            return state["name"]
        return self._param_names.get(param, "none")

    def _iter_group_params_with_names(self, group):
        names = group.get("param_names")
        for idx, param in enumerate(group["params"]):
            if names is not None and idx < len(names):
                yield names[idx], param
            else:
                yield self._param_name(param), param

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
        self._param_names = {}
        for group in self.param_groups:
            params = group["params"]
            names = list(group.get("param_names", ()))
            if len(names) != len(params):
                names = [self._param_name(p) for p in params]
            normalized_names = []
            for param, name in zip(params, names):
                name = str(name).lower()
                normalized_names.append(name)
                self._param_names[param] = name
                self.state[param]["name"] = name
            group["param_names"] = normalized_names

    def add_param_group(self, param_group):
        """Add a param group while preserving the caller-visible group boundary.

        Accepts the same group dict shape as the constructor (``params`` as
        tensors or ``(name, tensor)`` tuples; optional ``lr`` / ``betas`` (or
        ``beta1``/``beta2``) / ``eps`` / ``weight_decay``, defaulting to the
        constructor values). The public group is preserved; each parameter's
        stable lowercase name is stored in its per-param state and in the
        group's ``param_names`` list for introspection.
        """
        if not isinstance(param_group, dict):
            raise TypeError(
                "param_group must be a dict, got {}".format(
                    type(param_group).__name__
                )
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
        for param_index, (param_name, param) in enumerate(named_params):
            # Reject duplicates before registering any group, so the base class's
            # own (post-registration) duplicate check can't leave us partial.
            if param in existing_params or param in batch_params:
                raise ValueError(
                    "some parameters appear in more than one parameter group"
                )
            batch_params.add(param)
            params.append(param)
            if param_name is None:
                if group_name is not None and len(named_params) == 1:
                    param_name = group_name
                    param_name = self._unique_name(param_name, existing_names)
                elif implicit_group:
                    param_name = self._unique_name(
                        "param_{}".format(len(self._param_names) + param_index),
                        existing_names,
                    )
                else:
                    param_name = self._unique_name(
                        "group_{}_param_{}".format(group_index, param_index),
                        existing_names,
                    )
            else:
                param_name = str(param_name).lower()
            param_names.append(param_name)
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
                "lr": group_lr,
                "beta1": group_betas[0],
                "beta2": group_betas[1],
                "eps": group_eps,
                "weight_decay": group_weight_decay,
            }
        )
        super().add_param_group(new_group)
        for param, param_name in zip(params, param_names):
            self._param_names[param] = param_name
            self.state[param]["name"] = param_name

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
        if (
            not self.capturable
            and torch.cuda.is_available()
            and torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError(
                "Attempting CUDA graph capture of {}.step() but capturable=False. "
                "Construct the optimizer with capturable=True to make step() "
                "graph-safe.".format(type(self).__name__)
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
        acc += sum(
            map(id, chain.from_iterable(map(dict.values, state.values())))
        )
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
            self._gefen_codebook_lut_by_device[
                self._gefen_codebook.device
            ] = lut
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
            seed = torch.full(
                (), int(self._gefen_global_step),
                dtype=torch.int64, device=device,
            )
            self._sr_seed_by_device[device] = seed
        return seed

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
        if buf is None:
            device = state["step"].device
            buf = torch.ones(4, dtype=torch.float32, device=device)
            state["_capt_scalars"] = buf
            # [beta2, beta1] in float64, ordered to line up with the
            # [bc2_step, step] stack below.
            state["_capt_consts"] = torch.tensor(
                [group["beta2"], group["beta1"]],
                dtype=torch.float64,
                device=device,
            )
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
                        [
                            self.state[p][key].detach()
                            for p, key in zip(rows, d["keys"])
                        ]
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
        if stacks and all(
            len(device_stacks) == 1 for device_stacks in stacks.values()
        ):
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
                    or stack["betas"][i]
                    != (group["beta1"], group["beta2"])
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

    def _gefen_codebook_lut_on(
        self, device: torch.device
    ) -> Optional[torch.Tensor]:
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
        if self._force_1d_period_one and param.ndim == 1:
            return 1
        if self._force_2d_period_one and param.ndim == 2:
            return 1
        if self._period_one_substrings:
            lname = str(param_name).lower()
            if any(sub in lname for sub in self._period_one_substrings):
                return 1
        return self._predict_period_from_grad_sq(param_name, param, grad)

    def _predict_period_from_grad_sq(
        self, param_name: str, param: torch.Tensor, grad: torch.Tensor
    ) -> int:
        backend = _resolve_find_period_backend(grad)
        # If the fused CUDA toolchain GENUINELY failed to build (the step-1 probe
        # ran and returned False -> _fused_build_ok is False), fall back to the
        # pure-torch "gpu" period backend rather than crashing in the separate
        # period-variance extension load. Gate on the explicit build-failure
        # verdict, NOT on _gefen_fused_toolchain_ok(): the latter also returns
        # False for a user-chosen fused=False optimizer, which must keep using
        # the cuda_kernel period backend (bit-identical to before this scrub).
        # Only override the AUTO resolution -- an explicit FIND_PERIOD_BACKEND is
        # the user's deliberate choice.
        if (
            backend == "cuda_kernel"
            and FIND_PERIOD_BACKEND is None
            and grad.device.type == "cuda"
            and self._fused_build_ok is False
        ):
            backend = "gpu"
        if backend == "cuda_kernel":
            grad_work = grad.to_local() if hasattr(grad, "to_local") else grad
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
            period_input = grad.detach().float().square().reshape(-1).cpu().numpy()
        elif backend == "gpu":
            if grad.device.type != "cuda":
                raise ValueError("FIND_PERIOD_BACKEND='gpu' requires a CUDA tensor")
            period_input = grad.detach().float().square().reshape(-1)
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

    def _iter_gefen_grad_periods(self, reuse_existing_periods: bool = False):

        for group in self.param_groups:
            for param_name, p in self._iter_group_params_with_names(group):
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

                self.state[p]["automatic_period"] = period

                if flat.numel() % period != 0:
                    raise ValueError(
                        "Automatic partition period {} does not divide parameter {} with numel {} while learning Gefen codebook".format(
                            period,
                            param_name,
                            flat.numel(),
                        )
                    )

                yield param_name, flat, period, grad

    def _learn_gefen_exact_codebook(
        self,
        reuse_existing_periods: bool = False,
        compute_mse_logging: bool = True,
    ) -> Optional[torch.Tensor]:
        # Fire the fused-toolchain probe ONCE here, at the first fused CUDA
        # extension boundary (step-1 codebook learning), so a broken toolchain
        # is discovered and downgraded before the histogram / period-finding /
        # update paths run. On a healthy box this just builds the extension a
        # little earlier; for fused=False the probe is a no-op (never builds).
        if self.fused:
            self._gefen_fused_toolchain_ok()
        codebook = learn_gefen_exact_codebook_from_grad_periods(
            grad_periods=self._iter_gefen_grad_periods(
                reuse_existing_periods=reuse_existing_periods
            ),
            codebook_device=self._gefen_codebook_device(),
            num_codebooks=256,
            force_endpoints=True,
            verbose=self.verbose,
            compute_mse_logging=compute_mse_logging,
            # Use the fused histogram UNLESS the toolchain probe explicitly
            # failed (_fused_build_ok is False). Gating on the build-failure
            # verdict rather than self.fused preserves the pre-scrub behavior of
            # the histogram (it ran on CUDA tensors regardless of the optimizer's
            # fused flag), so a fused=False run stays bit-identical; only a
            # genuinely broken toolchain drops to the pure-torch bincount path.
            use_fused_histogram=(
                FUSE_HISTOGRAM_FOR_EXACT and self._fused_build_ok is not False
            ),
        )
        if codebook is None:
            return None

        return codebook

    def _ensure_gefen_codebook(self, reuse_existing_periods: bool = False) -> None:
        codebook = self._learn_gefen_exact_codebook(
            reuse_existing_periods=reuse_existing_periods,
            compute_mse_logging=False,
        )
        if codebook is not None:
            self._gefen_codebook = codebook
            self._gefen_codebook_by_device.clear()
            self._gefen_codebook_lut_by_device.clear()

    def _step_automatic_factored(
        self, group, param_name: str, p: torch.Tensor, grad: torch.Tensor
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
        state = self.state[p]
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
                automatic_period = self._resolve_automatic_period(
                    param_name, p, grad
                )
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
                    "Expected Gefen codebook to be initialized before the "
                    "factored update."
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

        # Decomposed fallback (CPU / fused disabled). Row/col mean-square EMAs
        # over bounded fp32 row-chunks (one chunk serves both reductions, so
        # the transient stays ~a few chunk sizes), then whole-tensor torch ops.
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

        # Decomposed fallback (CPU / fused disabled): whole-tensor torch ops.
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
        old_codebook = self._gefen_codebook
        if old_codebook is None:
            return
        new_codebook = self._learn_gefen_exact_codebook(
            reuse_existing_periods=True, compute_mse_logging=False
        )
        if new_codebook is None:
            return
        for pgroup in self.param_groups:
            for p in pgroup["params"]:
                pstate = self.state.get(p)
                if not pstate or "m_codebook" not in pstate:
                    continue
                stored_device = pstate["m_codebook"].device
                old_codebook_local = self._gefen_codebook_on(stored_device)
                new_codebook_local = new_codebook.to(stored_device)
                coeffs = gefen_dequantize_unpacked_indices(
                    old_codebook_local, pstate["m_codebook"], old_codebook_local
                )
                indices = gefen_nearest_codebook_indices(new_codebook_local, coeffs)
                gefen_set_unpacked_indices(pstate["m_codebook"], indices)
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

        # No codebook yet. On a fresh run this learns it and predicts periods.
        # On resume the persisted codebook is normally restored in
        # load_state_dict, but FSDP optim-state consolidation strips custom
        # top-level keys, so the codebook can be missing even though the
        # per-param state (incl. automatic_period) was restored. In that case we
        # must reuse the restored periods rather than re-predict them, otherwise
        # the refreshed periods desync from the restored vmean/m_codebook block
        # geometry and the vmean kernel aborts on a block-count mismatch.
        self._ensure_gefen_codebook(
            reuse_existing_periods=self._resuming_from_checkpoint()
        )

    def _maybe_save_gefen_grad_histogram(self) -> None:
        if not hasattr(quantization_module, "LIST_STEPS_SAVE_HIST_GRAD"):
            return
        requested_steps = quantization_module.LIST_STEPS_SAVE_HIST_GRAD
        if requested_steps is None:
            return
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
        self, state, grad_view, beta1
    ) -> torch.Tensor:
        # Muon-specific quantized-momentum update. Advances the quantized momentum
        # state (m_codebook / m_magnitude) by one EMA step and returns the DENSE
        # quantized momentum for Newton-Schulz in a single kernel pass -- no lr==0
        # dummy-stepsize parameter write, and no second full-size codebook gather.
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
            updated_m = current_m.lerp(
                grad_view[r0:r1].to(current_m.dtype), 1 - beta1
            )

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
        self, group, param_name: str, p: torch.Tensor, grad: torch.Tensor
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

        state = self.state[p]
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
                automatic_period = self._resolve_automatic_period(
                    param_name, p, grad
                )
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
            state["vmean_step"] = (
                step.clone() if torch.is_tensor(step) else step
            )

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
        route_v2 = _should_use_v2_full(
            grad_view.shape[0], automatic_period, grad_view.device
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
            p.to_local()
            if (hasattr(p, "to_local") and hasattr(p, "placements"))
            else p
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
            if torch.is_tensor(lr) and (use_full_fused or use_v2_full):
                # The fused kernel bindings take lr / wd_factor as host
                # doubles; scalarize a tensor lr through the cached no-sync
                # .item() hoist (_lr_scalar) instead of passing the tensor.
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
                1.0 if step_scalars is not None
                else 1.0 - lr * group["weight_decay"]
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
                1.0 if step_scalars is not None
                else 1.0 - lr * group["weight_decay"]
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

        bias_correction_1 = 1 - beta1 ** step_count
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
        torch._foreach_copy_(vmeans, list(merged_vmean.reshape(k, nblocks, 1).unbind(0)))
        torch._foreach_copy_(mags, list(merged_mag.reshape(k, nblocks, 1).unbind(0)))
        torch._foreach_copy_(
            codebooks, list(merged_codebook.reshape(k, nblocks, period).unbind(0))
        )

    def state_dict(self):
        """Serialize optimizer state, plus Gefen's run-level extras.

        Beyond the base ``torch.optim.Optimizer.state_dict`` contents (per-param
        state keyed by index and caller-preserved ``param_groups``), the dict
        carries each parameter's stable ``name`` in its state, plus
        ``gefen_global_step`` and the frozen exact-DP ``gefen_codebook`` (without
        it a resume would re-learn the codebook and desync the restored block
        periods). Per-step scratch buffers are stripped, and non-owning state
        views are compacted to tight clones so checkpoints stay small.
        """
        state_dict = super().state_dict()
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
            # Batched-refresh registration marker (a host int): scratch like
            # the buffers it indexes into; leaking it into a checkpoint would
            # falsely mark restored params as batch-covered.
            "_capt_stack",
            "_capt_row",
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
                {k: _compact(v) for k, v in pstate.items() if k not in scratch_keys}
                if isinstance(pstate, dict)
                else pstate
            )
            for pid, pstate in state_dict["state"].items()
        }
        state_dict["gefen_global_step"] = self._gefen_global_step
        # The exact-DP codebook is learned once on the first step and then frozen
        # for the rest of the run. It is not per-param state, so persist it
        # explicitly; without it resume re-learns the codebook (see
        # _maybe_refresh_gefen_codebook) which re-predicts and overwrites every
        # restored automatic_period, desyncing them from the saved
        # vmean/m_codebook. (Note: under FSDP optim-state consolidation strips
        # these custom top-level keys; _maybe_refresh_gefen_codebook handles that
        # fallback by reusing the restored periods.)
        state_dict["gefen_codebook"] = self._gefen_codebook
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
        ignored = {"params", "name", "param_names"}
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
        """Restore optimizer state saved by :meth:`state_dict`.

        On top of the base load this restores ``gefen_global_step`` and the
        frozen codebook, and undoes the base class's dtype coercion of aux
        state: ``torch.optim`` casts every floating state tensor to the owning
        param's dtype, which would corrupt Gefen's fp32 moments and uint8
        momentum indices on bf16 models, so each aux tensor is written back in
        its ORIGINAL saved dtype. Step counters are normalized to this
        optimizer's ``capturable`` mode (host ints vs 0-dim device tensors), so
        checkpoints are portable across the toggle in either direction.
        """
        # Shallow-copy so the pop()s below don't mutate the caller's checkpoint
        # dict (torch.optim.Optimizer.load_state_dict leaves its input intact);
        # otherwise a second load of the same dict loses the gefen_* keys.
        state_dict = dict(state_dict)
        gefen_global_step = state_dict.pop("gefen_global_step", 0)
        gefen_codebook = state_dict.pop("gefen_codebook", None)
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

        super().load_state_dict(state_dict)
        self._gefen_global_step = gefen_global_step
        # Capturable SR seeds are optimizer-level scratch (a device mirror of
        # gefen_global_step): drop them so the first post-load SR kernel call
        # rebuilds them from the restored counter. (Under manual graph-replay
        # training the host counter -- like every host-side value -- reflects
        # host step() calls, not replays; checkpoint semantics are unchanged
        # from the pre-SR capturable behavior.)
        self._sr_seed_by_device.clear()

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
                if torch.is_tensor(live_value) and live_value.dtype == saved_value.dtype:
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
        self._assert_capturable_if_capturing()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

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
                # Factored-v routing (opt-in): plain 2D params keep their
                # quantized momentum but step through the factored-second-
                # moment path (in-kernel elementwise stepsize on the fused
                # path) instead of any vmean path. Sharded (DTensor) params
                # fall through to the standard path: local-shard row/col
                # statistics would be wrong under sharding.
                if (
                    self._factored_v_2d
                    and p.ndim == 2
                    and not hasattr(p, "placements")
                ):
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
        self._gefen_global_step += 1

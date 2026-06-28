import math
import os
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
from gefen.kernels.automatic_gefen_fused import _should_use_v2

FIND_PERIOD_BACKEND = None
FUSE_AUTOMATIC_VMEAN_UPDATE = True
FUSE_GEFEN_AUTOMATIC_STEP = True
FUSE_HISTOGRAM_FOR_EXACT = True

# Dispatch between the two bit-identical fused update kernels (v1 single-pass,
# v2 two-phase) is decided per call inside automatic_gefen_fused_update_cuda from
# the param's num_blocks / period vs the device SM count -- the real driver of
# which kernel is faster (the old arch allowlist was a proxy for this). Env
# GEFEN_UPDATE_V2 still hard-forces the choice. See fused_bench/PERF.md.


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
        block_mean_grad_sq = torch.mean(grad_view * grad_view, dim=1, keepdim=True)
        vmean.mul_(beta2).add_(block_mean_grad_sq, alpha=1 - beta2)
        return
    vmean.mul_(beta2)
    for r0 in range(0, num_blocks, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, num_blocks)
        g = grad_view[r0:r1]
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
    if hasattr(quantization_module, "SAVE_CODEBOOK_PREFIX"):
        save_codebook_prefix = quantization_module.SAVE_CODEBOOK_PREFIX
        if save_codebook_prefix is not None:
            if not hasattr(quantization_module, "LIST_STEPS_SAVE_HIST_GRAD"):
                raise ValueError(
                    "Gefen exact codebook learning does not support SAVE_CODEBOOK_PREFIX exports. "
                    "Disable exports or define LIST_STEPS_SAVE_HIST_GRAD for the explicit histogram export path."
                )
            elif quantization_module.LIST_STEPS_SAVE_HIST_GRAD is None:
                raise ValueError(
                    "Gefen exact codebook learning does not support SAVE_CODEBOOK_PREFIX exports. "
                    "Disable exports or use a non-exact research path outside the publishable Gefen optimizer."
                )

    histogram_bins = num_codebooks * 16
    bin_width = 2.0 / float(histogram_bins)
    bin_counts_cpu = torch.zeros(histogram_bins, dtype=torch.float32, device="cpu")
    total_numel = 0

    prev_mode = torch.get_deterministic_debug_mode()
    torch.set_deterministic_debug_mode(0)
    try:
        for _, flat, period, _ in grad_periods:
            flat_float = flat.float()
            if use_fused_histogram and flat_float.device.type == "cuda":
                from gefen.kernels.exact_histogram_fused import (
                    gefen_exact_histogram_cuda,
                )

                local_counts_cuda = torch.zeros(
                    histogram_bins, dtype=torch.int64, device=flat_float.device
                )
                gefen_exact_histogram_cuda(flat_float, period, local_counts_cuda)
                bin_counts_cpu.add_(local_counts_cuda.cpu().to(torch.float32))
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
                ).float()
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

    if verbose:
        variant = "ForceOne (-1 and 1 fixed)" if force_endpoints else "standard"
        histogram_backend = (
            "fused cuda kernel" if use_fused_histogram else "reference pytorch"
        )

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
    global FIND_PERIOD_BACKEND
    if FIND_PERIOD_BACKEND is not None:
        return FIND_PERIOD_BACKEND

    grad_work = grad.to_local() if hasattr(grad, "to_local") else grad
    if grad_work.device.type == "cuda":
        FIND_PERIOD_BACKEND = "cuda_kernel"
    else:
        FIND_PERIOD_BACKEND = "cpu"
    return FIND_PERIOD_BACKEND


class Gefen(torch.optim.Optimizer):

    def __init__(
        self,
        params: Iterable[Union[nn.Parameter, Tuple[str, nn.Parameter]]],
        lr: Union[float, torch.Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        fused: bool = True,
        verbose: bool = False,
    ):
        if fused and not torch.cuda.is_available():
            print(
                "Gefen Optimizer:  got fused=True, but CUDA is not available. Changing fused to False."
            )
            fused = False
        print("Initializing Gefen optimizer (fused={}).".format(fused))

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        def validate_group_options(
            group_lr, group_betas, group_eps, group_weight_decay
        ):
            if not 0.0 <= group_lr:
                raise ValueError("Invalid learning rate: {}".format(group_lr))
            elif not 0.0 <= group_betas[0] < 1.0:
                raise ValueError(
                    "Invalid beta parameter at index 0: {}".format(group_betas[0])
                )
            elif not 0.0 <= group_betas[1] < 1.0:
                raise ValueError(
                    "Invalid beta parameter at index 1: {}".format(group_betas[1])
                )
            elif not 0.0 <= group_weight_decay:
                raise ValueError(
                    "Invalid weight_decay value: {}".format(group_weight_decay)
                )
            elif not 0.0 <= group_eps:
                raise ValueError("Invalid epsilon value: {}".format(group_eps))

        def iter_params_with_names(group_params):
            if isinstance(group_params, torch.Tensor):
                yield None, group_params
                return
            for item_index, param_or_named_param in enumerate(group_params):
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
                yield param_name, param

        def append_param_group(
            param_name, param, group_lr, group_betas, group_eps, group_weight_decay
        ):
            param_name = str(param_name).lower()
            if not param.requires_grad:
                return
            optim_groups.append(
                {
                    "name": param_name,
                    "params": param,
                    "lr": group_lr,
                    "beta1": group_betas[0],
                    "beta2": group_betas[1],
                    "eps": group_eps,
                    "weight_decay": group_weight_decay,
                }
            )

        self.fused = fused
        self.verbose = verbose

        self._printed_optimizer_memory = False
        self._gefen_codebook = None
        self._gefen_global_step = 0
        self._printed_gefen_materialization = False

        optim_groups = []
        for group_index, param_or_group in enumerate(params):
            if isinstance(param_or_group, dict):
                if "params" not in param_or_group:
                    raise ValueError(
                        "Gefen parameter group {} is missing the 'params' key.".format(
                            group_index
                        )
                    )
                group_lr = param_or_group.get("lr", lr)
                group_betas = param_or_group.get("betas", betas)
                group_eps = param_or_group.get("eps", eps)
                group_weight_decay = param_or_group.get("weight_decay", weight_decay)
                validate_group_options(
                    group_lr, group_betas, group_eps, group_weight_decay
                )
                for param_index, (param_name, param) in enumerate(
                    iter_params_with_names(param_or_group["params"])
                ):
                    if param_name is None:
                        param_name = "group_{}_param_{}".format(
                            group_index, param_index
                        )
                    append_param_group(
                        param_name,
                        param,
                        group_lr,
                        group_betas,
                        group_eps,
                        group_weight_decay,
                    )
            else:
                for param_name, param in iter_params_with_names((param_or_group,)):
                    if param_name is None:
                        param_name = "param_{}".format(group_index)
                    append_param_group(param_name, param, lr, betas, eps, weight_decay)

        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], eps=eps)
        super().__init__(optim_groups, defaults)

    def _use_fused_automatic_vmean(self) -> bool:
        return self.fused and FUSE_AUTOMATIC_VMEAN_UPDATE

    def _use_fused_gefen_automatic_step(self) -> bool:
        return self.fused and FUSE_GEFEN_AUTOMATIC_STEP

    def _automatic_vmean_update(
        self,
        vmean: torch.Tensor,
        grad_view: torch.Tensor,
        beta2: float,
    ) -> None:
        automatic_vmean_update(
            vmean, grad_view, beta2, use_fused=self._use_fused_automatic_vmean()
        )

    def _gefen_codebook_device(self) -> torch.device:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    return p.grad.device
                return p.device
        raise ValueError(
            "Expected at least one parameter when choosing the Gefen codebook device."
        )

    def _gefen_nearest_indices(self, normalized_vals: torch.Tensor) -> torch.Tensor:

        codebook = self._gefen_codebook
        if codebook is None:
            raise ValueError(
                "Expected Gefen codebook to be initialized before nearest-index lookup."
            )
        return gefen_nearest_codebook_indices(codebook, normalized_vals)

    def _gefen_dequantize_m_coefficients(
        self, state, like_tensor: torch.Tensor
    ) -> torch.Tensor:

        stored = state["m_codebook"]
        codebook = self._gefen_codebook
        if codebook is None:
            raise ValueError(
                "Expected Gefen codebook to be initialized before dequantizing m coefficients."
            )
        return gefen_dequantize_unpacked_indices(codebook, stored, like_tensor)

    def _gefen_set_indices(self, state, indices: torch.Tensor) -> None:
        gefen_set_unpacked_indices(state["m_codebook"], indices)

    def _print_gefen_codebook(self, label: str, codebook: torch.Tensor) -> None:
        if self.verbose:
            print("{} ({} entries):".format(label, codebook.numel()))

    def _automatic_view(self, flat_tensor: torch.Tensor, period: int) -> torch.Tensor:
        return automatic_partition_view(flat_tensor, period)

    def _automatic_reduce(
        self, flat_tensor: torch.Tensor, period: int, reduce_op: str
    ) -> torch.Tensor:
        return automatic_partition_reduce(flat_tensor, period, reduce_op)

    def _predict_period_from_grad_sq(
        self, param_name: str, param: torch.Tensor, grad: torch.Tensor
    ) -> int:
        backend = _resolve_find_period_backend(grad)
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
                if not self._printed_gefen_materialization:
                    pass

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
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
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
                                group["name"],
                                self._gefen_global_step,
                            )
                        )
                    period = state["automatic_period"]
                elif flat.numel() == 1:
                    period = 1
                else:
                    period = self._predict_period_from_grad_sq(group["name"], p, grad)

                self.state[p]["automatic_period"] = period

                if flat.numel() % period != 0:
                    raise ValueError(
                        "Automatic partition period {} does not divide parameter {} with numel {} while learning Gefen codebook".format(
                            period,
                            group["name"],
                            flat.numel(),
                        )
                    )

                yield group["name"], flat, period, grad

    def _learn_gefen_exact_codebook(
        self,
        reuse_existing_periods: bool = False,
        compute_mse_logging: bool = True,
    ) -> Optional[torch.Tensor]:
        codebook = learn_gefen_exact_codebook_from_grad_periods(
            grad_periods=self._iter_gefen_grad_periods(
                reuse_existing_periods=reuse_existing_periods
            ),
            codebook_device=self._gefen_codebook_device(),
            num_codebooks=256,
            force_endpoints=True,
            verbose=self.verbose,
            compute_mse_logging=compute_mse_logging,
            use_fused_histogram=FUSE_HISTOGRAM_FOR_EXACT,
        )
        if codebook is None:
            return None

        return codebook

    def _ensure_gefen_codebook(self, reuse_existing_periods: bool = False) -> None:
        compute_mse_logging = False
        if "COMPUTE_GEFEN_EXACT_MSE_LOGGING" in globals():
            compute_mse_logging = COMPUTE_GEFEN_EXACT_MSE_LOGGING
        codebook = self._learn_gefen_exact_codebook(
            reuse_existing_periods=reuse_existing_periods,
            compute_mse_logging=compute_mse_logging,
        )
        if codebook is not None:
            self._gefen_codebook = codebook

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
        state["m_codebook_shape"] = tuple(grad_view.shape)
        state["m_codebook"] = torch.zeros_like(
            grad_view, dtype=torch.uint8, memory_format=torch.preserve_format
        )

        state["m_magnitude"] = torch.zeros_like(
            grad_view[:, 0:1],
            dtype=torch.float32,
            memory_format=torch.preserve_format,
        )

    def _automatic_gefen_fused_update(
        self, p, state, grad_view, beta1, stepsize, lr
    ) -> None:

        codebook = self._gefen_codebook
        if codebook is None:
            raise ValueError(
                "Expected Gefen codebook to be initialized before fused update."
            )
        gefen_automatic_fused_update(
            p=p,
            grad_view=grad_view,
            m_codebook=state["m_codebook"],
            m_magnitude=state["m_magnitude"],
            stepsize=stepsize,
            codebook=codebook,
            packed_indices=False,
            beta1=beta1,
            lr=lr,
        )

    def _gefen_quantized_momentum_update(
        self, state, grad_view, beta1
    ) -> torch.Tensor:
        # Muon-specific quantized-momentum update. Advances the quantized momentum
        # state (m_codebook / m_magnitude) by one EMA step and returns the DENSE
        # quantized momentum for Newton-Schulz in a single kernel pass -- no lr==0
        # dummy-stepsize parameter write, and no second full-size codebook gather.
        codebook = self._gefen_codebook
        if codebook is None:
            raise ValueError(
                "Expected Gefen codebook to be initialized before quantized momentum update."
            )
        momentum_out = torch.empty_like(grad_view)
        _gefen_quantized_momentum_update_cuda(
            grad_view,
            state["m_codebook"],
            state["m_magnitude"],
            codebook,
            momentum_out,
            beta1,
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
    ) -> None:
        # Tier-1 fully-fused v1 path: the kernel computes the vmean EMA and the
        # per-block stepsize in-kernel, so the separate vmean kernel and the host
        # sqrt/div/reciprocal/mul launches are skipped. The scalar reciprocals are
        # precomputed here as Python doubles and cast to float32 in the kernel
        # exactly as PyTorch casts the scalar operands of the host div_/mul_.
        # PyTorch lowers `h.div_(sqrt(bc2))` to a multiply by float32(1/sqrt(bc2))
        # (a true divide differs by up to ~2 ULP), so pass the reciprocal directly.
        codebook = self._gefen_codebook
        if codebook is None:
            raise ValueError(
                "Expected Gefen codebook to be initialized before fused update."
            )
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
            lr,
            eps,
            1.0 / math.sqrt(bias_correction_2),
            1.0 / bias_correction_1,
            weight_decay_factor,
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
    ) -> None:
        # Tier-1 fully-fused v2 (two-phase) path for occupancy-flexible params:
        # the magnitude phase also accumulates Sum(grad^2), a tiny finalize forms
        # the vmean EMA + per-block stepsize, and weight decay is folded into the
        # update write -- so the separate vmean kernel and the host
        # stepsize/weight-decay passes are skipped. The scalar reciprocals are
        # precomputed as Python doubles and cast in-kernel exactly as PyTorch
        # casts the operands of the host div_/mul_ (see the v1-full path).
        codebook = self._gefen_codebook
        if codebook is None:
            raise ValueError(
                "Expected Gefen codebook to be initialized before fused update."
            )
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
            lr,
            eps,
            1.0 / math.sqrt(bias_correction_2),
            1.0 / bias_correction_1,
            weight_decay_factor,
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

        codebook = self._gefen_codebook
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

        codebook = self._gefen_codebook
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
                automatic_period = self._predict_period_from_grad_sq(
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
            state["step"] = 0
            grad_view = self._automatic_view(flat_grad, automatic_period)
            self._init_gefen_state(state, grad_view)

            state["vmean"] = torch.zeros_like(
                grad_view[:, 0:1],
                dtype=torch.float32,
                memory_format=torch.preserve_format,
            )
            self._print_v_period(param_name, p, state["vmean"])

        automatic_period = state["automatic_period"]
        grad_view = self._automatic_view(flat_grad, automatic_period)

        # Tier-1: well-occupied (v1-regime) params take the fully-fused kernel,
        # which folds the vmean EMA + per-block stepsize math (and, after K3,
        # weight decay) into the single update kernel -- skipping the separate
        # vmean kernel and the host sqrt/div/reciprocal/mul launches. Tiny-period
        # / few-block params stay on the decomposed v2 path, and the non-fused
        # path stays decomposed. The routing predicate is the same one the v1/v2
        # update dispatch uses, so the choice is consistent.
        fused_step = self._use_fused_gefen_automatic_step()
        route_v2 = _should_use_v2(
            grad_view.shape[0], automatic_period, grad_view.device
        )
        use_full_fused = fused_step and not route_v2
        use_v2_full = fused_step and route_v2

        # Only the non-fused fallback still needs the standalone vmean kernel; the
        # v1-full and v2-full fused paths both compute the vmean EMA in-kernel.
        if not (use_full_fused or use_v2_full):
            self._automatic_vmean_update(state["vmean"], grad_view, beta2)

        state["step"] += 1

        # Under FSDP2 the parameter is a sharded DTensor; every in-place write
        # below (weight decay, fused kernel, non-fused add_) must land on the
        # local shard's storage. DTensor.to_local() returns a view of that
        # storage, so in-place ops propagate back to the param.
        p_local = (
            p.to_local()
            if (hasattr(p, "to_local") and hasattr(p, "placements"))
            else p
        )

        bias_correction_1 = 1 - beta1 ** state["step"]
        bias_correction_2 = 1 - beta2 ** state["step"]

        if use_full_fused:
            # K1+K2+K3: vmean EMA, per-block stepsize, and weight decay are all
            # computed inside the fused kernel. weight_decay_factor == 1 - lr*wd
            # mirrors the host p.mul_(1 - lr*wd) pass (an exact identity when
            # wd == 0). The kernel flat-indexes p and requires it contiguous; a
            # non-contiguous shard is updated on a contiguous copy that is copied
            # back to preserve the in-place semantics.
            weight_decay_factor = 1.0 - lr * group["weight_decay"]
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
                )
                p_local.copy_(p_contig)
            return

        if use_v2_full:
            # K1+K2+K3 on the occupancy-flexible two-phase path: vmean EMA,
            # per-block stepsize and weight decay are folded into the v2 kernels
            # (see _automatic_gefen_fused_update_v2_full). Same contiguity
            # handling as the v1-full branch.
            weight_decay_factor = 1.0 - lr * group["weight_decay"]
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
        h.div_(math.sqrt(bias_correction_2)).add_(eps)
        torch.reciprocal(h, out=stepsize)
        stepsize.mul_(1.0 / bias_correction_1)

        if self._use_fused_gefen_automatic_step():
            # The fused kernel flat-indexes p and requires it contiguous. A
            # contiguous DTensor shard is written in place directly; a
            # non-contiguous shard is updated on a contiguous copy that is then
            # copied back so the in-place semantics are preserved.
            if p_local.is_contiguous():
                self._automatic_gefen_fused_update(
                    p_local,
                    state,
                    grad_view,
                    beta1,
                    stepsize,
                    lr,
                )
            else:
                p_contig = p_local.contiguous()
                self._automatic_gefen_fused_update(
                    p_contig,
                    state,
                    grad_view,
                    beta1,
                    stepsize,
                    lr,
                )
                p_local.copy_(p_contig)

            return

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
            self.state[p]["step"] += 1
        step_count = first_state["step"]

        if weight_decay > 0.0:
            torch._foreach_mul_(params, 1 - lr * weight_decay)

        bias_correction_1 = 1 - beta1 ** step_count
        bias_correction_2 = 1 - beta2 ** step_count
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

    def _optimizer_state_memory_bytes(self) -> int:
        total = 0
        for state in self.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    total += value.numel() * value.element_size()
        return total

    def state_dict(self):
        state_dict = super().state_dict()
        # stepsize/_h_buf are per-step scratch buffers (recomputed from vmean every
        # step); they live in self.state only to be reused across steps, so strip
        # them from the serialized dict instead of bloating every checkpoint. Build
        # fresh per-param dicts so live self.state is left untouched.
        scratch_keys = ("stepsize", "_h_buf")

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

    def load_state_dict(self, state_dict):
        # Shallow-copy so the pop()s below don't mutate the caller's checkpoint
        # dict (torch.optim.Optimizer.load_state_dict leaves its input intact);
        # otherwise a second load of the same dict loses the gefen_* keys.
        state_dict = dict(state_dict)
        gefen_global_step = state_dict.pop("gefen_global_step", 0)
        gefen_codebook = state_dict.pop("gefen_codebook", None)

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

        super().load_state_dict(state_dict)
        self._gefen_global_step = gefen_global_step

        # Restoring the frozen codebook keeps _maybe_refresh_gefen_codebook a
        # no-op on the first resume step, so the restored automatic_period values
        # stay consistent with the restored vmean/m_codebook block geometry.
        # Assign unconditionally: when FSDP strips the codebook (gefen_codebook is
        # None) any stale codebook from a prior train/load must be cleared, else
        # _maybe_refresh_gefen_codebook returns early and skips the restored-period
        # fallback.
        self._gefen_codebook = gefen_codebook

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

    @torch.no_grad()
    def step(self, closure=None):
        self._maybe_refresh_gefen_codebook()
        self._maybe_save_gefen_grad_histogram()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # The fused path has its own batched CUDA kernel per param; only the
        # non-fused Python-loop path benefits from foreach/merged batching.
        # Gate on BOTH fused flags: the merged path hard-codes use_fused=False for
        # the vmean update, so it must not engage while fused vmean is still on
        # (e.g. FUSE_GEFEN_AUTOMATIC_STEP off but FUSE_AUTOMATIC_VMEAN_UPDATE on),
        # which would otherwise change the vmean math vs the per-param path.
        batch_nonfused = (
            not self._use_fused_gefen_automatic_step()
            and not self._use_fused_automatic_vmean()
        )

        # Collect parameters that share block geometry + hyperparameters so they
        # can be processed as one merged tensor (see _step_automatic_merged).
        # Anything not eligible (fused path, first step before state exists,
        # DTensor/FSDP2 shards, LR-Tensor schedules, or LARGE params that should
        # stream) keeps the per-param path.
        nonfused_groups = {}
        for group in self.param_groups:
            name = group["name"]
            for p in group["params"]:
                grad = p.grad
                if grad is None:
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

        self._gefen_global_step += 1
        return loss

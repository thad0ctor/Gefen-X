import math
import warnings
from collections import OrderedDict
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn

from gefen.gefen import Gefen

EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5


def _stable_distributed_owner(stable_index: int, world: int) -> int:
    """Map a stable full-param-set index to a process-group rank coordinate."""
    if world <= 0:
        raise ValueError("world must be positive, got {}".format(world))
    if stable_index < 0:
        raise ValueError("stable_index must be non-negative, got {}".format(stable_index))
    return stable_index % world

# Tuned per-iteration Newton-Schulz coefficient schedules. Each entry is a list
# of (a, b, c) quintic coefficients applied one per iteration. Unlike the fixed
# (DEFAULT_A, DEFAULT_B, DEFAULT_C) quintic -- which uses identical, conservative
# coefficients for all ns_steps iterations -- these schedules deliberately
# overshoot in the early iterations (large a, very negative b) to amplify the
# small singular values fast, then refine, reaching the same orthogonality in
# fewer iterations. Derived by minimax optimization of the K-fold composition of
# p(s) = a s + b s^3 + c s^5 over s in [s_min, 1] (see
# benchmarks/microbench/derive_ns_schedule.py). NS is GEMM-FLOP-bound on the
# real Muon shapes, so K iterations cost ~K/5 of the standard 5-step NS.
# 3-step (40% fewer iterations). Robust over s in [1e-2, 1] with peak|p| <= 1.15
# (safe on near-low-rank gradients). Slightly below the standard 5-step on the
# very smallest singular values (3 safe quintic steps cannot amplify s < ~1e-2
# all the way to 1) -- a speed/quality trade, not a strict win.
NS_SCHEDULE_3STEP = [
    (5.0067, -14.4125, 10.6001),
    (3.7534, -6.8083, 3.3840),
    (3.2080, -4.3501, 2.0330),
]
# 4-step (20% fewer iterations). Robust over s in [1e-2, 1] with peak|p| <= 1.01
# (very safe). Beats the standard 5-step orthogonality on the real Muon shapes.
NS_SCHEDULE_4STEP = [
    (5.4261, -15.5285, 11.2657),
    (3.0766, -4.9701, 2.1297),
    (3.0008, -6.4486, 4.3696),
    (2.8386, -4.1421, 2.5758),
]
# Named schedules selectable via the GefenMuon ``ns_schedule`` group option.
NS_SCHEDULES = {
    "standard": None,  # use ns_coefficients/ns_steps as-is (classic quintic)
    "tuned3": NS_SCHEDULE_3STEP,
    "tuned4": NS_SCHEDULE_4STEP,
}

# fp8 (e4m3) Newton-Schulz support. e4m3 has a max representable magnitude of
# 448; we per-row scale every operand into [-448, 448] before the fp8 GEMM and
# undo the scale in the (bf16/fp32-accumulated) output via torch._scaled_mm.
_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = 448.0
_FP8_MIN_SCALE = 1e-12
# fp8 tensor-core GEMMs (the e4m3 path used here) need sm_89+ (Ada / Hopper /
# Blackwell). Older GPUs (e.g. Ampere sm_80/86) have no fp8 GEMM.
_FP8_MIN_CAPABILITY = (8, 9)
# fp8 only beats bf16 once the matrix is large enough for the fp8 tensor-core GEMM
# to outrun cuBLAS bf16; below that the fp8 path is a net loss. Measured square
# crossover on sm_120 (RTX 5090, NS call, torch.compile-fused): min-dim 1024->0.77x,
# 1280->0.77x, 1536->1.23x, 2048->1.31x. So even on a supported GPU, fp8 is used
# only when the matrix's smaller dim >= this. 1024 admitted the losing Qwen3-0.6B
# shapes (all min-dim 1024, 0.84-0.93x); 1536 is the measured break-even.
FP8_MIN_DIM = 1536
_ns_fp8_compiled = None
_fp8_fallback_warned = False

# Shape-batched Newton--Schulz is an explicit numerical/performance trade-off.
# Strided-batched GEMMs are dramatically faster for repeated small matrices on
# Ampere, but their reduction order is not bit-identical to launching one GEMM
# per matrix.  Keep the conservative, measured gate centralized so constructor,
# routing tests, and benchmarks describe the same supported region.
BATCHED_NS_MIN_BATCH = 8
BATCHED_NS_MAX_MIN_DIM = 512
BATCHED_NS_MAX_NUMEL = 1 << 20
BATCHED_NS_DEFAULT_WORKSPACE_BYTES = 256 << 20


def _batched_ns_shape_eligible(rows: int, cols: int) -> bool:
    """Whether one oriented matrix is inside the measured batching region."""
    return (
        rows > 0
        and cols > 0
        and min(rows, cols) <= BATCHED_NS_MAX_MIN_DIM
        and rows * cols <= BATCHED_NS_MAX_NUMEL
    )


def _fp8_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.get_device_capability() >= _FP8_MIN_CAPABILITY
    except Exception:
        return False


def _quantize_rowwise_fp8(x: torch.Tensor):
    # Per-row absmax scaling into the e4m3 range. Returns (fp8 tensor, fp32
    # scale [rows, 1]) such that fp8_value * scale ~= x.
    scale = (x.abs().amax(dim=1, keepdim=True).float() / _FP8_MAX).clamp(
        min=_FP8_MIN_SCALE
    )
    xq = (x.float() / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)
    return xq, scale


def _ns_fp8_core(ortho_grad: torch.Tensor, schedule) -> torch.Tensor:
    # fp8 Newton-Schulz iteration. The three matmuls (Gram, Gram^2, the
    # gram_update @ X update) run in e4m3 with bf16 accumulation via
    # torch._scaled_mm; the lightweight elementwise combine stays in bf16.
    # Newton-Schulz is iterative and self-correcting, so the per-matmul fp8
    # rounding error largely washes out across iterations. `ortho_grad` is the
    # already-normalized, already-oriented (rows <= cols) bf16 matrix. `schedule`
    # is the resolved per-iteration list of (a, b, c) coefficients, so the fp8
    # path benefits from the tuned schedules (tuned3/tuned4) exactly like bf16.
    for a, b, c in schedule:
        # gram = X @ X.T : both operands are X, so quantize X once. The B side
        # (X.T) is the transpose view of the row-major fp8 X -- it is
        # column-major (stride(0) == 1) and its per-column scale is X's per-row
        # scale, exactly what _scaled_mm rowwise scaling wants.
        xq, sx = _quantize_rowwise_fp8(ortho_grad)
        gram = torch._scaled_mm(
            xq, xq.T, scale_a=sx, scale_b=sx.T, out_dtype=torch.bfloat16
        )
        # gram is symmetric, so gram @ gram reuses a single quantization too.
        gq, sg = _quantize_rowwise_fp8(gram)
        gram_sq = torch._scaled_mm(
            gq, gq.T, scale_a=sg, scale_b=sg.T, out_dtype=torch.bfloat16
        )
        gram_update = (b * gram + c * gram_sq).bfloat16()
        # ortho_grad = a * X + gram_update @ X. gram_update is the row-major A
        # operand; the B operand X must be column-major, which needs a separate
        # column-wise quantization (== row-wise quantization of X.T).
        guq, sgu = _quantize_rowwise_fp8(gram_update)
        xt = ortho_grad.T.contiguous()
        sxt = (xt.abs().amax(dim=1, keepdim=True).float() / _FP8_MAX).clamp(
            min=_FP8_MIN_SCALE
        )
        x_colmajor = (xt.float() / sxt).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE).T
        gux = torch._scaled_mm(
            guq, x_colmajor, scale_a=sgu, scale_b=sxt.T, out_dtype=torch.bfloat16
        )
        ortho_grad = (a * ortho_grad + gux).bfloat16()
    return ortho_grad


def _get_ns_fp8_core(compile_fp8: bool):
    # The fp8 GEMM win only materializes once the per-matmul quantization
    # (amax / scale / cast) is fused into the surrounding kernels, which
    # torch.compile does. Eager fp8 is correctness-equivalent but slower than
    # bf16 at Muon matrix sizes, so compilation is on by default.
    global _ns_fp8_compiled
    if not compile_fp8:
        return _ns_fp8_core
    if _ns_fp8_compiled is None:
        _ns_fp8_compiled = torch.compile(
            _ns_fp8_core, mode="max-autotune-no-cudagraphs", dynamic=False
        )
    return _ns_fp8_compiled


def _normalize_ns_schedule(
    ns_coefficients, ns_steps: int
) -> "list[Tuple[float, float, float]]":
    """Resolve ``ns_coefficients`` / ``ns_steps`` into a per-iteration schedule.

    Accepts either form (backward compatible):
      * a single ``(a, b, c)`` tuple -> the classic fixed quintic, repeated for
        ``ns_steps`` iterations (identical behavior to the original helper);
      * a sequence of ``(a, b, c)`` tuples -> an explicit per-iteration schedule
        whose length IS the iteration count (``ns_steps`` is then ignored).
    """
    if len(ns_coefficients) == 0:
        raise ValueError("ns_coefficients must be non-empty")

    first = ns_coefficients[0]
    is_single_tuple = isinstance(first, (int, float))
    if is_single_tuple:
        # ns_steps only applies to the single-tuple path (it sets the repeat
        # count); an explicit schedule carries its own length and ignores it.
        if not 1 <= ns_steps < 100:
            raise ValueError(
                "ns_steps must be in [1, 100) but is: {}. 0 or a negative value "
                "would produce an EMPTY Newton-Schulz schedule (silently degrading "
                "Muon to normalized momentum SGD); 100+ is rejected for "
                "computational efficiency".format(ns_steps)
            )
        if len(ns_coefficients) != 3:
            raise ValueError(
                "A single coefficient set must be a tuple of exactly 3 values"
            )
        return [tuple(float(x) for x in ns_coefficients)] * int(ns_steps)

    schedule = []
    for entry in ns_coefficients:
        if len(entry) != 3:
            raise ValueError(
                "Each Newton-Schulz schedule entry must have exactly 3 values"
            )
        schedule.append(tuple(float(x) for x in entry))
    if len(schedule) >= 100:
        raise ValueError(
            "Newton-Schulz schedule length must be less than 100 for "
            "computational efficiency"
        )
    return schedule


def _zeropower_via_newtonschulz(
    grad: torch.Tensor,
    ns_coefficients,
    ns_steps: int,
    eps: float,
    use_fp8: bool = False,
    compile_fp8: bool = True,
) -> torch.Tensor:

    if len(grad.shape) != 2:
        raise ValueError("Input tensor gradient must be a 2D matrix")

    # Resolve the (possibly tuned per-iteration) schedule. A single (a, b, c)
    # tuple reproduces the classic fixed quintic bit-for-bit; an explicit
    # sequence is the tuned poly path. Both compose with the fp8 NS path below.
    schedule = _normalize_ns_schedule(ns_coefficients, ns_steps)

    ortho_grad = grad.bfloat16()
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T

    ortho_grad = ortho_grad.div(ortho_grad.norm().clamp(min=eps))
    # Arch-gate fp8: requested but unsupported GPU (pre-sm_89) -> fall back to
    # bf16 (warn once) so `fp8_ns=True` is a portable config, not a hard error.
    if use_fp8 and not _fp8_supported():
        global _fp8_fallback_warned
        if not _fp8_fallback_warned:
            warnings.warn(
                "fp8_ns=True but this GPU is pre-sm_89 (no fp8 GEMM); "
                "falling back to bf16 Newton-Schulz.",
                RuntimeWarning,
            )
            _fp8_fallback_warned = True
        use_fp8 = False
    # Size-gate fp8: it loses to bf16 on small/skinny matrices, so only use it
    # when the smaller dimension is large enough to amortize the quant overhead.
    if use_fp8 and min(grad.shape) < FP8_MIN_DIM:
        use_fp8 = False
    if use_fp8:
        core = _get_ns_fp8_core(compile_fp8)
        ortho_grad = core(ortho_grad.contiguous(), schedule)
    else:
        for a, b, c in schedule:
            gram_matrix = ortho_grad @ ortho_grad.T
            gram_update = torch.addmm(
                gram_matrix,
                gram_matrix,
                gram_matrix,
                beta=b,
                alpha=c,
            )
            ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)

    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    return ortho_grad


def _batched_ns_workspace_bytes(
    batch: int, rows: int, cols: int, element_size: int = 2
) -> int:
    """Conservative tensor-temporary budget for the batched bf16 NS path.

    With ``m=min(rows, cols)`` and ``n=max(rows, cols)``, the implementation
    can simultaneously retain up to three matrix batches and two Gram batches:
    ``3 * B*m*n + 2 * B*m*m`` elements.  The estimate deliberately includes
    the per-item momentum outputs retained while the stack is formed, so the
    configured cap remains meaningful even when allocator reuse is imperfect.
    cuBLAS-internal workspace and allocator fragmentation are outside this
    tensor accounting, so this is a routing budget rather than a hard process
    memory limit.
    """
    m, n = min(rows, cols), max(rows, cols)
    return int(batch) * int(element_size) * (3 * m * n + 2 * m * m)


def _batched_ns_chunk_sizes(
    count: int,
    rows: int,
    cols: int,
    workspace_bytes: int,
    min_batch: int = BATCHED_NS_MIN_BATCH,
) -> "Tuple[list[int], int]":
    """Return eligible balanced batch sizes plus a serial tail count.

    Every returned batch is in ``[min_batch, max_batch_for_workspace]``.  When
    a small remainder can be absorbed by rebalancing the last full batch, do
    that; otherwise leave the undersized tail on the bit-identical serial path.
    """
    if count < min_batch:
        return [], count
    per_item = _batched_ns_workspace_bytes(1, rows, cols)
    max_batch = workspace_bytes // per_item if per_item else count
    if max_batch < min_batch:
        return [], count
    max_batch = min(int(max_batch), int(count))

    full, remainder = divmod(int(count), max_batch)
    sizes = [max_batch] * full
    if remainder == 0:
        return sizes, 0
    if remainder >= min_batch:
        sizes.append(remainder)
        return sizes, 0

    # Rebalance the last full batch and the short remainder when their combined
    # population can form two legal batches.  Example: cap=10, count=17 -> 9+8.
    if sizes and max_batch + remainder >= 2 * min_batch:
        sizes.pop()
        combined = max_batch + remainder
        left = (combined + 1) // 2
        right = combined - left
        sizes.extend((left, right))
        return sizes, 0
    return sizes, remainder


def _zeropower_via_newtonschulz_batched(
    grads: torch.Tensor,
    ns_coefficients,
    ns_steps: int,
    eps: float,
) -> torch.Tensor:
    """Newton--Schulz over a homogeneous ``[B, rows, cols]`` bf16 stack.

    This intentionally uses strided-batched GEMMs, so it is close to but not
    bit-identical with a Python loop over ``_zeropower_via_newtonschulz``.
    Callers must keep it behind the explicit ``batched_ns`` opt-in. A bf16
    ``grads`` tensor is scratch and may be overwritten; optimizer routing passes
    a newly stacked temporary, never a user gradient.
    """
    if grads.ndim != 3:
        raise ValueError("Batched Newton-Schulz expects [batch, rows, cols]")
    schedule = _normalize_ns_schedule(ns_coefficients, ns_steps)
    ortho = grads.bfloat16()
    transposed = ortho.size(1) > ortho.size(2)
    if transposed:
        ortho = ortho.transpose(1, 2).contiguous()

    norms = torch.linalg.vector_norm(ortho, dim=(1, 2), keepdim=True)
    ortho.div_(norms.clamp(min=eps))
    # Ping-pong two matrix batches instead of allocating a third live iterate.
    # The conservative public workspace model intentionally retains its extra
    # headroom for the pre-stack momentum buffers and allocator behavior.
    next_ortho = torch.empty_like(ortho)
    for a, b, c in schedule:
        gram = torch.bmm(ortho, ortho.transpose(1, 2))
        gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        del gram
        torch.baddbmm(
            ortho, gram_update, ortho, beta=a, out=next_ortho
        )
        del gram_update
        ortho, next_ortho = next_ortho, ortho

    if transposed:
        ortho = ortho.transpose(1, 2)
    return ortho


def _cautious_mask_(update: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    # Cautious masking (Cautious Optimizers, Liang et al. 2024): zero every
    # update coordinate whose sign disagrees with the current gradient, then
    # rescale by 1/mean(mask) so the expected update magnitude is preserved.
    # Zero extra state; one elementwise pass over the (already materialized)
    # Newton-Schulz output.
    mask = (update * grad) > 0
    agree = mask.sum()
    scale = mask.numel() / agree.clamp(min=1).to(torch.float32)
    update.mul_(mask.to(update.dtype)).mul_(scale.to(update.dtype))
    return update


def _adjust_lr_ratio(
    adjust_lr_fn: Optional[str], param_shape: torch.Size
) -> float:
    # The lr-adjustment ratio is a static function of the (fixed) param shape,
    # so it is always a python constant -- safe to bake into a CUDA graph.
    rows, cols = param_shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        return math.sqrt(max(1, rows / cols))
    elif adjust_lr_fn == "match_rms_adamw":
        return 0.2 * math.sqrt(max(rows, cols))
    return 1.0


def _adjust_lr(
    lr: float, adjust_lr_fn: Optional[str], param_shape: torch.Size
) -> float:

    if isinstance(lr, torch.Tensor):
        lr = lr.item()
    return lr * _adjust_lr_ratio(adjust_lr_fn, param_shape)


def _swapped_param_groups_error(p: torch.Tensor) -> str:
    """Message for a non-2D parameter discovered in a Muon group at step time.

    GefenMuon validates 2D-ness at construction, so this only fires when an
    optimizer wrapper replaced ``param_groups[*]["params"]`` after construction.
    DeepSpeed ZeRO (stage 1/2/3) is the known case: it swaps in flattened 1-D
    fp32 partitions and steps those, which Muon fundamentally cannot do.
    """
    return (
        "GefenMuon found a {}-D parameter (shape {}) in its param groups at "
        "step time, but every parameter was validated as a 2D matrix at "
        "construction -- an optimizer wrapper has replaced "
        "param_groups[*]['params'] after construction. DeepSpeed ZeRO "
        "(stage 1/2/3) does exactly this: ZeRO steps flattened 1-D fp32 "
        "partitions, and Muon's 2D Newton-Schulz orthogonalization cannot be "
        "applied to a flat partition, so GefenMuon and GefenMuonHybrid cannot "
        "be used as the DeepSpeed ZeRO client optimizer. Use plain Gefen as "
        "the client optimizer under DeepSpeed ZeRO, or train the Muon family "
        "under FSDP2 / DDP / single-GPU instead.".format(p.ndim, tuple(p.shape))
    )


class GefenMuon(Gefen):
    """Muon with Gefen's quantized (8-bit codebook) momentum state.

    Muon (MomentUm Orthogonalized by Newton-Schulz; Keller Jordan et al., 2024,
    https://kellerjordan.github.io/posts/muon/) maintains a momentum buffer per
    2D weight matrix and orthogonalizes the momentum via a Newton-Schulz
    iteration before applying it. GefenMuon stores that momentum in Gefen's
    8-bit learned-codebook format (~1 byte/param optimizer state) and runs the
    same NS pipeline, bit-identically across the fused/non-fused and
    single-GPU/FSDP2-exact paths.

    ONLY 2D parameters are accepted, and 2D is not sufficient: embeddings, the
    (often tied) LM head, and classifier heads are vocabulary/class projections,
    not hidden matrices, and should NOT be given to raw GefenMuon -- route them
    (and every 1D norm/bias tensor) to a plain-Gefen/AdamW backup instead. Use
    ``GefenMuonHybrid`` / ``split_params_for_muon`` (gefen.params) to do that
    split automatically.

    Args:
        params: 2D parameters to optimize, ideally as ``(name, param)`` pairs
            (unique names key the per-param codebook cache; bare tensors get
            positional auto-names).
        lr: learning rate (float, or a 1-element tensor for capturable /
            on-device schedules). Interpretation depends on ``adjust_lr_fn``.
        weight_decay: decoupled (AdamW-style) weight decay. NOTE: defaults to
            0.1 (the common Muon recipe), NOT plain Gefen's 0.0.
        momentum: momentum EMA coefficient in [0, 1) (0.95 is the Muon paper
            default).
        nesterov: apply Nesterov-style momentum (default True).
        ns_coefficients: (a, b, c) quintic coefficients for the classic
            fixed-coefficient Newton-Schulz iteration.
        eps: strictly positive clamp floor for the NS input normalization
            (``grad / max(||grad||, eps)``).
        ns_steps: number of NS iterations in [1, 100) for the fixed-coefficient
            path (ignored when ``ns_schedule`` supplies a per-iteration list).
        ns_schedule: optional tuned per-iteration coefficient schedule: a name
            ("standard"/"tuned3"/"tuned4") or an explicit list of (a, b, c)
            tuples. Overrides ns_coefficients/ns_steps; see NS_SCHEDULES.
        adjust_lr_fn: None/"original" keeps Muon-native LR scaling
            (sqrt(rows/cols)); "match_rms_adamw" rescales each update to
            AdamW-equivalent RMS (0.2*sqrt(max(rows, cols))) so an AdamW-scale
            lr transfers directly (the GefenMuonHybrid default).
        fused: use the fused CUDA momentum kernel when available (default True;
            CPU / non-CUDA falls back to the decomposed path).
        sharded_mode: FSDP2/DTensor handling -- "exact" (default; all-gather the
            full gradient, NS on the full matrix on every rank, bit-identical to
            single-GPU), "distributed" (Parallel-Muon: one owner rank per matrix
            runs NS and broadcasts, bit-identical to "exact" on homogeneous
            GPUs), or "approx" (NS on the local shard only; cheaper and
            explicitly NON-parity, opt-in).
            "distributed" is EXPERIMENTAL but checkpoint-safe when every rank
            participates in ``state_dict()``: each owner broadcasts its local
            momentum state so rank 0 can write a complete checkpoint. Ownership
            is keyed by stable position in the full distributed parameter set
            (not by per-step active position), so a fluctuating grad set does not
            move an active matrix's momentum between ranks.
        fp8_ns: run the NS GEMMs in fp8 (e4m3) on sm_89+ for large matrices
            (opt-in; auto-falls back to bf16 elsewhere).
        fp8_ns_compile: torch.compile the fp8 NS core (default True; eager fp8
            is slower than bf16).
        batched_ns: opt into shape-batched bf16 Newton--Schulz for repeated
            small matrices (default False). This changes GEMM reduction order
            and is therefore not bit-identical to the serial path. It is gated
            to non-sharded, non-capturable CUDA bf16 matrices with batch >= 8,
            min dimension <= 512, and at most 2**20 elements per matrix.
        batched_ns_workspace_bytes: conservative temporary-memory cap used to
            chunk eligible shape groups (default 256 MiB). An undersized tail
            remains on the serial, bit-identical path. This budgets explicit
            tensors, not cuBLAS-internal workspace or allocator fragmentation.
            Changing the budget can change batch sizes and therefore the
            opt-in path's approximate numerical result.
        stochastic_round: stochastically round the 8-bit momentum quantization
            (debiases it; throughput-neutral opt-in).
        normuon: NorMuon-style per-row 2nd-moment normalization of the NS
            output (default False here; GefenMuonHybrid turns it on).
        normuon_beta2: EMA coefficient in [0, 1) for the normuon row statistic.
        normuon_eps: denominator floor for the normuon normalization.
        cautious: cautious-optimizer sign-agreement masking of the update
            (experimental opt-in; LOST in the hybrid's loss sweep).
        capturable: make step() CUDA-graph/torch.compile capturable (device-side
            step counters and scalars; see Gefen).
        verbose: print codebook/quantization diagnostics.
    """

    def __init__(
        self,
        params: Iterable[Union[nn.Parameter, Tuple[str, nn.Parameter]]],
        lr: Union[float, torch.Tensor] = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: Tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        ns_schedule: Optional[object] = None,
        adjust_lr_fn: Optional[str] = None,
        *,
        fused: bool = True,
        sharded_mode: str = "exact",
        fp8_ns: bool = False,
        fp8_ns_compile: bool = True,
        batched_ns: bool = False,
        batched_ns_workspace_bytes: int = BATCHED_NS_DEFAULT_WORKSPACE_BYTES,
        stochastic_round: bool = False,
        normuon: bool = False,
        normuon_beta2: float = 0.95,
        normuon_eps: float = 1e-8,
        cautious: bool = False,
        capturable: bool = False,
        verbose: bool = False,
    ) -> None:
        if isinstance(lr, torch.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError("Learning rate should be >= 0 but is: {}".format(lr))
        if not 0.0 <= momentum < 1.0:
            raise ValueError(
                "momentum should be in [0, 1) but is: {}".format(momentum)
            )
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(
                "eps should be finite and > 0 but is: {}. The Newton-Schulz input "
                "normalization divides by the momentum matrix norm clamped at "
                "eps, so eps=0 turns an all-zero momentum matrix into 0/0 = "
                "NaN".format(eps)
            )
        if not 0.0 <= weight_decay:
            raise ValueError(
                "weight decay should be >= 0 but is: {}".format(weight_decay)
            )
        if adjust_lr_fn is not None and adjust_lr_fn not in [
            "original",
            "match_rms_adamw",
        ]:
            raise ValueError(
                "Adjust learning rate function {} is not supported".format(adjust_lr_fn)
            )
        if sharded_mode not in ("exact", "approx", "distributed"):
            raise ValueError(
                "sharded_mode must be 'exact', 'approx' or 'distributed' but is: "
                "{}".format(sharded_mode)
            )
        # ns_schedule (optional): a tuned per-iteration Newton-Schulz coefficient
        # schedule that reaches orthogonality in fewer iterations than the fixed
        # quintic. Accepts a named schedule ("standard"/"tuned3"/"tuned4") or an
        # explicit sequence of (a, b, c) tuples (one per iteration). When set it
        # OVERRIDES ns_coefficients/ns_steps; the iteration count becomes the
        # schedule length. None (default) keeps the classic fixed-quintic path.
        if ns_schedule is not None:
            if isinstance(ns_schedule, str):
                if ns_schedule not in NS_SCHEDULES:
                    raise ValueError(
                        "Unknown ns_schedule {!r}; choose from {}".format(
                            ns_schedule, sorted(NS_SCHEDULES)
                        )
                    )
                resolved = NS_SCHEDULES[ns_schedule]
            else:
                resolved = ns_schedule
            if resolved is not None:
                # Validate eagerly so misconfigurations fail at construction.
                schedule = _normalize_ns_schedule(resolved, ns_steps)
                ns_coefficients = schedule
                ns_steps = len(schedule)
        # ns_steps only drives the classic fixed-coefficient path (a single
        # (a, b, c) tuple repeated ns_steps times). An explicit per-iteration
        # schedule -- a named tuned schedule or an explicit list of tuples --
        # carries its own length and IGNORES ns_steps, so only validate ns_steps
        # when it will actually be consumed. Mirror _normalize_ns_schedule's own
        # single-tuple detection (a scalar first element) on the RESOLVED
        # coefficients so a list schedule with an out-of-range ns_steps does not
        # raise spuriously.
        if (
            len(ns_coefficients) > 0
            and isinstance(ns_coefficients[0], (int, float))
            and not 1 <= ns_steps < 100
        ):
            raise ValueError(
                "ns_steps must be in [1, 100) but is: {}. 0 or a negative value "
                "would produce an EMPTY Newton-Schulz schedule (silently "
                "degrading Muon to normalized momentum SGD)".format(ns_steps)
            )
        # "exact" (default): under FSDP2 every rank gathers the full gradient and
        # runs Newton-Schulz on the full matrix -> bit-for-bit single-GPU parity.
        # "approx": each rank runs the whole pipeline on its LOCAL shard only --
        # no all-gather and NS on a smaller (row-sharded) matrix, so it is
        # cheaper, but Newton-Schulz of a row block is NOT the orthogonalization
        # of the full matrix: this mode is explicitly NON-PARITY. Opt-in only.
        # "distributed" ("Parallel Muon" / Moonshot): EXACT like "exact", but the
        # redundant Newton-Schulz is removed. Each 2D matrix is assigned by its
        # stable position in the full distributed parameter set to a single owner
        # rank; only that rank runs the quantized momentum + Newton-Schulz on the
        # full matrix, then broadcasts the
        # orthogonalized full-matrix update so every rank slices its own shard.
        # The NS/momentum compute (and the persistent momentum state) is therefore
        # cut ~world_size x while staying bit-for-bit identical to "exact". Only
        # the per-step gradient all-gather (a collective every rank must join) and
        # one extra update broadcast are replicated. Falls back to the "exact"
        # full-NS-everywhere path for non-1D meshes (e.g. HSDP x TP).
        #
        # Homogeneity assumption: "exact" runs Newton-Schulz redundantly on
        # every rank's own GPU and relies on the results agreeing bit-for-bit,
        # which holds only when all ranks have the SAME GPU architecture --
        # cross-arch (e.g. sm_86 + sm_120) bf16 GEMMs differ at ULP scale, so a
        # mixed rig drifts by ~1-2 bf16 ULP per step between "exact" ranks (and
        # between "exact" and "distributed", which broadcasts ONE owner's
        # result and is therefore internally consistent on any rig).
        self._sharded_mode = sharded_mode

        # fp8 Newton-Schulz (opt-in). Default False keeps the bf16 NS path
        # bit-for-bit unchanged. When True, the NS matmuls run in e4m3 with
        # bf16 accumulation; this requires sm_89+ and is only faster than bf16
        # once torch.compile fuses the quantization (fp8_ns_compile, default
        # True). At small Muon shapes (min-dim <~ 1024) bf16 is still faster.
        # Arch-gating is handled per-call in _zeropower_via_newtonschulz (fp8 on
        # sm_89+ and large matrices, bf16 fallback otherwise), so fp8_ns=True is a
        # portable config that simply no-ops on unsupported GPUs/shapes rather than
        # erroring. Warn once here if it can never engage on this machine.
        if fp8_ns and not _fp8_supported():
            warnings.warn(
                "fp8_ns=True but this GPU is pre-sm_89 (no fp8 GEMM); "
                "Newton-Schulz will run in bf16.",
                RuntimeWarning,
            )
        self._fp8_ns = fp8_ns
        self._fp8_ns_compile = fp8_ns_compile

        if not isinstance(batched_ns_workspace_bytes, int) or isinstance(
            batched_ns_workspace_bytes, bool
        ):
            raise TypeError("batched_ns_workspace_bytes must be an integer")
        if batched_ns_workspace_bytes <= 0:
            raise ValueError("batched_ns_workspace_bytes must be positive")
        self._batched_ns = bool(batched_ns)
        self._batched_ns_workspace_bytes = batched_ns_workspace_bytes

        # NorMuon-style per-neuron 2nd moment on the orthogonalized update
        # (opt-in). After Newton-Schulz, each ROW of the update (an output
        # neuron) is divided by the bias-corrected EMA-RMS of that row across
        # steps, then the whole matrix is rescaled to its pre-normalization
        # Frobenius norm so the global update scale (and therefore the
        # match_rms_adamw LR semantics) is unchanged. State is one fp32 scalar
        # per row (~4 bytes/row -- negligible next to the 8-bit momentum).
        if not 0.0 <= normuon_beta2 < 1.0:
            raise ValueError(
                "normuon_beta2 must be in [0, 1) but is: {}".format(normuon_beta2)
            )
        if not math.isfinite(normuon_eps) or normuon_eps <= 0.0:
            raise ValueError(
                "normuon_eps must be finite and > 0 but is: {}".format(normuon_eps)
            )
        # Cautious masking (opt-in): zero update coordinates whose sign
        # disagrees with the current gradient, rescaled to preserve magnitude.
        self._normuon = normuon
        self._cautious = cautious

        # torch.optim.Optimizer calls ``self.add_param_group`` while the base
        # constructor is still running. Publish the complete Muon group schema
        # first so both constructor groups and groups added later take the same
        # validation/default-injection path.
        self._muon_group_defaults = {
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
            "sharded_mode": sharded_mode,
            "fp8_ns": fp8_ns,
            "fp8_ns_compile": fp8_ns_compile,
            "batched_ns": bool(batched_ns),
            "batched_ns_workspace_bytes": batched_ns_workspace_bytes,
            "normuon": normuon,
            "normuon_beta2": normuon_beta2,
            "normuon_eps": normuon_eps,
            "cautious": cautious,
        }

        super().__init__(
            params,
            lr=lr,
            betas=(momentum, 0.0),
            eps=eps,
            weight_decay=weight_decay,
            fused=fused,
            stochastic_round=stochastic_round,
            capturable=capturable,
            verbose=verbose,
        )

    def add_param_group(self, param_group):
        """Add a validated 2D Muon parameter group atomically.

        Muon-specific options may be supplied per group; omitted values inherit
        the constructor defaults. The group is fully validated before the base
        Gefen registration mutates ``param_groups`` or per-parameter state.
        """
        if not isinstance(param_group, dict) or "params" not in param_group:
            # Preserve Gefen's public error types/messages for malformed group
            # containers and missing ``params``.
            return super().add_param_group(param_group)

        group = dict(param_group)
        raw_params = group["params"]
        if isinstance(raw_params, torch.Tensor):
            raw_params = [raw_params]
        else:
            raw_params = list(raw_params)
        group["params"] = raw_params

        # Validate tensor type/complex support and Muon's dimensionality before
        # registration. Gefen.add_param_group repeats its own checks, but doing
        # this first keeps a mixed valid/invalid addition all-or-nothing.
        for _, param in self._iter_params_with_names(raw_params):
            if param.ndim != 2:
                raise ValueError(
                    "GefenMuon only supports 2D parameters whereas we found a "
                    "parameter with size: {}".format(param.size())
                )

        defaults = self._muon_group_defaults
        momentum = group.get("momentum", defaults["momentum"])
        if not 0.0 <= momentum < 1.0:
            raise ValueError(
                "momentum should be in [0, 1) but is: {}".format(momentum)
            )

        adjust_lr_fn = group.get("adjust_lr_fn", defaults["adjust_lr_fn"])
        if adjust_lr_fn is not None and adjust_lr_fn not in (
            "original",
            "match_rms_adamw",
        ):
            raise ValueError(
                "Adjust learning rate function {} is not supported".format(
                    adjust_lr_fn
                )
            )

        sharded_mode = group.get("sharded_mode", defaults["sharded_mode"])
        if sharded_mode not in ("exact", "approx", "distributed"):
            raise ValueError(
                "sharded_mode must be 'exact', 'approx' or 'distributed' but is: "
                "{}".format(sharded_mode)
            )

        ns_coefficients = group.get(
            "ns_coefficients", defaults["ns_coefficients"]
        )
        ns_steps = group.get("ns_steps", defaults["ns_steps"])
        ns_schedule = group.get("ns_schedule")
        if ns_schedule is not None:
            if isinstance(ns_schedule, str):
                if ns_schedule not in NS_SCHEDULES:
                    raise ValueError(
                        "Unknown ns_schedule {!r}; choose from {}".format(
                            ns_schedule, sorted(NS_SCHEDULES)
                        )
                    )
                resolved_schedule = NS_SCHEDULES[ns_schedule]
            else:
                resolved_schedule = ns_schedule
            if resolved_schedule is not None:
                ns_coefficients = _normalize_ns_schedule(
                    resolved_schedule, ns_steps
                )
                ns_steps = len(ns_coefficients)
        # Validate the fixed-coefficient path and direct explicit schedules too.
        _normalize_ns_schedule(ns_coefficients, ns_steps)

        workspace_bytes = group.get(
            "batched_ns_workspace_bytes",
            defaults["batched_ns_workspace_bytes"],
        )
        if not isinstance(workspace_bytes, int) or isinstance(workspace_bytes, bool):
            raise TypeError("batched_ns_workspace_bytes must be an integer")
        if workspace_bytes <= 0:
            raise ValueError("batched_ns_workspace_bytes must be positive")

        normuon_beta2 = group.get("normuon_beta2", defaults["normuon_beta2"])
        if not 0.0 <= normuon_beta2 < 1.0:
            raise ValueError(
                "normuon_beta2 must be in [0, 1) but is: {}".format(normuon_beta2)
            )
        normuon_eps = group.get("normuon_eps", defaults["normuon_eps"])
        if not math.isfinite(normuon_eps) or normuon_eps <= 0.0:
            raise ValueError(
                "normuon_eps must be finite and > 0 but is: {}".format(normuon_eps)
            )

        group.update(
            {
                "momentum": momentum,
                "nesterov": group.get("nesterov", defaults["nesterov"]),
                "ns_coefficients": ns_coefficients,
                "ns_steps": ns_steps,
                "adjust_lr_fn": adjust_lr_fn,
                "sharded_mode": sharded_mode,
                "fp8_ns": group.get("fp8_ns", defaults["fp8_ns"]),
                "fp8_ns_compile": group.get(
                    "fp8_ns_compile", defaults["fp8_ns_compile"]
                ),
                "batched_ns": bool(
                    group.get("batched_ns", defaults["batched_ns"])
                ),
                "batched_ns_workspace_bytes": workspace_bytes,
                "normuon": group.get("normuon", defaults["normuon"]),
                "normuon_beta2": normuon_beta2,
                "normuon_eps": normuon_eps,
                "cautious": group.get("cautious", defaults["cautious"]),
                # Gefen's inherited storage still carries beta1/beta2. Keep it
                # consistent with the Muon momentum option instead of retaining
                # irrelevant or contradictory caller-supplied Adam betas.
                "betas": (momentum, 0.0),
            }
        )
        return super().add_param_group(group)

    def _init_gefen_muon_state(self, state, grad_view: torch.Tensor) -> None:
        self._init_gefen_state(state, grad_view)

    def _iter_gefen_grad_periods(self, reuse_existing_periods: bool = False):
        # Same as Gefen._iter_gefen_grad_periods, but for sharded (DTensor)
        # gradients gather the FULL matrix (full_tensor) instead of taking the
        # local shard. The exact-DP codebook and the per-param block period are
        # therefore learned from the full matrix on every rank -- identical to
        # the single-GPU reference, and identical across ranks (so quantization
        # matches). With full grads, flat.numel() is the global numel and is
        # never 0, so every rank iterates every parameter in the same order and
        # the full_tensor() collective is matched across ranks.
        for group in self.param_groups:
            for param_name, p in self._iter_group_params_with_names(group):
                if p.grad is None:
                    continue
                grad = p.grad
                # approx mode learns the codebook/period from the LOCAL shard
                # (no all-gather) so periods divide the local numel that the
                # approximate step operates on; exact mode gathers the full matrix.
                if group["sharded_mode"] == "approx" and hasattr(
                    grad, "to_local"
                ):
                    grad = grad.to_local()
                elif hasattr(grad, "full_tensor"):
                    grad = grad.full_tensor()
                elif hasattr(grad, "to_local"):
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
                    period = self._predict_period_from_grad_sq(param_name, p, grad)

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

    def _quantize_momentum_(self, state, momentum_view: torch.Tensor) -> None:
        period = state["automatic_period"]
        state["m_magnitude"].copy_(
            self._automatic_reduce(
                momentum_view.abs().reshape(-1), period, reduce_op="max"
            )
        )

        momentum_view.div_(
            state["m_magnitude"].clamp(min=torch.finfo(momentum_view.dtype).tiny)
        )
        momentum_view.masked_fill_(state["m_magnitude"] <= 0, 0.0)

        indices = self._gefen_nearest_indices(momentum_view)
        self._gefen_set_indices(state, indices)

        # Per-device resolver: with device_map-split models the momentum views
        # live on several GPUs while the codebook is learned once on one; the
        # resolver hands back a cached same-device copy (a no-op single-device).
        codebook = self._gefen_codebook_on(indices.device)
        if codebook is None:
            raise ValueError(
                "Expected Gefen codebook to be initialized before reconstructing quantized momentum."
            )

        momentum_view.copy_(codebook[indices.long()].to(dtype=momentum_view.dtype))
        momentum_view.mul_(state["m_magnitude"])

    def _fused_quantized_momentum_update(
        self,
        state,
        grad_view: torch.Tensor,
        momentum: float,
        *,
        nesterov: bool = False,
    ) -> torch.Tensor:
        # Single-pass Muon momentum update: the kernel advances the quantized
        # momentum state and emits the dense quantized momentum for Newton-Schulz
        # directly, so the old lr==0 dummy-stepsize call into the generic update
        # kernel followed by a second full-size codebook gather is gone. The
        # emitted base momentum is bit-identical to the old
        # `dequantize(m_codebook) * m_magnitude`; when requested, the kernel
        # additionally reproduces the old host Nesterov mul/add bit-for-bit in
        # this same output write without changing the stored EMA state.
        return self._gefen_quantized_momentum_update(
            state, grad_view, momentum, nesterov=nesterov
        )

    @staticmethod
    def _is_sharded(p: torch.Tensor) -> bool:
        # An FSDP2 / DTensor parameter carries a device mesh + placements. Plain
        # tensors (single-GPU / DDP) do not, and take the original code paths
        # verbatim so their behaviour is byte-for-byte unchanged.
        return (
            hasattr(p, "to_local")
            and hasattr(p, "placements")
            and hasattr(p, "device_mesh")
        )

    @staticmethod
    def _shard_like(
        full_tensor: torch.Tensor, dtensor_param: torch.Tensor
    ) -> torch.Tensor:
        # Slice an already-replicated full tensor (the Newton-Schulz update,
        # which is identical on every rank) down to this rank's shard, matching
        # the parameter's own placements exactly -- including uneven and empty
        # shards. Replicate -> Shard is a pure local narrow (no collective), so
        # rather than build a Replicated DTensor and call
        # .redistribute(...).to_local() (a DTensor object + redistribute dispatch
        # per matrix, every step) we narrow the full tensor directly, reproducing
        # DTensor's torch.chunk split arithmetic exactly. This is numerically
        # identical to the redistribute path, adds no communication, and cannot
        # deadlock.
        from torch.distributed.tensor.placement_types import Shard

        # Fast narrow only reproduces DTensor's chunking for *plain* Shard (and
        # Replicate, a no-op slice). Strided/other shard variants -- e.g.
        # `_StridedShard`, used for FSDP2 x TP (HSDP) composition, which is NOT a
        # `Shard` subclass -- have different split arithmetic, so fall back to the
        # exact (slower) DTensor redistribute for any non-plain-Shard placement
        # rather than silently slicing the wrong rows.
        placements = dtensor_param.placements
        if any(p.is_shard() and type(p) is not Shard for p in placements):
            from torch.distributed.tensor import DTensor, Replicate

            mesh = dtensor_param.device_mesh
            replicated = DTensor.from_local(
                full_tensor.contiguous(),
                mesh,
                [Replicate()] * mesh.ndim,
                run_check=False,
            )
            return replicated.redistribute(placements=placements).to_local()

        mesh = dtensor_param.device_mesh
        coords = mesh.get_coordinate()
        if coords is None:
            # This rank is not in the mesh -> it owns no shard. (The caller
            # already guards on p_local.numel()==0 before reaching here, so this
            # is a defensive empty return; dim 0 is fine because it is empty.)
            return full_tensor.narrow(0, 0, 0).contiguous()

        local = full_tensor
        for mesh_dim, placement in enumerate(placements):
            if isinstance(placement, Shard):
                dim = placement.dim
                size = local.size(dim)
                num_chunks = mesh.size(mesh_dim)
                rank = coords[mesh_dim]
                # ceil split == torch.chunk == DTensor Replicate->Shard
                full_chunk = (size + num_chunks - 1) // num_chunks
                start = full_chunk * rank
                length = max(0, min(size, start + full_chunk) - start)
                local = local.narrow(dim, min(start, size), length)
        return local.contiguous()

    def _prepare_muon_momentum(
        self,
        group,
        param_name: str,
        p: torch.Tensor,
        grad: torch.Tensor,
        eff_numel: int,
    ) -> "Tuple[dict, torch.Tensor]":
        """Advance quantized momentum and return the dense NS input."""
        state = self.state[p]
        momentum = group["momentum"]
        flat_grad = grad.reshape(-1)
        if "step" not in state:
            if "automatic_period" in state:
                automatic_period = state["automatic_period"]
            elif eff_numel == 1:
                automatic_period = 1
            elif eff_numel > 1:
                automatic_period = self._predict_period_from_grad_sq(
                    param_name, p, grad
                )
            else:
                raise ValueError(
                    "Automatic partition received an empty parameter {}".format(
                        param_name
                    )
                )

            if eff_numel % automatic_period != 0:
                raise ValueError(
                    "Automatic partition period {} does not divide parameter {} with numel {}".format(
                        automatic_period,
                        param_name,
                        eff_numel,
                    )
                )

            state["automatic_period"] = automatic_period
            state["step"] = self._new_step_counter(grad.device)
            self._init_gefen_muon_state(
                state, self._automatic_view(flat_grad, automatic_period)
            )

        automatic_period = state["automatic_period"]
        grad_view = self._automatic_view(flat_grad, automatic_period)

        nesterov_emitted = False
        if (
            self._fused_kernels_available()
            and grad_view.is_cuda
            and grad_view.dtype != torch.float64
        ):
            # Per-tensor device/dtype gating (see Gefen._step_automatic): the
            # momentum kernel is a raw CUDA-pointer launch with a reject-Double
            # dispatch, so a CPU-resident shard or an fp64 param takes the
            # pure-torch dequant/lerp/quantize path below instead.
            # The Muon momentum kernel reads grad + quantized state and emits the
            # dense quantized momentum; it never touches p, so no full-matrix
            # scratch is needed under sharding. Unlike the Gefen update kernels,
            # its only host scalar argument is the CONSTANT momentum beta, so it
            # stays enabled under capturable -- constants bake into a CUDA graph
            # safely, and the per-step stochastic-round seed (when
            # stochastic_round=True) flows through the device seed tensor
            # (Gefen._sr_seed_on) rather than a host argument a graph would
            # freeze.
            momentum_update = self._fused_quantized_momentum_update(
                state,
                grad_view,
                momentum,
                nesterov=group["nesterov"],
            )
            nesterov_emitted = group["nesterov"]
        else:
            momentum_update = self._gefen_dequantize_m_coefficients(state, grad_view)
            momentum_update.mul_(state["m_magnitude"])
            momentum_update.lerp_(grad_view, 1 - momentum)
            self._quantize_momentum_(state, momentum_update)

        state["step"] += 1

        if group["nesterov"] and not nesterov_emitted:
            momentum_update.mul_(momentum).add_(grad_view, alpha=1 - momentum)
        return state, momentum_update.view_as(grad)

    def _finish_muon_update(
        self,
        group,
        state,
        grad: torch.Tensor,
        ortho: torch.Tensor,
    ) -> torch.Tensor:
        """Apply optional post-NS levers shared by serial and batched NS."""
        if group.get("normuon", False):
            ortho = self._normuon_normalize(
                state, ortho, group["normuon_beta2"], group["normuon_eps"]
            )
        if group.get("cautious", False):
            ortho = _cautious_mask_(ortho.contiguous(), grad)
        return ortho

    def _compute_muon_update(
        self,
        group,
        param_name: str,
        p: torch.Tensor,
        grad: torch.Tensor,
        eff_numel: int,
    ) -> torch.Tensor:
        # The full quantized-momentum + Newton-Schulz pipeline that turns a
        # gradient matrix into the orthogonalized update. Shared verbatim by the
        # exact path (grad = full matrix, eff_numel = global numel), the approx
        # path (grad = local shard, eff_numel = local numel) and the distributed
        # owner path (grad = full matrix, eff_numel = global numel) so all three
        # are bit-identical on identical inputs. Reads/writes self.state[p].
        state, update = self._prepare_muon_momentum(
            group, param_name, p, grad, eff_numel
        )

        ortho = _zeropower_via_newtonschulz(
            update,
            group["ns_coefficients"],
            group["ns_steps"],
            group["eps"],
            use_fp8=group.get("fp8_ns", False),
            compile_fp8=group.get("fp8_ns_compile", True),
        )
        return self._finish_muon_update(group, state, grad, ortho)

    def _normuon_normalize(
        self,
        state,
        ortho: torch.Tensor,
        beta2: float,
        eps: float,
    ) -> torch.Tensor:
        # Per-neuron 2nd-moment normalization of the Newton-Schulz output
        # (NorMuon, Zhang et al. 2025). NS truncated at a few iterations leaves
        # the row norms of the "orthogonalized" update visibly non-uniform; an
        # EMA of each row's mean-square tracks that and divides it out, giving
        # each output neuron a uniform effective step. The final rescale to the
        # pre-normalization Frobenius norm keeps the overall update scale --
        # and thus the adjust_lr_fn / match_rms_adamw calibration -- intact.
        # Two passes over the matrix total: one fp32-accumulated row-norm
        # reduction, one in-place scale. Everything else is O(rows). The
        # Frobenius norms before/after row normalization are both derivable
        # from the row norms alone (||O||^2 = sum_r ||o_r||^2 and
        # ||O/denom||^2 = sum_r ||o_r||^2 / denom_r^2), so no full-size fp32
        # copy of the update is ever materialized.
        v = state.get("normuon_v")
        if v is None or v.shape[0] != ortho.shape[0]:
            v = torch.zeros(
                ortho.shape[0], 1, dtype=torch.float32, device=ortho.device
            )
            state["normuon_v"] = v
            state["normuon_step"] = self._new_step_counter(ortho.device)
        row_norm = torch.linalg.vector_norm(
            ortho, dim=1, keepdim=True, dtype=torch.float32
        )
        row_sq = row_norm.square()
        v.mul_(beta2).add_(row_sq, alpha=(1 - beta2) / ortho.shape[1])
        state["normuon_step"] += 1
        bias_correction = 1 - beta2 ** state["normuon_step"]
        denom = (v / bias_correction).sqrt_().add_(eps)
        frob_scale = row_sq.sum().sqrt_() / torch.linalg.vector_norm(
            row_norm / denom
        ).clamp(min=torch.finfo(torch.float32).tiny)
        return ortho.mul_((frob_scale / denom).to(ortho.dtype))

    # _lr_scalar is inherited unchanged from Gefen (single-slot tensor-lr cache).

    def _apply_muon_update(
        self,
        group,
        p: torch.Tensor,
        update: torch.Tensor,
        is_sharded: bool,
        approx: bool,
    ) -> None:
        lr = group["lr"]
        if self.capturable:
            # Graph-capturable apply: never .item() the lr (a D2H sync, and a
            # captured graph would freeze the read value anyway). The
            # adjust_lr ratio is a static function of the fixed param shape,
            # so it stays a python constant; a TENSOR lr flows through the
            # update math on device -- update it in place between replays to
            # drive an LR schedule. A float lr bakes into the graph. alpha=
            # only accepts host scalars, so the update is scaled by a tensor
            # multiply instead.
            adjusted_lr = _adjust_lr_ratio(group["adjust_lr_fn"], p.shape) * lr
            if is_sharded:
                p_local = p.to_local()
                if group["weight_decay"] > 0.0:
                    p_local.mul_(1 - lr * group["weight_decay"])
                if p_local.numel() > 0:
                    local_update = update if approx else self._shard_like(update, p)
                    # Scale in p's dtype (fp32), matching the promotion the
                    # non-capturable p.add_(update, alpha=...) form performs;
                    # a bare `update * adjusted_lr` would multiply in the NS
                    # output's bf16 and visibly lose precision.
                    p_local.sub_(local_update.to(dtype=p_local.dtype) * adjusted_lr)
            else:
                if group["weight_decay"] > 0.0:
                    p.mul_(1 - lr * group["weight_decay"])
                p.sub_(update.to(dtype=p.dtype) * adjusted_lr)
            return

        # p.shape is the GLOBAL shape for a DTensor, which is exactly what the
        # rows/cols LR ratio wants -- keep it unchanged under sharding.
        # Resolve a tensor LR (tensor-LR / capturable scheduler) to a python float
        # via a cached scalar instead of a fresh lr.item() per Muon param. A tensor
        # lr.item() is a D2H sync; the per-param _adjust_lr call would otherwise
        # mean ~one sync per param per step for tensor-lr schedules,
        # serializing the Newton-Schulz pipeline. The cache re-reads only when the
        # lr tensor's identity or _version changes (an in-place scheduler update
        # bumps _version), so a constant tensor lr is read once and reused; the
        # value is exactly lr.item(), making _adjust_lr bit-identical.
        adjusted_lr = _adjust_lr(
            self._lr_scalar(group), group["adjust_lr_fn"], p.shape
        )

        if is_sharded:
            # Apply to the local shard storage (to_local() is a view, so in-place
            # ops propagate back). In exact/distributed mode `update` is the
            # full-matrix Newton-Schulz result (identical on every rank -- computed
            # redundantly under exact, broadcast from the owner under distributed)
            # sliced to this shard; in approx mode `update` is already this shard's
            # own NS result.
            p_local = p.to_local()
            if group["weight_decay"] > 0.0:
                p_local.mul_(1 - lr * group["weight_decay"])
            if p_local.numel() > 0:
                local_update = update if approx else self._shard_like(update, p)
                p_local.add_(local_update, alpha=-adjusted_lr)
        else:
            if group["weight_decay"] > 0.0:
                p.mul_(1 - lr * group["weight_decay"])
            p.add_(update, alpha=-adjusted_lr)

    @staticmethod
    def _dist_available() -> bool:
        if not torch.distributed.is_available():
            return False
        return torch.distributed.is_initialized()

    def _distributed_process_group(self, p: torch.Tensor):
        if not self._dist_available() or not self._is_sharded(p):
            return None
        import torch.distributed as dist

        mesh = p.device_mesh
        if mesh.ndim != 1:
            return None
        pg = mesh.get_group()
        if dist.get_world_size(pg) < 2:
            return None
        return pg

    def _step_distributed(self, items) -> None:
        # "distributed" / Parallel-Muon path. Each 2D matrix is assigned to one
        # stable owner rank that alone runs the quantized-momentum + Newton-Schulz;
        # the result is broadcast so every rank slices its shard.
        #
        # CRITICAL for the speed-up: the gather, compute and broadcast are
        # SEPARATED into phases per bucket of `world` matrices. If we broadcast
        # each update right after its owner computes it, the non-owner ranks block
        # at the broadcast while the owner runs NS -- the NS serializes and there
        # is no win. By all-gathering every grad in the bucket first, then letting
        # each rank compute its owned matrix/matrices with no collective in
        # between, the owners' NS runs concurrently (NS critical path ~ NS_total /
        # world), and the per-bucket broadcasts are a pure communication phase.
        # Eligibility (1-D mesh, world>=2) is a property of each param's mesh and
        # so is identical on every rank -> the eligible/fallback split, and thus
        # the collective order, agrees globally. Non-eligible matrices (multi-dim
        # HSDP x TP meshes, world==1) keep the replicated exact full-NS path.
        eligible, fallback = [], []
        for (group, name, p, grad) in items:
            pg = self._distributed_process_group(p)
            if pg is not None:
                eligible.append((pg, group, name, p, grad))
            else:
                fallback.append((group, name, p, grad))

        for (group, name, p, grad) in fallback:
            if grad is not None:
                self._step_automatic(group, name, p, grad)

        if not eligible:
            return

        # Group by process group (one mesh under plain FSDP2; multiple only under
        # exotic setups). Insertion-ordered so the order is identical across ranks.
        by_pg = OrderedDict()
        for pg, group, name, p, grad in eligible:
            by_pg.setdefault(pg, []).append((group, name, p, grad))

        for pg, pg_items in by_pg.items():
            self._step_distributed_pg(pg, pg_items)

    def _step_distributed_pg(self, pg, pg_items) -> None:
        import torch.distributed as dist

        world = dist.get_world_size(pg)
        my_coord = dist.get_group_rank(pg, dist.get_rank())
        global_rank = [dist.get_global_rank(pg, c) for c in range(world)]

        active_items = [
            (_stable_distributed_owner(idx, world), group, name, p, grad)
            for idx, (group, name, p, grad) in enumerate(pg_items)
            if grad is not None
        ]
        if not active_items:
            return

        # Buckets of `world` active matrices bound peak full-gradient scratch.
        # Ownership is NOT active-position based: every matrix maps from its
        # stable position in the full distributed param set, so dropping/adding
        # unrelated gradients between steps does not move a matrix's momentum to a
        # different rank. When all grads are present this preserves the old
        # balanced model-order schedule.
        for b in range(0, len(active_items), world):
            bucket = active_items[b : b + world]

            # --- Phase 1: all-gather every grad in the bucket; keep only mine. ---
            # Every rank joins every full_tensor (matched collective). No compute
            # is interleaved, so ranks march through the gathers in lockstep.
            owned = []
            for bucket_idx, (owner, group, name, p, grad) in enumerate(bucket):
                fg = grad
                if hasattr(fg, "full_tensor"):
                    fg = fg.full_tensor()
                elif hasattr(fg, "to_local"):
                    fg = fg.to_local()
                if hasattr(fg, "wait"):
                    fg = fg.wait()
                if owner == my_coord:
                    if torch.is_complex(p):
                        raise RuntimeError(
                            "GefenMuon does not support complex parameters"
                        )
                    if fg.is_sparse:
                        raise RuntimeError(
                            "GefenMuon does not support sparse gradients"
                        )
                    if fg.ndim != 2:
                        raise ValueError(
                            "GefenMuon gradient must be a 2D matrix for {}".format(
                                name
                            )
                        )
                    owned.append((bucket_idx, group, name, p, fg))
                else:
                    fg = None  # drop the gathered full grad we do not own

            # --- Phase 2: compute MY owned update (no collectives -> parallel). ---
            owned_updates = [None] * len(bucket)
            for bucket_idx, group, name, p, full_grad in owned:
                owned_updates[bucket_idx] = self._compute_muon_update(
                    group, name, p, full_grad, p.numel()
                )
            owned = None

            # --- Phase 3: broadcast each update from its owner; apply locally. ---
            for bucket_idx, (owner, group, name, p, grad) in enumerate(bucket):
                if owner == my_coord:
                    buf = owned_updates[bucket_idx].contiguous()
                    if buf.dtype != torch.bfloat16:
                        buf = buf.to(torch.bfloat16)
                else:
                    buf = torch.empty(
                        tuple(p.shape),
                        dtype=torch.bfloat16,
                        device=p.to_local().device,
                    )
                dist.broadcast(buf, src=global_rank[owner], group=pg)
                self._apply_muon_update(
                    group, p, buf, is_sharded=True, approx=False
                )
                buf = None

    def _batched_ns_key(self, item, fused_available: bool):
        """Return a homogeneous batch key, or ``None`` for serial routing."""
        group, _name, p, grad = item
        if (
            not group.get("batched_ns", False)
            or not fused_available
            or self.capturable
            or torch.compiler.is_compiling()
            or self._is_sharded(p)
            or group.get("fp8_ns", False)
            or torch.is_complex(p)
            or grad.is_sparse
            or grad.ndim != 2
            or not p.is_cuda
            or not grad.is_cuda
            or p.dtype != torch.bfloat16
            or grad.dtype != torch.bfloat16
            or tuple(p.shape) != tuple(grad.shape)
        ):
            return None
        rows, cols = grad.shape
        if not _batched_ns_shape_eligible(rows, cols):
            return None
        schedule = tuple(
            _normalize_ns_schedule(group["ns_coefficients"], group["ns_steps"])
        )
        return (
            grad.device,
            grad.dtype,
            (rows, cols),
            schedule,
            float(group["eps"]),
            int(
                group.get(
                    "batched_ns_workspace_bytes",
                    BATCHED_NS_DEFAULT_WORKSPACE_BYTES,
                )
            ),
        )

    def _step_batched_ns_chunk(self, items) -> None:
        states = []
        momentum_updates = []
        for group, name, p, grad in items:
            state, update = self._prepare_muon_momentum(
                group, name, p, grad, p.numel()
            )
            states.append(state)
            momentum_updates.append(update)

        first_group = items[0][0]
        transposed = momentum_updates[0].size(0) > momentum_updates[0].size(1)
        # Orient before stacking so tall matrices do not require a second full
        # contiguous batch inside the NS helper (and therefore stay inside the
        # same 3S+2G workspace model as wide matrices).
        stack_inputs = (
            [update.T for update in momentum_updates]
            if transposed
            else momentum_updates
        )
        stacked = torch.stack(stack_inputs, dim=0)
        # The stacked tensor owns the NS inputs now. Drop the individual dense
        # momentum buffers before allocating Gram matrices so the allocator can
        # reuse them within the configured workspace envelope.
        del stack_inputs, momentum_updates
        ortho_batch = _zeropower_via_newtonschulz_batched(
            stacked,
            first_group["ns_coefficients"],
            first_group["ns_steps"],
            first_group["eps"],
        )
        if transposed:
            ortho_batch = ortho_batch.transpose(1, 2)
        for i, ((group, _name, p, grad), state) in enumerate(zip(items, states)):
            update = self._finish_muon_update(
                group, state, grad, ortho_batch[i]
            )
            self._apply_muon_update(
                group, p, update, is_sharded=False, approx=False
            )

    def _step_regular_items(self, items) -> None:
        if not items:
            return
        wants_batched = any(group.get("batched_ns", False) for group, *_ in items)
        if not wants_batched or self.capturable or torch.compiler.is_compiling():
            for group, name, p, grad in items:
                self._step_automatic(group, name, p, grad)
            return

        fused_available = self._fused_kernels_available()
        buckets = OrderedDict()
        for position, item in enumerate(items):
            key = self._batched_ns_key(item, fused_available)
            # Give every serial item a unique key so its relative position among
            # first-seen shape buckets remains deterministic.
            if key is None:
                key = ("serial", position)
            buckets.setdefault(key, []).append(item)

        for key, bucket in buckets.items():
            if key[0] == "serial":
                group, name, p, grad = bucket[0]
                self._step_automatic(group, name, p, grad)
                continue

            rows, cols = key[2]
            workspace_bytes = key[5]
            sizes, serial_tail = _batched_ns_chunk_sizes(
                len(bucket), rows, cols, workspace_bytes
            )
            offset = 0
            for size in sizes:
                self._step_batched_ns_chunk(bucket[offset : offset + size])
                offset += size
            # A group below the minimum, a workspace cap that cannot fit eight,
            # or an unavoidable short remainder stays bit-identical and serial.
            for group, name, p, grad in bucket[offset : offset + serial_tail]:
                self._step_automatic(group, name, p, grad)

    def _step_automatic(
        self, group, param_name: str, p: torch.Tensor, grad: torch.Tensor
    ) -> None:
        is_sharded = self._is_sharded(p)
        approx = is_sharded and group["sharded_mode"] == "approx"

        if approx:
            # NON-PARITY opt-in mode: operate on this rank's LOCAL shard only --
            # no all-gather, and Newton-Schulz runs on the smaller row-sharded
            # matrix. There are no collectives here, so an empty-shard rank may
            # return early without risking a deadlock.
            if hasattr(grad, "to_local"):
                grad = grad.to_local()
            if hasattr(grad, "wait"):
                grad = grad.wait()
            if grad.numel() == 0:
                return
        elif is_sharded:
            # FSDP2: reconstruct the FULL gradient matrix on every rank so the
            # period prediction, quantized momentum, and Newton-Schulz all run on
            # the full matrix exactly as on a single GPU. full_tensor() is a
            # collective (all-gather); every rank -- including a rank that owns an
            # empty shard -- must reach it, so it precedes any early return.
            if hasattr(grad, "full_tensor"):
                grad = grad.full_tensor()
            elif hasattr(grad, "to_local"):
                grad = grad.to_local()
            if hasattr(grad, "wait"):
                grad = grad.wait()
        elif not self.fused:
            if hasattr(grad, "to_local"):
                grad = grad.to_local()
            if hasattr(grad, "wait"):
                grad = grad.wait()

        if torch.is_complex(p):
            raise RuntimeError("GefenMuon does not support complex parameters")
        if grad.is_sparse:
            raise RuntimeError("GefenMuon does not support sparse gradients")
        if grad.ndim != 2:
            raise ValueError(
                "GefenMuon gradient must be a 2D matrix for {} but has {} "
                "dimension(s). If a training wrapper flattened the parameters "
                "or gradients (DeepSpeed ZeRO does this: it steps flattened "
                "1-D fp32 partitions), note that Muon's 2D orthogonalization "
                "cannot apply there -- use plain Gefen as the DeepSpeed ZeRO "
                "client optimizer instead.".format(param_name, grad.ndim)
            )

        # In approx mode the pipeline operates on the local shard, so the period
        # must divide the LOCAL numel; exact/non-sharded use the (global) numel.
        eff_numel = grad.reshape(-1).numel() if approx else p.numel()
        update = self._compute_muon_update(group, param_name, p, grad, eff_numel)
        self._apply_muon_update(group, p, update, is_sharded, approx)

    @staticmethod
    def _state_tensor_device(p: torch.Tensor) -> torch.device:
        if hasattr(p, "to_local"):
            local = p.to_local()
            if hasattr(local, "wait"):
                local = local.wait()
            return local.device
        return p.device

    def _distributed_state_items(self, state_dict):
        saved_ids = []
        for saved_group in state_dict["param_groups"]:
            saved_ids.extend(saved_group["params"])

        live_params = []
        for group in self.param_groups:
            live_params.extend(group["params"])
        param_to_saved_id = dict(zip(live_params, saved_ids))

        by_pg = OrderedDict()
        for group in self.param_groups:
            if group["sharded_mode"] != "distributed":
                continue
            for name, p in self._iter_group_params_with_names(group):
                pg = self._distributed_process_group(p)
                if pg is None:
                    continue
                saved_id = param_to_saved_id.get(p)
                if saved_id is None:
                    continue
                by_pg.setdefault(pg, []).append((name, p, saved_id))
        return by_pg

    def _consolidate_distributed_state_dict(self, state_dict):
        # In distributed mode the persistent momentum state is updated only on the
        # stable owner rank for each matrix. Before checkpointing, broadcast each
        # owner's serialized per-param state to every rank so rank-0-only writers
        # can save a complete optimizer state_dict AFTER all ranks have called
        # this method. This method is therefore collective for distributed
        # sharded params, matching the DTensor collectives used by step().
        if not self._dist_available():
            return state_dict

        import torch.distributed as dist

        by_pg = self._distributed_state_items(state_dict)
        if not by_pg:
            return state_dict

        for pg, pg_items in by_pg.items():
            world = dist.get_world_size(pg)
            my_coord = dist.get_group_rank(pg, dist.get_rank())
            global_rank = [dist.get_global_rank(pg, c) for c in range(world)]
            owner_by_saved_id = {
                saved_id: _stable_distributed_owner(idx, world)
                for idx, (_, _, saved_id) in enumerate(pg_items)
            }

            for name, p, saved_id in pg_items:
                owner = owner_by_saved_id[saved_id]
                src = global_rank[owner]
                tensor_values = []
                if my_coord == owner:
                    pstate = state_dict["state"].get(saved_id, {})
                    meta = []
                    for key, value in pstate.items():
                        if torch.is_tensor(value):
                            tensor = value.detach()
                            if not tensor.is_contiguous():
                                tensor = tensor.contiguous()
                            meta.append(
                                (
                                    key,
                                    "tensor",
                                    tuple(tensor.shape),
                                    tensor.dtype,
                                )
                            )
                            tensor_values.append(tensor)
                        else:
                            meta.append((key, "object", value))
                else:
                    meta = None

                obj = [meta]
                dist.broadcast_object_list(obj, src=src, group=pg)
                meta = obj[0]

                owner_state = {}
                tensor_idx = 0
                for entry in meta:
                    key = entry[0]
                    kind = entry[1]
                    if kind == "object":
                        owner_state[key] = entry[2]
                        continue

                    _, _, shape, dtype = entry
                    if my_coord == owner:
                        tensor = tensor_values[tensor_idx]
                    else:
                        tensor = torch.empty(
                            shape,
                            dtype=dtype,
                            device=self._state_tensor_device(p),
                        )
                    dist.broadcast(tensor, src=src, group=pg)
                    owner_state[key] = tensor
                    tensor_idx += 1

                state_dict["state"][saved_id] = owner_state

        state_dict["gefen_muon_distributed"] = {
            "version": 1,
            "ownership": "stable_full_param_index_v1",
            "consolidated": True,
        }
        return state_dict

    def _drop_non_owned_distributed_state(self) -> None:
        if not self._dist_available():
            return

        import torch.distributed as dist

        by_pg = OrderedDict()
        for group in self.param_groups:
            if group["sharded_mode"] != "distributed":
                continue
            for name, p in self._iter_group_params_with_names(group):
                pg = self._distributed_process_group(p)
                if pg is not None:
                    by_pg.setdefault(pg, []).append((name, p))

        for pg, pg_items in by_pg.items():
            for idx, (name, p) in enumerate(pg_items):
                world = dist.get_world_size(pg)
                my_coord = dist.get_group_rank(pg, dist.get_rank())
                owner = _stable_distributed_owner(idx, world)
                if owner == my_coord:
                    continue
                pstate = self.state[p]
                pstate.clear()
                pstate["name"] = str(name).lower()

    def state_dict(self):
        state_dict = super().state_dict()
        return self._consolidate_distributed_state_dict(state_dict)

    def _warn_if_unconsolidated_distributed_load(self, state_dict, marker) -> None:
        # In distributed mode the persistent momentum lives only on each matrix's
        # stable owner rank; state_dict() runs a collective that broadcasts every
        # owner's state to all ranks and stamps the consolidation marker. A
        # checkpoint that skipped that consolidation (an old/foreign save, or a
        # rank-0-only dump) leaves non-owner ranks without the owner's momentum, so
        # a resumed run silently diverges. Warn (do NOT raise) so loading such a
        # checkpoint degrades rather than crashes.
        if not self._dist_available():
            return
        consolidated = isinstance(marker, dict) and marker.get("consolidated") is True
        if consolidated:
            return
        try:
            by_pg = self._distributed_state_items(state_dict)
        except (KeyError, TypeError):
            return
        if not by_pg:
            return
        saved_state = state_dict.get("state", {})
        carries_state = any(
            saved_id in saved_state
            for pg_items in by_pg.values()
            for (_, _, saved_id) in pg_items
        )
        if not carries_state:
            return
        warnings.warn(
            "GefenMuon.load_state_dict: loading a sharded_mode='distributed' "
            "checkpoint that is not marked consolidated (missing/incomplete "
            "'gefen_muon_distributed' marker written by GefenMuon.state_dict()). "
            "Non-owner ranks may start with incomplete momentum and the resumed "
            "run can diverge. Re-save with GefenMuon.state_dict() (called on every "
            "rank) to produce a consolidated checkpoint.",
            RuntimeWarning,
            stacklevel=2,
        )

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        marker = state_dict.pop("gefen_muon_distributed", None)
        self._warn_if_unconsolidated_distributed_load(state_dict, marker)
        super().load_state_dict(state_dict)
        # Old checkpoints predate shape-batched NS. Keep their historical,
        # bit-identical serial behavior even when this optimizer was constructed
        # with the new opt-in; current checkpoints carry and restore both keys.
        # This mirrors torch optimizers' usual setdefault migration for newly
        # introduced group options.
        for group in self.param_groups:
            group.setdefault("batched_ns", False)
            group.setdefault(
                "batched_ns_workspace_bytes",
                BATCHED_NS_DEFAULT_WORKSPACE_BYTES,
            )
        self._drop_non_owned_distributed_state()

    @torch.no_grad()
    def step(self, closure=None):
        self._assert_capturable_if_capturing()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Partition the work once so distributed-mode sharded params can take the
        # stable-owner Parallel-Muon path while every other param keeps the normal
        # per-param path.
        distributed_items = []
        regular_items = []
        for group in self.param_groups:
            distributed = group["sharded_mode"] == "distributed"
            for name, p in self._iter_group_params_with_names(group):
                # The constructor validated every parameter as a 2D matrix, so a
                # non-2D parameter here means a wrapper replaced
                # param_groups[*]["params"] AFTER construction (bypassing
                # add_param_group). DeepSpeed ZeRO does exactly that -- it swaps
                # in flattened 1-D fp32 partitions -- and without this guard the
                # step fails later with a cryptic gradient-shape error carrying a
                # stale param name. Fail loudly before any state is touched.
                # (FSDP2/DTensor params keep their GLOBAL 2-D shape, so sharded
                # flows never trip this.)
                if p.ndim != 2:
                    raise ValueError(_swapped_param_groups_error(p))
                grad = p.grad
                if distributed and self._is_sharded(p):
                    distributed_items.append((group, name, p, grad))
                elif grad is not None:
                    regular_items.append((group, name, p, grad))

        self._maybe_refresh_gefen_codebook()
        self._maybe_save_gefen_grad_histogram()

        # "distributed"-mode SHARDED matrices are handled by the bucketed
        # Parallel-Muon path (stable full-param-index owner per matrix); every other
        # matrix (exact / approx / non-sharded) takes the per-param path.
        # The two passes run in the same order on every rank, so all collectives
        # stay matched.
        self._step_regular_items(regular_items)

        if distributed_items:
            self._step_distributed(distributed_items)

        # Capturable stochastic rounding: advance the per-device seed tensors
        # on device (the momentum kernel reads them; see Gefen._sr_seed_on).
        if self.capturable and self._stochastic_round:
            self._advance_sr_seeds()
        # Via the dynamo-disabled helper (NOT a raw += 1): dynamo guards on the
        # exact value of a python int read in a traced frame, so an in-trace
        # increment of this per-step counter would recompile every compiled step.
        self._advance_gefen_global_step()
        if self.capturable and not torch.compiler.is_compiling():
            self._mark_state_static_for_compile()
        return loss

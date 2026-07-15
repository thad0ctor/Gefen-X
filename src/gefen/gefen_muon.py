import math
import re
import warnings
from collections import OrderedDict
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn

from gefen.contracts import (
    OptimizerContract,
    ParameterLayout,
    TopologyChange,
    _gefen_muon_contract,
)
from gefen.gefen import (
    Gefen,
    _amp_optimizer_step_controls,
    _amp_prepare_optimizer_step,
    _assert_optimizer_gradients_structurally_valid,
    _synchronize_step_control_range,
    _synchronize_step_failure,
)

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

# Auto-generated names for bare (unnamed) parameters, as assigned during group
# registration in gefen.py: "param_{i}", "group_{g}_param_{i}", plus the
# "_{n}" uniquing suffix. These names follow registration position, so they
# carry no cross-rank identity of their own.
_AUTO_PARAM_NAME = re.compile(r"^(?:param|group_\d+_param)_\d+(?:_\d+)?$")

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
_DISTRIBUTED_CHECKPOINT_METADATA_KEY = "muon_distributed_state"


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
        deterministic: persist and enforce Gefen's replica-determinism policy
            across checkpoints. GefenMuon's fused momentum reduction is already
            replica-exact on homogeneous GPUs (it uses order-independent max
            reductions); the flag is forwarded so hybrid Muon/backup children
            share one explicit policy.
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

    @staticmethod
    def _step_non_2d_parameter_error(param: torch.Tensor) -> str:
        return _swapped_param_groups_error(param)

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
        deterministic: bool = False,
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
            # GefenMuon owns its second-moment/update pipeline; plain Gefen's
            # factored-v routing is unused here. Pinning it off also keeps the
            # valid deterministic+stochastic-round Muon combination distinct
            # from plain Gefen's deterministic factored fallback.
            factored_v_2d=False,
            stochastic_round=stochastic_round,
            deterministic=deterministic,
            capturable=capturable,
            verbose=verbose,
        )

    def optimizer_contract(self) -> OptimizerContract:
        """Return the immutable Muon state and capability contract."""

        sharded_modes = frozenset(
            group["sharded_mode"] for group in self.param_groups
        )
        normuon_modes = frozenset(
            group["sharded_mode"]
            for group in self.param_groups
            if group.get("normuon", False)
        )
        non_normuon_modes = frozenset(
            group["sharded_mode"]
            for group in self.param_groups
            if not group.get("normuon", False)
        )
        try:
            canonical_state_layouts = self._canonical_state_layouts()
        except Exception:
            canonical_state_layouts = frozenset()
        try:
            from gefen.portable_runtime import _portable_runtime_layouts

            canonical_global_same_topology = _portable_runtime_layouts(self)
        except Exception:
            canonical_global_same_topology = frozenset()
        if canonical_global_same_topology and sharded_modes == frozenset({"distributed"}):
            canonical_global_topology_changing = frozenset(
                {
                    ParameterLayout.REPLICATED,
                    ParameterLayout.WHOLE_PARAMETER_OWNER,
                }
            )
            canonical_global_topology_change_kinds = frozenset(
                {
                    TopologyChange.PLACEMENT_RESHARD,
                    TopologyChange.WORLD_SIZE_OWNER_REDISTRIBUTION,
                }
            )
        else:
            canonical_global_topology_changing = frozenset()
            canonical_global_topology_change_kinds = frozenset()
        return _gefen_muon_contract(
            sharded_modes=sharded_modes,
            normuon_modes=normuon_modes,
            non_normuon_modes=non_normuon_modes,
            canonical_parameter_fqns=self._canonical_identity_ready(),
            stable_shard_identity=self._canonical_identity_ready(),
            explicit_process_group_codebook_scope=True,
            canonical_state_layouts=canonical_state_layouts,
            canonical_global_same_topology=canonical_global_same_topology,
            canonical_global_topology_changing=canonical_global_topology_changing,
            canonical_global_topology_change_kinds=canonical_global_topology_change_kinds,
            atomic_state_movement=self._atomic_state_movement_supported(),
            whole_parameter_owner=(
                self._codebook_scope_ready()
                and any(
                    shard.layout is ParameterLayout.WHOLE_PARAMETER_OWNER
                    for _, shard in self._gefen_local_shard_bindings
                )
            ),
        )

    def _canonical_state_layout_supported(self, layout) -> bool:
        return layout is ParameterLayout.REPLICATED

    def _canonical_state_variant_layout(self):
        sharded_modes = frozenset(
            group["sharded_mode"] for group in self.param_groups
        )
        normuon_modes = frozenset(
            group["sharded_mode"]
            for group in self.param_groups
            if group.get("normuon", False)
        )
        non_normuon_modes = frozenset(
            group["sharded_mode"]
            for group in self.param_groups
            if not group.get("normuon", False)
        )
        return _gefen_muon_contract(
            sharded_modes=sharded_modes,
            normuon_modes=normuon_modes,
            non_normuon_modes=non_normuon_modes,
        ).state_layout

    def _validate_rebinding_layout(self, rebinding) -> None:
        shard = rebinding.shard
        target = rebinding.new_parameter
        if len(shard.parameter.global_shape) != 2:
            raise ValueError("GefenMuon canonical parameters must be logical matrices")
        if shard.layout is ParameterLayout.REPLICATED:
            if target is None or tuple(target.shape) != shard.parameter.global_shape:
                raise ValueError(
                    "replicated GefenMuon rebinding requires one complete 2-D matrix"
                )
        elif shard.layout is ParameterLayout.WHOLE_PARAMETER_OWNER:
            local_owns = shard.local_member == shard.owner
            if local_owns and (
                target is None
                or target.ndim != 2
                or tuple(target.shape) != shard.parameter.global_shape
            ):
                raise ValueError(
                    "a GefenMuon whole-parameter owner requires one complete 2-D matrix"
                )
            if not local_owns and target is not None:
                raise ValueError(
                    "a GefenMuon whole-parameter non-owner must not retain storage"
                )
        else:
            raise ValueError(
                "GefenMuon rebinding supports replicated complete matrices or "
                "whole-parameter ownership only"
            )
        if target is not None and target.numel() != shard.logical_slice.length:
            raise ValueError(
                "GefenMuon rebound tensor numel does not match its logical slice"
            )

    def _has_unscoped_whole_owner_bindings(self) -> bool:
        return self._gefen_codebook_process_group is None and any(
            shard.layout is ParameterLayout.WHOLE_PARAMETER_OWNER
            for _, shard in self._gefen_local_shard_bindings
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

    def _codebook_requires_2d_parameters(self) -> bool:
        return True

    def _iter_gefen_grad_periods(
        self, reuse_existing_periods: bool = False, staged_periods=None
    ):
        # Same as Gefen._iter_gefen_grad_periods, but for sharded (DTensor)
        # gradients gather the FULL matrix (full_tensor) instead of taking the
        # local shard. The exact-DP codebook and the per-param block period are
        # therefore learned from the full matrix on every rank -- identical to
        # the single-GPU reference, and identical across ranks (so quantization
        # matches). With full grads, flat.numel() is the global numel and is
        # never 0, so every rank iterates every parameter in the same order and
        # the full_tensor() collective is matched across ranks.
        for group, param_name, p in self._iter_codebook_params_with_names():
            if p.grad is None:
                continue
            grad = p.grad
            # approx mode learns the codebook/period from the LOCAL shard
            # (no all-gather) so periods divide the local numel that the
            # approximate step operates on; exact mode gathers the full matrix.
            if group["sharded_mode"] == "approx" and hasattr(grad, "to_local"):
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

    @torch._dynamo.disable
    def _step_failure_process_groups(self):
        """Return the control scope for eager exact/distributed Muon steps."""
        if not self._dist_available():
            return ()

        params = [param for group in self.param_groups for param in group["params"]]
        # The protocol returns host-readable flags and therefore cannot be
        # captured. Captured steps already require an eager warmup with a fixed
        # control-flow and gradient-presence pattern.
        if any(param.device.type == "cuda" for param in params) and (
            torch.cuda.is_current_stream_capturing()
        ):
            return ()

        import torch.distributed as dist

        meshes = {}
        for group in self.param_groups:
            if group["sharded_mode"] == "approx":
                continue
            for param in group["params"]:
                if not self._is_sharded(param):
                    continue
                mesh = param.device_mesh
                if mesh.get_coordinate() is None:
                    continue
                process_groups = tuple(mesh.get_all_groups())
                members = tuple(
                    int(item)
                    for item in mesh.mesh.detach().cpu().reshape(-1).tolist()
                )
                key = (
                    str(mesh.device_type),
                    tuple(int(item) for item in mesh.shape),
                    members,
                    tuple(str(pg.group_name) for pg in process_groups),
                )
                meshes.setdefault(key, (mesh, process_groups))

        if not meshes:
            return ()

        member_sets = {frozenset(key[2]) for key in meshes}
        world_members = frozenset(range(dist.get_world_size()))
        participant_union = frozenset().union(*member_sets)
        if world_members in member_sets:
            return (dist.group.WORLD,)

        enclosing_sets = [
            members for members in member_sets if participant_union <= members
        ]
        if len(member_sets) > 1 and not enclosing_sets:
            # No existing group encloses every upcoming collective participant.
            # Entering only a rank-local subset of those groups could create a
            # new deadlock before the established mesh/order preflight. Leave
            # this uncommon topology on its existing behavior until the caller
            # supplies an explicit enclosing control group.
            return ()

        selected_members = (
            min(enclosing_sets, key=lambda item: (len(item), sorted(item)))
            if enclosing_sets
            else next(iter(member_sets))
        )
        candidates = [
            (key, entry)
            for key, entry in meshes.items()
            if frozenset(key[2]) == selected_members
        ]
        _, (mesh, process_groups) = min(candidates, key=lambda item: item[0])
        if mesh.size() < 2:
            return ()
        if mesh.ndim == 1:
            return (process_groups[0],)
        return tuple(
            process_group
            for process_group in process_groups
            if dist.get_world_size(process_group) > 1
        )

    @staticmethod
    @torch._dynamo.disable
    def _synchronize_sharded_step_flag(local_value, process_groups) -> bool:
        synchronized = bool(local_value)
        for process_group in process_groups:
            synchronized = _synchronize_step_failure(
                synchronized, process_group
            )
        return synchronized

    @torch._dynamo.disable
    def _synchronize_sharded_step_error(
        self, error, phase: str, process_groups
    ) -> None:
        if not process_groups:
            if error is not None:
                raise error
            return
        failed = self._synchronize_sharded_step_flag(
            error is not None, process_groups
        )
        if not failed:
            return

        import torch.distributed as dist

        if error is not None:
            raise RuntimeError(
                "GefenMuon {} failed on local rank {}: {}".format(
                    phase, dist.get_rank(), error
                )
            ) from error
        raise RuntimeError(
            "GefenMuon {} failed on another process-group member".format(
                phase
            )
        )

    @staticmethod
    @torch._dynamo.disable
    def _synchronize_sharded_step_control_range(local_control, process_groups):
        minimum = tuple(float(item) for item in local_control)
        maximum = minimum
        for process_group in process_groups:
            minimum, maximum = _synchronize_step_control_range(
                minimum, maximum, process_group
            )
        return minimum, maximum

    @torch._dynamo.disable
    def _prepare_synchronized_amp_step(self, optimizer, process_groups) -> bool:
        """Agree on AMP controls before unscaling or entering Muon collectives."""
        local_present = hasattr(optimizer, "found_inf") or hasattr(
            optimizer, "grad_scale"
        )
        if not process_groups:
            if not local_present:
                return True
            return _amp_prepare_optimizer_step(optimizer)

        try:
            if local_present:
                local_overflow, grad_scale, scale_value = (
                    _amp_optimizer_step_controls(optimizer)
                )
                local_scale_present = grad_scale is not None
            else:
                local_overflow = False
                local_scale_present = False
                scale_value = 0.0
            local_amp_error = None
        except Exception as exc:
            local_overflow = False
            local_scale_present = False
            scale_value = 0.0
            local_amp_error = exc
        self._synchronize_sharded_step_error(
            local_amp_error, "AMP control preflight", process_groups
        )

        minimum, maximum = self._synchronize_sharded_step_control_range(
            (
                int(local_present),
                int(local_overflow),
                int(local_scale_present),
                scale_value,
            ),
            process_groups,
        )
        if minimum[0] != maximum[0]:
            raise RuntimeError(
                "GefenMuon AMP protocol presence differs across process-group "
                "members; use the same group-aware gradient scaler on every member"
            )
        if not bool(maximum[0]):
            return True
        if minimum[1] != maximum[1]:
            raise RuntimeError(
                "GefenMuon AMP found_inf differs across process-group members; "
                "use a group-aware gradient scaler"
            )
        if minimum[2] != maximum[2]:
            raise RuntimeError(
                "GefenMuon AMP grad_scale presence differs across process-group "
                "members; use the same group-aware gradient scaler on every member"
            )
        if bool(maximum[2]) and minimum[3] != maximum[3]:
            raise RuntimeError(
                "GefenMuon AMP grad_scale differs across process-group members; "
                "use a group-aware gradient scaler"
            )
        if bool(maximum[1]):
            return False

        try:
            should_step = _amp_prepare_optimizer_step(optimizer)
            local_amp_error = None
        except Exception as exc:
            should_step = False
            local_amp_error = exc
        self._synchronize_sharded_step_error(
            local_amp_error, "AMP preparation", process_groups
        )
        return should_step

    @torch._dynamo.disable
    def _assert_sharded_grad_presence_consistent(self) -> None:
        """Fail before collectives when mesh ranks disagree on the step inputs.

        Exact and distributed Muon reconstruct full DTensor gradients with
        collectives entered in parameter-group insertion order, and
        ``_step_distributed_pg`` assigns momentum owners by that same insertion
        index. Two rank-local properties therefore must agree mesh-wide before
        any of those collectives start: which parameters have gradients, and
        the order the parameters were registered in. Pack both per DeviceMesh
        — an activity bit summed across every mesh dimension, and an insertion
        position reduced to its mesh-wide max and min — so every participating
        rank sees the same global result and raises on the same disagreement
        before codebook learning or optimizer state/parameter mutation begins.

        Plain tensors/DDP and local-shard ``approx`` mode take no Muon gradient
        collectives and therefore pay no preflight collective. Manual CUDA graph
        capture also skips this host-branching check: capturable Gefen requires
        eager warmup before capture, and those warmup steps establish that the
        graph's fixed gradient-presence pattern is rank-consistent.
        """
        if not self._dist_available():
            return
        # Do not initialize or query CUDA for a CPU-only mesh optimizer.
        # Spawned Gloo workers may run alongside a CUDA-heavy parent process,
        # and an unrelated current-stream query can otherwise block before the
        # CPU collective preflight. CUDA-backed optimizers retain the capture
        # guard.
        if any(
            param.device.type == "cuda"
            for group in self.param_groups
            for param in group["params"]
        ) and torch.cuda.is_current_stream_capturing():
            return

        import torch.distributed as dist

        by_mesh = OrderedDict()
        # Optimizer-wide insertion positions: step() walks parameters and
        # first-seen process groups in this order, so cross-mesh order
        # divergence is just as collective-fatal as divergence inside one
        # mesh. Per-mesh enumeration would hide it.
        global_position = {}
        for group in self.param_groups:
            if group["sharded_mode"] == "approx":
                continue
            for name, p in self._iter_group_params_with_names(group):
                if not self._is_sharded(p):
                    continue
                mesh = p.device_mesh
                if mesh.get_coordinate() is None or mesh.size() < 2:
                    continue
                # Two equivalent DeviceMesh objects can still refer to distinct
                # c10d groups, and a mis-built training script can register the
                # same parameters in a different order per rank. Key each mesh
                # by content instead of object identity, and sort meshes and
                # items below, so this preflight's own collectives stay matched
                # under both — and rank-divergent registration order is then
                # detected positionally rather than corrupting the comparison.
                process_groups = tuple(mesh.get_all_groups())
                key = (
                    str(mesh.device_type),
                    tuple(int(item) for item in mesh.shape),
                    tuple(
                        int(item)
                        for item in mesh.mesh.detach().cpu().reshape(-1).tolist()
                    ),
                    tuple(str(pg.group_name) for pg in process_groups),
                )
                entry = by_mesh.get(key)
                if entry is None:
                    entry = {
                        "mesh": mesh,
                        "process_groups": process_groups,
                        "items": [],
                    }
                    by_mesh[key] = entry
                entry["items"].append((str(name), p, p.grad is not None))
                global_position[id(p)] = len(global_position)

        mismatches = []
        order_mismatches = []
        duplicate_labels = []
        duplicates_anywhere = False
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
            # Two parameters carrying the same explicit name, shape, and dtype
            # are indistinguishable across ranks, so neither this order probe
            # nor the presence vector below can align them; construction
            # accepts duplicate explicit names, so fail closed on them. The
            # flag rides the reduced probe below rather than raising here:
            # a rank whose own labels are clean must learn about duplicates on
            # its peers and take the same error path, not enter a collective
            # its peer already abandoned.
            duplicate_keys = sorted(
                {
                    items[index][0]
                    for index in range(1, len(items))
                    if items[index][0] == items[index - 1][0]
                    and tuple(items[index][1].shape)
                    == tuple(items[index - 1][1].shape)
                    and str(items[index][1].dtype)
                    == str(items[index - 1][1].dtype)
                }
            )
            # Bare parameters receive positional auto-names, so two of them
            # with the same shape and dtype have no rank-invariant identity at
            # all: the probes below would align labels that themselves follow
            # position, and no rank-local property can do better. Ranks built
            # by the same construction code (the normal case) remain correct;
            # warn once that divergence between such parameters is invisible
            # to this preflight, and recommend named construction.
            if not getattr(self, "_gefen_warned_unverifiable_order", False):
                collisions = {}
                for name, p, _ in items:
                    if _AUTO_PARAM_NAME.match(str(name)):
                        key = (tuple(p.shape), str(p.dtype))
                        collisions.setdefault(key, []).append(str(name))
                ambiguous = sorted(
                    name
                    for names in collisions.values()
                    if len(names) > 1
                    for name in names
                )
                if ambiguous:
                    warnings.warn(
                        "GefenMuon cannot verify cross-rank parameter order "
                        "for unnamed parameters that share a shape and dtype "
                        "({}): their auto-generated names follow registration "
                        "position, so rank-divergent registration order "
                        "between them is undetectable. Construct the "
                        "optimizer from model.named_parameters() to make "
                        "this check complete.".format(", ".join(ambiguous)),
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    self._gefen_warned_unverifiable_order = True
            device = self._state_tensor_device(items[0][1])
            active_counts = torch.tensor(
                [int(active) for _, _, active in items],
                dtype=torch.int32,
                device=device,
            )
            # Each sorted slot is the same parameter identity on every rank, so
            # reducing its optimizer-wide insertion position to the mesh-wide
            # max and min (one MAX all_reduce over [position, -position])
            # exposes any rank whose registration order differs — including
            # order swaps between parameters on different meshes. The final
            # slot carries this rank's duplicate-label count so the reduction
            # also propagates duplicates to every rank of the mesh.
            order_probe = torch.tensor(
                [
                    signed
                    for _, p, _ in items
                    for signed in (
                        global_position[id(p)],
                        -global_position[id(p)],
                    )
                ]
                + [len(duplicate_keys)],
                dtype=torch.int32,
                device=device,
            )
            # Reducing the same vectors once along every Cartesian mesh
            # dimension propagates mesh-global results to every rank. This is
            # two collectives per mesh dimension, independent of the number of
            # optimizer parameters.
            for process_group in entry["process_groups"]:
                if dist.get_world_size(process_group) > 1:
                    dist.all_reduce(
                        active_counts, op=dist.ReduceOp.SUM, group=process_group
                    )
                    dist.all_reduce(
                        order_probe, op=dist.ReduceOp.MAX, group=process_group
                    )

            # Do not raise inside this loop: with DTensors on overlapping but
            # non-identical meshes, a rank exiting early would strand peers
            # that only share a later mesh inside that mesh's all_reduce.
            # Accumulate every violation and raise after all local meshes
            # have completed their probe collectives, like the presence path.
            order_flat = order_probe.cpu().tolist()
            if order_flat[-1] > 0:
                duplicates_anywhere = True
                duplicate_labels.extend(duplicate_keys)
            order_mismatches.extend(
                items[index][0]
                for index in range(len(items))
                if order_flat[2 * index] != -order_flat[2 * index + 1]
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

        if duplicates_anywhere:
            raise RuntimeError(
                "GefenMuon cannot verify cross-rank parameter order or "
                "gradient presence when parameters in one "
                "sharded_mode='exact' or 'distributed' DTensor mesh share "
                "a name, shape, and dtype{}: the shared label makes them "
                "indistinguishable across ranks. Give every parameter a "
                "unique name.".format(
                    " ({})".format(", ".join(sorted(set(duplicate_labels))))
                    if duplicate_labels
                    else " on at least one mesh rank"
                )
            )
        if order_mismatches:
            raise RuntimeError(
                "GefenMuon requires identical parameter-group order on "
                "every rank of a DTensor/FSDP mesh before "
                "sharded_mode='exact' or 'distributed' stepping: gradient "
                "collectives and momentum-owner assignment follow that "
                "order. Parameters at rank-divergent positions: {}. "
                "Construct parameter groups in the same order on every "
                "rank.".format(", ".join(order_mismatches))
            )
        if mismatches:
            raise RuntimeError(
                "GefenMuon requires identical gradient presence on every rank "
                "of a DTensor/FSDP mesh before sharded_mode='exact' or "
                "'distributed' stepping. Mismatched parameters: {}. Ensure "
                "conditional or unused parameters produce the same `.grad is "
                "None` pattern on every mesh rank.".format(", ".join(mismatches))
            )

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

        marker_groups = []
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

            marker_groups.append(
                {
                    "world_size": world,
                    "params": [
                        {
                            "saved_id": saved_id,
                            "name": str(name),
                            "shape": tuple(p.shape),
                            "owner": owner_by_saved_id[saved_id],
                            "state_keys": tuple(
                                sorted(
                                    str(key)
                                    for key in state_dict["state"]
                                    .get(saved_id, {})
                                    .keys()
                                )
                            ),
                            "initialized": any(
                                str(key) != "name"
                                for key in state_dict["state"]
                                .get(saved_id, {})
                                .keys()
                            ),
                        }
                        for name, p, saved_id in pg_items
                    ],
                }
            )

        state_dict["gefen_muon_distributed"] = {
            "version": 2,
            "ownership": "stable_full_param_index_v1",
            "consolidated": True,
            "groups": marker_groups,
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

    def _state_dict_impl(self):
        # Parallel-Muon owner state must be made complete before Gefen's
        # rank-local DTensor adapter serializes the per-rank payload. This
        # ordering matters for mixed optimizers carrying both ``approx`` and
        # ``distributed`` groups: wrapping first replaces every real state entry
        # with opaque transport tensors, which are not owner momentum.
        state_dict = super()._state_dict_impl(consolidate_rank_local=False)
        state_dict = self._consolidate_distributed_state_dict(state_dict)

        checkpoint_metadata = None
        groups = state_dict.get("param_groups", ())
        if groups:
            checkpoint_metadata = groups[0].get("_gefen_checkpoint_metadata")
        if self._uses_rank_local_sharded_state():
            if not isinstance(checkpoint_metadata, dict):
                raise RuntimeError(
                    "GefenMuon rank-local checkpoint metadata was not initialized"
                )
            self._consolidate_rank_local_sharded_state(
                state_dict, checkpoint_metadata
            )

        marker = state_dict.get("gefen_muon_distributed")
        if marker is not None:
            # Generic DCP keeps only conventional state/param_groups top-level
            # keys. Mirror the owner proof into every group's private transport
            # metadata so ordinary and flattened DCP round-trips retain it.
            for group in groups:
                metadata = dict(group.get("_gefen_checkpoint_metadata", {}))
                metadata[_DISTRIBUTED_CHECKPOINT_METADATA_KEY] = marker
                group["_gefen_checkpoint_metadata"] = metadata
        return state_dict

    @staticmethod
    def _distributed_checkpoint_error(detail: str) -> ValueError:
        return ValueError(
            "GefenMuon.load_state_dict refused populated "
            "sharded_mode='distributed' optimizer state before mutation: {}. "
            "Parallel-Muon momentum must be consolidated by calling "
            "GefenMuon.state_dict() on every save rank and must carry either a "
            "canonical released version-1 consolidation proof or a valid "
            "version-2 'gefen_muon_distributed' saved-world/owner manifest. "
            "Re-save from the original distributed job; do not load a rank-local "
            "or manually stripped optimizer state.".format(detail)
        )

    def _distributed_checkpoint_marker(self, state_dict):
        """Recover one consistent owner proof from top-level/group transport."""

        marker = state_dict.pop("gefen_muon_distributed", None)
        groups = state_dict.get("param_groups", ())
        metadata_markers = []
        metadata_presence = []
        for group in groups if isinstance(groups, (list, tuple)) else ():
            metadata = (
                group.get("_gefen_checkpoint_metadata")
                if isinstance(group, dict)
                else None
            )
            present = (
                isinstance(metadata, dict)
                and _DISTRIBUTED_CHECKPOINT_METADATA_KEY in metadata
            )
            metadata_presence.append(present)
            if present:
                metadata_markers.append(
                    metadata[_DISTRIBUTED_CHECKPOINT_METADATA_KEY]
                )

        if any(metadata_presence) and not all(metadata_presence):
            raise self._distributed_checkpoint_error(
                "the parameter-group copies of the owner manifest are incomplete"
            )
        if metadata_markers:
            metadata_marker = metadata_markers[0]
            try:
                copies_agree = all(
                    item == metadata_marker for item in metadata_markers[1:]
                )
            except Exception:
                copies_agree = False
            if not copies_agree:
                raise self._distributed_checkpoint_error(
                    "the parameter-group copies of the owner manifest disagree"
                )
            if marker is None:
                marker = metadata_marker
            else:
                try:
                    top_level_agrees = marker == metadata_marker
                except Exception:
                    top_level_agrees = False
                if not top_level_agrees:
                    raise self._distributed_checkpoint_error(
                        "the top-level and parameter-group owner manifests disagree"
                    )
        return marker

    def _distributed_load_state_items(self, state_dict):
        """Bind saved distributed groups to live params without zip truncation."""
        saved_groups = state_dict.get("param_groups")
        if not isinstance(saved_groups, (list, tuple)):
            raise self._distributed_checkpoint_error(
                "the checkpoint param_groups payload is not a sequence"
            )
        if not any(
            isinstance(group, dict) and group.get("sharded_mode") == "distributed"
            for group in saved_groups
        ):
            return [], OrderedDict()
        if len(saved_groups) != len(self.param_groups):
            raise self._distributed_checkpoint_error(
                "the checkpoint has {} parameter groups but the live optimizer "
                "has {}".format(len(saved_groups), len(self.param_groups))
            )

        saved_state = state_dict.get("state", {})
        expected_items = []
        seen_saved_ids = set()
        by_pg = OrderedDict()
        for group_index, (saved_group, live_group) in enumerate(
            zip(saved_groups, self.param_groups)
        ):
            if not isinstance(saved_group, dict):
                raise self._distributed_checkpoint_error(
                    "checkpoint parameter group {} is not a mapping".format(
                        group_index
                    )
                )
            saved_ids = saved_group.get("params")
            if not isinstance(saved_ids, (list, tuple)):
                raise self._distributed_checkpoint_error(
                    "checkpoint parameter group {} has no parameter-id list".format(
                        group_index
                    )
                )
            live_items = list(self._iter_group_params_with_names(live_group))
            if len(saved_ids) != len(live_items):
                raise self._distributed_checkpoint_error(
                    "checkpoint parameter group {} has {} parameters but the live "
                    "group has {}".format(
                        group_index, len(saved_ids), len(live_items)
                    )
                )
            if saved_group.get("sharded_mode") != "distributed":
                continue

            saved_names = saved_group.get("param_names")
            if saved_names is not None and (
                not isinstance(saved_names, (list, tuple))
                or len(saved_names) != len(saved_ids)
            ):
                raise self._distributed_checkpoint_error(
                    "checkpoint distributed group {} has an invalid param_names "
                    "manifest".format(group_index)
                )
            for param_index, ((live_name, live_param), saved_id) in enumerate(
                zip(live_items, saved_ids)
            ):
                if saved_id in seen_saved_ids:
                    raise self._distributed_checkpoint_error(
                        "checkpoint distributed parameter id {!r} appears more "
                        "than once".format(saved_id)
                    )
                seen_saved_ids.add(saved_id)
                state_name = None
                if isinstance(saved_state, dict):
                    pstate = saved_state.get(saved_id)
                    if isinstance(pstate, dict):
                        state_name = pstate.get("name")
                saved_name = (
                    saved_names[param_index]
                    if saved_names is not None
                    else state_name
                )
                if saved_name is not None and str(saved_name).lower() != str(
                    live_name
                ).lower():
                    raise self._distributed_checkpoint_error(
                        "checkpoint distributed group {} parameter {} name {!r} "
                        "does not match live name {!r}".format(
                            group_index, param_index, saved_name, str(live_name)
                        )
                    )
                item = (str(live_name).lower(), live_param, saved_id)
                expected_items.append(item)
                pg = self._distributed_process_group(live_param)
                if pg is not None:
                    by_pg.setdefault(pg, []).append(item)
        return expected_items, by_pg

    @staticmethod
    def _distributed_checkpoint_is_pristine(state_dict, expected_items) -> bool:
        steps = []
        if "gefen_global_step" in state_dict:
            steps.append(state_dict.get("gefen_global_step"))
        codebooks = [state_dict.get("gefen_codebook")]
        for group in state_dict.get("param_groups", ()):
            if not isinstance(group, dict):
                return False
            metadata = group.get("_gefen_checkpoint_metadata")
            if isinstance(metadata, dict):
                if "global_step" in metadata:
                    steps.append(metadata.get("global_step"))
                codebooks.append(metadata.get("codebook"))
        if not steps or any(type(step) is not int or step != 0 for step in steps):
            return False
        if any(codebook is not None for codebook in codebooks):
            return False

        saved_state = state_dict.get("state", {})
        if not isinstance(saved_state, dict):
            return False
        for expected_name, _, saved_id in expected_items:
            pstate = saved_state.get(saved_id)
            if (
                not isinstance(pstate, dict)
                or set(pstate) != {"name"}
                or str(pstate.get("name")).lower() != expected_name
            ):
                return False
        return True

    def _distributed_checkpoint_codebook(self, state_dict):
        candidates = []
        if "gefen_codebook" in state_dict:
            candidates.append(state_dict.get("gefen_codebook"))
        for group in state_dict.get("param_groups", ()):
            if not isinstance(group, dict):
                continue
            metadata = group.get("_gefen_checkpoint_metadata")
            if isinstance(metadata, dict) and "codebook" in metadata:
                candidates.append(metadata.get("codebook"))
        non_null = [value for value in candidates if value is not None]
        if not non_null:
            return None
        codebook = non_null[0]
        for other in non_null[1:]:
            if not (
                torch.is_tensor(codebook)
                and torch.is_tensor(other)
                and torch.equal(codebook, other)
            ):
                raise self._distributed_checkpoint_error(
                    "checkpoint copies of the frozen codebook disagree"
                )
        return codebook

    def _validate_distributed_codebook(self, state_dict, *, required: bool) -> None:
        try:
            self._validate_rank_local_codebook(
                self._distributed_checkpoint_codebook(state_dict),
                required=required,
            )
        except ValueError as exc:
            raise self._distributed_checkpoint_error(str(exc)) from exc

    @staticmethod
    def _distributed_counter_is_valid(value, *, minimum: int = 1) -> bool:
        if torch.is_tensor(value):
            if (
                value.dim() != 0
                or value.dtype == torch.bool
                or not bool(torch.isfinite(value).all())
            ):
                return False
            scalar = float(value.detach().cpu().item())
            return scalar >= minimum and scalar.is_integer()
        return type(value) is int and value >= minimum

    def _validate_distributed_param_state(
        self, pstate, expected_name: str, param: torch.Tensor
    ) -> bool:
        """Validate owner momentum independently of the consolidation marker."""
        if not isinstance(pstate, dict):
            raise self._distributed_checkpoint_error(
                "state for parameter {!r} is missing or is not a mapping".format(
                    expected_name
                )
            )
        if not all(isinstance(key, str) for key in pstate):
            raise self._distributed_checkpoint_error(
                "state for parameter {!r} has non-string keys".format(expected_name)
            )
        if str(pstate.get("name")).lower() != expected_name:
            raise self._distributed_checkpoint_error(
                "state name {!r} does not match parameter {!r}".format(
                    pstate.get("name"), expected_name
                )
            )
        if set(pstate) == {"name"}:
            return False

        core = {"name", "automatic_period", "step", "m_codebook", "m_magnitude"}
        missing = core - set(pstate)
        if missing:
            raise self._distributed_checkpoint_error(
                "state for parameter {!r} is initialized but missing core keys "
                "{}".format(expected_name, sorted(missing))
            )
        period = pstate["automatic_period"]
        if type(period) is not int or period <= 0 or param.numel() % period != 0:
            raise self._distributed_checkpoint_error(
                "parameter {!r} has invalid automatic_period {!r} for {} "
                "elements".format(expected_name, period, param.numel())
            )
        blocks = param.numel() // period
        indices = pstate["m_codebook"]
        magnitude = pstate["m_magnitude"]
        if (
            not torch.is_tensor(indices)
            or indices.dtype != torch.uint8
            or tuple(indices.shape) != (blocks, period)
        ):
            raise self._distributed_checkpoint_error(
                "parameter {!r} has invalid m_codebook dtype/geometry".format(
                    expected_name
                )
            )
        if (
            not torch.is_tensor(magnitude)
            or magnitude.dtype != torch.float32
            or tuple(magnitude.shape) != (blocks, 1)
            or not bool(torch.isfinite(magnitude).all())
            or not bool((magnitude >= 0).all())
        ):
            raise self._distributed_checkpoint_error(
                "parameter {!r} has invalid m_magnitude dtype/geometry/values".format(
                    expected_name
                )
            )
        if not self._distributed_counter_is_valid(pstate["step"]):
            raise self._distributed_checkpoint_error(
                "parameter {!r} has invalid step counter".format(expected_name)
            )
        normuon_v = pstate.get("normuon_v")
        normuon_step = pstate.get("normuon_step")
        if (normuon_v is None) != (normuon_step is None):
            raise self._distributed_checkpoint_error(
                "parameter {!r} has incomplete NorMuon state".format(expected_name)
            )
        if normuon_v is not None and (
            not torch.is_tensor(normuon_v)
            or normuon_v.dtype != torch.float32
            or tuple(normuon_v.shape) != (param.shape[0], 1)
            or not bool(torch.isfinite(normuon_v).all())
            or not bool((normuon_v >= 0).all())
            or not self._distributed_counter_is_valid(normuon_step)
        ):
            raise self._distributed_checkpoint_error(
                "parameter {!r} has invalid NorMuon dtype/geometry/counter".format(
                    expected_name
                )
            )
        return True

    def _validate_distributed_checkpoint_load(self, state_dict, marker) -> None:
        # In distributed mode the persistent momentum lives only on each matrix's
        # stable owner rank; state_dict() runs a collective that broadcasts every
        # owner's state to all ranks and stamps a manifest describing the saved
        # process-group world, stable owner, parameter identity/order, and state
        # keys. A rank-local dump can otherwise look loadable on every rank while
        # silently resetting the non-owner momentum. Validate before delegating to
        # the base loader so every rejection is mutation-free.
        try:
            expected_items, by_pg = self._distributed_load_state_items(state_dict)
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith(
                "GefenMuon.load_state_dict refused"
            ):
                raise
            raise self._distributed_checkpoint_error(
                "the checkpoint parameter-group layout cannot be matched to the "
                "live distributed DTensor parameters ({})".format(exc)
            ) from exc
        if not expected_items:
            return

        saved_state = state_dict.get("state", {})
        if not isinstance(saved_state, dict):
            raise self._distributed_checkpoint_error(
                "the checkpoint 'state' payload is not a mapping"
            )
        pristine = self._distributed_checkpoint_is_pristine(
            state_dict, expected_items
        )
        if pristine and marker is None:
            return

        if not isinstance(marker, dict):
            # A world-1 or unsupported-mesh "distributed" group takes Muon's
            # replicated exact fallback and never had owner-local state to
            # consolidate. Preserve such markerless legacy loads only when every
            # expected entry is independently proven complete.
            if not by_pg:
                initialized = [
                    self._validate_distributed_param_state(
                        saved_state.get(saved_id), expected_name, param
                    )
                    for expected_name, param, saved_id in expected_items
                ]
                if initialized and all(initialized):
                    self._validate_distributed_codebook(
                        state_dict, required=True
                    )
                    return
            raise self._distributed_checkpoint_error(
                "the consolidation marker is missing or is not a mapping"
            )
        version = marker.get("version")
        if version not in (1, 2):
            raise self._distributed_checkpoint_error(
                "the consolidation marker version is {!r}, expected 1 or 2".format(
                    version
                )
            )
        if marker.get("ownership") != "stable_full_param_index_v1":
            raise self._distributed_checkpoint_error(
                "the ownership scheme is {!r}, expected "
                "'stable_full_param_index_v1'".format(marker.get("ownership"))
            )
        if marker.get("consolidated") is not True:
            raise self._distributed_checkpoint_error(
                "the consolidation marker does not assert consolidated=True"
            )

        if version == 1:
            if pristine:
                return
            initialized = [
                self._validate_distributed_param_state(
                    saved_state.get(saved_id), expected_name, param
                )
                for expected_name, param, saved_id in expected_items
            ]
            if not initialized or not all(initialized):
                raise self._distributed_checkpoint_error(
                    "released version-1 state mixes initialized and empty owner "
                    "entries, so completeness cannot be proven"
                )
            self._validate_distributed_codebook(state_dict, required=True)
            return

        marker_groups = marker.get("groups")
        if not isinstance(marker_groups, (list, tuple)):
            raise self._distributed_checkpoint_error(
                "the saved-world/owner group manifest is missing or is not a list"
            )

        # A sharded_mode='distributed' optimizer group may contain a mixture of
        # Parallel-Muon-eligible parameters and replicated fallbacks (plain
        # tensors, multi-dimensional meshes, or world-one meshes). Save-side
        # consolidation creates one owner manifest per eligible process group
        # and deliberately leaves fallback state to the ordinary Gefen loader.
        # Bind the proof to those same ordered process-group partitions instead
        # of requiring it to cover every parameter in the optimizer group.
        expected_pg_groups = list(by_pg.values())
        if not expected_pg_groups and marker_groups:
            # A consolidated version-2 checkpoint carries the complete owner
            # state on every rank, so it must stay loadable when the live
            # optimizer has no Parallel-Muon-eligible process group at all
            # (single-process resume/eval, uninitialized torch.distributed, or
            # an all-fallback topology) — released-v1 and markerless loads of
            # the same state already support exactly that. The manifest cannot
            # bind to live process groups then; rebind each saved partition to
            # the live parameters by saved id and run the identical
            # per-parameter owner-state proof below on the rebound partitions.
            items_by_saved_id = {item[2]: item for item in expected_items}
            rebound_ids = set()
            rebound_groups = []
            for group_index, saved_group in enumerate(marker_groups):
                group_items = []
                saved_params = (
                    saved_group.get("params")
                    if isinstance(saved_group, dict)
                    else None
                )
                if isinstance(saved_params, (list, tuple)):
                    for saved_param in saved_params:
                        if not isinstance(saved_param, dict):
                            continue
                        saved_id = saved_param.get("saved_id")
                        try:
                            item = items_by_saved_id.get(saved_id)
                        except TypeError:
                            item = None
                        if item is None or saved_id in rebound_ids:
                            raise self._distributed_checkpoint_error(
                                "owner manifest group {} saved id {!r} does not "
                                "map to exactly one live distributed "
                                "parameter".format(group_index, saved_id)
                            )
                        rebound_ids.add(saved_id)
                        group_items.append(item)
                rebound_groups.append(group_items)
            expected_pg_groups = rebound_groups
        if len(marker_groups) != len(expected_pg_groups):
            raise self._distributed_checkpoint_error(
                "the owner manifest has {} process-group entries but the live "
                "optimizer has {} eligible distributed process groups".format(
                    len(marker_groups), len(expected_pg_groups)
                )
            )

        eligible_ids = {
            saved_id
            for pg_items in expected_pg_groups
            for _, _, saved_id in pg_items
        }
        fallback_items = [
            item for item in expected_items if item[2] not in eligible_ids
        ]
        initialized_any = False
        for expected_name, param, saved_id in fallback_items:
            initialized_any = (
                self._validate_distributed_param_state(
                    saved_state.get(saved_id), expected_name, param
                )
                or initialized_any
            )

        for group_index, (saved_group, expected_pg_items) in enumerate(
            zip(marker_groups, expected_pg_groups)
        ):
            if not isinstance(saved_group, dict):
                raise self._distributed_checkpoint_error(
                    "owner manifest group {} is not a mapping".format(group_index)
                )
            saved_world = saved_group.get("world_size")
            if (
                not isinstance(saved_world, int)
                or isinstance(saved_world, bool)
                or saved_world <= 0
            ):
                raise self._distributed_checkpoint_error(
                    "owner manifest group {} has invalid saved world_size {!r}".format(
                        group_index, saved_world
                    )
                )
            saved_params = saved_group.get("params")
            if not isinstance(saved_params, (list, tuple)):
                raise self._distributed_checkpoint_error(
                    "owner manifest group {} has no ordered parameter list".format(
                        group_index
                    )
                )
            expected_ids = [saved_id for _, _, saved_id in expected_pg_items]
            manifest_ids = []
            for param_index, saved_param in enumerate(saved_params):
                if not isinstance(saved_param, dict):
                    raise self._distributed_checkpoint_error(
                        "owner manifest group {} parameter {} is not a mapping".format(
                            group_index, param_index
                        )
                    )
                expected_owner = _stable_distributed_owner(param_index, saved_world)
                saved_owner = saved_param.get("owner")
                if (
                    not isinstance(saved_owner, int)
                    or isinstance(saved_owner, bool)
                    or saved_owner != expected_owner
                ):
                    raise self._distributed_checkpoint_error(
                        "owner manifest group {} parameter {} has owner {!r}, "
                        "expected {} for saved world {}".format(
                            group_index,
                            param_index,
                            saved_owner,
                            expected_owner,
                            saved_world,
                        )
                    )
                manifest_ids.append(saved_param.get("saved_id"))

            try:
                manifest_id_set = set(manifest_ids)
            except TypeError as exc:
                raise self._distributed_checkpoint_error(
                    "the owner manifest contains an unhashable saved_id"
                ) from exc
            if (
                len(manifest_ids) != len(expected_ids)
                or len(manifest_id_set) != len(manifest_ids)
                or manifest_ids != expected_ids
            ):
                raise self._distributed_checkpoint_error(
                    "owner manifest group {} parameter ids {!r} do not exactly "
                    "match the ordered eligible checkpoint/live ids {!r}".format(
                        group_index, manifest_ids, expected_ids
                    )
                )

            for param_index, (saved_param, live_item) in enumerate(
                zip(saved_params, expected_pg_items)
            ):
                live_name, live_param, saved_id = live_item
                if saved_param.get("name") != str(live_name):
                    raise self._distributed_checkpoint_error(
                        "owner manifest group {} parameter {} name {!r} does not "
                        "match live name {!r}".format(
                            group_index,
                            param_index,
                            saved_param.get("name"),
                            str(live_name),
                        )
                    )
                saved_shape = saved_param.get("shape")
                if not isinstance(saved_shape, (list, tuple)) or tuple(
                    saved_shape
                ) != tuple(live_param.shape):
                    raise self._distributed_checkpoint_error(
                        "owner manifest group {} parameter {} shape {!r} does not "
                        "match live shape {!r}".format(
                            group_index,
                            param_index,
                            saved_shape,
                            tuple(live_param.shape),
                        )
                    )
                saved_keys = saved_param.get("state_keys")
                if (
                    not isinstance(saved_keys, (list, tuple))
                    or not all(isinstance(key, str) for key in saved_keys)
                    or len(set(saved_keys)) != len(saved_keys)
                ):
                    raise self._distributed_checkpoint_error(
                        "owner manifest group {} parameter {} has no state_keys "
                        "manifest".format(group_index, param_index)
                    )
                actual_param_state = saved_state.get(saved_id)
                initialized = self._validate_distributed_param_state(
                    actual_param_state, str(live_name).lower(), live_param
                )
                initialized_any = initialized_any or initialized
                actual_keys = tuple(sorted(str(key) for key in actual_param_state))
                if tuple(sorted(saved_keys)) != actual_keys:
                    raise self._distributed_checkpoint_error(
                        "owner manifest group {} parameter {} state keys {!r} do "
                        "not match checkpoint state keys {!r}".format(
                            group_index,
                            param_index,
                            tuple(saved_keys),
                            actual_keys,
                        )
                    )
                marker_initialized = saved_param.get("initialized")
                if (
                    type(marker_initialized) is not bool
                    or marker_initialized != initialized
                ):
                    raise self._distributed_checkpoint_error(
                        "owner manifest group {} parameter {} initialized={!r} "
                        "does not match checkpoint state initialized={}".format(
                            group_index,
                            param_index,
                            marker_initialized,
                            initialized,
                        )
                    )
        self._validate_distributed_codebook(state_dict, required=initialized_any)

    def _load_state_dict_impl(self, state_dict):
        state_dict = dict(state_dict)
        state_dict = self._pack_legacy_param_groups_for_load(state_dict)
        marker = self._distributed_checkpoint_marker(state_dict)

        # Unwrap only a copied transaction for Parallel-Muon validation. The
        # base loader unwraps and loads the original exactly once after every
        # owner/fallback invariant has passed, keeping all rejection paths
        # mutation-free while composing with rank-local approx state.
        validation_state = dict(state_dict)
        validation_state["param_groups"] = [
            dict(group) for group in state_dict.get("param_groups", ())
        ]
        self._unwrap_rank_local_sharded_checkpoint(validation_state)
        self._validate_distributed_checkpoint_load(validation_state, marker)
        super()._load_state_dict_impl(state_dict)
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
        self._install_rank_local_checkpoint_schema()

    @torch.no_grad()
    def step(self, closure=None):
        self._assert_finalized_binding_layout()
        self._assert_runtime_codebook_process_group()
        if self._has_unscoped_whole_owner_bindings():
            raise RuntimeError(
                "GefenMuon whole-parameter owner stepping requires the separate "
                "explicit process-group codebook scope"
            )
        loss = None
        hybrid_preflight_complete = bool(
            getattr(self, "_gefen_hybrid_precollective_preflight", False)
        )
        if not hybrid_preflight_complete and (
            self._gefen_codebook_process_group is not None
        ):
            # The explicit convention binding is the exclusive control scope:
            # synchronize on it before any operation header or mesh collective,
            # never in addition to the mainline DTensor-derived scope.
            try:
                self._assert_capturable_if_capturing()
                self._assert_codebook_capture_ready()
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()
                local_preamble_error = None
            except Exception as exc:
                loss = None
                local_preamble_error = exc
            self._synchronize_codebook_scope_failure(
                local_preamble_error, "step preamble"
            )

            self._assert_finalized_binding_layout()
            self._assert_runtime_codebook_process_group()
            try:
                _assert_optimizer_gradients_structurally_valid(
                    self, require_2d_params=True
                )
                local_preflight_error = None
            except Exception as exc:
                local_preflight_error = exc
            self._validate_codebook_scope_operation_header("step")
            self._synchronize_codebook_scope_failure(
                local_preflight_error, "gradient preflight"
            )
            self._ensure_codebook_scope_agreement()

            if not self._prepare_scoped_amp_optimizer_step():
                return loss
        elif not hybrid_preflight_complete:
            process_groups = self._step_failure_process_groups()
            try:
                self._assert_capturable_if_capturing()
                self._assert_codebook_capture_ready()
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()
                _assert_optimizer_gradients_structurally_valid(
                    self, require_2d_params=True
                )
                local_preflight_error = None
            except Exception as exc:
                loss = None
                local_preflight_error = exc
            self._synchronize_sharded_step_error(
                local_preflight_error, "step preflight", process_groups
            )

            # Every mesh member enters the AMP control agreement, including a
            # member with no local GradScaler attributes. This prevents the
            # protocol-presence decision itself from becoming rank-divergent.
            if not self._prepare_synchronized_amp_step(self, process_groups):
                return loss

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

        # This must precede _maybe_refresh_gefen_codebook(): the first-step
        # Muon codebook iterator itself calls full_tensor() in exact/distributed
        # modes, before either update dispatcher gets a chance to validate the
        # active set. The preflight is mutation-free and gives every mesh rank
        # the same clear error instead of leaving active ranks in an unmatched
        # collective.
        self._assert_sharded_grad_presence_consistent()

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

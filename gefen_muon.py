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
        if ns_steps >= 100:
            raise ValueError(
                "Number of steps must be less than 100 for computational efficiency"
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


def _adjust_lr(
    lr: float, adjust_lr_fn: Optional[str], param_shape: torch.Size
) -> float:

    if isinstance(lr, torch.Tensor):
        lr = lr.item()
    rows, cols = param_shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        adjusted_ratio = math.sqrt(max(1, rows / cols))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(rows, cols))
    else:
        adjusted_ratio = 1.0
    return lr * adjusted_ratio


class GefenMuon(Gefen):

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
        verbose: bool = False,
    ) -> None:
        if isinstance(lr, torch.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError("Learning rate should be >= 0 but is: {}".format(lr))
        if not 0.0 <= momentum:
            raise ValueError("momentum should be >= 0 but is: {}".format(momentum))
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
        # "exact" (default): under FSDP2 every rank gathers the full gradient and
        # runs Newton-Schulz on the full matrix -> bit-for-bit single-GPU parity.
        # "approx": each rank runs the whole pipeline on its LOCAL shard only --
        # no all-gather and NS on a smaller (row-sharded) matrix, so it is
        # cheaper, but Newton-Schulz of a row block is NOT the orthogonalization
        # of the full matrix: this mode is explicitly NON-PARITY. Opt-in only.
        # "distributed" ("Parallel Muon" / Moonshot): EXACT like "exact", but the
        # redundant Newton-Schulz is removed. Each 2D matrix is round-robin
        # assigned to a single owner rank; only that rank runs the quantized
        # momentum + Newton-Schulz on the full matrix, then broadcasts the
        # orthogonalized full-matrix update so every rank slices its own shard.
        # The NS/momentum compute (and the persistent momentum state) is therefore
        # cut ~world_size x while staying bit-for-bit identical to "exact". Only
        # the per-step gradient all-gather (a collective every rank must join) and
        # one extra update broadcast are replicated. Falls back to the "exact"
        # full-NS-everywhere path for non-1D meshes (e.g. HSDP x TP).
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

        super().__init__(
            params,
            lr=lr,
            betas=(momentum, 0.0),
            eps=eps,
            weight_decay=weight_decay,
            fused=fused,
            verbose=verbose,
        )

        for group in self.param_groups:
            group["momentum"] = momentum
            group["nesterov"] = nesterov
            group["ns_coefficients"] = ns_coefficients
            group["ns_steps"] = ns_steps
            group["adjust_lr_fn"] = adjust_lr_fn
            group["sharded_mode"] = sharded_mode
            group["fp8_ns"] = fp8_ns
            group["fp8_ns_compile"] = fp8_ns_compile
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        "GefenMuon only supports 2D parameters whereas we found a parameter with size: {}".format(
                            p.size(),
                        )
                    )

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
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                # approx mode learns the codebook/period from the LOCAL shard
                # (no all-gather) so periods divide the local numel that the
                # approximate step operates on; exact mode gathers the full matrix.
                if self._sharded_mode == "approx" and hasattr(grad, "to_local"):
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

        codebook = self._gefen_codebook
        if codebook is None:
            raise ValueError(
                "Expected Gefen codebook to be initialized before reconstructing quantized momentum."
            )
        if codebook.device != indices.device:
            raise ValueError(
                "Gefen codebook device {} does not match index device {}.".format(
                    codebook.device,
                    indices.device,
                )
            )

        momentum_view.copy_(codebook[indices.long()].to(dtype=momentum_view.dtype))
        momentum_view.mul_(state["m_magnitude"])

    def _fused_quantized_momentum_update(
        self,
        state,
        grad_view: torch.Tensor,
        momentum: float,
    ) -> torch.Tensor:
        # Single-pass Muon momentum update: the kernel advances the quantized
        # momentum state and emits the dense quantized momentum for Newton-Schulz
        # directly, so the old lr==0 dummy-stepsize call into the generic update
        # kernel followed by a second full-size codebook gather is gone. The
        # emitted momentum is bit-identical to the old
        # `dequantize(m_codebook) * m_magnitude`.
        return self._gefen_quantized_momentum_update(state, grad_view, momentum)

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
            state["step"] = 0
            self._init_gefen_muon_state(
                state, self._automatic_view(flat_grad, automatic_period)
            )

        automatic_period = state["automatic_period"]
        grad_view = self._automatic_view(flat_grad, automatic_period)

        if self._use_fused_gefen_automatic_step():
            # The Muon momentum kernel reads grad + quantized state and emits the
            # dense quantized momentum; it never touches p, so no full-matrix
            # scratch is needed under sharding.
            momentum_update = self._fused_quantized_momentum_update(
                state,
                grad_view,
                momentum,
            )
        else:
            momentum_update = self._gefen_dequantize_m_coefficients(state, grad_view)
            momentum_update.mul_(state["m_magnitude"])
            momentum_update.lerp_(grad_view, 1 - momentum)
            self._quantize_momentum_(state, momentum_update)

        state["step"] += 1

        if group["nesterov"]:
            momentum_update.mul_(momentum).add_(grad_view, alpha=1 - momentum)
        update = momentum_update.view_as(grad)

        return _zeropower_via_newtonschulz(
            update,
            group["ns_coefficients"],
            group["ns_steps"],
            group["eps"],
            use_fp8=group.get("fp8_ns", False),
            compile_fp8=group.get("fp8_ns_compile", True),
        )

    def _apply_muon_update(
        self,
        group,
        p: torch.Tensor,
        update: torch.Tensor,
        is_sharded: bool,
        approx: bool,
    ) -> None:
        lr = group["lr"]
        # p.shape is the GLOBAL shape for a DTensor, which is exactly what the
        # rows/cols LR ratio wants -- keep it unchanged under sharding.
        adjusted_lr = _adjust_lr(lr, group["adjust_lr_fn"], p.shape)

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

    def _step_distributed(self, items) -> None:
        # "distributed" / Parallel-Muon path. Each 2D matrix is round-robin
        # assigned to one owner rank that alone runs the quantized-momentum +
        # Newton-Schulz; the result is broadcast so every rank slices its shard.
        #
        # CRITICAL for the speed-up: the gather, compute and broadcast are
        # SEPARATED into phases per bucket of `world` matrices. If we broadcast
        # each update right after its owner computes it, the non-owner ranks block
        # at the broadcast while the owner runs NS -- the NS serializes and there
        # is no win. By all-gathering every grad in the bucket first, then letting
        # each rank compute its ONE owned matrix with no collective in between, the
        # owners' NS runs concurrently (NS critical path ~ NS_total / world), and
        # the per-bucket broadcasts are a pure communication phase.
        import torch.distributed as dist

        # Eligibility (1-D mesh, world>=2) is a property of each param's mesh and
        # so is identical on every rank -> the eligible/fallback split, and thus
        # the collective order, agrees globally. Non-eligible matrices (multi-dim
        # HSDP x TP meshes, world==1) keep the replicated exact full-NS path.
        eligible, fallback = [], []
        for (group, name, p, grad) in items:
            mesh = p.device_mesh
            if mesh.ndim == 1 and dist.get_world_size(mesh.get_group()) >= 2:
                eligible.append((group, name, p, grad))
            else:
                fallback.append((group, name, p, grad))

        for (group, name, p, grad) in fallback:
            self._step_automatic(group, name, p, grad)

        if not eligible:
            return

        # Group by process group (one mesh under plain FSDP2; multiple only under
        # exotic setups). Insertion-ordered so the order is identical across ranks.
        by_pg = OrderedDict()
        for item in eligible:
            pg = item[2].device_mesh.get_group()
            by_pg.setdefault(pg, []).append(item)

        for pg, pg_items in by_pg.items():
            self._step_distributed_pg(pg, pg_items)

    def _step_distributed_pg(self, pg, pg_items) -> None:
        import torch.distributed as dist

        world = dist.get_world_size(pg)
        my_coord = dist.get_group_rank(pg, dist.get_rank())
        global_rank = [dist.get_global_rank(pg, c) for c in range(world)]

        # Buckets of `world` consecutive matrices; within a bucket the matrix at
        # offset k is owned by rank k (so each bucket has exactly one matrix per
        # rank, except possibly a short tail bucket). idx = b + k, b % world == 0,
        # so owner == k -- the round-robin assignment, stable across steps.
        for b in range(0, len(pg_items), world):
            bucket = pg_items[b : b + world]

            # --- Phase 1: all-gather every grad in the bucket; keep only mine. ---
            # Every rank joins every full_tensor (matched collective). No compute
            # is interleaved, so ranks march through the gathers in lockstep.
            my_full_grad = None
            my_entry = None
            for k, (group, name, p, grad) in enumerate(bucket):
                fg = grad
                if hasattr(fg, "full_tensor"):
                    fg = fg.full_tensor()
                elif hasattr(fg, "to_local"):
                    fg = fg.to_local()
                if hasattr(fg, "wait"):
                    fg = fg.wait()
                if k == my_coord:
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
                    my_full_grad = fg
                    my_entry = (group, name, p)
                else:
                    fg = None  # drop the gathered full grad we do not own

            # --- Phase 2: compute MY owned update (no collectives -> parallel). ---
            my_update = None
            if my_entry is not None:
                group, name, p = my_entry
                my_update = self._compute_muon_update(
                    group, name, p, my_full_grad, p.numel()
                )
                my_full_grad = None

            # --- Phase 3: broadcast each update from its owner; apply locally. ---
            for k, (group, name, p, grad) in enumerate(bucket):
                if k == my_coord:
                    buf = my_update.contiguous()
                    if buf.dtype != torch.bfloat16:
                        buf = buf.to(torch.bfloat16)
                else:
                    buf = torch.empty(
                        tuple(p.shape),
                        dtype=torch.bfloat16,
                        device=p.to_local().device,
                    )
                dist.broadcast(buf, src=global_rank[k], group=pg)
                self._apply_muon_update(
                    group, p, buf, is_sharded=True, approx=False
                )
                buf = None

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
                "GefenMuon gradient must be a 2D matrix for {}".format(param_name)
            )

        # In approx mode the pipeline operates on the local shard, so the period
        # must divide the LOCAL numel; exact/non-sharded use the (global) numel.
        eff_numel = grad.reshape(-1).numel() if approx else p.numel()
        update = self._compute_muon_update(group, param_name, p, grad, eff_numel)
        self._apply_muon_update(group, p, update, is_sharded, approx)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._maybe_refresh_gefen_codebook()
        self._maybe_save_gefen_grad_histogram()

        # Partition the work. "distributed"-mode SHARDED matrices are handled by
        # the bucketed Parallel-Muon path (round-robin owner per matrix, in the
        # deterministic param-iteration order that agrees on every rank); every
        # other matrix (exact / approx / non-sharded) takes the per-param path.
        # The two passes run in the same order on every rank, so all collectives
        # stay matched.
        distributed_items = []
        for group in self.param_groups:
            name = group["name"]
            distributed = group["sharded_mode"] == "distributed"
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                if distributed and self._is_sharded(p):
                    distributed_items.append((group, name, p, grad))
                else:
                    self._step_automatic(group, name, p, grad)

        if distributed_items:
            self._step_distributed(distributed_items)

        self._gefen_global_step += 1
        return loss

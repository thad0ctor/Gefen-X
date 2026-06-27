import os
from typing import Optional

import torch

from ._build_notice import load_gefen_cuda_extension

_EXTENSION_MODULE = None

# Occupancy-driven dispatch between the two bit-identical update kernels.
# v1 (single serial pass, grid=num_blocks, threads=next_pow2(period)) is fastest
# only when the launch fills the machine: enough blocks to cover the SMs AND a
# full warp of work per block. v2 (two-phase, grid decoupled from num_blocks)
# wins at both extremes -- few blocks (large period) and tiny period (it uses
# flat full-warp grid-stride instead of a partial-warp per-block reduction).
#
# A microbench sweep of num_blocks from 2..numel on sm_86 (84 SM), sm_120
# (170 + 188 SM) showed a single SM-normalized rule fits all three archs
# (fused_bench/PERF.md): v1 only when period > 16 (full warp; the boundary is
# warpSize, arch-invariant) AND num_blocks >= 2 * SM_count. Both thresholds are
# bit-identical either way, so a misroute costs only speed, never correctness.
_V2_PERIOD_MAX = 16  # period <= 16 underfills the warp (threads = next_pow2)
_V2_BLOCKS_PER_SM = 2  # below 2 blocks/SM the single-pass kernel can't fill SMs
_GEFEN_UPDATE_V2_ENV = os.environ.get("GEFEN_UPDATE_V2")
_SM_COUNT_BY_DEVICE: dict = {}


def _load_extension():
    global _EXTENSION_MODULE
    if _EXTENSION_MODULE is not None:
        return _EXTENSION_MODULE

    _EXTENSION_MODULE = load_gefen_cuda_extension(
        "gefen_automatic_fused_ext",
        [
            "automatic_gefen_fused_binding.cpp",
            "automatic_gefen_fused_kernel.cu",
        ],
    )
    return _EXTENSION_MODULE


def _sm_count(device: torch.device) -> int:
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    cached = _SM_COUNT_BY_DEVICE.get(index)
    if cached is None:
        cached = torch.cuda.get_device_properties(index).multi_processor_count
        _SM_COUNT_BY_DEVICE[index] = cached
    return cached


def _should_use_v2(num_blocks: int, period: int, device: torch.device) -> bool:
    """Occupancy heuristic; env GEFEN_UPDATE_V2 hard-overrides it (1=v2, 0=v1)."""
    if _GEFEN_UPDATE_V2_ENV is not None:
        return _GEFEN_UPDATE_V2_ENV.lower() not in {"0", "false", "no", "off"}
    if period <= _V2_PERIOD_MAX:
        return True
    return num_blocks < _V2_BLOCKS_PER_SM * _sm_count(device)


def automatic_gefen_fused_update_cuda(
    p: torch.Tensor,
    grad_view: torch.Tensor,
    m_sign: torch.Tensor,
    m_magnitude: torch.Tensor,
    stepsize: torch.Tensor,
    codebook: torch.Tensor,
    packed_indices: bool,
    beta1: float,
    lr: float,
    *,
    use_v2: Optional[bool] = None,
) -> None:
    # Steady-state hot path: the dtype/shape/device/contiguity invariants are
    # enforced by the CUDA extension's C++ guards (which throw clear errors), so
    # the former full Python re-validation ran every step per param for no added
    # safety. Only normalize contiguity here, and only when actually needed --
    # an already-contiguous tensor skips the dispatch.
    if use_v2 is None:
        use_v2 = not packed_indices and _should_use_v2(
            grad_view.shape[0], grad_view.shape[1], grad_view.device
        )

    # Load-bearing guards for the raw-pointer launch: the C++ path reads grad_view
    # as p.scalar_type() and m_magnitude/stepsize/codebook as float*, all on p's
    # device. A dtype/device mismatch would reinterpret memory or read across
    # devices. (Shape/contiguity invariants are still enforced in C++.)
    if grad_view.dtype != p.dtype:
        raise ValueError(
            f"grad_view dtype {grad_view.dtype} must match p dtype {p.dtype}."
        )
    for name, t in (
        ("grad_view", grad_view),
        ("m_magnitude", m_magnitude),
        ("stepsize", stepsize),
        ("codebook", codebook),
    ):
        if t.device != p.device:
            raise ValueError(
                f"{name} device {t.device} must match p device {p.device}."
            )
    for name, t in (
        ("m_magnitude", m_magnitude),
        ("stepsize", stepsize),
        ("codebook", codebook),
    ):
        if t.dtype != torch.float32:
            raise ValueError(f"{name} must be float32, got {t.dtype}.")

    grad_c = grad_view if grad_view.is_contiguous() else grad_view.contiguous()
    step_c = stepsize if stepsize.is_contiguous() else stepsize.contiguous()
    cb_c = codebook if codebook.is_contiguous() else codebook.contiguous()

    module = _load_extension()
    if use_v2 and not packed_indices:
        module.automatic_gefen_fused_update_v2_cuda(
            p, grad_c, m_sign, m_magnitude, step_c, cb_c, packed_indices, beta1, lr
        )
        return
    module.automatic_gefen_fused_update_cuda(
        p, grad_c, m_sign, m_magnitude, step_c, cb_c, packed_indices, beta1, lr
    )

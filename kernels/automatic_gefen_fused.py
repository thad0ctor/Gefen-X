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
    tensors = {
        "p": p,
        "grad_view": grad_view,
        "m_sign": m_sign,
        "m_magnitude": m_magnitude,
        "stepsize": stepsize,
        "codebook": codebook,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Expected {} to be a torch.Tensor.".format(name))
        elif tensor.device.type != "cuda":
            raise ValueError(
                "Expected {} to be on CUDA, got device {}.".format(name, tensor.device)
            )

    if grad_view.dim() != 2:
        raise ValueError(
            "Expected grad_view to be 2D, got dim={}".format(grad_view.dim())
        )
    elif m_magnitude.dim() != 2 or m_magnitude.shape[1] != 1:
        raise ValueError(
            "Expected m_magnitude to have shape [num_blocks, 1], got {}".format(
                tuple(m_magnitude.shape)
            )
        )
    elif stepsize.dim() != 2 or stepsize.shape[1] != 1:
        raise ValueError(
            "Expected stepsize to have shape [num_blocks, 1], got {}".format(
                tuple(stepsize.shape)
            )
        )
    elif grad_view.shape[0] != m_magnitude.shape[0]:
        raise ValueError(
            "Expected grad_view and m_magnitude to have the same number of blocks."
        )
    elif grad_view.shape[0] != stepsize.shape[0]:
        raise ValueError(
            "Expected grad_view and stepsize to have the same number of blocks."
        )
    elif m_sign.dtype != torch.uint8:
        raise ValueError("Expected m_sign to be uint8, got {}".format(m_sign.dtype))
    elif codebook.dtype != torch.float32:
        raise ValueError(
            "Expected codebook to be float32, got {}".format(codebook.dtype)
        )
    elif codebook.numel() > 256:
        raise ValueError(
            "Expected codebook to have at most 256 entries, got {}".format(
                codebook.numel()
            )
        )
    elif packed_indices and m_sign.numel() != (grad_view.numel() + 1) // 2:
        raise ValueError(
            "Expected packed m_sign.numel()={} for grad_view.numel()={}, got {}.".format(
                (grad_view.numel() + 1) // 2,
                grad_view.numel(),
                m_sign.numel(),
            )
        )
    elif not packed_indices and m_sign.numel() != grad_view.numel():
        raise ValueError(
            "Expected unpacked m_sign.numel()={} for grad_view.numel()={}, got {}.".format(
                grad_view.numel(),
                grad_view.numel(),
                m_sign.numel(),
            )
        )

    # v2 has no packed-index path; for unpacked, decide per call from occupancy.
    if use_v2 is None:
        use_v2 = not packed_indices and _should_use_v2(
            grad_view.shape[0], grad_view.shape[1], grad_view.device
        )

    module = _load_extension()
    if use_v2 and not packed_indices:
        module.automatic_gefen_fused_update_v2_cuda(
            p,
            grad_view.contiguous(),
            m_sign,
            m_magnitude,
            stepsize.contiguous(),
            codebook.contiguous(),
            packed_indices,
            beta1,
            lr,
        )
        return
    module.automatic_gefen_fused_update_cuda(
        p,
        grad_view.contiguous(),
        m_sign,
        m_magnitude,
        stepsize.contiguous(),
        codebook.contiguous(),
        packed_indices,
        beta1,
        lr,
    )

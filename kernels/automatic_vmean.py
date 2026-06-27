import torch

from ._build_notice import load_gefen_cuda_extension

_EXTENSION_MODULE = None


def _load_extension():
    global _EXTENSION_MODULE
    if _EXTENSION_MODULE is not None:
        return _EXTENSION_MODULE

    _EXTENSION_MODULE = load_gefen_cuda_extension(
        "gefen_automatic_vmean_ext",
        [
            "automatic_vmean_binding.cpp",
            "automatic_vmean_kernel.cu",
        ],
    )
    return _EXTENSION_MODULE


def automatic_vmean_update_cuda(
    vmean: torch.Tensor,
    grad_view: torch.Tensor,
    beta2: float,
) -> None:
    # Steady-state hot path: dtype/shape/contiguity invariants are enforced by the
    # CUDA extension's C++ guards, so the former Python re-validation ran every
    # step per param for no added safety. Only normalize contiguity, and only when
    # the input is actually non-contiguous.
    # CUDAGuard only selects the launch device; a mismatched-device input would
    # still read across devices, so reject that here.
    if grad_view.device != vmean.device:
        raise ValueError(
            f"grad_view device {grad_view.device} must match vmean device "
            f"{vmean.device}."
        )
    grad_c = grad_view if grad_view.is_contiguous() else grad_view.contiguous()
    module = _load_extension()
    module.automatic_vmean_update_cuda(vmean, grad_c, beta2)

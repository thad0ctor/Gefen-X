#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

void automatic_gefen_fused_update_cuda(
    at::Tensor p,
    at::Tensor grad_view,
    at::Tensor m_sign,
    at::Tensor m_magnitude,
    at::Tensor stepsize,
    at::Tensor codebook,
    bool packed_indices,
    double beta1,
    double lr
);

void automatic_gefen_fused_update_v2_cuda(
    at::Tensor p,
    at::Tensor grad_view,
    at::Tensor m_sign,
    at::Tensor m_magnitude,
    at::Tensor stepsize,
    at::Tensor codebook,
    bool packed_indices,
    double beta1,
    double lr
);

void automatic_gefen_fused_full_update_cuda(
    at::Tensor p,
    at::Tensor grad_view,
    at::Tensor m_sign,
    at::Tensor m_magnitude,
    at::Tensor vmean,
    at::Tensor codebook,
    bool packed_indices,
    double beta1,
    double beta2,
    double lr,
    double eps,
    double inv_sqrt_bias_correction_2,
    double inv_bias_correction_1,
    double weight_decay_factor,
    bool stochastic_round,
    int64_t rng_seed
);

void automatic_gefen_fused_update_v2_full_cuda(
    at::Tensor p,
    at::Tensor grad_view,
    at::Tensor m_sign,
    at::Tensor m_magnitude,
    at::Tensor vmean,
    at::Tensor codebook,
    bool packed_indices,
    double beta1,
    double beta2,
    double lr,
    double eps,
    double inv_sqrt_bias_correction_2,
    double inv_bias_correction_1,
    double weight_decay_factor,
    bool stochastic_round,
    int64_t rng_seed
);

void gefen_quantized_momentum_update_cuda(
    at::Tensor grad_view,
    at::Tensor m_sign,
    at::Tensor m_magnitude,
    at::Tensor codebook,
    at::Tensor momentum_out,
    double beta1,
    bool stochastic_round,
    int64_t rng_seed
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "automatic_gefen_fused_update_cuda",
        &automatic_gefen_fused_update_cuda,
        "Fused automatic Gefen momentum/state/parameter update (CUDA)",
        py::arg("p"),
        py::arg("grad_view"),
        py::arg("m_sign"),
        py::arg("m_magnitude"),
        py::arg("stepsize"),
        py::arg("codebook"),
        py::arg("packed_indices"),
        py::arg("beta1"),
        py::arg("lr")
    );
    m.def(
        "automatic_gefen_fused_update_v2_cuda",
        &automatic_gefen_fused_update_v2_cuda,
        "Occupancy-flexible two-phase automatic Gefen update (CUDA)",
        py::arg("p"),
        py::arg("grad_view"),
        py::arg("m_sign"),
        py::arg("m_magnitude"),
        py::arg("stepsize"),
        py::arg("codebook"),
        py::arg("packed_indices"),
        py::arg("beta1"),
        py::arg("lr")
    );
    m.def(
        "automatic_gefen_fused_full_update_cuda",
        &automatic_gefen_fused_full_update_cuda,
        "Fully-fused automatic Gefen update: vmean EMA + in-kernel stepsize "
        "+ momentum/state/parameter update (CUDA)",
        py::arg("p"),
        py::arg("grad_view"),
        py::arg("m_sign"),
        py::arg("m_magnitude"),
        py::arg("vmean"),
        py::arg("codebook"),
        py::arg("packed_indices"),
        py::arg("beta1"),
        py::arg("beta2"),
        py::arg("lr"),
        py::arg("eps"),
        py::arg("inv_sqrt_bias_correction_2"),
        py::arg("inv_bias_correction_1"),
        py::arg("weight_decay_factor"),
        py::arg("stochastic_round") = false,
        py::arg("rng_seed") = 0
    );
    m.def(
        "automatic_gefen_fused_update_v2_full_cuda",
        &automatic_gefen_fused_update_v2_full_cuda,
        "Occupancy-flexible two-phase fully-fused Gefen update: vmean EMA + "
        "in-kernel stepsize + weight decay (CUDA)",
        py::arg("p"),
        py::arg("grad_view"),
        py::arg("m_sign"),
        py::arg("m_magnitude"),
        py::arg("vmean"),
        py::arg("codebook"),
        py::arg("packed_indices"),
        py::arg("beta1"),
        py::arg("beta2"),
        py::arg("lr"),
        py::arg("eps"),
        py::arg("inv_sqrt_bias_correction_2"),
        py::arg("inv_bias_correction_1"),
        py::arg("weight_decay_factor"),
        py::arg("stochastic_round") = false,
        py::arg("rng_seed") = 0
    );
    m.def(
        "gefen_quantized_momentum_update_cuda",
        &gefen_quantized_momentum_update_cuda,
        "Muon quantized-momentum update: update the quantized momentum state and "
        "emit the dense quantized momentum for Newton-Schulz in one pass (CUDA)",
        py::arg("grad_view"),
        py::arg("m_sign"),
        py::arg("m_magnitude"),
        py::arg("codebook"),
        py::arg("momentum_out"),
        py::arg("beta1"),
        py::arg("stochastic_round") = false,
        py::arg("rng_seed") = 0
    );
}

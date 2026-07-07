#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAMacros.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace {

constexpr int kMaxThreads = 256;
constexpr int kMaxHistogramBins = 4096;

template <typename scalar_t>
__global__ void gefen_exact_histogram_kernel(
    const scalar_t* __restrict__ grad_flat,
    int64_t period,
    int64_t num_blocks,
    int histogram_bins,
    int64_t* __restrict__ bin_counts
) {
    __shared__ float shared_absmax[kMaxThreads];
    // 64-bit shared counters: a single logical block can hold >= 2^31 elements
    // (a huge period), which would wrap a 32-bit int accumulator. unsigned long
    // long holds any count a block can produce; the flush to the int64 global
    // counts is exact (integer widening changes no result). At the max 4096 bins
    // this is 32 KiB of static shared -- under the 48 KiB budget with the
    // absmax buffer (~33 KiB total).
    __shared__ unsigned long long shared_counts[kMaxHistogramBins];

    const int64_t logical_block_idx = static_cast<int64_t>(blockIdx.x);
    if (logical_block_idx >= num_blocks) {
        return;
    }

    const int tid = static_cast<int>(threadIdx.x);
    const int64_t start = logical_block_idx * period;

    for (int idx = tid; idx < histogram_bins; idx += blockDim.x) {
        shared_counts[idx] = 0ULL;
    }

    float local_absmax = 0.0f;
    for (int64_t offset = tid; offset < period; offset += blockDim.x) {
        const float value = static_cast<float>(grad_flat[start + offset]);
        const float abs_value = fabsf(value);
        if (abs_value > local_absmax) {
            local_absmax = abs_value;
        }
    }

    shared_absmax[tid] = local_absmax;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < static_cast<int>(stride) && shared_absmax[tid + stride] > shared_absmax[tid]) {
            shared_absmax[tid] = shared_absmax[tid + stride];
        }
        __syncthreads();
    }

    const float absmax = shared_absmax[0];
    const float scale = 0.5f * static_cast<float>(histogram_bins);
    __syncthreads();

    for (int64_t offset = tid; offset < period; offset += blockDim.x) {
        const float grad_value = static_cast<float>(grad_flat[start + offset]);
        float normalized_value = 0.0f;
        if (absmax > 0.0f) {
            normalized_value = grad_value / absmax;
        }
        int bin_idx = static_cast<int>(floorf((normalized_value + 1.0f) * scale));
        if (bin_idx < 0) {
            bin_idx = 0;
        } else if (bin_idx >= histogram_bins) {
            bin_idx = histogram_bins - 1;
        }
        atomicAdd(&shared_counts[bin_idx], 1ULL);
    }
    __syncthreads();

    for (int idx = tid; idx < histogram_bins; idx += blockDim.x) {
        atomicAdd(
            reinterpret_cast<unsigned long long*>(&bin_counts[idx]),
            shared_counts[idx]
        );
    }
}

int choose_threads(int64_t period) {
    int threads = 32;
    while (threads < period && threads < kMaxThreads) {
        threads <<= 1;
    }
    if (threads > kMaxThreads) {
        threads = kMaxThreads;
    }
    return threads;
}

void validate_common_inputs(
    const at::Tensor& grad_flat,
    int64_t period,
    const at::Tensor& bin_counts
) {
    if (!grad_flat.is_cuda() || !bin_counts.is_cuda()) {
        throw std::invalid_argument("Expected grad_flat and bin_counts to be CUDA tensors.");
    }
    // The kernel accumulates into bin_counts on grad_flat's device; a
    // cross-device pair would launch on grad_flat's device but write across
    // devices. Reject it (CUDAGuard only selects the launch device).
    if (grad_flat.device() != bin_counts.device()) {
        throw std::invalid_argument("Expected grad_flat and bin_counts on the same device.");
    }
    if (!grad_flat.is_contiguous() || !bin_counts.is_contiguous()) {
        throw std::invalid_argument("Expected grad_flat and bin_counts to be contiguous.");
    }
    if (grad_flat.dim() != 1) {
        throw std::invalid_argument("Expected grad_flat to be 1D.");
    }
    if (bin_counts.dim() != 1) {
        throw std::invalid_argument("Expected bin_counts to be 1D.");
    }
    if (bin_counts.scalar_type() != at::kLong) {
        throw std::invalid_argument("Expected bin_counts to have dtype int64.");
    }
    if (bin_counts.numel() <= 0 || bin_counts.numel() > kMaxHistogramBins) {
        throw std::invalid_argument("Expected histogram size in [1, 4096].");
    }
    if (period <= 0) {
        throw std::invalid_argument("Expected period to be positive.");
    }
    if (grad_flat.numel() % period != 0) {
        throw std::invalid_argument("Expected grad_flat.numel() to be divisible by period.");
    }
}

}  // namespace

void gefen_exact_histogram_cuda(
    at::Tensor grad_flat,
    int64_t period,
    at::Tensor bin_counts
) {
    validate_common_inputs(grad_flat, period, bin_counts);

    // Empty grad (numel() == 0 -> num_blocks == 0): no elements to bin, and a
    // grid.x of 0 is an invalid launch. No-op before the launch (leaves
    // bin_counts untouched, which is the correct "added nothing" result).
    if (grad_flat.numel() == 0) {
        return;
    }

    c10::cuda::CUDAGuard device_guard(grad_flat.device());

    const int64_t num_blocks = grad_flat.numel() / period;
    const dim3 grid(static_cast<unsigned int>(num_blocks));
    const dim3 block(static_cast<unsigned int>(choose_threads(period)));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        grad_flat.scalar_type(),
        "gefen_exact_histogram_cuda",
        [&] {
            gefen_exact_histogram_kernel<scalar_t><<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
                grad_flat.data_ptr<scalar_t>(),
                period,
                num_blocks,
                static_cast<int>(bin_counts.numel()),
                bin_counts.data_ptr<int64_t>()
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

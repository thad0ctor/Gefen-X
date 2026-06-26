#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMacros.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace {

__device__ __forceinline__ uint8_t unpack_codebook_index(
    const uint8_t* __restrict__ packed_indices,
    int64_t logical_idx,
    bool packed
) {
    if (!packed) {
        return packed_indices[logical_idx];
    }
    const uint8_t packed_value = packed_indices[logical_idx >> 1];
    if ((logical_idx & 1) == 0) {
        return packed_value & 0x0F;
    }
    return (packed_value >> 4) & 0x0F;
}

__device__ __forceinline__ void store_packed_codebook_index(
    uint8_t* __restrict__ packed_indices,
    int64_t logical_idx,
    uint8_t quantized_index,
    bool packed
) {
    if (!packed) {
        packed_indices[logical_idx] = quantized_index;
        return;
    }

    const int64_t byte_idx = logical_idx >> 1;
    const int nibble_shift = (logical_idx & 1) == 0 ? 0 : 4;
    const uintptr_t raw_address = reinterpret_cast<uintptr_t>(packed_indices + byte_idx);
    const uintptr_t aligned_address = raw_address & ~static_cast<uintptr_t>(0x3);
    unsigned int* word_ptr = reinterpret_cast<unsigned int*>(aligned_address);
    const unsigned int byte_offset = static_cast<unsigned int>(raw_address - aligned_address);
    const unsigned int bit_shift = byte_offset * 8 + static_cast<unsigned int>(nibble_shift);
    const unsigned int nibble_mask = 0xFu << bit_shift;
    const unsigned int nibble_value = (static_cast<unsigned int>(quantized_index) & 0xFu) << bit_shift;

    unsigned int old_word = *word_ptr;
    unsigned int assumed_word = old_word;
    do {
        assumed_word = old_word;
        const unsigned int new_word = (assumed_word & ~nibble_mask) | nibble_value;
        old_word = atomicCAS(word_ptr, assumed_word, new_word);
    } while (old_word != assumed_word);
}

__device__ __forceinline__ uint8_t nearest_codebook_index(
    float normalized_value,
    const float* __restrict__ codebook,
    int codebook_size
) {
    int left = 0;
    int right = codebook_size;
    while (left < right) {
        const int mid = left + (right - left) / 2;
        if (codebook[mid] < normalized_value) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    int right_idx = left;
    if (right_idx < 0) {
        right_idx = 0;
    } else if (right_idx >= codebook_size) {
        right_idx = codebook_size - 1;
    }

    int left_idx = right_idx - 1;
    if (left_idx < 0) {
        left_idx = 0;
    }

    const float left_dist = fabsf(normalized_value - codebook[left_idx]);
    const float right_dist = fabsf(normalized_value - codebook[right_idx]);
    return static_cast<uint8_t>(left_dist <= right_dist ? left_idx : right_idx);
}

template <typename scalar_t>
__global__ void automatic_gefen_fused_update_kernel(
    scalar_t* __restrict__ p,
    const scalar_t* __restrict__ grad_view,
    uint8_t* __restrict__ m_sign,
    float* __restrict__ m_magnitude,
    const float* __restrict__ stepsize,
    const float* __restrict__ codebook,
    int codebook_size,
    bool packed_indices,
    int64_t period,
    int64_t num_blocks,
    float beta1,
    float lr
) {
    extern __shared__ float shared_max[];

    const int64_t block_idx = static_cast<int64_t>(blockIdx.x);
    if (block_idx >= num_blocks) {
        return;
    }

    const int64_t start = block_idx * period;
    const float old_magnitude = m_magnitude[block_idx];
    const float step = stepsize[block_idx];
    float local_absmax = 0.0f;

    for (int64_t offset = threadIdx.x; offset < period; offset += blockDim.x) {
        const int64_t idx = start + offset;
        const float coeff = codebook[static_cast<int>(unpack_codebook_index(m_sign, idx, packed_indices))];
        const float current_m = old_magnitude * coeff;
        const float grad_value = static_cast<float>(grad_view[idx]);
        const float updated_value = beta1 * current_m + (1.0f - beta1) * grad_value;
        const float abs_value = fabsf(updated_value);
        if (abs_value > local_absmax) {
            local_absmax = abs_value;
        }
    }

    shared_max[threadIdx.x] = local_absmax;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (shared_max[threadIdx.x + stride] > shared_max[threadIdx.x]) {
                shared_max[threadIdx.x] = shared_max[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    const float new_magnitude = shared_max[0];
    if (threadIdx.x == 0) {
        m_magnitude[block_idx] = new_magnitude;
    }
    __syncthreads();

    for (int64_t offset = threadIdx.x; offset < period; offset += blockDim.x) {
        const int64_t idx = start + offset;
        float normalized_value = 0.0f;
        const float coeff = codebook[static_cast<int>(unpack_codebook_index(m_sign, idx, packed_indices))];
        const float current_m = old_magnitude * coeff;
        const float grad_value = static_cast<float>(grad_view[idx]);
        const float updated_value = beta1 * current_m + (1.0f - beta1) * grad_value;
        if (new_magnitude > 0.0f) {
            normalized_value = updated_value / new_magnitude;
        }
        const uint8_t quantized_index = nearest_codebook_index(normalized_value, codebook, codebook_size);
        store_packed_codebook_index(m_sign, idx, quantized_index, packed_indices);
        if (lr != 0.0f) {
            const float quantized_value = codebook[static_cast<int>(quantized_index)] * new_magnitude;
            const float update_value = quantized_value * step * lr;
            p[idx] = static_cast<scalar_t>(static_cast<float>(p[idx]) - update_value);
        }
    }
}

int choose_threads(int64_t period) {
    int threads = 32;
    while (threads < period && threads < 256) {
        threads <<= 1;
    }
    if (threads > 256) {
        threads = 256;
    }
    return threads;
}

// ---------------------------------------------------------------------------
// v2: occupancy-flexible two-phase update.
//
// The single-kernel v1 launches grid=num_blocks, so its parallelism collapses
// when the partition period is large (few blocks, huge per-block serial loop)
// or when period==1 (millions of 32-thread blocks reducing one element). v2
// decouples the CUDA grid from num_blocks:
//   phase 1 computes the new per-block magnitude (absmax of the updated
//           momentum) with a grid sized to the work, not to num_blocks;
//   phase 2 quantizes + writes p elementwise over a flat grid.
// absmax is order-independent, so the magnitude is bit-identical to v1, and
// phase 2 recomputes the updated momentum with the identical float ops, so p
// and the stored indices are bit-identical to v1.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void atomic_max_nonneg(float* addr, float val) {
    // Valid only for val >= 0 and *addr >= 0 (positive-float bit patterns are
    // monotonic in their unsigned-int reinterpretation). magnitude is an
    // absolute value initialised to 0, so both hold.
    unsigned int* uaddr = reinterpret_cast<unsigned int*>(addr);
    unsigned int old = *uaddr;
    unsigned int assumed;
    do {
        assumed = old;
        if (__uint_as_float(assumed) >= val) {
            break;
        }
        old = atomicCAS(uaddr, assumed, __float_as_uint(val));
    } while (old != assumed);
}

template <typename scalar_t>
__device__ __forceinline__ float updated_momentum(
    const scalar_t* __restrict__ grad_view,
    const uint8_t* __restrict__ m_sign,
    const float* __restrict__ codebook,
    float old_magnitude,
    int64_t idx,
    float beta1
) {
    const float coeff = codebook[static_cast<int>(m_sign[idx])];
    const float current_m = old_magnitude * coeff;
    const float grad_value = static_cast<float>(grad_view[idx]);
    return beta1 * current_m + (1.0f - beta1) * grad_value;
}

// Small/medium period: flat grid-stride, one atomicMax per element. Contention
// per block is O(period); fine while period stays modest.
template <typename scalar_t>
__global__ void gefen_magnitude_flat_kernel(
    const scalar_t* __restrict__ grad_view,
    const uint8_t* __restrict__ m_sign,
    const float* __restrict__ old_magnitude,
    const float* __restrict__ codebook,
    float* __restrict__ new_magnitude,
    int64_t period,
    int64_t total_numel,
    float beta1
) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total_numel; idx += stride) {
        const int64_t block_idx = idx / period;
        const float updated = updated_momentum(
            grad_view, m_sign, codebook, old_magnitude[block_idx], idx, beta1);
        atomic_max_nonneg(&new_magnitude[block_idx], fabsf(updated));
    }
}

// Large period: split each partition block across blocks_per_row CUDA blocks,
// block-local reduction then one atomicMax per CUDA block (contention O(blocks
// _per_row) per row, not O(period)).
template <typename scalar_t>
__global__ void gefen_magnitude_split_kernel(
    const scalar_t* __restrict__ grad_view,
    const uint8_t* __restrict__ m_sign,
    const float* __restrict__ old_magnitude,
    const float* __restrict__ codebook,
    float* __restrict__ new_magnitude,
    int64_t period,
    int64_t num_blocks,
    int blocks_per_row,
    float beta1
) {
    extern __shared__ float shared_max[];
    const int64_t row = static_cast<int64_t>(blockIdx.x) / blocks_per_row;
    const int sub = static_cast<int>(static_cast<int64_t>(blockIdx.x) % blocks_per_row);
    if (row >= num_blocks) {
        return;
    }
    const float old_mag = old_magnitude[row];
    const int64_t row_start = row * period;
    float local_absmax = 0.0f;
    for (int64_t offset = static_cast<int64_t>(sub) * blockDim.x + threadIdx.x;
         offset < period; offset += static_cast<int64_t>(blocks_per_row) * blockDim.x) {
        const float updated = updated_momentum(
            grad_view, m_sign, codebook, old_mag, row_start + offset, beta1);
        const float a = fabsf(updated);
        if (a > local_absmax) {
            local_absmax = a;
        }
    }
    shared_max[threadIdx.x] = local_absmax;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (shared_max[threadIdx.x + s] > shared_max[threadIdx.x]) {
                shared_max[threadIdx.x] = shared_max[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomic_max_nonneg(&new_magnitude[row], shared_max[0]);
    }
}

template <typename scalar_t>
__global__ void gefen_update_flat_kernel(
    scalar_t* __restrict__ p,
    const scalar_t* __restrict__ grad_view,
    uint8_t* __restrict__ m_sign,
    const float* __restrict__ old_magnitude,
    const float* __restrict__ new_magnitude,
    const float* __restrict__ stepsize,
    const float* __restrict__ codebook,
    int codebook_size,
    int64_t period,
    int64_t total_numel,
    float beta1,
    float lr
) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total_numel; idx += stride) {
        const int64_t block_idx = idx / period;
        const float new_mag = new_magnitude[block_idx];
        const float updated = updated_momentum(
            grad_view, m_sign, codebook, old_magnitude[block_idx], idx, beta1);
        float normalized_value = 0.0f;
        if (new_mag > 0.0f) {
            normalized_value = updated / new_mag;
        }
        const uint8_t quantized_index =
            nearest_codebook_index(normalized_value, codebook, codebook_size);
        m_sign[idx] = quantized_index;
        if (lr != 0.0f) {
            const float quantized_value = codebook[static_cast<int>(quantized_index)] * new_mag;
            const float update_value = quantized_value * stepsize[block_idx] * lr;
            p[idx] = static_cast<scalar_t>(static_cast<float>(p[idx]) - update_value);
        }
    }
}

}  // namespace

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
) {
    if (!p.is_cuda() || !grad_view.is_cuda() || !m_sign.is_cuda() || !m_magnitude.is_cuda() || !stepsize.is_cuda() || !codebook.is_cuda()) {
        throw std::invalid_argument("Expected all tensors to be on CUDA.");
    }
    if (!p.is_contiguous()) {
        throw std::invalid_argument("Expected p to be contiguous.");
    }
    if (!grad_view.is_contiguous()) {
        throw std::invalid_argument("Expected grad_view to be contiguous.");
    }
    if (!m_sign.is_contiguous()) {
        throw std::invalid_argument("Expected m_sign to be contiguous.");
    }
    if (!m_magnitude.is_contiguous()) {
        throw std::invalid_argument("Expected m_magnitude to be contiguous.");
    }
    if (!stepsize.is_contiguous()) {
        throw std::invalid_argument("Expected stepsize to be contiguous.");
    }
    if (!codebook.is_contiguous()) {
        throw std::invalid_argument("Expected codebook to be contiguous.");
    }
    if (grad_view.dim() != 2) {
        throw std::invalid_argument("Expected grad_view to be 2D.");
    }
    if (m_magnitude.dim() != 2 || m_magnitude.size(1) != 1) {
        throw std::invalid_argument("Expected m_magnitude to have shape [num_blocks, 1].");
    }
    if (stepsize.dim() != 2 || stepsize.size(1) != 1) {
        throw std::invalid_argument("Expected stepsize to have shape [num_blocks, 1].");
    }
    if (m_sign.scalar_type() != at::kByte) {
        throw std::invalid_argument("Expected m_sign to have dtype uint8.");
    }
    if (codebook.scalar_type() != at::kFloat) {
        throw std::invalid_argument("Expected codebook to have dtype float32.");
    }

    c10::cuda::CUDAGuard device_guard(p.device());

    const int64_t num_blocks = grad_view.size(0);
    const int64_t period = grad_view.size(1);
    const int64_t total_numel = num_blocks * period;
    if (p.numel() != total_numel) {
        throw std::invalid_argument("Expected p.numel() to match grad_view.numel().");
    }
    if (packed_indices) {
        if (m_sign.numel() != (total_numel + 1) / 2) {
            throw std::invalid_argument("Expected packed m_sign.numel() to be ceil(total_numel / 2).");
        }
    } else if (!packed_indices && m_sign.numel() != total_numel) {
        throw std::invalid_argument("Expected unpacked m_sign.numel() to match grad_view.numel().");
    }
    if (m_magnitude.size(0) != num_blocks || stepsize.size(0) != num_blocks) {
        throw std::invalid_argument("Expected m_magnitude and stepsize to match the number of blocks.");
    }
    // these two may create additional memory footprint.
    const int threads = choose_threads(period);
    const dim3 grid(static_cast<unsigned int>(num_blocks));
    const dim3 block(static_cast<unsigned int>(threads));
    const size_t shared_bytes = static_cast<size_t>(threads) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        p.scalar_type(),
        "automatic_gefen_fused_update_cuda",
        [&] {
            automatic_gefen_fused_update_kernel<scalar_t><<<grid, block, shared_bytes>>>(
                p.data_ptr<scalar_t>(),
                grad_view.data_ptr<scalar_t>(),
                m_sign.data_ptr<uint8_t>(),
                m_magnitude.data_ptr<float>(),
                stepsize.data_ptr<float>(),
                codebook.data_ptr<float>(),
                static_cast<int>(codebook.numel()),
                packed_indices,
                period,
                num_blocks,
                static_cast<float>(beta1),
                static_cast<float>(lr)
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
) {
    if (packed_indices) {
        throw std::invalid_argument("v2 fused update does not support packed indices.");
    }
    if (!p.is_cuda() || !grad_view.is_cuda() || !m_sign.is_cuda() || !m_magnitude.is_cuda() || !stepsize.is_cuda() || !codebook.is_cuda()) {
        throw std::invalid_argument("Expected all tensors to be on CUDA.");
    }
    if (!p.is_contiguous() || !grad_view.is_contiguous() || !m_sign.is_contiguous() ||
        !m_magnitude.is_contiguous() || !stepsize.is_contiguous() || !codebook.is_contiguous()) {
        throw std::invalid_argument("Expected all tensors to be contiguous.");
    }
    if (grad_view.dim() != 2) {
        throw std::invalid_argument("Expected grad_view to be 2D.");
    }
    if (m_sign.scalar_type() != at::kByte) {
        throw std::invalid_argument("Expected m_sign to have dtype uint8.");
    }
    if (codebook.scalar_type() != at::kFloat) {
        throw std::invalid_argument("Expected codebook to have dtype float32.");
    }

    c10::cuda::CUDAGuard device_guard(p.device());

    const int64_t num_blocks = grad_view.size(0);
    const int64_t period = grad_view.size(1);
    const int64_t total_numel = num_blocks * period;
    if (p.numel() != total_numel || m_sign.numel() != total_numel) {
        throw std::invalid_argument("Expected p and m_sign numel to match grad_view.");
    }
    if (m_magnitude.size(0) != num_blocks || stepsize.size(0) != num_blocks) {
        throw std::invalid_argument("Expected m_magnitude and stepsize to match num_blocks.");
    }

    auto new_magnitude = at::zeros_like(m_magnitude);

    const int codebook_size = static_cast<int>(codebook.numel());
    const int threads = 256;
    int device_id = 0;
    cudaGetDevice(&device_id);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);
    if (sm_count < 1) sm_count = 1;
    // Target a few thousand resident blocks for the flat phases.
    const int64_t flat_blocks_cap = static_cast<int64_t>(sm_count) * 32;

    // Phase 1: per-block magnitude (absmax of updated momentum).
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, p.scalar_type(),
        "gefen_v2_magnitude", [&] {
            if (period <= 2048) {
                int64_t nblocks = (total_numel + threads - 1) / threads;
                if (nblocks > flat_blocks_cap) nblocks = flat_blocks_cap;
                if (nblocks < 1) nblocks = 1;
                gefen_magnitude_flat_kernel<scalar_t><<<static_cast<unsigned int>(nblocks), threads>>>(
                    grad_view.data_ptr<scalar_t>(), m_sign.data_ptr<uint8_t>(),
                    m_magnitude.data_ptr<float>(), codebook.data_ptr<float>(),
                    new_magnitude.data_ptr<float>(), period, total_numel,
                    static_cast<float>(beta1));
            } else {
                int blocks_per_row = static_cast<int>((period + threads * 64 - 1) / (threads * 64));
                if (blocks_per_row < 1) blocks_per_row = 1;
                // Keep total grid bounded but well above SM count.
                const int max_bpr = static_cast<int>((flat_blocks_cap + num_blocks - 1) / num_blocks);
                if (blocks_per_row > max_bpr && max_bpr >= 1) blocks_per_row = max_bpr;
                if (blocks_per_row < 1) blocks_per_row = 1;
                const int64_t grid = num_blocks * blocks_per_row;
                const size_t shmem = static_cast<size_t>(threads) * sizeof(float);
                gefen_magnitude_split_kernel<scalar_t><<<static_cast<unsigned int>(grid), threads, shmem>>>(
                    grad_view.data_ptr<scalar_t>(), m_sign.data_ptr<uint8_t>(),
                    m_magnitude.data_ptr<float>(), codebook.data_ptr<float>(),
                    new_magnitude.data_ptr<float>(), period, num_blocks, blocks_per_row,
                    static_cast<float>(beta1));
            }
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Phase 2: quantize + parameter update (elementwise).
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, p.scalar_type(),
        "gefen_v2_update", [&] {
            int64_t nblocks = (total_numel + threads - 1) / threads;
            if (nblocks > flat_blocks_cap) nblocks = flat_blocks_cap;
            if (nblocks < 1) nblocks = 1;
            gefen_update_flat_kernel<scalar_t><<<static_cast<unsigned int>(nblocks), threads>>>(
                p.data_ptr<scalar_t>(), grad_view.data_ptr<scalar_t>(), m_sign.data_ptr<uint8_t>(),
                m_magnitude.data_ptr<float>(), new_magnitude.data_ptr<float>(),
                stepsize.data_ptr<float>(), codebook.data_ptr<float>(), codebook_size,
                period, total_numel, static_cast<float>(beta1), static_cast<float>(lr));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    m_magnitude.copy_(new_magnitude);
}

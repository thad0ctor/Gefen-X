#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMacros.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <stdexcept>

// Param-dtype dispatch over Float/Half/BFloat16 *without* Double: every Gefen
// update kernel does its arithmetic in float and writes static_cast<scalar_t>,
// so a double param would be silently computed at single precision. Reject it at
// dispatch (clear "not implemented for 'Double'") instead of downcasting.
#define GEFEN_DISPATCH_FLOAT_HALF_BF16(TYPE, NAME, ...)        \
    AT_DISPATCH_SWITCH(                                        \
        TYPE, NAME,                                            \
        AT_DISPATCH_CASE(at::kFloat, __VA_ARGS__)              \
        AT_DISPATCH_CASE(at::kHalf, __VA_ARGS__)               \
        AT_DISPATCH_CASE(at::kBFloat16, __VA_ARGS__))

namespace {

// Cache the per-device SM count: cudaDeviceGetAttribute is a driver round-trip
// that the v2 launch path otherwise pays on every step per param, and the value
// is constant for the life of the process. The cache is an atomic array (0 ==
// unfilled, real SM counts are >= 1) so concurrent fills are a benign race on
// the same constant rather than UB on a plain shared array.
int cached_sm_count(int device_id) {
    constexpr int kMaxDevices = 64;
    static std::atomic<int> cache[kMaxDevices];  // static storage zero-inits
    if (device_id >= 0 && device_id < kMaxDevices) {
        const int cached = cache[device_id].load(std::memory_order_relaxed);
        if (cached > 0) {
            return cached;
        }
    }
    int sm = 0;
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, device_id);
    if (sm < 1) {
        sm = 1;
    }
    if (device_id >= 0 && device_id < kMaxDevices) {
        cache[device_id].store(sm, std::memory_order_relaxed);
    }
    return sm;
}

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
    // Shared layout: [codebook_size floats codebook] then [blockDim.x floats max].
    // Staging the (<=256-entry) codebook once per block keeps the per-element
    // coeff gather and the binary search in the quantize loop off global memory.
    extern __shared__ float smem[];
    float* s_codebook = smem;
    float* shared_max = smem + codebook_size;
    for (int i = threadIdx.x; i < codebook_size; i += blockDim.x) {
        s_codebook[i] = codebook[i];
    }
    __syncthreads();

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
        const float coeff = s_codebook[static_cast<int>(unpack_codebook_index(m_sign, idx, packed_indices))];
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
        const float coeff = s_codebook[static_cast<int>(unpack_codebook_index(m_sign, idx, packed_indices))];
        const float current_m = old_magnitude * coeff;
        const float grad_value = static_cast<float>(grad_view[idx]);
        const float updated_value = beta1 * current_m + (1.0f - beta1) * grad_value;
        if (new_magnitude > 0.0f) {
            normalized_value = updated_value / new_magnitude;
        }
        const uint8_t quantized_index = nearest_codebook_index(normalized_value, s_codebook, codebook_size);
        store_packed_codebook_index(m_sign, idx, quantized_index, packed_indices);
        if (lr != 0.0f) {
            const float quantized_value = s_codebook[static_cast<int>(quantized_index)] * new_magnitude;
            // Explicit round-to-nearest mul/sub: the parameter write does not
            // depend on nvcc's per-kernel FMA-contraction choice for
            // `p - q*step*lr`. v1, v2 and the fully-fused kernel all use this exact
            // form so their writes are bit-identical to each other and reproducible
            // run-to-run -- separately-compiled kernels otherwise contract `p -
            // q*step*lr` differently and disagree on fp32 params at rounding
            // boundaries.
            const float update_value = __fmul_rn(__fmul_rn(quantized_value, step), lr);
            p[idx] = static_cast<scalar_t>(__fsub_rn(static_cast<float>(p[idx]), update_value));
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
// Fully-fused v1 update (Tier-1 K1+K2): a single kernel that, in the same
// pass-1 grad read the absmax already needs, also accumulates Sum(grad^2) and
// emits the second-moment EMA (vmean), then computes the per-block stepsize
// in-kernel from the just-updated vmean. This collapses the separate vmean
// kernel + one full grad pass + the host sqrt/div/reciprocal/mul launches into
// this one kernel.
//
// Bit-exactness contract:
//   * absmax reduction + quantize/write are byte-for-byte the v1 single-kernel
//     path (identical threads = choose_threads(period), grid = num_blocks,
//     identical float ops), so p / m_sign / m_magnitude match v1.
//   * the Sum(grad^2) reduction reuses the exact thread/stride geometry and the
//     halving sum tree of automatic_vmean_update_kernel, so vmean is bit-
//     identical to the standalone vmean kernel.
//   * the in-kernel stepsize mirrors gefen.py's host math op-for-op:
//       h = sqrt(vmean); h = h / sqrt(bc2); h = h + eps;
//       stepsize = reciprocal(h) * (1/bc1)   [reciprocal*scalar, NOT div]
//     with sqrt(bc2) and 1/bc1 precomputed host-side (Python double) then cast
//     to float exactly as PyTorch casts the scalar operands of div_/mul_.
// Only routed for period > warpSize with full SM occupancy (the v1 regime);
// tiny-period / few-block shapes stay on the v2 decomposed path.
template <typename scalar_t>
__global__ void automatic_gefen_fused_full_update_kernel(
    scalar_t* __restrict__ p,
    const scalar_t* __restrict__ grad_view,
    uint8_t* __restrict__ m_sign,
    float* __restrict__ m_magnitude,
    float* __restrict__ vmean,
    const float* __restrict__ codebook,
    int codebook_size,
    bool packed_indices,
    int64_t period,
    int64_t num_blocks,
    float beta1,
    float beta2,
    float lr,
    float eps,
    float inv_sqrt_bias_correction_2,
    float inv_bias_correction_1,
    float weight_decay_factor
) {
    // Shared layout: [codebook_size codebook][blockDim.x max][blockDim.x sumsq].
    extern __shared__ float smem[];
    float* s_codebook = smem;
    float* shared_max = smem + codebook_size;
    float* shared_sum = shared_max + blockDim.x;
    for (int i = threadIdx.x; i < codebook_size; i += blockDim.x) {
        s_codebook[i] = codebook[i];
    }
    __syncthreads();

    const int64_t block_idx = static_cast<int64_t>(blockIdx.x);
    if (block_idx >= num_blocks) {
        return;
    }

    const int64_t start = block_idx * period;
    const float old_magnitude = m_magnitude[block_idx];
    float local_absmax = 0.0f;
    float local_sumsq = 0.0f;

    for (int64_t offset = threadIdx.x; offset < period; offset += blockDim.x) {
        const int64_t idx = start + offset;
        const float coeff = s_codebook[static_cast<int>(unpack_codebook_index(m_sign, idx, packed_indices))];
        const float current_m = old_magnitude * coeff;
        const float grad_value = static_cast<float>(grad_view[idx]);
        const float updated_value = beta1 * current_m + (1.0f - beta1) * grad_value;
        const float abs_value = fabsf(updated_value);
        if (abs_value > local_absmax) {
            local_absmax = abs_value;
        }
        // Same per-thread accumulation order as automatic_vmean_update_kernel.
        local_sumsq += grad_value * grad_value;
    }

    shared_max[threadIdx.x] = local_absmax;
    shared_sum[threadIdx.x] = local_sumsq;
    __syncthreads();

    // Fused max + sum tree: the sum half is bit-identical to the standalone
    // vmean kernel (same stride schedule, same `+=` order); the max half is
    // bit-identical to v1.
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (shared_max[threadIdx.x + stride] > shared_max[threadIdx.x]) {
                shared_max[threadIdx.x] = shared_max[threadIdx.x + stride];
            }
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float new_magnitude = shared_max[0];
    if (threadIdx.x == 0) {
        m_magnitude[block_idx] = new_magnitude;
        // vmean EMA, op-identical to automatic_vmean_update_kernel.
        const float mean_square = shared_sum[0] / static_cast<float>(period);
        const float previous_vmean = vmean[block_idx];
        const float updated_vmean = beta2 * previous_vmean + (1.0f - beta2) * mean_square;
        vmean[block_idx] = updated_vmean;
        // stepsize, op-identical to gefen.py host math. Each op is an explicit
        // round-to-nearest IEEE primitive so nvcc cannot contract or reassociate
        // away from PyTorch's five separately-rounded tensor ops
        // (sqrt -> div_ -> add_ -> reciprocal -> mul_). CRUCIAL: PyTorch lowers
        // `tensor.div_(scalar)` to a multiply by the float32 reciprocal of the
        // scalar (NOT a true divide -- they differ by up to ~2 ULP), so the host
        // passes 1/sqrt(bc2) precomputed and this multiplies by it.
        float h = __fsqrt_rn(updated_vmean);
        h = __fmul_rn(h, inv_sqrt_bias_correction_2);
        h = __fadd_rn(h, eps);
        // Broadcast the per-block stepsize through shared_sum[0] (the reduced
        // sum is already consumed above).
        shared_sum[0] = __fmul_rn(__frcp_rn(h), inv_bias_correction_1);
    }
    __syncthreads();
    const float stepsize_val = shared_sum[0];

    for (int64_t offset = threadIdx.x; offset < period; offset += blockDim.x) {
        const int64_t idx = start + offset;
        float normalized_value = 0.0f;
        const float coeff = s_codebook[static_cast<int>(unpack_codebook_index(m_sign, idx, packed_indices))];
        const float current_m = old_magnitude * coeff;
        const float grad_value = static_cast<float>(grad_view[idx]);
        const float updated_value = beta1 * current_m + (1.0f - beta1) * grad_value;
        if (new_magnitude > 0.0f) {
            normalized_value = updated_value / new_magnitude;
        }
        const uint8_t quantized_index = nearest_codebook_index(normalized_value, s_codebook, codebook_size);
        store_packed_codebook_index(m_sign, idx, quantized_index, packed_indices);
        if (lr != 0.0f) {
            const float quantized_value = s_codebook[static_cast<int>(quantized_index)] * new_magnitude;
            const float update_value = __fmul_rn(__fmul_rn(quantized_value, stepsize_val), lr);
            // K3: fold weight decay into the write. weight_decay_factor is a
            // kernel-wide scalar, so this branch is uniform (no warp divergence).
            // When it is 1.0f (wd == 0 -- the common case) take the exact same
            // write the K1+K2 path used, which is bit-identical to the v1 kernel;
            // restructuring it would change nvcc's FMA-contraction of `p - update`
            // and perturb fp32 params at rounding boundaries. Otherwise reproduce
            // the host p.mul_(1 - lr*wd) pass, which rounds back to scalar_t before
            // the update reads it (the inner scalar_t cast mirrors that round).
            if (weight_decay_factor == 1.0f) {
                p[idx] = static_cast<scalar_t>(__fsub_rn(static_cast<float>(p[idx]), update_value));
            } else {
                const float decayed = static_cast<float>(
                    static_cast<scalar_t>(__fmul_rn(static_cast<float>(p[idx]), weight_decay_factor)));
                p[idx] = static_cast<scalar_t>(__fsub_rn(decayed, update_value));
            }
        }
    }
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
    // Skip NaN and negatives: `!(val >= 0)` is true for NaN (all NaN compares
    // false) and for val < 0, so a NaN magnitude can never win the CAS and
    // corrupt the running max -- matching the v1 single-kernel path, which
    // leaves the max unchanged for NaN.
    if (!(val >= 0.0f)) {
        return;
    }
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
    int codebook_size,
    int64_t period,
    int64_t total_numel,
    float beta1
) {
    extern __shared__ float s_codebook[];
    for (int i = threadIdx.x; i < codebook_size; i += blockDim.x) {
        s_codebook[i] = codebook[i];
    }
    __syncthreads();
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total_numel; idx += stride) {
        const int64_t block_idx = idx / period;
        const float updated = updated_momentum(
            grad_view, m_sign, s_codebook, old_magnitude[block_idx], idx, beta1);
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
    int codebook_size,
    int64_t period,
    int64_t num_blocks,
    int blocks_per_row,
    float beta1
) {
    // Shared: [blockDim.x reduction max] then [codebook_size staged codebook].
    extern __shared__ float smem[];
    float* shared_max = smem;
    float* s_codebook = smem + blockDim.x;
    for (int i = threadIdx.x; i < codebook_size; i += blockDim.x) {
        s_codebook[i] = codebook[i];
    }
    __syncthreads();
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
            grad_view, m_sign, s_codebook, old_mag, row_start + offset, beta1);
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
    extern __shared__ float s_codebook[];
    for (int i = threadIdx.x; i < codebook_size; i += blockDim.x) {
        s_codebook[i] = codebook[i];
    }
    __syncthreads();
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total_numel; idx += stride) {
        const int64_t block_idx = idx / period;
        const float new_mag = new_magnitude[block_idx];
        const float updated = updated_momentum(
            grad_view, m_sign, s_codebook, old_magnitude[block_idx], idx, beta1);
        float normalized_value = 0.0f;
        if (new_mag > 0.0f) {
            normalized_value = updated / new_mag;
        }
        const uint8_t quantized_index =
            nearest_codebook_index(normalized_value, s_codebook, codebook_size);
        m_sign[idx] = quantized_index;
        if (lr != 0.0f) {
            const float quantized_value = s_codebook[static_cast<int>(quantized_index)] * new_mag;
            // Explicit round-to-nearest mul/sub: FMA-independent, bit-identical to
            // the v1 and fully-fused parameter writes (see the v1 kernel comment).
            const float update_value = __fmul_rn(__fmul_rn(quantized_value, stepsize[block_idx]), lr);
            p[idx] = static_cast<scalar_t>(__fsub_rn(static_cast<float>(p[idx]), update_value));
        }
    }
}

// ---------------------------------------------------------------------------
// Muon momentum emit (phase 2): identical magnitude/quantize/state-write ops as
// gefen_update_flat_kernel, but instead of writing the parameter it emits the
// DENSE quantized momentum (codebook[new_index] * new_magnitude) that Muon's
// Newton-Schulz consumes. This replaces the old lr==0 dummy-stepsize call into
// the generic update kernel followed by a separate full-size codebook gather:
// the dense momentum is written in the SAME pass that quantizes the state, so
// the second gather over every parameter is eliminated.
//
// Bit-exactness contract (vs the old `dequantize(m_sign) * m_magnitude`):
//   * m_sign / new_magnitude are produced by the identical ops as
//     gefen_update_flat_kernel (phase 1 magnitude kernels are reused verbatim),
//     so they are bit-identical to the generic v2 update -> and, by the v1/v2
//     parity suite, to v1 as well.
//   * momentum_out reproduces the host dequant rounding op-for-op: the old path
//     was `coeff = codebook[idx].to(scalar_t)` (a single fp32->scalar_t round)
//     then `coeff.mul_(m_magnitude)` (an in-place bf16/fp16 multiply, i.e. the
//     fp32 product rounded back to scalar_t). So: cast codebook value to
//     scalar_t FIRST, multiply by the fp32 magnitude, then cast the product to
//     scalar_t. For fp32 params both casts are no-ops.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void gefen_momentum_emit_flat_kernel(
    const scalar_t* __restrict__ grad_view,
    uint8_t* __restrict__ m_sign,
    const float* __restrict__ old_magnitude,
    const float* __restrict__ new_magnitude,
    const float* __restrict__ codebook,
    scalar_t* __restrict__ momentum_out,
    int codebook_size,
    int64_t period,
    int64_t total_numel,
    float beta1
) {
    extern __shared__ float s_codebook[];
    for (int i = threadIdx.x; i < codebook_size; i += blockDim.x) {
        s_codebook[i] = codebook[i];
    }
    __syncthreads();
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total_numel; idx += stride) {
        const int64_t block_idx = idx / period;
        const float new_mag = new_magnitude[block_idx];
        const float updated = updated_momentum(
            grad_view, m_sign, s_codebook, old_magnitude[block_idx], idx, beta1);
        float normalized_value = 0.0f;
        if (new_mag > 0.0f) {
            normalized_value = updated / new_mag;
        }
        const uint8_t quantized_index =
            nearest_codebook_index(normalized_value, s_codebook, codebook_size);
        m_sign[idx] = quantized_index;
        // Dense quantized momentum, rounded exactly as the old host
        // `codebook[idx].to(scalar_t).mul_(m_magnitude)` two-step round.
        const scalar_t coeff = static_cast<scalar_t>(s_codebook[static_cast<int>(quantized_index)]);
        momentum_out[idx] = static_cast<scalar_t>(static_cast<float>(coeff) * new_mag);
    }
}

// ---------------------------------------------------------------------------
// v2 fully-fused (Tier-1 K1/K2/K3 on the two-phase path).
//
// Phase 1 already streams every grad element to form the per-block magnitude;
// these variants additionally accumulate Sum(grad^2) per block in the SAME read
// (K1), so the separate automatic_vmean kernel + its redundant grad pass are
// eliminated for v2-routed params. A tiny finalize over num_blocks forms the
// vmean EMA and the per-block stepsize in-kernel (K2), removing the host
// sqrt/div/reciprocal/mul launches. Phase 2 folds weight decay into the write
// (K3).
//
// Bit-exactness contract (vs the decomposed v2 reference it replaces):
//   * magnitude is an order-free absmax -> bit-identical to the plain v2
//     magnitude kernels (the Sum(grad^2) accumulator is independent of it).
//   * given the kernel's own vmean, the finalize stepsize uses the identical
//     IEEE primitives as gefen.py's host math (and the v1 fully-fused kernel,
//     which is gated bit-exact vs that host math), and phase 2's write copies
//     the v2 update kernel op-for-op (+ the v1-style weight-decay fold), so p,
//     m_sign and m_magnitude are BIT-IDENTICAL to the reference pipeline run on
//     that same vmean.
//   * vmean itself is NOT bit-identical to the standalone tree reduction: the
//     per-block Sum(grad^2) is formed by atomic accumulation, whose summation
//     order is non-deterministic. The deviation is a sub-ULP-scale perturbation
//     of a 2nd-moment EMA (convergence-neutral); the parity suite asserts it
//     stays within a tight rtol of the standalone kernel. period==1 has a single
//     term and no atomic contention across distinct blocks, so it stays
//     bit-identical there.
// ---------------------------------------------------------------------------

// Small/medium period: flat grid-stride, one atomicMax + one atomicAdd per
// element. Magnitude path is op-identical to gefen_magnitude_flat_kernel.
template <typename scalar_t>
__global__ void gefen_magnitude_sumsq_flat_kernel(
    const scalar_t* __restrict__ grad_view,
    const uint8_t* __restrict__ m_sign,
    const float* __restrict__ old_magnitude,
    const float* __restrict__ codebook,
    float* __restrict__ new_magnitude,
    float* __restrict__ sumsq,
    int codebook_size,
    int64_t period,
    int64_t total_numel,
    float beta1
) {
    extern __shared__ float s_codebook[];
    for (int i = threadIdx.x; i < codebook_size; i += blockDim.x) {
        s_codebook[i] = codebook[i];
    }
    __syncthreads();
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total_numel; idx += stride) {
        const int64_t block_idx = idx / period;
        // Use the shared updated_momentum() helper for the magnitude so nvcc's
        // FMA-contraction of `beta1*m + (1-beta1)*g` is byte-identical to the
        // plain v2 magnitude kernel (an inlined copy contracts differently and
        // perturbs the absmax). The extra grad read for the square is L2-hot.
        const float updated = updated_momentum(
            grad_view, m_sign, s_codebook, old_magnitude[block_idx], idx, beta1);
        atomic_max_nonneg(&new_magnitude[block_idx], fabsf(updated));
        const float grad_value = static_cast<float>(grad_view[idx]);
        atomicAdd(&sumsq[block_idx], grad_value * grad_value);
    }
}

// Large period: split each row across blocks_per_row CUDA blocks; block-local
// max + sum trees, then one atomicMax and one atomicAdd per CUDA block.
template <typename scalar_t>
__global__ void gefen_magnitude_sumsq_split_kernel(
    const scalar_t* __restrict__ grad_view,
    const uint8_t* __restrict__ m_sign,
    const float* __restrict__ old_magnitude,
    const float* __restrict__ codebook,
    float* __restrict__ new_magnitude,
    float* __restrict__ sumsq,
    int codebook_size,
    int64_t period,
    int64_t num_blocks,
    int blocks_per_row,
    float beta1
) {
    // Shared: [blockDim.x max][blockDim.x sum][codebook_size codebook].
    extern __shared__ float smem[];
    float* shared_max = smem;
    float* shared_sum = smem + blockDim.x;
    float* s_codebook = shared_sum + blockDim.x;
    for (int i = threadIdx.x; i < codebook_size; i += blockDim.x) {
        s_codebook[i] = codebook[i];
    }
    __syncthreads();
    const int64_t row = static_cast<int64_t>(blockIdx.x) / blocks_per_row;
    const int sub = static_cast<int>(static_cast<int64_t>(blockIdx.x) % blocks_per_row);
    if (row >= num_blocks) {
        return;
    }
    const float old_mag = old_magnitude[row];
    const int64_t row_start = row * period;
    float local_absmax = 0.0f;
    float local_sumsq = 0.0f;
    for (int64_t offset = static_cast<int64_t>(sub) * blockDim.x + threadIdx.x;
         offset < period; offset += static_cast<int64_t>(blocks_per_row) * blockDim.x) {
        const int64_t idx = row_start + offset;
        // Helper for the magnitude (FMA-identical to the plain split kernel);
        // separate grad read for the square.
        const float updated = updated_momentum(
            grad_view, m_sign, s_codebook, old_mag, idx, beta1);
        const float a = fabsf(updated);
        if (a > local_absmax) {
            local_absmax = a;
        }
        const float grad_value = static_cast<float>(grad_view[idx]);
        local_sumsq += grad_value * grad_value;
    }
    shared_max[threadIdx.x] = local_absmax;
    shared_sum[threadIdx.x] = local_sumsq;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (shared_max[threadIdx.x + s] > shared_max[threadIdx.x]) {
                shared_max[threadIdx.x] = shared_max[threadIdx.x + s];
            }
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomic_max_nonneg(&new_magnitude[row], shared_max[0]);
        atomicAdd(&sumsq[row], shared_sum[0]);
    }
}

// K2 finalize over num_blocks: form the vmean EMA from the per-block Sum(grad^2)
// and the per-block stepsize, op-identical to automatic_vmean_update_kernel's
// EMA and gefen.py's host stepsize math. stepsize is written back into the
// sumsq buffer (its input is consumed first).
__global__ void gefen_finalize_vmean_stepsize_kernel(
    float* __restrict__ vmean,
    float* __restrict__ sumsq_to_stepsize,
    int64_t num_blocks,
    int64_t period,
    float beta2,
    float eps,
    float inv_sqrt_bias_correction_2,
    float inv_bias_correction_1
) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < num_blocks; i += stride) {
        const float mean_square = sumsq_to_stepsize[i] / static_cast<float>(period);
        const float previous_vmean = vmean[i];
        const float updated_vmean = beta2 * previous_vmean + (1.0f - beta2) * mean_square;
        vmean[i] = updated_vmean;
        float h = __fsqrt_rn(updated_vmean);
        h = __fmul_rn(h, inv_sqrt_bias_correction_2);
        h = __fadd_rn(h, eps);
        sumsq_to_stepsize[i] = __fmul_rn(__frcp_rn(h), inv_bias_correction_1);
    }
}

// Phase 2: quantize + parameter update, with weight decay folded into the write
// (K3). Magnitude/quantize/momentum ops are op-identical to
// gefen_update_flat_kernel; the wd==0 (factor==1) branch is bit-identical to it.
template <typename scalar_t>
__global__ void gefen_update_flat_full_kernel(
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
    float lr,
    float weight_decay_factor
) {
    extern __shared__ float s_codebook[];
    for (int i = threadIdx.x; i < codebook_size; i += blockDim.x) {
        s_codebook[i] = codebook[i];
    }
    __syncthreads();
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total_numel; idx += stride) {
        const int64_t block_idx = idx / period;
        const float new_mag = new_magnitude[block_idx];
        const float updated = updated_momentum(
            grad_view, m_sign, s_codebook, old_magnitude[block_idx], idx, beta1);
        float normalized_value = 0.0f;
        if (new_mag > 0.0f) {
            normalized_value = updated / new_mag;
        }
        const uint8_t quantized_index =
            nearest_codebook_index(normalized_value, s_codebook, codebook_size);
        m_sign[idx] = quantized_index;
        if (lr != 0.0f) {
            const float quantized_value = s_codebook[static_cast<int>(quantized_index)] * new_mag;
            const float update_value = __fmul_rn(__fmul_rn(quantized_value, stepsize[block_idx]), lr);
            if (weight_decay_factor == 1.0f) {
                p[idx] = static_cast<scalar_t>(__fsub_rn(static_cast<float>(p[idx]), update_value));
            } else {
                const float decayed = static_cast<float>(
                    static_cast<scalar_t>(__fmul_rn(static_cast<float>(p[idx]), weight_decay_factor)));
                p[idx] = static_cast<scalar_t>(__fsub_rn(decayed, update_value));
            }
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
    // nearest_codebook_index() returns uint8_t, so >256 entries would wrap the
    // stored indices; an empty table would size shared memory at 0. Packed
    // indices store 2 entries per byte (4-bit), so cap at 16 in that mode.
    const int64_t codebook_numel = codebook.numel();
    if (codebook_numel < 1 || codebook_numel > 256) {
        throw std::invalid_argument("Expected codebook size in [1, 256].");
    }
    if (packed_indices && codebook_numel > 16) {
        throw std::invalid_argument("Expected packed codebook size in [1, 16].");
    }
    // these two may create additional memory footprint.
    const int threads = choose_threads(period);
    const dim3 grid(static_cast<unsigned int>(num_blocks));
    const dim3 block(static_cast<unsigned int>(threads));
    // Shared holds the staged codebook (<=256) plus the per-thread reduction max.
    const size_t shared_bytes =
        (static_cast<size_t>(threads) + static_cast<size_t>(codebook.numel())) * sizeof(float);

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
    double weight_decay_factor
) {
    if (!p.is_cuda() || !grad_view.is_cuda() || !m_sign.is_cuda() || !m_magnitude.is_cuda() || !vmean.is_cuda() || !codebook.is_cuda()) {
        throw std::invalid_argument("Expected all tensors to be on CUDA.");
    }
    if (!p.is_contiguous() || !grad_view.is_contiguous() || !m_sign.is_contiguous() ||
        !m_magnitude.is_contiguous() || !vmean.is_contiguous() || !codebook.is_contiguous()) {
        throw std::invalid_argument("Expected all tensors to be contiguous.");
    }
    if (grad_view.dim() != 2) {
        throw std::invalid_argument("Expected grad_view to be 2D.");
    }
    if (m_magnitude.dim() != 2 || m_magnitude.size(1) != 1) {
        throw std::invalid_argument("Expected m_magnitude to have shape [num_blocks, 1].");
    }
    if (vmean.dim() != 2 || vmean.size(1) != 1) {
        throw std::invalid_argument("Expected vmean to have shape [num_blocks, 1].");
    }
    if (m_sign.scalar_type() != at::kByte) {
        throw std::invalid_argument("Expected m_sign to have dtype uint8.");
    }
    if (codebook.scalar_type() != at::kFloat) {
        throw std::invalid_argument("Expected codebook to have dtype float32.");
    }
    if (m_magnitude.scalar_type() != at::kFloat || vmean.scalar_type() != at::kFloat) {
        throw std::invalid_argument("Expected m_magnitude and vmean to have dtype float32.");
    }
    // Raw-pointer launch reads grad_view as scalar_t (p's dispatch dtype); a
    // mismatch would reinterpret memory at the wrong width.
    if (grad_view.scalar_type() != p.scalar_type()) {
        throw std::invalid_argument("Expected grad_view dtype to match p.");
    }
    // All tensors are dereferenced on p's device; reject cross-device inputs that
    // the is_cuda() checks above would otherwise let through.
    if (grad_view.device() != p.device() || m_sign.device() != p.device() ||
        m_magnitude.device() != p.device() || vmean.device() != p.device() ||
        codebook.device() != p.device()) {
        throw std::invalid_argument("Expected all tensors on the same device as p.");
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
    } else if (m_sign.numel() != total_numel) {
        throw std::invalid_argument("Expected unpacked m_sign.numel() to match grad_view.numel().");
    }
    if (m_magnitude.size(0) != num_blocks || vmean.size(0) != num_blocks) {
        throw std::invalid_argument("Expected m_magnitude and vmean to match the number of blocks.");
    }
    if (period <= 0) {
        throw std::invalid_argument("Expected grad_view to have a positive period.");
    }
    // nearest_codebook_index() returns uint8_t, so >256 entries would wrap the
    // stored indices; an empty table would size shared memory at 0. Packed
    // indices store 2 entries per byte (4-bit), so cap at 16 in that mode.
    const int64_t codebook_numel = codebook.numel();
    if (codebook_numel < 1 || codebook_numel > 256) {
        throw std::invalid_argument("Expected codebook size in [1, 256].");
    }
    if (packed_indices && codebook_numel > 16) {
        throw std::invalid_argument("Expected packed codebook size in [1, 16].");
    }

    const int threads = choose_threads(period);
    const dim3 grid(static_cast<unsigned int>(num_blocks));
    const dim3 block(static_cast<unsigned int>(threads));
    // Shared holds the staged codebook plus two per-thread reduction buffers
    // (absmax and sum-of-squares).
    const size_t shared_bytes =
        (static_cast<size_t>(codebook.numel()) + 2 * static_cast<size_t>(threads)) * sizeof(float);

    GEFEN_DISPATCH_FLOAT_HALF_BF16(
        p.scalar_type(),
        "automatic_gefen_fused_full_update_cuda",
        [&] {
            automatic_gefen_fused_full_update_kernel<scalar_t><<<grid, block, shared_bytes>>>(
                p.data_ptr<scalar_t>(),
                grad_view.data_ptr<scalar_t>(),
                m_sign.data_ptr<uint8_t>(),
                m_magnitude.data_ptr<float>(),
                vmean.data_ptr<float>(),
                codebook.data_ptr<float>(),
                static_cast<int>(codebook.numel()),
                packed_indices,
                period,
                num_blocks,
                static_cast<float>(beta1),
                static_cast<float>(beta2),
                static_cast<float>(lr),
                static_cast<float>(eps),
                static_cast<float>(inv_sqrt_bias_correction_2),
                static_cast<float>(inv_bias_correction_1),
                static_cast<float>(weight_decay_factor)
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
    // Raw-pointer dtype guards: grad_view is read as scalar_t (the dispatch type
    // from p), and m_magnitude/stepsize are read as float*; a mismatch would
    // reinterpret memory at the wrong width.
    if (grad_view.scalar_type() != p.scalar_type()) {
        throw std::invalid_argument("Expected grad_view dtype to match p.");
    }
    if (m_magnitude.scalar_type() != at::kFloat || stepsize.scalar_type() != at::kFloat) {
        throw std::invalid_argument("Expected m_magnitude and stepsize to have dtype float32.");
    }
    // nearest_codebook_index() returns uint8_t, so a codebook larger than 256
    // entries would wrap the stored indices; also reject an empty table.
    if (codebook.numel() < 1 || codebook.numel() > 256) {
        throw std::invalid_argument("Expected codebook size in [1, 256].");
    }

    auto new_magnitude = at::zeros_like(m_magnitude);

    const int codebook_size = static_cast<int>(codebook.numel());
    const int threads = 256;
    const int sm_count = cached_sm_count(p.get_device());
    // Target a few thousand resident blocks for the flat phases.
    const int64_t flat_blocks_cap = static_cast<int64_t>(sm_count) * 32;
    const size_t codebook_bytes = static_cast<size_t>(codebook_size) * sizeof(float);

    // Phase 1: per-block magnitude (absmax of updated momentum).
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, p.scalar_type(),
        "gefen_v2_magnitude", [&] {
            if (period <= 2048) {
                int64_t nblocks = (total_numel + threads - 1) / threads;
                if (nblocks > flat_blocks_cap) nblocks = flat_blocks_cap;
                if (nblocks < 1) nblocks = 1;
                gefen_magnitude_flat_kernel<scalar_t><<<static_cast<unsigned int>(nblocks), threads, codebook_bytes>>>(
                    grad_view.data_ptr<scalar_t>(), m_sign.data_ptr<uint8_t>(),
                    m_magnitude.data_ptr<float>(), codebook.data_ptr<float>(),
                    new_magnitude.data_ptr<float>(), codebook_size, period, total_numel,
                    static_cast<float>(beta1));
            } else {
                int blocks_per_row = static_cast<int>((period + threads * 64 - 1) / (threads * 64));
                if (blocks_per_row < 1) blocks_per_row = 1;
                // Keep total grid bounded but well above SM count.
                const int max_bpr = static_cast<int>((flat_blocks_cap + num_blocks - 1) / num_blocks);
                if (blocks_per_row > max_bpr && max_bpr >= 1) blocks_per_row = max_bpr;
                if (blocks_per_row < 1) blocks_per_row = 1;
                const int64_t grid = num_blocks * blocks_per_row;
                const size_t shmem = static_cast<size_t>(threads) * sizeof(float) + codebook_bytes;
                gefen_magnitude_split_kernel<scalar_t><<<static_cast<unsigned int>(grid), threads, shmem>>>(
                    grad_view.data_ptr<scalar_t>(), m_sign.data_ptr<uint8_t>(),
                    m_magnitude.data_ptr<float>(), codebook.data_ptr<float>(),
                    new_magnitude.data_ptr<float>(), codebook_size, period, num_blocks, blocks_per_row,
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
            gefen_update_flat_kernel<scalar_t><<<static_cast<unsigned int>(nblocks), threads, codebook_bytes>>>(
                p.data_ptr<scalar_t>(), grad_view.data_ptr<scalar_t>(), m_sign.data_ptr<uint8_t>(),
                m_magnitude.data_ptr<float>(), new_magnitude.data_ptr<float>(),
                stepsize.data_ptr<float>(), codebook.data_ptr<float>(), codebook_size,
                period, total_numel, static_cast<float>(beta1), static_cast<float>(lr));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    m_magnitude.copy_(new_magnitude);
}

void gefen_quantized_momentum_update_cuda(
    at::Tensor grad_view,
    at::Tensor m_sign,
    at::Tensor m_magnitude,
    at::Tensor codebook,
    at::Tensor momentum_out,
    double beta1
) {
    if (!grad_view.is_cuda() || !m_sign.is_cuda() || !m_magnitude.is_cuda() ||
        !codebook.is_cuda() || !momentum_out.is_cuda()) {
        throw std::invalid_argument("Expected all tensors to be on CUDA.");
    }
    if (grad_view.device() != momentum_out.device() || m_sign.device() != momentum_out.device() ||
        m_magnitude.device() != momentum_out.device() || codebook.device() != momentum_out.device()) {
        throw std::invalid_argument("Expected all tensors on the same device.");
    }
    if (!grad_view.is_contiguous() || !m_sign.is_contiguous() || !m_magnitude.is_contiguous() ||
        !codebook.is_contiguous() || !momentum_out.is_contiguous()) {
        throw std::invalid_argument("Expected all tensors to be contiguous.");
    }
    if (grad_view.dim() != 2 || momentum_out.dim() != 2) {
        throw std::invalid_argument("Expected grad_view and momentum_out to be 2D.");
    }
    if (m_magnitude.dim() != 2 || m_magnitude.size(1) != 1) {
        throw std::invalid_argument("Expected m_magnitude to have shape [num_blocks, 1].");
    }
    if (m_sign.scalar_type() != at::kByte) {
        throw std::invalid_argument("Expected m_sign to have dtype uint8.");
    }
    if (codebook.scalar_type() != at::kFloat) {
        throw std::invalid_argument("Expected codebook to have dtype float32.");
    }
    if (m_magnitude.scalar_type() != at::kFloat) {
        throw std::invalid_argument("Expected m_magnitude to have dtype float32.");
    }
    // momentum_out is written as scalar_t (grad_view's dispatch dtype); a
    // mismatch would reinterpret memory at the wrong width.
    if (grad_view.scalar_type() != momentum_out.scalar_type()) {
        throw std::invalid_argument("Expected momentum_out dtype to match grad_view.");
    }

    c10::cuda::CUDAGuard device_guard(grad_view.device());

    const int64_t num_blocks = grad_view.size(0);
    const int64_t period = grad_view.size(1);
    const int64_t total_numel = num_blocks * period;
    if (m_sign.numel() != total_numel || momentum_out.numel() != total_numel) {
        throw std::invalid_argument("Expected m_sign and momentum_out numel to match grad_view.");
    }
    if (m_magnitude.size(0) != num_blocks) {
        throw std::invalid_argument("Expected m_magnitude to match num_blocks.");
    }
    if (codebook.numel() < 1 || codebook.numel() > 256) {
        throw std::invalid_argument("Expected codebook size in [1, 256].");
    }
    if (period <= 0) {
        throw std::invalid_argument("Expected grad_view to have a positive period.");
    }
    // Empty param: nothing to quantize/emit, and the launch math below divides by
    // num_blocks. Bail before any launch / divide-by-zero.
    if (total_numel == 0) {
        return;
    }

    // new per-block magnitude (absmax of the updated momentum); the atomicMax
    // target must start at zero, exactly as the generic v2 path.
    auto new_magnitude = at::zeros_like(m_magnitude);

    const int codebook_size = static_cast<int>(codebook.numel());
    const int threads = 256;
    const int sm_count = cached_sm_count(grad_view.get_device());
    const int64_t flat_blocks_cap = static_cast<int64_t>(sm_count) * 32;
    const size_t codebook_bytes = static_cast<size_t>(codebook_size) * sizeof(float);

    // Phase 1: per-block magnitude (reuses the generic v2 magnitude kernels, so
    // new_magnitude is bit-identical to the generic update's m_magnitude).
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, grad_view.scalar_type(),
        "gefen_momentum_magnitude", [&] {
            if (period <= 2048) {
                int64_t nblocks = (total_numel + threads - 1) / threads;
                if (nblocks > flat_blocks_cap) nblocks = flat_blocks_cap;
                if (nblocks < 1) nblocks = 1;
                gefen_magnitude_flat_kernel<scalar_t><<<static_cast<unsigned int>(nblocks), threads, codebook_bytes>>>(
                    grad_view.data_ptr<scalar_t>(), m_sign.data_ptr<uint8_t>(),
                    m_magnitude.data_ptr<float>(), codebook.data_ptr<float>(),
                    new_magnitude.data_ptr<float>(), codebook_size, period, total_numel,
                    static_cast<float>(beta1));
            } else {
                int blocks_per_row = static_cast<int>((period + threads * 64 - 1) / (threads * 64));
                if (blocks_per_row < 1) blocks_per_row = 1;
                const int max_bpr = static_cast<int>((flat_blocks_cap + num_blocks - 1) / num_blocks);
                if (blocks_per_row > max_bpr && max_bpr >= 1) blocks_per_row = max_bpr;
                if (blocks_per_row < 1) blocks_per_row = 1;
                const int64_t grid = num_blocks * blocks_per_row;
                const size_t shmem = static_cast<size_t>(threads) * sizeof(float) + codebook_bytes;
                gefen_magnitude_split_kernel<scalar_t><<<static_cast<unsigned int>(grid), threads, shmem>>>(
                    grad_view.data_ptr<scalar_t>(), m_sign.data_ptr<uint8_t>(),
                    m_magnitude.data_ptr<float>(), codebook.data_ptr<float>(),
                    new_magnitude.data_ptr<float>(), codebook_size, period, num_blocks, blocks_per_row,
                    static_cast<float>(beta1));
            }
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Phase 2: quantize state + emit the dense quantized momentum (no p write).
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, grad_view.scalar_type(),
        "gefen_momentum_emit", [&] {
            int64_t nblocks = (total_numel + threads - 1) / threads;
            if (nblocks > flat_blocks_cap) nblocks = flat_blocks_cap;
            if (nblocks < 1) nblocks = 1;
            gefen_momentum_emit_flat_kernel<scalar_t><<<static_cast<unsigned int>(nblocks), threads, codebook_bytes>>>(
                grad_view.data_ptr<scalar_t>(), m_sign.data_ptr<uint8_t>(),
                m_magnitude.data_ptr<float>(), new_magnitude.data_ptr<float>(),
                codebook.data_ptr<float>(), momentum_out.data_ptr<scalar_t>(), codebook_size,
                period, total_numel, static_cast<float>(beta1));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    m_magnitude.copy_(new_magnitude);
}

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
    double weight_decay_factor
) {
    if (packed_indices) {
        throw std::invalid_argument("v2 fused update does not support packed indices.");
    }
    if (!p.is_cuda() || !grad_view.is_cuda() || !m_sign.is_cuda() || !m_magnitude.is_cuda() || !vmean.is_cuda() || !codebook.is_cuda()) {
        throw std::invalid_argument("Expected all tensors to be on CUDA.");
    }
    // All tensors are dereferenced on p's device; reject cross-device inputs that
    // the is_cuda() checks above would otherwise let through (CUDAGuard only
    // selects the launch device, it does not relocate the operands).
    if (grad_view.device() != p.device() || m_sign.device() != p.device() ||
        m_magnitude.device() != p.device() || vmean.device() != p.device() ||
        codebook.device() != p.device()) {
        throw std::invalid_argument("Expected all tensors on the same device as p.");
    }
    if (!p.is_contiguous() || !grad_view.is_contiguous() || !m_sign.is_contiguous() ||
        !m_magnitude.is_contiguous() || !vmean.is_contiguous() || !codebook.is_contiguous()) {
        throw std::invalid_argument("Expected all tensors to be contiguous.");
    }
    if (grad_view.dim() != 2) {
        throw std::invalid_argument("Expected grad_view to be 2D.");
    }
    if (m_magnitude.dim() != 2 || m_magnitude.size(1) != 1) {
        throw std::invalid_argument("Expected m_magnitude to have shape [num_blocks, 1].");
    }
    if (vmean.dim() != 2 || vmean.size(1) != 1) {
        throw std::invalid_argument("Expected vmean to have shape [num_blocks, 1].");
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
    if (m_magnitude.size(0) != num_blocks || vmean.size(0) != num_blocks) {
        throw std::invalid_argument("Expected m_magnitude and vmean to match num_blocks.");
    }
    if (grad_view.scalar_type() != p.scalar_type()) {
        throw std::invalid_argument("Expected grad_view dtype to match p.");
    }
    if (m_magnitude.scalar_type() != at::kFloat || vmean.scalar_type() != at::kFloat) {
        throw std::invalid_argument("Expected m_magnitude and vmean to have dtype float32.");
    }
    if (codebook.numel() < 1 || codebook.numel() > 256) {
        throw std::invalid_argument("Expected codebook size in [1, 256].");
    }
    if (period <= 0) {
        throw std::invalid_argument("Expected grad_view to have a positive period.");
    }
    // Empty param (num_blocks == 0 -> total_numel == 0): nothing to update, and
    // the split-path block-per-row math below divides by num_blocks. Bail before
    // any launch / divide-by-zero.
    if (total_numel == 0) {
        return;
    }

    // One zeroed scratch instead of two separate zeros_like fills (one fill
    // launch, not two): row 0 is the new per-block magnitude (atomicMax target),
    // row 1 is the K1 Sum(grad^2) accumulator (atomicAdd target, later reused as
    // the stepsize buffer by the finalize K2). Both rows are contiguous
    // [num_blocks, 1] views, so the kernels index them exactly as before.
    auto scratch = at::zeros({2, num_blocks, 1}, m_magnitude.options());
    auto new_magnitude = scratch[0];
    auto sumsq = scratch[1];

    const int codebook_size = static_cast<int>(codebook.numel());
    const int threads = 256;
    const int sm_count = cached_sm_count(p.get_device());
    const int64_t flat_blocks_cap = static_cast<int64_t>(sm_count) * 32;
    const size_t codebook_bytes = static_cast<size_t>(codebook_size) * sizeof(float);

    // Phase 1: per-block magnitude (absmax) + Sum(grad^2) in one grad read.
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, p.scalar_type(),
        "gefen_v2_full_magnitude", [&] {
            if (period <= 2048) {
                int64_t nblocks = (total_numel + threads - 1) / threads;
                if (nblocks > flat_blocks_cap) nblocks = flat_blocks_cap;
                if (nblocks < 1) nblocks = 1;
                gefen_magnitude_sumsq_flat_kernel<scalar_t><<<static_cast<unsigned int>(nblocks), threads, codebook_bytes>>>(
                    grad_view.data_ptr<scalar_t>(), m_sign.data_ptr<uint8_t>(),
                    m_magnitude.data_ptr<float>(), codebook.data_ptr<float>(),
                    new_magnitude.data_ptr<float>(), sumsq.data_ptr<float>(),
                    codebook_size, period, total_numel, static_cast<float>(beta1));
            } else {
                int blocks_per_row = static_cast<int>((period + threads * 64 - 1) / (threads * 64));
                if (blocks_per_row < 1) blocks_per_row = 1;
                const int max_bpr = static_cast<int>((flat_blocks_cap + num_blocks - 1) / num_blocks);
                if (blocks_per_row > max_bpr && max_bpr >= 1) blocks_per_row = max_bpr;
                if (blocks_per_row < 1) blocks_per_row = 1;
                const int64_t grid = num_blocks * blocks_per_row;
                // Two per-thread reduction buffers (max, sum) plus the codebook.
                const size_t shmem = 2 * static_cast<size_t>(threads) * sizeof(float) + codebook_bytes;
                gefen_magnitude_sumsq_split_kernel<scalar_t><<<static_cast<unsigned int>(grid), threads, shmem>>>(
                    grad_view.data_ptr<scalar_t>(), m_sign.data_ptr<uint8_t>(),
                    m_magnitude.data_ptr<float>(), codebook.data_ptr<float>(),
                    new_magnitude.data_ptr<float>(), sumsq.data_ptr<float>(),
                    codebook_size, period, num_blocks, blocks_per_row, static_cast<float>(beta1));
            }
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // K2: vmean EMA + per-block stepsize (writes stepsize into sumsq).
    {
        const int fthreads = 256;
        int64_t fblocks = (num_blocks + fthreads - 1) / fthreads;
        if (fblocks > flat_blocks_cap) fblocks = flat_blocks_cap;
        if (fblocks < 1) fblocks = 1;
        gefen_finalize_vmean_stepsize_kernel<<<static_cast<unsigned int>(fblocks), fthreads>>>(
            vmean.data_ptr<float>(), sumsq.data_ptr<float>(), num_blocks, period,
            static_cast<float>(beta2), static_cast<float>(eps),
            static_cast<float>(inv_sqrt_bias_correction_2),
            static_cast<float>(inv_bias_correction_1));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Phase 2: quantize + parameter update with weight decay folded in (K3).
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, p.scalar_type(),
        "gefen_v2_full_update", [&] {
            int64_t nblocks = (total_numel + threads - 1) / threads;
            if (nblocks > flat_blocks_cap) nblocks = flat_blocks_cap;
            if (nblocks < 1) nblocks = 1;
            gefen_update_flat_full_kernel<scalar_t><<<static_cast<unsigned int>(nblocks), threads, codebook_bytes>>>(
                p.data_ptr<scalar_t>(), grad_view.data_ptr<scalar_t>(), m_sign.data_ptr<uint8_t>(),
                m_magnitude.data_ptr<float>(), new_magnitude.data_ptr<float>(),
                sumsq.data_ptr<float>(), codebook.data_ptr<float>(), codebook_size,
                period, total_numel, static_cast<float>(beta1), static_cast<float>(lr),
                static_cast<float>(weight_decay_factor));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Repoint m_magnitude at the just-computed magnitudes instead of copying them
    // back. copy_ was a full-size D2D memcpy + its own launch every step per
    // param; set_ is a metadata-only storage rebind (no kernel launch, no copy)
    // and the values m_magnitude ends up holding are identical. m_magnitude now
    // shares `scratch`'s storage; the tiny row-1 (sumsq) half stays alive until
    // the next step rebinds m_magnitude (bounded by num_blocks floats per param).
    // phase 2 above has already consumed the old m_magnitude (old_magnitude), so
    // the rebind is safe to do after it.
    m_magnitude.set_(new_magnitude);
}

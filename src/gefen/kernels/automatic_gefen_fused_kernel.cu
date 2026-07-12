#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAMacros.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/util/Optional.h>

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

// Counter-based (stateless) uniform in [0, 1) from a 64-bit key, SplitMix64
// finalizer over (rng_seed, element_idx). Stateless so it needs no per-thread
// curand setup, is reproducible run-to-run for a fixed (seed, idx), and costs a
// handful of integer ops -- negligible against the per-element global-memory
// codebook gather + binary search already in the quantize loop. The seed is the
// optimizer's step count -- delivered as a host kernel argument on the default
// path, or read from a device-side step tensor (rng_seed_dev, advanced in
// place once per step) under capturable so captured CUDA graphs replay fresh
// seeds -- so the stochastic rounding decorrelates across steps, which is what
// debiases the EMA trajectory.
__device__ __forceinline__ float gefen_sr_uniform(
    uint64_t rng_seed, int64_t element_idx
) {
    uint64_t z = rng_seed + 0x9E3779B97F4A7C15ULL *
        (static_cast<uint64_t>(element_idx) + 1ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    // top 24 bits -> [0, 1) with 2^-24 resolution (exact in fp32).
    return static_cast<float>(z >> 40) * (1.0f / 16777216.0f);
}

// Quantize one normalized value to a codebook index. When ``stochastic`` is
// false this is bit-for-bit ``nearest_codebook_index`` (the parity-preserving
// default). When true it rounds to one of the two bracketing codewords with
// probability proportional to closeness so that E[codebook[idx]] == value
// exactly (unbiased stochastic rounding): P(round up to the right codeword) =
// (value - lo) / (hi - lo). This removes the systematic bias that deterministic
// nearest-rounding of the EMA momentum accumulates over a long horizon.
__device__ __forceinline__ uint8_t quantize_codebook_index(
    float normalized_value,
    const float* __restrict__ codebook,
    int codebook_size,
    bool stochastic,
    uint64_t rng_seed,
    int64_t element_idx
) {
    if (!stochastic) {
        return nearest_codebook_index(normalized_value, codebook, codebook_size);
    }

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
    if (left_idx == right_idx) {
        return static_cast<uint8_t>(right_idx);
    }

    const float lo = codebook[left_idx];
    const float hi = codebook[right_idx];
    const float span = hi - lo;
    // value clamps into [lo, hi] at the codebook ends (searchsorted can land out
    // of the bracket); inside the bracket the clamp is a no-op.
    float v = normalized_value;
    if (v < lo) {
        v = lo;
    } else if (v > hi) {
        v = hi;
    }
    const float p_up = (span > 0.0f) ? (v - lo) / span : 0.0f;
    const float u = gefen_sr_uniform(rng_seed, element_idx);
    return static_cast<uint8_t>((u < p_up) ? right_idx : left_idx);
}

// ---------------------------------------------------------------------------
// LUT-narrowed codebook search. lut[b] = #codebook entries whose bucket < b,
// with bucket(v) = clamp(floor((v+1) * buckets / 2), 0, buckets-1) over the
// normalized domain [-1, 1]. Bucketization is monotone in v, so for any v the
// lower_bound answer lies in [lut[bucket(v)], lut[bucket(v)+1]]: every entry
// in an earlier bucket is strictly < v and every entry in a later bucket is
// > v. Running the IDENTICAL binary search on that sub-range therefore reaches
// the identical lower_bound, and the identical left/right tie-break (nearest)
// or bracketing pair (stochastic) returns the identical index -- the LUT is a
// pure search accelerator, bit-exact by construction. The LUT is built host-
// side once per codebook (the exact-DP codebook is frozen after step 1) and
// read via __ldg, so it stays resident in L1/L2. A null lut falls back to the
// full-range search (bit-identical, just slower).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void gefen_lut_search_bounds(
    float normalized_value,
    const int16_t* __restrict__ lut,
    int lut_buckets,
    int codebook_size,
    int* left,
    int* right
) {
    if (lut == nullptr) {
        *left = 0;
        *right = codebook_size;
        return;
    }
    int b = static_cast<int>(
        floorf((normalized_value + 1.0f) * (0.5f * static_cast<float>(lut_buckets))));
    if (b < 0) {
        b = 0;
    } else if (b > lut_buckets - 1) {
        b = lut_buckets - 1;
    }
    // Defensive clamp: the host-built LUT is monotone in [0, codebook_size] by
    // construction (torch.searchsorted over the bucketized sorted codebook), so
    // for every legitimate caller these clamps are no-ops and the search stays
    // bit-identical. They exist so a malformed LUT handed to the raw extension
    // API (negative / non-monotonic / out-of-range int16 values -- which the
    // host wrapper cannot value-check without a D2H sync) degrades to a safe
    // in-bounds search instead of out-of-bounds codebook indexing.
    int lo = __ldg(&lut[b]);
    int hi = __ldg(&lut[b + 1]);
    if (lo < 0) {
        lo = 0;
    } else if (lo > codebook_size) {
        lo = codebook_size;
    }
    if (hi < lo) {
        hi = lo;
    } else if (hi > codebook_size) {
        hi = codebook_size;
    }
    *left = lo;
    *right = hi;
}

// quantize_codebook_index with a LUT-narrowed lower_bound. Bit-identical to
// quantize_codebook_index for both rounding modes (see gefen_lut_search_bounds);
// lut == nullptr degrades to the exact full-range behavior.
__device__ __forceinline__ uint8_t quantize_codebook_index_lut(
    float normalized_value,
    const float* __restrict__ codebook,
    int codebook_size,
    const int16_t* __restrict__ lut,
    int lut_buckets,
    bool stochastic,
    uint64_t rng_seed,
    int64_t element_idx
) {
    int left;
    int right;
    gefen_lut_search_bounds(
        normalized_value, lut, lut_buckets, codebook_size, &left, &right);
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

    if (!stochastic) {
        const float left_dist = fabsf(normalized_value - codebook[left_idx]);
        const float right_dist = fabsf(normalized_value - codebook[right_idx]);
        return static_cast<uint8_t>(left_dist <= right_dist ? left_idx : right_idx);
    }

    if (left_idx == right_idx) {
        return static_cast<uint8_t>(right_idx);
    }
    const float lo = codebook[left_idx];
    const float hi = codebook[right_idx];
    const float span = hi - lo;
    float v = normalized_value;
    if (v < lo) {
        v = lo;
    } else if (v > hi) {
        v = hi;
    }
    const float p_up = (span > 0.0f) ? (v - lo) / span : 0.0f;
    const float u = gefen_sr_uniform(rng_seed, element_idx);
    return static_cast<uint8_t>((u < p_up) ? right_idx : left_idx);
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

// Muon's momentum-only two-phase path: above this period, reducing locally
// before one atomic per CUDA block is substantially faster than issuing one
// atomicMax per element. The same cutoff keeps the lower-overhead scalar emit
// kernel for the small-period tail; larger periods use the chunked/LUT emitter.
constexpr int64_t GEFEN_MOMENTUM_FLAT_MAX_PERIOD = 64;

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
//
// Multi-row packing: one CUDA block hosts blockDim.y gefen-blocks (rows), each
// processed by its own blockDim.x-thread slice with the IDENTICAL per-row
// thread-stride schedule, halving reduction tree, and scalar math as the
// one-row-per-block original -- so every per-row result is bit-exact by
// construction. The wins are (a) the 256-entry codebook is staged into shared
// once per ~256 threads instead of once per row (the dominant cost at small
// periods, where a row is a single warp), and (b) blocks are >= 2 warps, so
// small-period shapes stop running at 1-warp occupancy. Rows past num_blocks
// (the remainder of the last CUDA block) stay resident but inactive: they skip
// all global reads/writes yet join every __syncthreads (no early return).
template <typename scalar_t>
__global__ void automatic_gefen_fused_full_update_kernel(
    scalar_t* __restrict__ p,
    const scalar_t* __restrict__ grad_view,
    uint8_t* __restrict__ m_sign,
    float* __restrict__ m_magnitude,
    float* __restrict__ vmean,
    const float* __restrict__ codebook,
    const int16_t* __restrict__ lut,
    int lut_buckets,
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
    float weight_decay_factor,
    bool stochastic_round,
    uint64_t rng_seed,
    const int64_t* __restrict__ rng_seed_dev,
    const float* __restrict__ step_scalars
) {
    // Capturable mode: the per-STEP-varying scalars live in device memory (a
    // tiny fp32 buffer refreshed by the host with tensor ops each step) so a
    // captured CUDA graph replays fresh values instead of the host kernel args
    // frozen at capture time. Layout: [lr, 1/sqrt(bc2), 1/bc1, wd_factor],
    // already-fp32 values, so the math below stays fp32 either way (the host
    // path casts python doubles to float at launch). nullptr keeps the legacy
    // host-scalar behavior bit-identically.
    if (step_scalars != nullptr) {
        lr = __ldg(&step_scalars[0]);
        inv_sqrt_bias_correction_2 = __ldg(&step_scalars[1]);
        inv_bias_correction_1 = __ldg(&step_scalars[2]);
        weight_decay_factor = __ldg(&step_scalars[3]);
    }
    // Capturable stochastic rounding: the per-step seed likewise lives in
    // device memory (a 0-dim int64 step tensor advanced in place once per
    // optimizer step), so graph replays dither with a FRESH seed instead of
    // the host rng_seed frozen at capture time. nullptr keeps the legacy
    // host-seed path bit-identically.
    if (stochastic_round && rng_seed_dev != nullptr) {
        rng_seed = static_cast<uint64_t>(
            __ldg(reinterpret_cast<const long long*>(rng_seed_dev)));
    }
    // Shared layout: [codebook_size codebook][tpr*rows max][tpr*rows sumsq],
    // where tpr = blockDim.x (threads per row) and rows = blockDim.y.
    extern __shared__ float smem[];
    float* s_codebook = smem;
    const int tpr = blockDim.x;
    const int rows = blockDim.y;
    float* shared_max_all = smem + codebook_size;
    float* shared_sum_all = shared_max_all + tpr * rows;

    const int lin = threadIdx.y * tpr + threadIdx.x;
    for (int i = lin; i < codebook_size; i += tpr * rows) {
        s_codebook[i] = codebook[i];
    }
    __syncthreads();

    const int64_t block_idx =
        static_cast<int64_t>(blockIdx.x) * rows + threadIdx.y;
    const bool active = block_idx < num_blocks;

    // Per-row slices of the reduction buffers: each row reduces over its own
    // tpr-wide slice with the same stride schedule the one-row kernel used.
    float* shared_max = shared_max_all + threadIdx.y * tpr;
    float* shared_sum = shared_sum_all + threadIdx.y * tpr;

    const int64_t start = active ? block_idx * period : 0;
    const float old_magnitude = active ? m_magnitude[block_idx] : 0.0f;
    float local_absmax = 0.0f;
    float local_sumsq = 0.0f;

    if (active) {
        for (int64_t offset = threadIdx.x; offset < period; offset += tpr) {
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
    }

    shared_max[threadIdx.x] = local_absmax;
    shared_sum[threadIdx.x] = local_sumsq;
    __syncthreads();

    // Fused max + sum tree per row slice: the sum half is bit-identical to the
    // standalone vmean kernel (same stride schedule over tpr, same `+=` order);
    // the max half is bit-identical to v1. Inactive rows reduce zeros.
    for (unsigned int stride = tpr / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (shared_max[threadIdx.x + stride] > shared_max[threadIdx.x]) {
                shared_max[threadIdx.x] = shared_max[threadIdx.x + stride];
            }
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float new_magnitude = shared_max[0];
    if (threadIdx.x == 0 && active) {
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

    if (active) {
        for (int64_t offset = threadIdx.x; offset < period; offset += tpr) {
            const int64_t idx = start + offset;
            float normalized_value = 0.0f;
            const float coeff = s_codebook[static_cast<int>(unpack_codebook_index(m_sign, idx, packed_indices))];
            const float current_m = old_magnitude * coeff;
            const float grad_value = static_cast<float>(grad_view[idx]);
            const float updated_value = beta1 * current_m + (1.0f - beta1) * grad_value;
            if (new_magnitude > 0.0f) {
                normalized_value = updated_value / new_magnitude;
            }
            const uint8_t quantized_index = quantize_codebook_index_lut(
                normalized_value, s_codebook, codebook_size, lut, lut_buckets,
                stochastic_round, rng_seed, idx);
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

// Value form of the momentum recurrence: the ONE canonical expression every
// magnitude/update kernel shares, so nvcc's FMA contraction is identical at
// every site by construction (an inlined copy can contract differently and
// perturb the absmax -- see gefen_magnitude_sumsq_flat_kernel's comment).
// Kernels that stage coeff/grad in registers call this form directly; the
// pointer form below delegates to it.
__device__ __forceinline__ float updated_momentum_val(
    float coeff,
    float old_magnitude,
    float grad_value,
    float beta1
) {
    const float current_m = old_magnitude * coeff;
    return beta1 * current_m + (1.0f - beta1) * grad_value;
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
    const float grad_value = static_cast<float>(grad_view[idx]);
    return updated_momentum_val(coeff, old_magnitude, grad_value, beta1);
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
//   * when nesterov is requested, only the DENSE output is advanced again; the
//     quantized state remains the underlying EMA.  The host expression is two
//     in-place TensorIterator kernels, `m.mul_(beta).add_(g, alpha=1-beta)`.
//     Preserve both scalar-dtype rounds explicitly.  In particular, fp32 needs
//     __fmul_rn to prevent contraction across the first round, and the final
//     add is the same round-to-nearest FMA as TensorIterator.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__device__ __forceinline__ scalar_t gefen_nesterov_momentum(
    scalar_t dense_momentum,
    float grad,
    float beta1,
    float alpha
) {
    const scalar_t rounded_momentum = static_cast<scalar_t>(
        __fmul_rn(static_cast<float>(dense_momentum), beta1));
    return static_cast<scalar_t>(__fmaf_rn(
        alpha, grad, static_cast<float>(rounded_momentum)));
}

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
    float beta1,
    float nesterov_alpha,
    bool nesterov,
    bool stochastic_round,
    uint64_t rng_seed,
    const int64_t* __restrict__ rng_seed_dev
) {
    // Capturable stochastic rounding: read the per-step seed from device
    // memory (a 0-dim int64 step tensor advanced once per optimizer step) so
    // graph replays dither with a fresh seed. nullptr == legacy host seed.
    if (stochastic_round && rng_seed_dev != nullptr) {
        rng_seed = static_cast<uint64_t>(
            __ldg(reinterpret_cast<const long long*>(rng_seed_dev)));
    }
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
        const uint8_t quantized_index = quantize_codebook_index(
            normalized_value, s_codebook, codebook_size, stochastic_round, rng_seed, idx);
        m_sign[idx] = quantized_index;
        // Dense quantized momentum, rounded exactly as the old host
        // `codebook[idx].to(scalar_t).mul_(m_magnitude)` two-step round.
        const scalar_t coeff = static_cast<scalar_t>(s_codebook[static_cast<int>(quantized_index)]);
        if (nesterov) {
            // The explicit multiply also forces the fp32 dense-momentum round
            // that the unfused path gets by storing the emitter output before
            // launching its separate Nesterov kernels.
            const scalar_t dense_momentum = static_cast<scalar_t>(
                __fmul_rn(static_cast<float>(coeff), new_mag));
            momentum_out[idx] = gefen_nesterov_momentum(
                dense_momentum, static_cast<float>(grad_view[idx]), beta1,
                nesterov_alpha);
        } else {
            // Leave the historical non-Nesterov expression untouched.
            momentum_out[idx] = static_cast<scalar_t>(
                static_cast<float>(coeff) * new_mag);
        }
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

// 8-element vector copy for the chunked kernels (block-vmean and factored): single 16B load for
// 2-byte dtypes, two 16B loads for fp32, per-element fallback otherwise
// (double is rejected at dispatch but keep it correct). Callers guarantee
// 16B alignment of src.
template <typename scalar_t>
__device__ __forceinline__ void gefen_load_vec8(
    const scalar_t* __restrict__ src, scalar_t* dst
) {
    if constexpr (sizeof(scalar_t) == 2) {
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
    } else if constexpr (sizeof(scalar_t) == 4) {
        reinterpret_cast<uint4*>(dst)[0] = reinterpret_cast<const uint4*>(src)[0];
        reinterpret_cast<uint4*>(dst)[1] = reinterpret_cast<const uint4*>(src)[1];
    } else {
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            dst[k] = src[k];
        }
    }
}

template <typename scalar_t>
__device__ __forceinline__ void gefen_store_vec8(
    scalar_t* __restrict__ dst, const scalar_t* src
) {
    if constexpr (sizeof(scalar_t) == 2) {
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
    } else if constexpr (sizeof(scalar_t) == 4) {
        reinterpret_cast<uint4*>(dst)[0] = reinterpret_cast<const uint4*>(src)[0];
        reinterpret_cast<uint4*>(dst)[1] = reinterpret_cast<const uint4*>(src)[1];
    } else {
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            dst[k] = src[k];
        }
    }
}

// The chunk width is pinned by the helpers above and below: the vec8 pair
// moves exactly 8 elements and the sign8 pair exactly 8 bytes. Retuning the
// chunk therefore requires new helpers, not a define edit.
#define GEFEN_UPD_CHUNK 8
static_assert(GEFEN_UPD_CHUNK == 8,
              "gefen_load_vec8/gefen_store_vec8 and the sign8 helpers move "
              "exactly 8 elements");

// 8-byte vectorized m_sign transfer (one uint2). Deliberately NOT expressed
// via gefen_load_vec8<uint8_t>: its sizeof==1 branch degrades to scalar
// copies, which would silently devectorize the byte stream.
__device__ __forceinline__ void gefen_load_sign8(
    const uint8_t* __restrict__ src, uint8_t* dst
) {
    const uint2 sv = *reinterpret_cast<const uint2*>(src);
    const uint8_t* ss = reinterpret_cast<const uint8_t*>(&sv);
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        dst[k] = ss[k];
    }
}

__device__ __forceinline__ void gefen_store_sign8(
    uint8_t* __restrict__ dst, const uint8_t* src
) {
    uint2 so;
    uint8_t* sp = reinterpret_cast<uint8_t*>(&so);
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        sp[k] = src[k];
    }
    *reinterpret_cast<uint2*>(dst) = so;
}

// Alignment contract of the vector paths, in one place: grad/p move as uint4
// (16B), m_sign as uint2 (8B). Flat chunked kernels get element-offset
// alignment for free (chunk bases are multiples of GEFEN_UPD_CHUNK); the
// row-strided kernels must ALSO check their per-row start offset (see the
// sumsq-split and factored-stats kernels).
__device__ __forceinline__ bool gefen_aligned16(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}
__device__ __forceinline__ bool gefen_aligned8(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 7) == 0;
}

// Shared chunk frame of the two chunked UPDATE kernels (block-vmean full and
// factored): stage one GEFEN_UPD_CHUNK-wide chunk of grad/m_sign(/p) into
// registers, and write the results back, with the vector fast path and the
// tail/unaligned scalar fallback in ONE place. load_p/store_p mirror the
// kernels' lr != 0 gating: p is neither read nor written when lr == 0.
// p_v/p_out must be alignas(16) at the caller (vec8 contract).
template <typename scalar_t>
__device__ __forceinline__ void gefen_load_update_chunk(
    const scalar_t* __restrict__ grad_view,
    const uint8_t* __restrict__ m_sign,
    const scalar_t* __restrict__ p,
    int64_t base,
    int64_t total_numel,
    bool full,
    bool load_p,
    float* g_f,
    uint8_t* sign_in,
    scalar_t* p_v
) {
    if (full) {
        alignas(16) scalar_t g_v[GEFEN_UPD_CHUNK];
        gefen_load_vec8(&grad_view[base], g_v);
        gefen_load_sign8(&m_sign[base], sign_in);
        if (load_p) {
            gefen_load_vec8(&p[base], p_v);
        }
        #pragma unroll
        for (int k = 0; k < GEFEN_UPD_CHUNK; ++k) {
            g_f[k] = static_cast<float>(g_v[k]);
        }
    } else {
        #pragma unroll
        for (int k = 0; k < GEFEN_UPD_CHUNK; ++k) {
            if (base + k < total_numel) {
                g_f[k] = static_cast<float>(grad_view[base + k]);
                sign_in[k] = m_sign[base + k];
                if (load_p) {
                    p_v[k] = p[base + k];
                }
            }
        }
    }
}

template <typename scalar_t>
__device__ __forceinline__ void gefen_store_update_chunk(
    uint8_t* __restrict__ m_sign,
    scalar_t* __restrict__ p,
    int64_t base,
    int64_t total_numel,
    bool full,
    bool store_p,
    const uint8_t* sign_out,
    const scalar_t* p_out
) {
    if (full) {
        gefen_store_sign8(&m_sign[base], sign_out);
        if (store_p) {
            gefen_store_vec8(&p[base], p_out);
        }
    } else {
        #pragma unroll
        for (int k = 0; k < GEFEN_UPD_CHUNK; ++k) {
            if (base + k < total_numel) {
                m_sign[base + k] = sign_out[k];
                if (store_p) {
                    p[base + k] = p_out[k];
                }
            }
        }
    }
}

// Large period: split each row across blocks_per_row CUDA blocks; block-local
// max + sum trees, then one atomicMax and one atomicAdd per CUDA block.
//
// Each thread owns 8 CONTIGUOUS elements per slice iteration so grad and
// m_sign move as 16B/8B vector transactions (the stride form read 1-2 bytes
// per instruction); tails and unaligned rows take the scalar fallback.
// absmax is order-independent, so the magnitude is bit-identical to the
// stride form. The per-thread sumsq partials compose differently than the
// stride form, which shifts the row Sum(grad^2) at sub-ULP scale. With
// blocks_per_row > 1 that lands in the kernel's pre-existing run-to-run
// atomicAdd nondeterminism (verified by old-vs-old control runs); with
// blocks_per_row == 1 (periods up to threads*64, or when the max_bpr clamp
// bites) the stride form was deterministic, so there this is a
// DETERMINISTIC sub-ULP change to sumsq -> stepsize -> p relative to the old
// kernel. m_magnitude and m_sign are unaffected either way, and a 1-ulp fp32
// stepsize perturbation is far below the documented bf16 training noise.
#define GEFEN_SUMSQ_CHUNK GEFEN_UPD_CHUNK  // same 8-element vec8/sign8 contract

template <typename scalar_t>
__global__ __launch_bounds__(256) void gefen_magnitude_sumsq_split_kernel(
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
    // Vector loads need 16B/8B-aligned base pointers AND a 16B-aligned row
    // start (period % 8 != 0 misaligns every odd row).
    const bool row_aligned = gefen_aligned16(grad_view) &&
                             gefen_aligned8(m_sign) &&
                             ((row_start & 7) == 0);
    float local_absmax = 0.0f;
    float local_sumsq = 0.0f;
    const int64_t chunk_stride =
        static_cast<int64_t>(blocks_per_row) * blockDim.x * GEFEN_SUMSQ_CHUNK;
    for (int64_t o0 = (static_cast<int64_t>(sub) * blockDim.x + threadIdx.x) *
             GEFEN_SUMSQ_CHUNK;
         o0 < period; o0 += chunk_stride) {
        const bool full = row_aligned && (o0 + GEFEN_SUMSQ_CHUNK <= period);
        float g_f[GEFEN_SUMSQ_CHUNK];
        uint8_t s_in[GEFEN_SUMSQ_CHUNK];
        if (full) {
            alignas(16) scalar_t g_v[GEFEN_SUMSQ_CHUNK];
            gefen_load_vec8(&grad_view[row_start + o0], g_v);
            gefen_load_sign8(&m_sign[row_start + o0], s_in);
            #pragma unroll
            for (int k = 0; k < GEFEN_SUMSQ_CHUNK; ++k) {
                g_f[k] = static_cast<float>(g_v[k]);
            }
        } else {
            #pragma unroll
            for (int k = 0; k < GEFEN_SUMSQ_CHUNK; ++k) {
                if (o0 + k < period) {
                    g_f[k] = static_cast<float>(grad_view[row_start + o0 + k]);
                    s_in[k] = m_sign[row_start + o0 + k];
                }
            }
        }
        #pragma unroll
        for (int k = 0; k < GEFEN_SUMSQ_CHUNK; ++k) {
            if (!full && o0 + k >= period) {
                break;
            }
            const float coeff = s_codebook[static_cast<int>(s_in[k])];
            const float updated =
                updated_momentum_val(coeff, old_mag, g_f[k], beta1);
            const float a = fabsf(updated);
            if (a > local_absmax) {
                local_absmax = a;
            }
            local_sumsq += g_f[k] * g_f[k];
        }
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
    float inv_bias_correction_1,
    const float* __restrict__ step_scalars
) {
    // Capturable mode: read the per-step bias-correction reciprocals from the
    // device buffer ([lr, 1/sqrt(bc2), 1/bc1, wd]; slots 1 and 2 are consumed
    // here) so graph replays see fresh values. nullptr == legacy host scalars.
    if (step_scalars != nullptr) {
        inv_sqrt_bias_correction_2 = __ldg(&step_scalars[1]);
        inv_bias_correction_1 = __ldg(&step_scalars[2]);
    }
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
//
// Chunked layout: the block-vmean twin of gefen_update_flat_factored_kernel
// -- same chunk frame (gefen_load/store_update_chunk), same one-divide-per-
// chunk integer carry, same bit-exactness contract (float op order unchanged,
// SR hash keyed on the exact global element index); see the factored twin's
// header for the full rationale. Differences here: a single period carry (no
// row/col tracking) and a precomputed per-block stepsize. KEEP THE TWO CHUNK
// FRAMES IN LOCKSTEP -- a carry/tail fix in one twin applies to the other.
template <typename scalar_t>
__global__ __launch_bounds__(256) void gefen_update_flat_full_kernel(
    scalar_t* __restrict__ p,
    const scalar_t* __restrict__ grad_view,
    uint8_t* __restrict__ m_sign,
    const float* __restrict__ old_magnitude,
    const float* __restrict__ new_magnitude,
    const float* __restrict__ stepsize,  // [num_blocks]; read per chunk even
                                         // when lr == 0 (must always be valid)
    const float* __restrict__ codebook,
    const int16_t* __restrict__ lut,
    int lut_buckets,
    int codebook_size,
    int64_t period,
    int64_t total_numel,
    float beta1,
    float lr,
    float weight_decay_factor,
    bool stochastic_round,
    uint64_t rng_seed,
    const int64_t* __restrict__ rng_seed_dev,
    const float* __restrict__ step_scalars
) {
    // Capturable mode: read lr and the weight-decay factor from the device
    // buffer ([lr, 1/sqrt(bc2), 1/bc1, wd]; slots 0 and 3 are consumed here --
    // the bias-correction slots feed the finalize kernel) so graph replays see
    // fresh values. nullptr == legacy host scalars.
    if (step_scalars != nullptr) {
        lr = __ldg(&step_scalars[0]);
        weight_decay_factor = __ldg(&step_scalars[3]);
    }
    // Capturable stochastic rounding: read the per-step seed from device
    // memory (a 0-dim int64 step tensor advanced once per optimizer step) so
    // graph replays dither with a fresh seed. nullptr == legacy host seed.
    if (stochastic_round && rng_seed_dev != nullptr) {
        rng_seed = static_cast<uint64_t>(
            __ldg(reinterpret_cast<const long long*>(rng_seed_dev)));
    }
    extern __shared__ float s_codebook[];
    for (int i = threadIdx.x; i < codebook_size; i += blockDim.x) {
        s_codebook[i] = codebook[i];
    }
    __syncthreads();
    const bool ptrs_aligned = gefen_aligned16(p) &&
                              gefen_aligned16(grad_view) &&
                              gefen_aligned8(m_sign);

    const int64_t chunk_stride =
        static_cast<int64_t>(gridDim.x) * blockDim.x * GEFEN_UPD_CHUNK;
    for (int64_t base =
             (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
             GEFEN_UPD_CHUNK;
         base < total_numel; base += chunk_stride) {
        // base is a multiple of GEFEN_UPD_CHUNK, so a fully in-bounds chunk
        // is 16B/8B aligned whenever the base pointers are.
        const bool full =
            ptrs_aligned && (base + GEFEN_UPD_CHUNK <= total_numel);
        // ONE divide per chunk; the exact integer carry below replaces the
        // per-element division.
        int64_t blk = base / period;
        int64_t pin = base - blk * period;  // position inside the gefen block

        float g_f[GEFEN_UPD_CHUNK];
        uint8_t sign_in[GEFEN_UPD_CHUNK];
        alignas(16) scalar_t p_v[GEFEN_UPD_CHUNK];
        gefen_load_update_chunk(grad_view, m_sign, p, base, total_numel,
                                full, lr != 0.0f, g_f, sign_in, p_v);

        uint8_t sign_out[GEFEN_UPD_CHUNK];
        alignas(16) scalar_t p_out[GEFEN_UPD_CHUNK];
        float old_mag = old_magnitude[blk];
        float new_mag = new_magnitude[blk];
        float step = stepsize[blk];
        #pragma unroll
        for (int k = 0; k < GEFEN_UPD_CHUNK; ++k) {
            if (!full && base + k >= total_numel) {
                break;
            }
            // Per-element math: identical ops in identical order to the
            // flat form (updated_momentum_val is the same expression the
            // pointer-form helper delegates to).
            const float coeff = s_codebook[static_cast<int>(sign_in[k])];
            const float updated =
                updated_momentum_val(coeff, old_mag, g_f[k], beta1);
            float normalized_value = 0.0f;
            if (new_mag > 0.0f) {
                normalized_value = updated / new_mag;
            }
            const uint8_t quantized_index = quantize_codebook_index_lut(
                normalized_value, s_codebook, codebook_size, lut, lut_buckets,
                stochastic_round, rng_seed, base + k);
            sign_out[k] = quantized_index;
            if (lr != 0.0f) {
                const float quantized_value =
                    s_codebook[static_cast<int>(quantized_index)] * new_mag;
                const float update_value =
                    __fmul_rn(__fmul_rn(quantized_value, step), lr);
                const float p_f = static_cast<float>(p_v[k]);
                if (weight_decay_factor == 1.0f) {
                    p_out[k] = static_cast<scalar_t>(
                        __fsub_rn(p_f, update_value));
                } else {
                    const float decayed = static_cast<float>(
                        static_cast<scalar_t>(
                            __fmul_rn(p_f, weight_decay_factor)));
                    p_out[k] = static_cast<scalar_t>(
                        __fsub_rn(decayed, update_value));
                }
            }
            // Exact integer carry; the reload fires only when a next element
            // exists in this chunk (k+1 bound covers the chunk end, the
            // in-bounds check covers tail chunks).
            if (++pin == period) {
                pin = 0;
                ++blk;
                if (k + 1 < GEFEN_UPD_CHUNK && base + k + 1 < total_numel) {
                    old_mag = old_magnitude[blk];
                    new_mag = new_magnitude[blk];
                    step = stepsize[blk];
                }
            }
        }

        gefen_store_update_chunk(m_sign, p, base, total_numel, full,
                                 lr != 0.0f, sign_out, p_out);
    }
}

// Muon phase 2: quantize the advanced momentum and emit the dense tensor that
// Newton--Schulz consumes.  This is the no-parameter-write sibling of the
// chunked full-update kernel above.  The old emitter assigned one element to
// each thread, paying an int64 idx/period division plus scalar grad/sign/output
// transactions per element and an eight-step full-codebook search.  Each
// thread now owns eight adjacent elements: one divmod plus exact carries,
// vector traffic, and the same bit-exact LUT-narrowed lower_bound used by the
// Gefen update kernels.
//
// The dense output deliberately retains the old two-round contract:
// codebook fp32 -> scalar_t, then scalar_t*fp32 magnitude -> scalar_t.  Do not
// collapse those casts or replace updated/new_mag with reciprocal-multiply;
// either change would perturb the bf16/fp16 Newton--Schulz input.
template <typename scalar_t>
__global__ __launch_bounds__(256) void gefen_momentum_emit_chunked_kernel(
    const scalar_t* __restrict__ grad_view,
    uint8_t* __restrict__ m_sign,
    const float* __restrict__ old_magnitude,
    const float* __restrict__ new_magnitude,
    const float* __restrict__ codebook,
    const int16_t* __restrict__ lut,
    int lut_buckets,
    scalar_t* __restrict__ momentum_out,
    int codebook_size,
    int64_t period,
    int64_t total_numel,
    float beta1,
    float nesterov_alpha,
    bool nesterov,
    bool stochastic_round,
    uint64_t rng_seed,
    const int64_t* __restrict__ rng_seed_dev
) {
    if (stochastic_round && rng_seed_dev != nullptr) {
        rng_seed = static_cast<uint64_t>(
            __ldg(reinterpret_cast<const long long*>(rng_seed_dev)));
    }
    extern __shared__ float s_codebook[];
    for (int i = threadIdx.x; i < codebook_size; i += blockDim.x) {
        s_codebook[i] = codebook[i];
    }
    __syncthreads();

    const bool ptrs_aligned = gefen_aligned16(grad_view) &&
                              gefen_aligned8(m_sign) &&
                              gefen_aligned16(momentum_out);
    const int64_t chunk_stride =
        static_cast<int64_t>(gridDim.x) * blockDim.x * GEFEN_UPD_CHUNK;
    for (int64_t base =
             (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
             GEFEN_UPD_CHUNK;
         base < total_numel; base += chunk_stride) {
        const bool full =
            ptrs_aligned && (base + GEFEN_UPD_CHUNK <= total_numel);
        int64_t blk = base / period;
        int64_t pin = base - blk * period;

        float g_f[GEFEN_UPD_CHUNK];
        uint8_t sign_in[GEFEN_UPD_CHUNK];
        alignas(16) scalar_t unused_out[GEFEN_UPD_CHUNK];
        gefen_load_update_chunk(
            grad_view, m_sign, momentum_out, base, total_numel, full,
            false, g_f, sign_in, unused_out);

        uint8_t sign_out[GEFEN_UPD_CHUNK];
        alignas(16) scalar_t momentum_v[GEFEN_UPD_CHUNK];
        float old_mag = old_magnitude[blk];
        float new_mag = new_magnitude[blk];
        #pragma unroll
        for (int k = 0; k < GEFEN_UPD_CHUNK; ++k) {
            if (!full && base + k >= total_numel) {
                break;
            }
            const float coeff = s_codebook[static_cast<int>(sign_in[k])];
            const float updated =
                updated_momentum_val(coeff, old_mag, g_f[k], beta1);
            float normalized_value = 0.0f;
            if (new_mag > 0.0f) {
                normalized_value = updated / new_mag;
            }
            const uint8_t quantized_index = quantize_codebook_index_lut(
                normalized_value, s_codebook, codebook_size, lut, lut_buckets,
                stochastic_round, rng_seed, base + k);
            sign_out[k] = quantized_index;
            const scalar_t quantized_coeff = static_cast<scalar_t>(
                s_codebook[static_cast<int>(quantized_index)]);
            if (nesterov) {
                const scalar_t dense_momentum = static_cast<scalar_t>(
                    __fmul_rn(static_cast<float>(quantized_coeff), new_mag));
                momentum_v[k] = gefen_nesterov_momentum(
                    dense_momentum, g_f[k], beta1, nesterov_alpha);
            } else {
                // Leave the historical non-Nesterov expression untouched.
                momentum_v[k] = static_cast<scalar_t>(
                    static_cast<float>(quantized_coeff) * new_mag);
            }

            if (++pin == period) {
                pin = 0;
                ++blk;
                if (k + 1 < GEFEN_UPD_CHUNK && base + k + 1 < total_numel) {
                    old_mag = old_magnitude[blk];
                    new_mag = new_magnitude[blk];
                }
            }
        }
        gefen_store_update_chunk(
            m_sign, momentum_out, base, total_numel, full, true,
            sign_out, momentum_v);
    }
}

// Factored-mode phase 1: ONE pass over grad + m_sign computing, per element,
// (a) the updated-momentum absmax per gefen block (atomic max, order-free) and
// (b) the row/col sum-of-squares of the RAW gradient for the Adafactor EMAs.
// Layout: each CUDA block owns a (TR rows x TC cols) matrix tile; warp w scans
// row (tile_row + w), each lane owning LC CONTIGUOUS columns so the grad and
// m_sign reads are 16-byte / 8-byte vector loads (one 512B transaction per
// warp) instead of per-element scalar loads. Row sums reduce via warp
// shuffles (one atomicAdd per row per tile); col sums accumulate in shared
// partials (one atomicAdd per col per tile). The atomicAdd accumulation makes
// v_row/v_col run-to-run nondeterministic at sub-ULP scale -- the same
// documented, convergence-neutral property as the v2-full atomic vmean.
//
// The per-block magnitude absmax accumulates in a REGISTER while the lane
// stays inside one gefen block (one idx/period divide per lane per row strip,
// advanced by exact integer carries) and flushes with a single warp-reduced
// atomic per strip when every lane still holds the same block -- the common
// case, since the period search picks huge blocks (often numel/2). A block
// boundary inside a lane's strip flushes that lane early; mixed-block strips
// flush per lane. absmax is order-independent, so the flushing schedule
// cannot change the result: bit-identical to the per-element atomic form.
#define GEFEN_FACTORED_TILE_COLS 256
#define GEFEN_FACTORED_TILE_ROWS 8
#define GEFEN_FACTORED_LANE_COLS 8
static_assert(GEFEN_FACTORED_LANE_COLS == GEFEN_UPD_CHUNK,
              "lane strips ride the same 8-element vec8/sign8 helpers");

template <typename scalar_t>
__global__ __launch_bounds__(256) void gefen_factored_stats_kernel(
    const scalar_t* __restrict__ grad_view,
    const uint8_t* __restrict__ m_sign,
    const float* __restrict__ old_magnitude,
    const float* __restrict__ codebook,
    float* __restrict__ new_magnitude,
    float* __restrict__ row_sq,
    float* __restrict__ col_sq,
    int codebook_size,
    int64_t period,
    int64_t rows,
    int64_t cols,
    float beta1
) {
    // Shared: [codebook][TC col partials]
    extern __shared__ float smem[];
    float* s_codebook = smem;
    float* s_col = smem + codebook_size;
    for (int i = threadIdx.x; i < codebook_size; i += blockDim.x) {
        s_codebook[i] = codebook[i];
    }
    for (int i = threadIdx.x; i < GEFEN_FACTORED_TILE_COLS; i += blockDim.x) {
        s_col[i] = 0.0f;
    }
    __syncthreads();

    // Vector loads additionally need 16B/8B-aligned base pointers; fresh
    // optimizer-state allocations always are, but a caller handing in offset
    // views must degrade to the scalar path, not misalign.
    const bool ptrs_aligned =
        gefen_aligned16(grad_view) && gefen_aligned8(m_sign);
    const int64_t col0 =
        static_cast<int64_t>(blockIdx.x) * GEFEN_FACTORED_TILE_COLS;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;

    // Grid-stride over row tiles: gridDim.y is capped at 65535 by the host
    // launch (the CUDA grid.y limit), so tall tensors (> 65535 * TILE_ROWS
    // rows) are covered by striding. The shared col partials for this fixed
    // col0 accumulate across all strided row tiles and flush once at the end.
    const int64_t row_tile_stride =
        static_cast<int64_t>(gridDim.y) * GEFEN_FACTORED_TILE_ROWS;
    for (int64_t row0 = static_cast<int64_t>(blockIdx.y) * GEFEN_FACTORED_TILE_ROWS;
         row0 < rows; row0 += row_tile_stride) {
    // Each warp owns rows row0+warp, row0+warp+warps_per_block, ...
    for (int64_t r = row0 + warp; r < row0 + GEFEN_FACTORED_TILE_ROWS && r < rows;
         r += warps_per_block) {
        const int64_t row_base = r * cols;
        const int64_t c_lo =
            col0 + static_cast<int64_t>(lane) * GEFEN_FACTORED_LANE_COLS;
        float row_acc = 0.0f;
        // Running per-block magnitude max: ONE divide per lane per strip;
        // the block index then advances by integer carries. blk < 0 marks a
        // fully out-of-bounds lane (edge tile past the last column).
        int64_t blk = -1;
        int64_t pin = 0;
        float blk_max = 0.0f;
        if (c_lo < cols) {
            const int64_t idx0 = row_base + c_lo;
            blk = idx0 / period;
            pin = idx0 - blk * period;
        }
        // Vector path needs the whole lane strip in bounds AND 16B-aligned
        // (row_base % 8 can be nonzero when cols % 8 != 0).
        const bool full = ptrs_aligned &&
                          (c_lo + GEFEN_FACTORED_LANE_COLS <= cols) &&
                          (((row_base + c_lo) & 7) == 0);
        float g_f[GEFEN_FACTORED_LANE_COLS];
        uint8_t s_in[GEFEN_FACTORED_LANE_COLS];
        if (full) {
            alignas(16) scalar_t g_v[GEFEN_FACTORED_LANE_COLS];
            gefen_load_vec8(&grad_view[row_base + c_lo], g_v);
            gefen_load_sign8(&m_sign[row_base + c_lo], s_in);
            #pragma unroll
            for (int k = 0; k < GEFEN_FACTORED_LANE_COLS; ++k) {
                g_f[k] = static_cast<float>(g_v[k]);
            }
        } else {
            #pragma unroll
            for (int k = 0; k < GEFEN_FACTORED_LANE_COLS; ++k) {
                if (c_lo + k < cols) {
                    g_f[k] = static_cast<float>(grad_view[row_base + c_lo + k]);
                    s_in[k] = m_sign[row_base + c_lo + k];
                }
            }
        }
        float old_mag = (blk >= 0) ? old_magnitude[blk] : 0.0f;
        #pragma unroll
        for (int k = 0; k < GEFEN_FACTORED_LANE_COLS; ++k) {
            if (!full && c_lo + k >= cols) {
                break;
            }
            const float g = g_f[k];
            const float g2 = g * g;
            row_acc += g2;
            // Swizzled [k][lane] layout: at unrolled step k every lane hits
            // a distinct bank (slot = k*32 + lane); the natural lane-major
            // layout (slot = lane*8 + k) would be an 8-way bank conflict per
            // instruction. The flush below undoes the swizzle.
            atomicAdd(&s_col[k * 32 + lane], g2);
            const float coeff = s_codebook[static_cast<int>(s_in[k])];
            const float updated = updated_momentum_val(coeff, old_mag, g, beta1);
            blk_max = fmaxf(blk_max, fabsf(updated));
            if (++pin == period) {
                // Gefen-block boundary inside the strip: flush this lane.
                atomic_max_nonneg(&new_magnitude[blk], blk_max);
                blk_max = 0.0f;
                pin = 0;
                ++blk;
                if (k + 1 < GEFEN_FACTORED_LANE_COLS && c_lo + k + 1 < cols) {
                    old_mag = old_magnitude[blk];
                }
            }
        }
        // Warp-reduce the row partial; one global atomic per row per tile.
        for (int offset = 16; offset > 0; offset >>= 1) {
            row_acc += __shfl_down_sync(0xffffffffu, row_acc, offset);
        }
        if (lane == 0) {
            atomicAdd(&row_sq[r], row_acc);
        }
        // Magnitude flush. All 32 lanes reach here (the strip loops rejoin
        // and the row loop bound is warp-uniform). Fast path: every lane
        // still holds the SAME gefen block -> warp-reduce the max, one
        // atomic per warp-strip. NaN handling matches atomic_max_nonneg:
        // fmaxf returns the non-NaN operand, so a NaN |updated| can never
        // win the reduction, exactly as it can never win the CAS.
        const int64_t blk_lo = __shfl_sync(0xffffffffu, blk, 0);
        const bool uniform = __all_sync(0xffffffffu, blk == blk_lo && blk >= 0);
        if (uniform) {
            float m = blk_max;
            for (int offset = 16; offset > 0; offset >>= 1) {
                m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, offset));
            }
            if (lane == 0) {
                atomic_max_nonneg(&new_magnitude[blk_lo], m);
            }
        } else if (blk >= 0) {
            atomic_max_nonneg(&new_magnitude[blk], blk_max);
        }
    }
    }
    __syncthreads();

    // One global atomic per column per CUDA block (accumulated across all of
    // this block's strided row tiles).
    for (int i = threadIdx.x; i < GEFEN_FACTORED_TILE_COLS; i += blockDim.x) {
        // Undo the [k][lane] swizzle: slot i = k*32 + lane holds column
        // col0 + lane*LC + k.
        const int k = i >> 5;
        const int ln = i & 31;
        const int64_t c =
            col0 + static_cast<int64_t>(ln) * GEFEN_FACTORED_LANE_COLS + k;
        if (c < cols && s_col[i] != 0.0f) {
            atomicAdd(&col_sq[c], s_col[i]);
        }
    }
}

// Factored-second-moment (Adafactor-style) phase 2 for 2D params: quantize +
// apply with a PER-ELEMENT stepsize computed in registers from the row/col
// grad^2 EMAs, V_ij ~= v_row[i] * v_col[j] / mean(v_row), instead of the
// per-block vmean stepsize. Everything else (momentum recompute, LUT-narrowed
// quantize, m_sign write, weight-decay-folded p write) is op-identical to
// gefen_update_flat_full_kernel. mean(v_row) arrives as a 1-element device
// tensor so the host never synchronizes; v_row/v_col are tiny and served from
// L1/L2 via __ldg. This kernel is the canonical numerics for factored_v_2d --
// the decomposed python fallback matches it within float tolerance, not
// bit-for-bit (different op associations).
//
// Chunked layout: each thread owns GEFEN_UPD_CHUNK (8) CONTIGUOUS elements
// per grid-stride iteration, so (a) the two per-element int64 divisions of
// the flat form (idx/period, idx/cols -- ~40+ cycles each, the dominant ALU
// cost at the huge periods the search picks) collapse to ONE divmod pair per
// chunk advanced by exact integer carries, and (b) grad/p/m_sign move as
// 16B/8B vector transactions. Per-element float ops and their order are
// UNCHANGED, and the SR hash still keys on the exact global element index,
// so every output (p, m_sign, magnitude writeback) is bit-identical to the
// flat form for both rounding modes. Tails and unaligned bases take an
// in-kernel scalar fallback. __launch_bounds__(256) pins the block size the
// host launches (helps the register allocator; the kernel uses ~73 regs).
// Twin: gefen_update_flat_full_kernel shares this chunk frame -- KEEP THE
// TWO IN LOCKSTEP; a carry/tail fix in one twin applies to the other.
template <typename scalar_t>
__global__ __launch_bounds__(256) void gefen_update_flat_factored_kernel(
    scalar_t* __restrict__ p,
    const scalar_t* __restrict__ grad_view,
    uint8_t* __restrict__ m_sign,
    const float* __restrict__ old_magnitude,
    const float* __restrict__ new_magnitude,
    const float* __restrict__ v_row,
    const float* __restrict__ v_col,
    const float* __restrict__ mean_v_row,  // [1], device-computed (see below)
    int64_t mean_sum_rows,  // 0: mean_v_row IS the mean; >0: it is
                            // Sum(v_row) and the mean is sum/rows (capturable
                            // path, accumulated by the EMA launch)
    const float* __restrict__ codebook,
    const int16_t* __restrict__ lut,
    int lut_buckets,
    int codebook_size,
    int64_t period,
    int64_t cols,
    int64_t total_numel,
    float beta1,
    float lr,
    float eps,
    float inv_bias_correction_2,
    float inv_bias_correction_1,
    float weight_decay_factor,
    bool stochastic_round,
    uint64_t rng_seed,
    const int64_t* __restrict__ rng_seed_dev,
    const float* __restrict__ step_scalars,
    float* __restrict__ mag_writeback
) {
    // Capturable mode: the per-step-varying scalars live in device memory so a
    // captured graph replays fresh values instead of the frozen host args.
    // Layout: [lr, 1/bc2, 1/bc1, wd_factor] (note slot 1 is 1/bc2, NOT
    // 1/sqrt(bc2), on this factored path). nullptr == legacy host scalars.
    //
    // mag_writeback (capturable only): the persistent m_magnitude state, which
    // this kernel then updates IN PLACE (one write per gefen block) instead of
    // the host launching a separate m_magnitude.copy_(new_magnitude) afterward
    // -- on a many-param model those per-param copy launches are a measurable
    // slice of a replayed graph. Safe because in that mode `old_magnitude`
    // points at the scratch STAGED COPY of the old magnitudes (written by the
    // EMA launch before this kernel), so no thread reads the buffer being
    // overwritten. nullptr == legacy flow (old_magnitude IS m_magnitude; the
    // host set_()s the state onto new_magnitude afterwards, zero launches).
    if (step_scalars != nullptr) {
        lr = __ldg(&step_scalars[0]);
        inv_bias_correction_2 = __ldg(&step_scalars[1]);
        inv_bias_correction_1 = __ldg(&step_scalars[2]);
        weight_decay_factor = __ldg(&step_scalars[3]);
    }
    // Capturable stochastic rounding: read the per-step seed from device
    // memory (a 0-dim int64 step tensor advanced once per optimizer step) so
    // graph replays dither with a fresh seed. nullptr == legacy host seed.
    if (stochastic_round && rng_seed_dev != nullptr) {
        rng_seed = static_cast<uint64_t>(
            __ldg(reinterpret_cast<const long long*>(rng_seed_dev)));
    }
    extern __shared__ float s_codebook[];
    for (int i = threadIdx.x; i < codebook_size; i += blockDim.x) {
        s_codebook[i] = codebook[i];
    }
    __syncthreads();
    float mean_r = __ldg(mean_v_row);
    if (mean_sum_rows > 0) {
        mean_r = __fdiv_rn(mean_r, static_cast<float>(mean_sum_rows));
    }
    const float inv_mean_r = 1.0f / fmaxf(mean_r, 1.1754944e-38f);
    const bool ptrs_aligned = gefen_aligned16(p) &&
                              gefen_aligned16(grad_view) &&
                              gefen_aligned8(m_sign);

    const int64_t chunk_stride =
        static_cast<int64_t>(gridDim.x) * blockDim.x * GEFEN_UPD_CHUNK;
    for (int64_t base =
             (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
             GEFEN_UPD_CHUNK;
         base < total_numel; base += chunk_stride) {
        // base is a multiple of GEFEN_UPD_CHUNK, so a fully in-bounds chunk
        // is 16B/8B aligned whenever the base pointers are.
        const bool full =
            ptrs_aligned && (base + GEFEN_UPD_CHUNK <= total_numel);
        // ONE divmod pair per chunk; exact integer carries below replace the
        // per-element divisions.
        int64_t row = base / cols;
        int64_t col = base - row * cols;
        int64_t blk = base / period;
        int64_t pin = base - blk * period;  // position inside the gefen block

        float g_f[GEFEN_UPD_CHUNK];
        uint8_t sign_in[GEFEN_UPD_CHUNK];
        alignas(16) scalar_t p_v[GEFEN_UPD_CHUNK];
        gefen_load_update_chunk(grad_view, m_sign, p, base, total_numel,
                                full, lr != 0.0f, g_f, sign_in, p_v);

        uint8_t sign_out[GEFEN_UPD_CHUNK];
        alignas(16) scalar_t p_out[GEFEN_UPD_CHUNK];
        float old_mag = old_magnitude[blk];
        float new_mag = new_magnitude[blk];
        float vr = __ldg(&v_row[row]);
        #pragma unroll
        for (int k = 0; k < GEFEN_UPD_CHUNK; ++k) {
            if (!full && base + k >= total_numel) {
                break;
            }
            if (mag_writeback != nullptr && pin == 0) {
                // Exactly one element per gefen block writes the new
                // magnitude into the persistent state (see the arg comment
                // above); pin == 0 is the chunked form of
                // idx == block_idx * period.
                mag_writeback[blk] = new_mag;
            }
            // Per-element math: identical ops in identical order to the
            // flat form (updated_momentum_val is the same expression the
            // pointer-form helper delegates to).
            const float coeff = s_codebook[static_cast<int>(sign_in[k])];
            const float updated =
                updated_momentum_val(coeff, old_mag, g_f[k], beta1);
            float normalized_value = 0.0f;
            if (new_mag > 0.0f) {
                normalized_value = updated / new_mag;
            }
            const uint8_t quantized_index = quantize_codebook_index_lut(
                normalized_value, s_codebook, codebook_size, lut, lut_buckets,
                stochastic_round, rng_seed, base + k);
            sign_out[k] = quantized_index;
            if (lr != 0.0f) {
                const float v_hat = __fmul_rn(
                    __fmul_rn(vr, __ldg(&v_col[col])), inv_mean_r);
                float h = __fsqrt_rn(__fmul_rn(v_hat, inv_bias_correction_2));
                h = __fadd_rn(h, eps);
                const float stepsize_val =
                    __fmul_rn(__frcp_rn(h), inv_bias_correction_1);
                const float quantized_value =
                    s_codebook[static_cast<int>(quantized_index)] * new_mag;
                const float update_value =
                    __fmul_rn(__fmul_rn(quantized_value, stepsize_val), lr);
                const float p_f = static_cast<float>(p_v[k]);
                if (weight_decay_factor == 1.0f) {
                    p_out[k] = static_cast<scalar_t>(
                        __fsub_rn(p_f, update_value));
                } else {
                    const float decayed = static_cast<float>(
                        static_cast<scalar_t>(
                            __fmul_rn(p_f, weight_decay_factor)));
                    p_out[k] = static_cast<scalar_t>(
                        __fsub_rn(decayed, update_value));
                }
            }
            // Exact integer carries replace the per-element divisions. The
            // reloads fire only when a next element exists IN THIS CHUNK (the
            // k+1 bound covers the chunk end -- without it the tensor's final
            // full chunk would express reads of v_row[rows] /
            // *_magnitude[num_blocks], dead loads that only compiler DCE
            // removed); the in-bounds check covers tail chunks.
            if (++col == cols) {
                col = 0;
                ++row;
                if (k + 1 < GEFEN_UPD_CHUNK && base + k + 1 < total_numel) {
                    vr = __ldg(&v_row[row]);
                }
            }
            if (++pin == period) {
                pin = 0;
                ++blk;
                if (k + 1 < GEFEN_UPD_CHUNK && base + k + 1 < total_numel) {
                    old_mag = old_magnitude[blk];
                    new_mag = new_magnitude[blk];
                }
            }
        }

        gefen_store_update_chunk(m_sign, p, base, total_numel, full,
                                 lr != 0.0f, sign_out, p_out);
    }
}

// Fused v_row/v_col EMA advance for the factored launcher: one launch replaces
// the six aten launches of
//     v_row.mul_(beta2).add_(row_sq.div_(cols), 1 - beta2);
//     v_col.mul_(beta2).add_(col_sq.div_(rows), 1 - beta2);
// BIT-IDENTICALLY. The aten chain lowers, per element, to exactly
//     t  = x * (1.0f / (float)n)      (div-by-scalar is a reciprocal multiply
//                                      with the reciprocal formed in fp32)
//     vb = v * (float)beta2           (rounded to memory by mul_)
//     v' = fmaf((float)(1-beta2), t, vb)   (CUDAFunctor_add's a + alpha*b,
//                                      contracted to a single fma by nvcc)
// and the intrinsics below pin that exact rounding sequence (verified
// elementwise-equal against the aten chain over randomized magnitudes from
// denormal to 1e10, including the fma-vs-mul+add distinction, which DOES
// differ). Both vectors ride one grid: [0, rows) -> v_row, [rows, rows+cols)
// -> v_col.
//
// The optional tail range [rows+cols, rows+cols+mag_blocks) additionally
// stages a verbatim copy of the old per-block magnitudes (mag_src ->
// mag_copy) for the capturable factored path: the phase-2 kernel then reads
// the old magnitudes from the copy and writes the new ones straight into the
// persistent m_magnitude state, eliminating the separate per-param
// m_magnitude.copy_ launch. The copy rides this (already tiny) launch for
// free; mag_blocks == 0 skips it.
//
// vrow_sum (optional, capturable only): a zero-initialized 1-elem accumulator
// that receives Sum(v_row_new) via one block-reduced atomicAdd per CUDA block.
// The phase-2 kernel derives mean(v_row) from it in registers, replacing the
// per-param at::mean launch. The atomicAdd ordering makes the sum run-to-run
// nondeterministic at ulp level -- the same class of nondeterminism the stats
// kernel's row/col grad^2 atomics already have -- which is why the
// capturable=False path keeps at::mean (bit-exact history) and does not pass
// this pointer.
__global__ void gefen_factored_v_ema_kernel(
    float* __restrict__ v_row,
    float* __restrict__ v_col,
    const float* __restrict__ row_sq,
    const float* __restrict__ col_sq,
    int64_t rows,
    int64_t cols,
    float beta2,
    float alpha,
    float inv_cols,
    float inv_rows,
    const float* __restrict__ mag_src,
    float* __restrict__ mag_copy,
    int64_t mag_blocks,
    float* __restrict__ vrow_sum
) {
    extern __shared__ float s_partials[];
    float local_sum = 0.0f;
    const int64_t total = rows + cols + mag_blocks;
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
         i < total;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        if (i < rows) {
            const float t = __fmul_rn(row_sq[i], inv_cols);
            const float vb = __fmul_rn(v_row[i], beta2);
            const float v_new = __fmaf_rn(alpha, t, vb);
            v_row[i] = v_new;
            local_sum += v_new;
        } else if (i < rows + cols) {
            const int64_t j = i - rows;
            const float t = __fmul_rn(col_sq[j], inv_rows);
            const float vb = __fmul_rn(v_col[j], beta2);
            v_col[j] = __fmaf_rn(alpha, t, vb);
        } else {
            const int64_t b = i - rows - cols;
            mag_copy[b] = mag_src[b];
        }
    }
    if (vrow_sum != nullptr) {
        // Block-reduce the v_row partial sums; ONE global atomic per block.
        s_partials[threadIdx.x] = local_sum;
        __syncthreads();
        for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            if (threadIdx.x < static_cast<unsigned int>(offset)) {
                s_partials[threadIdx.x] += s_partials[threadIdx.x + offset];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0 && s_partials[0] != 0.0f) {
            atomicAdd(vrow_sum, s_partials[0]);
        }
    }
}

// Validate + unwrap the optional capturable device-scalar buffer. The kernels
// read 4 fp32 values ([lr, inv(_sqrt)_bc2, inv_bc1, wd_factor]) straight from
// this pointer, so it must be a contiguous fp32 CUDA tensor with >= 4 elements
// on p's device. nullopt / undefined -> nullptr (legacy host-scalar path).
const float* resolve_step_scalars_ptr(
    const c10::optional<at::Tensor>& step_scalars,
    const at::Tensor& p
) {
    if (!step_scalars.has_value() || !step_scalars->defined()) {
        return nullptr;
    }
    const at::Tensor& ss = *step_scalars;
    if (!ss.is_cuda() || ss.device() != p.device()) {
        throw std::invalid_argument(
            "Expected step_scalars on the same CUDA device as p.");
    }
    if (!ss.is_contiguous() || ss.scalar_type() != at::kFloat) {
        throw std::invalid_argument(
            "Expected step_scalars to be contiguous float32.");
    }
    if (ss.numel() < 4) {
        throw std::invalid_argument(
            "Expected step_scalars to have at least 4 elements.");
    }
    return ss.data_ptr<float>();
}

// Validate + unwrap the optional device-side stochastic-rounding seed. The
// kernels read ONE int64 from this pointer at launch (the optimizer's step
// count, advanced in place once per step by the capturable python path), so it
// must be a contiguous int64 CUDA tensor with >= 1 element on the reference
// tensor's device. nullopt / undefined -> nullptr (legacy host-seed path,
// bit-identical).
const int64_t* resolve_seed_dev_ptr(
    const c10::optional<at::Tensor>& seed_dev,
    const at::Tensor& ref
) {
    if (!seed_dev.has_value() || !seed_dev->defined()) {
        return nullptr;
    }
    const at::Tensor& sd = *seed_dev;
    if (!sd.is_cuda() || sd.device() != ref.device()) {
        throw std::invalid_argument(
            "Expected seed_dev on the same CUDA device as the updated tensor.");
    }
    if (!sd.is_contiguous() || sd.scalar_type() != at::kLong) {
        throw std::invalid_argument(
            "Expected seed_dev to be a contiguous int64 tensor.");
    }
    if (sd.numel() < 1) {
        throw std::invalid_argument(
            "Expected seed_dev to have at least 1 element.");
    }
    return sd.data_ptr<int64_t>();
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
    // Packed 4-bit index mode is unsupported: its only call site always passed
    // packed_indices=false, and the aligned-down 32-bit CAS in the packed
    // store can touch memory outside the logical tensor extent. Reject it hard
    // rather than route into that path.
    if (packed_indices) {
        throw std::invalid_argument("packed index mode is unsupported.");
    }
    if (!p.is_cuda() || !grad_view.is_cuda() || !m_sign.is_cuda() || !m_magnitude.is_cuda() || !stepsize.is_cuda() || !codebook.is_cuda()) {
        throw std::invalid_argument("Expected all tensors to be on CUDA.");
    }
    // All tensors are dereferenced on p's device; reject cross-device inputs the
    // is_cuda() checks above would otherwise let through (CUDAGuard only selects
    // the launch device, it does not relocate the operands).
    if (grad_view.device() != p.device() || m_sign.device() != p.device() ||
        m_magnitude.device() != p.device() || stepsize.device() != p.device() ||
        codebook.device() != p.device()) {
        throw std::invalid_argument("Expected all tensors on the same device as p.");
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
    if (period <= 0) {
        throw std::invalid_argument("Expected grad_view to have a positive period.");
    }
    // Empty param (num_blocks == 0 -> total_numel == 0): nothing to update, and a
    // grid.x of 0 is an invalid launch. Bail before the launch. (v1-full already
    // guards this via its period/total_numel checks.)
    if (num_blocks == 0) {
        return;
    }
    // these two may create additional memory footprint.
    const int threads = choose_threads(period);
    const dim3 grid(static_cast<unsigned int>(num_blocks));
    const dim3 block(static_cast<unsigned int>(threads));
    // Shared holds the staged codebook (<=256) plus the per-thread reduction max.
    const size_t shared_bytes =
        (static_cast<size_t>(threads) + static_cast<size_t>(codebook.numel())) * sizeof(float);

    GEFEN_DISPATCH_FLOAT_HALF_BF16(
        p.scalar_type(),
        "automatic_gefen_fused_update_cuda",
        [&] {
            automatic_gefen_fused_update_kernel<scalar_t><<<grid, block, shared_bytes, c10::cuda::getCurrentCUDAStream()>>>(
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
    at::Tensor lut,
    bool packed_indices,
    double beta1,
    double beta2,
    double lr,
    double eps,
    double inv_sqrt_bias_correction_2,
    double inv_bias_correction_1,
    double weight_decay_factor,
    bool stochastic_round,
    int64_t rng_seed,
    c10::optional<at::Tensor> step_scalars,
    c10::optional<at::Tensor> seed_dev
) {
    // Packed 4-bit index mode is unsupported (see the v1-legacy update): the
    // only call site always passes false, and the packed store's aligned-down
    // 32-bit CAS can touch memory outside the logical tensor extent.
    if (packed_indices) {
        throw std::invalid_argument("packed index mode is unsupported.");
    }
    if (!p.is_cuda() || !grad_view.is_cuda() || !m_sign.is_cuda() || !m_magnitude.is_cuda() || !vmean.is_cuda() || !codebook.is_cuda()) {
        throw std::invalid_argument("Expected all tensors to be on CUDA.");
    }
    // Optional capturable device-scalar buffer: when present the kernel reads
    // the per-step-varying scalars (lr / bias-correction reciprocals / weight-
    // decay factor) from it instead of the host args, so a captured CUDA graph
    // replays fresh values. Null -> bit-identical legacy host-scalar path.
    const float* step_scalars_dev = resolve_step_scalars_ptr(step_scalars, p);
    // Optional capturable device-side stochastic-rounding seed: when present
    // (and stochastic_round is set) the kernel reads the seed from it instead
    // of the host rng_seed, so replays dither with fresh per-step seeds.
    const int64_t* seed_dev_ptr = resolve_seed_dev_ptr(seed_dev, p);
    // Optional search LUT (empty tensor -> full-range binary search). Built on
    // the host once per (frozen) codebook; int16 [buckets + 1] on p's device.
    const bool use_lut = lut.numel() > 0;
    if (use_lut) {
        if (!lut.is_cuda() || lut.device() != p.device()) {
            throw std::invalid_argument("Expected lut on the same device as p.");
        }
        if (!lut.is_contiguous() || lut.scalar_type() != at::kShort) {
            throw std::invalid_argument("Expected lut to be contiguous int16.");
        }
        if (lut.numel() < 2) {
            throw std::invalid_argument("Expected lut to have at least 2 entries.");
        }
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
    // Empty param (num_blocks == 0 -> total_numel == 0): nothing to update, and
    // grid_blocks would round down to 0 -> dim3 grid(0), an invalid launch. Bail
    // before building the grid (matches the v1-legacy update path guard).
    if (num_blocks == 0) {
        return;
    }

    const int threads = choose_threads(period);
    // Multi-row packing: host ~256 threads (>= 2 warps) per CUDA block by
    // stacking 256/threads rows, so small-period shapes stop launching 1-warp
    // blocks and the codebook is staged once per ~256 threads instead of once
    // per row. threads is capped at 256, so rows >= 1 always.
    int rows = 256 / threads;
    if (rows < 1) {
        rows = 1;
    }
    const int64_t grid_blocks = (num_blocks + rows - 1) / rows;
    const dim3 grid(static_cast<unsigned int>(grid_blocks));
    const dim3 block(static_cast<unsigned int>(threads), static_cast<unsigned int>(rows));
    // Shared holds the staged codebook plus two per-thread reduction buffers
    // (absmax and sum-of-squares) for every hosted row.
    const size_t shared_bytes =
        (static_cast<size_t>(codebook.numel()) +
         2 * static_cast<size_t>(threads) * static_cast<size_t>(rows)) * sizeof(float);

    GEFEN_DISPATCH_FLOAT_HALF_BF16(
        p.scalar_type(),
        "automatic_gefen_fused_full_update_cuda",
        [&] {
            automatic_gefen_fused_full_update_kernel<scalar_t><<<grid, block, shared_bytes, c10::cuda::getCurrentCUDAStream()>>>(
                p.data_ptr<scalar_t>(),
                grad_view.data_ptr<scalar_t>(),
                m_sign.data_ptr<uint8_t>(),
                m_magnitude.data_ptr<float>(),
                vmean.data_ptr<float>(),
                codebook.data_ptr<float>(),
                use_lut ? lut.data_ptr<int16_t>() : nullptr,
                use_lut ? static_cast<int>(lut.numel() - 1) : 0,
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
                static_cast<float>(weight_decay_factor),
                stochastic_round,
                static_cast<uint64_t>(rng_seed),
                seed_dev_ptr,
                step_scalars_dev
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
    // All tensors are dereferenced on p's device; reject cross-device inputs the
    // is_cuda() checks above would otherwise let through (CUDAGuard only selects
    // the launch device, it does not relocate the operands).
    if (grad_view.device() != p.device() || m_sign.device() != p.device() ||
        m_magnitude.device() != p.device() || stepsize.device() != p.device() ||
        codebook.device() != p.device()) {
        throw std::invalid_argument("Expected all tensors on the same device as p.");
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
    GEFEN_DISPATCH_FLOAT_HALF_BF16(
        p.scalar_type(),
        "gefen_v2_magnitude", [&] {
            if (period <= 2048) {
                int64_t nblocks = (total_numel + threads - 1) / threads;
                if (nblocks > flat_blocks_cap) nblocks = flat_blocks_cap;
                if (nblocks < 1) nblocks = 1;
                gefen_magnitude_flat_kernel<scalar_t><<<static_cast<unsigned int>(nblocks), threads, codebook_bytes, c10::cuda::getCurrentCUDAStream()>>>(
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
                gefen_magnitude_split_kernel<scalar_t><<<static_cast<unsigned int>(grid), threads, shmem, c10::cuda::getCurrentCUDAStream()>>>(
                    grad_view.data_ptr<scalar_t>(), m_sign.data_ptr<uint8_t>(),
                    m_magnitude.data_ptr<float>(), codebook.data_ptr<float>(),
                    new_magnitude.data_ptr<float>(), codebook_size, period, num_blocks, blocks_per_row,
                    static_cast<float>(beta1));
            }
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Phase 2: quantize + parameter update (elementwise).
    GEFEN_DISPATCH_FLOAT_HALF_BF16(
        p.scalar_type(),
        "gefen_v2_update", [&] {
            int64_t nblocks = (total_numel + threads - 1) / threads;
            if (nblocks > flat_blocks_cap) nblocks = flat_blocks_cap;
            if (nblocks < 1) nblocks = 1;
            gefen_update_flat_kernel<scalar_t><<<static_cast<unsigned int>(nblocks), threads, codebook_bytes, c10::cuda::getCurrentCUDAStream()>>>(
                p.data_ptr<scalar_t>(), grad_view.data_ptr<scalar_t>(), m_sign.data_ptr<uint8_t>(),
                m_magnitude.data_ptr<float>(), new_magnitude.data_ptr<float>(),
                stepsize.data_ptr<float>(), codebook.data_ptr<float>(), codebook_size,
                period, total_numel, static_cast<float>(beta1), static_cast<float>(lr));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    m_magnitude.copy_(new_magnitude);
}

void gefen_factored_update_cuda(
    at::Tensor p,
    at::Tensor grad_view,
    at::Tensor m_sign,
    at::Tensor m_magnitude,
    at::Tensor v_row,
    at::Tensor v_col,
    at::Tensor codebook,
    at::Tensor lut,
    int64_t cols,
    double beta1,
    double beta2,
    double lr,
    double eps,
    double inv_bias_correction_2,
    double inv_bias_correction_1,
    double weight_decay_factor,
    bool stochastic_round,
    int64_t rng_seed,
    c10::optional<at::Tensor> step_scalars,
    c10::optional<at::Tensor> seed_dev
) {
    // Factored-second-moment (Adafactor-style) fused update for 2D params.
    // Phase 1 is the combined stats kernel: ONE pass over grad + m_sign
    // computes the per-block absmax AND the raw row/col grad^2 sums; the
    // v_row/v_col EMAs are then advanced IN PLACE here as tiny device ops and
    // their mean stays on device (no host sync). Phase 2 quantizes
    // (LUT-narrowed) + applies with the per-element stepsize
    // V_ij ~= v_row[i] * v_col[j] / mean(v_row) computed in registers -- no
    // vmean state and no full-size temporaries anywhere in the step.
    if (!p.is_cuda() || !grad_view.is_cuda() || !m_sign.is_cuda() ||
        !m_magnitude.is_cuda() || !v_row.is_cuda() || !v_col.is_cuda() ||
        !codebook.is_cuda()) {
        throw std::invalid_argument("Expected all tensors to be on CUDA.");
    }
    if (grad_view.device() != p.device() || m_sign.device() != p.device() ||
        m_magnitude.device() != p.device() || v_row.device() != p.device() ||
        v_col.device() != p.device() || codebook.device() != p.device()) {
        throw std::invalid_argument("Expected all tensors on the same device as p.");
    }
    if (!p.is_contiguous() || !grad_view.is_contiguous() || !m_sign.is_contiguous() ||
        !m_magnitude.is_contiguous() || !v_row.is_contiguous() ||
        !v_col.is_contiguous() || !codebook.is_contiguous()) {
        throw std::invalid_argument("Expected all tensors to be contiguous.");
    }
    const bool use_lut = lut.numel() > 0;
    if (use_lut) {
        if (!lut.is_cuda() || lut.device() != p.device()) {
            throw std::invalid_argument("Expected lut on the same device as p.");
        }
        if (!lut.is_contiguous() || lut.scalar_type() != at::kShort) {
            throw std::invalid_argument("Expected lut to be contiguous int16.");
        }
        if (lut.numel() < 2) {
            throw std::invalid_argument("Expected lut to have at least 2 entries.");
        }
    }
    if (grad_view.dim() != 2) {
        throw std::invalid_argument("Expected grad_view to be 2D.");
    }
    if (m_sign.scalar_type() != at::kByte) {
        throw std::invalid_argument("Expected m_sign to have dtype uint8.");
    }
    if (codebook.scalar_type() != at::kFloat || m_magnitude.scalar_type() != at::kFloat ||
        v_row.scalar_type() != at::kFloat || v_col.scalar_type() != at::kFloat) {
        throw std::invalid_argument(
            "Expected codebook/m_magnitude/v_row/v_col to have dtype float32.");
    }
    if (grad_view.scalar_type() != p.scalar_type()) {
        throw std::invalid_argument("Expected grad_view dtype to match p.");
    }

    // Optional capturable device-scalar buffer ([lr, 1/bc2, 1/bc1, wd]); null
    // -> bit-identical legacy host-scalar path.
    const float* step_scalars_dev = resolve_step_scalars_ptr(step_scalars, p);
    // Optional capturable device-side stochastic-rounding seed (read by the
    // phase-2 quantize kernel); null -> legacy host-seed path.
    const int64_t* seed_dev_ptr = resolve_seed_dev_ptr(seed_dev, p);

    c10::cuda::CUDAGuard device_guard(p.device());

    const int64_t num_blocks = grad_view.size(0);
    const int64_t period = grad_view.size(1);
    const int64_t total_numel = num_blocks * period;
    if (p.numel() != total_numel || m_sign.numel() != total_numel) {
        throw std::invalid_argument("Expected p and m_sign numel to match grad_view.");
    }
    if (m_magnitude.size(0) != num_blocks) {
        throw std::invalid_argument("Expected m_magnitude to match num_blocks.");
    }
    if (cols <= 0 || total_numel % cols != 0) {
        throw std::invalid_argument("Expected cols to divide the total numel.");
    }
    const int64_t rows = total_numel / cols;
    if (v_row.numel() != rows || v_col.numel() != cols) {
        throw std::invalid_argument("Expected v_row [rows] and v_col [cols].");
    }
    if (codebook.numel() < 1 || codebook.numel() > 256) {
        throw std::invalid_argument("Expected codebook size in [1, 256].");
    }
    if (period <= 0) {
        throw std::invalid_argument("Expected grad_view to have a positive period.");
    }
    if (total_numel == 0) {
        return;
    }

    // One pooled zero-filled scratch serves new_magnitude (atomicMax target),
    // row_sq and col_sq (atomicAdd targets): a single fill launch instead of
    // three per-tensor at::zeros fills -- the per-param launch count is what
    // bounds this step (the fills are ~1 us of GPU each but ~2-3 us of enqueue
    // apiece). Zero-init semantics are unchanged, so this is bit-identical.
    // The non-capturable tail set_()s m_magnitude onto the new_magnitude view,
    // which keeps the whole (rows+cols floats larger) pool storage alive until
    // the next step -- a few KB per 2D param; state_dict's _compact already
    // clones such non-owning views on serialization.
    //
    // Capturable (in-place magnitude writeback, see the phase-2 arg comment):
    // the pool grows by num_blocks + 1 floats -- a staged copy of the OLD
    // magnitudes (filled by the EMA launch, read by phase 2, so phase 2 can
    // write the new magnitudes straight into the persistent m_magnitude and
    // the per-param copy_ launch disappears from the captured graph) plus a
    // 1-elem Sum(v_row) accumulator (block-reduced atomicAdd in the EMA
    // launch, consumed as mean = sum/rows in phase-2 registers, so the
    // per-param at::mean launch disappears too).
    const bool inplace_mag = step_scalars_dev != nullptr;
    auto scratch = at::zeros(
        {num_blocks + rows + cols + (inplace_mag ? num_blocks + 1 : 0)},
        m_magnitude.options());
    auto new_magnitude = scratch.narrow(0, 0, num_blocks).view({num_blocks, 1});
    auto row_sq = scratch.narrow(0, num_blocks, rows);
    auto col_sq = scratch.narrow(0, num_blocks + rows, cols);
    float* old_mag_copy = inplace_mag
        ? scratch.data_ptr<float>() + num_blocks + rows + cols
        : nullptr;
    float* vrow_sum = inplace_mag
        ? scratch.data_ptr<float>() + 2 * num_blocks + rows + cols
        : nullptr;

    const int codebook_size = static_cast<int>(codebook.numel());
    const int threads = 256;
    const int sm_count = cached_sm_count(p.get_device());
    const int64_t flat_blocks_cap = static_cast<int64_t>(sm_count) * 32;
    const size_t codebook_bytes = static_cast<size_t>(codebook_size) * sizeof(float);

    // Phase 1 (combined stats): ONE pass over grad + m_sign computes the
    // per-block absmax AND the raw row/col sum-of-squares for the Adafactor
    // EMAs -- the separate magnitude launch and the host-side chunked grad^2
    // reduction are both gone.
    {
        const int64_t grid_x =
            (cols + GEFEN_FACTORED_TILE_COLS - 1) / GEFEN_FACTORED_TILE_COLS;
        int64_t grid_y =
            (rows + GEFEN_FACTORED_TILE_ROWS - 1) / GEFEN_FACTORED_TILE_ROWS;
        // CUDA caps gridDim.y at 65535; taller tensors are covered by the
        // kernel's y-grid-stride loop.
        if (grid_y > 65535) {
            grid_y = 65535;
        }
        const dim3 grid(static_cast<unsigned int>(grid_x),
                        static_cast<unsigned int>(grid_y));
        const size_t shmem =
            codebook_bytes + GEFEN_FACTORED_TILE_COLS * sizeof(float);
        GEFEN_DISPATCH_FLOAT_HALF_BF16(
            p.scalar_type(),
            "gefen_factored_stats", [&] {
                gefen_factored_stats_kernel<scalar_t><<<grid, threads, shmem, c10::cuda::getCurrentCUDAStream()>>>(
                    grad_view.data_ptr<scalar_t>(), m_sign.data_ptr<uint8_t>(),
                    m_magnitude.data_ptr<float>(), codebook.data_ptr<float>(),
                    new_magnitude.data_ptr<float>(), row_sq.data_ptr<float>(),
                    col_sq.data_ptr<float>(), codebook_size, period, rows, cols,
                    static_cast<float>(beta1));
            });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // EMA advance as ONE tiny device launch (no host sync): the caller's
    // v_row/v_col are updated in place from the raw sums. Bit-identical to the
    // former aten chain v.mul_(beta2).add_(sq.div_(n), 1-beta2) -- see the
    // kernel comment for the pinned rounding sequence. Non-capturable keeps
    // the mean on at::mean (its reduction order is bit-preserved); under
    // capturable the same launch also stages the old-magnitude copy for
    // phase 2's in-place magnitude writeback AND accumulates Sum(v_row) for
    // phase 2's in-register mean, so neither a copy_ nor an at::mean launch
    // remains in the captured graph.
    {
        const int64_t total_rc =
            rows + cols + (inplace_mag ? num_blocks : 0);
        int64_t ema_blocks = (total_rc + threads - 1) / threads;
        if (ema_blocks > flat_blocks_cap) ema_blocks = flat_blocks_cap;
        if (ema_blocks < 1) ema_blocks = 1;
        const size_t ema_shmem = inplace_mag
            ? static_cast<size_t>(threads) * sizeof(float)
            : 0;
        gefen_factored_v_ema_kernel<<<static_cast<unsigned int>(ema_blocks), threads, ema_shmem, c10::cuda::getCurrentCUDAStream()>>>(
            v_row.data_ptr<float>(), v_col.data_ptr<float>(),
            row_sq.data_ptr<float>(), col_sq.data_ptr<float>(),
            rows, cols, static_cast<float>(beta2),
            static_cast<float>(1.0 - beta2),
            1.0f / static_cast<float>(static_cast<double>(cols)),
            1.0f / static_cast<float>(static_cast<double>(rows)),
            inplace_mag ? m_magnitude.data_ptr<float>() : nullptr,
            old_mag_copy,
            inplace_mag ? num_blocks : 0,
            vrow_sum);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    at::Tensor mean_v_row;
    if (!inplace_mag) {
        mean_v_row = v_row.mean().reshape({1}).contiguous();
    }


    // Phase 2: quantize + parameter update with the in-register factored
    // per-element stepsize. Grid sized to chunks (GEFEN_UPD_CHUNK contiguous
    // elements per thread), still capped to a resident grid.
    GEFEN_DISPATCH_FLOAT_HALF_BF16(
        p.scalar_type(),
        "gefen_factored_update", [&] {
            const int64_t chunk_span =
                static_cast<int64_t>(threads) * GEFEN_UPD_CHUNK;
            int64_t nblocks = (total_numel + chunk_span - 1) / chunk_span;
            if (nblocks > flat_blocks_cap) nblocks = flat_blocks_cap;
            if (nblocks < 1) nblocks = 1;
            gefen_update_flat_factored_kernel<scalar_t><<<static_cast<unsigned int>(nblocks), threads, codebook_bytes, c10::cuda::getCurrentCUDAStream()>>>(
                p.data_ptr<scalar_t>(), grad_view.data_ptr<scalar_t>(), m_sign.data_ptr<uint8_t>(),
                inplace_mag ? old_mag_copy : m_magnitude.data_ptr<float>(),
                new_magnitude.data_ptr<float>(),
                v_row.data_ptr<float>(), v_col.data_ptr<float>(),
                inplace_mag ? vrow_sum : mean_v_row.data_ptr<float>(),
                inplace_mag ? rows : static_cast<int64_t>(0),
                codebook.data_ptr<float>(),
                use_lut ? lut.data_ptr<int16_t>() : nullptr,
                use_lut ? static_cast<int>(lut.numel() - 1) : 0,
                codebook_size, period, cols, total_numel,
                static_cast<float>(beta1), static_cast<float>(lr), static_cast<float>(eps),
                static_cast<float>(inv_bias_correction_2),
                static_cast<float>(inv_bias_correction_1),
                static_cast<float>(weight_decay_factor),
                stochastic_round, static_cast<uint64_t>(rng_seed),
                seed_dev_ptr,
                step_scalars_dev,
                inplace_mag ? m_magnitude.data_ptr<float>() : nullptr);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (!inplace_mag) {
        // Legacy (non-capturable) flow, byte-for-byte unchanged: rebind the
        // state onto the freshly written scratch -- a host-side set_(), zero
        // launches. A CUDA graph cannot replay a set_(), which is why the
        // capturable path instead had phase 2 write the new magnitudes
        // straight into the persistent m_magnitude (reading the old values
        // from the scratch copy staged by the EMA launch), keeping replays on
        // stable addresses with no extra copy launch.
        m_magnitude.set_(new_magnitude);
    }
}

void gefen_factored_v_ema_cuda(
    at::Tensor v_row,
    at::Tensor v_col,
    at::Tensor row_sq,
    at::Tensor col_sq,
    double beta2
) {
    // Test/inspection entry for the fused factored-v EMA advance (the factored
    // launcher calls the same kernel inline). Updates v_row/v_col in place:
    //     v_row = v_row*beta2 + (row_sq/cols)*(1-beta2)   (cols = v_col.numel())
    //     v_col = v_col*beta2 + (col_sq/rows)*(1-beta2)   (rows = v_row.numel())
    // bit-identical to the aten chain it replaced; the parity test pins that.
    if (!v_row.is_cuda() || !v_col.is_cuda() || !row_sq.is_cuda() || !col_sq.is_cuda()) {
        throw std::invalid_argument("Expected all tensors to be on CUDA.");
    }
    if (v_col.device() != v_row.device() || row_sq.device() != v_row.device() ||
        col_sq.device() != v_row.device()) {
        throw std::invalid_argument("Expected all tensors on the same device.");
    }
    if (!v_row.is_contiguous() || !v_col.is_contiguous() ||
        !row_sq.is_contiguous() || !col_sq.is_contiguous()) {
        throw std::invalid_argument("Expected all tensors to be contiguous.");
    }
    if (v_row.scalar_type() != at::kFloat || v_col.scalar_type() != at::kFloat ||
        row_sq.scalar_type() != at::kFloat || col_sq.scalar_type() != at::kFloat) {
        throw std::invalid_argument("Expected all tensors to have dtype float32.");
    }
    const int64_t rows = v_row.numel();
    const int64_t cols = v_col.numel();
    if (row_sq.numel() != rows || col_sq.numel() != cols) {
        throw std::invalid_argument(
            "Expected row_sq/col_sq numel to match v_row/v_col.");
    }
    if (rows == 0 && cols == 0) {
        return;
    }
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument(
            "Expected both v_row and v_col to be non-empty.");
    }

    c10::cuda::CUDAGuard device_guard(v_row.device());
    const int threads = 256;
    const int sm_count = cached_sm_count(v_row.get_device());
    const int64_t flat_blocks_cap = static_cast<int64_t>(sm_count) * 32;
    const int64_t total_rc = rows + cols;
    int64_t ema_blocks = (total_rc + threads - 1) / threads;
    if (ema_blocks > flat_blocks_cap) ema_blocks = flat_blocks_cap;
    if (ema_blocks < 1) ema_blocks = 1;
    gefen_factored_v_ema_kernel<<<static_cast<unsigned int>(ema_blocks), threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        v_row.data_ptr<float>(), v_col.data_ptr<float>(),
        row_sq.data_ptr<float>(), col_sq.data_ptr<float>(),
        rows, cols, static_cast<float>(beta2),
        static_cast<float>(1.0 - beta2),
        1.0f / static_cast<float>(static_cast<double>(cols)),
        1.0f / static_cast<float>(static_cast<double>(rows)),
        nullptr, nullptr, 0, nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gefen_quantized_momentum_update_cuda(
    at::Tensor grad_view,
    at::Tensor m_sign,
    at::Tensor m_magnitude,
    at::Tensor codebook,
    at::Tensor lut,
    at::Tensor momentum_out,
    double beta1,
    bool stochastic_round,
    int64_t rng_seed,
    c10::optional<at::Tensor> seed_dev,
    bool nesterov
) {
    if (!grad_view.is_cuda() || !m_sign.is_cuda() || !m_magnitude.is_cuda() ||
        !codebook.is_cuda() || !momentum_out.is_cuda()) {
        throw std::invalid_argument("Expected all tensors to be on CUDA.");
    }
    const bool use_lut = lut.numel() > 0;
    if (use_lut) {
        if (!lut.is_cuda() || lut.device() != grad_view.device()) {
            throw std::invalid_argument(
                "Expected lut on the same device as grad_view.");
        }
        if (!lut.is_contiguous() || lut.scalar_type() != at::kShort) {
            throw std::invalid_argument("Expected lut to be contiguous int16.");
        }
        if (lut.numel() < 2) {
            throw std::invalid_argument("Expected lut to have at least 2 entries.");
        }
    }
    // Optional capturable device-side stochastic-rounding seed (read by the
    // phase-2 emit kernel); null -> legacy host-seed path.
    const int64_t* seed_dev_ptr = resolve_seed_dev_ptr(seed_dev, grad_view);
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
    const float beta1_f = static_cast<float>(beta1);
    // Match Python's `1 - momentum`: subtraction happens in double, then
    // TensorIterator narrows the scalar to its fp32 opmath type.  Computing
    // `1.0f - beta1_f` would differ for ordinary betas such as 0.9/0.95/0.99.
    const float nesterov_alpha = static_cast<float>(1.0 - beta1);
    const int threads = 256;
    const int sm_count = cached_sm_count(grad_view.get_device());
    const int64_t flat_blocks_cap = static_cast<int64_t>(sm_count) * 32;
    const size_t codebook_bytes = static_cast<size_t>(codebook_size) * sizeof(float);

    // Phase 1: per-block magnitude (reuses the generic v2 magnitude kernels, so
    // new_magnitude is bit-identical to the generic update's m_magnitude).
    GEFEN_DISPATCH_FLOAT_HALF_BF16(
        grad_view.scalar_type(),
        "gefen_momentum_magnitude", [&] {
            // Per-element atomics scale poorly once many elements share a
            // magnitude slot.  The split reducer wins decisively on both
            // Ampere and Blackwell above this small-period tail (notably the
            // period-512/2048 K/V matrices common in transformer Muon paths).
            if (period <= GEFEN_MOMENTUM_FLAT_MAX_PERIOD) {
                int64_t nblocks = (total_numel + threads - 1) / threads;
                if (nblocks > flat_blocks_cap) nblocks = flat_blocks_cap;
                if (nblocks < 1) nblocks = 1;
                gefen_magnitude_flat_kernel<scalar_t><<<static_cast<unsigned int>(nblocks), threads, codebook_bytes, c10::cuda::getCurrentCUDAStream()>>>(
                    grad_view.data_ptr<scalar_t>(), m_sign.data_ptr<uint8_t>(),
                    m_magnitude.data_ptr<float>(), codebook.data_ptr<float>(),
                    new_magnitude.data_ptr<float>(), codebook_size, period, total_numel,
                    beta1_f);
            } else {
                int blocks_per_row = static_cast<int>((period + threads * 64 - 1) / (threads * 64));
                if (blocks_per_row < 1) blocks_per_row = 1;
                const int max_bpr = static_cast<int>((flat_blocks_cap + num_blocks - 1) / num_blocks);
                if (blocks_per_row > max_bpr && max_bpr >= 1) blocks_per_row = max_bpr;
                if (blocks_per_row < 1) blocks_per_row = 1;
                const int64_t grid = num_blocks * blocks_per_row;
                const size_t shmem = static_cast<size_t>(threads) * sizeof(float) + codebook_bytes;
                gefen_magnitude_split_kernel<scalar_t><<<static_cast<unsigned int>(grid), threads, shmem, c10::cuda::getCurrentCUDAStream()>>>(
                    grad_view.data_ptr<scalar_t>(), m_sign.data_ptr<uint8_t>(),
                    m_magnitude.data_ptr<float>(), codebook.data_ptr<float>(),
                    new_magnitude.data_ptr<float>(), codebook_size, period, num_blocks, blocks_per_row,
                    beta1_f);
            }
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Phase 2: quantize state + emit the dense quantized momentum (no p write).
    GEFEN_DISPATCH_FLOAT_HALF_BF16(
        grad_view.scalar_type(),
        "gefen_momentum_emit", [&] {
            if (period <= GEFEN_MOMENTUM_FLAT_MAX_PERIOD) {
                // At tiny periods phase 1 is already a flat grid and the
                // scalar emitter has slightly lower register/control overhead;
                // keep it instead of regressing the elementwise tail.
                int64_t nblocks = (total_numel + threads - 1) / threads;
                if (nblocks > flat_blocks_cap) nblocks = flat_blocks_cap;
                if (nblocks < 1) nblocks = 1;
                gefen_momentum_emit_flat_kernel<scalar_t><<<static_cast<unsigned int>(nblocks), threads, codebook_bytes, c10::cuda::getCurrentCUDAStream()>>>(
                    grad_view.data_ptr<scalar_t>(), m_sign.data_ptr<uint8_t>(),
                    m_magnitude.data_ptr<float>(), new_magnitude.data_ptr<float>(),
                    codebook.data_ptr<float>(), momentum_out.data_ptr<scalar_t>(),
                    codebook_size, period, total_numel, beta1_f,
                    nesterov_alpha, nesterov,
                    stochastic_round, static_cast<uint64_t>(rng_seed),
                    seed_dev_ptr);
                return;
            }
            const int64_t chunk_span =
                static_cast<int64_t>(threads) * GEFEN_UPD_CHUNK;
            int64_t nblocks = (total_numel + chunk_span - 1) / chunk_span;
            if (nblocks > flat_blocks_cap) nblocks = flat_blocks_cap;
            if (nblocks < 1) nblocks = 1;
            gefen_momentum_emit_chunked_kernel<scalar_t><<<static_cast<unsigned int>(nblocks), threads, codebook_bytes, c10::cuda::getCurrentCUDAStream()>>>(
                grad_view.data_ptr<scalar_t>(), m_sign.data_ptr<uint8_t>(),
                m_magnitude.data_ptr<float>(), new_magnitude.data_ptr<float>(),
                codebook.data_ptr<float>(),
                use_lut ? lut.data_ptr<int16_t>() : nullptr,
                use_lut ? static_cast<int>(lut.numel() - 1) : 0,
                momentum_out.data_ptr<scalar_t>(), codebook_size,
                period, total_numel, beta1_f, nesterov_alpha, nesterov,
                stochastic_round, static_cast<uint64_t>(rng_seed),
                seed_dev_ptr);
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
    at::Tensor lut,
    bool packed_indices,
    double beta1,
    double beta2,
    double lr,
    double eps,
    double inv_sqrt_bias_correction_2,
    double inv_bias_correction_1,
    double weight_decay_factor,
    bool stochastic_round,
    int64_t rng_seed,
    c10::optional<at::Tensor> step_scalars,
    c10::optional<at::Tensor> seed_dev
) {
    if (packed_indices) {
        throw std::invalid_argument("v2 fused update does not support packed indices.");
    }
    // Optional capturable device-scalar buffer ([lr, 1/sqrt(bc2), 1/bc1, wd]):
    // the finalize kernel reads slots 1-2, the phase-2 update reads slots 0/3.
    // Null -> bit-identical legacy host-scalar path.
    const float* step_scalars_dev = resolve_step_scalars_ptr(step_scalars, p);
    // Optional capturable device-side stochastic-rounding seed (read by the
    // phase-2 quantize kernel); null -> legacy host-seed path.
    const int64_t* seed_dev_ptr = resolve_seed_dev_ptr(seed_dev, p);
    // Optional search LUT (empty tensor -> full-range binary search), used by
    // the phase-2 quantize kernel. Same contract as the v1-full path.
    const bool use_lut = lut.numel() > 0;
    if (use_lut) {
        if (!lut.is_cuda() || lut.device() != p.device()) {
            throw std::invalid_argument("Expected lut on the same device as p.");
        }
        if (!lut.is_contiguous() || lut.scalar_type() != at::kShort) {
            throw std::invalid_argument("Expected lut to be contiguous int16.");
        }
        if (lut.numel() < 2) {
            throw std::invalid_argument("Expected lut to have at least 2 entries.");
        }
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
    GEFEN_DISPATCH_FLOAT_HALF_BF16(
        p.scalar_type(),
        "gefen_v2_full_magnitude", [&] {
            if (period <= 2048) {
                int64_t nblocks = (total_numel + threads - 1) / threads;
                if (nblocks > flat_blocks_cap) nblocks = flat_blocks_cap;
                if (nblocks < 1) nblocks = 1;
                gefen_magnitude_sumsq_flat_kernel<scalar_t><<<static_cast<unsigned int>(nblocks), threads, codebook_bytes, c10::cuda::getCurrentCUDAStream()>>>(
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
                gefen_magnitude_sumsq_split_kernel<scalar_t><<<static_cast<unsigned int>(grid), threads, shmem, c10::cuda::getCurrentCUDAStream()>>>(
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
        gefen_finalize_vmean_stepsize_kernel<<<static_cast<unsigned int>(fblocks), fthreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
            vmean.data_ptr<float>(), sumsq.data_ptr<float>(), num_blocks, period,
            static_cast<float>(beta2), static_cast<float>(eps),
            static_cast<float>(inv_sqrt_bias_correction_2),
            static_cast<float>(inv_bias_correction_1),
            step_scalars_dev);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Phase 2: quantize + parameter update with weight decay folded in (K3).
    // Grid sized to chunks (GEFEN_UPD_CHUNK contiguous elements per thread),
    // still capped to a resident grid.
    GEFEN_DISPATCH_FLOAT_HALF_BF16(
        p.scalar_type(),
        "gefen_v2_full_update", [&] {
            const int64_t chunk_span =
                static_cast<int64_t>(threads) * GEFEN_UPD_CHUNK;
            int64_t nblocks = (total_numel + chunk_span - 1) / chunk_span;
            if (nblocks > flat_blocks_cap) nblocks = flat_blocks_cap;
            if (nblocks < 1) nblocks = 1;
            gefen_update_flat_full_kernel<scalar_t><<<static_cast<unsigned int>(nblocks), threads, codebook_bytes, c10::cuda::getCurrentCUDAStream()>>>(
                p.data_ptr<scalar_t>(), grad_view.data_ptr<scalar_t>(), m_sign.data_ptr<uint8_t>(),
                m_magnitude.data_ptr<float>(), new_magnitude.data_ptr<float>(),
                sumsq.data_ptr<float>(), codebook.data_ptr<float>(),
                use_lut ? lut.data_ptr<int16_t>() : nullptr,
                use_lut ? static_cast<int>(lut.numel() - 1) : 0,
                codebook_size,
                period, total_numel, static_cast<float>(beta1), static_cast<float>(lr),
                static_cast<float>(weight_decay_factor),
                stochastic_round, static_cast<uint64_t>(rng_seed),
                seed_dev_ptr,
                step_scalars_dev);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (step_scalars_dev != nullptr) {
        // Capturable path: set_() is a HOST-side storage rebind, which a CUDA
        // graph cannot replay -- after capture the state tensor would stay
        // pointed at the capture-time scratch while the kernels keep reading
        // the (stale, never-again-updated) pre-capture pointer. Copy the new
        // magnitudes back into the persistent state buffer on-stream instead,
        // so replays read/write stable addresses. ([num_blocks, 1] D2D copy --
        // tiny next to the update kernels.)
        m_magnitude.copy_(new_magnitude);
        return;
    }
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

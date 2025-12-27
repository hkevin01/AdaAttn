/**
 * @file entropy_estimate.cu
 * @brief CUDA kernel for fast entropy estimation
 *
 * Computes attention entropy to guide adaptive rank selection.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>

namespace adaattn {
namespace kernels {
namespace adaptive {

constexpr int ENTROPY_BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

/**
 * Compute Shannon entropy of attention weights.
 * H = -sum(p * log(p))
 *
 * @param attn_weights Attention weights [batch * heads, seq_len, seq_len]
 * @param entropy Output entropy per row [batch * heads, seq_len]
 * @param seq_len Sequence length
 */
template <typename T>
__global__ void compute_attention_entropy(
    const T* __restrict__ attn_weights,
    float* __restrict__ entropy,
    const int seq_len
) {
    const int batch_head_idx = blockIdx.x;
    const int row_idx = blockIdx.y;
    const int tid = threadIdx.x;

    __shared__ float shared_sum[ENTROPY_BLOCK_SIZE / WARP_SIZE];

    const int row_start = batch_head_idx * seq_len * seq_len + row_idx * seq_len;

    // Compute local entropy contribution
    float local_entropy = 0.0f;
    for (int col = tid; col < seq_len; col += blockDim.x) {
        float p = static_cast<float>(attn_weights[row_start + col]);
        if (p > 1e-10f) {
            local_entropy -= p * logf(p);
        }
    }

    // Warp reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_entropy += __shfl_xor_sync(0xffffffff, local_entropy, offset);
    }

    // Block reduction
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    if (lane_id == 0) {
        shared_sum[warp_id] = local_entropy;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < blockDim.x / WARP_SIZE) ? shared_sum[lane_id] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            val += __shfl_xor_sync(0xffffffff, val, offset);
        }
        if (lane_id == 0) {
            entropy[batch_head_idx * seq_len + row_idx] = val;
        }
    }
}

/**
 * Estimate effective rank based on entropy.
 * Effective rank = exp(average_entropy)
 *
 * @param entropy Per-row entropy [batch * heads, seq_len]
 * @param effective_rank Output effective rank per head [batch * heads]
 * @param seq_len Sequence length
 */
__global__ void estimate_effective_rank(
    const float* __restrict__ entropy,
    float* __restrict__ effective_rank,
    const int seq_len
) {
    const int batch_head_idx = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float shared_sum[ENTROPY_BLOCK_SIZE / WARP_SIZE];

    // Compute average entropy for this head
    float local_sum = 0.0f;
    for (int row = tid; row < seq_len; row += blockDim.x) {
        local_sum += entropy[batch_head_idx * seq_len + row];
    }

    // Warp reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }

    // Block reduction
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    if (lane_id == 0) {
        shared_sum[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < blockDim.x / WARP_SIZE) ? shared_sum[lane_id] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            val += __shfl_xor_sync(0xffffffff, val, offset);
        }
        if (lane_id == 0) {
            float avg_entropy = val / static_cast<float>(seq_len);
            effective_rank[batch_head_idx] = expf(avg_entropy);
        }
    }
}

/**
 * Determine target rank for each attention head based on effective rank.
 *
 * @param effective_rank Effective rank per head [batch * heads]
 * @param target_rank Output target rank per head [batch * heads]
 * @param min_rank Minimum allowed rank
 * @param max_rank Maximum allowed rank
 * @param threshold Threshold for switching to low-rank
 */
__global__ void determine_target_rank(
    const float* __restrict__ effective_rank,
    int* __restrict__ target_rank,
    const int min_rank,
    const int max_rank,
    const float threshold
) {
    const int batch_head_idx = blockIdx.x * blockDim.x + threadIdx.x;

    float eff_rank = effective_rank[batch_head_idx];

    // Determine target rank
    int rank;
    if (eff_rank < threshold) {
        // Low effective rank -> use low-rank approximation
        rank = max(min_rank, static_cast<int>(eff_rank * 1.5f));
    } else {
        // High effective rank -> use dense or high rank
        rank = max_rank;  // -1 could indicate dense attention
    }

    target_rank[batch_head_idx] = min(rank, max_rank);
}

// Explicit instantiations
template __global__ void compute_attention_entropy<float>(
    const float*, float*, int
);

template __global__ void compute_attention_entropy<__half>(
    const __half*, float*, int
);

}  // namespace adaptive
}  // namespace kernels
}  // namespace adaattn

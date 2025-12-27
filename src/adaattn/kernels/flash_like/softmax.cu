/**
 * @file softmax.cu
 * @brief CUDA kernel for numerically stable softmax
 *
 * Implements online softmax computation to avoid materializing
 * the full attention matrix, following FlashAttention design.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

namespace adaattn {
namespace kernels {

constexpr int WARP_SIZE = 32;
constexpr int SOFTMAX_BLOCK_SIZE = 256;

/**
 * Warp-level reduction for max
 */
template <typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

/**
 * Warp-level reduction for sum
 */
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Online softmax kernel - computes softmax in a single pass
 * using the online algorithm for numerical stability.
 *
 * @param input Score matrix [batch * num_heads, seq_len, seq_len]
 * @param output Attention weights [batch * num_heads, seq_len, seq_len]
 * @param seq_len Sequence length
 * @param causal Whether to apply causal masking
 */
template <typename T>
__global__ void online_softmax_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int seq_len,
    const bool causal
) {
    // Each block handles one row of the attention matrix
    const int batch_head_idx = blockIdx.x;
    const int row_idx = blockIdx.y;

    const int row_start = batch_head_idx * seq_len * seq_len + row_idx * seq_len;
    const int tid = threadIdx.x;

    // Determine valid range for causal masking
    const int valid_len = causal ? min(row_idx + 1, seq_len) : seq_len;

    // Step 1: Find max value (for numerical stability)
    float thread_max = -FLT_MAX;

    for (int col = tid; col < valid_len; col += blockDim.x) {
        float val = static_cast<float>(input[row_start + col]);
        thread_max = max(thread_max, val);
    }

    // Warp reduction for max
    thread_max = warp_reduce_max(thread_max);

    // Block reduction for max using shared memory
    __shared__ float shared_max[SOFTMAX_BLOCK_SIZE / WARP_SIZE];
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    if (lane_id == 0) {
        shared_max[warp_id] = thread_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < blockDim.x / WARP_SIZE) ?
                    shared_max[lane_id] : -FLT_MAX;
        val = warp_reduce_max(val);
        if (lane_id == 0) {
            shared_max[0] = val;
        }
    }
    __syncthreads();

    const float row_max = shared_max[0];

    // Step 2: Compute exp and sum
    float thread_sum = 0.0f;

    for (int col = tid; col < valid_len; col += blockDim.x) {
        float val = static_cast<float>(input[row_start + col]);
        thread_sum += expf(val - row_max);
    }

    // Warp reduction for sum
    thread_sum = warp_reduce_sum(thread_sum);

    // Block reduction for sum
    __shared__ float shared_sum[SOFTMAX_BLOCK_SIZE / WARP_SIZE];

    if (lane_id == 0) {
        shared_sum[warp_id] = thread_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < blockDim.x / WARP_SIZE) ?
                    shared_sum[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            shared_sum[0] = val;
        }
    }
    __syncthreads();

    const float row_sum = shared_sum[0];
    const float inv_sum = 1.0f / (row_sum + 1e-10f);

    // Step 3: Write normalized output
    for (int col = tid; col < seq_len; col += blockDim.x) {
        float val;
        if (col < valid_len) {
            val = expf(static_cast<float>(input[row_start + col]) - row_max) * inv_sum;
        } else {
            val = 0.0f;  // Masked positions
        }
        output[row_start + col] = static_cast<T>(val);
    }
}

// Explicit instantiations
template __global__ void online_softmax_kernel<float>(
    const float*, float*, int, bool
);

template __global__ void online_softmax_kernel<__half>(
    const __half*, __half*, int, bool
);

}  // namespace kernels
}  // namespace adaattn

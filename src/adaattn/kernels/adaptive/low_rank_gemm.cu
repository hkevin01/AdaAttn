/**
 * @file low_rank_gemm.cu
 * @brief Low-rank GEMM kernels for adaptive attention
 *
 * Implements efficient low-rank matrix multiplication for
 * attention approximation.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace adaattn {
namespace kernels {
namespace adaptive {

constexpr int LR_BLOCK_SIZE = 128;

/**
 * Low-rank attention: output = Q @ (K^T @ V) with rank-r approximation
 *
 * For low-rank attention, we compute:
 *   1. K_proj = P @ K  (project K to rank r)
 *   2. V_proj = P @ V  (project V to rank r)
 *   3. scores = Q @ K_proj^T
 *   4. output = softmax(scores) @ V_proj
 *
 * This reduces O(n^2) to O(n*r) complexity.
 */
template <typename T>
__global__ void low_rank_projection_kernel(
    const T* __restrict__ input,     // [batch, seq_len, num_heads, head_dim]
    const T* __restrict__ projection, // [seq_len, rank]
    T* __restrict__ output,          // [batch, rank, num_heads, head_dim]
    const int batch_size,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const int rank
) {
    // Each block handles one (batch, head, head_dim_idx) combination
    const int batch_idx = blockIdx.z / num_heads;
    const int head_idx = blockIdx.z % num_heads;
    const int d_idx = blockIdx.x;

    const int tid = threadIdx.x;

    __shared__ float shared_sum[LR_BLOCK_SIZE];

    // Compute one output element: sum over seq_len
    for (int r = blockIdx.y; r < rank; r += gridDim.y) {
        float acc = 0.0f;

        for (int s = tid; s < seq_len; s += blockDim.x) {
            float proj_val = static_cast<float>(projection[s * rank + r]);
            float input_val = static_cast<float>(
                input[batch_idx * seq_len * num_heads * head_dim +
                      s * num_heads * head_dim +
                      head_idx * head_dim + d_idx]
            );
            acc += proj_val * input_val;
        }

        shared_sum[tid] = acc;
        __syncthreads();

        // Reduction
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                shared_sum[tid] += shared_sum[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            output[batch_idx * rank * num_heads * head_dim +
                   r * num_heads * head_dim +
                   head_idx * head_dim + d_idx] = static_cast<T>(shared_sum[0]);
        }
    }
}

/**
 * Compute low-rank attention scores: Q @ K_proj^T
 */
template <typename T>
__global__ void low_rank_score_kernel(
    const T* __restrict__ Q,        // [batch, seq_len, num_heads, head_dim]
    const T* __restrict__ K_proj,   // [batch, rank, num_heads, head_dim]
    T* __restrict__ scores,         // [batch, num_heads, seq_len, rank]
    const int batch_size,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const int rank,
    const float scale
) {
    const int batch_head_idx = blockIdx.z;
    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    const int row = blockIdx.y * blockDim.y + threadIdx.y;  // seq position
    const int col = blockIdx.x * blockDim.x + threadIdx.x;  // rank position

    if (row < seq_len && col < rank) {
        float acc = 0.0f;

        for (int d = 0; d < head_dim; ++d) {
            float q_val = static_cast<float>(
                Q[batch_idx * seq_len * num_heads * head_dim +
                  row * num_heads * head_dim +
                  head_idx * head_dim + d]
            );
            float k_val = static_cast<float>(
                K_proj[batch_idx * rank * num_heads * head_dim +
                       col * num_heads * head_dim +
                       head_idx * head_dim + d]
            );
            acc += q_val * k_val;
        }

        scores[batch_head_idx * seq_len * rank + row * rank + col] =
            static_cast<T>(acc * scale);
    }
}

/**
 * Compute low-rank output: attn_weights @ V_proj
 */
template <typename T>
__global__ void low_rank_output_kernel(
    const T* __restrict__ attn,     // [batch, num_heads, seq_len, rank]
    const T* __restrict__ V_proj,   // [batch, rank, num_heads, head_dim]
    T* __restrict__ output,         // [batch, seq_len, num_heads, head_dim]
    const int batch_size,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const int rank
) {
    const int batch_head_idx = blockIdx.z;
    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    const int row = blockIdx.y * blockDim.y + threadIdx.y;  // seq position
    const int col = blockIdx.x * blockDim.x + threadIdx.x;  // head_dim position

    if (row < seq_len && col < head_dim) {
        float acc = 0.0f;

        for (int r = 0; r < rank; ++r) {
            float attn_val = static_cast<float>(
                attn[batch_head_idx * seq_len * rank + row * rank + r]
            );
            float v_val = static_cast<float>(
                V_proj[batch_idx * rank * num_heads * head_dim +
                       r * num_heads * head_dim +
                       head_idx * head_dim + col]
            );
            acc += attn_val * v_val;
        }

        output[batch_idx * seq_len * num_heads * head_dim +
               row * num_heads * head_dim +
               head_idx * head_dim + col] = static_cast<T>(acc);
    }
}

// Explicit instantiations
template __global__ void low_rank_projection_kernel<float>(
    const float*, const float*, float*,
    int, int, int, int, int
);

template __global__ void low_rank_projection_kernel<__half>(
    const __half*, const __half*, __half*,
    int, int, int, int, int
);

template __global__ void low_rank_score_kernel<float>(
    const float*, const float*, float*,
    int, int, int, int, int, float
);

template __global__ void low_rank_score_kernel<__half>(
    const __half*, const __half*, __half*,
    int, int, int, int, int, float
);

template __global__ void low_rank_output_kernel<float>(
    const float*, const float*, float*,
    int, int, int, int, int
);

template __global__ void low_rank_output_kernel<__half>(
    const __half*, const __half*, __half*,
    int, int, int, int, int
);

}  // namespace adaptive
}  // namespace kernels
}  // namespace adaattn

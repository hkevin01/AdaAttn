/**
 * @file qk_gemm.cu
 * @brief CUDA kernel for QK^T matrix multiplication
 *
 * This kernel computes the attention score matrix Q @ K^T
 * with optimizations for memory bandwidth and tensor core usage.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace adaattn {
namespace kernels {

// Block sizes optimized for A100 GPU
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 32;

/**
 * Tiled QK^T computation with shared memory
 *
 * @param Q Query matrix [batch, seq_len, num_heads, head_dim]
 * @param K Key matrix [batch, seq_len, num_heads, head_dim]
 * @param output Score matrix [batch, num_heads, seq_len, seq_len]
 * @param batch_size Batch dimension
 * @param seq_len Sequence length
 * @param num_heads Number of attention heads
 * @param head_dim Head dimension
 * @param scale Softmax scaling factor
 */
template <typename T>
__global__ void qk_gemm_kernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    T* __restrict__ output,
    const int batch_size,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const float scale
) {
    // Shared memory for tiles
    __shared__ T Q_tile[BLOCK_M][BLOCK_K];
    __shared__ T K_tile[BLOCK_N][BLOCK_K];

    // Block and thread indices
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int batch_head_idx = blockIdx.z;

    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = by * BLOCK_M + ty;
    const int col = bx * BLOCK_N + tx;

    // Accumulator
    float acc = 0.0f;

    // Compute base pointers
    const int q_base = batch_idx * seq_len * num_heads * head_dim +
                       head_idx * head_dim;
    const int k_base = batch_idx * seq_len * num_heads * head_dim +
                       head_idx * head_dim;

    // Tile loop over head_dim
    for (int k_tile = 0; k_tile < head_dim; k_tile += BLOCK_K) {
        // Load Q tile
        if (row < seq_len && (k_tile + tx) < head_dim) {
            Q_tile[ty][tx] = Q[q_base + row * num_heads * head_dim + k_tile + tx];
        } else {
            Q_tile[ty][tx] = static_cast<T>(0);
        }

        // Load K tile (transposed)
        if (col < seq_len && (k_tile + ty) < head_dim) {
            K_tile[tx][ty] = K[k_base + col * num_heads * head_dim + k_tile + ty];
        } else {
            K_tile[tx][ty] = static_cast<T>(0);
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < BLOCK_K; ++k) {
            acc += static_cast<float>(Q_tile[ty][k]) *
                   static_cast<float>(K_tile[tx][k]);
        }

        __syncthreads();
    }

    // Write output with scaling
    if (row < seq_len && col < seq_len) {
        const int out_idx = batch_head_idx * seq_len * seq_len +
                           row * seq_len + col;
        output[out_idx] = static_cast<T>(acc * scale);
    }
}

// Explicit instantiations
template __global__ void qk_gemm_kernel<float>(
    const float*, const float*, float*,
    int, int, int, int, float
);

template __global__ void qk_gemm_kernel<__half>(
    const __half*, const __half*, __half*,
    int, int, int, int, float
);

}  // namespace kernels
}  // namespace adaattn

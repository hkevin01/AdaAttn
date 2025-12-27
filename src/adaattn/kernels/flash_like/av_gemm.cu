/**
 * @file av_gemm.cu
 * @brief CUDA kernel for attention-weighted value computation
 *
 * Computes output = Attention @ V with optimized memory access patterns.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace adaattn {
namespace kernels {

constexpr int AV_BLOCK_M = 64;
constexpr int AV_BLOCK_N = 64;
constexpr int AV_BLOCK_K = 32;

/**
 * Compute Attention @ V
 *
 * @param attn Attention weights [batch * num_heads, seq_len, seq_len]
 * @param V Value matrix [batch, seq_len, num_heads, head_dim]
 * @param output Output [batch, seq_len, num_heads, head_dim]
 */
template <typename T>
__global__ void av_gemm_kernel(
    const T* __restrict__ attn,
    const T* __restrict__ V,
    T* __restrict__ output,
    const int batch_size,
    const int seq_len,
    const int num_heads,
    const int head_dim
) {
    __shared__ T attn_tile[AV_BLOCK_M][AV_BLOCK_K];
    __shared__ T v_tile[AV_BLOCK_K][AV_BLOCK_N];

    const int bx = blockIdx.x;  // head_dim block
    const int by = blockIdx.y;  // seq_len block
    const int batch_head_idx = blockIdx.z;

    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = by * AV_BLOCK_M + ty;
    const int col = bx * AV_BLOCK_N + tx;

    float acc = 0.0f;

    // Tile loop over seq_len (the K dimension for A @ V)
    for (int k_tile = 0; k_tile < seq_len; k_tile += AV_BLOCK_K) {
        // Load attention tile
        const int attn_row = row;
        const int attn_col = k_tile + tx;
        if (attn_row < seq_len && attn_col < seq_len) {
            attn_tile[ty][tx] = attn[batch_head_idx * seq_len * seq_len +
                                     attn_row * seq_len + attn_col];
        } else {
            attn_tile[ty][tx] = static_cast<T>(0);
        }

        // Load V tile
        const int v_row = k_tile + ty;
        const int v_col = col;
        if (v_row < seq_len && v_col < head_dim) {
            v_tile[ty][tx] = V[batch_idx * seq_len * num_heads * head_dim +
                               v_row * num_heads * head_dim +
                               head_idx * head_dim + v_col];
        } else {
            v_tile[ty][tx] = static_cast<T>(0);
        }

        __syncthreads();

        // Compute partial result
        #pragma unroll
        for (int k = 0; k < AV_BLOCK_K; ++k) {
            acc += static_cast<float>(attn_tile[ty][k]) *
                   static_cast<float>(v_tile[k][tx]);
        }

        __syncthreads();
    }

    // Write output
    if (row < seq_len && col < head_dim) {
        output[batch_idx * seq_len * num_heads * head_dim +
               row * num_heads * head_dim +
               head_idx * head_dim + col] = static_cast<T>(acc);
    }
}

template __global__ void av_gemm_kernel<float>(
    const float*, const float*, float*,
    int, int, int, int
);

template __global__ void av_gemm_kernel<__half>(
    const __half*, const __half*, __half*,
    int, int, int, int
);

}  // namespace kernels
}  // namespace adaattn

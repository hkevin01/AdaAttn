"""
Linear algebra utility functions.

General-purpose utilities for numerical linear algebra operations.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import Tensor


def stable_softmax(
    x: Tensor,
    dim: int = -1,
    temperature: float = 1.0,
) -> Tensor:
    """
    Numerically stable softmax implementation.

    Args:
        x: Input tensor
        dim: Dimension along which to compute softmax
        temperature: Temperature scaling factor

    Returns:
        Softmax output
    """
    x_scaled = x / temperature
    x_max = x_scaled.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x_scaled - x_max)
    return x_exp / (x_exp.sum(dim=dim, keepdim=True) + 1e-10)


def log_softmax_stable(
    x: Tensor,
    dim: int = -1,
    temperature: float = 1.0,
) -> Tensor:
    """
    Numerically stable log-softmax implementation.

    Args:
        x: Input tensor
        dim: Dimension along which to compute log-softmax
        temperature: Temperature scaling factor

    Returns:
        Log-softmax output
    """
    x_scaled = x / temperature
    x_max = x_scaled.max(dim=dim, keepdim=True).values
    x_shifted = x_scaled - x_max
    log_sum_exp = torch.log(torch.exp(x_shifted).sum(dim=dim, keepdim=True) + 1e-10)
    return x_shifted - log_sum_exp


def batch_trace(A: Tensor) -> Tensor:
    """
    Compute trace for batched matrices.

    Args:
        A: Batched square matrices of shape (..., n, n)

    Returns:
        Trace values of shape (...)
    """
    return torch.diagonal(A, dim1=-2, dim2=-1).sum(dim=-1)


def batch_frobenius_norm(A: Tensor) -> Tensor:
    """
    Compute Frobenius norm for batched matrices.

    Args:
        A: Batched matrices of shape (..., m, n)

    Returns:
        Frobenius norms of shape (...)
    """
    return torch.sqrt((A ** 2).sum(dim=(-2, -1)))


def symmetric_part(A: Tensor) -> Tensor:
    """
    Extract symmetric part of a matrix: (A + A^T) / 2

    Args:
        A: Input matrix of shape (..., n, n)

    Returns:
        Symmetric part of A
    """
    return (A + A.transpose(-2, -1)) / 2


def antisymmetric_part(A: Tensor) -> Tensor:
    """
    Extract antisymmetric part of a matrix: (A - A^T) / 2

    Args:
        A: Input matrix of shape (..., n, n)

    Returns:
        Antisymmetric part of A
    """
    return (A - A.transpose(-2, -1)) / 2


def condition_number_estimate(
    A: Tensor,
    method: str = "power",
    num_iterations: int = 10,
) -> Tensor:
    """
    Estimate condition number of a matrix.

    Args:
        A: Input matrix of shape (..., m, n)
        method: Estimation method ("power" or "svd")
        num_iterations: Number of iterations for power method

    Returns:
        Estimated condition numbers
    """
    if method == "svd":
        S = torch.linalg.svdvals(A)
        return S[..., 0] / (S[..., -1] + 1e-10)

    # Power iteration for largest and smallest singular values
    *batch_dims, m, n = A.shape

    # Largest singular value
    v = torch.randn(*batch_dims, n, 1, device=A.device, dtype=A.dtype)
    for _ in range(num_iterations):
        u = torch.matmul(A, v)
        u = u / (u.norm(dim=-2, keepdim=True) + 1e-10)
        v = torch.matmul(A.transpose(-2, -1), u)
        v = v / (v.norm(dim=-2, keepdim=True) + 1e-10)

    sigma_max = torch.matmul(u.transpose(-2, -1), torch.matmul(A, v))
    sigma_max = sigma_max.squeeze(-1).squeeze(-1).abs()

    # Smallest singular value (via inverse power iteration on A^T A)
    ATA = torch.matmul(A.transpose(-2, -1), A)
    v = torch.randn(*batch_dims, n, 1, device=A.device, dtype=A.dtype)

    for _ in range(num_iterations):
        # Solve (A^T A) v_new = v_old
        try:
            v_new = torch.linalg.solve(ATA + 1e-6 * torch.eye(n, device=A.device), v)
        except RuntimeError:
            # Fallback if solve fails
            return sigma_max * 1e6  # Return large condition number

        v = v_new / (v_new.norm(dim=-2, keepdim=True) + 1e-10)

    sigma_min = 1.0 / (torch.matmul(v.transpose(-2, -1), torch.matmul(ATA, v)) + 1e-10)
    sigma_min = sigma_min.squeeze(-1).squeeze(-1).abs().sqrt()

    return sigma_max / (sigma_min + 1e-10)


def block_diagonal(blocks: list) -> Tensor:
    """
    Create a block diagonal matrix from a list of blocks.

    Args:
        blocks: List of tensors representing diagonal blocks

    Returns:
        Block diagonal matrix
    """
    sizes = [b.shape[-1] for b in blocks]
    total_size = sum(sizes)
    device = blocks[0].device
    dtype = blocks[0].dtype

    # Handle batched case
    batch_dims = blocks[0].shape[:-2]
    result = torch.zeros(*batch_dims, total_size, total_size, device=device, dtype=dtype)

    offset = 0
    for block in blocks:
        size = block.shape[-1]
        result[..., offset:offset + size, offset:offset + size] = block
        offset += size

    return result


def tile_matrix(A: Tensor, tile_size: int) -> Tensor:
    """
    Reshape matrix into tiles for blocked algorithms.

    Args:
        A: Input matrix of shape (..., m, n)
        tile_size: Size of each tile

    Returns:
        Tiled matrix of shape (..., m//tile_size, n//tile_size, tile_size, tile_size)
    """
    *batch_dims, m, n = A.shape

    # Pad if necessary
    m_pad = (tile_size - m % tile_size) % tile_size
    n_pad = (tile_size - n % tile_size) % tile_size

    if m_pad > 0 or n_pad > 0:
        A = torch.nn.functional.pad(A, (0, n_pad, 0, m_pad))

    m_new, n_new = A.shape[-2:]
    num_tiles_m = m_new // tile_size
    num_tiles_n = n_new // tile_size

    # Reshape into tiles
    A = A.reshape(*batch_dims, num_tiles_m, tile_size, num_tiles_n, tile_size)
    A = A.permute(*range(len(batch_dims)), -4, -2, -3, -1)

    return A


def untile_matrix(A_tiled: Tensor, original_shape: Tuple[int, int]) -> Tensor:
    """
    Reshape tiled matrix back to original form.

    Args:
        A_tiled: Tiled matrix of shape (..., num_tiles_m, num_tiles_n, tile_size, tile_size)
        original_shape: Original (m, n) shape

    Returns:
        Matrix of shape (..., m, n)
    """
    *batch_dims, num_tiles_m, num_tiles_n, tile_size, _ = A_tiled.shape

    # Permute and reshape
    A = A_tiled.permute(*range(len(batch_dims)), -4, -2, -3, -1)
    A = A.reshape(*batch_dims, num_tiles_m * tile_size, num_tiles_n * tile_size)

    # Remove padding
    m, n = original_shape
    return A[..., :m, :n]


def masked_fill_inf(x: Tensor, mask: Tensor) -> Tensor:
    """
    Fill masked positions with -inf for attention masking.

    Args:
        x: Input tensor
        mask: Boolean mask (True = mask out)

    Returns:
        Masked tensor
    """
    return x.masked_fill(mask, float('-inf'))


def causal_mask(seq_len: int, device: torch.device = None) -> Tensor:
    """
    Create a causal (lower triangular) attention mask.

    Args:
        seq_len: Sequence length
        device: Target device

    Returns:
        Boolean mask of shape (seq_len, seq_len) where True = mask out
    """
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


def sliding_window_mask(
    seq_len: int,
    window_size: int,
    device: torch.device = None,
) -> Tensor:
    """
    Create a sliding window attention mask.

    Args:
        seq_len: Sequence length
        window_size: Size of attention window
        device: Target device

    Returns:
        Boolean mask of shape (seq_len, seq_len) where True = mask out
    """
    positions = torch.arange(seq_len, device=device)
    distance = positions.unsqueeze(0) - positions.unsqueeze(1)
    return distance.abs() > window_size

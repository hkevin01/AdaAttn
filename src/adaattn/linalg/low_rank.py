"""
Low-rank approximation utilities.

This module provides efficient low-rank decomposition and
approximation methods for attention matrices.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


def power_iteration(
    A: Tensor,
    num_iterations: int = 3,
    eps: float = 1e-10,
) -> Tuple[Tensor, Tensor]:
    """
    Compute the largest singular value and vectors using power iteration.

    Args:
        A: Input matrix of shape (..., m, n)
        num_iterations: Number of power iterations
        eps: Small constant for numerical stability

    Returns:
        Tuple of:
        - Largest singular value (...,)
        - Right singular vector (..., n)
    """
    *batch_dims, m, n = A.shape

    # Initialize random vector
    v = torch.randn(*batch_dims, n, 1, device=A.device, dtype=A.dtype)
    v = v / (v.norm(dim=-2, keepdim=True) + eps)

    for _ in range(num_iterations):
        # u = Av / ||Av||
        u = torch.matmul(A, v)
        u = u / (u.norm(dim=-2, keepdim=True) + eps)

        # v = A^T u / ||A^T u||
        v = torch.matmul(A.transpose(-2, -1), u)
        v = v / (v.norm(dim=-2, keepdim=True) + eps)

    # Compute singular value: sigma = u^T A v
    sigma = torch.matmul(u.transpose(-2, -1), torch.matmul(A, v))
    sigma = sigma.squeeze(-1).squeeze(-1)

    return sigma.abs(), v.squeeze(-1)


def randomized_svd(
    A: Tensor,
    rank: int,
    num_oversampling: int = 10,
    num_power_iterations: int = 2,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute truncated SVD using randomized algorithm.

    This is more efficient than full SVD for low-rank approximations.

    Args:
        A: Input matrix of shape (..., m, n)
        rank: Target rank
        num_oversampling: Oversampling parameter for accuracy
        num_power_iterations: Number of power iterations for accuracy

    Returns:
        Tuple of (U, S, Vh) where:
        - U: (..., m, rank)
        - S: (..., rank)
        - Vh: (..., rank, n)
    """
    *batch_dims, m, n = A.shape
    k = min(rank + num_oversampling, min(m, n))

    # Random projection matrix
    Omega = torch.randn(*batch_dims, n, k, device=A.device, dtype=A.dtype)

    # Form Y = A @ Omega
    Y = torch.matmul(A, Omega)

    # Power iterations for better accuracy
    for _ in range(num_power_iterations):
        Y = torch.matmul(A, torch.matmul(A.transpose(-2, -1), Y))

    # QR decomposition of Y
    Q, _ = torch.linalg.qr(Y)

    # Form B = Q^T @ A
    B = torch.matmul(Q.transpose(-2, -1), A)

    # SVD of B
    U_b, S, Vh = torch.linalg.svd(B, full_matrices=False)

    # Recover U
    U = torch.matmul(Q, U_b)

    # Truncate to target rank
    U = U[..., :rank]
    S = S[..., :rank]
    Vh = Vh[..., :rank, :]

    return U, S, Vh


def estimate_effective_rank(
    A: Tensor,
    threshold: float = 0.99,
    method: str = "entropy",
) -> Tensor:
    """
    Estimate the effective rank of a matrix.

    Args:
        A: Input matrix of shape (..., m, n)
        threshold: Cumulative energy threshold for rank estimation
        method: Method to use ("entropy", "energy", "count")

    Returns:
        Effective rank for each matrix in batch
    """
    # Compute singular values
    S = torch.linalg.svdvals(A)

    # Normalize
    S_normalized = S / (S.sum(dim=-1, keepdim=True) + 1e-10)

    if method == "entropy":
        # Shannon entropy-based rank
        entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum(dim=-1)
        effective_rank = torch.exp(entropy)

    elif method == "energy":
        # Energy-based rank (cumulative sum threshold)
        S_squared = S ** 2
        cumsum = torch.cumsum(S_squared, dim=-1)
        total_energy = S_squared.sum(dim=-1, keepdim=True)
        cumsum_normalized = cumsum / (total_energy + 1e-10)

        # Count how many singular values needed to reach threshold
        effective_rank = (cumsum_normalized < threshold).sum(dim=-1).float() + 1

    else:  # count
        # Simple count above threshold
        effective_rank = (S > threshold * S[..., 0:1]).sum(dim=-1).float()

    return effective_rank


def low_rank_approx(
    A: Tensor,
    rank: int,
    method: str = "svd",
) -> Tensor:
    """
    Compute low-rank approximation of a matrix.

    Args:
        A: Input matrix of shape (..., m, n)
        rank: Target rank
        method: Method to use ("svd", "randomized")

    Returns:
        Low-rank approximation of A
    """
    if method == "svd":
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        U = U[..., :rank]
        S = S[..., :rank]
        Vh = Vh[..., :rank, :]
    else:
        U, S, Vh = randomized_svd(A, rank)

    # Reconstruct: A ≈ U @ diag(S) @ Vh
    return torch.matmul(U * S.unsqueeze(-2), Vh)


def spectral_norm(
    A: Tensor,
    num_iterations: int = 3,
) -> Tensor:
    """
    Estimate spectral norm (largest singular value) of a matrix.

    Args:
        A: Input matrix of shape (..., m, n)
        num_iterations: Number of power iterations

    Returns:
        Spectral norm for each matrix in batch
    """
    sigma, _ = power_iteration(A, num_iterations)
    return sigma

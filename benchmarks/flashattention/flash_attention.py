"""FlashAttention baseline implementation for benchmarking.

This module provides a clean interface to FlashAttention v2 for comparison
with AdaAttn implementations. Uses official flash-attn package when available,
falls back to PyTorch SDPA or custom implementation.
"""

import math
from typing import Optional, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    warnings.warn("flash-attn not available, falling back to PyTorch SDPA")


class FlashAttentionBaseline(nn.Module):
    """FlashAttention baseline for benchmarking.
    
    Provides a clean interface to FlashAttention with fallback options.
    Used as the primary baseline for comparing AdaAttn performance.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        causal: bool = False,
        use_flash_attn: bool = True,
        bias: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.causal = causal
        self.use_flash_attn = use_flash_attn and FLASH_ATTN_AVAILABLE
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass using FlashAttention or fallback.
        
        Args:
            query: Query tensor [batch, seq_len, embed_dim]
            key: Key tensor [batch, seq_len, embed_dim]
            value: Value tensor [batch, seq_len, embed_dim]
            attention_mask: Optional mask [batch, seq_len] or [batch, seq_len, seq_len]
            need_weights: Whether to return attention weights (forces fallback)
            
        Returns:
            Tuple of (output, attention_weights)
            attention_weights is None if using FlashAttention
        """
        batch_size, seq_len, _ = query.shape
        
        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Use FlashAttention if available and not needing weights
        if self.use_flash_attn and not need_weights and attention_mask is None:
            # FlashAttention expects [batch, seq_len, num_heads, head_dim]
            attn_output = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=self.causal
            )
            attn_weights = None
            
        else:
            # Fallback to PyTorch SDPA or manual implementation
            attn_output, attn_weights = self._pytorch_attention(
                q, k, v, attention_mask, need_weights
            )
        
        # Reshape and project output
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)
        
        return output, attn_weights
    
    def _pytorch_attention(
        self,
        q: torch.Tensor,  # [batch, seq_len, num_heads, head_dim]
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        need_weights: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Fallback attention using PyTorch SDPA or manual computation."""
        
        # Transpose to [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Try PyTorch SDPA first (if available and compatible)
        if hasattr(F, 'scaled_dot_product_attention') and not need_weights:
            try:
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attention_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=self.causal and attention_mask is None
                )
                return attn_output.transpose(1, 2), None
            except Exception:
                pass  # Fall through to manual implementation
        
        # Manual attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if self.causal:
            seq_len = scores.size(-1)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                # [batch, seq_len] -> [batch, 1, 1, seq_len]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.ndim == 3:
                # [batch, seq_len, seq_len] -> [batch, 1, seq_len, seq_len]
                attention_mask = attention_mask.unsqueeze(1)
            
            if attention_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attention_mask, float('-inf'))
            else:
                scores = scores + attention_mask
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Transpose back to [batch, seq_len, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2)
        
        if need_weights:
            return attn_output, attn_weights
        else:
            return attn_output, None
    
    def extra_repr(self) -> str:
        return f'embed_dim={self.embed_dim}, num_heads={self.num_heads}, ' \
               f'dropout={self.dropout}, causal={self.causal}, ' \
               f'use_flash_attn={self.use_flash_attn}'

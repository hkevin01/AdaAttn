"""
Transformer implementation with AdaAttn.

This module provides a complete transformer architecture using
adaptive attention mechanisms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from adaattn.attention import AdaAttn, AdaAttnConfig


@dataclass
class AdaAttnTransformerConfig:
    """Configuration for AdaAttn Transformer."""

    # Model architecture
    vocab_size: int = 50257
    max_seq_len: int = 2048
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    head_dim: Optional[int] = None  # Defaults to hidden_dim // num_heads
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    # Attention configuration
    attention_config: AdaAttnConfig = field(default_factory=AdaAttnConfig)

    # Training settings
    tie_word_embeddings: bool = True
    use_cache: bool = True

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_dim // self.num_heads

        assert self.hidden_dim % self.num_heads == 0, (
            f"hidden_dim ({self.hidden_dim}) must be divisible by "
            f"num_heads ({self.num_heads})"
        )


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class MLP(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        intermediate_dim = intermediate_dim or int(hidden_dim * 4 * 2 / 3)
        # Round to multiple of 256 for efficiency
        intermediate_dim = ((intermediate_dim + 255) // 256) * 256

        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class AdaAttnTransformerLayer(nn.Module):
    """Single transformer layer with AdaAttn."""

    def __init__(self, config: AdaAttnTransformerConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.attention = AdaAttn(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            config=config.attention_config,
        )

        self.mlp = MLP(
            hidden_dim=config.hidden_dim,
            intermediate_dim=int(config.hidden_dim * config.mlp_ratio),
            dropout=config.dropout,
        )

        self.attn_norm = RMSNorm(config.hidden_dim)
        self.mlp_norm = RMSNorm(config.hidden_dim)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)

        attn_output, present_key_value = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        hidden_states = residual + attn_output

        # MLP with residual
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        return hidden_states, present_key_value


class AdaAttnTransformer(nn.Module):
    """
    Transformer model with adaptive attention.

    This model can be used as a language model or as a backbone
    for other tasks.
    """

    def __init__(self, config: AdaAttnTransformerConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Position embeddings (rotary position encoding is applied in attention)
        self.embed_positions = nn.Embedding(config.max_seq_len, config.hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            AdaAttnTransformerLayer(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        # Output normalization
        self.norm = RMSNorm(config.hidden_dim)

        # Output head
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Tie embeddings
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        return_dict: bool = True,
    ) -> dict:
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Add position embeddings
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        position_embeds = self.embed_positions(position_ids)
        hidden_states = hidden_states + position_embeds

        # Process through layers
        presents = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values else None

            hidden_states, present = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )

            if use_cache:
                presents.append(present)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        # Compute logits
        logits = self.lm_head(hidden_states)

        if return_dict:
            return {
                "logits": logits,
                "hidden_states": hidden_states,
                "past_key_values": presents,
            }

        return logits

    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Tensor:
        """Generate tokens autoregressively."""
        generated = input_ids
        past_key_values = None

        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.forward(
                generated if past_key_values is None else generated[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]

            # Apply temperature
            logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)

        return generated

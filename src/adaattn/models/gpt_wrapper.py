"""
GPT wrapper for AdaAttn integration.

This module provides utilities to replace standard attention
in pre-trained GPT models with AdaAttn.
"""

from __future__ import annotations

import logging
from typing import Optional, Type

import torch
import torch.nn as nn
from torch import Tensor

from adaattn.attention import AdaAttn, AdaAttnConfig

logger = logging.getLogger(__name__)


def replace_attention_with_adaattn(
    model: nn.Module,
    config: Optional[AdaAttnConfig] = None,
    target_modules: Optional[list] = None,
) -> nn.Module:
    """
    Replace attention modules in a model with AdaAttn.

    This function recursively traverses the model and replaces
    attention layers with AdaAttn equivalents.

    Args:
        model: The model to modify
        config: AdaAttn configuration
        target_modules: List of module names to replace (e.g., ["attn", "attention"])

    Returns:
        Modified model with AdaAttn layers
    """
    if config is None:
        config = AdaAttnConfig()

    if target_modules is None:
        target_modules = ["attn", "attention", "self_attn", "self_attention"]

    replacements = 0

    for name, module in model.named_modules():
        # Check if this is an attention module to replace
        module_name = name.split(".")[-1]

        if module_name in target_modules:
            # Try to infer dimensions from the existing module
            hidden_dim = None
            num_heads = None

            # Look for common attribute names
            for attr in ["embed_dim", "hidden_size", "d_model", "dim"]:
                if hasattr(module, attr):
                    hidden_dim = getattr(module, attr)
                    break

            for attr in ["num_heads", "n_head", "num_attention_heads"]:
                if hasattr(module, attr):
                    num_heads = getattr(module, attr)
                    break

            if hidden_dim is None or num_heads is None:
                logger.warning(
                    f"Could not infer dimensions for {name}, skipping"
                )
                continue

            # Create AdaAttn replacement
            ada_attn = AdaAttn(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                config=config,
            )

            # Copy weights if possible
            ada_attn = _copy_attention_weights(module, ada_attn)

            # Replace the module
            parent = _get_parent_module(model, name)
            setattr(parent, module_name, ada_attn)
            replacements += 1

            logger.info(f"Replaced {name} with AdaAttn")

    logger.info(f"Replaced {replacements} attention modules with AdaAttn")
    return model


def _get_parent_module(model: nn.Module, name: str) -> nn.Module:
    """Get the parent module for a given module name."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent


def _copy_attention_weights(
    source: nn.Module,
    target: AdaAttn,
) -> AdaAttn:
    """Copy attention weights from source to target AdaAttn module."""
    try:
        # Try to find QKV projections
        if hasattr(source, "q_proj"):
            target.q_proj.weight.data.copy_(source.q_proj.weight.data)
            target.k_proj.weight.data.copy_(source.k_proj.weight.data)
            target.v_proj.weight.data.copy_(source.v_proj.weight.data)
        elif hasattr(source, "query"):
            target.q_proj.weight.data.copy_(source.query.weight.data)
            target.k_proj.weight.data.copy_(source.key.weight.data)
            target.v_proj.weight.data.copy_(source.value.weight.data)
        elif hasattr(source, "c_attn"):
            # GPT-2 style combined QKV
            qkv_weight = source.c_attn.weight.data
            hidden_dim = qkv_weight.shape[0] // 3
            target.q_proj.weight.data.copy_(qkv_weight[:hidden_dim])
            target.k_proj.weight.data.copy_(qkv_weight[hidden_dim:2*hidden_dim])
            target.v_proj.weight.data.copy_(qkv_weight[2*hidden_dim:])

        # Try to find output projection
        if hasattr(source, "out_proj"):
            target.o_proj.weight.data.copy_(source.out_proj.weight.data)
        elif hasattr(source, "o_proj"):
            target.o_proj.weight.data.copy_(source.o_proj.weight.data)
        elif hasattr(source, "c_proj"):
            target.o_proj.weight.data.copy_(source.c_proj.weight.data)

        logger.debug("Successfully copied attention weights")

    except Exception as e:
        logger.warning(f"Could not copy weights: {e}")

    return target


class GPTWithAdaAttn(nn.Module):
    """
    Wrapper for GPT models with AdaAttn.

    This class wraps a pre-trained GPT model and replaces
    its attention layers with AdaAttn.
    """

    def __init__(
        self,
        base_model: nn.Module,
        config: Optional[AdaAttnConfig] = None,
    ):
        super().__init__()

        self.config = config or AdaAttnConfig()
        self.model = replace_attention_with_adaattn(
            base_model,
            config=self.config,
        )

        # Track attention statistics
        self.attention_stats = {}

    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped model."""
        return self.model(*args, **kwargs)

    def get_attention_stats(self) -> dict:
        """
        Collect attention statistics from all AdaAttn layers.

        Returns:
            Dictionary with per-layer statistics
        """
        stats = {}

        for name, module in self.model.named_modules():
            if isinstance(module, AdaAttn):
                layer_stats = module.get_stats()
                if layer_stats:
                    stats[name] = layer_stats

        return stats

    def reset_attention_stats(self):
        """Reset attention statistics for all layers."""
        for module in self.model.modules():
            if isinstance(module, AdaAttn):
                module.reset_stats()

    def set_adaptive_mode(self, enable: bool = True):
        """Enable or disable adaptive behavior in all AdaAttn layers."""
        for module in self.model.modules():
            if isinstance(module, AdaAttn):
                module.config.enable_adaptive_rank = enable
                module.config.enable_adaptive_precision = enable

    def get_memory_savings(self) -> dict:
        """
        Estimate memory savings from adaptive attention.

        Returns:
            Dictionary with memory statistics
        """
        total_theoretical = 0
        total_actual = 0

        for name, module in self.model.named_modules():
            if isinstance(module, AdaAttn):
                stats = module.get_stats()
                if stats:
                    # Theoretical: seq_len^2 * num_heads * batch
                    # Actual: depends on rank used
                    total_theoretical += stats.get("theoretical_memory", 0)
                    total_actual += stats.get("actual_memory", 0)

        savings = 1.0 - (total_actual / (total_theoretical + 1e-10))

        return {
            "theoretical_memory_mb": total_theoretical / 1e6,
            "actual_memory_mb": total_actual / 1e6,
            "savings_percent": savings * 100,
        }

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        config: Optional[AdaAttnConfig] = None,
        **kwargs,
    ) -> "GPTWithAdaAttn":
        """
        Load a pre-trained model and wrap it with AdaAttn.

        Args:
            model_name_or_path: HuggingFace model name or path
            config: AdaAttn configuration
            **kwargs: Additional arguments for model loading

        Returns:
            GPTWithAdaAttn instance
        """
        try:
            from transformers import AutoModelForCausalLM

            base_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                **kwargs,
            )

            return cls(base_model, config=config)

        except ImportError:
            raise ImportError(
                "transformers library is required for from_pretrained. "
                "Install it with: pip install transformers"
            )

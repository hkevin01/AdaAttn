"""
FlashAttention integration for AdaAttn.
"""

import torch
from typing import Optional, Tuple, Union
import logging
from ..attention.base import PrecisionMode

logger = logging.getLogger(__name__)

try:
    import flash_attn
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
    logger.info(f"FlashAttention available: version {flash_attn.__version__}")
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    logger.warning("FlashAttention not available. Install with: pip install flash-attn")


class FlashAttentionConfig:
    """Configuration for FlashAttention."""
    
    def __init__(
        self,
        enable_flash: bool = True,
        causal: bool = False,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        return_attn_probs: bool = False,
        deterministic: bool = False,
    ):
        self.enable_flash = enable_flash and FLASH_ATTN_AVAILABLE
        self.causal = causal
        self.dropout_p = dropout_p
        self.softmax_scale = softmax_scale
        self.return_attn_probs = return_attn_probs
        self.deterministic = deterministic


def flash_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    config: FlashAttentionConfig,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    FlashAttention forward pass with automatic fallback.
    
    Args:
        query: Query tensor [batch, seq_len, num_heads, head_dim]
        key: Key tensor [batch, seq_len, num_heads, head_dim] 
        value: Value tensor [batch, seq_len, num_heads, head_dim]
        config: FlashAttention configuration
        attention_mask: Optional attention mask
    
    Returns:
        output: Attention output [batch, seq_len, num_heads, head_dim]
        attn_weights: Attention weights if requested, else None
    """
    if not config.enable_flash:
        return _fallback_attention(query, key, value, config, attention_mask)
    
    # FlashAttention expects specific tensor format and device placement
    if not query.is_cuda:
        logger.warning("FlashAttention requires CUDA tensors. Using fallback.")
        return _fallback_attention(query, key, value, config, attention_mask)
    
    # Check tensor dimensions and dtypes
    if query.dim() != 4:
        logger.warning("FlashAttention requires 4D tensors. Using fallback.")
        return _fallback_attention(query, key, value, config, attention_mask)
    
    if query.dtype not in [torch.float16, torch.bfloat16]:
        logger.warning("FlashAttention requires FP16/BF16. Using fallback.")
        return _fallback_attention(query, key, value, config, attention_mask)
    
    try:
        # FlashAttention expects [batch, seq_len, num_heads, head_dim]
        batch_size, seq_len, num_heads, head_dim = query.shape
        
        # Handle attention mask - FlashAttention doesn't support arbitrary masks
        if attention_mask is not None and not config.causal:
            logger.warning("FlashAttention doesn't support custom masks. Using fallback.")
            return _fallback_attention(query, key, value, config, attention_mask)
        
        # Call FlashAttention
        output = flash_attn_func(
            q=query,
            k=key, 
            v=value,
            dropout_p=config.dropout_p,
            softmax_scale=config.softmax_scale,
            causal=config.causal,
            return_attn_probs=config.return_attn_probs,
            deterministic=config.deterministic,
        )
        
        if config.return_attn_probs:
            attn_output, attn_weights = output
            return attn_output, attn_weights
        else:
            return output, None
            
    except Exception as e:
        logger.warning(f"FlashAttention failed: {e}. Using fallback.")
        return _fallback_attention(query, key, value, config, attention_mask)


def _fallback_attention(
    query: torch.Tensor,
    key: torch.Tensor, 
    value: torch.Tensor,
    config: FlashAttentionConfig,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Fallback attention implementation when FlashAttention is not available."""
    
    # Ensure proper dimensions [batch, seq_len, num_heads, head_dim]
    if query.dim() == 3:
        # Add num_heads dimension
        query = query.unsqueeze(2)
        key = key.unsqueeze(2)
        value = value.unsqueeze(2)
    
    batch_size, seq_len_q, num_heads, head_dim = query.shape
    seq_len_k = key.shape[1]
    
    # Transpose to [batch, num_heads, seq_len, head_dim] for torch operations
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    
    scale = config.softmax_scale or (head_dim ** -0.5)
    
    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Apply causal mask
    if config.causal:
        causal_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=query.device),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))
    
    # Apply attention mask
    if attention_mask is not None:
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        
        if attention_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attention_mask, float('-inf'))
        else:
            scores = scores + attention_mask
    
    # Compute attention weights
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Apply dropout
    if config.dropout_p > 0.0 and query.training:
        attn_weights = torch.dropout(attn_weights, config.dropout_p, train=True)
    
    # Compute output
    output = torch.matmul(attn_weights, v)
    
    # Transpose back to [batch, seq_len, num_heads, head_dim]
    output = output.transpose(1, 2)
    
    if config.return_attn_probs:
        return output, attn_weights.transpose(1, 2)
    else:
        return output, None


class AdaptiveFlashAttention(torch.nn.Module):
    """
    Adaptive attention layer with FlashAttention optimization.
    Automatically selects between FlashAttention and fallback based on constraints.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = False,
        precision_mode: Union[str, PrecisionMode] = "auto",
        enable_flash: bool = True,
        chunk_size: int = 2048,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.causal = causal
        self.chunk_size = chunk_size
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # QKV projections
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.o_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # FlashAttention config
        self.flash_config = FlashAttentionConfig(
            enable_flash=enable_flash and FLASH_ATTN_AVAILABLE,
            causal=causal,
            dropout_p=dropout,
        )
        
        # Precision management
        if isinstance(precision_mode, str):
            if precision_mode == "auto":
                self.precision_mode = self._auto_detect_precision()
            else:
                self.precision_mode = getattr(PrecisionMode, precision_mode.upper())
        else:
            self.precision_mode = precision_mode
        
        self._init_weights()
        
        # Statistics tracking
        self.register_buffer('_total_calls', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_flash_calls', torch.zeros(1, dtype=torch.long))
        
    def _auto_detect_precision(self) -> PrecisionMode:
        """Auto-detect optimal precision based on hardware."""
        if not torch.cuda.is_available():
            return PrecisionMode.FP32
        
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        
        # Ampere and newer support BF16
        if props.major >= 8:
            return PrecisionMode.BF16
        # Volta and newer support FP16
        elif props.major >= 7:
            return PrecisionMode.FP16
        else:
            return PrecisionMode.FP32
    
    def _init_weights(self):
        """Initialize layer weights."""
        torch.nn.init.xavier_uniform_(self.q_proj.weight)
        torch.nn.init.xavier_uniform_(self.k_proj.weight) 
        torch.nn.init.xavier_uniform_(self.v_proj.weight)
        torch.nn.init.xavier_uniform_(self.o_proj.weight)
        
        if self.q_proj.bias is not None:
            torch.nn.init.zeros_(self.q_proj.bias)
            torch.nn.init.zeros_(self.k_proj.bias)
            torch.nn.init.zeros_(self.v_proj.bias)
            torch.nn.init.zeros_(self.o_proj.bias)
    
    def _should_use_chunked(self, seq_len: int) -> bool:
        """Determine if chunked attention should be used."""
        if seq_len <= self.chunk_size:
            return False
        
        # Estimate memory usage
        batch_size = 1  # Conservative estimate
        memory_estimate = batch_size * self.num_heads * seq_len * seq_len * 4
        
        # If estimated memory > 2GB, use chunked
        return memory_estimate > (2 * 1024**3)
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with adaptive FlashAttention."""
        
        batch_size, seq_len, _ = query.shape
        
        # Self-attention if key/value not provided
        if key is None:
            key = query
        if value is None:
            value = key
        
        # Project to QKV
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape to [batch, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, k.shape[1], self.num_heads, self.head_dim)
        v = v.view(batch_size, v.shape[1], self.num_heads, self.head_dim)
        
        # Convert to target precision for computation
        target_dtype = self.precision_mode.to_dtype()
        if target_dtype != q.dtype:
            q = q.to(target_dtype)
            k = k.to(target_dtype)
            v = v.to(target_dtype)
        
        # Update FlashAttention config
        self.flash_config.dropout_p = self.dropout if self.training else 0.0
        self.flash_config.return_attn_probs = return_attention_weights
        self.flash_config.softmax_scale = (self.head_dim ** -0.5)
        
        # Statistics
        self._total_calls += 1
        
        # Choose attention implementation
        if self._should_use_chunked(seq_len):
            # Use chunked attention for very long sequences
            output, attn_weights = self._chunked_attention(q, k, v, attention_mask)
        else:
            # Try FlashAttention first
            output, attn_weights = flash_attention_forward(
                q, k, v, self.flash_config, attention_mask
            )
            if self.flash_config.enable_flash and attn_weights is None:
                self._flash_calls += 1
        
        # Project output
        output = output.view(batch_size, seq_len, self.embed_dim)
        output = self.o_proj(output.to(query.dtype))
        
        return output, attn_weights
    
    def _chunked_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Chunked attention for memory efficiency."""
        batch_size, seq_len_q, num_heads, head_dim = query.shape
        seq_len_k = key.shape[1]
        
        output = torch.zeros_like(query)
        attn_weights = None
        
        if self.flash_config.return_attn_probs:
            attn_weights = torch.zeros(
                batch_size, seq_len_q, num_heads, seq_len_k,
                dtype=query.dtype, device=query.device
            )
        
        for i in range(0, seq_len_q, self.chunk_size):
            end_i = min(i + self.chunk_size, seq_len_q)
            q_chunk = query[:, i:end_i]
            
            chunk_mask = attention_mask[:, i:end_i] if attention_mask is not None else None
            
            chunk_out, chunk_weights = flash_attention_forward(
                q_chunk, key, value, self.flash_config, chunk_mask
            )
            
            output[:, i:end_i] = chunk_out
            
            if chunk_weights is not None:
                attn_weights[:, i:end_i] = chunk_weights
        
        return output, attn_weights
    
    def get_flash_statistics(self) -> dict:
        """Get FlashAttention usage statistics."""
        total = self._total_calls.item()
        flash = self._flash_calls.item()
        
        return {
            "total_calls": total,
            "flash_calls": flash,
            "flash_ratio": flash / total if total > 0 else 0.0,
            "fallback_calls": total - flash,
            "flash_available": FLASH_ATTN_AVAILABLE,
        }
    
    def reset_statistics(self):
        """Reset usage statistics."""
        self._total_calls.zero_()
        self._flash_calls.zero_()

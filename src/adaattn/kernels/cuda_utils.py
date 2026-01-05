"""
CUDA utilities and kernel management for AdaAttn.
"""

import torch
import torch.cuda
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CUDAManager:
    """Manages CUDA context and kernel compilation."""
    
    _instance = None
    _kernels_compiled = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.device_capability = None
        self.max_shared_memory = None
        self.max_threads_per_block = None
        self.warp_size = 32
        self._initialize_cuda_info()
    
    def _initialize_cuda_info(self):
        """Initialize CUDA device information."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available")
            return
        
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        
        self.device_capability = (properties.major, properties.minor)
        self.max_shared_memory = properties.shared_memory_per_block
        self.max_threads_per_block = properties.max_threads_per_block
        
        logger.info(f"CUDA Device: {properties.name}")
        logger.info(f"Compute Capability: {self.device_capability}")
        logger.info(f"Max Shared Memory: {self.max_shared_memory} bytes")
        logger.info(f"Max Threads per Block: {self.max_threads_per_block}")
    
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return torch.cuda.is_available()
    
    def get_optimal_block_size(self, seq_len: int, embed_dim: int) -> Tuple[int, int]:
        """Calculate optimal CUDA block dimensions."""
        if not self.is_cuda_available():
            return (1, 1)
        
        # Target around 256-512 threads per block for good occupancy
        target_threads = 256
        
        # For attention, we typically parallelize over sequence length
        if seq_len <= 32:
            block_x = seq_len
            block_y = min(target_threads // seq_len, embed_dim)
        else:
            block_x = 32
            block_y = min(target_threads // 32, embed_dim, 32)
        
        return (block_x, block_y)
    
    def supports_bf16(self) -> bool:
        """Check if device supports BF16."""
        if not self.is_cuda_available() or self.device_capability is None:
            return False
        
        # BF16 requires compute capability >= 8.0 (Ampere)
        major, minor = self.device_capability
        return major >= 8
    
    def supports_fp16(self) -> bool:
        """Check if device supports FP16."""
        if not self.is_cuda_available() or self.device_capability is None:
            return False
        
        # FP16 requires compute capability >= 5.3
        major, minor = self.device_capability
        return major > 5 or (major == 5 and minor >= 3)
    
    def get_memory_info(self) -> Tuple[int, int]:
        """Get GPU memory information."""
        if not self.is_cuda_available():
            return (0, 0)
        
        free, total = torch.cuda.mem_get_info()
        return (free, total)


def fused_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor, 
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Fused scaled dot-product attention using PyTorch's native implementation
    when available, falling back to manual implementation.
    """
    batch_size, num_heads, seq_len_q, head_dim = query.shape
    seq_len_k = key.shape[2]
    
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)
    
    # Use PyTorch 2.0+ fused attention if available
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        try:
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, 
                attn_mask=attn_mask,
                dropout_p=dropout_p if query.training else 0.0,
                is_causal=is_causal,
                scale=scale
            )
        except Exception as e:
            logger.warning(f"Fused attention failed: {e}. Falling back to manual implementation.")
    
    # Manual implementation fallback
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=query.device),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))
    
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float('-inf'))
        else:
            scores = scores + attn_mask
    
    attn_weights = torch.softmax(scores, dim=-1)
    
    if dropout_p > 0.0 and query.training:
        attn_weights = torch.dropout(attn_weights, dropout_p, train=True)
    
    return torch.matmul(attn_weights, value)


def adaptive_attention_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    rank_proj_down: Optional[torch.Tensor] = None,
    use_low_rank: bool = False,
    precision_mode: str = "fp16",
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Adaptive attention kernel that switches between dense and low-rank computation
    with precision optimization.
    """
    original_dtype = query.dtype
    
    # Convert to target precision
    target_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(precision_mode, torch.float16)
    
    q = query.to(target_dtype)
    k = key.to(target_dtype)
    v = value.to(target_dtype)
    
    if use_low_rank and rank_proj_down is not None:
        # Low-rank attention
        batch_size, seq_len, embed_dim = q.shape
        num_heads = q.shape[1] if q.dim() == 4 else 1
        
        if q.dim() == 3:
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
        
        # Project to low-rank space
        q_proj = torch.matmul(q, rank_proj_down.to(target_dtype))
        k_proj = torch.matmul(k, rank_proj_down.to(target_dtype))
        
        output = fused_scaled_dot_product_attention(
            q_proj, k_proj, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale
        )
    else:
        # Dense attention
        output = fused_scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale
        )
    
    # Return attention weights as well (dummy for now)
    attn_weights = torch.empty(0, device=query.device, dtype=target_dtype)
    
    return output.to(original_dtype), attn_weights.to(original_dtype)


def memory_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    chunk_size: int = 1024,
    **kwargs
) -> torch.Tensor:
    """
    Memory-efficient attention using chunked computation.
    Reduces memory usage for long sequences.
    """
    batch_size, num_heads, seq_len_q, head_dim = query.shape
    seq_len_k = key.shape[2]
    
    if seq_len_q <= chunk_size and seq_len_k <= chunk_size:
        # Small enough, use regular attention
        return fused_scaled_dot_product_attention(query, key, value, **kwargs)
    
    # Chunked computation
    output = torch.zeros_like(query)
    
    for i in range(0, seq_len_q, chunk_size):
        end_i = min(i + chunk_size, seq_len_q)
        q_chunk = query[:, :, i:end_i]
        
        chunk_output = fused_scaled_dot_product_attention(
            q_chunk, key, value, **kwargs
        )
        output[:, :, i:end_i] = chunk_output
    
    return output


class AttentionKernelManager:
    """Manages different attention kernel implementations."""
    
    def __init__(self):
        self.cuda_manager = CUDAManager()
        self.kernel_cache = {}
    
    def get_optimal_kernel(
        self,
        seq_len: int,
        embed_dim: int,
        num_heads: int,
        use_low_rank: bool = False,
        memory_budget: Optional[int] = None
    ) -> str:
        """Select optimal kernel based on input characteristics."""
        
        if not self.cuda_manager.is_cuda_available():
            return "cpu_fallback"
        
        # Memory considerations
        if memory_budget is not None:
            estimated_memory = seq_len * seq_len * num_heads * 4  # rough estimate
            if estimated_memory > memory_budget:
                return "memory_efficient"
        
        # Sequence length considerations  
        if seq_len >= 4096:
            return "memory_efficient"
        elif seq_len >= 1024:
            return "fused_attention"
        elif use_low_rank:
            return "adaptive_kernel"
        else:
            return "fused_attention"
    
    def execute_kernel(
        self,
        kernel_name: str,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute the specified kernel."""
        
        if kernel_name == "fused_attention":
            output = fused_scaled_dot_product_attention(query, key, value, **kwargs)
            weights = torch.empty(0, device=query.device)
            return output, weights
            
        elif kernel_name == "adaptive_kernel":
            return adaptive_attention_kernel(query, key, value, **kwargs)
            
        elif kernel_name == "memory_efficient":
            output = memory_efficient_attention(query, key, value, **kwargs)
            weights = torch.empty(0, device=query.device)
            return output, weights
            
        else:
            # CPU fallback
            scale = kwargs.get('scale', 1.0 / (query.shape[-1] ** 0.5))
            scores = torch.matmul(query, key.transpose(-2, -1)) * scale
            weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(weights, value)
            return output, weights


# Global kernel manager instance
kernel_manager = AttentionKernelManager()

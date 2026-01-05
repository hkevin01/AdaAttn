"""
End-to-end transformer integration example using AdaAttn.
Demonstrates full transformer block with adaptive attention.
"""

import torch
import torch.nn as nn
from adaattn.attention.adaattn import AdaAttention
from adaattn.attention.base import PrecisionMode


class TransformerBlock(nn.Module):
    """Transformer block with AdaAttn."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        enable_adaptive_rank: bool = True,
        enable_adaptive_precision: bool = True,
    ):
        super().__init__()
        
        # Adaptive attention layer
        self.attention = AdaAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            enable_adaptive_rank=enable_adaptive_rank,
            enable_adaptive_precision=enable_adaptive_precision,
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor, attention_mask=None):
        """Forward pass with residual connections."""
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, attention_mask=attention_mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class AdaptiveTransformer(nn.Module):
    """Full transformer model with AdaAttn."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        enable_adaptive_rank: bool = True,
        enable_adaptive_precision: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                enable_adaptive_rank=enable_adaptive_rank,
                enable_adaptive_precision=enable_adaptive_precision,
            )
            for _ in range(num_layers)
        ])
        
        # Output head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Tie weights
        self.head.weight = self.token_embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, attention_mask=None):
        """Forward pass."""
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos_ids)
        
        x = self.embed_dropout(token_emb + pos_emb)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        
        # Output
        x = self.norm(x)
        logits = self.head(x)
        
        return logits
    
    def get_attention_statistics(self):
        """Get statistics from all attention layers."""
        stats = []
        for i, block in enumerate(self.blocks):
            layer_stats = block.attention.get_statistics()
            layer_stats['layer'] = i
            stats.append(layer_stats)
        return stats


def demo_transformer():
    """Demonstrate transformer with AdaAttn."""
    print("=" * 80)
    print("AdaAttn Transformer Integration Demo")
    print("=" * 80)
    
    # Model configuration
    vocab_size = 1000
    embed_dim = 256
    num_heads = 8
    num_layers = 4
    ff_dim = 1024
    batch_size = 4
    seq_len = 128
    
    print(f"\nModel Configuration:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Embed dim: {embed_dim}")
    print(f"  Num heads: {num_heads}")
    print(f"  Num layers: {num_layers}")
    print(f"  FF dim: {ff_dim}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    
    # Create model
    print("\nCreating adaptive transformer...")
    model = AdaptiveTransformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        max_seq_len=512,
        enable_adaptive_rank=True,
        enable_adaptive_precision=True,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create sample input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    print("\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    
    # Get attention statistics
    print("\nAttention Statistics per Layer:")
    print("-" * 80)
    stats = model.get_attention_statistics()
    for layer_stats in stats:
        layer = layer_stats['layer']
        print(f"\nLayer {layer}:")
        print(f"  Calls: {layer_stats['call_count']}")
        print(f"  Low-rank ratio: {layer_stats['low_rank_ratio']:.2%}")
        print(f"  Precision distribution:")
        for prec, ratio in layer_stats['precision_distribution'].items():
            print(f"    {prec}: {ratio:.2%}")
    
    # Test training mode
    print("\n" + "=" * 80)
    print("Training Mode Test")
    print("=" * 80)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create sample batch
    input_ids = torch.randint(0, vocab_size, (2, 64))
    target_ids = torch.randint(0, vocab_size, (2, 64))
    
    print("\nPerforming training step...")
    optimizer.zero_grad()
    logits = model(input_ids)
    
    # Compute loss (simple cross-entropy)
    loss = nn.functional.cross_entropy(
        logits.view(-1, vocab_size),
        target_ids.view(-1)
    )
    
    print(f"  Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    print("  Gradients computed successfully")
    print("  Optimizer step completed")
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


def benchmark_configurations():
    """Compare different attention configurations."""
    print("\n" + "=" * 80)
    print("Benchmarking Different Configurations")
    print("=" * 80)
    
    configs = [
        ("Dense only", False, False),
        ("Adaptive rank", True, False),
        ("Adaptive precision", False, True),
        ("Both adaptive", True, True),
    ]
    
    vocab_size = 1000
    embed_dim = 256
    num_heads = 8
    num_layers = 2
    ff_dim = 1024
    batch_size = 4
    seq_len = 128
    
    results = []
    
    for name, enable_rank, enable_prec in configs:
        print(f"\nTesting: {name}")
        
        model = AdaptiveTransformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            enable_adaptive_rank=enable_rank,
            enable_adaptive_precision=enable_prec,
        )
        
        model.eval()
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Warmup
        with torch.no_grad():
            _ = model(input_ids)
        
        # Benchmark
        import time
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_ids)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = (time.perf_counter() - start) / 10 * 1000  # ms
        
        # Get stats
        stats = model.get_attention_statistics()
        avg_low_rank = sum(s['low_rank_ratio'] for s in stats) / len(stats)
        
        print(f"  Time: {elapsed:.2f}ms")
        print(f"  Avg low-rank ratio: {avg_low_rank:.2%}")
        
        results.append((name, elapsed, avg_low_rank))
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'Configuration':<20} {'Time (ms)':<12} {'Low-Rank %':<12}")
    print("-" * 80)
    for name, time_ms, lr_ratio in results:
        print(f"{name:<20} {time_ms:<12.2f} {lr_ratio * 100:<12.1f}")


if __name__ == "__main__":
    demo_transformer()
    benchmark_configurations()

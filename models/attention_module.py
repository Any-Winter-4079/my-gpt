import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Multi-Head Self-Attention module with causal masking.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections for Q, K, V
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj.RESIDUAL_SCALE_INIT = 1  # Mark this for weight initialization scaling
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """Compute self-attention over input sequence x using F.scaled_dot_product_attention"""
        batch_size, seq_len, embed_dim = x.shape
        assert embed_dim == self.embed_dim, "Input embedding dim does not match model"
        
        # Compute Q, K, V (batch, seq_len, 3 * embed_dim) -> (batch, seq_len, embed_dim) * 3
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2), qkv)
        
        # Use PyTorch's optimized scaled_dot_product_attention
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0.0)
        
        # Merge heads back
        output = output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        
        # Final output projection
        return self.out_proj(output)

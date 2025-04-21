import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mlp_module import MLP
from models.attention_module import CausalSelfAttention

class Block(nn.Module):
    """
    Decoder Block consisting of:
    - Multi-Head Self-Attention with causal masking
    - Feedforward MLP with up projection
    - LayerNorm before each sublayer
    - Residual connections
    """
    def __init__(self, embed_dim, num_heads, up_proj_factor, dropout=0.1):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, up_proj_factor, dropout=dropout)
    
    def forward(self, x):        
        x = x + self.attn(self.ln1(x))
        
        x = x + self.mlp(self.ln2(x))
        
        return x

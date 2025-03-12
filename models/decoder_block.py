import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mlp_module import MLP
from models.attention_module import Attention

class Block(nn.Module):
    """
    Transformer Decoder Block consisting of:
    - Multi-Head Self-Attention
    - Feedforward MLP
    - LayerNorm before each sublayer
    - Residual connections
    """
    def __init__(self, embed_dim, num_heads, up_proj_factor, dropout=0.1):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(embed_dim)  # LayerNorm before Attention
        self.attn = Attention(embed_dim, num_heads, dropout)
        
        self.ln2 = nn.LayerNorm(embed_dim)  # LayerNorm before MLP
        self.mlp = MLP(embed_dim, up_proj_factor, dropout=dropout)
    
    def forward(self, x):
        """Apply attention and MLP with residual connections"""
        
        # Self-Attention + Residual
        attn_out = self.attn(self.ln1(x))  # LayerNorm before attention
        x = x + attn_out  # Residual connection
        
        # MLP + Residual
        mlp_out = self.mlp(self.ln2(x))  # LayerNorm before MLP
        x = x + mlp_out  # Residual connection
        
        return x

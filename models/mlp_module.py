import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Feedforward network module.
    """
    def __init__(self, embed_dim, up_proj_factor=4, dropout=0.1):
        super().__init__()
        
        self.fc1 = nn.Linear(embed_dim, embed_dim * up_proj_factor)
        self.gelu = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(embed_dim * up_proj_factor, embed_dim)
        self.fc2.RESIDUAL_SCALE_INIT = 1  # Mark this for weight initialization scaling
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Feedforward network module.
    """
    def __init__(self, embed_dim, up_proj_factor=4, dropout=0.1):
        super().__init__()
        expanded_hidden_dim = int(embed_dim * up_proj_factor)  # Scale hidden dim
        
        self.fc1 = nn.Linear(embed_dim, expanded_hidden_dim)
        self.fc2 = nn.Linear(expanded_hidden_dim, embed_dim)
        self.fc2.RESIDUAL_SCALE_INIT = 1  # Mark this for weight initialization scaling
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """Apply the feedforward network with ReLU activation"""
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

import torch
import torch.nn as nn
from models.decoder_block import Block

class GPT(nn.Module):
    """
    Full GPT-style language model.
    - Token & positional embeddings
    - Stacked Decoder blocks
    - Final LayerNorm
    - Language modeling head (optionally tied with token embeddings)
    """
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, up_proj_factor, max_seq_len, dropout=0.1, tie_weights=True):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Token & Positional Embeddings
        self.wte = nn.Embedding(vocab_size, embed_dim)  # Token embeddings
        self.wpe = nn.Embedding(max_seq_len, embed_dim)  # Positional embeddings
        
        # Decoder Blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, up_proj_factor, dropout) for _ in range(num_layers)
        ])
        
        # Final LayerNorm
        self.ln_f = nn.LayerNorm(embed_dim)
        
        # Language Modeling Head (linear projection to vocab size)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Optionally tie lm_head and wte weights
        if tie_weights:
            self.lm_head.weight = self.wte.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Apply proper weight initialization"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            std = 0.02
            if isinstance(module, nn.Linear) and hasattr(module, 'RESIDUAL_SCALE_INIT'):
                std *= (2 * len(self.blocks)) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    
    def forward(self, input_ids):
        """Run input tokens through the model"""
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        token_embeddings = self.wte(input_ids)  # (batch, seq_len, embed_dim)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeddings = self.wpe(pos_ids)  # (batch, seq_len, embed_dim)
        
        # Sum token and positional embeddings
        x = token_embeddings + pos_embeddings
        
        # Forward through Decoder blocks
        for block in self.blocks:
            x = block(x)
        
        # Final LayerNorm
        x = self.ln_f(x)
        
        # Compute logits
        logits = self.lm_head(x)
        return logits

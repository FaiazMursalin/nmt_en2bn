import torch
import torch.nn as nn


class HierarchicalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim

        # Primary multi-head attention
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(embed_dim)

        # Context-aware attention for capturing sentence-level information
        self.context_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1, bias=False)
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Primary attention for capturing word and phrase level patterns
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)

        # Context-aware attention for sentence level
        # Calculate attention weights
        attn_weights = self.context_projection(x).softmax(dim=0)
        # Generate context vector through weighted sum
        context = torch.sum(attn_weights * x, dim=0, keepdim=True)
        # Apply context to all tokens
        result = self.layer_norm2(x + context)

        return result

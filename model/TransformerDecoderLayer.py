import torch.nn as nn
from triton.ops import attention


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # layer norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # dropout
        self.dropout = nn.Dropout(dropout)
        # activation
        self.activation = nn.ReLU()

    def forward(self, target, memory, target_mask=None, memory_mask=None, target_key_padding_mask=None,
                memory_key_padding_mask=None):
        # self attention first
        target2, _ = self.self_attention(query=target,
                                         key=target,
                                         value=target,
                                         attn_mask=target_mask,
                                         key_padding_mask=target_key_padding_mask)
        target = self.norm1(target + self.dropout(target2))

        # cross attention
        target2, _ = self.cross_attention(query=target,
                                          key=memory,
                                          value=memory,
                                          attn_mask=memory_mask,
                                          key_padding_mask=memory_key_padding_mask)
        target = self.norm2(target + self.dropout(target2))

        # finally feed forward
        target2 = self.linear2(self.dropout(self.activation(self.linear1(target))))
        target = self.norm3(target + self.dropout(target2))

        return target

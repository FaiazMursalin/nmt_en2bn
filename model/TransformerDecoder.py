import torch.nn as nn


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])

    def forward(self, target, memory, target_mask=None, memory_mask=None, target_key_padding_mask=None,
                memory_key_padding_mask=None):
        output = target

        for layer in self.layers:
            output = layer(output, memory, target_mask=target_mask, memory_mask=memory_mask,
                           target_key_padding_mask=target_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)

        return output

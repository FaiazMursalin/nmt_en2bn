import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from helper.PositionalEncoding import PositionalEncoding
from model.HierarchicalAttention import HierarchicalAttention
from model.TransformerDecoder import TransformerDecoder
from model.TransformerDecoderLayer import TransformerDecoderLayer
import sentencepiece as spm
from model.mBERTEncoder import mBERTEncoder


class BengaliEnglishNMT(nn.Module):
    def __init__(self, mbert_model_name='bert-base-multilingual-cased',
                 tgt_vocab_size=32000, embed_dim=768, num_decoder_layers=6, bn_spm_path=None):
        super().__init__()
        self.encoder = mBERTEncoder(model_name=mbert_model_name)
        self.encoder_tokenizer = BertTokenizer.from_pretrained(mbert_model_name)
        self.bn_spm = spm.SentencePieceProcessor(model_file=bn_spm_path)
        self.hierarchical_attention = HierarchicalAttention(embed_dim)

        # target embedding
        self.target_embedding = nn.Embedding(tgt_vocab_size, embed_dim)

        # positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim)

        # decoder layers
        decoder_layer = TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.3
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # output projection
        self.output_projection = nn.Linear(embed_dim, tgt_vocab_size)

        # initialize zero weights
        self._init_weights()

    def _init_weights(self):
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform_(self.target_embedding.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, src_input_ids, src_attention_mask, tgt_input):
        # 1. Verify input shapes
        print(f"Input shapes - src: {src_input_ids.shape}, tgt: {tgt_input.shape}")  # Debug

        # 2. Proper encoder handling
        encoder_output = self.encoder(src_input_ids, src_attention_mask)
        if isinstance(encoder_output, tuple):  # Handle BERT output format
            encoder_output = encoder_output[0]

        # 3. Add dimension checks
        assert encoder_output.dim() == 3, f"Encoder output should be 3D, got {encoder_output.dim()}"

        # encode source first
        encoder_output = self.encoder(
            input_ids=src_input_ids,
            attention_mask=src_attention_mask
        )
        encoder_hidden_states = encoder_output # Extract the tensor

        # second hierarchical attention
        memory = self.hierarchical_attention(encoder_hidden_states)

        # target embeddings and positional embedding
        target_embeddings = self.target_embedding(tgt_input)
        target_embeddings = self.positional_encoding(target_embeddings)

        # decoder
        decoder_output = self.decoder(
            # seq_len, batch, embed_dim
            target_embeddings.transpose(0, 1),
            # seq_len, batch, embed_dim
            memory.transpose(0, 1),
            target_mask=self.generate_square_subsequent_mask(tgt_input.size(1)),
            memory_key_padding_mask=(src_attention_mask == 0)
        )

        # project to vocabs
        # batch, seq_len, vocab
        output = self.output_projection(decoder_output.transpose(0, 1))

        return output

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(next(self.parameters()).device)

    def encode(self, src_input_ids, src_attention_mask):
        encoder_output = self.encoder(src_input_ids, src_attention_mask)
        return self.hierarchical_attention(encoder_output)

    def decode(self, memory, target_input, target_mask=None):
        target_embeddings = self.target_embedding(target_input)
        target_embeddings = self.positional_encoding(target_embeddings)

        decoder_output = self.decoder(target_embeddings.transpose(0, 1), memory.transpose(0, 1),
                                      target_mask=target_mask)

        return self.output_projection(decoder_output.transpose(0, 1))

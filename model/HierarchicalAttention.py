import torch.nn as nn


class HierarchicalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads_word=12, num_heads_phrase=8, num_heads_sentence=6):
        super().__init__()
        self.embed_dim = embed_dim

        # word level attention
        self.word_attention = nn.MultiheadAttention(embed_dim, num_heads_word)
        self.word_layer_norm = nn.LayerNorm(embed_dim)

        # phrase level attention
        self.phrase_attention = nn.MultiheadAttention(embed_dim, num_heads_phrase)
        self.phrase_layer_norm = nn.LayerNorm(embed_dim)

        # sentence level attention
        self.sentence_attention = nn.MultiheadAttention(embed_dim, num_heads_sentence)
        self.sentence_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # word level attention
        word_out, _ = self.word_attention(x, x, x)
        word_out = self.word_layer_norm(x + word_out)

        # phrase level attention
        phrase_out, _ = self.phrase_attention(word_out, word_out, word_out)
        phrase_out = self.phrase_layer_norm(word_out + phrase_out)

        # sentence level attention
        # first we need to create sentence representation by averaging
        sentence_representation = phrase_out.mean(dim=0, keepdim=True)
        sentence_out, _ = self.sentence_attention(sentence_representation, sentence_representation,
                                                  sentence_representation)
        sentence_out = self.sentence_layer_norm(sentence_representation + sentence_out)

        # combining all the three attentions
        combined = phrase_out + sentence_out.expand_as(phrase_out)

        return combined

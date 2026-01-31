from transformer.sublayers import MultiHeadAttention, PositionWiseFeedForward
from transformer.modules import NormLayer

import torch
import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim: int = 512, n_head: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super(EncoderLayer, self).__init__()
        self.norm1 = NormLayer(embed_dim)
        self.norm2 = NormLayer(embed_dim)

        self.self_attn = MultiHeadAttention(embed_dim, n_head, dropout=dropout)
        self.pos_ffn = PositionWiseFeedForward(embed_dim, d_ff=d_ff, dropout=dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        For sublayer in [MultiHeadAttetion, PostionWiseFeedForward]:
             1. Normalize(x)
             2. Do sublayer (like MultiHeadAttention)
             3. Dropout(0.1)

        논문에서는 다음과 같이 씌여져 있음
             We apply dropout to the output of each sub-layer,
             before it is added to the sub-layer input and normalized
        """
        x2 = self.norm1(x)
        h = x + self.dropout1(self.self_attn(x2, x2, x2, mask))
        h = x + self.dropout2(self.pos_ffn(self.norm2(h)))
        return h


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, n_head: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super(DecoderLayer, self).__init__()
        self.norm1 = NormLayer(embed_dim)
        self.norm2 = NormLayer(embed_dim)

        self.self_attn1 = MultiHeadAttention(embed_dim, n_head, dropout=dropout)
        self.self_attn2 = MultiHeadAttention(embed_dim, n_head, dropout=dropout)
        self.pos_ffn = PositionWiseFeedForward(embed_dim, d_ff=d_ff, dropout=dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, trg: torch.Tensor, trg_mask: torch.Tensor, enc_output: torch.Tensor, enc_mask: torch.Tensor):
        # First Multi Head Attention : target_input
        x2 = self.norm1(trg)
        dec_output = trg + self.dropout1(self.self_attn1(x2, x2, x2, trg_mask))

        # Second Multi Head Attention : 1st multi-head attetion output + encoder output
        # TODO: 현재 dec_output에만 normalization이 들어갔는데, enc_output도 normalization도 실험 필요함
        dec_output2 = self.norm2(dec_output)
        dec_output = dec_output + self.dropout2(self.self_attn2(dec_output2, enc_output, enc_output, enc_mask))

        # Postion-Wise Feed Forward
        dec_output = trg + self.dropout3(self.pos_ffn(dec_output))
        return dec_output

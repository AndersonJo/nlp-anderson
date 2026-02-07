import torch
from torch import nn
from torch import Tensor
from typing import Optional, Tuple
from .transformer_encoder import SwitchTransformerEncoderLayer, SwitchTransformerEncoder
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SwitchTransformerLM(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_ff: int, num_experts: int, num_layers: int,
                 dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.encoder = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = SwitchTransformerEncoderLayer(d_model, nhead, d_ff, num_experts, dropout)
        self.transformer_encoder = SwitchTransformerEncoder(encoder_layer, num_layers)

        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        output, load_balancing_loss = self.transformer_encoder(src, mask=src_mask)

        output = self.decoder(output)
        return output, load_balancing_loss

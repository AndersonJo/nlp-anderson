import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_seq_len=400):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_len, embed_dim)  # (400, 512) shape 의 matrix
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, 400, 512) shape 으로 만든다
        self.register_buffer('pe', pe)  # 논문에서 positional emcodding은 constant matrix 임으로 register_buffer 사용

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].detach()  # constant matrix 이기 때문에 detach 시킨다

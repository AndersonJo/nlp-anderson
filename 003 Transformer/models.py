import math

import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, padding_idx: int):
        super(EmbeddingLayer, self).__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, dropout=0.1, max_seq_len=400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, embed_dim)  # (400, 512) shape 의 matrix
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, 400, 512) shape 으로 만든다
        self.register_buffer('pe', pe)  # 논문에서 positional emcodding은 constant matrix 임으로 register_buffer 사용

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].detach()  # constant matrix 이기 때문에 detach 시킨다


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, padding_idx: int, n_layers: int, dropout: float = 0.1):
        """
        Embedding Parameters
        :param vocab_size: the size of the source vocabulary
        :param embed_dim: embedding dimension. 512 is used in the paper
        :param padding_idx: padding index

        Flow
           1. embedding layer
           2. positional encoding
           3. residual dropout (I don't know why it is named by "residual")
           4. multiple layers (6 layers are used in paper)
        """
        super(Encoder, self).__init__()
        self.embed = EmbeddingLayer(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pe = PositionalEncoding(embed_dim, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)  # Residual Dropout(0.1) in paper
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])


class Transformer(nn.Module):

    def __init__(self, vocab_size: int, padding_idx: int, embed_dim: int = 512):
        """
        :param vocab_size: the size of the source vocabulary
        :param embed_dim: embedding dimension. 512 is used in the paper
        :param padding_idx: padding index
        """
        super(Transformer, self).__init__()

        self.encoder = Encoder(vocab_size, embed_dim, padding_idx=padding_idx)

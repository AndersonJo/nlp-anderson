import torch
import torch.nn as nn
from torch.nn.modules.container import ModuleList

from transformer.embed import PositionalEncoding
from transformer.layers import EncoderLayer, DecoderLayer
from transformer.mask import create_mask
from transformer.modules import NormLayer


class Transformer(nn.Module):

    def __init__(self, embed_dim: int, src_vocab_size: int, trg_vocab_size: int,
                 src_pad_idx: int, trg_pad_idx: int,
                 n_layers: int = 6, n_head: int = 8, d_ff: int = 2048, dropout: float = 0.1,
                 share_embed_weights: bool = True,
                 share_embed_weights_to_output_linear: bool = True):
        """
        # Vocab
        :param src_vocab_size: the size of the source vocabulary
        :param trg_vocab_size: the size of the target vocabulary
        :param src_pad_idx: source padding index
        :param trg_pad_idx: target padding index

        # Embedding
        :param embed_dim: embedding dimension. 512 is used in the paper
        :param share_embed_weights: share target embedding weights with source embedding
        :param share_embed_weights_to_output_linear: share target embedding weights with output linear weights

        # Transformer
        :param n_layers: the number of sub-layers
        :param n_head: the number of heads in Multi Head Attention
        :param d_ff: inner dimension of position-wise feed-forward

        super(Transformer, self).__init__()

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.encoder = Encoder(src_vocab_size, embed_dim=embed_dim, n_head=n_head, d_ff=d_ff,
                               pad_idx=src_pad_idx, n_layers=n_layers, dropout=dropout)
        self.decoder = Decoder(trg_vocab_size, embed_dim=embed_dim, n_head=n_head, d_ff=d_ff,
                               pad_idx=trg_pad_idx, n_layers=n_layers, dropout=dropout)

        self.out_linear = nn.Linear(embed_dim, trg_vocab_size, bias=False)

        self.logit_scale = 1
        if share_embed_weights_to_output_linear:
            # 다음은 논문에 나온 내용
            # 3.4  Embeddings and Softmax
            # We also use the usual learned linear transformation and so

        """
        super(Transformer, self).__init__()

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.encoder = Encoder(src_vocab_size, embed_dim=embed_dim, n_head=n_head, d_ff=d_ff,
                               pad_idx=src_pad_idx, n_layers=n_layers, dropout=dropout)
        self.decoder = Decoder(trg_vocab_size, embed_dim=embed_dim, n_head=n_head, d_ff=d_ff,
                               pad_idx=trg_pad_idx, n_layers=n_layers, dropout=dropout)

        self.out_linear = nn.Linear(embed_dim, trg_vocab_size, bias=False)

        self.logit_scale = 1
        if share_embed_weights_to_output_linear:
            # 다음은 논문에 나온 내용
            # 3.4  Embeddings and Softmax
            # We also use the usual learned linear transformation and softmax function to convert the decoder output
            # to predicted next-token probabilities. In our model, we share the same weight matrix between
            # the two embedding layers and the pre-softmax linear transformation, similar to [30].
            # In the embedding layers, we multiply those weights by sqrt(d_model).
            self.out_linear.weight = self.decoder.embed.weight
            self.logit_scale = (embed_dim ** -0.5)

        if share_embed_weights:
            self.encoder.embed.weight = self.decoder.embed.weight

    def forward(self, src: torch.Tensor, trg: torch.Tensor):
        """
        :param src: (batch_size, maximum_sequence_length)
        :param trg: (batch_size, maximum_sequence_length)
        :return (batch, seq_len, trg_vocab_size) ex.(256, 33, 9473)
        """
        src_mask, trg_mask = create_mask(src, trg, src_pad_idx=self.src_pad_idx, trg_pad_idx=self.trg_pad_idx)

        enc_output = self.encoder(src, src_mask)  # (batch, seq_len, embed_dim) like (256, 33, 512)
        dec_output = self.decoder(trg, trg_mask, enc_output, src_mask)  # (batch, seq_len, embed_dim)
        output = self.out_linear(dec_output) * self.logit_scale  # (batch, seq_len, trg_vocab_size) like (256, 33, 9473)
        return output


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, n_head: int, d_ff: int,
                 pad_idx: int, n_layers: int, dropout: float = 0.1):
        """
        Embedding Parameters
        :param vocab_size: the size of the source vocabulary
        :param embed_dim: embedding dimension. 512 is used in the paper
        :param n_head: the number of multi head. (split the embed_dim to 8.. such that 8 * 64 = 512)
        :param d_ff: inner dimension of position-wise feed-forward
        :param pad_idx: padding index
        :param n_layers: the number of sub-layers

        Flow
           1. embedding layer
           2. positional encoding
           3. residual dropout(0.1)
           4. iterate sub-layers (6 layers are used in paper)
        """
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(embed_dim)
        self.dropout = nn.Dropout(dropout)  # Residual Dropout(0.1) in paper

        self.layer_stack: ModuleList[EncoderLayer] = nn.ModuleList(
            [EncoderLayer(embed_dim, n_head, d_ff=d_ff, dropout=dropout) for _ in range(n_layers)])

        self.layer_norm = NormLayer(embed_dim)

    def forward(self, src: torch.Tensor, mask: torch.Tensor):
        """
        Sublayers 에 들어가기 전에 논문에서는 다음 3가지를 해야 함
        Word Tensor -> Embedding -> Positional Embedding -> Dropout(0.1)

        아래는 논문 내용
            we apply dropout to the sums of the embeddings and the
            positional encodings in both the encoder and decoder stacks

        :return encoder output : (batch, seq_len, embed_dim) like (256, 33, 512)
        """
        x = self.embed(src)  # (256 batch, 33 seqence, 512 embedding)
        x = self.position_enc(x)  # (256, 33, 512)
        x = self.dropout(x)  # Layer Stack 사용전에 Dropout 을 해야 함 (Decoder 에도 해야 됨)

        for enc_layer in self.layer_stack:
            x = enc_layer(x, mask)

        enc_output = self.layer_norm(x)  # (256, 33, 512)
        return enc_output


class Decoder(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int, n_head: int, d_ff: int,
                 pad_idx: int, n_layers: int, dropout: float = 0.1):
        """
        :param vocab_size: the size of the target vocabulary
        :param embed_dim: embedding dimension. 512 is used in the paper
        :param n_head: the number of multi head. (split the embed_dim to 8.. such that 8 * 64 = 512)
        :param d_ff: inner dimension of position-wise feed-forward
        :param pad_idx: target padding index
        :param n_layers: the number of sub-layers
        """
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(embed_dim)
        self.dropout = nn.Dropout(dropout)  # Residual Dropout(0.1) in paper

        self.layer_stack: ModuleList[DecoderLayer] = nn.ModuleList(
            [DecoderLayer(embed_dim, n_head, d_ff=d_ff, dropout=dropout) for _ in range(n_layers)])

        self.layer_norm = NormLayer(embed_dim)

    def forward(self, trg: torch.Tensor, trg_mask: torch.Tensor,
                enc_output: torch.Tensor, enc_mask: torch.Tensor):
        dec_output = self.embed(trg)  # (256 batch, 33 seqence, 512 embedding)
        dec_output = self.position_enc(dec_output)  # (256, 33, 512)
        dec_output = self.dropout(dec_output)  # Layer Stack 사용전에 Dropout 을 해야 함

        for dec_layer in self.layer_stack:
            dec_output = dec_layer(dec_output, trg_mask, enc_output, enc_mask)

        dec_output = self.layer_norm(dec_output)

        return dec_output

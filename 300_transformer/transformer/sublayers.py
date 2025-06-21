"""
Implementation of Multi-Head Self Attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    512인 embedding vector를 -> n_head에 따라서 나눈다.
    예를 들어, n_head=8일 경우 512 vector를 -> 8 * 64 vector 로 변형한다

    """

    def __init__(self, embed_dim: int, n_head: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.n_head = n_head
        self.dk = embed_dim // n_head
        self.dv = embed_dim // n_head

        self.linear_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_f = nn.Linear(embed_dim, embed_dim, bias=False)  # Final linear layer

        self.attention = ScaleDotProductAttention(self.dk, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
         * 마지막 skip connection 은 layer 부분에서 구현함
        """
        batch_size, n_head, dk, dv = q.size(0), self.n_head, self.dk, self.dv

        # Linear Transformation (256, 33, 512)
        # Multi Head : d_model(512) vector부분을 h개 (8개) 로 나눈다
        q = self.linear_q(q).view(batch_size, -1, n_head, dk)
        k = self.linear_k(k).view(batch_size, -1, n_head, dk)
        v = self.linear_v(v).view(batch_size, -1, n_head, dv)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, mask)

        # multi head dimension 을 원래의 형태로 되돌린다
        # (batch, n_head, seq_len, d_v) (256, 8, 33, 64) --> (batch, seq_len, n_head, d_v) (256, 33, 8, 64)
        scores = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        # Final linear Layer
        scores = self.linear_f(scores)

        return scores


class ScaleDotProductAttention(nn.Module):
    """
    Attention(Q, K, V) = softmax( (QK^T)/sqrt(d_k) )
    """

    def __init__(self, d_k: int, dropout: float):
        """
        :param d_k: the number of heads
        """
        super(ScaleDotProductAttention, self).__init__()
        self.sqrt_dk = d_k ** 0.5  # 8 = 64**0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        :param q: Queries (256 batch, 8 d_k, 33 sequence, 64)
        :param k: Keys    (256, 8, 33, 64)
        :param v: Values  (256, 8, 33, 64)
        :param mask: mask (256, 1, 28) Source Mask
        :return: scaled dot attention: (256, 8, 33, 64)
        """
        attn = (q @ k.transpose(-2, -1)) / self.sqrt_dk
        if mask is not None:  # 논문에는 안나온 내용. 하지만 masking을 해주어야 한다
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(~mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))  # softmax 이후 dropout도 논문에는 없으나 해야 한다
        output = attn @ v  # (256, 8, 33, 64)

        return output


class PositionWiseFeedForward(nn.Module):

    def __init__(self, embed_dim: int, d_ff: int = 2048, dropout: float = 0.1):
        super(PositionWiseFeedForward, self).__init__()

        self.w_1 = nn.Linear(embed_dim, d_ff)
        self.w_2 = nn.Linear(d_ff, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        논문에서는 Position-wise Feed Forward를 할때 skip connection에 대한 이야기는 없습니다.
        다만 MultiHead 부분에서도
        """
        residual = x
        x = F.relu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        return x + residual

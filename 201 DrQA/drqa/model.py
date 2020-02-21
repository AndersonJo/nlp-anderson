from argparse import Namespace
from typing import Tuple

import torch
import torch.nn as nn
import numpy as np


class DocumentReaderModel(nn.Module):

    def __init__(self, opt: Namespace, embeddings: torch.Tensor):
        super(DocumentReaderModel, self).__init__()
        self.opt = opt
        self.partial_tune_top_k = opt.tune_top_k  # Embeddings 의 top-n 까지 fine-tuning 하고, 나머지는 freeze

        # Embedding Layer
        self.embedding: nn.Embedding = None
        self._init_embedding_layer(embeddings, embed_shape=(opt.vocab_size, opt.embed_dim))

    def _init_embedding_layer(self, embeddings: torch.Tensor, embed_shape: Tuple[int, int]):
        # 여기서 Embedding 의 일부만 tuning 을 할 수 있는 코드가 나옵니다.
        # 즉, 빈번도가 높은 상위 단어들, <PAD>, <UNK>, the, of, and, is, was 같은 단어들 top-n (기본값 1000)
        # 을 잡아서 학습시 gradient 값에 따라 학습하고, 나머지는 gradient 값을 0으로 주어서 학습을 안시킵니다.

        if self.partial_tune_top_k == 0:
            self.embedding: nn.Embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)
            self.embedding.weight.requires_grad = False  # freeze the whole embedding layer
        elif self.partial_tune_top_k < 0:
            # 랜덤값으로 embedding layer를 만든다
            # padding_idx는 해당 row전체를 0으로 만든다
            # nn.Embedding(4, 3, padding_idx=0).weight
            # tensor([[ 0.0000,  0.0000,  0.0000],
            #         [ 1.8362,  0.4903,  0.4222],
            #         [ 0.1502,  1.1995,  1.2751],
            #         [-0.5338,  1.2036, -0.0191]], requires_grad=True)

            self.embedding = nn.Embedding(embed_shape[0], embed_shape[1], padding_idx=0)
        else:
            # register_hook 등록시, gradient값을 통해서 학습할때마다 실행하게 됨
            assert self.partial_tune_top_k + 2 < embeddings.size(0)
            self.embedding: nn.Embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)
            offset = self.partial_tune_top_k + 2

            def partial_tune_hook(grad, offset=offset):
                grad[offset:] = 0
                return grad

            self.embedding.weight.register_hook(partial_tune_hook)

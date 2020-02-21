from argparse import Namespace

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
        self._init_embedding_layer(embeddings)

    def _init_embedding_layer(self, embeddings: torch.Tensor):
        # 여기서 Embedding 의 일부만 tuning 을 할 수 있는 코드가 나옵니다.
        # 즉, 빈번도가 높은 상위 단어들, <PAD>, <UNK>, the, of, and, is, was 같은 단어들 top-n (기본값 1000)
        # 을 잡아서 학습시 gradient 값에 따라 학습하고, 나머지는 gradient 값을 0으로 주어서 학습을 안시킵니다.
        self.embedding: nn.Embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)
        if self.partial_tune_top_k <= 0:
            assert self.partial_tune_top_k == 0, 'tune_top_k should not be less than zero'
            self.embedding.weight.requires_grad = False  # freeze the whole embedding layer
        else:
            assert self.partial_tune_top_k + 2 < embeddings.size(0)
            offset = self.partial_tune_top_k + 2

            def partial_tune_hook(grad, offset=offset):
                grad[offset:] = 0
                return grad

            self.embedding.weight.register_hook(partial_tune_hook)

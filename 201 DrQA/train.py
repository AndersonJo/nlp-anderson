# 아래의 코드는 https://github.com/hitvoice/DrQA/blob/master/train.py 에서 대부분의 코드를 참고 하였습니다.
# 모델이 어떻게 만들어졌는지 알아내기 위해서 이미 만들어진 코드를 뜯어보면서 재구성했습니다.

import os
from argparse import ArgumentParser, Namespace
from typing import Tuple

import msgpack
import torch
from tqdm import tqdm

from drqa.model import DocumentReaderModel


def init() -> Namespace:
    parser = ArgumentParser()
    # Data anb Embedding Data
    parser.add_argument('--data_dir', default='./data/datasets', type=str, help='SQuAD dataset directory path')
    parser.add_argument('--embed_dir', default='./data/embeddings', type=str)

    # Embedding Word Vector
    parser.add_argument('--tune_top_k', default=1000, type=int, help='Find tune top-k vectors in embeddings')

    args = parser.parse_args()
    return args


def load_dataset(opt, sample=False) -> Tuple[list, list, list, torch.Tensor]:
    _tqdm = tqdm(total=3, desc='Load Dataset')
    _tqdm.display()
    # Load Meta
    with open(os.path.join(opt.data_dir, 'meta.msgpack'), 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    embed_matrix = torch.Tensor(meta['embed_matrix'])
    _tqdm.update(1)

    # Load Data
    data_path = 'sample.msgpack' if sample else 'data.msgpack'
    data_path = os.path.join(opt.data_dir, data_path)
    with open(data_path, 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    _tqdm.update(1)

    # Set options
    opt['vocab_size'] = embed_matrix.size(0)
    opt['embed_dim'] = embed_matrix.size(1)
    opt['pos_size'] = len(meta['ctx_tags'])  # the size of part of speech
    opt['ner_size'] = len(meta['ctx_ents'])  # the size of named entity recognition tags

    train = data['train']
    dev = data['dev']
    dev_y = [d['answers'] for d in dev]  # Retrieve answers
    [d.pop('answers') for d in dev]  # Remove answers from dev
    _tqdm.update(1)
    _tqdm.display()

    return train, dev, dev_y, embed_matrix


def main():
    opt = init()

    # Load Dataset
    train, dev, dev_y, embed_matrix = load_dataset(opt)

    # Load Reader Model
    model = DocumentReaderModel(opt, embeddings=embed_matrix)

    import ipdb
    ipdb.set_trace()


if __name__ == '__main__':
    main()

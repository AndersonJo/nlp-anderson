import argparse
from argparse import Namespace

import torch
from tqdm import tqdm

from tools.data_loader import load_preprocessed_data
from transformer import get_transformer
from transformer.models import Transformer


def init() -> Namespace:
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_pkl', default='.data/data.pkl', type=str)

    # System
    parser.add_argument('--cuda', action='store_true', default=True)

    # Hyper Parameters
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--embed_dim', default=512, type=int)
    parser.add_argument('--n_head', default=8, type=int, help='the number of multi heads')

    # Parse
    parser.set_defaults(share_embed_weights=True)
    opt = parser.parse_args()

    assert opt.embed_dim % opt.n_head == 0, 'the number of heads should be the multiple of embed_dim'

    opt.device = torch.device('cuda' if opt.cuda else 'cpu')
    return opt


def train(opt: Namespace, model: Transformer):
    train_data, val_data, src_vocab, trg_vocab = load_preprocessed_data(opt)

    for epoch in range(opt.epoch):
        for batch in tqdm(train_data, leave=False):
            # Prepare data
            src_input = batch.src.transpose(0, 1).to(opt.device)  # (seq_length, batch) -> (batch, seq_length)
            trg = batch.trg.transpose(0, 1).to(opt.device)  # (seq_length, batch) -> (batch, seq_length)
            ys_input = trg[:, 1:].contiguous().view(-1)
            trg_input = trg[:, :-1]

            # Forward
            model(src_input, trg_input)


def main():
    opt = init()
    train_data, val_data, src_vocab, trg_vocab = load_preprocessed_data(opt)

    transformer = get_transformer(opt)

    train(opt, transformer)


if __name__ == '__main__':
    main()

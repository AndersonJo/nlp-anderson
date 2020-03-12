import argparse
import torch
from argparse import Namespace

from models import Transformer
from tools.data_loader import load_preprocessed_data


def init() -> Namespace:
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_pkl', default='.data/data.pkl', type=str)

    # System
    parser.add_argument('--cuda', action='store_true', default=True)

    # Hyper Parameters
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--embed_dim', default=512, type=int)

    # Parse
    parser.set_defaults(share_embed_weights=True)
    opt = parser.parse_args()

    opt.device = torch.device('cuda' if opt.cuda else 'cpu')
    return opt


def get_transformer(opt):
    # Encoder hyper-parameters
    return Transformer(vocab_size=opt.src_vocab_size,
                       embed_dim=opt.embed_dim,
                       padding_idx=opt.src_pad_idx)


def main():
    opt = init()
    train_data, val_data = load_preprocessed_data(opt)

    transformer = get_transformer(opt)


if __name__ == '__main__':
    main()

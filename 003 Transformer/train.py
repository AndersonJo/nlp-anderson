import argparse
import torch
from argparse import Namespace

from tools.data_loader import load_preprocessed_data


def init() -> Namespace:
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_pkl', default='.data/data.pkl', type=str)

    # System
    parser.add_argument('--cuda', action='store_true', default=True)

    # Hyper Parameters
    parser.add_argument('--batch_size', default=128, type=int)

    # Parse
    opt = parser.parse_args()
    opt.device = torch.device('cuda' if opt.cuda else 'cpu')
    return opt


def main():
    opt = init()
    load_preprocessed_data(opt)


if __name__ == '__main__':
    main()

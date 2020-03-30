import argparse
from argparse import Namespace

import torch
import torch.nn.functional as F
from tqdm import tqdm

from tools.data_loader import load_preprocessed_data
from transformer import get_transformer
from transformer.models import Transformer
from transformer.optimizer import ScheduledAdam


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
    parser.add_argument('--warmup_steps', default=4000, type=int, help='the number of warmup steps')

    # Parse
    parser.set_defaults(share_embed_weights=True)
    opt = parser.parse_args()

    assert opt.embed_dim % opt.n_head == 0, 'the number of heads should be the multiple of embed_dim'

    opt.device = torch.device('cuda' if opt.cuda else 'cpu')
    return opt


def train(opt: Namespace, model: Transformer, optimizer: ScheduledAdam):
    train_data, val_data, src_vocab, trg_vocab = load_preprocessed_data(opt)

    for epoch in range(opt.epoch):
        for batch in tqdm(train_data, leave=False):
            # Prepare data
            #  - src_input: <sos>, 외국단어_1, 외국단어_2, ..., 외국단어_n, <eos>, pad_1, ..., pad_n
            #  - trg_input:  (256, 33) -> <sos>, 영어_1, 영어_2, ..., 영어_n, <eos>, pad_1, ..., pad_n-1
            #  - y_true   : (256 * 33) -> 영어_1, 영어_2, ... 영어_n, <eos>, pad_1, ..., pad_n
            src_input = batch.src.transpose(0, 1).to(opt.device)  # (seq_length, batch) -> (batch, seq_length)
            trg_input = batch.trg.transpose(0, 1).to(opt.device)  # (seq_length, batch) -> (batch, seq_length)
            trg_input, y_true = trg_input[:, :-1], trg_input[:, 1:].contiguous().view(-1)

            # Forward
            optimizer.zero_grad()
            y_pred = model(src_input, trg_input)

            # Backward and update parameters

            calculate_loss(y_pred, y_true, opt.trg_pad_idx)


def calculate_performance(y_pred, y_true):
    pass


def calculate_loss(y_pred, y_true, trg_pad_idx):
    """
    여기서 재미있는건.. y_pred는 trg_vocab_size인 vector 형태로 들어오고,
    y_true값은 index값으로 들어옴.
    F.cross_entropy에 그대로 집어 넣으면 vector에서 가장 큰 값과,

    :param y_pred: (batch * seq_len, trg_vocab_size) ex. (256*33, 9473)
    :param y_true: (batch * seq_len)
    :param trg_pad_idx:
    :return:
    """
    y_pred = y_pred.view(-1, y_pred.size(-1))
    y_true = y_true.contiguous().view(-1)

    return F.cross_entropy(y_pred, y_true, ignore_index=trg_pad_idx, reduction='sum')


def main():
    opt = init()
    train_data, val_data, src_vocab, trg_vocab = load_preprocessed_data(opt)

    transformer = get_transformer(opt)
    optimizer = ScheduledAdam(transformer.parameters(), opt.embed_dim, warmup_steps=opt.warmup_steps)

    train(opt, transformer, optimizer)


if __name__ == '__main__':
    main()

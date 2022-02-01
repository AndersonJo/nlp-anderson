import argparse
import os
from argparse import Namespace
from datetime import datetime
from typing import Tuple

import numpy as np
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
    parser.add_argument('--checkpoint_path', default='checkpoints', type=str)

    # Hyper Parameters
    parser.add_argument('--epoch', default=500, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--embed_dim', default=512, type=int)
    parser.add_argument('--n_head', default=8, type=int, help='the number of multi heads')
    parser.add_argument('--warmup_steps', default=124000, type=int, help='the number of warmup steps')

    # Parse
    parser.set_defaults(share_embed_weights=True)
    opt = parser.parse_args()

    assert opt.embed_dim % opt.n_head == 0, 'the number of heads should be the multiple of embed_dim'

    opt.device = torch.device('cuda' if opt.cuda else 'cpu')
    return opt


def train(opt: Namespace, model: Transformer, optimizer: ScheduledAdam):
    if not os.path.exists(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)

    train_data, val_data, src_vocab, trg_vocab = load_preprocessed_data(opt)
    min_loss = float('inf')

    for epoch in range(opt.epoch):
        # Training and Evaluation
        _t = train_per_epoch(opt, model, optimizer, train_data)
        _v = evaluate_epoch(opt, model, val_data)

        # Checkpoint
        is_checkpointed = False
        if _v['total_loss'] < min_loss:
            min_loss = _v['total_loss']
            checkpoint = {'epoch': epoch,
                          'opt': opt,
                          'weights': model.state_dict(),
                          'loss': min_loss,
                          '_t': _t,
                          '_v': _v}
            model_name = os.path.join(opt.checkpoint_path, f'checkpoint_{min_loss:.4f}.chkpt')
            torch.save(checkpoint, model_name)
            is_checkpointed = True

        # Print performance
        _show_performance(epoch=epoch, step=optimizer.n_step, lr=optimizer.lr, t=_t, v=_v, checkpoint=is_checkpointed)


def train_per_epoch(opt: Namespace,
                    model: Transformer,
                    optimizer: ScheduledAdam,
                    train_data) -> dict:
    model.train()
    start_time = datetime.now()
    total_loss = total_word = total_corrected_word = 0

    for i, batch in tqdm(enumerate(train_data), total=len(train_data), leave=False):
        src_input, trg_input, y_true = _prepare_batch_data(batch, opt.device)

        # Forward
        optimizer.zero_grad()
        y_pred = model(src_input, trg_input)

        # Backward and update parameters
        loss = calculate_loss(y_pred, y_true, opt.trg_pad_idx)
        n_word, n_corrected = calculate_performance(y_pred, y_true, opt.trg_pad_idx)
        loss.backward()
        optimizer.step()

        # Training Logs
        total_loss += loss.item()
        total_word += n_word
        total_corrected_word += n_corrected

    loss_per_word = total_loss / total_word
    accuracy = total_corrected_word / total_word

    return {'total_seconds': (datetime.now() - start_time).total_seconds(),
            'total_loss': total_loss,
            'total_word': total_word,
            'total_corrected_word': total_corrected_word,
            'loss_per_word': loss_per_word,
            'accuracy': accuracy}


def evaluate_epoch(opt: Namespace, model: Transformer, val_data):
    model.eval()
    start_time = datetime.now()
    total_loss = total_word = total_corrected_word = 0

    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_data), total=len(val_data), leave=False):
            # Prepare validation data
            src_input, trg_input, y_true = _prepare_batch_data(batch, opt.device)

            # Forward
            y_pred = model(src_input, trg_input)
            loss = calculate_loss(y_pred, y_true, opt.trg_pad_idx)
            n_word, n_corrected = calculate_performance(y_pred, y_true, opt.trg_pad_idx)

            # Validation Logs
            total_loss += loss.item()
            total_word += n_word
            total_corrected_word += n_corrected

    loss_per_word = total_loss / total_word
    accuracy = total_corrected_word / total_word

    return {'total_seconds': (datetime.now() - start_time).total_seconds(),
            'total_loss': total_loss,
            'total_word': total_word,
            'total_corrected_word': total_corrected_word,
            'loss_per_word': loss_per_word,
            'accuracy': accuracy}


def _prepare_batch_data(batch, device):
    """
    Prepare data
     - src_input: <sos>, 외국단어_1, 외국단어_2, ..., 외국단어_n, <eos>, pad_1, ..., pad_n
     - trg_inprint_performancesput:  (256, 33) -> <sos>, 영어_1, 영어_2, ..., 영어_n, <eos>, pad_1, ..., pad_n-1
     - y_true   : (256 * 33) -> 영어_1, 영어_2, ... 영어_n, <eos>, pad_1, ..., pad_n
    """
    src_input = batch.src.transpose(0, 1).to(device)  # (seq_length, batch) -> (batch, seq_length)
    trg_input = batch.trg.transpose(0, 1).to(device)  # (seq_length, batch) -> (batch, seq_length)
    trg_input, y_true = trg_input[:, :-1], trg_input[:, 1:].contiguous().view(-1)
    return src_input, trg_input, y_true


def calculate_performance(y_pred: torch.Tensor, y_true: torch.Tensor, trg_pad_idx: int) -> Tuple[int, int]:
    y_pred = y_pred.view(-1, y_pred.size(-1))
    y_argmax = y_pred.argmax(dim=1)
    y_true = y_true.contiguous().view(-1)

    non_pad_mask = y_true != trg_pad_idx
    n_corrected = (y_argmax == y_true).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return n_word, n_corrected


def calculate_loss(y_pred, y_true, trg_pad_idx):
    """
    y_pred는 trg_vocab_size인 vector 형태로 들어오고,
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


def _show_performance(epoch, step, lr, t, v, checkpoint):
    mins = int(t['total_seconds'] / 60)
    secs = int(t['total_seconds'] % 60)

    t_loss = t['total_loss']
    t_accuracy = t['accuracy']
    t_loss_per_word = t['loss_per_word']

    v_loss = v['total_loss']
    v_accuracy = v['accuracy']
    v_loss_per_word = v['loss_per_word']

    print(f'[{epoch + 1:02}] {mins:02}:{secs:02} | loss:{t_loss:10.2f}/{v_loss:10.2f} | '
          f'acc:{t_accuracy:7.4f}/{v_accuracy:7.4f} | '
          f'loss_per_word:{t_loss_per_word:5.2f}/{v_loss_per_word:5.2f} | step:{step:5} | lr:{lr:6.4f}'
          f'{" | checkpoint" if checkpoint else ""}')


def main():
    opt = init()
    train_data, val_data, src_vocab, trg_vocab = load_preprocessed_data(opt)

    transformer = get_transformer(opt)
    optimizer = ScheduledAdam(transformer.parameters(), opt.embed_dim, warmup_steps=opt.warmup_steps)

    train(opt, transformer, optimizer)


if __name__ == '__main__':
    main()

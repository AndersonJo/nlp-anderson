import argparse
import pickle
from argparse import Namespace
from typing import Tuple

import numpy as np
import torchtext
from torchtext.data import Field

from tools import const
from tools.tokenizer import Tokenizer


def init() -> Namespace:
    spacy_support_langs = ['de', 'el', 'en', 'es', 'fr', 'it', 'lt', 'nb', 'nl', 'pt']

    parser = argparse.ArgumentParser()
    parser.add_argument('--lang_src', default='de', choices=spacy_support_langs)
    parser.add_argument('--lang_trg', default='en', choices=spacy_support_langs)
    parser.add_argument('--data_src', type=str)
    parser.add_argument('--save_data', type=str, default='.data/data.pkl')

    parser.add_argument('--max_seq_len', type=int, default=100)
    parser.add_argument('--min_word_freq', type=int, default=3, help='minimum word count')
    parser.add_argument('--share_vocab', action='store_true', default=True, help='Merge source vocab with target vocab')

    opt = parser.parse_args()
    return opt


def create_torch_fields(opt: Namespace) -> Tuple[Field, Field]:
    tokenizer_src = Tokenizer(opt.lang_src)
    tokenizer_trg = Tokenizer(opt.lang_trg)

    src = Field(tokenize=tokenizer_src.tokenizer, lower=True,
                pad_token=const.PAD, init_token=const.SOS, eos_token=const.EOS)
    trg = Field(tokenize=tokenizer_trg.tokenizer, lower=True,
                pad_token=const.PAD, init_token=const.SOS, eos_token=const.EOS)

    return src, trg


def merge_source_and_target(src, trg):
    for word in set(trg.vocab.stoi) - set(src.vocab.stoi):
        l = len(src.vocab.stoi)
        src.vocab.stoi[word] = l
        src.vocab.itos.append(word)
        src.vocab.freqs[word] = trg.vocab.freqs[word]
    trg.vocab.stoi = src.vocab.stoi
    trg.vocab.itos = src.vocab.itos
    trg.vocab.freqs = src.vocab.freqs

    print(f'Merged source vocabulary: {len(src.vocab)}')
    print(f'Merged target vocabulary: {len(trg.vocab)}')
    return src, trg


def main():
    opt = init()
    max_seq_len = opt.max_seq_len
    min_word_freq = opt.min_word_freq

    # Create Fields
    src, trg = create_torch_fields(opt)

    # Data - max_seq_len 값 이상 넘어가는 단어로 이루어진 문장을 제외 시킨다
    def filter_with_length(x):
        return len(x.src) <= max_seq_len and len(x.trg) <= max_seq_len

    train, val, test = torchtext.datasets.Multi30k.splits(exts=('.' + opt.lang_src, '.' + opt.lang_trg),
                                                          fields=(src, trg),
                                                          filter_pred=filter_with_length)
    src.build_vocab(train.src, min_freq=min_word_freq)  # src.vocab.stoi, src.vocab.itos 생성
    print(f'Source vocabulary: {len(src.vocab)}')

    trg.build_vocab(train.trg, min_freq=min_word_freq)
    print(f'Target vocabulary: {len(trg.vocab)}')

    # Merge source vocabulary and target vocabulary
    if opt.share_vocab:
        src, trg = merge_source_and_target(src, trg)

    # Save data as a pickle
    data = {
        'opt': opt,
        'src': src,
        'trg': trg,
        'train': train.examples,
        'val': val.examples,
        'test': test.examples
    }

    pickle.dump(data, open(opt.save_data, 'wb'))
    print(f'The data saved at: {opt.save_data}')
    print('Preprocessing Completed Successfully')

    sentences = np.random.choice(train.examples, 5)
    print('\n[Train Data Examples]')
    for i, sentence in enumerate(sentences):
        print(f'[{i + 1}] Source:', sentence.src)
        print(f'[{i + 1}] Target:', sentence.trg)
        print()


if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()

import pickle
from typing import Tuple

from torchtext.vocab import Vocab

from tools import const
from torchtext.data import Dataset, BucketIterator


def load_preprocessed_data(opt) -> Tuple[BucketIterator, BucketIterator, Vocab, Vocab]:
    batch_size = opt.batch_size
    device = opt.device
    data = pickle.load(open(opt.data_pkl, 'rb'))

    opt.max_seq_len = data['opt'].max_seq_len
    opt.src_pad_idx = data['src'].vocab.stoi[const.PAD]
    opt.trg_pad_idx = data['trg'].vocab.stoi[const.PAD]
    opt.src_vocab_size = len(data['src'].vocab)
    opt.trg_vocab_size = len(data['trg'].vocab)

    if opt.share_embed_weights:
        assert data['src'].vocab.stoi == data['trg'].vocab.stoi

    fields = {'src': data['src'], 'trg': data['trg']}
    train = Dataset(examples=data['train'], fields=fields)
    val = Dataset(examples=data['val'], fields=fields)

    train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
    val_iterator = BucketIterator(val, batch_size=batch_size, device=device)

    return train_iterator, val_iterator, data['src'].vocab, data['trg'].vocab

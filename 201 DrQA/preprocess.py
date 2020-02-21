# 아래의 코드는 https://github.com/hitvoice/DrQA/blob/master/prepro.py 에서 많은 참고를 하였습니다.


import json
import logging
import msgpack
import os
import pickle
import re
import unicodedata
from argparse import ArgumentParser, Namespace
from collections import Counter
from functools import partial
from multiprocessing import Pool
from typing import List, Tuple, Union, Set

import numpy as np
import spacy
from spacy.lang.en import English
from spacy.tokens import Doc
from tqdm import tqdm

logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.DEBUG, datefmt='%Y/%m/%d %H:%M:%S')
logger = logging.getLogger('drqa')
spacy_: English = None


def init() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='./data/datasets', type=str, help='SQuAD dataset directory path')
    parser.add_argument('--embed_dir', default='./data/embeddings', type=str)
    parser.add_argument('--out', default='./data/datasets', type=str, help='preprocessed output directory path')
    parser.add_argument('--threads', default=64, type=int, help='the number of multiprocess jobs')
    parser.add_argument('--chunk', default=64, type=int, help='chunk size of multiprocess tokenizing and tagging')
    parser.add_argument('--embedding_size', default=300, type=int, help='the size of embedding vector')
    parser.add_argument('--sample_size', type=int, default=5000)

    args = parser.parse_args()
    return args


def load_dataset(path: str) -> List[Tuple[int, str, str, Tuple[int, int, str]]]:
    """
    :param path:
    :return: - context: [context, ...]
             - qas: [(context_id, question string, (start_idx, end_idx, answer)), ... ]
    """
    with open(path, 'rt') as f:
        data = json.load(f)['data']

    output = []

    for article in data:
        # title = article['title']
        for ctx_idx, paragraph in enumerate(article['paragraphs']):
            context = paragraph['context']

            for qa in paragraph['qas']:
                _id, question, answers = qa['id'], qa['question'], qa['answers']
                answers: tuple = tuple((a['answer_start'],
                                        a['answer_start'] + len(a['text']),
                                        a['text']) for a in answers)

                answers = tuple(set(answers))
                output.append((_id, context, question, answers))

    return output


def load_vocab_data(parser: Namespace) -> Set[str]:
    glove_path = os.path.join(parser.embed_dir, 'glove.840B.300d.txt')
    pickle_path = os.path.join(parser.embed_dir, 'glove.840B.300d.pkl')

    if os.path.exists(pickle_path):
        return pickle.load(open(pickle_path, 'rb'))

    vocab = set()

    with open(glove_path, 'rt') as f:
        for line in tqdm(f, total=2196018, desc='Vocab'):
            word = line[:line.find(' ')]
            word = unicodedata.normalize('NFD', word)
            # word2 = unicodedata.normalize('NFD', line.rstrip().split(' ')[0])
            # assert word == word2
            vocab.add(word)

    pickle.dump(vocab, open(pickle_path, 'wb'))
    return vocab


def load_preprocessed_squad_data(parser: Namespace, force=False):
    squad_train_path = os.path.join(parser.data_dir, 'SQuAD-v1.1-train.json')
    squad_dev_path = os.path.join(parser.data_dir, 'SQuAD-v1.1-dev.json')

    squad_train_prep_path = os.path.join(parser.data_dir, 'SQuAD-v1.1-train-preprocessed.pkl')
    squad_dev_prep_path = os.path.join(parser.data_dir, 'SQuAD-v1.1-dev-preprocessed.pkl')

    if not force and os.path.exists(squad_train_prep_path) and os.path.exists(squad_dev_prep_path):
        pbar = tqdm(total=2, desc='Load Pickle')
        pbar.display()
        train = pickle.load(open(squad_train_prep_path, 'rb'))
        pbar.update(1)
        pbar.display()
        dev = pickle.load(open(squad_dev_prep_path, 'rb'))
        pbar.update(1)
        pbar.display()
        pbar.close()

        return train, dev

    data_train = load_dataset(squad_train_path)
    data_dev = load_dataset(squad_dev_path)
    logger.info('Loading dataset is done')

    # Initialize Pool
    with Pool(parser.threads, initializer=initialize_pool) as pool:
        initialize_pool()
        annotate(data_train[0])

        train = list(tqdm(pool.imap(annotate,
                                    data_train,
                                    chunksize=parser.chunk),
                          total=len(data_train), desc='TRAIN'))

        dev = list(tqdm(pool.imap(annotate,
                                  data_dev,
                                  chunksize=parser.chunk),
                        total=len(data_dev), desc='TEST '))

    n_train = len(train)
    train = list(filter(lambda x: x is not None, train))
    dev = list(filter(lambda x: x is not None, dev))

    pickle.dump(train, open(squad_train_prep_path, 'wb'))
    pickle.dump(dev, open(squad_dev_prep_path, 'wb'))

    logger.info(f'Removed inconsistent samples: {n_train - len(train)} samples have been removed in {n_train} samples')
    return train, dev


def initialize_pool():
    global spacy_
    spacy_ = spacy.load('en', parser=False)


def annotate(row) -> Union[dict, None]:
    _id, context, question, answers = row
    ctx_doc, ctx_tokens, ctx_lowers = clean_text(context)
    q_doc, q_tokens, q_lowers = clean_text(question)

    # Context -> Span, Tag, Entity
    ctx_spans = [(w.idx, w.idx + len(w.text)) for w in ctx_doc]
    ctx_tags = [w.tag_ for w in ctx_doc]
    ctx_ents = [w.ent_type_ for w in ctx_doc]

    # Question -> Lemmatization(표제어추출), Stemming(어간추출)
    # Lemmatization은 am are is -> be 처럼
    q_lemmas = [w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in q_doc]
    q_lemmas_set = set(q_lemmas)
    q_tokens_set = set(q_tokens)
    q_lowers_set = set(q_lowers)

    # Matching between Context and Question
    match_origins = [w in q_tokens_set for w in ctx_tokens]
    match_lowers = [w in q_lowers_set for w in ctx_lowers]
    match_lemmas = [(w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in q_lemmas_set for w in ctx_doc]

    # Term Frequencies
    ctx_counter = Counter(ctx_lowers)
    ctx_total = len(ctx_lowers)
    ctx_tf = [ctx_counter[w] / ctx_total for w in ctx_tokens]

    # Final Context Features
    ctx_features = list(zip(match_origins, match_lowers, match_lemmas, ctx_tf))

    # Preprocess Answers
    start_indicex, end_indices = list(zip(*ctx_spans))
    new_answers = []
    for _a in answers:
        try:
            new_answers.append((_a[0], _a[1], start_indicex.index(_a[0]), end_indices.index(_a[1]), _a[2]))
        except ValueError:
            pass

    if not len(new_answers):
        return None
    answers = tuple(new_answers)

    return {'_id': _id, 'context': context, 'ctx_tokens': ctx_tokens, 'ctx_features': ctx_features,
            'ctx_spans': ctx_spans, 'ctx_tags': ctx_tags, 'ctx_ents': ctx_ents, 'q_tokens': q_tokens,
            'answers': answers}


def clean_text(text: str) -> Tuple[Doc, List[str], List[str]]:
    global spacy_
    text_doc = re.sub(r'\s', ' ', text.strip())
    text_doc: Doc = spacy_(text_doc)

    tokens = [unicodedata.normalize('NFD', t.text) for t in text_doc]
    lowers = [w.lower() for w in tokens]

    return text_doc, tokens, lowers


def build_vocabulary(vocab, train, dev) -> Tuple[List[str], List[str], List[str],
                                                 Counter, Counter, Counter,
                                                 dict, dict, dict]:
    """
    빈도에 따라서 단어를 정렬하는 것인데.. 이거 왜 하는걸까?
    """
    full = train + dev
    q_tokens = [d['q_tokens'] for d in full]
    ctx_tokens = [d['ctx_tokens'] for d in full]
    ctx_tags = [d['ctx_tags'] for d in full]
    ctx_ents = [d['ctx_ents'] for d in full]

    # Counting
    counter_vocab = Counter(w for doc in q_tokens + ctx_tokens for w in doc)
    counter_tag = Counter(w for row in full for w in row['ctx_tags'])
    counter_ent = Counter(w for row in full for w in row['ctx_ents'])

    # Sort and add PAD, UNK
    ctx_tags = sorted(counter_tag, key=counter_tag.get, reverse=True)
    ctx_ents = sorted(counter_ent, key=counter_ent.get, reverse=True)
    vocab = sorted([t for t in counter_vocab if t in vocab], key=counter_vocab.get, reverse=True)
    vocab.insert(0, '<PAD>')
    vocab.insert(1, '<UNK>')

    # build Index
    vocab2id = {w: i for i, w in enumerate(vocab)}
    tag2id = {w: i for i, w in enumerate(ctx_tags)}
    ent2id = {w: i for i, w in enumerate(ctx_ents)}

    logger.info('Vocabulary size: {}'.format(len(vocab)))
    logger.info('Found {} POS tags.'.format(len(ctx_tags)))
    logger.info('Found {} entity tags: {}'.format(len(ctx_ents), ctx_ents))

    # Add ID
    add_id_ = partial(add_id, vocab2id=vocab2id, tag2id=tag2id, ent2id=ent2id)
    [add_id_(row) for row in train]
    [add_id_(row) for row in dev]

    return vocab, ctx_tags, ctx_ents, counter_vocab, counter_tag, counter_ent, vocab2id, tag2id, ent2id


def build_embedding(parser, vocab, vocab2id: dict) -> Tuple[np.array, np.array]:
    vocab_size, vector_size = (len(vocab), parser.embedding_size)

    embed_matrix = np.zeros([vocab_size, vector_size])
    embed_counts = np.zeros(vocab_size)
    embed_counts[:2] = 1

    glove_path = os.path.join(parser.embed_dir, 'glove.840B.300d.txt')

    with open(glove_path, 'rt') as f:
        for line in f:
            elems = line.rstrip().split(' ')
            word = unicodedata.normalize('NFD', elems[0])
            if word in vocab2id:
                word_id = vocab2id[word]
                embed_counts[word_id] += 1
                embed_matrix[word_id] += [float(v) for v in elems[1:]]
    embed_matrix /= embed_counts.reshape((-1, 1))

    return embed_matrix, embed_counts


def add_id(row, vocab2id, tag2id, ent2id, unk_id=1):
    row['q_token_ids'] = [vocab2id[word] if word in vocab2id else unk_id for word in row['q_tokens']]
    row['ctx_token_ids'] = [vocab2id[word] if word in vocab2id else unk_id for word in row['ctx_tokens']]
    row['ctx_tag_ids'] = [tag2id[tag] for tag in row['ctx_tags']]
    row['ctx_ent_ids'] = [ent2id[ent] for ent in row['ctx_ents']]


def main():
    parser = init()

    # Load raw dataset - SQuAD dataset
    train, dev = load_preprocessed_squad_data(parser, force=False)

    # Load voccvulary data (word data from word vectors)
    vocab = load_vocab_data(parser)

    # Build Vocabulary
    _r = build_vocabulary(vocab, train, dev)
    vocab, ctx_tags, ctx_ents, counter_vocab, counter_tag, counter_ent, vocab2id, tag2id, ent2id = _r

    # Build Embedding Layer
    embed_matrix, embed_counts = build_embedding(parser, vocab, vocab2id)

    # Save data and meta as message pack
    meta = {
        'vocab': vocab,
        'ctx_tags': ctx_tags,
        'ctx_ents': ctx_ents,
        'embed_matrix': embed_matrix.tolist()
    }
    data = {
        'train': train,
        'dev': dev
    }

    sample = {
        'train': train[:parser.sample_size],
        'dev': dev[:parser.sample_size]
    }

    with open(os.path.join(parser.data_dir, 'meta.msgpack'), 'wb') as f:
        msgpack.dump(meta, f)

    with open(os.path.join(parser.data_dir, 'data.msgpack'), 'wb') as f:
        msgpack.dump(data, f)

    with open(os.path.join(parser.data_dir, 'sample.msgpack'), 'wb') as f:
        msgpack.dump(sample, f)

    logger.info(f'train   : {len(train)}')
    logger.info(f'dev     : {len(dev)}')
    logger.info(f'vocab   : {len(vocab)}')
    logger.info(f'ctx_tags: {len(ctx_tags)}')
    logger.info(f'ctx_ents: {len(ctx_ents)}')
    logger.info(f'embed   : {embed_matrix.shape}')
    logger.info('Preprocessing Done successfully')


if __name__ == '__main__':
    main()

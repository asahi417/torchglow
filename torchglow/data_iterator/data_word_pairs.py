""" Data iterator for word pairs dataset """
import os
import random
import logging
# from itertools import chain
from typing import List

import torch
import numpy as np
from gensim.models import KeyedVectors, fasttext

from .language_models import BERT
from ..util import wget, load_pickle, word_pair_format

CACHE_DIR = '{}/.cache/torchglow/word_embedding'.format(os.path.expanduser('~'))
COMMON_WORD_URL = 'https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/common_word.pkl'
COMMON_WORD_PAIRS_URL = 'https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/common_word_pairs.pkl'

__all__ = ('get_dataset', 'get_iterator_word_embedding', 'get_iterator_bert')


def get_dataset(data_iterator,
                data_name: str = 'common_word',
                parallel: bool = True,
                validation_rate: float = 0,
                relative_format: bool = True):
    """ Get word dataset iterator.

    Parameters
    ----------
    data_iterator : torch.utils.data.Dataset
        Iterator produced by `torchglow.data_iterator.get_iterator_bert` or `torchglow.data_iterator.get_iterator_fasttext`.
    data_name : str
        Data name ('common_word' or 'common_word_pair').
    parallel : bool
        Parallel processing.
    validation_rate : float
        Ratio of validation set.
    relative_format : bool
        Convert each pair (A, B) to the relative embedding input format 'A__B'.

    Returns
    -------
    (iterator_train, iterator_valid)
    """
    if data_name == 'common_word':
        path_data = '{}/common_word.pkl'.format(CACHE_DIR)
        if not os.path.exists(path_data):
            wget(COMMON_WORD_URL, CACHE_DIR)
        data = load_pickle(path_data)
        random.Random(0).shuffle(data)
    elif data_name == 'common_pair_word':
        path_data = '{}/common_word_pairs.pkl'.format(CACHE_DIR)
        if not os.path.exists(path_data):
            wget(COMMON_WORD_PAIRS_URL, CACHE_DIR)
        data = load_pickle(path_data)
        random.Random(0).shuffle(data)

        # assert data_format in ['relative', 'fasttext', 'bert'], data_format
        if relative_format:
            # convert word pair to pair-format of relative embeddings
            data = [word_pair_format(d) for d in data]
        # if data_format == 'relative':
        #     # convert word pair to pair-format of relative embeddings
        #     data = [word_pair_format(d) for d in data]
        # elif data_format == 'fasttext':
        #     # convert word pair to word-level data_iterator
        #     data = list(set(list(chain(*data))))
    else:
        raise ValueError('unknown data: {}'.format(data_name))
    try:
        if data_iterator.model_vocab is not None:
            data = list(filter(lambda x: x in data_iterator.model_vocab, data))
    except AttributeError:
        pass

    if validation_rate == 0:
        return data_iterator(data, parallel=parallel), None
    n = int(len(data) * validation_rate)
    valid_set = data_iterator(data[:n], parallel=parallel)
    train_set = data_iterator(data[n:], parallel=parallel)
    return train_set, valid_set


def get_iterator_bert(model: str, max_length: int = 32, embedding_layers: List = -1, mode: str = 'mask'):
    """ Get data iterator with all pipelines required as preprocessing for BERT embedding.

    Parameters
    ----------
    model : str
        Transformers model alias.
    max_length : int
        Model length.
    embedding_layers : int
        Embedding layers to get average.
    mode : str
        - `mask` to get the embedding for a word pair by [MASK] token, eg) (A, B) -> A [MASK] B
        - `cls` to get the embedding on the [CLS] token

    Returns
    -------
    ((torch.utils.data.Dataset, a function to map the encoded input to BERT embedding) hidden dimension)
    """
    lm = BERT(model=model, max_length=max_length, embedding_layers=embedding_layers, mode=mode)
    return (lm.preprocess, lm.to_embedding), lm.hidden_size


def get_iterator_word_embedding(model_type: str):
    """ Get data iterator with all pipelines required as preprocessing for word embedding.

    Parameters
    ----------
    model_type : str
        Model type (glove, fasttext, w2v).

    Returns
    -------
    (torch.utils.data.Dataset, embedding dimension)
    """
    urls = {
        'fasttext': ['https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip', 'crawl-300d-2M-subword.zip'],
        'glove': ['https://drive.google.com/u/0/uc?id=1DbLuxwDlTRDbhBroOVgn2_fhVUQAVIqN&export=download', 'glove.840B.300d.gensim.bin.tar.gz'],
        'w2v': ["https://drive.google.com/u/0/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download", 'GoogleNews-vectors-negative300.bin.gz']
    }

    # load embedding model
    url, filename = urls[model_type]
    model_path_bin = '{}/{}'.format(CACHE_DIR, filename).replace('.tar.gz', '').replace('.zip', '.bin').replace('.gz', '')
    if not os.path.exists(model_path_bin):
        logging.debug('downloading word embedding model from {}'.format(url))
        wget(url, CACHE_DIR, gdrive_filename=filename)

    if model_type == 'fasttext':
        model = fasttext.load_facebook_model(model_path_bin)
    else:
        model = KeyedVectors.load_word2vec_format(model_path_bin, binary=True)

    class Dataset(torch.utils.data.Dataset):
        """ 1D data_iterator iterator with RELATIVE word embedding """
        model_vocab = set(model.vocab.keys()) if model_type != 'fasttext' else None

        def __init__(self, vocab, **kwargs):
            self.vocab = vocab

        def __len__(self):
            return len(self.vocab)

        def __getitem__(self, idx):
            vector = model.wv.__getitem__(self.vocab[idx])
            tensor = torch.tensor(np.array(vector), dtype=torch.float32)
            return tensor.reshape(len(tensor), 1, 1)  # return in CHW shape

    return Dataset, model.vector_size

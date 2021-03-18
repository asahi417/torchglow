""" Data iterator for word pairs dataset """
import os
import random
import logging
from itertools import chain
from typing import List

import torch
import numpy as np
from gensim.models import KeyedVectors, fasttext

from .language_models import BERT
from ..util import open_compressed_file, load_pickle

CACHE_DIR = '{}/.cache/torchglow/word_embedding'.format(os.path.expanduser('~'))
COMMON_WORD_PAIRS_URL = 'https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/common_word_pairs.pkl.tar.gz'


def get_dataset_word_pairs(data_iterator,
                           parallel: bool = True,
                           validation_rate: float = 0.2,
                           data_format: str = None):
    """ Get word pairs dataset iterator.

    Parameters
    ----------
    data_iterator : torch.utils.data.Dataset
        Iterator produced by `torchglow.data_iterator.get_iterator_bert` or `torchglow.data_iterator.get_iterator_fasttext`.
    parallel : bool
        Parallel processing.
    validation_rate : float
        Ratio of validation set.
    data_format : str
        If provided, convert each pair (A, B) to the format
        - 'pair':  'A__B'
        - 'word': flatten all pair to be a list of individual words

    Returns
    -------
    (iterator_train, iterator_valid)
    """
    # download common-word-pairs data_iterator
    path_data = '{}/common_word_pairs.pkl'.format(CACHE_DIR)
    if not os.path.exists(path_data):
        open_compressed_file(COMMON_WORD_PAIRS_URL, CACHE_DIR)
    data = load_pickle(path_data)
    random.Random(0).shuffle(data)

    assert data_format is None or data_format in ['pair', 'word'], data_format
    if data_format == 'pair':
        # convert word pair to pair-format of relative embeddings
        data = ['__'.join([a.lower(), b.lower()]) for a, b in data]
    elif data_format == 'word':
        # convert word pair to word-level data_iterator
        data = list(set(list(chain(*data))))

    if validation_rate == 0:
        return data_iterator(data, parallel=parallel), None
    n = int(len(data) * validation_rate)
    valid_set = data_iterator(data[:n], parallel=parallel)
    train_set = data_iterator(data[n:], parallel=parallel)
    return train_set, valid_set


def get_iterator_bert(model: str, max_length: int = 32, embedding_layers: List = -1, mode: str = 'cls'):
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


def get_iterator_fasttext(model_type: str):
    """ Get data iterator with all pipelines required as preprocessing for word embedding.

    Parameters
    ----------
    model_type : str
        Model type from `fasttext`, `relative_init`, `fasttext_diff`, or `concat_relative_fasttext`

    Returns
    -------
    (torch.utils.data.Dataset, embedding dimension)
    """
    urls = {
        'fasttext': 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip',
        'relative_init': 'https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/relative_init_vectors.bin.tar.gz',
        'fasttext_diff': 'https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/fasttext_diff_vectors.bin.tar.gz',
        'concat_relative_fasttext': 'https://drive.google.com/u/0/uc?id=1CkdsxEl21TUiBmLS6uq55tH6SiHvWGDn&export=download',
    }

    # load embedding model
    url = urls[model_type]
    if model_type == 'concat_relative_fasttext':
        filename = 'concat_relative_fasttext_vectors.bin.tar.gz'
    else:
        filename = os.path.basename(url)
    model_path_bin = '{}/{}'.format(CACHE_DIR, filename).replace('.tar.gz', '').replace('.zip', '.bin')
    if not os.path.exists(model_path_bin):
        logging.debug('downloading word embedding model from {}'.format(url))
        open_compressed_file(url, CACHE_DIR, filename=filename)

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
            tensor = torch.tensor(np.array(model.wv.__getitem__(self.vocab[idx]), dtype=torch.float32))
            return tensor.reshape(len(tensor), 1, 1)  # return in CHW shape

    return Dataset, model.vector_size

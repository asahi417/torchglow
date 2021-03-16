""" Data iterator for gensim embedding model: data is over the all vocabulary of the embedding model """
import os
import logging
import random

import numpy as np
import torch
from gensim.models import KeyedVectors

from ..util import open_compressed_file

CACHE_DIR = '{}/.cache/torchglow/word_embedding'.format(os.path.expanduser('~'))
URL_MODEL = {
    'relative_init': ['https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/relative_init_vectors.bin.tar.gz', 'relative_init_vectors.bin.tar.gz'],
    'fasttext_diff': ['https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/fasttext_diff_vectors.bin.tar.gz', 'fasttext_diff_vectors.bin.tar.gz'],
    'concat_relative_fasttext': ['https://drive.google.com/u/0/uc?id=1CkdsxEl21TUiBmLS6uq55tH6SiHvWGDn&export=download', 'concat_relative_fasttext_vectors.bin.tar.gz']
}

N_DIM = {'relative': 300, 'fasttext_diff': 300, 'concat_relative_fasttext': 600}
__all__ = ('get_dataset_word_embedding', 'get_iterator_word_embedding', 'N_DIM')


def get_dataset_word_embedding(model_type: str, cache_dir: str = None, validation_rate: float = 0.2):
    data_iterator = get_iterator_word_embedding(model_type, cache_dir)
    data = list(data_iterator.model_vocab)
    random.Random(0).shuffle(data)
    if validation_rate == 0:
        return data_iterator(data), None
    n = int(len(data) * validation_rate)

    valid_set = data_iterator(data[:n])
    train_set = data_iterator(data[n:])
    return train_set, valid_set


def get_iterator_word_embedding(model_type: str, cache_dir: str = None):
    cache_dir = cache_dir if cache_dir is not None else CACHE_DIR
    url, filename = URL_MODEL[model_type]
    model_path = '{}/{}'.format(cache_dir, os.path.basename(url))
    model_path_bin = model_path.replace('.tar.gz', '')
    if not os.path.exists(model_path_bin):
        logging.debug('downloading word embedding model from {}'.format(url))
        open_compressed_file(url, cache_dir, filename=filename)

    model = KeyedVectors.load_word2vec_format(model_path_bin, binary=True)

    class DatasetWordEmbedding(torch.utils.data.Dataset):
        """ 1D data iterator with RELATIVE word embedding """
        model_vocab = set(model.vocab.keys())

        def __init__(self, vocab):
            self.vocab = vocab

        def __len__(self):
            return len(self.vocab)

        def __getitem__(self, idx):
            tensor = torch.tensor(np.array(model[self.vocab[idx]]), dtype=torch.float32)
            return tensor.reshape(len(tensor), 1, 1),  # return in CHW shape

    return DatasetWordEmbedding


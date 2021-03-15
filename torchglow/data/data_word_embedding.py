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
    'relative': 'https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/relative_init_vectors.bin.tar.gz',
    'diff_fasttext': 'https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/fasttext_diff_vectors.bin.tar.gz'
}
N_DIM = {
    'relative': 300,
    'diff_fasttext': 300
}
__all__ = ('get_dataset_word_embedding', 'N_DIM')


def get_dataset_word_embedding(model_type: str, cache_dir: str = None, validation_rate: float = 0.2):
    cache_dir = cache_dir if cache_dir is not None else CACHE_DIR
    url = URL_MODEL[model_type]
    model_path = '{}/{}'.format(cache_dir, os.path.basename(url))
    model_path_bin = model_path.replace('.tar.gz', '')
    if not os.path.exists(model_path_bin):
        logging.debug('downloading word embedding model from {}'.format(url))
        open_compressed_file(url, cache_dir)

    model = KeyedVectors.load_word2vec_format(model_path_bin, binary=True)

    class DatasetWordEmbedding(torch.utils.data.Dataset):
        """ 1D data iterator with RELATIVE word embedding """

        def __init__(self, vocab):
            self.vocab = vocab

        def __len__(self):
            return len(self.vocab)

        def __getitem__(self, idx):
            tensor = torch.tensor(np.array(model[self.vocab[idx]]), dtype=torch.float32)
            return tensor.reshape(len(tensor), 1, 1),  # return in CHW shape

    data = list(model.vocab.keys())
    random.Random(0).shuffle(data)
    n = int(len(data) * validation_rate)

    valid_set = DatasetWordEmbedding(data[:n])
    train_set = DatasetWordEmbedding(data[n:])
    return train_set, valid_set

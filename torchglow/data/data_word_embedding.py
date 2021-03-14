""" Data iterators: `celba` and `cifar10`. """
import os
import logging

import torch
from gensim.models import KeyedVectors

from ..util import open_compressed_file

CACHE_DIR = '{}/.cache/torchglow/word_embedding'.format(os.path.expanduser('~'))
URL_MODEL = {
    'relative': 'https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/relative_init_vectors.bin.tar.gz',
    'fasttext': 'https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/fasttext_diff_vectors.bin.tar.gz'
}


class DatasetWordEmbedding(torch.utils.data.Dataset):
    """ 1D data iterator with RELATIVE word embedding """

    def __init__(self, relative: bool = False, root: str = None):

        root = root if root is not None else CACHE_DIR
        url = URL_MODEL['relative'] if relative else URL_MODEL['fasttext']
        model_path = '{}/{}'.format(root, os.path.dirname(url))
        if not os.path.exists(model_path):
            logging.debug('downloading word embedding model from {}'.format(url))
            open_compressed_file(url, root)

        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        self.data = list(self.model.vocab.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tensor = torch.tensor(self.model[self.data[idx]], dtype=torch.float32)
        return tensor.reshape(len(tensor), 1, 1)  # return in CHW shape


def get_dataset_word_embedding(relative: bool = False, cache_dir: str = None):
    return DatasetWordEmbedding(relative, cache_dir)

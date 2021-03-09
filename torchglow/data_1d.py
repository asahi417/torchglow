""" Data iterators: `celba` and `cifar10`. """
import os
import pickle
import struct
import logging
import requests
import tarfile
import zipfile
from glob import glob

import torch

from .util import load_pickle

CACHE_DIR = '{}/.cache/torchglow'.format(os.path.expanduser('~'))


class Dataset1D(torch.utils.data.Dataset):
    """ 1D data iterator """

    def __init__(self, path_to_data: str = None, root: str = None, data: str = None):
        if path_to_data.endswith('.pkl'):
            path_to_data = os.path.dirname(path_to_data)
        self.files = glob(os.path.join(path_to_data, '*'))

        self.data_index = {}
        ind = 0
        for n, p in enumerate(self.files):
            vector = load_pickle(p)
            for i in range(len(vector)):
                self.data_index[ind] = (n, i)
                ind += 1

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        file_index, data_index = self.data_index[idx]
        vector_list = load_pickle(self.files[file_index])
        tensor = torch.tensor(vector_list[data_index], dtype=torch.float32)
        return tensor.reshape(len(tensor), 1, 1)  # return in CHW shape


def get_dataset_1d(path_to_data: str = None, cache_dir: str = None, data: str = None):
    return Dataset1D(path_to_data, cache_dir, data)

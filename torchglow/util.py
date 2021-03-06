import random
import os
import pickle
import tarfile
import zipfile
import gzip
import requests
import json
# from typing import List
from copy import deepcopy

import gdown
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

home_dir = '{}/.cache/torchglow'.format(os.path.expanduser('~'))
module_output_dir = './torchglow_output'


def flatten_list(nested_list):
    """Flatten an arbitrarily nested list, without recursion (to avoid
    stack overflows). Returns a new list, the original list is unchanged.
    >> list(flatten_list([1, 2, 3, [4], [], [[[[[[[[[5]]]]]]]]]]))
    [1, 2, 3, 4, 5]
    >> list(flatten_list([[1, 2], 3]))
    [1, 2, 3]
    """

    def _flatten_list(_nested_list):
        if type(_nested_list) is torch.Tensor:
            _nested_list = _nested_list.cpu().tolist()
        else:
            _nested_list = deepcopy(_nested_list)

        while _nested_list:
            sublist = _nested_list.pop(0)

            if type(sublist) is torch.Tensor:
                sublist = sublist.cpu().tolist()

            if isinstance(sublist, list):
                _nested_list = sublist + _nested_list
            else:
                yield sublist

    return list(_flatten_list(nested_list))


def wget(url, cache_dir: str, gdrive_filename: str = None):
    """ wget and uncompress data_iterator """
    path = _wget(url, cache_dir, gdrive_filename=gdrive_filename)
    if path.endswith('.tar.gz') or path.endswith('.tgz') or path.endswith('.tar'):
        if path.endswith('.tar'):
            tar = tarfile.open(path)
        else:
            tar = tarfile.open(path, "r:gz")
        tar.extractall(cache_dir)
        tar.close()
        os.remove(path)
    elif path.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            with open(path.replace('.gz', ''), 'wb') as f_write:
                f_write.write(f.read())
        os.remove(path)
    elif path.endswith('.zip'):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
        os.remove(path)


def _wget(url: str, cache_dir, gdrive_filename: str = None):
    """ get data from web """
    os.makedirs(cache_dir, exist_ok=True)
    if url.startswith('https://drive.google.com'):
        assert gdrive_filename is not None, 'please provide fileaname for gdrive download'
        return gdown.download(url, '{}/{}'.format(cache_dir, gdrive_filename), quiet=False)
    filename = os.path.basename(url)
    with open('{}/{}'.format(cache_dir, filename), "wb") as f:
        r = requests.get(url)
        f.write(r.content)
    return '{}/{}'.format(cache_dir, filename)


def fix_seed(seed: int = 12):
    """ Fix random seed. """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps=None, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    https://huggingface.co/transformers/_modules/transformers/optimization.html#get_linear_schedule_with_warmup

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        current_step += 1
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if num_training_steps is None:
            return 1
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def save_pickle(_list, export_dir, chunk_size: int):
    os.makedirs(export_dir, exist_ok=True)
    for s_n, n in enumerate(range(0, len(_list), chunk_size)):
        _list_sub = _list[n:min(n + chunk_size, len(_list))]
        with open('{}/list.{}.pkl'.format(export_dir, s_n), "wb") as fp:
            pickle.dump(_list_sub, fp)


def load_pickle(path):
    with open(path, "rb") as fp:
        return pickle.load(fp)


import random
import os
import pickle
import tarfile
import zipfile
import requests
import json
from typing import List

import gdown
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR


def open_compressed_file(url, cache_dir, filename: str = None):
    """ wget and uncompress data_iterator """
    path = wget(url, cache_dir, filename=filename)
    if path.endswith('.tar.gz') or path.endswith('.tgz'):
        tar = tarfile.open(path, "r:gz")
        tar.extractall(cache_dir)
        tar.close()
    elif path.endswith('.zip'):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)


def wget(url: str, cache_dir, filename: str = None):
    """ wget """
    os.makedirs(cache_dir, exist_ok=True)
    if url.startswith('https://drive.google.com'):
        if filename:
            return gdown.download(url, '{}/{}'.format(cache_dir, filename), quiet=False)
        else:
            return gdown.download(url, cache_dir, quiet=False)
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


def get_analogy_baseline(_type: str = 'fasttext_diff'):
    """ Get prediction files for analogy dataset on the plain relative embedding for baseline. """
    cache_dir = '{}/.cache/torchglow/analogy/baseline'.format(os.path.expanduser('~'))
    os.makedirs(cache_dir, exist_ok=True)
    assert _type in ['fasttext_diff', 'concat_relative_fasttext', 'relative_init'], _type
    prediction_files = '{}/{}.json'.format(cache_dir, _type)
    if not os.path.exists(prediction_files):
        open_compressed_file(
            url='https://raw.githubusercontent.com/asahi417/AnalogyDataset/master/predictions/{}.json'.format(_type),
            cache_dir=cache_dir)
    with open(prediction_files) as f:
        return json.load(f)


def get_analogy_dataset(data_name: str):
    """ Get SAT-type dataset: a list of (answer: int, prompts: list, stem: list, choice: list)"""
    cache_dir = '{}/.cache/torchglow/analogy/dataset'.format(os.path.expanduser('~'))
    root_url_analogy = 'https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0'
    assert data_name in ['sat', 'u2', 'u4', 'google', 'bats'], 'unknown data_iterator: {}'.format(data_name)
    if not os.path.exists('{}/{}'.format(cache_dir, data_name)):
        open_compressed_file('{}/{}.zip'.format(root_url_analogy, data_name), cache_dir)
    with open('{}/{}/test.jsonl'.format(cache_dir, data_name), 'r') as f:
        test_set = list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))
    with open('{}/{}/valid.jsonl'.format(cache_dir, data_name), 'r') as f:
        val_set = list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))
    return val_set, test_set


def word_pair_format(pair: List):
    """ transform word pair into the format of relative format """
    return '__'.join(pair).replace(' ', '_').lower()

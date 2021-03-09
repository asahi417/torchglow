import random
import os
import pickle
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR


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

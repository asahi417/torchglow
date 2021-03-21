""" configuration manager """
import os
import random
import json
import string
import logging
import torch
import shutil
from glob import glob
from typing import List

__all__ = 'Config'


class Config:

    def __init__(self, export_dir: str = None, checkpoint_path: str = None, **kwargs):

        if checkpoint_path is not None:
            assert os.path.exists(checkpoint_path), checkpoint_path
            self.config = self.safe_open('{}/config.json'.format(checkpoint_path))
            self.cache_dir = checkpoint_path
            if os.path.exists('{}/config_train.json'.format(checkpoint_path)):
                self.epoch_elapsed = self.safe_open('{}/config_train.json'.format(checkpoint_path))['epoch_elapsed']
            self.epoch_elapsed = None
        else:
            assert export_dir, 'either `export_dir` or `checkpoint_path` is required'
            self.config = kwargs
            logging.info('hyperparameters')
            for k, v in self.config.items():
                logging.info('\t * {}: {}'.format(k, v))
            ex_configs = {i: self.safe_open(i) for i in glob('{}/*/config.json'.format(export_dir))}
            same_config = list(filter(lambda x: x[1] == self.config, ex_configs.items()))
            if len(same_config) != 0:
                input('\ncheckpoint already exists: {}\n enter to overwrite >>>'.format(same_config[0]))
                for _p, _ in same_config:
                    shutil.rmtree(os.path.dirname(_p))
            self.cache_dir = '{}/{}'.format(export_dir, self.get_random_string(
                [os.path.basename(i.replace('/config.json', '')) for i in ex_configs.keys()]
            ))
            self.epoch_elapsed = 0
        self.__dict__.update(self.config)
        self.model_weight_path = '{}/model.pt'.format(self.cache_dir)
        self.model_weight_path_inter = {k.split('model.')[-1].replace('.pt', ''): k
                                        for k in glob('{}/model.*.pt'.format(self.cache_dir))}
        self.optimizer_path = '{}/optimizer.pt'.format(self.cache_dir)
        self.optimizer_path_inter = {k.split('optimizer.')[-1].replace('.pt', ''): k
                                     for k in glob('{}/optimizer.*.pt'.format(self.cache_dir))}
        self.__cache_init()

    @property
    def is_trained(self):
        return os.path.exists(self.model_weight_path) or len(self.model_weight_path_inter) > 0

    @property
    def is_fully_trained(self):
        return self.epoch_elapsed >= self.epoch

    def __cache_init(self):
        if not os.path.exists('{}/config.json'.format(self.cache_dir)):
            os.makedirs(self.cache_dir, exist_ok=True)
            with open('{}/config.json'.format(self.cache_dir), 'w') as f:
                json.dump(self.config, f)

    def save(self,
             model_state_dict,
             epoch: int,
             optimizer_state_dict=None,
             scheduler_state_dict=None,
             last_model: bool = False):
        logging.info('saving model weight in {}'.format(self.cache_dir))
        if last_model:
            torch.save(model_state_dict, self.model_weight_path)
            if optimizer_state_dict and scheduler_state_dict:
                torch.save({
                    'optimizer_state_dict': optimizer_state_dict,
                    'scheduler_state_dict': scheduler_state_dict,
                    'epoch_elapsed': epoch
                }, self.optimizer_path)
        else:
            torch.save(model_state_dict, '{}/model.{}.pt'.format(self.cache_dir, epoch))
            if optimizer_state_dict and scheduler_state_dict:
                torch.save({
                    'optimizer_state_dict': optimizer_state_dict,
                    'scheduler_state_dict': scheduler_state_dict,
                    'epoch_elapsed': epoch
                }, '{}/optimizer.{}.pt'.format(self.cache_dir, epoch))

    @staticmethod
    def get_random_string(exclude: List = None, length: int = 6):
        while True:
            tmp = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
            if exclude is None:
                break
            elif tmp not in exclude:
                break
        return tmp

    @staticmethod
    def safe_open(_file):
        with open(_file, 'r') as f:
            return json.load(f)

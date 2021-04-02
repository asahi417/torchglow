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
            assert len(glob('{}/*.pt'.format(checkpoint_path))) > 0, checkpoint_path
            self.config = self.safe_open('{}/config.json'.format(checkpoint_path))
            self.cache_dir = checkpoint_path
            self.epoch_saved = [int(k.split('model.')[-1].replace('.pt', '')) for k in glob(
                '{}/model.*.pt'.format(self.cache_dir))]
            self.path_model = {e: '{}/model.{}.pt'.format(self.cache_dir, e) for e in self.epoch_saved}
            self.path_optimizer = {e: '{}/optimizer.{}.pt'.format(self.cache_dir, e) for e in self.epoch_saved}
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
            self.path_model = None
            self.path_optimizer = None
        self.__dict__.update(self.config)
        self.__cache_init()

    @property
    def is_trained(self):
        return self.path_model

    def __cache_init(self):
        if not os.path.exists('{}/config.json'.format(self.cache_dir)):
            os.makedirs(self.cache_dir, exist_ok=True)
            with open('{}/config.json'.format(self.cache_dir), 'w') as f:
                json.dump(self.config, f)

    def save(self,
             model_state_dict,
             epoch: int,
             optimizer_state_dict=None,
             scheduler_state_dict=None):
        logging.info('saving model weight/optimizer at {}'.format(self.cache_dir))
        torch.save(model_state_dict, '{}/model.{}.pt'.format(self.cache_dir, epoch))
        if optimizer_state_dict is not None or scheduler_state_dict is not None:
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

""" configuration manager """
import os
import random
import json
import string
import logging
import torch
from glob import glob
from typing import List

__all__ = 'Config'


class Config:

    def __init__(self, export_dir: str = None, checkpoint_path: str = None, **kwargs):

        if checkpoint_path is not None:
            assert os.path.exists(checkpoint_path), checkpoint_path
            self.config = self.safe_open('{}/config.json'.format(checkpoint_path))
            self.cache_dir = checkpoint_path
        else:
            assert export_dir, 'either `export_dir` or `checkpoint_path` is required'
            self.config = kwargs
            logging.info('hyperparameters')
            for k, v in self.config.items():
                logging.info('\t * {}: {}'.format(k, v))
            ex_configs = {i: self.safe_open(i) for i in glob('{}/*/config.json'.format(export_dir))}
            print(ex_configs)
            same_config = list(filter(lambda x: x[1] == self.config, ex_configs.items()))
            print(same_config)
            assert len(same_config) == 0, 'checkpoint already exists: '.format(same_config[0])
            self.cache_dir = '{}/{}'.format(export_dir, self.get_random_string(
                [os.path.basename(i.replace('/config.json', '')) for i in ex_configs.keys()]
            ))
        self.__dict__.update(self.config)
        self.model_weight_path = '{}/model.pt'.format(self.cache_dir)

    @property
    def is_trained(self):
        return os.path.exists(self.model_weight_path)

    def __cache_init(self):
        assert not self.is_trained, 'model has already been trained'
        if not os.path.exists('{}/config.json'.format(self.cache_dir)):
            os.makedirs(self.cache_dir, exist_ok=True)
            with open('{}/config.json'.format(self.cache_dir), 'w') as f:
                json.dump(self.config, f)

    def save(self, model_state_dict, epoch: int = None):
        self.__cache_init()
        logging.info('saving model weight in {}'.format(self.cache_dir))
        if epoch:
            torch.save(model_state_dict, '{}/model.{}.pt'.format(self.cache_dir, epoch))
        else:
            torch.save(model_state_dict, '{}/model.pt'.format(self.cache_dir))

    @staticmethod
    def get_random_string(exclude: List = None, length: int = 6):
        print(exclude)
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
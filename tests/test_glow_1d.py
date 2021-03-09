""" UnitTest """
import os
import unittest
import logging
import shutil
from glob import glob
from random import random

import torchglow

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
test_embedding_file = './tests/data_sample'
dim = 10


def generate_sample_embedding():
    if not glob('{}/*.pkl'.format(test_embedding_file)):
        tmp = [list(map(lambda x: random(), range(dim))) for _ in range(15)]
        torchglow.save_pickle(tmp, test_embedding_file, chunk_size=3)


class Test(unittest.TestCase):
    """ Test """

    def test(self):
        generate_sample_embedding()
        # torchglow.get_dataset(test_embedding_file)

        if os.path.exists('tests/ckpt_1d'):
            shutil.rmtree('tests/ckpt_1d')
        model = torchglow.Glow1D(
            n_channel=dim,
            path_to_data_valid=test_embedding_file,
            path_to_data=test_embedding_file,
            lr=0.001,
            training_step=2,
            epoch=2,
            export_dir='tests/ckpt_1d',
            batch=2
        )
        model.train()


if __name__ == "__main__":
    unittest.main()

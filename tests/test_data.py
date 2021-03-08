""" UnitTest """
import os
import unittest
import logging
import shutil
import random

import torch
import numpy as np
from torchglow.data import get_dataset, get_decoder


def fix_seed(seed=12):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
n_img = 5


def get_image(images, decoder, data):
    for i in range(n_img):
        decoder(images[i]).save('./tests/img/test_data/{}.{}.png'.format(data, i))


class Test(unittest.TestCase):
    """ Test """

    def test(self):

        if os.path.exists('./tests/img/test_data'):
            shutil.rmtree('./tests/img/test_data')
        os.makedirs('./tests/img/test_data', exist_ok=True)

        for data in ['cifar10', 'celeba']:
            fix_seed()
            logging.info('get loader: {}'.format(data))
            decoder = get_decoder()
            train, val = get_dataset(data)
            logging.info('initialize iterator')
            train = iter(train)
            val = iter(val)
            logging.info('generating images')
            for i in range(n_img):
                img = next(train)[0]
                decoder(img).save('./tests/img/test_data/{}.train.{}.png'.format(data, i))
                img = next(val)[0]
                decoder(img).save('./tests/img/test_data/{}.valid.{}.png'.format(data, i))

        fix_seed()
        train, val = get_dataset('celeba', image_size=64, n_bits_x=5)
        decoder = get_decoder(n_bits_x=5)
        val = iter(val)
        for i in range(n_img):
            img = next(val)[0]
            decoder(img).save('./tests/img/test_data/{}.valid.{}.transform.png'.format(data, i))


if __name__ == "__main__":
    unittest.main()

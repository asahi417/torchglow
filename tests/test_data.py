""" UnitTest """
import os
import unittest
import logging
import shutil
import random

import torch
import torchvision
import numpy as np
from torchglow.util import fix_seed
from torchglow.data_iterator import get_dataset_image, get_image_decoder

# torchvision.utils.save_image()


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
n_img = 16


def get_image(images, decoder, data):
    for i in range(n_img):
        decoder(images[i]).save('./tests/img/test_data/{}.{}.png'.format(data, i))


class Test(unittest.TestCase):
    """ Test """

    def test_image(self):

        if os.path.exists('./tests/img/test_data'):
            shutil.rmtree('./tests/img/test_data')
        os.makedirs('./tests/img/test_data', exist_ok=True)

        for data in ['cifar10', 'celeba']:
            fix_seed()
            logging.info('get loader: {}'.format(data))
            decoder = get_image_decoder()
            _, val = get_dataset_image(data)
            val = iter(val)
            images = []
            for i in range(n_img):
                x, y = next(val)
                images.append(x)

            image_tensor_batch = decoder(images)
            torchvision.utils.save_image(
                image_tensor_batch,
                './tests/img/test_data/{}.png'.format(data)
                          # nrow=12
                )
            # .save('./tests/img/test_data/{}.train.{}.png'.format(data, i))
            # img = next(val)[0]
            # decoder(img).save('./tests/img/test_data/{}.valid.{}.png'.format(data, i))
            input()

        fix_seed()
        train, val = get_dataset_image('celeba', image_size=64, n_bits_x=5)
        decoder = get_image_decoder(n_bits_x=5)
        val = iter(val)
        for i in range(n_img):
            img = next(val)[0]
            decoder(img).save('./tests/img/test_data/{}.valid.{}.transform.png'.format(data, i))

    # def test_bert(self):
    #     for model in ['roberta-large', 'bert-large-cased']:
    #         (iterator, _), dim = get_iterator_bert(model, mode='mask')
    #         logging.info('\t hidden dimension: {}'.format(dim))
    #         get_dataset_word_pairs(iterator, data_format='bert')
    #
    #         # for future bertflow implementation
    #         # (iterator, _), dim = get_iterator_bert(model, mode='cls')
    #
    # def test_fasttext(self):
    #     for model in ['fasttext']:
    #         iterator, dim = get_iterator_fasttext(model)
    #         logging.info('\t hidden dimension: {}'.format(dim))
    #         get_dataset_word_pairs(iterator, data_format='fasttext')
    #
    #     for model in ['relative_init', 'fasttext_diff', 'concat_relative_fasttext']:
    #         iterator, _ = get_iterator_fasttext(model)
    #         logging.info('\t hidden dimension: {}'.format(dim))
    #         get_dataset_word_pairs(iterator, data_format='relative')


if __name__ == "__main__":
    unittest.main()

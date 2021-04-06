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
from torchglow.data_iterator import get_dataset, get_iterator_word_embedding, get_dataset_image, get_image_decoder

# torchvision.utils.save_image()


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')


# def get_image(images, decoder, data, n_img = 32):
#     for i in range(n_img):
#         decoder(images[i]).save('./tests/img/test_data/{}.{}.png'.format(data, i))


class Test(unittest.TestCase):
    """ Test """

    # def test_image(self):
    #
    #     if os.path.exists('./tests/img/test_data'):
    #         shutil.rmtree('./tests/img/test_data')
    #     os.makedirs('./tests/img/test_data', exist_ok=True)
    #
    #     for data in ['cifar10', 'celeba']:
    #         fix_seed()
    #         logging.info('get loader: {}'.format(data))
    #         decoder = get_image_decoder()
    #         dataset, _ = get_dataset_image(data)
    #         dataset = iter(dataset)
    #         images = []
    #         if data == 'cifar10':
    #             n_img = 64
    #         else:
    #             n_img = 32
    #         for i in range(n_img):
    #             x, y = next(dataset)
    #             images.append(x)
    #
    #         image_tensor_batch = decoder(images, keep_tensor=True)
    #         torchvision.utils.save_image(
    #             image_tensor_batch,
    #             './tests/img/test_data/{}.png'.format(data),
    #             normalize=True,
    #         )

    # def test_bert(self):
    #     for model in ['roberta-large', 'bert-large-cased']:
    #         (iterator, _), dim = get_iterator_bert(model, mode='mask')
    #         logging.info('\t hidden dimension: {}'.format(dim))
    #         get_dataset_word_pairs(iterator, data_format='bert')
    #
    #         # for future bertflow implementation
    #         # (iterator, _), dim = get_iterator_bert(model, mode='cls')
    #
    def test_fasttext(self):
        iterator, dim = get_iterator_word_embedding('glove')
        logging.info('\t hidden dimension: {}'.format(dim))
        data = get_dataset(iterator)

        for model in ['relative_init', 'fasttext_diff', 'concat_relative_fasttext']:
            iterator, _ = get_iterator_fasttext(model)
            logging.info('\t hidden dimension: {}'.format(dim))
            get_dataset_word_pairs(iterator, data_format='relative')


if __name__ == "__main__":
    unittest.main()

""" UnitTest """
import os
import unittest
import logging
import shutil

import torchvision
from torchglow.util import fix_seed
from torchglow.data_iterator import get_dataset_image, get_image_decoder


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')


def get_image(images, decoder, data, n_img=32):
    for i in range(n_img):
        decoder(images[i]).save('./tests/img/test_data/{}.{}.png'.format(data, i))


class Test(unittest.TestCase):
    """ Test """

    def test_image(self):

        if os.path.exists('./tests/img/test_data'):
            shutil.rmtree('./tests/img/test_data')
        os.makedirs('./tests/img/test_data', exist_ok=True)

        for data in ['celeba', 'cifar10']:
            fix_seed()
            logging.info('get loader: {}'.format(data))
            decoder = get_image_decoder()
            dataset, _ = get_dataset_image(data)
            dataset = iter(dataset)
            images = []
            if data == 'cifar10':
                n_img = 64
            else:
                n_img = 32
            for i in range(n_img):
                x, y = next(dataset)
                images.append(x)

            image_tensor_batch = decoder(images, keep_tensor=True)
            torchvision.utils.save_image(
                image_tensor_batch,
                './tests/img/test_data/{}.png'.format(data),
                normalize=True,
            )


if __name__ == "__main__":
    unittest.main()

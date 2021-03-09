""" UnitTest """
import os
import unittest
import logging
import shutil
import pickle
from glob import glob
from random import random

from torchglow import Glow

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
test_embedding_file = './tests/embedding_sample'
dim = 10


def generate_sample_embedding():
    if not glob('{}/*.pkl'.format(test_embedding_file)):
        for i in range(3):
            e = [list(map(random, range(10))), list(map(random, range(10))), list(map(random, range(10)))]
            with open('{}/embedding_{}.pkl'.format(test_embedding_file, i), "wb") as fp:
                pickle.dump(e, fp)
    # with open(p, "rb") as fp:
    #     h_list += map(lambda x: x[layer], pickle.load(fp))


class Test(unittest.TestCase):
    """ Test """

    def test(self):
        generate_sample_embedding()
        test_embedding_file

        if os.path.exists('tests/ckpt'):
            shutil.rmtree('tests/ckpt')
        model = Glow(
            lr=0.0001,
            training_step=5,
            epoch=2,
            export_dir='tests/ckpt',
            data='cifar10',
            image_size=32,
            batch=2,
        )
        model.train(epoch_valid=10)
        ckpt = model.checkpoint_dir

        # test reverse mode
        model = Glow(checkpoint_path=ckpt)
        x, y = model.reconstruct(sample_size=5)
        for n, (_x, _y) in enumerate(zip(x, y)):
            _x.save('./tests/ckpt/image.{}.org.png'.format(n))
            _y.save('./tests/ckpt/image.{}.rec.png'.format(n))


if __name__ == "__main__":
    unittest.main()

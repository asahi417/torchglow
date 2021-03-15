""" UnitTest """
import os
import unittest
import logging
import shutil
from torchglow import Glow, GlowWordEmbedding

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')


class Test(unittest.TestCase):
    """ Test """

    def test(self):
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

    def test_we(self):
        if os.path.exists('tests/ckpt_we'):
            shutil.rmtree('tests/ckpt_we')
        model = GlowWordEmbedding(
            lr=0.0001,
            training_step=5,
            epoch=2,
            export_dir='tests/ckpt_we',
            batch=2,
        )
        model.train(epoch_valid=10)
        x, y = model.reconstruct(10)
        p = 3
        for n, (_x, _y) in enumerate(zip(x, y)):
            # reduce precision
            assert all(round(a[0][0], p) == round(b[0][0], p) for a, b in zip(_x, _y))

        z = model.embed(['paris__france', 'vienna__austria'])
        print(z)


if __name__ == "__main__":
    unittest.main()

""" UnitTest """
import os
import unittest
import logging
import shutil
from torchglow import Glow, GlowFasttext, GlowBERT

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')


class Test(unittest.TestCase):
    """ Test """

    def test_image(self):
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

    def test_fasttext(self):
        if os.path.exists('tests/ckpt_we'):
            shutil.rmtree('tests/ckpt_we')
        model = GlowFasttext(
            lr=0.0001,
            training_step=5,
            epoch=2,
            export_dir='tests/ckpt_we',
            batch=2)
        model.train(epoch_valid=10)
        x, y = model.reconstruct(10)
        p = 3
        for n, (_x, _y) in enumerate(zip(x, y)):
            # reduce precision
            assert all(round(a[0][0], p) == round(b[0][0], p) for a, b in zip(_x, _y))

        z = model.embed(['paris__france', 'vienna__austria'])
        logging.info('\t latent vector from GlowFasttext: \t {}'.format(z))

    def test_bert(self):
        if os.path.exists('tests/ckpt_bert'):
            shutil.rmtree('tests/ckpt_bert')
        model = GlowBERT(
            lr=0.0001,
            training_step=5,
            epoch=2,
            export_dir='tests/ckpt_bert',
            batch=2)
        model.train(epoch_valid=10)
        x, y = model.reconstruct(10)
        p = 3
        for n, (_x, _y) in enumerate(zip(x, y)):
            # reduce precision
            assert all(round(a[0][0], p) == round(b[0][0], p) for a, b in zip(_x, _y))

        z = model.embed([['paris', 'france'], ['vienna, austria']])
        logging.info('\t latent vector from GlowBERT: \t {}'.format(z))


if __name__ == "__main__":
    unittest.main()

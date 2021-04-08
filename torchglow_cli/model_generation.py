""" Sampling image from GLOW's latent space """
import argparse
import logging
import os
from glob import glob

import torchglow

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main():
    # argument
    parser = argparse.ArgumentParser(description='Sampling image from latent space.')
    parser.add_argument('-b', '--batch', help='batch size', default=4, type=int)
    parser.add_argument('-n', '--n-image', help='number of image', default=12, type=int)
    parser.add_argument('-s', '--sample-size', help='sample size per image', default=16, type=int)
    parser.add_argument('--checkpoint-path', help='train existing checkpoint', default='./ckpt/celeba_128', type=str)
    parser.add_argument('-e', '--epoch', help='model epoch (the last epoch as default)', default=None, type=int)
    parser.add_argument('--export-dir', help='directory to export generated image (`./output/{ckpt}/` as default)',
                        default=None, type=str)
    parser.add_argument('--eps-std', help='std for sampling distribution', default=1, type=float)
    parser.add_argument('--random-seed', help='random seed', default=0, type=int)
    opt = parser.parse_args()

    # main
    torchglow.util.fix_seed(opt.random_seed)
    generate_sample(None, opt)


def generate_sample(epoch, opt):
    logging.info('loading model with epoch {}'.format(epoch))
    export_dir = opt.export_dir if opt.export_dir else './output/{}/'.format(os.path.basename(opt.checkpoint_path))
    os.makedirs(os.path.dirname(export_dir), exist_ok=True)

    model = torchglow.Glow(checkpoint_path=opt.checkpoint_path, epoch=epoch)

    for i in range(opt.n_image):
        logging.info('\t * generating image: {}/{}'.format(i + 1, opt.n_image))
        model.generate(
            sample_size=opt.sample_size,
            batch=opt.batch,
            export_path='{}/sample.{}.{}.{}.png'.format(export_dir, model.epoch_elapsed, i, opt.eps_std)
        )


if __name__ == '__main__':
    main()

""" Export latent space. """
import argparse
import logging
import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import umap
import torchglow

sns.set_theme(style="darkgrid")


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main():
    # argument
    parser = argparse.ArgumentParser(description='Display latent space')
    parser.add_argument('-b', '--batch', help='batch size', default=4, type=int)
    parser.add_argument('-s', '--sample-size', help='sample size per image', default=4000, type=int)
    parser.add_argument('-c', '--checkpoint-path', help='train existing checkpoint',
                        default='{}/ckpt/cifar10'.format(torchglow.util.module_output_dir), type=str)
    parser.add_argument('-e', '--epoch', help='model epoch (the last epoch as default)', default=None, type=int)
    parser.add_argument('--export-dir', help='directory to export generated image', default=None, type=str)
    parser.add_argument('--random-seed', help='random seed', default=0, type=int)
    parser.add_argument('--all-epoch', help='generate on all epochs', action='store_true')
    opt = parser.parse_args()

    # main

    if opt.all_epoch:
        for k in glob('{}/model.*.pt'.format(opt.checkpoint_path)):
            embed(int(k.split('model.')[-1].replace('.pt', '')), opt)
    else:
        embed(opt.epoch, opt)


def embed(epoch, opt):
    logging.info('loading model with epoch {}'.format(epoch))
    torchglow.util.fix_seed(opt.random_seed)

    if opt.export_dir is None:
        export_dir = '{}/2d_latent_space/{}'.format(
            torchglow.util.module_output_dir, os.path.basename(opt.checkpoint_path))
    else:
        export_dir = opt.export_dir
    os.makedirs(export_dir, exist_ok=True)
    model = torchglow.Glow(checkpoint_path=opt.checkpoint_path, checkpoint_epoch=epoch)
    # datasize x dimension
    embeddings, label = model.embed_data(sample_size=opt.sample_size, batch=opt.batch)
    logging.info('latent embedding consists of {} depth'.format(len(embeddings)))
    for n, embedding in enumerate(embeddings):
        plt.figure()
        export_image = '{}/latent.epoch_{}.depth_{}.png'.format(export_dir, epoch, n)
        embedding = np.array(embedding)
        reducer = umap.UMAP(random_state=opt.random_seed)
        reducer.fit(embedding)
        embedding = reducer.transform(embedding)

        plt.scatter(embedding[:, 0], embedding[:, 1], c=label, cmap='Spectral', s=5)
        plt.gca().set_aspect('equal', 'datalim')
        plt.colorbar(boundaries=np.arange(len(set(label)) + 1) - 0.5).set_ticks(np.arange(len(set(label))))
        plt.title('Latent space 2D projection (depth {}, epoch {:4})'.format(n, epoch), fontsize=16)
        # plt.tight_layout()
        plt.savefig(export_image)


if __name__ == '__main__':
    main()

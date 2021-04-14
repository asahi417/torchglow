""" Export latent space. """
import argparse
import logging
import os


import numpy as np
from sklearn.datasets import load_digits
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
    parser.add_argument('--checkpoint-path', help='train existing checkpoint', default='./ckpt/cifar10', type=str)
    parser.add_argument('-e', '--epoch', help='model epoch (the last epoch as default)', default=None, type=int)
    parser.add_argument('--export-dir', help='directory to export generated image (`./output/{ckpt}/` as default)',
                        default=None, type=str)
    parser.add_argument('--random-seed', help='random seed', default=0, type=int)
    opt = parser.parse_args()

    # main
    torchglow.util.fix_seed(opt.random_seed)
    embed(None, opt)


def embed(epoch, opt):
    logging.info('loading model with epoch {}'.format(epoch))
    export_dir = opt.export_dir if opt.export_dir else './output/{}'.format(os.path.basename(opt.checkpoint_path))
    os.makedirs(export_dir, exist_ok=True)
    export_image = '{}/latent.{}.png'.format(export_dir, epoch)

    model = torchglow.Glow(checkpoint_path=opt.checkpoint_path, epoch=epoch)
    # datasize x dimension
    embedding, label = model.embed_data(sample_size=opt.sample_size, batch=opt.batch)
    embedding = np.array(embedding)
    print(len(label))
    print(embedding.shape)
    input()
    reducer = umap.UMAP(random_state=opt.random_seed)
    reducer.fit(embedding)
    embedding = reducer.transform(embedding)

    plt.scatter(embedding[:, 0], embedding[:, 1], c=label, cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(len(set(label)) + 1) - 0.5).set_ticks(np.arange(len(set(label))))
    plt.title('UMAP 2D projection', fontsize=16)
    plt.savefig(export_image)


if __name__ == '__main__':
    main()

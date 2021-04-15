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
    parser.add_argument('-s', '--sample-size', help='sample size per image', default=16, type=int)
    parser.add_argument('-n', '--n-image', help='number of image', default=2, type=int)
    parser.add_argument('--nrow', help='number of row in image', default=8, type=int)
    parser.add_argument('-b', '--batch', help='batch size', default=4, type=int)
    parser.add_argument('-c', '--checkpoint-path', help='train existing checkpoint',
                        default='{}/ckpt/celba256'.format(torchglow.util.module_output_dir), type=str)
    parser.add_argument('-e', '--epoch', help='model epoch (the last epoch as default)', default=None, type=int)
    parser.add_argument('--export-dir', help='directory to export generated image (`./output/{ckpt}/` as default)',
                        default=None, type=str)
    parser.add_argument('--eps-std', help='std for sampling distribution', default=1, type=float)
    parser.add_argument('--random-seed', help='random seed', default=0, type=int)
    parser.add_argument('--all-epoch', help='generate on all epochs', action='store_true')
    opt = parser.parse_args()

    # main
    if opt.all_epoch:
        for k in glob('{}/model.*.pt'.format(opt.checkpoint_path)):
            generate_sample(int(k.split('model.')[-1].replace('.pt', '')), opt)
    else:
        generate_sample(opt.epoch, opt)


def generate_sample(epoch, opt):
    logging.info('loading model with epoch {}'.format(epoch))
    torchglow.util.fix_seed(opt.random_seed)
    if opt.export_dir is None:
        export_dir = '{}/generated_image/{}'.format(torchglow.util.module_output_dir, os.path.basename(opt.checkpoint_path))
    else:
        export_dir = opt.export_dir
    os.makedirs(export_dir, exist_ok=True)

    model = torchglow.Glow(checkpoint_path=opt.checkpoint_path, checkpoint_epoch=epoch)

    for i in range(opt.n_image):
        logging.info('\t * generating image: {}/{}'.format(i + 1, opt.n_image))
        model.generate(
            sample_size=opt.sample_size,
            batch=opt.batch,
            nrow=opt.nrow,
            eps_std=opt.eps_std,
            export_path='{}/sample.{}.{}.{}.png'.format(export_dir, model.checkpoint_epoch, i, opt.eps_std)
        )


if __name__ == '__main__':
    main()

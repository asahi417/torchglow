""" Train Glow model on images. """
import argparse
import logging
import torchglow


def get_options():
    parser = argparse.ArgumentParser(description='Train Glow model on images.')
    # image model specific parameters
    parser.add_argument('-d', '--data', help='dataset from `celeba`, `cifar10`', default='cifar10', type=str)
    parser.add_argument('--n-level', help='number of block', default=3, type=int)
    parser.add_argument('--n-bits-x', help='number of bits', default=8, type=int)
    # model parameter
    parser.add_argument('-s', '--training-step', help='training step in single epoch', default=50000, type=int)
    parser.add_argument('-e', '--epoch', help='training epochs', default=2000, type=int)
    parser.add_argument('--export-dir', help='directory to export model weight file', default='./ckpt', type=str)
    parser.add_argument('-b', '--batch', help='batch size', default=128, type=int)
    parser.add_argument('--lr', help='learning rate', default=0.001, type=float)
    parser.add_argument('--image-size', help='image size', default=32, type=int)
    parser.add_argument('--batch-init', help='number of batch for data_iterator dependent initialization', default=256, type=int)
    parser.add_argument('--filter-size', help='CNN filter size', default=512, type=int)
    parser.add_argument('--n-flow-step', help='number of flow in single block', default=32, type=int)
    parser.add_argument('--actnorm-scale', help='actnorm scaler', default=1, type=float)
    parser.add_argument('--lu-decomposition', help='LU decompose invertible CNN', action='store_true')
    parser.add_argument('--random-seed', help='random seed', default=0, type=int)
    parser.add_argument('--decay-lr', help='linear decay of learning rate after warmup', action='store_true')
    parser.add_argument("--epoch-warmup", help="warmup epochs", default=10, type=int)
    parser.add_argument("--weight-decay", help="l2 penalty for weight decay", default=0, type=float)
    parser.add_argument('--optimizer', help='optimizer `adam`/`adamax`/`adamw`', default='adamax', type=str)
    parser.add_argument("--momentum", help="sgd momentum", default=0.9, type=float)
    parser.add_argument('--unit-gaussian', help='unit gaussian instead of learnt gaussian', action='store_true')
    # optimization parameter
    parser.add_argument('--batch-valid', help='batch size for validation', default=64, type=int)
    parser.add_argument('--cache-dir', help='cache directory to store dataset', default=None, type=str)
    parser.add_argument('--num-workers', help='workers for dataloder', default=0, type=int)
    parser.add_argument('--fp16', help='fp16 for training', action='store_true')
    parser.add_argument('--progress-interval', help='log interval during training', default=100, type=int)
    parser.add_argument('--epoch-valid', help='interval to run validation', default=10, type=int)
    parser.add_argument('--epoch-save', help='interval to save model weight', default=1000, type=int)
    # load existing checkpoint
    parser.add_argument('--checkpoint-path', help='train existing checkpoint', default=None, type=str)
    # misc
    parser.add_argument('--debug', help='log level', action='store_true')
    return parser.parse_args()


def main():
    opt = get_options()
    level = logging.DEBUG if opt.debug else logging.INFO
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=level, datefmt='%Y-%m-%d %H:%M:%S')

    trainer = torchglow.Glow(
        training_step=opt.training_step,
        epoch=opt.epoch,
        data=opt.data,
        export_dir=opt.export_dir,
        batch=opt.batch,
        lr=opt.lr,
        image_size=opt.image_size,
        batch_init=opt.batch_init,
        filter_size=opt.filter_size,
        n_flow_step=opt.n_flow_step,
        n_level=opt.n_level,
        actnorm_scale=opt.actnorm_scale,
        lu_decomposition=opt.lu_decomposition,
        random_seed=opt.random_seed,
        n_bits_x=opt.n_bits_x,
        decay_lr=opt.decay_lr,
        epoch_warmup=opt.epoch_warmup,
        weight_decay=opt.weight_decay,
        optimizer=opt.optimizer,
        momentum=opt.momentum,
        checkpoint_path=opt.checkpoint_path,
        cache_dir=opt.cache_dir)
    trainer.train(
        batch_valid=opt.batch_valid,
        num_workers=opt.num_workers,
        fp16=opt.fp16,
        progress_interval=opt.progress_interval,
        epoch_valid=opt.epoch_valid,
        epoch_save=opt.epoch_save
    )


if __name__ == '__main__':
    main()

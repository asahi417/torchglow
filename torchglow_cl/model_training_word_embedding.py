""" Train Glow model on built-in dataset """
import argparse
import logging
import torchglow


def get_options():
    parser = argparse.ArgumentParser(description='Train Glow model on built-in dataset.')
    # model parameter
    parser.add_argument('-m', '--model-type', help='embedding model type (`relative`, `diff_fasttext`)',
                        default='relative', type=str)
    parser.add_argument('--validation-rate', help='validation set ratio', default=0.2, type=float)

    parser.add_argument('-s', '--training-step', help='training step in single epoch', default=950000, type=int)
    parser.add_argument('-e', '--epoch', help='training epochs', default=20, type=int)
    parser.add_argument('--export-dir', help='directory to export model weight file', default='./ckpt', type=str)
    parser.add_argument('-b', '--batch', help='batch size', default=1024, type=int)
    parser.add_argument('--lr', help='learning rate', default=0.0001, type=float)
    parser.add_argument('--batch-init', help='number of batch for data dependent initialization', default=256, type=int)
    parser.add_argument('--filter-size', help='CNN filter size', default=256, type=int)
    parser.add_argument('--n-flow-step', help='number of flow in single block', default=16, type=int)
    parser.add_argument('--actnorm-scale', help='actnorm scaler', default=1, type=float)
    parser.add_argument('--lu-decomposition', help='LU decompose invertible CNN', action='store_true')
    parser.add_argument('--random-seed', help='random seed', default=0, type=int)
    parser.add_argument('--decay-lr', help='linear decay of learning rate after warmup', action='store_true')
    parser.add_argument("--epoch-warmup", help="warmup epochs", default=2, type=int)
    parser.add_argument("--weight-decay", help="l2 penalty for weight decay", default=0, type=float)
    parser.add_argument('--optimizer', help='optimizer `adam`/`adamax`/`adamw`', default='adamax', type=str)
    parser.add_argument("--momentum", help="sgd momentum", default=0.9, type=float)
    # optimization parameter
    parser.add_argument('--batch-valid', help='batch size for validation', default=1024, type=int)
    parser.add_argument('--cache-dir', help='cache directory to store dataset', default=None, type=str)
    parser.add_argument('--num-workers', help='workers for dataloder', default=0, type=int)
    parser.add_argument('--fp16', help='fp16 for training', action='store_true')
    parser.add_argument('--progress-interval', help='log interval during training', default=100, type=int)
    parser.add_argument('--epoch-valid', help='interval to run validation', default=1, type=int)
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

    # train model
    trainer = torchglow.GlowWordEmbedding(
        model_type=opt.model_type,
        validation_rate=opt.validation_rate,
        training_step=opt.training_step,
        epoch=opt.epoch,
        export_dir=opt.export_dir,
        batch=opt.batch,
        lr=opt.lr,
        batch_init=opt.batch_init,
        filter_size=opt.filter_size,
        n_flow_step=opt.n_flow_step,
        actnorm_scale=opt.actnorm_scale,
        lu_decomposition=opt.lu_decomposition,
        random_seed=opt.random_seed,
        decay_lr=opt.decay_lr,
        epoch_warmup=opt.epoch_warmup,
        weight_decay=opt.weight_decay,
        optimizer=opt.optimizer,
        momentum=opt.momentum,
        checkpoint_path=opt.checkpoint_path
    )
    trainer.train(
        batch_valid=opt.batch_valid,
        cache_dir=opt.cache_dir,
        num_workers=opt.num_workers,
        fp16=opt.fp16,
        progress_interval=opt.progress_interval,
        epoch_valid=opt.epoch_valid,
        epoch_save=opt.epoch_save
    )


if __name__ == '__main__':
    main()

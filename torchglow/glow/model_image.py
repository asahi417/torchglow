""" Glow for 2D image data """
import logging
from math import log

import torch
from torch.utils.tensorboard import SummaryWriter

from .model_base import GlowBase
from .module import GlowNetwork
from ..config import Config
from ..data.data_image import get_dataset_image, get_image_decoder
from ..util import fix_seed

__all__ = 'Glow'


class Glow(GlowBase):
    """ Glow for 2D image data """

    def __init__(self,
                 training_step: int = 50000,
                 epoch: int = 1000000,
                 data: str = 'cifar10',
                 export_dir: str = './ckpt',
                 batch: int = 64,
                 lr: float = 0.001,
                 image_size: int = 32,
                 batch_init: int = 256,
                 filter_size: int = 512,
                 n_flow_step: int = 32,
                 n_level: int = 3,
                 actnorm_scale: float = 1.0,
                 lu_decomposition: bool = False,
                 random_seed: int = 0,
                 n_bits_x: int = 8,
                 decay_lr: bool = False,
                 epoch_warmup: int = 10,
                 weight_decay: float = 0,
                 optimizer: str = 'adamax',
                 momentum: float = 0.9,
                 checkpoint_path: str = None):
        """ Glow for 2D image data

        Parameters
        ----------
        training_step : int
            Training step in single epoch.
        epoch : int
            Number of epochs.
        data : str
            Dataset (`cifar10` or `celeba`).
        export_dir : str
            Directory to ecxport model weight file.
        batch : int
            The size of batch.
        lr : float
            Learning rate.
        image_size : int
            Image resolution.
        batch_init : int
            The number of batch for data-dependent initialization.
        filter_size : int
            CNN filter size.
        n_flow_step : int
            The number of flow.
        n_level : int
            The number of blocks.
        actnorm_scale : float
            Factor to scale ActNorm layer.
        lu_decomposition : bool
            LU decomposed invertible CNN
        random_seed : int
            Random seed.
        n_bits_x : int
            The number of bit.
        decay_lr : bool
            Linear decay of learning rate after warmup.
        epoch_warmup : int
            Epochs to linearly warmup learning rate.
        weight_decay : float
            Penalty for l2 weight decay.
        checkpoint_path : str
            Path to checkpoint to load trained weight.
        """
        super(Glow, self).__init__()
        fix_seed(random_seed)
        # config
        self.config = Config(
            checkpoint_path=checkpoint_path,
            lr=lr,
            decay_lr=decay_lr,
            n_bits_x=n_bits_x,
            epoch_warmup=epoch_warmup,
            image_size=image_size,
            training_step=training_step,
            epoch=epoch,
            batch=batch,
            data=data,
            export_dir=export_dir,
            filter_size=filter_size,
            n_flow_step=n_flow_step,
            n_level=n_level,
            actnorm_scale=actnorm_scale,
            lu_decomposition=lu_decomposition,
            random_seed=random_seed,
            weight_decay=weight_decay,
            optimizer=optimizer,
            batch_init=batch_init,
            momentum=momentum
        )
        # model
        self.model = GlowNetwork(
            image_shape=[self.config.image_size, self.config.image_size, 3],
            filter_size=self.config.filter_size,
            n_flow_step=self.config.n_flow_step,
            n_level=self.config.n_level,
            actnorm_scale=self.config.actnorm_scale,
            lu_decomposition=self.config.lu_decomposition
        )
        # model size
        model_size = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # for p in self.model.parameters():
        #     if p.requires_grad:
        #         print(p.numel(), p.shape)
        # input(model_size)
        logging.info('Glow model: {}M parameters'.format(round(model_size/10**6, 4)))

        if self.config.is_trained:
            logging.info('loading weight from {}'.format(self.config.cache_dir))
            self.model.load_state_dict(torch.load(self.config.model_weight_path))

        # model on gpu
        self.model.to(self.device)
        logging.info('running on {} GPUs'.format(self.n_gpu))

        self.checkpoint_dir = self.config.cache_dir

    def setup_data(self, cache_dir):
        return get_dataset_image(
            self.config.data, cache_dir=cache_dir, n_bits_x=self.config.n_bits_x, image_size=self.config.image_size)

    def reconstruct(self, sample_size: int = 5, cache_dir: str = None, batch: int = 5):
        """ Reconstruct validation image by Glow """
        assert self.config.is_trained, 'model is not trained'
        _, data_valid = get_dataset_image(
            self.config.data, cache_dir=cache_dir, n_bits_x=self.config.n_bits_x, image_size=self.config.image_size)
        decoder = get_image_decoder(n_bits_x=self.config.n_bits_x)
        loader = torch.utils.data.DataLoader(data_valid, batch_size=batch)
        image_original = []
        image_reconstruct = []
        with torch.no_grad():
            for x, _ in loader:
                z, _ = self.model(x.to(self.device), return_loss=False)
                y, _ = self.model(latent_states=z, reverse=True, return_loss=False)
                image_original += decoder(x)

                image_reconstruct += decoder(y)
                if len(image_original) > sample_size:
                    break
        return image_original[:sample_size], image_reconstruct[:sample_size]

    def train_single_epoch(self, data_loader, epoch_n: int, writer, progress_interval):
        self.model.train()
        n_bins = 2 ** self.config.n_bits_x
        step_in_epoch = int(round(self.config.training_step / self.config.batch))
        data_loader = iter(data_loader)
        total_bpd = 0
        data_size = 0
        for i in range(step_in_epoch):
            try:
                x, _ = next(data_loader)
            except StopIteration:
                break
            x = x.to(self.device)
            # https://github.com/openai/glow/issues/43
            x = x + torch.rand_like(x) / n_bins
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward: output prediction and get loss
            _, nll = self.model(x)
            bpd = (nll + log(n_bins)) / log(2)

            # backward: calculate gradient
            self.scaler.scale(bpd.mean()).backward()

            # bits per dimension
            inst_bpd = bpd.mean().cpu().item()
            writer.add_scalar('train/bits_per_dim', inst_bpd, i + epoch_n * step_in_epoch)

            # update optimizer
            inst_lr = self.optimizer.param_groups[0]['lr']
            writer.add_scalar('train/learning_rate', inst_lr, i + epoch_n * step_in_epoch)

            if i % progress_interval == 0:
                logging.debug('[epoch {}/{}] (step {}/{}) instant bpd: {}: lr: {}'.format(
                    epoch_n, self.config.epoch, i, step_in_epoch, round(inst_bpd, 3), inst_lr))

            # aggregate average bpd over epoch
            total_bpd += bpd.sum().cpu().item()
            data_size += len(x)

            self.scaler.step(self.optimizer)
            self.scaler.update()

        return total_bpd / data_size

    def valid_single_epoch(self, data_loader, epoch_n: int, writer):
        self.model.eval()
        n_bins = 2 ** self.config.n_bits_x
        total_bpd = 0
        data_size = 0
        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(self.device)
                # https://github.com/openai/glow/issues/43
                x = x + torch.rand_like(x) / n_bins
                # forward: output prediction and get loss
                _, nll = self.model(x)
                bpd = (nll + log(n_bins)) / log(2)
                total_bpd += bpd.sum().cpu().item()
                data_size += len(x)

        # bits per dimension
        bpd = total_bpd / data_size
        writer.add_scalar('valid/bits_per_dim', bpd, epoch_n)
        return bpd

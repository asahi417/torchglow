""" Glow for 2D image data_iterator """
import logging

import torch
import torchvision

from .model_base import GlowBase
from .module import GlowNetwork
from ..config import Config
from ..data_iterator.data_image import get_dataset_image, get_image_decoder
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
                 checkpoint_path: str = None,
                 unit_gaussian: bool = False,
                 additive_coupling: bool = False,
                 cache_dir: str = None,
                 checkpoint_epoch: int = None):
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
        additive_coupling : bool
            Additive coupling instead of affine coupling.
        """
        super(Glow, self).__init__()
        fix_seed(random_seed)
        self.cache_dir = cache_dir
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
            momentum=momentum,
            unit_gaussian=unit_gaussian,
            additive_coupling=additive_coupling
        )
        # model
        self.model = GlowNetwork(
            image_shape=[self.config.image_size, self.config.image_size, 3],
            filter_size=self.config.filter_size,
            n_flow_step=self.config.n_flow_step,
            n_level=self.config.n_level,
            actnorm_scale=self.config.actnorm_scale,
            lu_decomposition=self.config.lu_decomposition,
            unit_gaussian=self.config.unit_gaussian,
            additive_coupling=self.config.additive_coupling
        )

        # model size
        model_size = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info('{}M trainable parameters'.format(round(model_size/10**6, 4)))

        self.checkpoint_dir = self.config.cache_dir
        self.checkpoint_epoch = None
        if self.config.is_trained:
            logging.info('loading weight from {}'.format(self.config.cache_dir))
            if checkpoint_epoch is not None:
                self.checkpoint_epoch = checkpoint_epoch
                model_weight_path = self.config.path_model[self.checkpoint_epoch]
            else:
                # use the longest trained model
                self.checkpoint_epoch = sorted(list(self.config.path_model.keys()))[-1]
                model_weight_path = self.config.path_model[self.checkpoint_epoch]
            self.model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))

        # for multi GPUs
        self.parallel = False
        if torch.cuda.device_count() > 1:
            self.parallel = True
            self.model = torch.nn.DataParallel(self.model)

        # model on gpu
        self.model.to(self.device)
        logging.info('Glow running on {} GPUs'.format(self.n_gpu))

        self.n_bins = 2 ** self.config.n_bits_x

    def setup_data(self):
        """ Initialize training dataset. """
        return get_dataset_image(
            self.config.data, cache_dir=self.cache_dir, n_bits_x=self.config.n_bits_x,
            image_size=self.config.image_size)

    def generate(self, sample_size: int = 16, batch: int = 4, export_path: str = 'glow_generate_image.png',
                 eps_std: float = 1):
        """ Generate image from trained GLOW by sampling from learnt latent embedding. """
        assert self.config.is_trained
        self.model.eval()
        decoder = get_image_decoder(n_bits_x=self.config.n_bits_x)
        images = []
        with torch.no_grad():
            for _ in range(batch):
                y, _ = self.model(sample_size=sample_size, reverse=True, return_loss=False, eps_std=eps_std)
                images.append(y)
        images = torch.cat(images)
        images = decoder(images, keep_tensor=True)
        torchvision.utils.save_image(images, export_path, normalize=True)
        logging.info('Glow generated image saved at {}'.format(export_path))
        return images

    def reconstruct(self, sample_size: int = 5, batch: int = 5):
        decoder = get_image_decoder(n_bits_x=self.config.n_bits_x)
        return self.reconstruct_base(sample_size, batch, decoder)

""" Main network for Glow """
import logging
from math import log
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .module import GlowNetwork
from ..config import Config
from ..data.data import get_dataset, get_decoder
from ..util import fix_seed, get_linear_schedule_with_warmup


class Glow(nn.Module):
    """ Main network for Glow """

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
        """ Main network for Glow

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
        super().__init__()
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
        logging.info('Glow model: {}M parameters'.format(round(model_size/10**6)))

        # a few initialization related to optimization
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.n_gpu = torch.cuda.device_count()
        self.device = 'cuda' if self.n_gpu > 0 else 'cpu'

        if self.config.is_trained:
            logging.info('loading weight from {}'.format(self.config.cache_dir))
            self.model.load_state_dict(torch.load(self.config.model_weight_path))

        # model on gpu
        self.model.to(self.device)
        logging.info('running on {} GPUs'.format(self.n_gpu))

        self.checkpoint_dir = self.config.cache_dir

    def __setup_optimizer(self, fp16):
        # optimizer
        if self.config.optimizer == 'adamax':
            self.optimizer = torch.optim.Adamax(
                self.model.parameters(), lr=self.config.lr)
        elif self.config.optimizer == 'adam':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.config.lr)
        elif self.config.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.config.lr, momentum=self.config.momentum)
        elif self.config.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        else:
            raise ValueError('unknown optimizer: {}'.format(self.config.optimizer))
        # scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.epoch_warmup,
            num_training_steps=self.config.epoch if self.config.decay_lr else None)
        # load from exsiting config
        if self.config.is_trained:
            optimizer_stat = torch.load(self.config.optimizer_path, map_location='cpu')  # allocate stats on cpu
            self.optimizer.load_state_dict(optimizer_stat['optimizer_state_dict'])
            self.scheduler.load_state_dict(optimizer_stat['scheduler_state_dict'])

        # GPU mixture precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=fp16)
        # multi-gpus
        if self.n_gpu > 1:
            # multi-gpu training (should be after apex fp16 initialization)
            self.model = torch.nn.DataParallel(self.model)
            logging.info('using `torch.nn.DataParallel`')

    def reconstruct(self, sample_size: int = 5, cache_dir: str = None, batch: int = 5):
        """ Reconstruct validation image by Glow """
        assert self.config.is_trained, 'model is not trained'
        _, data_valid = get_dataset(
            self.config.data, cache_dir=cache_dir, n_bits_x=self.config.n_bits_x, image_size=self.config.image_size)
        decoder = get_decoder(n_bits_x=self.config.n_bits_x)
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

    def train(self,
              batch_valid: int = 32,
              cache_dir: str = None,
              num_workers: int = 0,
              fp16: bool = False,
              progress_interval: int = 100,
              epoch_valid: int = 100,
              epoch_save: int = 100000):
        """ Train Glow model

        Parameters
        ----------
        batch_valid : int
            The size of batch for validation.
        cache_dir : int
            Directory for cache the datasets.
        num_workers : int
            Workers for DataLoader.
        fp16 : bool
            Mixed precision to save memory.
        progress_interval : int
            Interval to log loss during training.
        epoch_valid : int
            Epoch to run validation eg) Every 100 epoch, it will run validation as default.
        epoch_save : int
            Epoch to run validation eg) Every 100000 epoch, it will save model weight as default.
        """
        # assert not self.config.is_trained, 'model has already been trained'
        assert not self.config.is_fully_trained, 'model was fully trained over all epochs'
        batch_valid = self.config.batch if batch_valid is None else batch_valid
        writer = SummaryWriter(log_dir=self.config.cache_dir)

        logging.debug('setting up optimizer')
        self.__setup_optimizer(fp16=fp16)

        logging.debug('loading data iterator')
        data_train, data_valid = get_dataset(
            self.config.data, cache_dir=cache_dir, n_bits_x=self.config.n_bits_x, image_size=self.config.image_size)
        if self.config.epoch_elapsed == 0:
            logging.debug('data-dependent initialization')
            loader = torch.utils.data.DataLoader(
                data_train, batch_size=self.config.batch_init, shuffle=True)
            self.__data_dependent_initialization(loader)

        logging.info('start model training')
        loader = torch.utils.data.DataLoader(
            data_train, batch_size=self.config.batch, shuffle=True, num_workers=num_workers)
        loader_valid = torch.utils.data.DataLoader(
            data_valid, batch_size=batch_valid, shuffle=False, num_workers=num_workers)

        try:
            with torch.cuda.amp.autocast(enabled=fp16):
                for e in range(self.config.epoch_elapsed, self.config.epoch):  # loop over the epoch

                    mean_bpd = self.__train_single_epoch(
                        loader, epoch_n=e, progress_interval=progress_interval, writer=writer)
                    inst_lr = self.optimizer.param_groups[0]['lr']
                    logging.info('[epoch {}/{}] average bpd: {}: lr {}'.format(
                        e, self.config.epoch, round(mean_bpd, 3), inst_lr))

                    if e % epoch_valid == 0 and e != 0:
                        logging.debug('running validation')
                        mean_bpd = self.__valid_single_epoch(loader_valid, epoch_n=e, writer=writer)
                        logging.info('[epoch {}/{}] average bpd: {} (valid)'.format(
                            e, self.config.epoch, round(mean_bpd, 3)))

                    if e % epoch_save == 0 and e != 0:
                        self.config.save(self.model.state_dict(), epoch=e)

                    self.scheduler.step()

        except KeyboardInterrupt:
            logging.info('*** KeyboardInterrupt ***')

        writer.close()
        self.config.save(
            self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=self.scheduler.state_dict(),
            epoch=e,
            last_model=True)
        logging.info('complete training: model ckpt was saved at {}'.format(self.config.cache_dir))

    def __data_dependent_initialization(self, data_loader):
        with torch.no_grad():
            loader = iter(data_loader)
            x, _ = next(loader)
            x = x.to(self.device)
            self.model(x, return_loss=False, initialize_actnorm=True)

    def __train_single_epoch(self, data_loader, epoch_n: int, writer, progress_interval):
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

    def __valid_single_epoch(self, data_loader, epoch_n: int, writer):
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


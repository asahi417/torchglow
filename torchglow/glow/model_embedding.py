import logging
from typing import List

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from .module import FlowStep, Split
from ..util import fix_seed, get_linear_schedule_with_warmup
from ..config import Config


class GlowNetwork1D(nn.Module):
    """ Glow network architecture for 1d input """

    def __init__(self,
                 n_channel: int,
                 filter_size: int = 512,
                 n_flow_step: int = 10,
                 actnorm_scale: float = 1.0,
                 lu_decomposition: bool = False):
        """ Glow network architecture

        Parameters
        ----------
        n_channel: List
            Input channel size.
        filter_size : int
            Filter size for CNN layer.
        n_flow_step : int
            Number of flow block.
        actnorm_scale : float
            Factor to scale ActNorm.
        lu_decomposition : bool
            Whether use LU decomposition in invertible CNN layer.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_flow_step = n_flow_step
        flow_config = {'filter_size': filter_size, 'kernel_size': 1, 'stride': 1,
                       'actnorm_scale': actnorm_scale, 'lu_decomposition': lu_decomposition}
        self.n_channel = n_channel
        for _ in range(self.n_flow_step):
            self.layers.append(FlowStep(in_channels=self.n_channel, **flow_config))
        self.layers.append(Split(in_channels=self.n_channel, split=False))
        self.last_latent_shape = [self.n_channel, 1, 1]

    def forward(self,
                x: torch.Tensor = None,
                initialize_actnorm: bool = False,
                sample_size: int = 1,
                return_loss: bool = True,
                reverse: bool = False,
                latent_states: List = None,
                eps_std: float = None):
        """ Glow forward inference/reverse sampling module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, which can be left as None to generate random sample from learnt posterior
            for reverse mode.
        reverse : bool
            Switch to reverse mode to sample data from latent variable.
        latent_states: List
            Latent variable to generate data, which is None as default for random sampling, otherwise
            generate data conditioned by the given latent variable. This should be the output from the forward
            inference.
        sample_size : int
            Sampling size for random generation in reverse mode.
        return_loss : bool
            Switch to not computing log-det mode (should be True in non-training usecases).
        eps_std : bool
            Factor to scale sampling distribution.

        Returns
        -------
        output :
            (forward mode) List containing latent variables of x
            (reverse mode) torch.Tensor of the generated data
        nll : negative log likelihood
        """
        if reverse:
            if x is None:
                # seed variables for sampling data
                x = torch.zeros([sample_size] + self.last_latent_shape)

            for layer in reversed(self.layers):
                if isinstance(layer, Split):
                    x, _ = layer(x, reverse=True, eps_std=eps_std, z=latent_states)
                else:
                    x, _ = layer(x, reverse=True)
            return x, None
        else:
            assert x is not None, '`x` have to be a tensor, not None'
            log_det = 0 if return_loss else None
            for layer in self.layers:
                if isinstance(layer, FlowStep):
                    x, log_det = layer(x, log_det=log_det, initialize_actnorm=initialize_actnorm)
                else:
                    x, log_det = layer(x, log_det=log_det)
            (_, z) = x
            nll = None if log_det is None else - log_det
            return z, nll



class GlowWordEmbedding(nn.Module):
    """ Glow on Word Embeddings """

    def __init__(self,
                 path_to_data: str,
                 path_to_data_valid: str = None,
                 training_step: int = 500,
                 epoch: int = 1000,
                 export_dir: str = './ckpt',
                 batch: int = 64,
                 lr: float = 0.001,
                 n_channel: int = 1024,
                 batch_init: int = 256,
                 filter_size: int = 512,
                 n_flow_step: int = 32,
                 actnorm_scale: float = 1.0,
                 lu_decomposition: bool = False,
                 random_seed: int = 0,
                 decay_lr: bool = False,
                 epoch_warmup: int = 10,
                 weight_decay: float = 0,
                 optimizer: str = 'adamax',
                 momentum: float = 0.9,
                 checkpoint_path: str = None):
        super().__init__()
        fix_seed(random_seed)
        # config
        self.config = Config(
            path_to_data=path_to_data,
            path_to_data_valid=path_to_data_valid,
            n_channel=n_channel,
            checkpoint_path=checkpoint_path,
            lr=lr,
            decay_lr=decay_lr,
            epoch_warmup=epoch_warmup,
            training_step=training_step,
            epoch=epoch,
            batch=batch,
            export_dir=export_dir,
            filter_size=filter_size,
            n_flow_step=n_flow_step,
            actnorm_scale=actnorm_scale,
            lu_decomposition=lu_decomposition,
            random_seed=random_seed,
            weight_decay=weight_decay,
            optimizer=optimizer,
            batch_init=batch_init,
            momentum=momentum
        )
        # model
        self.model = GlowNetwork1D(
            n_channel=self.config.n_channel,
            filter_size=self.config.filter_size,
            n_flow_step=self.config.n_flow_step,
            actnorm_scale=self.config.actnorm_scale,
            lu_decomposition=self.config.lu_decomposition
        )
        # model size
        model_size = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info('1D Glow model: {}M parameters'.format(round(model_size/10**6)))

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
            # assert self.config.weight_decay == 0, ''
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
        # GPU mixture precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=fp16)
        # multi-gpus
        if self.n_gpu > 1:
            # multi-gpu training (should be after apex fp16 initialization)
            self.model = torch.nn.DataParallel(self.model)
            logging.info('using `torch.nn.DataParallel`')

    def train(self,
              batch_valid: int = 32,
              cache_dir: str = None,
              num_workers: int = 0,
              gradient_checkpointing: bool = False,
              fp16: bool = False,
              progress_interval: int = 100,
              epoch_valid: int = 100,
              epoch_save: int = 1000):
        """ Train 1D Glow model

        Parameters
        ----------
        batch_valid : int
            The size of batch for validation.
        cache_dir : int
            Directory for cache the datasets.
        num_workers : int
            Workers for DataLoader.
        gradient_checkpointing : bool
            Gradient checkpointing to save memory.
        fp16 : bool
            Mixed precision to save memory.
        progress_interval : int
            Interval to log loss during training.
        epoch_valid : int
            Epoch to run validation eg) Every 100 epoch, it will run validation as default.
        epoch_save : int
            Epoch to run validation eg) Every 100000 epoch, it will save model weight as default.
        """
        assert not self.config.is_trained, 'model has already been trained'
        writer = SummaryWriter(log_dir=self.config.cache_dir)

        logging.debug('setting up optimizer')
        self.__setup_optimizer(fp16=fp16)

        logging.debug('loading data iterator')
        data = get_dataset_1d(self.config.path_to_data, cache_dir=cache_dir)

        logging.debug('data-dependent initialization')
        loader = torch.utils.data.DataLoader(data, batch_size=self.config.batch_init, shuffle=True)
        self.__data_dependent_initialization(loader)

        logging.info('start model training')
        loader = torch.utils.data.DataLoader(data, batch_size=self.config.batch, shuffle=True, num_workers=num_workers)
        if self.config.path_to_data_valid is not None:
            data_valid = get_dataset_1d(self.config.path_to_data_valid, cache_dir=cache_dir)
            loader_valid = torch.utils.data.DataLoader(data_valid, batch_size=batch_valid, num_workers=num_workers)
        else:
            loader_valid = None

        try:
            with torch.cuda.amp.autocast(enabled=fp16):
                for e in range(self.config.epoch):  # loop over the epoch

                    mean_nll = self.__train_single_epoch(
                        loader, epoch_n=e, progress_interval=progress_interval, writer=writer,
                        gradient_checkpointing=gradient_checkpointing)
                    inst_lr = self.optimizer.param_groups[0]['lr']
                    logging.info('[epoch {}/{}] average nll: {}: lr {}'.format(
                        e, self.config.epoch, round(mean_nll, 3), inst_lr))

                    if e % epoch_valid == 0 and e != 0 and loader_valid is not None:
                        logging.debug('running validation')
                        mean_nll = self.__valid_single_epoch(loader_valid, epoch_n=e, writer=writer)
                        logging.info('[epoch {}/{}] average nll: {} (valid)'.format(
                            e, self.config.epoch, round(mean_nll, 3)))

                    if e % epoch_save == 0 and e != 0:
                        self.config.save(self.model.state_dict(), epoch=e)

                    self.scheduler.step()

        except KeyboardInterrupt:
            logging.info('*** KeyboardInterrupt ***')

        writer.close()
        self.config.save(self.model.state_dict())
        logging.info('complete training: model ckpt was saved at {}'.format(self.config.cache_dir))

    def __data_dependent_initialization(self, data_loader):
        with torch.no_grad():
            loader = iter(data_loader)
            self.model(next(loader).to(self.device), return_loss=False, initialize_actnorm=True)

    def __train_single_epoch(self, data_loader, epoch_n: int, writer, progress_interval, gradient_checkpointing):
        if gradient_checkpointing:
            raise NotImplementedError('gradient_checkpointing is not implemented yet')

        self.model.train()
        step_in_epoch = int(round(self.config.training_step / self.config.batch))
        data_loader = iter(data_loader)
        total_nll = 0
        data_size = 0
        for i in range(step_in_epoch):
            try:
                x = next(data_loader)
            except StopIteration:
                break
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward: output prediction and get loss
            _, nll = self.model(x.to(self.device))
            # backward: calculate gradient
            self.scaler.scale(nll.mean()).backward()

            inst_nll = nll.mean().cpu().item()
            writer.add_scalar('train/nll', inst_nll, i + epoch_n * step_in_epoch)

            # update optimizer
            inst_lr = self.optimizer.param_groups[0]['lr']
            writer.add_scalar('train/learning_rate', inst_lr, i + epoch_n * step_in_epoch)

            if i % progress_interval == 0:
                logging.debug('[epoch {}/{}] (step {}/{}) instant nll: {}: lr: {}'.format(
                    epoch_n, self.config.epoch, i, step_in_epoch, round(inst_nll, 3), inst_lr))

            # aggregate average nll over epoch
            total_nll += nll.sum().cpu().item()
            data_size += len(x)

            self.scaler.step(self.optimizer)
            self.scaler.update()

        return total_nll / data_size

    def __valid_single_epoch(self, data_loader, epoch_n: int, writer):
        self.model.eval()
        total_nll = 0
        data_size = 0
        with torch.no_grad():
            for x in data_loader:
                _, nll = self.model(x.to(self.device))
                total_nll += nll.sum().cpu().item()
                data_size += len(x)

        # bits per dimension
        bpd = total_nll / data_size
        writer.add_scalar('valid/bits_per_dim', bpd, epoch_n)
        return bpd


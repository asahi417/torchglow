""" Base Glow Class """
import logging
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from ..util import get_linear_schedule_with_warmup

__all__ = 'GlowBase'


class GlowBase(nn.Module):
    """ Base Glow Class """

    def __init__(self):
        super().__init__()
        # a few initialization related to optimization
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.n_gpu = torch.cuda.device_count()
        self.device = 'cuda' if self.n_gpu > 0 else 'cpu'

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
        self.setup_optimizer(fp16=fp16)

        logging.debug('loading data iterator')
        data_train, data_valid = self.setup_data(cache_dir)
        if self.config.epoch_elapsed == 0:
            logging.debug('data-dependent initialization')
            loader = torch.utils.data.DataLoader(
                data_train, batch_size=self.config.batch_init, shuffle=True)
            self.data_dependent_initialization(loader)

        logging.info('start model training')
        loader = torch.utils.data.DataLoader(
            data_train, batch_size=self.config.batch, shuffle=True, num_workers=num_workers)
        loader_valid = torch.utils.data.DataLoader(
            data_valid, batch_size=batch_valid, shuffle=False, num_workers=num_workers)

        try:
            with torch.cuda.amp.autocast(enabled=fp16):
                for e in range(self.config.epoch_elapsed, self.config.epoch):  # loop over the epoch

                    mean_bpd = self.train_single_epoch(
                        loader, epoch_n=e, progress_interval=progress_interval, writer=writer)
                    inst_lr = self.optimizer.param_groups[0]['lr']
                    logging.info('[epoch {}/{}] average bpd: {}: lr {}'.format(
                        e, self.config.epoch, round(mean_bpd, 3), inst_lr))

                    if e % epoch_valid == 0 and e != 0:
                        logging.debug('running validation')
                        mean_bpd = self.valid_single_epoch(loader_valid, epoch_n=e, writer=writer)
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

    def setup_optimizer(self, fp16):
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
        # load from existing config
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

    def data_dependent_initialization(self, data_loader):
        with torch.no_grad():
            loader = iter(data_loader)
            data = next(loader)
            x = data[0].to(self.device)
            self.model(x, return_loss=False, initialize_actnorm=True)

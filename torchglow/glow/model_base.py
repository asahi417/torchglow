""" Base Glow Class mainly for model training """
import logging
from math import log

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from ..util import get_linear_schedule_with_warmup
from ..data_iterator.data_image import get_image_decoder

__all__ = 'GlowBase'


class GlowBase(nn.Module):
    """ Base Glow Class """

    def __init__(self):
        super().__init__()
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.model = None
        self.n_bins = None
        self.epoch_elapsed = None
        self.parallel = False

    def train(self,
              batch_valid: int = 32,
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
        assert self.epoch_elapsed is None or not self.epoch_elapsed < self.config.epoch,\
            'model has been trained over all epochs already.'
        batch_valid = self.config.batch if batch_valid is None else batch_valid
        writer = SummaryWriter(log_dir=self.config.cache_dir)

        logging.debug('setting up optimizer')
        self.setup_optimizer(fp16=fp16)

        logging.debug('loading data iterator')
        data_train, data_valid = self.setup_data()
        if self.epoch_elapsed == 0:
            logging.debug('data-dependent initialization')
            loader = torch.utils.data.DataLoader(
                data_train, batch_size=self.config.batch_init, shuffle=True)
            self.data_dependent_initialization(loader)

        logging.info('start model training')

        assert self.config.batch <= len(data_train)
        loader = torch.utils.data.DataLoader(
            data_train, batch_size=self.config.batch, shuffle=True, num_workers=num_workers, drop_last=True)
        logging.info('\t * train data: {}, batch number: {}'.format(len(data_train), len(loader)))

        if data_valid is not None:
            loader_valid = torch.utils.data.DataLoader(
                data_valid, batch_size=batch_valid, shuffle=False, num_workers=num_workers)
            logging.info('\t * valid data: {}, batch number: {}'.format(len(data_valid), len(loader_valid)))
        else:
            loader_valid = None

        with torch.cuda.amp.autocast(enabled=fp16):
            for e in range(self.epoch_elapsed, self.config.epoch):  # loop over the epoch

                mean_bpd = self.train_single_epoch(
                    loader, epoch_n=e, progress_interval=progress_interval, writer=writer)
                inst_lr = self.optimizer.param_groups[0]['lr']
                logging.info('[epoch {}/{}] average bpd: {}, lr: {}'.format(
                    e, self.config.epoch, round(mean_bpd, 3), inst_lr))

                if e % epoch_valid == 0 and e != 0 and loader_valid is not None:
                    logging.debug('running validation')
                    mean_bpd = self.valid_single_epoch(loader_valid, epoch_n=e, writer=writer)
                    logging.info('[epoch {}/{}] average bpd: {} (valid)'.format(
                        e, self.config.epoch, round(mean_bpd, 3)))

                if e % epoch_save == 0 and e != 0:
                    if self.parallel:
                        state = self.model.module.state_dict()
                    else:
                        state = self.model.state_dict()
                    self.config.save(state,
                                     optimizer_state_dict=self.optimizer.state_dict(),
                                     scheduler_state_dict=self.scheduler.state_dict(),
                                     epoch=e)

                self.scheduler.step()

        writer.close()
        if self.parallel:
            state = self.model.module.state_dict()
        else:
            state = self.model.state_dict()
        self.config.save(
            state,
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=self.scheduler.state_dict(),
            epoch=e+1)
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
            optimizer_path = self.config.path_optimizer[self.checkpoint_epoch]
            optimizer_stat = torch.load(optimizer_path, map_location=torch.device('cpu'))
            self.optimizer.load_state_dict(optimizer_stat['optimizer_state_dict'])
            self.scheduler.load_state_dict(optimizer_stat['scheduler_state_dict'])
            self.epoch_elapsed = int(optimizer_stat['epoch_elapsed']) + 1
        else:
            self.epoch_elapsed = 0
        # GPU mixture precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=fp16)

    def data_dependent_initialization(self, data_loader):
        with torch.no_grad():
            loader = iter(data_loader)
            x = next(loader)
            if type(x) is not torch.Tensor:
                x = x[0]
            x = x.to(self.device)
            self.model(x, return_loss=False, initialize_actnorm=True)

    def train_single_epoch(self, data_loader, epoch_n: int, writer, progress_interval):
        self.model.train()
        total_bpd = 0
        data_size = 0
        if self.config.training_step is not None:
            step_in_epoch = min(len(data_loader), self.config.training_step)
        else:
            step_in_epoch = len(data_loader)
        for i, x in enumerate(data_loader):
            if i >= step_in_epoch:
                break
            if type(x) is not torch.Tensor:
                x = x[0]
            x = x.to(self.device)
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward: output prediction and get loss
            if self.n_bins is not None:  # for training on image data
                # forward: output prediction and get loss, https://github.com/openai/glow/issues/43
                _, nll = self.model(x + torch.rand_like(x) / self.n_bins)
                bpd = (nll + log(self.n_bins)) / log(2)
            else:  # for training on other data
                _, bpd = self.model(x)

            # backward: calculate gradient
            self.scaler.scale(bpd.mean()).backward()

            inst_bpd = bpd.mean().cpu().item()
            writer.add_scalar('train/bpd', inst_bpd, i + epoch_n * step_in_epoch)

            # update optimizer
            inst_lr = self.optimizer.param_groups[0]['lr']
            writer.add_scalar('train/learning_rate', inst_lr, i + epoch_n * step_in_epoch)

            if i % progress_interval == 0:
                logging.debug('[epoch {}/{}] (step {}/{}) instant bpd: {}: lr: {}'.format(
                    epoch_n, self.config.epoch, i + 1, step_in_epoch, round(inst_bpd, 3), inst_lr))

            # aggregate average bpd over epoch
            total_bpd += bpd.sum().cpu().item()
            data_size += len(x)

            self.scaler.step(self.optimizer)
            self.scaler.update()

        return total_bpd / data_size

    def valid_single_epoch(self, data_loader, epoch_n: int, writer):
        self.model.eval()
        total_bpd = 0
        data_size = 0
        with torch.no_grad():
            for x in data_loader:
                if type(x) is not torch.Tensor:
                    x = x[0]
                x = x.to(self.device)
                # forward: output prediction and get loss
                if self.n_bins is not None:
                    # forward: output prediction and get loss, https://github.com/openai/glow/issues/43
                    _, nll = self.model(x + torch.rand_like(x) / self.n_bins)
                    bpd = (nll + log(self.n_bins)) / log(2)
                else:  # for training on other data
                    _, bpd = self.model(x)

                total_bpd += bpd.sum().cpu().item()
                data_size += len(x)

        # bits per dimension
        bpd = total_bpd / data_size
        writer.add_scalar('valid/bpd', bpd, epoch_n)
        return bpd

    def reconstruct(self, sample_size: int = 5, batch: int = 5):
        """ Reconstruct validation data_iterator """
        decoder = get_image_decoder(n_bits_x=self.config.n_bits_x)
        assert self.config.is_trained, 'model is not trained'
        data_train, data_valid = self.setup_data()
        if data_valid is None:
            data_valid = data_train
        loader = torch.utils.data.DataLoader(data_valid, batch_size=batch)
        data_original = []
        data_reconstruct = []
        with torch.no_grad():
            for x in loader:
                if type(x) is not torch.Tensor:
                    x = x[0]
                x = x.to(self.device)
                z, _ = self.model(x, return_loss=False)
                y, _ = self.model(latent_states=z, reverse=True, return_loss=False)
                if decoder is not None:
                    data_original += decoder(x)
                    data_reconstruct += decoder(y)
                else:
                    data_original += x.cpu().tolist()
                    data_reconstruct += y.cpu().tolist()
                if len(data_original) > sample_size:
                    break
        return data_original[:sample_size], data_reconstruct[:sample_size]


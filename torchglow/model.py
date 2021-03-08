import logging
from math import log
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.checkpoint import checkpoint

from .module import GlowNetwork
from .config import Config
from .data import get_dataset, get_decoder
from .util import fix_seed, get_linear_schedule_with_warmup


class Glow(nn.Module):

    def __init__(self,
                 training_step: int = 10,
                 epoch: int = 10,
                 data: str = 'cifar10',
                 export_dir: str = None,
                 batch: int = 8,
                 lr: float = 0.0001,
                 image_size: int = 32,
                 n_batch_init: int = 2,
                 filter_size: int = 512,
                 n_flow_step: int = 32,
                 n_level: int = 3,
                 actnorm_scale: float = 1.0,
                 lu_decomposition: bool = True,
                 random_seed: int = 1234,
                 n_bits_x: int = 8,
                 optimizer: str = 'adam',
                 decay_lr: bool = True,
                 epoch_warmup: int = 10,
                 weight_decay: float = 0,
                 checkpoint_path: str = None):
        super().__init__()
        fix_seed(random_seed)
        # config
        self.config = Config(
            checkpoint_path=checkpoint_path,
            lr=lr,
            decay_lr=decay_lr,
            n_bits_x=n_bits_x,
            optimizer=optimizer,
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
            n_batch_init=n_batch_init
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
            logging.info('loading weight from '.format(self.config.cache_dir))
            self.model.load_state_dict(torch.load(self.config.model_weight_path))

        # model on gpu
        self.model.to(self.device)
        logging.info('running on {} GPUs'.format(self.n_gpu))

        self.checkpoint_dir = self.config.cache_dir

    def __setup_optimizer(self, fp16):
        # optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        # scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.epoch_warmup,
            num_training_steps=self.config.epoch if self.config.decay_lr else None)
        # GPU mixture precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=fp16)
        # if fp16:
        #     try:
        #         from apex import amp  # noqa: F401
        #         self.model, self.optimizer = amp.initialize(
        #             self.model, self.optimizer, opt_level='O1', max_loss_scale=2 ** 13, min_loss_scale=1e-5)
        #         self.master_params = amp.master_params
        #         self.scale_loss = amp.scale_loss
        #         logging.debug('using `apex.amp`')
        #     except ImportError:
        #         ImportError("Skip apex: please install apex from https://www.github.com/nvidia/apex to use fp16")
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

    # def sample(self, sample_size: int = 5, batch: int = 5):
    #     """ Sapmle from learnt posterior """
    #     assert self.config.is_trained, 'model is not trained'
    #     _, data_valid, decoder = get_dataset(
    #         self.config.data, cache_dir=cache_dir, n_bits_x=self.config.n_bits_x, image_size=self.config.image_size)
    #     loader = torch.utils.data.DataLoader(data_valid, batch_size=batch)
    #     image_original = []
    #     image_reconstruct = []
    #     with torch.no_grad():
    #         for x, _ in loader:
    #             z, _ = self.model(x.to(self.device), return_loss=False)
    #             y, _ = self.model(latent_states=z, reverse=True, return_loss=False)
    #             image_original += decoder(x)
    #
    #             image_reconstruct += decoder(y)
    #             if len(image_original) > sample_size:
    #                 break
    #     return image_original[:sample_size], image_reconstruct[:sample_size]

    def train(self,
              batch_valid: int = None,
              cache_dir: str = None,
              num_workers: int = 0,
              gradient_checkpointing: bool = True,
              fp16: bool = False,
              progress_interval: int = 100,
              epoch_validation: int = 5):
        assert not self.config.is_trained, 'model has already been trained'
        batch_valid = self.config.batch if batch_valid is None else batch_valid
        writer = SummaryWriter(log_dir=self.config.cache_dir)

        logging.debug('setting up optimizer')
        self.__setup_optimizer(fp16=fp16)

        logging.debug('loading data iterator')
        data_train, data_valid = get_dataset(
            self.config.data, cache_dir=cache_dir, n_bits_x=self.config.n_bits_x, image_size=self.config.image_size)

        logging.debug('data-dependent initialization')
        loader = torch.utils.data.DataLoader(
            data_train, batch_size=self.config.batch * self.config.n_batch_init)
        self.__data_dependent_initialization(loader)

        logging.info('start model training')
        loader = torch.utils.data.DataLoader(
            data_train, batch_size=self.config.batch, shuffle=True, num_workers=num_workers)
        loader_valid = torch.utils.data.DataLoader(
            data_valid, batch_size=batch_valid, shuffle=False, num_workers=num_workers)

        try:
            with torch.cuda.amp.autocast(enabled=fp16):
                for e in range(self.config.epoch):  # loop over the epoch

                    mean_bpd = self.__train_single_epoch(
                        loader, epoch_n=e, progress_interval=progress_interval, writer=writer,
                        gradient_checkpointing=gradient_checkpointing)
                    logging.info('[epoch {}/{}] average bpd: {}'.format(
                        e, self.config.epoch, round(mean_bpd, 2)))

                    if e % epoch_validation == 0 and e != 0:
                        logging.debug('running validation')
                        mean_bpd = self.__valid_single_epoch(loader_valid, epoch_n=e, writer=writer)
                        logging.info('[epoch {}/{}] average bpd: {} (valid)'.format(
                            e, self.config.epoch, round(mean_bpd, 2)))
                        self.config.save(self.model.state_dict(), epoch=e)

                    self.scheduler.step()

        except KeyboardInterrupt:
            logging.info('*** KeyboardInterrupt ***')

        writer.close()
        self.config.save(self.model.state_dict())
        logging.info('complete training: model ckpt was saved at {}'.format(self.config.cache_dir))

    def __data_dependent_initialization(self, data_loader):
        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(self.device)
                self.model(x, return_loss=False)
                break

    def __train_single_epoch(self, data_loader, epoch_n: int, writer, progress_interval, gradient_checkpointing):
        self.model.train()
        n_bins = 2 ** self.config.n_bits_x
        if self.config.training_step is not None:
            step_in_epoch = int(round(self.config.training_step / self.config.batch))
        else:
            step_in_epoch = len(data_loader)

        total_bpd = 0.0
        data_size = 0
        for i, (x, _) in enumerate(data_loader, 1):

            if i > step_in_epoch:
                break
            x = x.to(self.device)

            # forward: output prediction and get loss
            if gradient_checkpointing:
                raise NotImplementedError('TBA')
                # _, nll = checkpoint(self.model, x)
            else:
                _, nll = self.model(x)
            nll = nll.sum()

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # backward: calculate gradient
            self.scaler.scale(nll).backward()

            # bits per dimension
            bpd = log(n_bins) / log(2) - nll.cpu().item()
            total_bpd += bpd
            data_size += len(x)
            writer.add_scalar('train/bits_per_dim', bpd/len(x), i + epoch_n * len(data_loader))

            # update optimizer
            inst_lr = self.optimizer.param_groups[0]['lr']
            writer.add_scalar('train/learning_rate', inst_lr, i + epoch_n * len(data_loader))
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if i % progress_interval == 0:
                logging.info('[epoch {}/{}] (step {}/{}) instant bpd: {}: lr: {}'.format(
                    epoch_n, self.config.epoch, i, len(data_loader), round(bpd/len(x), 2), inst_lr))

        mean_bpe = total_bpd / data_size
        return mean_bpe

    def __valid_single_epoch(self, data_loader, epoch_n: int, writer):
        self.model.eval()
        n_bins = 2 ** self.config.n_bits_x
        total_bpd = 0
        data_size = 0
        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(self.device)
                # forward: output prediction and get loss
                _, nll = self.model(x)
                nll = nll.sum()
                # bits per dimension
                total_bpd += log(n_bins) / log(2) - nll.cpu().item()
                data_size += len(x)
        mean_bpe = total_bpd / data_size
        writer.add_scalar('valid/bits_per_dim', mean_bpe, epoch_n)
        return mean_bpe




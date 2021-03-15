""" Glow on 1D Word Embeddings """
import logging
import torch

from .model_base import GlowBase
from .module import GlowNetwork1D
from ..config import Config
from ..data.data_word_embedding import get_dataset_word_embedding, N_DIM
from ..util import fix_seed

__all__ = 'GlowWordEmbedding'


class GlowWordEmbedding(GlowBase):
    """ Glow on 1D Word Embeddings """

    def __init__(self,
                 model_type: str = 'relative',
                 validation_rate: float = 0.2,
                 training_step: int = 500,
                 epoch: int = 1000,
                 export_dir: str = './ckpt',
                 batch: int = 64,
                 lr: float = 0.001,
                 batch_init: int = 256,
                 filter_size: int = 256,
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
        """ Glow on 1D Word Embeddings

        Parameters
        ----------
        model_type : str
            Word embedding model type ('relative', 'diff_fasttext').
        validation_rate : float
            Ratio of validation set.
        training_step : int
            Training step in single epoch.
        epoch : int
            Number of epochs.
        export_dir : str
            Directory to ecxport model weight file.
        batch : int
            The size of batch.
        lr : float
            Learning rate.
        batch_init : int
            The number of batch for data-dependent initialization.
        filter_size : int
            CNN filter size.
        n_flow_step : int
            The number of flow.
        actnorm_scale : float
            Factor to scale ActNorm layer.
        lu_decomposition : bool
            LU decomposed invertible CNN
        random_seed : int
            Random seed.
        decay_lr : bool
            Linear decay of learning rate after warmup.
        epoch_warmup : int
            Epochs to linearly warmup learning rate.
        weight_decay : float
            Penalty for l2 weight decay.
        checkpoint_path : str
            Path to checkpoint to load trained weight.
        """
        super(GlowWordEmbedding, self).__init__()
        fix_seed(random_seed)
        # config
        self.config = Config(
            model_type=model_type,
            validation_rate=validation_rate,
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
            n_channel=N_DIM[self.config.model_type],
            filter_size=self.config.filter_size,
            n_flow_step=self.config.n_flow_step,
            actnorm_scale=self.config.actnorm_scale,
            lu_decomposition=self.config.lu_decomposition
        )
        # model size
        model_size = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info('1D Glow model: {}M parameters'.format(round(model_size/10**6)))

        if self.config.is_trained:
            logging.info('loading weight from {}'.format(self.config.cache_dir))
            self.model.load_state_dict(torch.load(self.config.model_weight_path))

        # model on gpu
        self.model.to(self.device)
        logging.info('running on {} GPUs'.format(self.n_gpu))

        self.checkpoint_dir = self.config.cache_dir

    def setup_data(self, cache_dir):
        return get_dataset_word_embedding(
            self.config.model_type, validation_rate=self.config.validation_rate, cache_dir=cache_dir)

    def train_single_epoch(self, data_loader, epoch_n: int, writer, progress_interval):
        self.model.train()
        step_in_epoch = int(round(self.config.training_step / self.config.batch))
        data_loader = iter(data_loader)
        total_nll = 0
        data_size = 0
        for i in range(step_in_epoch):
            try:
                data = next(data_loader)
            except StopIteration:
                break
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward: output prediction and get loss
            _, nll = self.model(data[0].to(self.device))
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
            data_size += len(data[0])

            self.scaler.step(self.optimizer)
            self.scaler.update()

        return total_nll / data_size

    def valid_single_epoch(self, data_loader, epoch_n: int, writer):
        self.model.eval()
        total_nll = 0
        data_size = 0
        with torch.no_grad():
            for data in data_loader:
                _, nll = self.model(data[0].to(self.device))
                total_nll += nll.sum().cpu().item()
                data_size += len(data[0])

        # bits per dimension
        bpd = total_nll / data_size
        writer.add_scalar('valid/nll', bpd, epoch_n)
        return bpd

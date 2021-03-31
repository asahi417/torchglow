""" Glow on 1D Word Embeddings """
import os
import logging
from typing import Dict, List
import torch

from .model_base import GlowBase
from .module import GlowNetwork1D
from ..config import Config
from ..data_iterator.data_word_pairs import get_dataset_word_pairs, get_iterator_bert
from ..util import fix_seed

__all__ = 'GlowBERT'


class GlowBERT(GlowBase):
    """ Glow on BERT embeddings """

    def __init__(self,
                 lm_model: str = 'roberta-large',
                 lm_max_length: int = 32,
                 lm_embedding_layers: List = -1,
                 data: str = 'common_word_pairs',
                 validation_rate: float = 0.2,
                 training_step: int = None,
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
                 unit_gaussian: bool = True,
                 additive_coupling: bool = False,
                 checkpoint_path: str = None,
                 cache_dir: str = None,
                 checkpoint_epoch: int = None):
        """ Glow on BERT Embeddings

        Parameters
        ----------
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
            The number of batch for bpd-dependent initialization.
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
        additive_coupling : bool
            Additive coupling instead of affine coupling.
        """
        super(GlowBERT, self).__init__()
        fix_seed(random_seed)
        self.cache_dir = cache_dir
        # config
        self.config = Config(
            lm_model=lm_model,
            lm_max_length=lm_max_length,
            lm_embedding_layers=lm_embedding_layers,
            data=data,
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
            momentum=momentum,
            unit_gaussian=unit_gaussian,
            additive_coupling=additive_coupling
        )
        # get preprocessing module
        if data == 'common_word_pairs':
            mode = 'mask'
        else:
            raise ValueError('unknown dataset: {}'.format(data))
        # self.data_iterator to encode input, and self.converter to embed the encoded input to BERT embedding
        (self.data_iterator, self.converter), self.hidden_size = get_iterator_bert(
            lm_model, lm_max_length, lm_embedding_layers, mode=mode)
        # model
        self.model = GlowNetwork1D(
            n_channel=self.hidden_size,
            filter_size=self.config.filter_size,
            n_flow_step=self.config.n_flow_step,
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

        self.data_format = 'bert'

    def setup_data(self):
        """ Initialize training dataset. """
        return get_dataset_word_pairs(self.data_iterator,  validation_rate=self.config.validation_rate)

    def reconstruct(self, sample_size: int = 5, batch: int = 5):
        return self.reconstruct_base(sample_size, batch)

    def embed(self, data: List, batch: int = None, return_original_embedding: bool = False):
        assert self.config.is_trained, 'model is not trained'
        self.model.eval()
        latent_variable, bert_embedding = self.embed_base(data, batch)
        if return_original_embedding:
            return latent_variable, bert_embedding
        return latent_variable

""" Get embedding from transformers language model. """
import os
import logging
from typing import List
from multiprocessing import Pool

import transformers
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning message
__all__ = 'BERT'


class EncodePlus:
    """ Wrapper of encode_plus for multiprocessing. """

    def __init__(self, tokenizer, max_length: int, prefix: str = None, mode: str = 'relative'):
        self.prefix = prefix
        self.tokenizer = tokenizer
        self.max_length = self.tokenizer.model_max_length
        self.mode = mode
        if max_length is not None:
            assert self.max_length >= max_length, '{} < {}'.format(self.max_length, max_length)
            self.max_length = max_length

    def __call__(self, x):
        """ Encoding a word pair or sentence. """
        if self.mode == 'mask':
            assert len(x) == 2, 'word_pair contains wrong number of tokens: {}'.format(len(x))
            h, t = x
            sentence = ' '.join([h] + [self.tokenizer.mask_token] + [t])
            if self.prefix:
                sentence = self.prefix + ' ' + sentence
        elif self.mode == 'cls':
            assert type(x) is str, x
            sentence = x
        else:
            raise ValueError('unknown mode: {}'.format(self.mode))
        param = {'max_length': self.max_length, 'truncation': True, 'padding': 'max_length'}
        encode = self.tokenizer.encode_plus(sentence, **param)
        assert encode['input_ids'][-1] == self.tokenizer.pad_token_id, 'exceeded max_length'
        if self.mode == 'mask':
            encode['mask_position'] = encode['input_ids'].index(self.tokenizer.mask_token_id)
        elif self.mode == 'cls':
            encode['mask_position'] = 0
        return encode


class Dataset(torch.utils.data.Dataset):
    """ `torch.utils.data_iterator.Dataset` """
    float_tensors = ['attention_mask']

    def __init__(self, data: List):
        self.data = data  # a list of dictionaries

    def __len__(self):
        return len(self.data)

    def to_tensor(self, name, data):
        if name in self.float_tensors:
            return torch.tensor(data, dtype=torch.float32)
        return torch.tensor(data, dtype=torch.long)

    def __getitem__(self, idx):
        return {k: self.to_tensor(k, v) for k, v in self.data[idx].items()}


class BERT:
    """ Get embedding from transformers language model. """


    def __init__(self,
                 model: str,
                 max_length: int = 32,
                 cache_dir: str = None,
                 embedding_layers: List = -1,
                 mode: str = 'cls'):
        """ Get embedding from transformers language model.

        Parameters
        ----------
        model : str
            Transformers model alias.
        max_length : int
            Model length.
        cache_dir : str
        embedding_layers : int
            Embedding layers to get average.
        mode : str
            - `mask` to get the embedding for a word pair by [MASK] token, eg) (A, B) -> A [MASK] B
            - `cls` to get the embedding on the [CLS] token
        """
        assert 'bert' in model, '{} is not BERT'.format(model)
        self.model_name = model
        self.cache_dir = cache_dir
        self.mode = mode
        self.embedding_layers = [embedding_layers] if type(embedding_layers) is not list else embedding_layers
        self.config = transformers.AutoConfig.from_pretrained(
            self.model_name, cache_dir=self.cache_dir, output_hidden_states=True)
        self.hidden_size = self.config.hidden_size
        self.num_hidden_layers = self.config.num_hidden_layers
        self.max_length = max_length

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.model = transformers.AutoModelForMaskedLM.from_pretrained(
            self.model_name, config=self.config, cache_dir=self.cache_dir)
        self.model.eval()
        # GPU setup
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.model.to(self.device)
        logging.info('BERT running on {} GPU'.format(torch.cuda.device_count()))

    def preprocess(self, x: List, parallel: bool = True):
        """ Preprocess textual data.

        Parameters
        ----------
        x : list
            List of word pairs (`relative` mode) or sentences (`cls` mode).
        parallel : bool
            Parallelize data processing part over CPUs.

        Returns
        -------
        """
        if self.mode == 'mask':
            x = [x] if type(x[0]) is str else x
        elif self.mode == 'cls':
            x = [x] if type(x) is str else x
        logging.debug('{} data to encode'.format(len(x)))
        if parallel:
            pool = Pool()
            data = pool.map(EncodePlus(self.tokenizer, self.max_length, mode=self.mode), x)
            pool.close()
        else:
            data = list(map(EncodePlus(self.tokenizer, self.max_length, mode=self.mode), x))
        return Dataset(data)

    def to_embedding(self, encode, return_tensor: bool = True):
        """ Compute embedding from batch of encode. """
        with torch.no_grad():
            encode = {k: v.to(self.device) for k, v in encode.items()}
            mask_position = encode.pop('mask_position').cpu().tolist()
            output = self.model(**encode, return_dict=True)
            # hidden state of masked token: layer x batch x length x h_n
            hidden_states = [output['hidden_states'][h] for h in self.embedding_layers]
            # get average over the specified layer: batch x length x h_n
            hidden_states = sum(hidden_states) / len(hidden_states)
            embedding = list(map(lambda y: y[0][y[1]], zip(hidden_states.cpu().tolist(), mask_position)))
            if return_tensor:
                return torch.tensor(embedding)
            return embedding

    def get_embedding(self, x: List, batch_size: int = None, num_worker: int = 0, parallel: bool = True):
        """ Get embedding from BERT.

        Parameters
        ----------
        x : list
            List of word pairs (`relative` mode) or sentences (`cls` mode).
        batch_size : int
            Batch size.
        num_worker : int
            Dataset worker number.
        parallel : boo;
            Parallelize data processing part over CPUs.

        Returns
        -------
        Embedding (len(word_pairs), n_hidden).
        """
        data = self.preprocess(x, parallel=parallel)
        batch_size = len(x) if batch_size is None else batch_size
        data_loader = torch.utils.data.DataLoader(
            data, num_workers=num_worker, batch_size=batch_size, shuffle=False, drop_last=False)

        logging.debug('\t* run LM inference')
        h_list = []
        for encode in data_loader:
            h_list += self.to_embedding(encode, return_tensor=False)
        return h_list

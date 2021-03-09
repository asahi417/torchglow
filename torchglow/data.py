""" Data iterators: `celba` and `cifar10`. """
import os
import struct
import logging
import requests
import tarfile
import zipfile
from glob import glob

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from tfrecord.torch.dataset import TFRecordDataset

# The original processed data used in Glow paper
URLS = {'celeba': 'https://openaipublic.azureedge.net/glow-demo/data/celeba-tfr.tar'}
CACHE_DIR = '{}/.cache/torchglow'.format(os.path.expanduser('~'))
__all__ = ('get_dataset', 'get_decoder')


def wget(url, cache_dir):
    filename = os.path.basename(url)
    dst = '{}/{}'.format(cache_dir, filename)
    if os.path.exists(dst):
        logging.debug('found file at `{}`, skip `wget`'.format(dst))
    else:
        with open(dst, "wb") as f:
            r = requests.get(url)
            f.write(r.content)
    return dst


def open_compressed_file(url, cache_dir):
    path = wget(url, cache_dir)
    if path.endswith('.tar') or path.endswith('.tar.gz') or path.endswith('.tgz'):
        tar = tarfile.open(path)
        tar.extractall(cache_dir)
        tar.close()
    elif path.endswith('.zip'):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)


def create_index(tfrecord_dir: str):
    """ Create index from the directory of tfrecords files.
    Stores starting location (byte) and length (in bytes) of each serialized record.

    Params:
    -------
    tfrecord_dir: str
        Path to the TFRecord file.
    """
    tfr_files = glob('{}/*.tfrecords'.format(tfrecord_dir))
    assert len(tfr_files),  'no tfrecords found at {}'.format(tfr_files)
    logging.debug('creating index from tfrecords: {} files will be processed'.format(len(tfr_files)))
    for n, tfrecord_file in enumerate(tfr_files):
        index_file = tfrecord_file.replace('.tfrecords', '.index')
        if os.path.exists(index_file):
            continue
        logging.debug('\t * process {}/{}: {}'.format(n, len(tfr_files), tfrecord_file))
        infile = open(tfrecord_file, "rb")
        outfile = open(index_file, "w")

        while True:
            current = infile.tell()
            byte_len = infile.read(8)
            if len(byte_len) == 0:
                break
            infile.read(4)
            proto_len = struct.unpack("q", byte_len)[0]
            infile.read(proto_len)
            infile.read(4)
            outfile.write(str(current) + " " + str(infile.tell() - current) + "\n")
        infile.close()
        outfile.close()
    logging.debug('finish indexing')


class Dataset(torch.utils.data.Dataset):
    """ CelebA data iterator """

    def __init__(self,
                 tfrecord_dir: str,
                 root: str,
                 train: bool,
                 transform,
                 n_bits_x: int = 5,
                 data: str = 'celeba'):
        self.root = root
        self.train = train
        self.transform = transform
        self.n_bits_x = n_bits_x

        # download celeba tfrecord files
        if not os.path.exists(tfrecord_dir):
            open_compressed_file(URLS[data], root)
        self.tfr_files = sorted(glob('{}/*.tfrecords'.format(tfrecord_dir)))

        # create unique index
        create_index(tfrecord_dir)

        # create global index, id: (tfrecord file, index in the tfr file)
        self.data_index = {}
        n = 0
        for i in self.tfr_files:
            with open(i.replace('tfrecords', 'index'), 'r') as f:
                data_size = len(list(filter(len, f.read().split('\n'))))
            for i_ in range(data_size):
                self.data_index[n] = (i, i_)
                n += 1

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        tfr_file, n = self.data_index[idx]
        # load tfrecord
        dataset = TFRecordDataset(tfr_file, tfr_file.replace('tfrecords', 'index'))
        single_data = list(dataset)[n]
        img = single_data['data'].reshape(single_data['shape']).astype('float32')
        # normalize image to [0, 1]
        img = (img / 2 ** (8 - self.n_bits_x)).round() / (2. ** self.n_bits_x)
        # apply transformation
        img = self.transform(img)
        return img, single_data['label'][0]


class AddNoise(object):
    """ Add uniform noise to image. """

    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + (torch.rand(tensor.size()) - 0.5 + self.mean) * self.std

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_dataset(data: str, cache_dir: str = None, n_bits_x: int = 8, image_size: int = None):
    """ Get dataset iterator.

    Parameters
    ----------
    data : str
        Dataset either of `celeba` or `cifar10`.
    cache_dir : str
        (optional) Root directory to store cached data files.
    n_bits_x : int
        (optional) Number of bits of image.
    image_size : int
        (optional) Image size to rescale.

    Returns
    -------
    train_set : torch.utils.data.Dataset
        Iterator of training set for torch.utils.data.DataLoader.
    valid_set : torch.utils.data.Dataset
        Iterator of validation set for torch.utils.data.DataLoader.
    """
    cache_dir = CACHE_DIR if cache_dir is None else cache_dir
    n_bins = 2 ** n_bits_x

    t_valid = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [1, 1, 1])])

    t_train = []
    if image_size:
        t_train.append(transforms.Resize(image_size))
    t_train += [
        transforms.ToTensor(),  # convert to float tensor scaled in [0, 1], and reshape HWC to CHW
        transforms.Normalize([0.5, 0.5, 0.5], [1, 1, 1]),  # centering pixel
    ]

    if data == 'cifar10':
        assert n_bits_x == 8, 'cifar10 does not support n_bits_x != 8'

        # t_train.append(transforms.RandomAffine(degrees=0, translate=(.1, .1)))  # add random shift
        # t_train.append(AddNoise(0, 1 / n_bins))  # add random noise to pixel
        t_train = transforms.Compose(t_train)

        train_set = torchvision.datasets.CIFAR10(root=cache_dir, train=True, download=True, transform=t_train)
        valid_set = torchvision.datasets.CIFAR10(root=cache_dir, train=False, download=True, transform=t_valid)
    elif data == 'celeba':
        t_train.append(AddNoise(0, 1 / n_bins))  # add random noise to pixel
        t_train = transforms.Compose(t_train)
        train_set = Dataset(
            '{}/celeba-tfr/train'.format(cache_dir), root=cache_dir, train=True, transform=t_train)
        valid_set = Dataset(
            '{}/celeba-tfr/validation'.format(cache_dir), root=cache_dir, train=False, transform=t_valid)
    else:
        raise ValueError('unknown data: {}'.format(data))
    return train_set, valid_set


def get_decoder(n_bits_x: int = 8):
    """ Get tensor decoder to get image. """
    n_bins = 2 ** n_bits_x

    def convert_tensor_to_img(v, pil: bool = True):
        """ decoder to recover image from tensor """

        def single_img(v_):
            v_ = v_.transpose(1, 2, 0)
            img = (((v_ + .5) * n_bins).round(0) * (256 / n_bins)).clip(0, 255).astype('uint8')
            if pil:
                img = Image.fromarray(img, 'RGB')
            return img

        if type(v) is torch.Tensor:
            v = v.cpu().numpy()
        assert v.ndim in [3, 4], v.shape
        if v.ndim == 3:
            return single_img(v)
        return [single_img(_v) for _v in v]

    return convert_tensor_to_img

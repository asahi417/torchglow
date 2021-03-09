""" Modules for Glow """
from typing import List
from math import log
from scipy import linalg
import torch
import numpy as np
from torch import nn
from torch.nn.functional import conv2d

__all__ = 'GlowNetwork'
EPS = 1e-5


def pixels(tensor: torch.Tensor):
    return int(tensor.size(2) * tensor.size(3))


def mean_tensor(tensor: torch.Tensor, dim: List = None, keepdim: bool = False):
    """ Take the mean along multiple dimensions. """
    if dim is None:
        # mean all dim
        return torch.mean(tensor)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(dim=d, keepdim=True)
        if not keepdim:
            for i, d in enumerate(dim):
                tensor.squeeze_(d - i)
        return tensor


class ActNorm2d(nn.Module):
    """ Activation normalization for 2D inputs.

    The bias and scale get initialized using the mean and variance of the
    first mini-batch. After the init, bias and scale are trainable parameters.
    """

    def __init__(self, num_features, scale: float = 1.):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.num_features = num_features
        self.scale = float(scale)

    def initialize_parameters(self, x):
        bias = - mean_tensor(x.clone(), dim=[0, 2, 3], keepdim=True)
        v = mean_tensor((x.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
        logs = (self.scale / (v.sqrt() + EPS)).log()
        self.bias.data.copy_(bias.data)
        self.logs.data.copy_(logs.data)

    def forward(self, x, log_det=None, reverse: bool = False, initialize: bool = False):
        if initialize:
            self.initialize_parameters(x)
        self._check_input_dim(x)

        if reverse:  # scale and center
            x, log_det = self._scale(x, log_det, reverse)
            x = self._center(x, reverse)
        else:  # center and scale
            x = self._center(x, reverse)
            x, log_det = self._scale(x, log_det, reverse)
        return x, log_det

    def _center(self, x, reverse: bool = False):
        flag = -1 if reverse else 1
        return x + self.bias * flag

    def _scale(self, x, log_det=None, reverse: bool = False):
        flag = -1 if reverse else 1
        x = x * torch.exp(flag * self.logs)
        if log_det is not None:
            log_det = log_det + self.logs.sum() * pixels(x) * flag
        return x, log_det

    def _check_input_dim(self, x):
        assert len(x.size()) == 4, x.size()
        assert x.size(1) == self.num_features, (
            "[ActNorm]: x should be in shape as `BCHW`, channels should be {} rather than {}".format(
                self.num_features, x.size()))


class InvertibleConv2d(nn.Module):
    """ Invertible 1x1 Convolution for 2D inputs. """

    def __init__(self, num_channels: int, lu_decomposition: bool = False):
        super().__init__()
        self.w_shape = [num_channels, num_channels]
        # sample a random orthogonal matrix
        w_init = np.linalg.qr(np.random.randn(*self.w_shape))[0].astype(np.float32)

        if lu_decomposition:
            lu_p, lu_l, lu_u = linalg.lu(w_init)

            # trainable parameters
            lu_u = np.triu(lu_u, k=1)
            self.lu_u = nn.Parameter(torch.Tensor(lu_u.astype(np.float32)))
            lu_s = np.diag(lu_u)
            log_s = np.log(np.abs(lu_s) + EPS)
            self.log_s = nn.Parameter(torch.Tensor(log_s.astype(np.float32)))
            self.lu_l = nn.Parameter(torch.Tensor(lu_l.astype(np.float32)))

            # fixed parameters
            sign_s = np.sign(lu_s)
            self.register_buffer('lu_p', torch.Tensor(lu_p.astype(np.float32)))
            self.register_buffer('eye', torch.Tensor(np.eye(*self.w_shape, dtype=np.float32)))
            self.register_buffer('sign_s', torch.Tensor(sign_s.astype(np.float32)))
            self.register_buffer('l_mask', torch.Tensor(np.tril(np.ones(self.w_shape, dtype=np.float32), -1)))
        else:
            self.weight = nn.Parameter(torch.Tensor(w_init))
            self.log_s = self.lu_u = self.lu_l = self.lu_p = self.eye = self.sign_s = self.l_mask = None
        self.lu_decomposition = lu_decomposition

    def forward(self, x, log_det=None, reverse: bool = False):
        flag = -1 if reverse else 1

        if self.lu_decomposition:
            lu_l = self.lu_l * self.l_mask + self.eye
            lu_u = self.lu_u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
            if log_det is not None:
                log_det = log_det + flag * self.log_s.sum() * pixels(x)
            if reverse:
                lu_l = torch.inverse(lu_l.double()).float()
                lu_u = torch.inverse(lu_u.double()).float()
                weight = torch.matmul(lu_u, torch.matmul(lu_l, self.p.inverse()))
            else:
                weight = torch.matmul(self.lu_p, torch.matmul(lu_l, lu_u))
            weight = weight.view(self.w_shape[0], self.w_shape[1], 1, 1)
        else:
            if log_det is not None:
                # log_det = log|abs(|W|)| * pixels
                log_det = log_det + flag * torch.slogdet(self.weight)[1] * pixels(x)
            if reverse:
                weight = torch.inverse(self.weight.double()).float().view(self.w_shape[0], self.w_shape[1], 1, 1)
            else:
                weight = self.weight.view(self.w_shape[0], self.w_shape[1], 1, 1)
        z = conv2d(x, weight)
        return z, log_det


class ZeroConv2d(nn.Conv2d):
    """ Zero initialized convolution for 2D """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 log_scale_factor: float = 3):
        padding = tuple([((kernel_size - 1) * stride + 1) // 2] * 2)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding=padding)

        # log_scale_factor
        self.log_scale_factor = log_scale_factor
        self.logs = nn.Parameter(torch.zeros(out_channels, 1, 1))

        # init
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, x):
        output = super().forward(x)
        return output * torch.exp(self.logs * self.log_scale_factor)


class AffineCoupling(nn.Module):
    """ Affine coupling layer """

    def __init__(self,
                 in_channels: int,
                 filter_size: int = 512,
                 kernel_size: int = 3,
                 stride: int = 1):
        assert in_channels % 2 == 0, in_channels
        super().__init__()

        # affine sub-network: take the half of the input channel and produce feature sized full channel
        kernel_size_mid = 1  # bottle neck layer
        self.net = nn.Sequential(
            nn.Conv2d(int(in_channels / 2), filter_size, kernel_size, stride,
                      padding=tuple([((kernel_size - 1) * stride + 1) // 2] * 2)),
            nn.ReLU(),
            nn.Conv2d(filter_size, filter_size, kernel_size_mid, stride,
                      padding=tuple([((kernel_size_mid - 1) * stride + 1) // 2] * 2)),
            nn.ReLU(),
            ZeroConv2d(filter_size, in_channels, kernel_size, stride),
        )
        # initialization
        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()
        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

        self.scale = nn.Parameter(torch.ones(in_channels, 1, 1))

    def forward(self, x, log_det=None, reverse: bool = False):
        x_a, x_b = x.chunk(2, 1)
        log_s, t = self.net(x_b).chunk(2, 1)
        s = torch.sigmoid(log_s + 2)
        if reverse:
            y_a = x_a / s - t
        else:
            y_a = (x_a + t) * s
            if log_det is not None:
                log_det = log_det + torch.sum(torch.log(s).view(x.shape[0], -1), 1)
        return torch.cat([y_a, x_b], 1), log_det


class FlowStep(nn.Module):
    """ One step of flow: input -> ActNorm -> Invertible Conv -> Affine Coupling -> output """

    def __init__(self,
                 in_channels: int,
                 filter_size: int = 512,
                 kernel_size: int = 3,
                 stride: int = 1,
                 actnorm_scale: float = 1.0,
                 lu_decomposition: bool = False):
        super().__init__()

        self.actnorm = ActNorm2d(in_channels, actnorm_scale)
        self.invconv = InvertibleConv2d(in_channels, lu_decomposition)
        self.coupling = AffineCoupling(
            in_channels, filter_size=filter_size, kernel_size=kernel_size, stride=stride)

    def forward(self, x, log_det=None, reverse: bool = False, initialize_actnorm: bool = False):
        if reverse:
            x, log_det = self.coupling(x, log_det, reverse=True)
            x, log_det = self.invconv(x, log_det, reverse=True)
            x, log_det = self.actnorm(x, log_det, reverse=True)
        else:
            x, log_det = self.actnorm(x, log_det, initialize=initialize_actnorm)
            x, log_det = self.invconv(x, log_det)
            x, log_det = self.coupling(x, log_det)
        return x, log_det


class Squeeze(nn.Module):

    def __init__(self, factor: int = 2):
        super().__init__()
        assert factor >= 1 and isinstance(factor, int)
        self.factor = factor

    def forward(self, x, log_det=None, reverse: bool = False):
        b, c, h, w = x.size()
        if self.factor == 1:
            return x
        factor2 = self.factor ** 2
        if reverse:
            assert c % factor2 == 0, "{}".format(c)
            x = x.view(b, c // factor2, self.factor, self.factor, h, w)
            x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
            x = x.view(b, c // factor2, h * self.factor, w * self.factor)
        else:
            assert h % self.factor == 0 and w % self.factor == 0, x.size()
            x = x.view(b, c, h // self.factor, self.factor, w // self.factor, self.factor)
            x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
            x = x.view(b, c * factor2, h // self.factor, w // self.factor)
        return x, log_det


class Split(nn.Module):
    """ Split input by channel dimension and compute log likelihood of Gaussian distribution with trainable parameters.
    input -> [a, b]
    a -> CNN -> [mean, log_sd]
    b -> Gaussian_likelihood(b, mean, log_sd) -> log likelihood
    * the reason to use learnt prior is described in https://github.com/openai/glow/issues/66
    """
    log2pi = float(np.log(2 * np.pi))

    def __init__(self, in_channels, split: bool = True):
        super().__init__()
        self.split = split
        if self.split:
            assert in_channels % 2 == 0, in_channels
            self.conv = ZeroConv2d(in_channels // 2, in_channels)
        else:
            self.conv = ZeroConv2d(in_channels, in_channels * 2)

    def forward(self, x, z=None, log_det=None, reverse: bool = False, eps_std: float = None):
        """ Splitting forward inference/reverse sampling module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, to split (forward) or concatenate with z or sampled variable (reverse)
        reverse : bool
            Switch to reverse mode to sample data from latent variable.
        z : torch.Tensor
            Latent state, to concatenate with x in reverse mode, which is None as the default to
            generate latent state by sampling from the learnt distribution by x.
        log_det : torch.Tensor
            Log determinant for model training in forward mode. Skip to compute if None or reverse mode.
        eps_std : bool
            Factor to scale sampling distribution.

        Returns
        -------
        output :
            (forward mode) List containing two torch.Tensor (z1, z2)
                - z1: variable further to be processed
                - z2: variable for latent state
            (reverse mode) torch.Tensor of generated variables
        log_det : Log determinant.
        """
        if reverse:
            if self.split:
                if z is None:
                    mean, log_sd = self.conv(x).chunk(2, 1)
                    z = self.gaussian_sample(mean, log_sd, eps_std)
                z = torch.cat((x, z), dim=1)
            else:
                if z is None:
                    assert x.sum() == 0, 'sampling seed should be zero tensor'
                    mean, log_sd = self.conv(x).chunk(2, 1)
                    z = self.gaussian_sample(mean, log_sd, eps_std)
            return z, log_det
        else:
            # z1: variable further to be processed, z2: variable for latent state
            if self.split:
                z1, z2 = x.chunk(2, 1)
            else:
                z1 = torch.zeros_like(x)
                z2 = x
            mean, log_sd = self.conv(z1).chunk(2, 1)
            if log_det is not None:
                log_likeli = self.gaussian_log_likelihood(z2, mean, log_sd)
                log_det = log_det + log_likeli.view(x.shape[0], -1).sum(1)
            return (z1, z2), log_det

    def gaussian_log_likelihood(self, x, mean, log_sd):
        """ Gaussian log likelihood
            lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
            k = 1 (Independent)
            Var = logs ** 2 """
        return -0.5 * self.log2pi - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)

    @staticmethod
    def gaussian_sample(mean, log_sd, eps_std=None):
        eps_std = eps_std or 1
        eps = torch.normal(mean=torch.zeros_like(mean), std=torch.ones_like(log_sd) * eps_std)
        return mean + torch.exp(log_sd) * eps


class GlowNetwork(nn.Module):
    """ Glow network architecture
                     n_flow_step                           n_flow_step
    --> [Squeeze] -> [FlowStep] -> [Split] -> [Squeeze] -> [FlowStep]
           ^                           v
           |       (n_level - 1)       |
           + --------------------------+
    """

    def __init__(self,
                 image_shape: List,
                 filter_size: int = 512,
                 kernel_size: int = 3,
                 stride: int = 1,
                 n_flow_step: int = 32,
                 n_level: int = 3,
                 actnorm_scale: float = 1.0,
                 lu_decomposition: bool = False):
        """ Glow network architecture

        Parameters
        ----------
        image_shape: List
            Input image size.
        filter_size : int
            Filter size for CNN layer.
        n_flow_step : int
            Number of flow block.
        n_level : int
            Number of single block: -[squeeze -> flow x n_flow_step -> split]->.
        actnorm_scale : float
            Factor to scale ActNorm.
        lu_decomposition : bool
            Whether use LU decomposition in invertible CNN layer.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_flow_step = n_flow_step
        self.n_level = n_level
        flow_config = {'filter_size': filter_size, 'kernel_size': kernel_size, 'stride': stride,
                       'actnorm_scale': actnorm_scale, 'lu_decomposition': lu_decomposition}
        h, w, c = image_shape
        self.pixel_size = h * w * c
        assert c in [1, 3], "image_shape should be HWC, like (64, 64, 3): {}".format(image_shape)
        for i in range(self.n_level):

            # 1. Squeeze
            assert h % 2 == 0 and w % 2 == 0, (h, w)
            h, w, c = h // 2, w // 2, c * 4
            self.layers.append(Squeeze(factor=2))

            # 2. K FlowStep
            for _ in range(self.n_flow_step):
                self.layers.append(FlowStep(in_channels=c, **flow_config))

            # 3. Split2d
            split = i != (self.n_level - 1)
            self.layers.append(Split(in_channels=c, split=split))
            if split:
                assert c % 2 == 0, c
                c = c // 2
        self.last_latent_shape = [c, h, w]

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
                    if latent_states is not None:
                        # reconstruct from latent variable
                        x, _ = layer(x, reverse=True, eps_std=eps_std, z=latent_states.pop(-1))
                    else:
                        # random generation
                        x, _ = layer(x, reverse=True, eps_std=eps_std)
                else:
                    x, _ = layer(x, reverse=True)
            return x, None
        else:
            assert x is not None, '`x` have to be a tensor, not None'
            if return_loss:
                log_det = 0
            else:
                log_det = None  # skip log det computation

            latent_states = []
            for layer in self.layers:
                if isinstance(layer, FlowStep):
                    x, log_det = layer(x, log_det=log_det, initialize_actnorm=initialize_actnorm)
                elif isinstance(layer, Split):
                    (x, z), log_det = layer(x, log_det=log_det)
                    latent_states.append(z)
                else:
                    x, log_det = layer(x, log_det=log_det)

            if log_det is not None:
                nll = - log_det / self.pixel_size
            else:
                nll = None
            return latent_states, nll

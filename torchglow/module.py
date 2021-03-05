from typing import List
import scipy
import torch
import numpy as np
from torch import nn
from torch.nn.functional import conv2d, sigmoid

EPS = 1e-6


def pixels(tensor: torch.Tensor):
    return int(tensor.size(2) * tensor.size(3))


def mean_tensor(tensor: torch.Tensor, dim: List = None, keepdim: bool = False):
    """ Take the mean along multiple dimensions.
    :param tensor: Tensor of values to average.
    :param dim: List of dimensions along which to take the mean.
    :param keepdim: Keep dimensions rather than squeezing.
    :return (torch.Tensor): New tensor of mean value(s).
    """
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


def conv2d_padding(padding: str, kernel_size: (int, List), stride: (int, List)):
    """ Get padding values for 2D convolution layer """
    if isinstance(padding, str):
        kernel_size = [kernel_size, kernel_size] if isinstance(kernel_size, int) else kernel_size
        stride = [stride, stride] if isinstance(stride, int) else stride
        if padding.lower() == 'same':
            return [((k - 1) * s + 1) // 2 for k, s in zip(kernel_size, stride)]
        elif padding.lower() == 'valid':
            return [0] * len(kernel_size)
    raise ValueError("{} is not supported".format(padding))


class ActNorm2d(nn.Module):
    """ Activation normalization for 2D inputs.

    The bias and scale get initialized using the mean and variance of the
    first mini-batch. After the init, bias and scale are trainable parameters.
    """

    def __init__(self, num_features, scale: float = 1.):
        super().__init__()
        self.is_initialized = False
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.num_features = num_features
        self.scale = float(scale)

    def initialize_parameters(self, x):
        if self.training and not self.is_initialized:
            with torch.no_grad():
                bias = - mean_tensor(x.clone(), dim=[0, 2, 3], keepdim=True)
                v = mean_tensor((x.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
                logs = (self.scale / (v.sqrt() + EPS)).log()
                self.bias.data.copy_(bias.data)
                self.logs.data.copy_(logs.data)
            self.is_initialized = True

    def forward(self, x, log_det=None, reverse: bool = False):
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
        return x - self.bias * flag

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
            lu_p, lu_l, lu_u = scipy.linalg.lu(w_init)

            # trainable parameters
            lu_u = np.triu(lu_u, k=1)
            self.lu_u = nn.Parameter(torch.Tensor(lu_u.astype(np.float32)))
            lu_s = np.diag(lu_u)
            log_s = np.log(np.abs(lu_s))
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
            lu_u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
            if log_det is not None:
                log_det = log_det + flag * self.log_s.sum() * pixels(x)
            if reverse:
                lu_l = torch.inverse(lu_l.double()).float()
                lu_u = torch.inverse(lu_u.double()).float()
                weight = torch.matmul(lu_u, torch.matmul(lu_l, self.p.inverse()))
            else:
                weight = torch.matmul(self.p, torch.matmul(lu_l, lu_u))
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
                 kernel_size: (List, int) = 3,
                 stride: (List, int) = 1,
                 padding: (List, int) = None,
                 log_scale_factor: float = 3):
        # padding_value = conv2d_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)

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
                 kernel_size: (List, int) = 3,
                 stride: (List, int) = 1,
                 padding: str = "same"):
        assert in_channels % 2 == 0, in_channels
        super().__init__()

        # affine sub-network: take the half of the input channel and produce feature sized full channel
        kernel_size_mid = 1
        self.net = nn.Sequential(
            nn.Conv2d(int(in_channels / 2), filter_size, kernel_size, stride,
                      padding=conv2d_padding(padding, kernel_size, stride)),
            nn.ReLU(),
            nn.Conv2d(filter_size, filter_size, kernel_size_mid, stride,
                      padding=conv2d_padding(padding, kernel_size_mid, stride)),
            nn.ReLU(),
            ZeroConv2d(filter_size, in_channels, kernel_size, stride,
                       padding=conv2d_padding(padding, kernel_size, stride)),
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
        s = sigmoid(log_s + 2)
        if reverse:
            y_a = x_a / s - t
        else:
            y_a = (x_a + t) * s
            log_det = log_det + torch.sum(torch.log(s).view(x.shape[0], -1), 1)
        return torch.cat([y_a, x_b], 1), log_det


class FlowStep(nn.Module):
    """ One step of flow: input -> ActNorm -> Invertible Conv -> Affine Coupling -> output """

    def __init__(self,
                 in_channels: int,
                 filter_size: int = 512,
                 actnorm_scale: float = 1.0,
                 lu_decomposition: bool = False):
        super().__init__()

        self.actnorm = ActNorm2d(in_channels, actnorm_scale)
        self.invconv = InvertibleConv2d(in_channels, lu_decomposition)
        self.coupling = AffineCoupling(in_channels, filter_size=filter_size)

    def forward(self, x, log_det=None, reverse: bool = False):
        if reverse:
            x, log_det = self.coupling(x, log_det, reverse=True)
            x, log_det = self.invconv(x, log_det, reverse=True)
            x, log_det = self.actnorm(x, log_det, reverse=True)
        else:
            x, log_det = self.actnorm(x, log_det)
            x, log_det = self.invconv(x, log_det)
            x, log_det = self.coupling(x, log_det)
        return x, log_det


class Squeeze(nn.Module):

    def __init__(self, factor: int = 2):
        super().__init__()
        assert factor >= 1 and isinstance(factor, int)
        self.factor = factor

    def forward(self, x, log_det=None, reverse: bool=False):
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
    """ Split input by channel dimension and compute log likelihood of Gaussian distribution with trainable parameters
    input -> [a, b]
    a -> CNN -> [mean, log_sd]
    b -> Gaussian_likelihood(b, mean, log_sd) -> log likelihood
    * the reason to use learnt prior is described in https://github.com/openai/glow/issues/66
    """
    log2pi = float(np.log(2 * np.pi))

    def __init__(self, in_channels, split: bool = True):
        super().__init__()
        assert in_channels & 2 == 0, in_channels
        self.split = split
        if self.split:
            self.conv = ZeroConv2d(in_channels // 2, in_channels)
        else:
            self.conv = ZeroConv2d(in_channels, in_channels * 2)

    def forward(self, x, log_det=None, reverse: bool = False, eps_std: float = None):
        if reverse:
            if self.split:
                z1 = x
                mean, log_sd = self.conv(z1).chunk(2, 1)
                z2 = self.gaussian_sample(mean, log_sd, eps_std)
                z = torch.cat((z1, z2), dim=1)
            else:
                z1 = torch.zeros_like(x)
                mean, log_sd = self.conv(z1).chunk(2, 1)
                z = self.gaussian_sample(mean, log_sd, eps_std)
            return z, log_det
        else:
            if self.split:
                z1, z2 = x.chunk(2, 1)
            else:
                z1 = torch.zeros_like(x)
                z2 = x
            mean, log_sd = self.conv(z1).chunk(2, 1)

            if log_det is not None:
                log_det = self.gaussian_log_likelihood(z2, mean, log_sd) + log_det
            return z1, log_det

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
    """ Overall network of Glow
                     n_flow_step                           n_flow_step
    --> [Squeeze] -> [FlowStep] -> [Split] -> [Squeeze] -> [FlowStep]
           ^                           v
           |       (n_level - 1)       |
           + --------------------------+
    """

    def __init__(self,
                 image_shape,
                 filter_size: int = 512,
                 n_flow_step: int = 32,
                 n_level: int = 3,
                 actnorm_scale: float = 1.0,
                 lu_decomposition: bool = False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.n_flow_step = n_flow_step
        self.n_level = n_level
        flow_config = {'filter_size': filter_size, 'actnorm_scale': actnorm_scale, 'lu_decomposition': lu_decomposition}
        h, w, c = image_shape
        assert c in [1, 3], "image_shape should be HWC, like (64, 64, 3): {}".format(image_shape)
        for i in range(self.n_level):

            # 1. Squeeze
            assert h % 2 == 0 and w % 2 == 0, (h, w)
            h, w, c = h // 2, w // 2, c * 4
            self.layers.append(Squeeze(factor=2))
            self.output_shapes.append([-1, c, h, w])

            # 2. K FlowStep
            for _ in range(self.n_flow_step):
                self.layers.append(FlowStep(in_channels=c, **flow_config))
                self.output_shapes.append([-1, c, h, w])

            # 3. Split2d
            split = i == (self.n_level - 1)
            self.layers.append(Split(in_channels=c, split=split))
            assert c % 2 == 0, c
            c = c // 2
            self.output_shapes.append([-1, c, h, w])

    def forward(self, x, log_det=0., reverse=False, eps_std=None):
        if reverse:
            for layer in reversed(self.layers):
                if isinstance(layer, Split):
                    x, _ = layer(x, reverse=True, eps_std=eps_std)
                else:
                    x, _ = layer(x, reverse=True)
            return x
        else:
            for layer in self.layers:
                x, log_det = layer(x, log_det)
            return x, log_det


class Glow(nn.Module):

    def __init__(self,
                 image_shape,
                 filter_size: int = 512,
                 n_flow_step: int = 32,
                 n_level: int = 3,
                 actnorm_scale: float = 1.0,
                 lu_decomposition: bool = False):
        super().__init__()
        self.flow = GlowNetwork(
            image_shape=image_shape,
            filter_size=filter_size,
            n_flow_step=n_flow_step,
            n_level=n_level,
            actnorm_scale=actnorm_scale,
            lu_decomposition=lu_decomposition
        )
        # for prior
        if hparams.Glow.learn_top:
            C = self.flow.output_shapes[-1][1]
            self.learn_top = modules.Conv2dZeros(C * 2, C * 2)
        # register prior hidden
        num_device = len(utils.get_proper_device(hparams.Device.glow, False))
        assert hparams.Train.batch_size % num_device == 0
        self.register_parameter(
            "prior_h",
            nn.Parameter(torch.zeros([hparams.Train.batch_size // num_device,
                                      self.flow.output_shapes[-1][1] * 2,
                                      self.flow.output_shapes[-1][2],
                                      self.flow.output_shapes[-1][3]])))

    def prior(self, y_onehot=None):
        B, C = self.prior_h.size(0), self.prior_h.size(1)
        h = self.prior_h.detach().clone()
        assert torch.sum(h) == 0.0
        if self.hparams.Glow.learn_top:
            h = self.learn_top(h)
        if self.hparams.Glow.y_condition:
            assert y_onehot is not None
            yp = self.project_ycond(y_onehot).view(B, C, 1, 1)
            h += yp
        return thops.split_feature(h, "split")

    def forward(self, x=None, y_onehot=None, z=None,
                eps_std=None, reverse=False):
        if not reverse:
            return self.normal_flow(x, y_onehot)
        else:
            return self.reverse_flow(z, y_onehot, eps_std)

    def normal_flow(self, x, y_onehot):
        pixels = thops.pixels(x)
        z = x + torch.normal(mean=torch.zeros_like(x),
                             std=torch.ones_like(x) * (1. / 256.))
        logdet = torch.zeros_like(x[:, 0, 0, 0])
        logdet += float(-np.log(256.) * pixels)
        # encode
        z, objective = self.flow(z, logdet=logdet, reverse=False)
        # prior
        mean, logs = self.prior(y_onehot)
        objective += modules.GaussianDiag.logp(mean, logs, z)

        if self.hparams.Glow.y_condition:
            y_logits = self.project_class(z.mean(2).mean(2))
        else:
            y_logits = None

        # return
        nll = (-objective) / float(np.log(2.) * pixels)
        return z, nll, y_logits

    def reverse_flow(self, z, y_onehot, eps_std):
        with torch.no_grad():
            mean, logs = self.prior(y_onehot)
            if z is None:
                z = modules.GaussianDiag.sample(mean, logs, eps_std)
            x = self.flow(z, eps_std=eps_std, reverse=True)
        return x

    def set_actnorm_init(self, inited=True):
        for name, m in self.named_modules():
            if (m.__class__.__name__.find("ActNorm") >= 0):
                m.inited = inited

    def generate_z(self, img):
        self.eval()
        B = self.hparams.Train.batch_size
        x = img.unsqueeze(0).repeat(B, 1, 1, 1).cuda()
        z,_, _ = self(x)
        self.train()
        return z[0].detach().cpu().numpy()

    def generate_attr_deltaz(self, dataset):
        assert "y_onehot" in dataset[0]
        self.eval()
        with torch.no_grad():
            B = self.hparams.Train.batch_size
            N = len(dataset)
            attrs_pos_z = [[0, 0] for _ in range(self.y_classes)]
            attrs_neg_z = [[0, 0] for _ in range(self.y_classes)]
            for i in tqdm(range(0, N, B)):
                j = min([i + B, N])
                # generate z for data from i to j
                xs = [dataset[k]["x"] for k in range(i, j)]
                while len(xs) < B:
                    xs.append(dataset[0]["x"])
                xs = torch.stack(xs).cuda()
                zs, _, _ = self(xs)
                for k in range(i, j):
                    z = zs[k - i].detach().cpu().numpy()
                    # append to different attrs
                    y = dataset[k]["y_onehot"]
                    for ai in range(self.y_classes):
                        if y[ai] > 0:
                            attrs_pos_z[ai][0] += z
                            attrs_pos_z[ai][1] += 1
                        else:
                            attrs_neg_z[ai][0] += z
                            attrs_neg_z[ai][1] += 1
                # break
            deltaz = []
            for ai in range(self.y_classes):
                if attrs_pos_z[ai][1] == 0:
                    attrs_pos_z[ai][1] = 1
                if attrs_neg_z[ai][1] == 0:
                    attrs_neg_z[ai][1] = 1
                z_pos = attrs_pos_z[ai][0] / float(attrs_pos_z[ai][1])
                z_neg = attrs_neg_z[ai][0] / float(attrs_neg_z[ai][1])
                deltaz.append(z_pos - z_neg)
        self.train()
        return deltaz

    @staticmethod
    def loss_generative(nll):
        # Generative loss
        return torch.mean(nll)

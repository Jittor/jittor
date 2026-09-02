"""Building blocks shared by the torchvision-derived model zoo.

Each of these existed in three or four copies, one per model file. The copies
had drifted: ``ConvNormActivation`` in particular had a version that replaced
``activation_layer=None`` with a default before testing it, which silently gave
EfficientNet's projection layers an activation they must not have. Keep one
implementation and let the models parameterise it.

Semantics follow torchvision: ``norm_layer=None`` and ``activation_layer=None``
mean *skip that part*, they are not "use the default".
"""

import jittor as jt
from jittor import nn

__all__ = [
    "make_divisible",
    "ConvNormActivation",
    "SqueezeExcitation",
    "StochasticDepth",
]


def make_divisible(v, divisor=8, min_value=None):
    """Round a channel count to a multiple of ``divisor`` (torchvision rule).

    Never drops more than 10% of the original value.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvNormActivation(nn.Sequential):
    """Conv -> (Norm) -> (Activation), mirroring torchvision's
    ``Conv2dNormActivation``.

    ``norm_layer=None`` builds no normalization and ``activation_layer=None``
    builds no activation. ``bias`` defaults to ``norm_layer is None``, because a
    following normalization makes the conv bias redundant.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=None, groups=1, norm_layer=nn.BatchNorm,
                 activation_layer=nn.ReLU, dilation=1, bias=None):
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None
        layers = [
            nn.Conv(in_channels, out_channels, kernel_size, stride, padding,
                    dilation=dilation, groups=groups, bias=bias),
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer())
        super(ConvNormActivation, self).__init__(*layers)
        self.out_channels = out_channels


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block (torchvision variant).

    Two 1x1 convolutions stand in for the fully-connected layers, with
    ``activation`` on the reduction and ``scale_activation`` as the gate
    (Sigmoid for EfficientNet/RegNet/MaxViT, Hardsigmoid for MobileNetV3).
    """

    def __init__(self, input_channels, squeeze_channels,
                 activation=nn.ReLU, scale_activation=nn.Sigmoid):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Conv(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, x):
        scale = x.mean([2, 3], keepdims=True)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return scale

    def execute(self, x):
        return self._scale(x) * x


class StochasticDepth(nn.Module):
    """Stochastic Depth (drop whole residual branches), torchvision semantics.

    In training, each sample's branch is zeroed with probability ``p`` and the
    survivors are rescaled by ``1 / (1 - p)`` so the expectation is unchanged.
    In evaluation it is the identity. ``mode="row"`` decides per sample,
    ``mode="batch"`` decides once for the whole batch.
    """

    def __init__(self, p, mode="row"):
        super(StochasticDepth, self).__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError("drop probability has to be between 0 and 1, "
                             "but got {}".format(p))
        if mode not in ("batch", "row"):
            raise ValueError("mode has to be either 'batch' or 'row', "
                             "but got {}".format(mode))
        self.p = p
        self.mode = mode

    def execute(self, x):
        if not self.is_training() or self.p == 0.0:
            return x
        survival_rate = 1.0 - self.p
        if self.mode == "row":
            size = [x.shape[0]] + [1] * (x.ndim - 1)
        else:
            size = [1] * x.ndim
        noise = (jt.rand(size) < survival_rate).float32()
        if survival_rate > 0.0:
            noise = noise / survival_rate
        return x * noise

    def __repr__(self):
        return "{}(p={}, mode={})".format(
            self.__class__.__name__, self.p, self.mode)

# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Jittor Group
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# EfficientNet (b0 - b3), faithfully aligned with torchvision.
# Reference:
#   "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
#   https://arxiv.org/abs/1905.11946
#   torchvision.models.efficientnet (V1 family)

import math
import copy

import jittor as jt
from jittor import nn

__all__ = [
    'EfficientNet',
    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
]


def _make_divisible(v, divisor=8, min_value=None):
    """Make a channel count divisible by ``divisor`` (matches torchvision)."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that rounding down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class StochasticDepth(nn.Module):
    """Stochastic Depth (drop whole residual branches), torchvision "row" mode.

    During training, with probability ``p`` the entire input batch element's
    residual branch is zeroed (and the kept samples are rescaled by
    ``1 / (1 - p)``). During evaluation it is the identity function.
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
        # Bernoulli(survival_rate) mask, then rescale to keep expectation.
        noise = (jt.rand(size) < survival_rate).float32()
        if survival_rate > 0.0:
            noise = noise / survival_rate
        return x * noise

    def __repr__(self):
        return "{}(p={}, mode={})".format(
            self.__class__.__name__, self.p, self.mode)


class ConvBNActivation(nn.Sequential):
    """Conv -> BatchNorm -> Activation, mirroring torchvision's ConvNormActivation."""

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
                 groups=1, norm_layer=None, activation_layer=None, dilation=1):
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm
        if activation_layer is None:
            activation_layer = nn.SiLU
        layers = [
            nn.Conv(in_planes, out_planes, kernel_size, stride, padding,
                    dilation=dilation, groups=groups, bias=False),
            norm_layer(out_planes),
        ]
        if activation_layer is not None:
            layers.append(activation_layer())
        super(ConvBNActivation, self).__init__(*layers)
        self.out_channels = out_planes


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block (torchvision variant).

    Uses 1x1 convolutions for the two fully-connected layers, an activation on
    the reduction (SiLU for EfficientNet) and a Sigmoid gate.
    """

    def __init__(self, input_channels, squeeze_channels,
                 activation=nn.SiLU, scale_activation=nn.Sigmoid):
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


class MBConvConfig:
    """Stores the configuration of one EfficientNet MBConv stage."""

    def __init__(self, expand_ratio, kernel, stride, input_channels,
                 out_channels, num_layers, width_mult=1.0, depth_mult=1.0):
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.num_layers = self.adjust_depth(num_layers, depth_mult)

    @staticmethod
    def adjust_channels(channels, width_mult, min_value=None):
        return _make_divisible(channels * width_mult, 8, min_value)

    @staticmethod
    def adjust_depth(num_layers, depth_mult):
        return int(math.ceil(num_layers * depth_mult))

    def __repr__(self):
        return ("MBConvConfig(expand_ratio={}, kernel={}, stride={}, "
                "input_channels={}, out_channels={}, num_layers={})".format(
                    self.expand_ratio, self.kernel, self.stride,
                    self.input_channels, self.out_channels, self.num_layers))


class MBConv(nn.Module):
    """EfficientNet MBConv block: expand -> dwise -> SE -> project (+ stoch. depth)."""

    def __init__(self, cnf, stochastic_depth_prob, norm_layer=None,
                 se_layer=SqueezeExcitation):
        super(MBConv, self).__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')
        if norm_layer is None:
            norm_layer = nn.BatchNorm

        self.use_res_connect = (cnf.stride == 1 and
                                cnf.input_channels == cnf.out_channels)

        layers = []
        activation_layer = nn.SiLU

        # expand (1x1)
        expanded_channels = cnf.adjust_channels(cnf.input_channels,
                                                cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(ConvBNActivation(
                cnf.input_channels, expanded_channels, kernel_size=1,
                norm_layer=norm_layer, activation_layer=activation_layer))

        # depthwise
        layers.append(ConvBNActivation(
            expanded_channels, expanded_channels, kernel_size=cnf.kernel,
            stride=cnf.stride, groups=expanded_channels,
            norm_layer=norm_layer, activation_layer=activation_layer))

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels,
                               activation=nn.SiLU))

        # project (1x1, no activation)
        layers.append(ConvBNActivation(
            expanded_channels, cnf.out_channels, kernel_size=1,
            norm_layer=norm_layer, activation_layer=None))

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def execute(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result = result + x
        return result


class EfficientNet(nn.Module):
    """EfficientNet model architecture, aligned with torchvision.

    Args:

    * inverted_residual_setting: List of ``MBConvConfig`` describing the network.
    * dropout: Dropout probability in the classifier.
    * stochastic_depth_prob: Maximum stochastic-depth drop probability. Default: 0.2.
    * num_classes: Number of classes. Default: 1000.
    * block: MBConv building block. If None, use ``MBConv``.
    * norm_layer: Normalization module. If None, use ``nn.BatchNorm``.
    """

    def __init__(self, inverted_residual_setting, dropout,
                 stochastic_depth_prob=0.2, num_classes=1000,
                 block=None, norm_layer=None):
        super(EfficientNet, self).__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        if not isinstance(inverted_residual_setting, (list, tuple)):
            raise TypeError("The inverted_residual_setting should be a list")
        if not all(isinstance(s, MBConvConfig) for s in inverted_residual_setting):
            raise TypeError("inverted_residual_setting should be "
                            "List[MBConvConfig]")

        if block is None:
            block = MBConv
        if norm_layer is None:
            norm_layer = nn.BatchNorm

        layers = []

        # building first layer (stem)
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(ConvBNActivation(
            3, firstconv_output_channels, kernel_size=3, stride=2,
            norm_layer=norm_layer, activation_layer=nn.SiLU))

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers
                                 for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage = []
            for _ in range(cnf.num_layers):
                # copy config to mutate per-block (later blocks: stride 1, in==out)
                block_cnf = copy.copy(cnf)
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1
                # adjust stochastic depth probability based on the depth.
                sd_prob = (stochastic_depth_prob * float(stage_block_id) /
                           total_stage_blocks)
                stage.append(block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))

        # building last several layers (head)
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 4 * lastconv_input_channels
        layers.append(ConvBNActivation(
            lastconv_input_channels, lastconv_output_channels, kernel_size=1,
            norm_layer=norm_layer, activation_layer=nn.SiLU))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(lastconv_output_channels, num_classes),
        )

    def _forward_impl(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = jt.reshape(x, (x.shape[0], -1))
        x = self.classifier(x)
        return x

    def execute(self, x):
        return self._forward_impl(x)


def _efficientnet_conf(arch, width_mult, depth_mult):
    bneck_conf = lambda *args: MBConvConfig(
        *args, width_mult=width_mult, depth_mult=depth_mult)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 5, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    return inverted_residual_setting


# (width_mult, depth_mult, default classifier dropout) per torchvision.
_EFFICIENTNET_PARAMS = {
    "efficientnet_b0": (1.0, 1.0, 0.2),
    "efficientnet_b1": (1.0, 1.1, 0.2),
    "efficientnet_b2": (1.1, 1.2, 0.3),
    "efficientnet_b3": (1.2, 1.4, 0.3),
}


def _efficientnet(arch, **kwargs):
    width_mult, depth_mult, dropout = _EFFICIENTNET_PARAMS[arch]
    inverted_residual_setting = _efficientnet_conf(arch, width_mult, depth_mult)
    # allow caller to override dropout via kwargs, else use the per-arch default
    kwargs.setdefault("dropout", dropout)
    model = EfficientNet(inverted_residual_setting, **kwargs)
    return model


def efficientnet_b0(pretrained=False, **kwargs):
    """EfficientNet-B0 model architecture.

    Args:

    * pretrained: If True, load pretrained weights. Default: False.
    * num_classes: Number of classes. Default: 1000.
    """
    if pretrained:
        raise NotImplementedError("pretrained weights not yet on jittorhub")
    return _efficientnet("efficientnet_b0", **kwargs)


def efficientnet_b1(pretrained=False, **kwargs):
    """EfficientNet-B1 model architecture.

    Args:

    * pretrained: If True, load pretrained weights. Default: False.
    * num_classes: Number of classes. Default: 1000.
    """
    if pretrained:
        raise NotImplementedError("pretrained weights not yet on jittorhub")
    return _efficientnet("efficientnet_b1", **kwargs)


def efficientnet_b2(pretrained=False, **kwargs):
    """EfficientNet-B2 model architecture.

    Args:

    * pretrained: If True, load pretrained weights. Default: False.
    * num_classes: Number of classes. Default: 1000.
    """
    if pretrained:
        raise NotImplementedError("pretrained weights not yet on jittorhub")
    return _efficientnet("efficientnet_b2", **kwargs)


def efficientnet_b3(pretrained=False, **kwargs):
    """EfficientNet-B3 model architecture.

    Args:

    * pretrained: If True, load pretrained weights. Default: False.
    * num_classes: Number of classes. Default: 1000.
    """
    if pretrained:
        raise NotImplementedError("pretrained weights not yet on jittorhub")
    return _efficientnet("efficientnet_b3", **kwargs)

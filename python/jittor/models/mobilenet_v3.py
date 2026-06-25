# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Jittor Group
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# MobileNetV3 (large & small), faithfully aligned with torchvision.
# Reference:
#   "Searching for MobileNetV3" https://arxiv.org/abs/1905.02244
#   torchvision.models.mobilenetv3

import jittor as jt
from jittor import nn

__all__ = ['MobileNetV3', 'mobilenet_v3_large', 'mobilenet_v3_small']


def _make_divisible(v, divisor=8, min_value=None):
    """Make a channel count divisible by ``divisor`` (matches torchvision)."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that rounding down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Hardsigmoid(nn.Module):
    """Hardsigmoid: clamp(x + 3, 0, 6) / 6 (torchvision / paper definition)."""

    def execute(self, x):
        return jt.clamp(x + 3.0, min_v=0.0, max_v=6.0) / 6.0


class Hardswish(nn.Module):
    """Hardswish: x * hardsigmoid(x) = x * clamp(x + 3, 0, 6) / 6."""

    def execute(self, x):
        return x * (jt.clamp(x + 3.0, min_v=0.0, max_v=6.0) / 6.0)


class ConvBNActivation(nn.Sequential):
    """Conv -> BatchNorm -> Activation, mirroring torchvision's ConvBNActivation."""

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
                 groups=1, norm_layer=None, activation_layer=None, dilation=1):
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm
        if activation_layer is None:
            activation_layer = nn.ReLU
        super(ConvBNActivation, self).__init__(
            nn.Conv(in_planes, out_planes, kernel_size, stride, padding,
                    dilation=dilation, groups=groups, bias=False),
            norm_layer(out_planes),
            activation_layer(),
        )
        self.out_channels = out_planes


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block (torchvision MobileNetV3 variant).

    Uses 1x1 convolutions for the two fully-connected layers, ReLU on the
    reduction and Hardsigmoid as the gate. ``squeeze_channels`` defaults to
    ``_make_divisible(input_channels // 4, 8)``.
    """

    def __init__(self, input_channels, squeeze_factor=4):
        super(SqueezeExcitation, self).__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv(squeeze_channels, input_channels, 1)
        self.hardsigmoid = Hardsigmoid()

    def _scale(self, x):
        scale = x.mean([2, 3], keepdims=True)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.hardsigmoid(scale)
        return scale

    def execute(self, x):
        return self._scale(x) * x


class InvertedResidualConfig:
    """Stores the configuration of one MobileNetV3 inverted-residual block."""

    def __init__(self, input_channels, kernel, expanded_channels, out_channels,
                 use_se, activation, stride, dilation, width_mult):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels, width_mult):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):
    """MobileNetV3 inverted residual block: expand -> dwise -> (SE) -> project."""

    def __init__(self, cnf, norm_layer=None, se_layer=SqueezeExcitation):
        super(InvertedResidual, self).__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')
        if norm_layer is None:
            norm_layer = nn.BatchNorm

        self.use_res_connect = (cnf.stride == 1 and
                                cnf.input_channels == cnf.out_channels)

        layers = []
        activation_layer = Hardswish if cnf.use_hs else nn.ReLU

        # expand (1x1)
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(ConvBNActivation(
                cnf.input_channels, cnf.expanded_channels, kernel_size=1,
                norm_layer=norm_layer, activation_layer=activation_layer))

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(ConvBNActivation(
            cnf.expanded_channels, cnf.expanded_channels, kernel_size=cnf.kernel,
            stride=stride, dilation=cnf.dilation, groups=cnf.expanded_channels,
            norm_layer=norm_layer, activation_layer=activation_layer))

        # squeeze and excitation
        if cnf.use_se:
            layers.append(se_layer(cnf.expanded_channels))

        # project (1x1, no activation)
        layers.append(ConvBNActivation(
            cnf.expanded_channels, cnf.out_channels, kernel_size=1,
            norm_layer=norm_layer, activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels

    def execute(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result = result + x
        return result


class MobileNetV3(nn.Module):
    """MobileNetV3 model architecture, aligned with torchvision.

    Args:

    * inverted_residual_setting: List of ``InvertedResidualConfig`` describing the network.
    * last_channel: Number of channels on the penultimate (classifier) layer.
    * num_classes: Number of classes. Default: 1000.
    * block: Inverted-residual building block. If None, use ``InvertedResidual``.
    * norm_layer: Normalization module. If None, use ``nn.BatchNorm``.
    * dropout: Dropout probability in the classifier. Default: 0.2.
    """

    def __init__(self, inverted_residual_setting, last_channel, num_classes=1000,
                 block=None, norm_layer=None, dropout=0.2):
        super(MobileNetV3, self).__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        if not isinstance(inverted_residual_setting, (list, tuple)):
            raise TypeError("The inverted_residual_setting should be a list")

        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = nn.BatchNorm

        layers = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(ConvBNActivation(
            3, firstconv_output_channels, kernel_size=3, stride=2,
            norm_layer=norm_layer, activation_layer=Hardswish))

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(ConvBNActivation(
            lastconv_input_channels, lastconv_output_channels, kernel_size=1,
            norm_layer=norm_layer, activation_layer=Hardswish))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(last_channel, num_classes),
        )

    def _forward_impl(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = jt.reshape(x, (x.shape[0], -1))
        x = self.classifier(x)
        return x

    def execute(self, x):
        return self._forward_impl(x)


def _mobilenet_v3_conf(arch, width_mult=1.0, reduced_tail=False, dilated=False):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = lambda *args: InvertedResidualConfig(*args, width_mult=width_mult)
    adjust_channels = lambda c: InvertedResidualConfig.adjust_channels(c, width_mult)

    if arch == "mobilenet_v3_large":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),   # C1
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),    # C2
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider,
                       160 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider,
                       160 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1280 // reduce_divider)  # C5
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),    # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),   # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),    # C3
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider,
                       96 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider,
                       96 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1024 // reduce_divider)  # C5
    else:
        raise ValueError("Unsupported model type {}".format(arch))

    return inverted_residual_setting, last_channel


def _mobilenet_v3(arch, **kwargs):
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch)
    model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)
    return model


def mobilenet_v3_large(pretrained=False, **kwargs):
    """MobileNetV3 (Large) model architecture.

    Args:

    * pretrained: If True, load pretrained weights. Default: False.
    * num_classes: Number of classes. Default: 1000.
    """
    if pretrained:
        raise NotImplementedError("pretrained weights not yet on jittorhub")
    return _mobilenet_v3("mobilenet_v3_large", **kwargs)


def mobilenet_v3_small(pretrained=False, **kwargs):
    """MobileNetV3 (Small) model architecture.

    Args:

    * pretrained: If True, load pretrained weights. Default: False.
    * num_classes: Number of classes. Default: 1000.
    """
    if pretrained:
        raise NotImplementedError("pretrained weights not yet on jittorhub")
    return _mobilenet_v3("mobilenet_v3_small", **kwargs)

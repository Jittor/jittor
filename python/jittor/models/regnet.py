# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Jittor Group
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# RegNet (X and Y families), faithfully aligned with torchvision.
# Reference:
#   "Designing Network Design Spaces" https://arxiv.org/abs/2003.13678
#   torchvision.models.regnet
#
# A RegNet is the "regularized" form of an AnyNet:
#   stem -> 4 stages (each = N ResBottleneckBlocks, optionally with SE) -> head
# The per-stage widths/depths are produced by quantizing the linear width
# parameterisation (w_0, w_a, w_m) of the paper (see ``BlockParams``).

import math

import numpy as np

import jittor as jt
from jittor import nn

__all__ = [
    'RegNet',
    'regnet_y_400mf', 'regnet_y_800mf',
    'regnet_x_400mf', 'regnet_x_800mf',
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


class ConvNormActivation(nn.Sequential):
    """Conv -> BatchNorm -> Activation, mirroring torchvision's ConvNormActivation."""

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
                 groups=1, norm_layer=None, activation_layer=None, dilation=1,
                 bias=None):
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm
        if bias is None:
            # torchvision: bias is disabled when a norm layer follows.
            bias = norm_layer is None
        layers = [
            nn.Conv(in_planes, out_planes, kernel_size, stride, padding,
                    dilation=dilation, groups=groups, bias=bias),
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_planes))
        if activation_layer is not None:
            layers.append(activation_layer())
        super(ConvNormActivation, self).__init__(*layers)
        self.out_channels = out_planes


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block (torchvision variant).

    Uses 1x1 convolutions for the two fully-connected layers, an activation on
    the reduction (ReLU for RegNet) and a Sigmoid gate.
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


class SimpleStemIN(ConvNormActivation):
    """Simple stem for ImageNet: 3x3 conv, stride 2."""

    def __init__(self, width_in, width_out, norm_layer, activation_layer):
        super(SimpleStemIN, self).__init__(
            width_in, width_out, kernel_size=3, stride=2,
            norm_layer=norm_layer, activation_layer=activation_layer)


class BottleneckTransform(nn.Sequential):
    """Bottleneck transformation: 1x1 -> 3x3 (grouped) -> (SE) -> 1x1."""

    def __init__(self, width_in, width_out, stride, norm_layer,
                 activation_layer, group_width, bottleneck_multiplier,
                 se_ratio):
        w_b = int(round(width_out * bottleneck_multiplier))
        g = w_b // group_width

        layers = []
        layers.append(ConvNormActivation(
            width_in, w_b, kernel_size=1,
            norm_layer=norm_layer, activation_layer=activation_layer))
        layers.append(ConvNormActivation(
            w_b, w_b, kernel_size=3, stride=stride, groups=g,
            norm_layer=norm_layer, activation_layer=activation_layer))

        if se_ratio:
            # se_ratio is measured relative to the bottleneck's input width.
            width_se_out = int(round(se_ratio * width_in))
            layers.append(SqueezeExcitation(
                input_channels=w_b,
                squeeze_channels=width_se_out,
                activation=activation_layer))

        layers.append(ConvNormActivation(
            w_b, width_out, kernel_size=1,
            norm_layer=norm_layer, activation_layer=None))

        super(BottleneckTransform, self).__init__(*layers)


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: y = f(x) + proj(x), then activation."""

    def __init__(self, width_in, width_out, stride, norm_layer,
                 activation_layer, group_width=1, bottleneck_multiplier=1.0,
                 se_ratio=None):
        super(ResBottleneckBlock, self).__init__()

        # Use a projection shortcut when the dimensions change.
        self.proj = None
        should_proj = (width_in != width_out) or (stride != 1)
        if should_proj:
            self.proj = ConvNormActivation(
                width_in, width_out, kernel_size=1, stride=stride,
                norm_layer=norm_layer, activation_layer=None)

        self.f = BottleneckTransform(
            width_in, width_out, stride, norm_layer, activation_layer,
            group_width, bottleneck_multiplier, se_ratio)
        self.activation = activation_layer()

    def execute(self, x):
        if self.proj is not None:
            x = self.proj(x) + self.f(x)
        else:
            x = x + self.f(x)
        return self.activation(x)


class AnyStage(nn.Sequential):
    """A RegNet stage: ``depth`` ResBottleneckBlocks (first one strided)."""

    def __init__(self, width_in, width_out, stride, depth, block_constructor,
                 norm_layer, activation_layer, group_width,
                 bottleneck_multiplier, se_ratio=None, stage_index=0):
        super(AnyStage, self).__init__()
        for i in range(depth):
            block = block_constructor(
                width_in if i == 0 else width_out,
                width_out,
                stride if i == 0 else 1,
                norm_layer,
                activation_layer,
                group_width,
                bottleneck_multiplier,
                se_ratio,
            )
            self.add_module("block{}-{}".format(stage_index, i), block)


class BlockParams:
    """Per-stage block parameters generated from the RegNet parameterisation.

    Holds, for each of the (up to) 4 stages: output width, stride, depth,
    group width and bottleneck multiplier.
    """

    def __init__(self, depths, widths, group_widths, bottleneck_multipliers,
                 strides, se_ratio=None):
        self.depths = depths
        self.widths = widths
        self.group_widths = group_widths
        self.bottleneck_multipliers = bottleneck_multipliers
        self.strides = strides
        self.se_ratio = se_ratio

    @classmethod
    def from_init_params(cls, depth, w_0, w_a, w_m, group_width,
                         bottleneck_multiplier=1.0, se_ratio=None, **kwargs):
        """Build per-stage params from the linear width parameterisation.

        Follows torchvision / the paper exactly:

        * Compute a per-block "continuous" width with ``w_0 + w_a * j``.
        * Quantise each block to ``w_0 * w_m ** s`` for an integer stage index
          ``s = round(log(w/w_0) / log(w_m))``.
        * Group consecutive blocks of equal quantised width into stages.
        """
        QUANT = 8
        STRIDE = 2

        if w_a < 0 or w_0 <= 0 or w_m <= 1 or w_0 % 8 != 0:
            raise ValueError("Invalid RegNet settings")

        # NOTE: this is one-time config math (runs once at construction). We use
        # numpy rather than jittor ops so it stays cheap and avoids depending on
        # optional Var helpers; results are identical to torchvision's tensors.

        # Continuous (un-quantised) per-block widths.
        widths_cont = np.arange(depth) * w_a + w_0
        # Stage index s for each block (real-valued), then rounded.
        block_capacity = np.round(np.log(widths_cont / w_0) / math.log(w_m))
        # Quantised per-block widths, rounded to the nearest multiple of QUANT.
        block_widths = (
            np.round(np.divide(w_0 * np.power(w_m, block_capacity), QUANT))
            * QUANT
        ).astype(int).tolist()
        num_stages = len(set(block_widths))

        # Convert to per-stage (width, depth) by collapsing runs of equal width.
        split_helper = zip(
            block_widths + [0],
            [0] + block_widths,
            block_widths + [0],
            [0] + block_widths,
        )
        splits = [w != wp for w, wp, _, _ in split_helper]

        stage_widths = [w for w, t in zip(block_widths, splits[:-1]) if t]
        stage_depths = np.diff(
            np.array([d for d, t in enumerate(splits) if t])).tolist()

        strides = [STRIDE] * num_stages
        bottleneck_multipliers = [bottleneck_multiplier] * num_stages
        group_widths = [group_width] * num_stages

        # Adjust the compatibility of stage widths and group widths.
        stage_widths, group_widths = cls._adjust_widths_groups_compatibilty(
            stage_widths, bottleneck_multipliers, group_widths)

        return cls(
            depths=stage_depths,
            widths=stage_widths,
            group_widths=group_widths,
            bottleneck_multipliers=bottleneck_multipliers,
            strides=strides,
            se_ratio=se_ratio,
        )

    def _get_expanded_params(self):
        return zip(self.widths, self.strides, self.depths,
                   self.group_widths, self.bottleneck_multipliers)

    @staticmethod
    def _adjust_widths_groups_compatibilty(stage_widths, bottleneck_ratios,
                                           group_widths):
        """Adjusts the compatibility of widths and groups (torchvision).

        Ensures each (bottleneck) width is divisible by its group width.
        """
        # Compute all widths for the current settings.
        widths = [int(w * b) for w, b in zip(stage_widths, bottleneck_ratios)]
        group_widths_min = [min(g, w_bot)
                            for g, w_bot in zip(group_widths, widths)]

        # Compute the adjusted widths so that they are compatible with groups.
        ws_bot = [_make_divisible(w_bot, g)
                  for w_bot, g in zip(widths, group_widths_min)]
        stage_widths = [int(w_bot / b)
                        for w_bot, b in zip(ws_bot, bottleneck_ratios)]
        return stage_widths, group_widths_min


class RegNet(nn.Module):
    """RegNet model architecture, aligned with torchvision.

    Args:

    * block_params: A ``BlockParams`` describing the per-stage configuration.
    * num_classes: Number of classes. Default: 1000.
    * stem_width: Width of the stem convolution. Default: 32.
    * stem_type: Stem building block. If None, use ``SimpleStemIN``.
    * block_type: Stage building block. If None, use ``ResBottleneckBlock``.
    * norm_layer: Normalization module. If None, use ``nn.BatchNorm``.
    * activation: Activation module. If None, use ``nn.ReLU``.
    """

    def __init__(self, block_params, num_classes=1000, stem_width=32,
                 stem_type=None, block_type=None, norm_layer=None,
                 activation=None):
        super(RegNet, self).__init__()

        if stem_type is None:
            stem_type = SimpleStemIN
        if norm_layer is None:
            norm_layer = nn.BatchNorm
        if block_type is None:
            block_type = ResBottleneckBlock
        if activation is None:
            activation = nn.ReLU

        # Ad-hoc stem.
        self.stem = stem_type(3, stem_width, norm_layer, activation)

        current_width = stem_width

        blocks = []
        for i, (width_out, stride, depth, group_width, bottleneck_multiplier) \
                in enumerate(block_params._get_expanded_params()):
            blocks.append(
                AnyStage(
                    current_width,
                    width_out,
                    stride,
                    depth,
                    block_type,
                    norm_layer,
                    activation,
                    group_width,
                    bottleneck_multiplier,
                    block_params.se_ratio,
                    stage_index=i + 1,
                )
            )
            current_width = width_out

        self.trunk_output = nn.Sequential(*blocks)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(current_width, num_classes)

        # Initialisation, matching torchvision.
        for m in self.modules():
            if isinstance(m, nn.Conv):
                # Kaiming/He normal, fan_out, for ReLU.
                jt.init.relu_invariant_gauss_(m.weight, mode="fan_out")
                if m.bias is not None:
                    jt.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm):
                jt.init.constant_(m.weight, 1.0)
                jt.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                jt.init.gauss_(m.weight, 0.0, 0.01)
                if m.bias is not None:
                    jt.init.constant_(m.bias, 0.0)

    def execute(self, x):
        x = self.stem(x)
        x = self.trunk_output(x)
        x = self.avgpool(x)
        x = jt.reshape(x, (x.shape[0], -1))
        x = self.fc(x)
        return x


def _regnet(arch, block_params, **kwargs):
    norm_layer = kwargs.pop("norm_layer", None)
    if norm_layer is None:
        norm_layer = nn.BatchNorm
    model = RegNet(block_params, norm_layer=norm_layer, **kwargs)
    return model


def regnet_y_400mf(pretrained=False, **kwargs):
    """RegNetY_400MF model architecture (Y family => with Squeeze-Excitation).

    Args:

    * pretrained: If True, load pretrained weights. Default: False.
    * num_classes: Number of classes. Default: 1000.
    """
    if pretrained:
        raise NotImplementedError("pretrained weights not yet on jittorhub")
    params = BlockParams.from_init_params(
        depth=16, w_0=48, w_a=27.89, w_m=2.09, group_width=8,
        se_ratio=0.25, **kwargs)
    # se_ratio is consumed by BlockParams; do not forward it to RegNet.
    kwargs.pop("se_ratio", None)
    return _regnet("regnet_y_400mf", params, **kwargs)


def regnet_y_800mf(pretrained=False, **kwargs):
    """RegNetY_800MF model architecture (Y family => with Squeeze-Excitation).

    Args:

    * pretrained: If True, load pretrained weights. Default: False.
    * num_classes: Number of classes. Default: 1000.
    """
    if pretrained:
        raise NotImplementedError("pretrained weights not yet on jittorhub")
    params = BlockParams.from_init_params(
        depth=14, w_0=56, w_a=38.84, w_m=2.4, group_width=16,
        se_ratio=0.25, **kwargs)
    kwargs.pop("se_ratio", None)
    return _regnet("regnet_y_800mf", params, **kwargs)


def regnet_x_400mf(pretrained=False, **kwargs):
    """RegNetX_400MF model architecture (X family => no Squeeze-Excitation).

    Args:

    * pretrained: If True, load pretrained weights. Default: False.
    * num_classes: Number of classes. Default: 1000.
    """
    if pretrained:
        raise NotImplementedError("pretrained weights not yet on jittorhub")
    params = BlockParams.from_init_params(
        depth=22, w_0=24, w_a=24.48, w_m=2.54, group_width=16, **kwargs)
    return _regnet("regnet_x_400mf", params, **kwargs)


def regnet_x_800mf(pretrained=False, **kwargs):
    """RegNetX_800MF model architecture (X family => no Squeeze-Excitation).

    Args:

    * pretrained: If True, load pretrained weights. Default: False.
    * num_classes: Number of classes. Default: 1000.
    """
    if pretrained:
        raise NotImplementedError("pretrained weights not yet on jittorhub")
    params = BlockParams.from_init_params(
        depth=16, w_0=56, w_a=35.73, w_m=2.28, group_width=16, **kwargs)
    return _regnet("regnet_x_800mf", params, **kwargs)

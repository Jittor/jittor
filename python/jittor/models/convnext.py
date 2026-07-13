# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Jittor Group
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# ConvNeXt (tiny / small / base / large), faithfully aligned with torchvision.
# Reference:
#   "A ConvNet for the 2020s" https://arxiv.org/abs/2201.03545
#   torchvision.models.convnext
#
# Pure ``jittor.nn`` so it runs on both NVIDIA (CUDA) and Ascend (NPU) and under
# ``import jittor as torch``.

from functools import partial

import jittor as jt
from jittor import nn

__all__ = [
    'ConvNeXt',
    'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large',
]


class StochasticDepth(nn.Module):
    """Stochastic Depth (drop whole residual branches), torchvision "row" mode.

    During training, with probability ``p`` each batch element's residual branch
    is zeroed (and the kept samples are rescaled by ``1 / (1 - p)``). During
    evaluation it is the identity function.
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


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channel dim of an ``(N, C, H, W)`` tensor.

    ConvNeXt normalizes in *channels-last* layout. jittor's ``nn.LayerNorm``
    normalizes over the trailing ``normalized_shape`` dims, so we permute
    ``NCHW -> NHWC``, normalize, then permute back. Mirrors torchvision's
    ``LayerNorm2d`` (which uses ``F.layer_norm`` after two permutes).
    """

    def execute(self, x):
        x = x.permute(0, 2, 3, 1)
        x = nn.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                          self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class CNBlock(nn.Module):
    """ConvNeXt block (torchvision ``CNBlock``).

    7x7 depthwise conv -> LayerNorm (channels-last) -> 1x1 (pointwise, 4x expand
    via ``Linear``) -> GELU -> 1x1 (``Linear``) -> LayerScale -> StochasticDepth,
    then a residual add. The two pointwise convolutions are implemented as
    ``Linear`` layers operating on the channels-last ``(N, H, W, C)`` view, which
    is exactly how torchvision does it.
    """

    def __init__(self, dim, layer_scale, stochastic_depth_prob,
                 norm_layer=None):
        super(CNBlock, self).__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim,
                                bias=True)
        self.norm = norm_layer(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim, bias=True)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim, bias=True)
        # LayerScale: a learnable per-channel scale, initialised small (1e-6).
        self.layer_scale = jt.ones((dim,)) * layer_scale
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def execute(self, input):
        x = self.dwconv(input)
        x = x.permute(0, 2, 3, 1)               # NCHW -> NHWC (channels-last)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.layer_scale * x                # per-channel scale on last dim
        x = x.permute(0, 3, 1, 2)               # NHWC -> NCHW
        x = self.stochastic_depth(x)
        x = x + input
        return x


class CNBlockConfig:
    """Stores the configuration of one ConvNeXt stage."""

    def __init__(self, input_channels, out_channels, num_layers):
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

    def __repr__(self):
        return ("CNBlockConfig(input_channels={}, out_channels={}, "
                "num_layers={})".format(self.input_channels, self.out_channels,
                                        self.num_layers))


class ConvNeXt(nn.Module):
    """ConvNeXt model architecture, aligned with torchvision.

    Args:

    * block_setting: List of ``CNBlockConfig`` describing the network. The last
      stage has ``out_channels`` of ``None`` (no further downsampling).
    * stochastic_depth_prob: Maximum stochastic-depth drop probability. Default: 0.0.
    * layer_scale: Initial value of the LayerScale parameter. Default: 1e-6.
    * num_classes: Number of classes. Default: 1000.
    * block: ConvNeXt building block. If None, use ``CNBlock``.
    * norm_layer: Normalization module. If None, use ``LayerNorm2d`` with eps 1e-6.
    """

    def __init__(self, block_setting, stochastic_depth_prob=0.0,
                 layer_scale=1e-6, num_classes=1000, block=None,
                 norm_layer=None):
        super(ConvNeXt, self).__init__()

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        if not isinstance(block_setting, (list, tuple)):
            raise TypeError("The block_setting should be a list")
        if not all(isinstance(s, CNBlockConfig) for s in block_setting):
            raise TypeError("block_setting should be List[CNBlockConfig]")

        if block is None:
            block = CNBlock
        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        layers = []

        # Stem: 4x4 conv stride 4, then LayerNorm (channels-first).
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(nn.Sequential(
            nn.Conv2d(3, firstconv_output_channels, kernel_size=4, stride=4,
                      padding=0, bias=True),
            norm_layer(firstconv_output_channels),
        ))

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            # Bottlenecks of one stage.
            stage = []
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth.
                sd_prob = (stochastic_depth_prob * float(stage_block_id) /
                           (total_stage_blocks - 1.0)) \
                    if total_stage_blocks > 1 else 0.0
                stage.append(block(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # Downsampling between stages: LayerNorm (channels-first) + 2x2 conv
            # stride 2. The last stage has out_channels=None and is skipped.
            if cnf.out_channels is not None:
                layers.append(nn.Sequential(
                    norm_layer(cnf.input_channels),
                    nn.Conv2d(cnf.input_channels, cnf.out_channels,
                              kernel_size=2, stride=2),
                ))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        lastblock = block_setting[-1]
        lastconv_output_channels = (lastblock.out_channels
                                    if lastblock.out_channels is not None
                                    else lastblock.input_channels)
        self.classifier = nn.Sequential(
            norm_layer(lastconv_output_channels),
            nn.Flatten(1),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        # Weight init: trunc-normal on conv/linear weights, zeros on biases
        # (matches torchvision's ConvNeXt initialisation).
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zero_(m.bias)

    def _forward_impl(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def execute(self, x):
        return self._forward_impl(x)


def _convnext(block_setting, stochastic_depth_prob, **kwargs):
    model = ConvNeXt(block_setting,
                     stochastic_depth_prob=stochastic_depth_prob, **kwargs)
    return model


def convnext_tiny(pretrained=False, **kwargs):
    """ConvNeXt-Tiny model architecture.

    Args:

    * pretrained: If True, load pretrained weights. Default: False.
    * num_classes: Number of classes. Default: 1000.
    """
    if pretrained:
        raise NotImplementedError("pretrained weights not yet on jittorhub")
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
    return _convnext(block_setting, stochastic_depth_prob, **kwargs)


def convnext_small(pretrained=False, **kwargs):
    """ConvNeXt-Small model architecture.

    Args:

    * pretrained: If True, load pretrained weights. Default: False.
    * num_classes: Number of classes. Default: 1000.
    """
    if pretrained:
        raise NotImplementedError("pretrained weights not yet on jittorhub")
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 27),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.4)
    return _convnext(block_setting, stochastic_depth_prob, **kwargs)


def convnext_base(pretrained=False, **kwargs):
    """ConvNeXt-Base model architecture.

    Args:

    * pretrained: If True, load pretrained weights. Default: False.
    * num_classes: Number of classes. Default: 1000.
    """
    if pretrained:
        raise NotImplementedError("pretrained weights not yet on jittorhub")
    block_setting = [
        CNBlockConfig(128, 256, 3),
        CNBlockConfig(256, 512, 3),
        CNBlockConfig(512, 1024, 27),
        CNBlockConfig(1024, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return _convnext(block_setting, stochastic_depth_prob, **kwargs)


def convnext_large(pretrained=False, **kwargs):
    """ConvNeXt-Large model architecture.

    Args:

    * pretrained: If True, load pretrained weights. Default: False.
    * num_classes: Number of classes. Default: 1000.
    """
    if pretrained:
        raise NotImplementedError("pretrained weights not yet on jittorhub")
    block_setting = [
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 3),
        CNBlockConfig(768, 1536, 27),
        CNBlockConfig(1536, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return _convnext(block_setting, stochastic_depth_prob, **kwargs)

# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Jittor Group
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# MaxViT (maxvit_t), faithfully aligned with torchvision.models.maxvit.
# Reference:
#   "MaxViT: Multi-Axis Vision Transformer" https://arxiv.org/abs/2204.01697
#   torchvision.models.maxvit (MaxVit / MaxVitBlock / MaxVitLayer)
#
# Pure ``jittor.nn`` so it runs on both NVIDIA (CUDA) and Ascend (NPU) and under
# ``import jittor as torch``.
#
#     from jittor.models import maxvit_t
#     m = maxvit_t(num_classes=1000); y = m(jt.randn(2, 3, 224, 224))   # (2, 1000)
#
# A MaxViT block is the repeating unit: an MBConv (with Squeeze-Excitation) for
# local conv mixing, followed by two attention layers -- ``window`` (block-local
# 7x7 attention) and ``grid`` (sparse global / dilated attention). The grid
# attention reuses the very same window-partition op but swaps the partition and
# token axes (SwapAxes(-2, -3)) so attention is computed across a dilated grid.

import math
from collections import OrderedDict
from functools import partial

import jittor as jt
from jittor import nn

__all__ = [
    "MaxVit",
    "maxvit_t",
]


def _get_conv_output_shape(input_size, kernel_size, stride, padding):
    return (
        (input_size[0] - kernel_size + 2 * padding) // stride + 1,
        (input_size[1] - kernel_size + 2 * padding) // stride + 1,
    )


def _make_block_input_shapes(input_size, n_blocks):
    """Check that the input size is valid for a MaxViT configuration (each block
    feature map must be divisible by the partition size). Returns the per-block
    input spatial shapes after the stem stride and each block's stride-2 layer."""
    shapes = []
    block_input_shape = _get_conv_output_shape(input_size, 3, 2, 1)
    for _ in range(n_blocks):
        block_input_shape = _get_conv_output_shape(block_input_shape, 3, 2, 1)
        shapes.append(block_input_shape)
    return shapes


def _get_relative_position_index(height, width):
    """Pair-wise relative position index for the (height*width) tokens inside a
    partition, flattened to gather rows from the relative-position-bias table.
    Mirrors torchvision ``_get_relative_position_index``."""
    coords = jt.stack(jt.meshgrid(jt.arange(height), jt.arange(width), indexing="ij"))  # 2, H, W
    coords_flat = jt.flatten(coords, 1)  # 2, H*W
    relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # 2, H*W, H*W
    relative_coords = relative_coords.permute(1, 2, 0)  # H*W, H*W, 2
    relative_coords[:, :, 0] += height - 1
    relative_coords[:, :, 1] += width - 1
    relative_coords[:, :, 0] *= 2 * width - 1
    return relative_coords.sum(-1)  # H*W, H*W


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
        noise = (jt.rand(size) < survival_rate).float32()
        if survival_rate > 0.0:
            noise = noise / survival_rate
        return x * noise

    def __repr__(self):
        return "{}(p={}, mode={})".format(
            self.__class__.__name__, self.p, self.mode)


class ConvNormActivation(nn.Sequential):
    """Conv -> (Norm) -> (Activation), mirroring torchvision ``Conv2dNormActivation``.

    ``norm_layer`` / ``activation_layer`` may be ``None`` to skip that part. The
    bias of the conv defaults to ``norm_layer is None`` unless overridden."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=None, groups=1, norm_layer=nn.BatchNorm2d,
                 activation_layer=nn.ReLU, dilation=1, bias=None):
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      dilation=dilation, groups=groups, bias=bias),
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer())
        super(ConvNormActivation, self).__init__(*layers)
        self.out_channels = out_channels


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block (torchvision variant), 1x1-conv FCs, an
    activation on the reduction and a Sigmoid gate."""

    def __init__(self, input_channels, squeeze_channels,
                 activation=nn.ReLU, scale_activation=nn.Sigmoid):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
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


class MBConv(nn.Module):
    """MBConv: Mobile Inverted Residual Bottleneck (the conv mixer of a MaxViT
    layer). pre-norm -> 1x1 expand -> 3x3 depthwise (stride) -> SE -> 1x1 project,
    with stochastic depth on the residual branch. When the layer downsamples
    (stride 2) or changes width, the skip uses AvgPool + 1x1 conv. Matches
    torchvision ``MBConv``."""

    def __init__(self, in_channels, out_channels, expansion_ratio, squeeze_ratio,
                 stride, activation_layer, norm_layer, p_stochastic_dropout=0.0):
        super(MBConv, self).__init__()

        should_proj = stride != 1 or in_channels != out_channels
        if should_proj:
            proj = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                              bias=True)]
            if stride == 2:
                proj = [nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)] + proj
            self.proj = nn.Sequential(*proj)
        else:
            self.proj = nn.Identity()

        mid_channels = int(out_channels * expansion_ratio)
        sqz_channels = int(out_channels * squeeze_ratio)

        if p_stochastic_dropout:
            self.stochastic_depth = StochasticDepth(p_stochastic_dropout, mode="row")
        else:
            self.stochastic_depth = nn.Identity()

        _layers = OrderedDict()
        _layers["pre_norm"] = norm_layer(in_channels)
        _layers["conv_a"] = ConvNormActivation(
            in_channels, mid_channels, kernel_size=1, stride=1, padding=0,
            activation_layer=activation_layer, norm_layer=norm_layer)
        _layers["conv_b"] = ConvNormActivation(
            mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1,
            activation_layer=activation_layer, norm_layer=norm_layer,
            groups=mid_channels)
        _layers["squeeze_excitation"] = SqueezeExcitation(
            mid_channels, sqz_channels, activation=nn.SiLU)
        _layers["conv_c"] = nn.Conv2d(mid_channels, out_channels, kernel_size=1,
                                      bias=True)

        self.layers = nn.Sequential(_layers)

    def execute(self, x):
        """x: [B, C, H, W] -> [B, C, H/stride, W/stride]."""
        res = self.proj(x)
        x = self.stochastic_depth(self.layers(x))
        return res + x


class RelativePositionalMultiHeadAttention(nn.Module):
    """Relative-Positional Multi-Head Attention over a partitioned feature map.

    Operates on tokens of layout [B, G, P, D] (G partitions, P tokens each, D
    channels). Adds a learnable relative-position bias (shared across partitions
    and the batch) to the attention logits. Matches torchvision.
    """

    def __init__(self, feat_dim, head_dim, max_seq_len):
        super(RelativePositionalMultiHeadAttention, self).__init__()

        if feat_dim % head_dim != 0:
            raise ValueError("feat_dim: {} must be divisible by head_dim: {}".format(
                feat_dim, head_dim))

        self.n_heads = feat_dim // head_dim
        self.head_dim = head_dim
        self.size = int(math.sqrt(max_seq_len))
        self.max_seq_len = max_seq_len

        self.to_qkv = nn.Linear(feat_dim, self.n_heads * self.head_dim * 3)
        self.scale_factor = feat_dim ** -0.5

        self.merge = nn.Linear(self.head_dim * self.n_heads, feat_dim)
        self.relative_position_bias_table = jt.zeros(
            ((2 * self.size - 1) * (2 * self.size - 1), self.n_heads))

        # non-trainable buffer used to gather from the bias table
        self.relative_position_index = _get_relative_position_index(
            self.size, self.size).stop_grad()
        # initialize with truncated normal the bias
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def get_relative_positional_bias(self):
        bias_index = self.relative_position_index.reshape(-1)
        relative_bias = self.relative_position_bias_table[bias_index].reshape(
            self.max_seq_len, self.max_seq_len, -1)
        relative_bias = relative_bias.permute(2, 0, 1)  # nH, P, P
        return relative_bias.unsqueeze(0)  # 1, nH, P, P

    def execute(self, x):
        """x: [B, G, P, D] -> [B, G, P, D]."""
        B, G, P, D = x.shape
        H, DH = self.n_heads, self.head_dim

        qkv = self.to_qkv(x)
        q, k, v = jt.chunk(qkv, 3, dim=-1)

        q = q.reshape(B, G, P, H, DH).permute(0, 1, 3, 2, 4)
        k = k.reshape(B, G, P, H, DH).permute(0, 1, 3, 2, 4)
        v = v.reshape(B, G, P, H, DH).permute(0, 1, 3, 2, 4)

        k = k * self.scale_factor
        dot_prod = jt.einsum("B G H I D, B G H J D -> B G H I J", q, k)
        pos_bias = self.get_relative_positional_bias()

        dot_prod = nn.softmax(dot_prod + pos_bias, dim=-1)

        out = jt.einsum("B G H I J, B G H J D -> B G H I D", dot_prod, v)
        out = out.permute(0, 1, 3, 2, 4).reshape(B, G, P, D)

        out = self.merge(out)
        return out


class SwapAxes(nn.Module):
    """Swap two axes of a tensor (torchvision ``SwapAxes``)."""

    def __init__(self, a, b):
        super(SwapAxes, self).__init__()
        self.a = a
        self.b = b

    def execute(self, x):
        return x.transpose(self.a, self.b)


class WindowPartition(nn.Module):
    """Partition [B, C, H, W] into non-overlapping P x P windows, producing
    [B, H/P * W/P, P*P, C]. Matches torchvision ``WindowPartition``."""

    def __init__(self):
        super(WindowPartition, self).__init__()

    def execute(self, x, p):
        B, C, H, W = x.shape
        P = p
        # chunk up H and W dimensions
        x = x.reshape(B, C, H // P, P, W // P, P)
        x = x.permute(0, 2, 4, 3, 5, 1)
        # collapse P * P dimension
        x = x.reshape(B, (H // P) * (W // P), P * P, C)
        return x


class WindowDepartition(nn.Module):
    """Reverse :class:`WindowPartition`: [B, (H/P * W/P), P*P, C] -> [B, C, H, W].
    Matches torchvision ``WindowDepartition``."""

    def __init__(self):
        super(WindowDepartition, self).__init__()

    def execute(self, x, p, h_partitions, w_partitions):
        B, G, PP, C = x.shape
        P = p
        HP, WP = h_partitions, w_partitions
        # split P * P dimension into 2 P tile dimensions
        x = x.reshape(B, HP, WP, P, P, C)
        # permute into B, C, HP, P, WP, P
        x = x.permute(0, 5, 1, 3, 2, 4)
        # reshape into B, C, H, W
        x = x.reshape(B, C, HP * P, WP * P)
        return x


class PartitionAttentionLayer(nn.Module):
    """Partition the input, run relative-positional multi-head attention + an MLP
    on each partition (both pre-norm residual with stochastic depth), then reverse
    the partition. ``partition_type`` selects ``window`` (block-local) or ``grid``
    (sparse global) attention -- the latter reuses the window partition op but
    swaps the partition / token axes so attention runs across a dilated grid.
    Matches torchvision ``PartitionAttentionLayer``.
    """

    def __init__(self, in_channels, head_dim, partition_size, partition_type,
                 grid_size, mlp_ratio, activation_layer, norm_layer,
                 attention_dropout, mlp_dropout, p_stochastic_dropout):
        super(PartitionAttentionLayer, self).__init__()

        self.n_heads = in_channels // head_dim
        self.head_dim = head_dim
        self.n_partitions = grid_size[0] // partition_size
        self.partition_type = partition_type
        self.grid_size = grid_size

        if partition_type not in ["grid", "window"]:
            raise ValueError("partition_type must be either 'grid' or 'window'")

        if partition_type == "window":
            self.p, self.g = partition_size, self.n_partitions
        else:
            self.p, self.g = self.n_partitions, partition_size

        self.partition_op = WindowPartition()
        self.departition_op = WindowDepartition()
        self.partition_swap = SwapAxes(-2, -3) if partition_type == "grid" else nn.Identity()
        self.departition_swap = SwapAxes(-2, -3) if partition_type == "grid" else nn.Identity()

        self.attn_layer = nn.Sequential(
            norm_layer(in_channels),
            # it's always going to be partition_size ** 2 because of the axis swap
            # in the case of grid partitioning
            RelativePositionalMultiHeadAttention(in_channels, head_dim, partition_size ** 2),
            nn.Dropout(attention_dropout),
        )

        # pre-normalization similar to transformer layers
        self.mlp_layer = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels * mlp_ratio),
            activation_layer(),
            nn.Linear(in_channels * mlp_ratio, in_channels),
            nn.Dropout(mlp_dropout),
        )

        self.stochastic_dropout = StochasticDepth(p_stochastic_dropout, mode="row")

    def execute(self, x):
        """x: [B, C, H, W] -> [B, C, H, W]."""
        # Undefined behavior if H or W are not divisible by p.
        gh, gw = self.grid_size[0] // self.p, self.grid_size[1] // self.p
        assert (self.grid_size[0] % self.p == 0 and self.grid_size[1] % self.p == 0), (
            "Grid size must be divisible by partition size. Got grid size of {} "
            "and partition size of {}".format(self.grid_size, self.p))

        x = self.partition_op(x, self.p)
        x = self.partition_swap(x)
        x = x + self.stochastic_dropout(self.attn_layer(x))
        x = x + self.stochastic_dropout(self.mlp_layer(x))
        x = self.departition_swap(x)
        x = self.departition_op(x, self.p, gh, gw)

        return x


class MaxVitLayer(nn.Module):
    """One MaxViT layer: an MBConv followed by a ``window`` PartitionAttentionLayer
    and a ``grid`` PartitionAttentionLayer. Matches torchvision ``MaxVitLayer``."""

    def __init__(self, in_channels, out_channels, squeeze_ratio, expansion_ratio,
                 stride, norm_layer, activation_layer, head_dim, mlp_ratio,
                 mlp_dropout, attention_dropout, p_stochastic_dropout,
                 partition_size, grid_size):
        super(MaxVitLayer, self).__init__()

        layers = OrderedDict()

        # convolutional layer
        layers["MBconv"] = MBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            expansion_ratio=expansion_ratio,
            squeeze_ratio=squeeze_ratio,
            stride=stride,
            activation_layer=activation_layer,
            norm_layer=norm_layer,
            p_stochastic_dropout=p_stochastic_dropout,
        )
        # attention layers, block (window) -> grid
        layers["window_attention"] = PartitionAttentionLayer(
            in_channels=out_channels,
            head_dim=head_dim,
            partition_size=partition_size,
            partition_type="window",
            grid_size=grid_size,
            mlp_ratio=mlp_ratio,
            activation_layer=activation_layer,
            norm_layer=nn.LayerNorm,
            attention_dropout=attention_dropout,
            mlp_dropout=mlp_dropout,
            p_stochastic_dropout=p_stochastic_dropout,
        )
        layers["grid_attention"] = PartitionAttentionLayer(
            in_channels=out_channels,
            head_dim=head_dim,
            partition_size=partition_size,
            partition_type="grid",
            grid_size=grid_size,
            mlp_ratio=mlp_ratio,
            activation_layer=activation_layer,
            norm_layer=nn.LayerNorm,
            attention_dropout=attention_dropout,
            mlp_dropout=mlp_dropout,
            p_stochastic_dropout=p_stochastic_dropout,
        )
        self.layers = nn.Sequential(layers)

    def execute(self, x):
        """x: (B, C, H, W) -> (B, C, H, W)."""
        return self.layers(x)


class MaxVitBlock(nn.Module):
    """A MaxViT block: ``n_layers`` MaxVitLayers. The first layer downsamples
    (stride 2). Matches torchvision ``MaxVitBlock``."""

    def __init__(self, in_channels, out_channels, squeeze_ratio, expansion_ratio,
                 norm_layer, activation_layer, head_dim, mlp_ratio, mlp_dropout,
                 attention_dropout, partition_size, input_grid_size, n_layers,
                 p_stochastic):
        super(MaxVitBlock, self).__init__()
        if not len(p_stochastic) == n_layers:
            raise ValueError("p_stochastic must have length n_layers={}, got "
                             "p_stochastic={}.".format(n_layers, p_stochastic))

        self.layers = nn.ModuleList()
        # account for the first stride of the first layer
        self.grid_size = _get_conv_output_shape(input_grid_size, kernel_size=3,
                                                stride=2, padding=1)

        for idx, p in enumerate(p_stochastic):
            stride = 2 if idx == 0 else 1
            self.layers.append(
                MaxVitLayer(
                    in_channels=in_channels if idx == 0 else out_channels,
                    out_channels=out_channels,
                    squeeze_ratio=squeeze_ratio,
                    expansion_ratio=expansion_ratio,
                    stride=stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    mlp_dropout=mlp_dropout,
                    attention_dropout=attention_dropout,
                    partition_size=partition_size,
                    grid_size=self.grid_size,
                    p_stochastic_dropout=p,
                )
            )

    def execute(self, x):
        """x: (B, C, H, W) -> (B, C, H, W)."""
        for layer in self.layers:
            x = layer(x)
        return x


class MaxVit(nn.Module):
    """MaxViT, aligned with torchvision.

    Args:

    * input_size: Size of the input image (H, W). Used to validate divisibility.
    * stem_channels: Number of channels in the stem.
    * partition_size: Size of the partitions (window/grid). Default uses 7.
    * block_channels: Number of channels in each of the 4 blocks.
    * block_layers: Number of layers in each of the 4 blocks.
    * head_dim: Dimension of each attention head.
    * stochastic_depth_prob: Max stochastic-depth drop probability (linearly
      ramped across all layers). Default: 0.2.
    * norm_layer: Normalization module for conv layers. If None, use
      ``BatchNorm2d(eps=1e-3, momentum=0.01)``.
    * activation_layer: Activation for conv and transformer layers. Default: GELU.
    * squeeze_ratio: Squeeze ratio in the SE layer. Default: 0.25.
    * expansion_ratio: Expansion ratio in the MBConv bottleneck. Default: 4.
    * mlp_ratio: Expansion ratio of the attention MLP. Default: 4.
    * mlp_dropout: Dropout for the MLP. Default: 0.0.
    * attention_dropout: Dropout for the attention layer. Default: 0.0.
    * num_classes: Number of classes. Default: 1000.
    """

    def __init__(self, input_size, stem_channels, partition_size, block_channels,
                 block_layers, head_dim, stochastic_depth_prob, norm_layer=None,
                 activation_layer=nn.GELU, squeeze_ratio=0.25, expansion_ratio=4,
                 mlp_ratio=4, mlp_dropout=0.0, attention_dropout=0.0,
                 num_classes=1000):
        super(MaxVit, self).__init__()

        input_channels = 3

        # exact batchnorm parameters from the reference google-research impl
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)

        # Make sure input size will be divisible by the partition size in all
        # blocks (undefined behavior otherwise).
        block_input_sizes = _make_block_input_shapes(input_size, len(block_channels))
        for idx, block_input_size in enumerate(block_input_sizes):
            if block_input_size[0] % partition_size != 0 or block_input_size[1] % partition_size != 0:
                raise ValueError(
                    "Input size {} of block {} is not divisible by partition size "
                    "{}. Consider changing the partition size or the input size.\n"
                    "Current configuration yields the following block input sizes: "
                    "{}.".format(block_input_size, idx, partition_size,
                                 block_input_sizes))

        # stem: 3x3 conv stride 2 (norm + act) then 3x3 conv stride 1 (bias only).
        self.stem = nn.Sequential(
            ConvNormActivation(
                input_channels, stem_channels, 3, stride=2, norm_layer=norm_layer,
                activation_layer=activation_layer, bias=False),
            ConvNormActivation(
                stem_channels, stem_channels, 3, stride=1, norm_layer=None,
                activation_layer=None, bias=True),
        )

        # account for stem stride
        input_size = _get_conv_output_shape(input_size, kernel_size=3, stride=2,
                                            padding=1)
        self.partition_size = partition_size

        # blocks
        self.blocks = nn.ModuleList()
        in_channels = [stem_channels] + block_channels[:-1]
        out_channels = block_channels

        # stochastic depth probabilities linearly spaced over [0, prob], one per
        # layer across all blocks.
        total_layers = sum(block_layers)
        if total_layers > 1:
            p_stochastic = [stochastic_depth_prob * i / (total_layers - 1)
                            for i in range(total_layers)]
        else:
            p_stochastic = [0.0] * total_layers

        p_idx = 0
        for in_channel, out_channel, num_layers in zip(in_channels, out_channels,
                                                       block_layers):
            self.blocks.append(
                MaxVitBlock(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    squeeze_ratio=squeeze_ratio,
                    expansion_ratio=expansion_ratio,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    mlp_dropout=mlp_dropout,
                    attention_dropout=attention_dropout,
                    partition_size=partition_size,
                    input_grid_size=input_size,
                    n_layers=num_layers,
                    p_stochastic=p_stochastic[p_idx: p_idx + num_layers],
                )
            )
            input_size = self.blocks[-1].grid_size
            p_idx += num_layers

        # head: avgpool -> LN -> Linear -> Tanh -> Linear (Linear -> Tanh -> Linear
        # follows the google-research reference).
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(block_channels[-1]),
            nn.Linear(block_channels[-1], block_channels[-1]),
            nn.Tanh(),
            nn.Linear(block_channels[-1], num_classes, bias=False),
        )

        self._init_weights()

    def execute(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.gauss_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zero_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.gauss_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zero_(m.bias)


def _maxvit(stem_channels, block_channels, block_layers, stochastic_depth_prob,
            partition_size, head_dim, **kwargs):
    input_size = kwargs.pop("input_size", (224, 224))
    model = MaxVit(
        stem_channels=stem_channels,
        block_channels=block_channels,
        block_layers=block_layers,
        stochastic_depth_prob=stochastic_depth_prob,
        head_dim=head_dim,
        partition_size=partition_size,
        input_size=input_size,
        **kwargs,
    )
    return model


def maxvit_t(pretrained=False, **kwargs):
    """MaxViT ``maxvit_t`` (tiny) architecture.

    block_layers=(2, 2, 5, 2), block_channels=(64, 128, 256, 512),
    stem_channels=64, head_dim=32, partition_size=7.

    Args:

    * pretrained: If True, load pretrained weights. Default: False.
    * num_classes: Number of classes. Default: 1000.
    """
    if pretrained:
        raise NotImplementedError("pretrained weights not yet on jittorhub")
    return _maxvit(
        stem_channels=64,
        block_channels=[64, 128, 256, 512],
        block_layers=[2, 2, 5, 2],
        head_dim=32,
        stochastic_depth_prob=0.2,
        partition_size=7,
        **kwargs,
    )

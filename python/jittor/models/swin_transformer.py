# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Jittor Group
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# Swin Transformer (V1: swin_t / swin_s / swin_b), faithfully aligned with
# torchvision.models.swin_transformer.
# Reference:
#   "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
#   https://arxiv.org/abs/2103.14030
#   torchvision.models.swin_transformer (SwinTransformer / SwinTransformerBlock)
#
# Pure ``jittor.nn`` so it runs on both NVIDIA (CUDA) and Ascend (NPU) and under
# ``import jittor as torch``.
#
#     from jittor.models import swin_t
#     m = swin_t(num_classes=1000); y = m(jt.randn(2, 3, 224, 224))   # (2, 1000)

import math

import jittor as jt
from jittor import nn

__all__ = [
    "SwinTransformer",
    "swin_t", "swin_s", "swin_b",
]


class Permute(nn.Module):
    """Permute the dimensions of the input tensor (torchvision ``ops.Permute``)."""

    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims

    def execute(self, x):
        return x.permute(self.dims)


class MLP(nn.Module):
    """torchvision ``ops.MLP`` (the subset used by Swin): Linear -> GELU -> drop
    -> Linear -> drop, all with bias."""

    def __init__(self, in_dim, hidden_dim, dropout=0.0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, in_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def execute(self, x):
        x = self.dropout(nn.gelu(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x


def _patch_merging_pad(x):
    """Pad (B, H, W, C) so H and W are even, then gather the four 2x2 phases and
    concatenate them on the channel axis -> (B, H/2, W/2, 4C). Matches
    torchvision ``_patch_merging_pad``."""
    H, W, _ = x.shape[-3:]
    # F.pad on the last 3 dims; pad order is (C_l, C_r, W_l, W_r, H_l, H_r).
    x = nn.pad(x, (0, 0, 0, W % 2, 0, H % 2))
    x0 = x[..., 0::2, 0::2, :]   # ... H/2 W/2 C
    x1 = x[..., 1::2, 0::2, :]
    x2 = x[..., 0::2, 1::2, :]
    x3 = x[..., 1::2, 1::2, :]
    x = jt.concat([x0, x1, x2, x3], dim=-1)   # ... H/2 W/2 4*C
    return x


class PatchMerging(nn.Module):
    """Patch-merging downsample layer (V1): even-pad, gather 2x2 -> 4C, LayerNorm
    on 4C, then a bias-free Linear 4C -> 2C. Order (norm before reduction) matches
    torchvision."""

    def __init__(self, dim, norm_layer=None):
        super(PatchMerging, self).__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def execute(self, x):
        """x: (B, H, W, C) -> (B, H/2, W/2, 2*C)."""
        x = _patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)
        return x


def _get_relative_position_bias(relative_position_bias_table,
                                relative_position_index, window_size):
    """Gather the (Wh*Ww, Wh*Ww, nHeads) relative position bias from the learnable
    table and reshape to (nHeads, Wh*Ww, Wh*Ww). Matches torchvision."""
    N = window_size[0] * window_size[1]
    relative_position_bias = relative_position_bias_table[relative_position_index]  # N*N? no -> (N*N, nH)
    relative_position_bias = relative_position_bias.reshape(N, N, -1)
    # (nH, N, N), contiguous
    relative_position_bias = relative_position_bias.permute(2, 0, 1)
    return relative_position_bias


def shifted_window_attention(input, qkv_weight, proj_weight, relative_position_bias,
                             window_size, num_heads,
                             shift_size, attention_dropout=0.0, dropout=0.0,
                             qkv_bias=None, proj_bias=None, training=True):
    """Window / shifted-window multi-head self attention (functional form, mirrors
    torchvision ``shifted_window_attention``).

    input: (B, H, W, C). Pads (H, W) up to a multiple of the window size, optionally
    rolls (cyclic shift) before partitioning into windows, runs scaled dot-product
    attention with the precomputed relative position bias (and, for shifted windows,
    an additive attention mask), then reverses partition / roll / pad.
    """
    B, H, W, C = input.shape
    # pad feature maps to multiples of window size
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    # F.pad pads last dims; order (C_l, C_r, W_l, W_r, H_l, H_r).
    x = nn.pad(input, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, _ = x.shape

    shift_size = list(shift_size)
    # If window size is larger than feature size, there is no need to shift the window.
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    # cyclic shift
    if sum(shift_size) > 0:
        x = jt.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    # partition windows
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = x.reshape(B, pad_H // window_size[0], window_size[0],
                  pad_W // window_size[1], window_size[1], C)
    # B, nWh, nWw, Wh, Ww, C  ->  (B*nW, Wh*Ww, C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows,
                                            window_size[0] * window_size[1], C)

    # multi-head attention
    qkv = nn.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.shape[0], x.shape[1], 3, num_heads,
                      C // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q * (C // num_heads) ** -0.5
    attn = jt.matmul(q, k.transpose(0, 1, 3, 2))
    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask
        attn_mask = jt.zeros((pad_H, pad_W), dtype=x.dtype)
        h_slices = ((0, -window_size[0]),
                    (-window_size[0], -shift_size[0]),
                    (-shift_size[0], None))
        w_slices = ((0, -window_size[1]),
                    (-window_size[1], -shift_size[1]),
                    (-shift_size[1], None))
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0]:h[1], w[0]:w[1]] = count
                count += 1
        attn_mask = attn_mask.reshape(pad_H // window_size[0], window_size[0],
                                      pad_W // window_size[1], window_size[1])
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(
            num_windows, window_size[0] * window_size[1])
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        # nonzero entries -> -100 (masked), zero entries -> 0 (kept)
        attn_mask = (attn_mask != 0).float32() * (-100.0)
        attn = attn.reshape(x.shape[0] // num_windows, num_windows, num_heads,
                            x.shape[1], x.shape[1])
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.reshape(-1, num_heads, x.shape[1], x.shape[1])

    attn = nn.softmax(attn, dim=-1)
    attn = nn.dropout(attn, p=attention_dropout, is_train=training)

    x = jt.matmul(attn, v).transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], C)
    x = nn.linear(x, proj_weight, proj_bias)
    x = nn.dropout(x, p=dropout, is_train=training)

    # reverse windows
    x = x.reshape(B, pad_H // window_size[0], pad_W // window_size[1],
                  window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = jt.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    # unpad features
    x = x[:, :H, :W, :]
    return x


class ShiftedWindowAttention(nn.Module):
    """See :func:`shifted_window_attention`. Holds the qkv/proj Linears and the
    learnable relative position bias table + (buffer) index. V1 layout."""

    def __init__(self, dim, window_size, shift_size, num_heads,
                 qkv_bias=True, proj_bias=True, attention_dropout=0.0, dropout=0.0):
        super(ShiftedWindowAttention, self).__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.window_size = list(window_size)
        self.shift_size = list(shift_size)
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.define_relative_position_bias_table()
        self.define_relative_position_index()

    def define_relative_position_bias_table(self):
        # define a parameter table of relative position bias
        self.relative_position_bias_table = jt.zeros(
            ((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
             self.num_heads))   # (2*Wh-1 * 2*Ww-1, nH)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def define_relative_position_index(self):
        # get pair-wise relative position index for each token inside the window
        coords_h = jt.arange(self.window_size[0])
        coords_w = jt.arange(self.window_size[1])
        coords = jt.stack(jt.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = jt.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0)  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww
        # keep as a non-trainable buffer (int) used to gather from the table
        self.relative_position_index = relative_position_index.stop_grad()

    def get_relative_position_bias(self):
        return _get_relative_position_bias(
            self.relative_position_bias_table,
            self.relative_position_index, self.window_size)

    def execute(self, x):
        """x: (B, H, W, C) -> (B, H, W, C)."""
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            training=self.is_training(),
        )


class SwinTransformerBlock(nn.Module):
    """One Swin block: (shifted) window attention + MLP, both pre-norm residual
    with stochastic depth. Matches torchvision ``SwinTransformerBlock``."""

    def __init__(self, dim, num_heads, window_size, shift_size,
                 mlp_ratio=4.0, dropout=0.0, attention_dropout=0.0,
                 stochastic_depth_prob=0.0, norm_layer=None,
                 attn_layer=ShiftedWindowAttention):
        super(SwinTransformerBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm

        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim, window_size, shift_size, num_heads,
            attention_dropout=attention_dropout, dropout=dropout)
        self.stochastic_depth = nn.DropPath(stochastic_depth_prob)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=dropout)

    def execute(self, x):
        # is_training() drives DropPath (it caches self.is_train at construction).
        self.stochastic_depth.is_train = self.is_training()
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        return x


class SwinTransformer(nn.Module):
    """Swin Transformer (V1), aligned with torchvision.

    Args:

    * patch_size: Patch size of the first conv embedding (V1 uses (4, 4)).
    * embed_dim: Patch-embedding dimension (channels after the stem).
    * depths: Number of SwinTransformerBlocks in each of the 4 stages.
    * num_heads: Number of attention heads in each stage.
    * window_size: Window size (V1 uses (7, 7)).
    * mlp_ratio: Expansion ratio of the MLP hidden dim. Default: 4.0.
    * dropout: Dropout rate. Default: 0.0.
    * attention_dropout: Attention dropout rate. Default: 0.0.
    * stochastic_depth_prob: Max stochastic-depth drop probability. Default: 0.1.
    * num_classes: Number of classes. Default: 1000.
    * norm_layer: Normalization module. If None, use ``nn.LayerNorm``.
    * block: SwinTransformerBlock building block. If None, use the default.
    """

    def __init__(self, patch_size, embed_dim, depths, num_heads, window_size,
                 mlp_ratio=4.0, dropout=0.0, attention_dropout=0.0,
                 stochastic_depth_prob=0.1, num_classes=1000,
                 norm_layer=None, block=None, downsample_layer=PatchMerging):
        super(SwinTransformer, self).__init__()
        self.num_classes = num_classes

        if block is None:
            block = SwinTransformerBlock
        if norm_layer is None:
            norm_layer = nn.LayerNorm

        layers = []
        # split image into non-overlapping patches (stem): Conv -> NHWC -> norm.
        layers.append(nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=(patch_size[0], patch_size[1]),
                      stride=(patch_size[0], patch_size[1])),
            Permute([0, 2, 3, 1]),   # B C H W -> B H W C
            norm_layer(embed_dim),
        ))

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage = []
            dim = embed_dim * 2 ** i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the
                # stage block
                sd_prob = (stochastic_depth_prob * float(stage_block_id) /
                           (total_stage_blocks - 1)) if total_stage_blocks > 1 else 0.0
                stage.append(block(
                    dim,
                    num_heads[i_stage],
                    window_size=window_size,
                    # even blocks: no shift; odd blocks: shift by half a window.
                    shift_size=[0 if i_layer % 2 == 0 else w // 2
                                for w in window_size],
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    stochastic_depth_prob=sd_prob,
                    norm_layer=norm_layer,
                ))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # add patch merging layer (between stages, not after the last one)
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer))
        self.features = nn.Sequential(*layers)

        num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(num_features)
        self.permute = Permute([0, 3, 1, 2])  # B H W C -> B C H W
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.head = nn.Linear(num_features, num_classes)

        # weight init (torchvision: trunc_normal_ on Linear weights, zero bias).
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def execute(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.head(x)
        return x


def _swin_transformer(patch_size, embed_dim, depths, num_heads, window_size,
                      stochastic_depth_prob, **kwargs):
    model = SwinTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        stochastic_depth_prob=stochastic_depth_prob,
        **kwargs,
    )
    return model


def swin_t(pretrained=False, **kwargs):
    """Swin Transformer ``swin_t`` (tiny) architecture.

    depths=(2, 2, 6, 2), heads=(3, 6, 12, 24), embed_dim=96, window=7.

    Args:

    * pretrained: If True, load pretrained weights. Default: False.
    * num_classes: Number of classes. Default: 1000.
    """
    if pretrained:
        raise NotImplementedError("pretrained weights not yet on jittorhub")
    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stochastic_depth_prob=0.2,
        **kwargs,
    )


def swin_s(pretrained=False, **kwargs):
    """Swin Transformer ``swin_s`` (small) architecture.

    depths=(2, 2, 18, 2), heads=(3, 6, 12, 24), embed_dim=96, window=7.

    Args:

    * pretrained: If True, load pretrained weights. Default: False.
    * num_classes: Number of classes. Default: 1000.
    """
    if pretrained:
        raise NotImplementedError("pretrained weights not yet on jittorhub")
    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stochastic_depth_prob=0.3,
        **kwargs,
    )


def swin_b(pretrained=False, **kwargs):
    """Swin Transformer ``swin_b`` (base) architecture.

    depths=(2, 2, 18, 2), heads=(4, 8, 16, 32), embed_dim=128, window=7.

    Args:

    * pretrained: If True, load pretrained weights. Default: False.
    * num_classes: Number of classes. Default: 1000.
    """
    if pretrained:
        raise NotImplementedError("pretrained weights not yet on jittorhub")
    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[7, 7],
        stochastic_depth_prob=0.5,
        **kwargs,
    )

"""Functional neural-network operations with explicit public exports."""

# ruff: noqa: F401

from jittor.pool import max_pool2d, max_pool3d, pool, pool3d

from ..attention import (
    cumulative_sequence_lengths,
    sequence_lengths,
    varlen_scaled_dot_product_attention,
)
from ..dual_grid import finalize_dual_grid_mesh_cuda
from ..packed_qkv_cuda import packed_qkv_rms_rope_cuda
from ..rms_norm_cuda import multihead_rms_norm_cuda
from ..rope_cuda import partial_rotary_embedding_cuda
from jittor.sparse.convolution import (
    build_submanifold_conv3d_neighbors,
    submanifold_conv3d,
)
from ..utils import skip_init
from .attention import scaled_dot_product_attention
from .activation import (
    elu,
    gelu,
    get_init_var_rand,
    hardsigmoid,
    hardswish,
    hardtanh,
    leaky_relu,
    mish,
    prelu,
    relu,
    relu6,
    rrelu,
    sigmoid,
    sign,
    silu,
    softplus,
)
from .autograd import backward
from .complex import polar, view_as_complex, view_as_real
from .convolution import conv1d, conv2d, conv3d
from .convolution_transpose import conv_transpose, conv_transpose1d, conv_transpose3d
from .dropout import dropout, dropout2d, droppath
from .embedding import embedding, embedding_bag
from .fold import fold, unfold
from .grid import (
    affine_grid,
    affine_grid_generator_4D,
    affine_grid_generator_5D,
    clip_coordinates,
    grid_sample,
    grid_sample_v0,
    grid_sampler,
    grid_sampler_2d,
    grid_sampler_3d,
    grid_sampler_compute_source_index,
    grid_sampler_unnormalize,
    linspace_from_neg_one,
    make_base_grid_4D,
    make_base_grid_5D,
    reflect_coordinates,
)
from .interpolation import interpolate, resize
from .linear import linear
from .loss import (
    bce_loss,
    binary_cross_entropy,
    binary_cross_entropy_with_logits,
    cosine_embedding_loss,
    cross_entropy,
    cross_entropy_loss,
    gaussian_nll_loss,
    huber_loss,
    kl_div,
    l1_loss,
    margin_ranking_loss,
    mse_loss,
    nll_loss,
    smooth_l1_loss,
)
from .matrix import baddbmm, bilinear, bmm, bmm_transpose, matmul, matmul_transpose
from .multihead_attention import multi_head_attention_forward
from .normalization import (
    _ln_function_cls,
    _ln_normalize,
    batch_norm,
    fp32_guard,
    group_norm,
    instance_norm,
    layer_norm,
)
from .padding import pad
from .pooling import adaptive_avg_pool2d, avg_pool2d
from .shape import flatten, identity
from .softmax import _get_softmax_dim, log_sigmoid, log_softmax, logsumexp, softmax
from .tensor import kron, one_hot, tensordot
from .vector import cosine_similarity, glu, normalize, pairwise_distance, softsign


conv = conv2d
conv_transpose2d = conv_transpose
pool2d = pool
upsample = resize


__all__ = sorted(
    (
        "adaptive_avg_pool2d",
        "affine_grid",
        "affine_grid_generator_4D",
        "affine_grid_generator_5D",
        "avg_pool2d",
        "backward",
        "baddbmm",
        "batch_norm",
        "bce_loss",
        "binary_cross_entropy",
        "bilinear",
        "binary_cross_entropy_with_logits",
        "bmm",
        "bmm_transpose",
        "build_submanifold_conv3d_neighbors",
        "clip_coordinates",
        "conv",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "cosine_similarity",
        "cosine_embedding_loss",
        "cross_entropy",
        "cross_entropy_loss",
        "cumulative_sequence_lengths",
        "dropout",
        "dropout2d",
        "droppath",
        "elu",
        "embedding",
        "embedding_bag",
        "finalize_dual_grid_mesh_cuda",
        "flatten",
        "fold",
        "gaussian_nll_loss",
        "fp32_guard",
        "gelu",
        "get_init_var_rand",
        "glu",
        "grid_sample",
        "grid_sample_v0",
        "grid_sampler",
        "grid_sampler_2d",
        "grid_sampler_3d",
        "grid_sampler_compute_source_index",
        "grid_sampler_unnormalize",
        "group_norm",
        "hardsigmoid",
        "hardswish",
        "hardtanh",
        "huber_loss",
        "identity",
        "instance_norm",
        "interpolate",
        "kron",
        "kl_div",
        "l1_loss",
        "layer_norm",
        "leaky_relu",
        "linear",
        "linspace_from_neg_one",
        "log_sigmoid",
        "log_softmax",
        "logsumexp",
        "margin_ranking_loss",
        "make_base_grid_4D",
        "make_base_grid_5D",
        "matmul",
        "matmul_transpose",
        "max_pool2d",
        "max_pool3d",
        "mish",
        "mse_loss",
        "multi_head_attention_forward",
        "multihead_rms_norm_cuda",
        "nll_loss",
        "normalize",
        "one_hot",
        "pad",
        "pairwise_distance",
        "packed_qkv_rms_rope_cuda",
        "partial_rotary_embedding_cuda",
        "polar",
        "pool",
        "pool2d",
        "pool3d",
        "prelu",
        "reflect_coordinates",
        "relu",
        "relu6",
        "resize",
        "rrelu",
        "scaled_dot_product_attention",
        "sequence_lengths",
        "sigmoid",
        "sign",
        "silu",
        "skip_init",
        "smooth_l1_loss",
        "softmax",
        "softplus",
        "softsign",
        "submanifold_conv3d",
        "tensordot",
        "unfold",
        "upsample",
        "varlen_scaled_dot_product_attention",
        "view_as_complex",
        "view_as_real",
    )
)

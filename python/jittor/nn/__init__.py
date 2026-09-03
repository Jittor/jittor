"""Neural-network facade composed from stable physical implementation modules."""

# ruff: noqa: F401, F403

from abc import abstractmethod
import collections
from collections import OrderedDict
from functools import partial
import math
import os

import jittor as jt
from jittor import Module, init
import numpy as np

from jittor.misc import CTCLoss, _pair, _triple
from jittor.optim import *
from jittor.pool import (
    AdaptiveAvgPool1d,
    AdaptiveAvgPool3d,
    AdaptiveMaxPool2d,
    AdaptiveMaxPool3d,
    AvgPool1d,
    AvgPool3d,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
    MaxUnpool2d,
    MaxUnpool3d,
    Pool,
    Pool3d,
    max_pool2d,
    max_pool3d,
    pool,
    pool2d,
    pool3d,
    pool_use_code_op,
)
from jittor_utils import LOG

from . import backends as backends
from . import functional as functional
from . import modules as modules
from .attention import (
    cumulative_sequence_lengths,
    sequence_lengths,
    varlen_scaled_dot_product_attention,
)
from .fused_moe import fused_moe
from .paged_attention import paged_attention, reshape_and_cache
from .serving_ops import (dual_rms_norm, fused_add_rms_norm, rms_norm,
                          rotary_embedding, silu_and_mul)
from .backends.cudnn import (
    _CUDNN_3D_HALF_DTYPES,
    _cudnn_conv3d_fp16_safe,
    _try_cudnn_conv2d,
    _try_cudnn_conv_transpose2d,
)
from .backends.batch_norm_training_cuda import _batch_norm_cuda, _batch_norm_eval_cuda
from .backends.channel_bias_cuda import _channel_bias_add_cuda
from .backends.layer_norm_cuda import _layer_norm_no_grad_cuda
from .backends.layer_norm_training_cuda import _layer_norm_cuda
from .backends.group_norm_cuda import _group_norm_cuda
from .backends.rms_norm_training_cuda import _rms_norm_training_cuda
from .dual_grid import finalize_dual_grid_mesh_cuda
from .functional import *
from .functional.complex import (
    _Complex64ToReal2,
    _Real2ToComplex64,
    _complex64_imag_unit,
    _complex64_imag_unit_cache,
    _complex64_to_real2,
    _complex64_to_real2_raw,
    _real2_to_complex64,
    _real2_to_complex64_raw,
    _var_angle,
    _var_imag,
    _var_real,
)
from .functional.interpolation import _bicubic, _interpolate
from .functional.matrix import (
    _broadcast_batch_dims,
    _matmul_2d_cublas,
    _transpose_base_last2,
)
from .functional.normalization import _ln_function_cls, _ln_normalize
from .functional.softmax import _get_softmax_dim
from .legacy_complex import ComplexNumber, _fft2
from .backends.modulated_layer_norm_cuda import _modulated_layer_norm_no_grad_cuda
from .modules import *
from .packed_qkv_cuda import packed_qkv_rms_rope_cuda
from .rms_norm_cuda import _rms_norm_cuda, multihead_rms_norm_cuda
from .rope_cuda import partial_rotary_embedding_cuda
from jittor.sparse.convolution import (
    build_submanifold_conv3d_neighbors,
    submanifold_conv3d,
)
from .utils import skip_init
from ._bindings import install_var_bindings as _install_var_bindings
from .._install_order import record as _record_install


_install_var_bindings()
# See jittor/_install_order.py: the order these patches run in is the contract.
_record_install("nn.var_bindings")
del _install_var_bindings
del _record_install

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

from jittor.misc import CTCLoss
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
from .dual_grid import finalize_dual_grid_mesh_cuda
from .functional import *
from .legacy_complex import ComplexNumber
from .modules import *
from .packed_qkv_cuda import packed_qkv_rms_rope_cuda
from .rms_norm_cuda import multihead_rms_norm_cuda
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

# Importing a private submodule attaches it to its parent package. These are
# implementation details used during facade construction, not facade exports.
globals().pop("_bindings", None)
globals().pop("_cuda_inference", None)

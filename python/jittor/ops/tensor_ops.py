"""Ordered tensor API composition; implementations live in domain owners."""

from jittor_core import Var, ops as _native_ops
import numpy as np
import math
import builtins as _builtins
from collections.abc import Sequence,Iterable
from .. import _arg_policy
from .._runtime.dispatch import dispatch_context, optional_kernel, register_kernel, select_kernel, try_dispatch

# Bootstrap registers CUDA kernels explicitly before composing this facade.
# These historical private attributes are same-object references, not wrappers.
def __getattr__(name):
    if name == "_cuda_ctc":
        from jittor.backends.cuda.kernels.misc import ctc
        return ctc
    if name == "_cuda_codegen":
        from jittor.backends.cuda.kernels.misc import codegen
        return codegen
    if name in ("_cuda_tensor_ops", "_repeat_interleave_dim0_cuda",
                "_stack_no_grad_cuda_fast", "_unbind_no_grad_cuda_fast",
                "_unique_code_cuda", "_scan_2d_cuda"):
        from jittor.backends.cuda.kernels.misc import tensor_ops
        return tensor_ops if name == "_cuda_tensor_ops" else getattr(tensor_ops, name)
    raise AttributeError(name)


from .shape_ops import _repeat_interleave_cpu_source
from .shape_ops import _stack_cpu_source
from .shape_ops import _unbind_cpu_source
from .geometry import knn
from .advanced_indexing import index_add_
Var.index_add_ = index_add_
from .tensor_protocol import __copy__
Var.__copy__ = __copy__
from .tensor_protocol import __deepcopy__
Var.__deepcopy__ = __deepcopy__
from .tensor_protocol import __len__
Var.__len__ = __len__
from .tensor_protocol import __iter__
Var.__iter__ = __iter__
from .tensor_protocol import __contains__
Var.__contains__ = __contains__
from .tensor_protocol import new
Var.new = new
from .tensor_protocol import __index__
Var.__index__ = __index__
from .sorting import sort
Var.sort = sort
from .numerical import all
Var.all = all
from .numerical import any
Var.any = any
from .random import bernoulli
from .shape_transforms import chunk, expand, repeat
Var.repeat = repeat
ne = Var.ne = Var.not_equal
from .shape_ops import repeat_interleave
Var.repeat_interleave = repeat_interleave
Var.chunk = chunk
Var.expand = expand
from .shape_ops import t
Var.t = t
from .sorting import median
Var.median = median
from .shape_ops import stack
Var.stack = stack
from .shape_ops import flip
Var.flip = flip
from .geometry import cross
Var.cross = cross
from .numerical import normalize
Var.normalize = normalize
from .shape_ops import unbind
Var.unbind = unbind
from .image import make_grid
from .image import save_image
from .tuples import _ntuple
from .tuples import _single
from .tuples import _pair
from .tuples import _triple
from .tuples import _quadruple
from .sorting import _unique_code_generic
register_kernel("misc.unique_code", "*", _unique_code_generic)
from .sorting import unique
Var.unique = unique
from .sorting import unique_consecutive
Var.unique_consecutive = unique_consecutive
from .numerical import hypot
from .numerical import _DEGREES_PER_RADIAN
from .numerical import _RADIANS_PER_DEGREE
from .numerical import rad2deg
Var.rad2deg = rad2deg
from .numerical import deg2rad
Var.deg2rad = deg2rad
from .numerical import arctan2
from .numerical import atan2
from .advanced_indexing import nonzero
Var.nonzero = nonzero
from .random import arange
from .numerical import log2
Var.log2 = log2
from .shape_ops import meshgrid
from .shape_ops import _split_slice
from .shape_ops import _split_slice_acl
register_kernel("misc.split_slice", "*", _split_slice)
register_kernel("misc.split_slice", "acl", _split_slice_acl)
from .shape_ops import split
Var.split = split
from .tensor_protocol import tolist
Var.tolist = tolist
from .shape_ops import view_as
Var.view_as = view_as
Var.reshape_as = view_as
from .shape_ops import diag
from .shape_ops import diagonal
Var.diag = diag
from .sorting import topk
Var.topk = topk
from . import sorting as _sorting
_sorting._kthvalue_native_argsort = _native_ops.argsort
from .sorting import _kthvalue_native_argsort
from .sorting import _kthvalue_argsort
register_kernel("misc.kthvalue_argsort", "*", _kthvalue_argsort)
register_kernel("misc.kthvalue_argsort", "acl", _kthvalue_native_argsort)
from .sorting import kthvalue
Var.kthvalue = kthvalue
from .scan import _prod
from .scan import numpy_cumsum
from .scan import cub_cumsum
Var.cub_cumsum = cub_cumsum
from .scan import _cumsum_dim
from .scan import _scan_2d
from .scan import _scan_2d_cpu
register_kernel("misc.scan_2d", "cpu", _scan_2d_cpu)
from .scan import _Cumsum
from .scan import cumsum
Var.cumsum = cumsum
from .scan import cumprod
Var.cumprod=cumprod
import collections as _collections
from .scan import _CumMax
from .scan import _CumMin
from .scan import _cummax_min
from .scan import cummax
from .scan import cummin
Var.cummax = cummax
Var.cummin = cummin
from .geometry import nms
Var.expand_as = Var.broadcast_var
from .advanced_indexing import index_fill_
from .advanced_indexing import index_fill
Var.index_fill_ = index_fill_
Var.index_fill = index_fill
from .diagnostics import print_tree
from .diagnostics import get_max_memory_treemap
from .codegen import python_pass_wrapper
from .codegen import auto_parallel
from .scan import numpy_cumprod
from .random import linspace
from .random import randperm
from .random import set_global_seed
import time
from .random import _seed_jittor_at_import
_seed_jittor_at_import()
from .search import _searchsorted_acl
from .search import searchsorted
from .advanced_indexing import _scatter_into
from .advanced_indexing import scatter
from .advanced_indexing import scatter_
Var.scatter = scatter
Var.scatter_ = scatter_
from .advanced_indexing import scatter_add
from .advanced_indexing import scatter_add_
Var.scatter_add = scatter_add
Var.scatter_add_ = scatter_add_
from .advanced_indexing import _SCATTER_REDUCE_JT
from .advanced_indexing import _segment_reduce
from .advanced_indexing import scatter_reduce
Var.scatter_reduce = scatter_reduce
from .advanced_indexing import index_add
Var.index_add = index_add
from .advanced_indexing import gather
Var.gather = gather
from .shape_ops import roll
Var.roll = roll
from .numerical import safe_log
Var.safe_log = safe_log
from .ctc import _CTCLossFunction
from .ctc import ctc_loss
from .ctc import CTCLoss
from .numerical import _simple_for
from .numerical import _isnan_acl
from .numerical import _isinf_acl
from .numerical import _isfinite_acl
from .numerical import _classify_value
from .numerical import _classify
from .numerical import _classify_acl
from .numerical import _classify_code
register_kernel("misc.classify", "*", _classify_code)
register_kernel("misc.classify", "acl", _classify_acl)
from .numerical import isnan
Var.isnan = isnan
from .numerical import isfinite
Var.isfinite = isfinite
from .numerical import isinf
Var.isinf = isinf
from .numerical import isneginf
Var.isneginf = isneginf
from .numerical import isposinf
Var.isposinf = isposinf
from .tensor_protocol import contiguous
Var.contiguous = contiguous
from .tensor_protocol import cpu
Var.cpu = cpu
from .tensor_protocol import _device_spec
from .tensor_protocol import _dtype_spec
from .tensor_protocol import _parse_to
from .tensor_protocol import to
Var.to = to
from .numerical import rsqrt
Var.rsqrt = rsqrt
from .tensor_protocol import from_torch
from .shape_ops import triu
Var.triu = triu
Var.triu_ = lambda x,diagonal=0: x.assign(x.triu(diagonal))
from .shape_ops import tril
Var.tril = tril
Var.tril_ = lambda x: x.assign(x.tril())
from .numerical import all_equal
from .numerical import _to_float
Var._to_float = _to_float
from .selection import _index_select_acl
from .selection import index_select
Var.index_select = index_select
from .random import multinomial
from .random import histc
from .tensor_protocol import peek_s
from .tensor_protocol import peek
from .numerical import Finfo
from .numerical import bfloat16_finfo
bfloat16_finfo.min = -1e38
bfloat16_finfo.max = 1e38
from .numerical import finfo
from .numerical import iinfo
from .tensor_protocol import _accelerator_index
from .tensor_protocol import cuda
Var.cuda = cuda
from .tensor_protocol import npu
Var.npu = npu
from .numerical import expm1
from .advanced_indexing import isin
from .shape_composition import (
    atleast_1d, atleast_2d, atleast_3d, block_diag, cartesian_prod,
)
__all__ = [
    "CTCLoss",
    "Finfo",
    "all",
    "all_equal",
    "any",
    "arange",
    "arctan2",
    "atan2",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "auto_parallel",
    "bernoulli",
    "bfloat16_finfo",
    "block_diag",
    "cartesian_prod",
    "chunk",
    "contiguous",
    "cpu",
    "cross",
    "ctc_loss",
    "cub_cumsum",
    "cuda",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "deg2rad",
    "diag",
    "diagonal",
    "expand",
    "expm1",
    "finfo",
    "flip",
    "from_torch",
    "gather",
    "get_max_memory_treemap",
    "histc",
    "hypot",
    "iinfo",
    "index_add",
    "index_add_",
    "index_fill",
    "index_fill_",
    "index_select",
    "isfinite",
    "isin",
    "isinf",
    "isnan",
    "isneginf",
    "isposinf",
    "knn",
    "kthvalue",
    "linspace",
    "log2",
    "make_grid",
    "median",
    "meshgrid",
    "multinomial",
    "ne",
    "new",
    "nms",
    "nonzero",
    "normalize",
    "numpy_cumprod",
    "numpy_cumsum",
    "peek",
    "peek_s",
    "print_tree",
    "python_pass_wrapper",
    "rad2deg",
    "randperm",
    "repeat",
    "repeat_interleave",
    "roll",
    "rsqrt",
    "safe_log",
    "save_image",
    "scatter",
    "scatter_",
    "scatter_add",
    "scatter_add_",
    "scatter_reduce",
    "searchsorted",
    "set_global_seed",
    "sort",
    "split",
    "stack",
    "t",
    "to",
    "tolist",
    "topk",
    "tril",
    "triu",
    "unbind",
    "unique",
    "unique_consecutive",
    "view_as",
]

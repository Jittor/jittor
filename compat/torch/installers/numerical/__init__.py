"""Family-owned Torch compatibility installer.

This module contains source moved from the former monolithic installer without
changing the compatibility semantics.
"""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import jittor as jt

from jittor import nn

import numpy as np

from builtins import all as _py_all, any as _py_any

from collections import namedtuple as _namedtuple

from ...functional import _diff, _isin, _repeat_interleave, _trapz

from ...grad import _AutocastContext

from ...nested import _NestedTensor

from ...types import _dtype_to_str

from ...fidelity import Fidelity, register_fidelity, register_api_bindings


from ....diagnostics import EXPECTED, swallowed

from .batching import vmap

register_fidelity(
    "torch.vmap", vmap, Fidelity.APPROXIMATE,
    "runs loop/broadcast batching over the shared native graph; randomness "
    "and advanced batching arguments retain existing compatibility limitations",
)

autocast = _AutocastContext

register_fidelity(
    "torch.autocast",
    autocast,
    Fidelity.APPROXIMATE,
    "matches Torch context/decorator enable semantics on supported CPU/CUDA "
    "paths; cache, device-specific dtype, and unsupported dtype diagnostics "
    "remain compatibility-layer limitations",
)

_native_all = jt.all

_native_any = jt.any

from .reductions import _reduce_alias

from .reductions import all

from .reductions import any

for _reduce_name, _reduce_impl in (("all", all), ("any", any)):
    register_fidelity(
        "torch." + _reduce_name,
        _reduce_impl,
        Fidelity.APPROXIMATE,
        "matches Torch boolean reduction values and keepdims/axis shape for "
        "CPU tensors; device, dtype, named-dimension, and out semantics are "
        "not implemented",
    )

del _reduce_name, _reduce_impl

_COMPLEX_FIDELITY_DETAIL = (
    "matches Torch complex construction and real/imag round-trips for CPU "
    "real tensors; device, layout, out, and complex128 semantics are not "
    "implemented by the Jittor complex owner"
)

from .complex import complex

from .complex import view_as_complex

from .complex import view_as_real

for _complex_name, _complex_impl in (
    ("complex", complex),
    ("view_as_complex", view_as_complex),
    ("view_as_real", view_as_real),
):
    register_fidelity(
        "torch." + _complex_name,
        _complex_impl,
        Fidelity.APPROXIMATE,
        _COMPLEX_FIDELITY_DETAIL,
    )

del _complex_name, _complex_impl

from .complex import _is_complex_value

from .complex import is_complex

from .complex import real

from .complex import imag

from .complex import conj

from .complex import angle

_native_abs = jt.abs

from .complex import abs

for _accessor_name, _accessor_impl in (
    ("is_complex", is_complex), ("real", real), ("imag", imag),
    ("conj", conj), ("angle", angle), ("abs", abs),
):
    register_fidelity(
        "torch." + _accessor_name,
        _accessor_impl,
        Fidelity.APPROXIMATE,
        "matches Torch complex accessor values for CPU tensors; device, "
        "layout, out, and complex128 semantics are not implemented",
    )

del _accessor_name, _accessor_impl

from .complex import polar

register_fidelity(
    "torch.polar",
    polar,
    Fidelity.APPROXIMATE,
    "matches Torch magnitude/phase values for CPU real tensors; device, "
    "layout, out, and dtype keyword semantics are not implemented",
)

from .linalg import eye

_PAIRWISE_DISTANCE_FIDELITY_DETAIL = (
    "matches Torch p-norm distance values and keepdim shape through Jittor's "
    "native nn implementation but omits device, layout, and dtype keyword semantics"
)

from .linalg import pairwise_distance

register_fidelity(
    "torch.pairwise_distance",
    pairwise_distance,
    Fidelity.APPROXIMATE,
    _PAIRWISE_DISTANCE_FIDELITY_DETAIL,
)

_COSINE_SIMILARITY_FIDELITY_DETAIL = (
    "matches Torch cosine similarity values, dim reduction shape, and the eps "
    "denominator floor through Jittor's native nn implementation but omits "
    "device, layout, and dtype keyword semantics"
)

from .linalg import cosine_similarity

register_fidelity(
    "torch.cosine_similarity",
    cosine_similarity,
    Fidelity.APPROXIMATE,
    _COSINE_SIMILARITY_FIDELITY_DETAIL,
)

_SVD_FIDELITY_DETAIL = (
    "matches Torch real-matrix decomposition values through Jittor's native "
    "linalg.svd but omits some/compute_uv/driver, device, and dtype keyword semantics"
)

from .linalg import svd

register_fidelity(
    "torch.svd",
    svd,
    Fidelity.APPROXIMATE,
    _SVD_FIDELITY_DETAIL,
)

_SVD_LOWRANK_FIDELITY_DETAIL = (
    "matches Torch low-rank decomposition outputs through Jittor's native SVD "
    "for supported real matrices but omits niter, device, and dtype semantics"
)

from .linalg import svd_lowrank

register_fidelity(
    "torch.svd_lowrank",
    svd_lowrank,
    Fidelity.APPROXIMATE,
    _SVD_LOWRANK_FIDELITY_DETAIL,
)

_PCA_LOWRANK_FIDELITY_DETAIL = (
    "matches Torch centered low-rank decomposition through the compatibility "
    "SVD owner for supported real matrices but omits niter, device, and dtype semantics"
)

from .linalg import pca_lowrank

register_fidelity(
    "torch.pca_lowrank",
    pca_lowrank,
    Fidelity.APPROXIMATE,
    _PCA_LOWRANK_FIDELITY_DETAIL,
)

from .sparse import _SparseCOO

from .sparse import sparse_coo_tensor

_SPARSE_COO_TENSOR_FIDELITY_DETAIL = (
    "matches Torch COO indices/values materialization through a dense-backed "
    "compatibility object but omits sparse storage, device, and dtype semantics"
)

register_fidelity(
    "torch.sparse_coo_tensor",
    sparse_coo_tensor,
    Fidelity.APPROXIMATE,
    _SPARSE_COO_TENSOR_FIDELITY_DETAIL,
)

_RANDINT_LIKE_FIDELITY_DETAIL = (
    "matches Torch integer bounds and shape, with optional dtype casting, for "
    "supported tensors but omits device and requires_grad keyword semantics"
)

from .factories import randint_like

register_fidelity(
    "torch.randint_like",
    randint_like,
    Fidelity.APPROXIMATE,
    _RANDINT_LIKE_FIDELITY_DETAIL,
)

_DET_FIDELITY_DETAIL = (
    "matches Torch determinant values through Jittor's native linalg owner for "
    "supported square real tensors but omits device, layout, and dtype semantics"
)

from .linalg import det

register_fidelity(
    "torch.det",
    det,
    Fidelity.APPROXIMATE,
    _DET_FIDELITY_DETAIL,
)

_INVERSE_FIDELITY_DETAIL = (
    "matches Torch matrix inverse values through Jittor's native linalg owner for "
    "supported square real tensors but omits device, layout, and dtype semantics"
)

from .linalg import inverse

register_fidelity(
    "torch.inverse",
    inverse,
    Fidelity.APPROXIMATE,
    _INVERSE_FIDELITY_DETAIL,
)

_TAKE_ALONG_DIM_FIDELITY_DETAIL = (
    "matches Torch gather values and broadcasted index shape for supported "
    "integer indices but omits out, device, layout, and dtype keyword semantics"
)

from .indexing import take_along_dim

register_fidelity(
    "torch.take_along_dim",
    take_along_dim,
    Fidelity.APPROXIMATE,
    _TAKE_ALONG_DIM_FIDELITY_DETAIL,
)

_LOG1P_FIDELITY_DETAIL = (
    "matches Torch elementwise log1p values for supported real tensors but "
    "omits device, layout, and dtype keyword semantics"
)

from .elementwise import log1p

register_fidelity(
    "torch.log1p",
    log1p,
    Fidelity.APPROXIMATE,
    _LOG1P_FIDELITY_DETAIL,
)

_RECIPROCAL_FIDELITY_DETAIL = (
    "matches Torch elementwise reciprocal values for supported real tensors but "
    "omits device, layout, and dtype keyword semantics"
)

from .elementwise import reciprocal

register_fidelity(
    "torch.reciprocal",
    reciprocal,
    Fidelity.APPROXIMATE,
    _RECIPROCAL_FIDELITY_DETAIL,
)

_LERP_FIDELITY_DETAIL = (
    "matches Torch linear interpolation values for supported real tensors but "
    "omits device, layout, and dtype keyword semantics"
)

from .elementwise import lerp

register_fidelity(
    "torch.lerp",
    lerp,
    Fidelity.APPROXIMATE,
    _LERP_FIDELITY_DETAIL,
)

_SOFTMAX_FIDELITY_DETAIL = (
    "matches Torch softmax values along an explicit dimension through Jittor's "
    "native nn owner and honours the dtype keyword by casting the input first, "
    "which is what Torch's dtype does; device and layout keyword semantics are "
    "not implemented"
)

from .elementwise import softmax

register_fidelity(
    "torch.softmax",
    softmax,
    Fidelity.APPROXIMATE,
    _SOFTMAX_FIDELITY_DETAIL,
)

_LOG_SOFTMAX_FIDELITY_DETAIL = (
    "matches Torch log-softmax values along an explicit dimension through "
    "Jittor's native nn owner and honours the dtype keyword by casting the "
    "input first, which is what Torch's dtype does; device and layout keyword "
    "semantics are not implemented"
)

from .elementwise import log_softmax

register_fidelity(
    "torch.log_softmax",
    log_softmax,
    Fidelity.APPROXIMATE,
    _LOG_SOFTMAX_FIDELITY_DETAIL,
)

_RELU_FIDELITY_DETAIL = (
    "matches Torch elementwise ReLU values through Jittor's native nn owner but "
    "omits inplace, device, layout, and dtype keyword semantics"
)

from .elementwise import relu

register_fidelity(
    "torch.relu",
    relu,
    Fidelity.APPROXIMATE,
    _RELU_FIDELITY_DETAIL,
)

_SHAPE_AS_TENSOR_FIDELITY_DETAIL = (
    "matches Torch int64 shape materialization for supported tensors but omits "
    "device, layout, and dynamic-shape keyword semantics"
)

from .elementwise import _shape_as_tensor

register_fidelity(
    "torch._shape_as_tensor",
    _shape_as_tensor,
    Fidelity.APPROXIMATE,
    _SHAPE_AS_TENSOR_FIDELITY_DETAIL,
)

_NATIVE_OWNER_FIDELITY_DETAIL = (
    "re-exports Jittor's native implementation, whose signature and values "
    "already match Torch for supported real tensors; device, layout, and dtype "
    "keyword semantics are not implemented, and out is not accepted"
)

outer = jt.outer

register_fidelity(
    "torch.outer",
    outer,
    Fidelity.APPROXIMATE,
    _NATIVE_OWNER_FIDELITY_DETAIL,
)

_NATIVE_ISIN = jt.isin

_ISIN_FIDELITY_DETAIL = (
    "re-exports Jittor's native isin implementation for supported tensors but "
    "omits device, layout, and dtype keyword semantics"
)

from .elementwise import isin

register_fidelity(
    "torch.isin",
    isin,
    Fidelity.APPROXIMATE,
    _ISIN_FIDELITY_DETAIL,
)

tensordot = jt.tensordot

register_fidelity(
    "torch.tensordot",
    tensordot,
    Fidelity.APPROXIMATE,
    _NATIVE_OWNER_FIDELITY_DETAIL,
)

repeat_interleave = jt.repeat_interleave

register_fidelity(
    "torch.repeat_interleave",
    repeat_interleave,
    Fidelity.APPROXIMATE,
    _NATIVE_OWNER_FIDELITY_DETAIL,
)

_NAN_TO_NUM_INPLACE_FIDELITY_DETAIL = (
    "matches Torch in-place NaN/Inf replacement and return identity for supported "
    "real tensors but omits device, layout, dtype, and narrow custom-bound semantics"
)

from .elementwise import nan_to_num_

register_fidelity(
    "torch.nan_to_num_",
    nan_to_num_,
    Fidelity.APPROXIMATE,
    _NAN_TO_NUM_INPLACE_FIDELITY_DETAIL,
)

_STACKING_FIDELITY_DETAIL = (
    "matches Torch values and shapes for tensor inputs but omits Torch "
    "device, dtype, layout, pin-memory, and out keyword semantics"
)

from .shape import _vstack_impl

from .shape import vstack

from .shape import row_stack

from .shape import hstack

from .shape import dstack

from .shape import column_stack

for _stacking_name in ("vstack", "row_stack", "hstack", "dstack", "column_stack"):
    register_fidelity(
        "torch." + _stacking_name,
        globals()[_stacking_name],
        Fidelity.APPROXIMATE,
        _STACKING_FIDELITY_DETAIL,
    )

del _stacking_name

_MOVEDIM_FIDELITY_DETAIL = (
    "matches Torch axis permutation for valid tensor inputs but omits Torch "
    "layout, device, out, and named-dimension semantics"
)

from .shape import _movedim_impl

from .shape import movedim

from .shape import moveaxis

for _movedim_name in ("movedim", "moveaxis"):
    register_fidelity(
        "torch." + _movedim_name,
        globals()[_movedim_name],
        Fidelity.APPROXIMATE,
        _MOVEDIM_FIDELITY_DETAIL,
    )

del _movedim_name

_SHAPE_HELPER_FIDELITY_DETAIL = (
    "matches Torch values and shapes for valid tensor inputs but omits Torch "
    "device, layout, named-dimension, and out keyword semantics"
)

from .shape import _unflatten_impl

from .shape import unflatten

from .shape import _swapaxes_impl

from .shape import swapaxes

from .shape import swapdims

from .shape import _ravel_impl

from .shape import ravel

for _shape_helper_name in ("unflatten", "swapaxes", "swapdims", "ravel"):
    register_fidelity(
        "torch." + _shape_helper_name,
        globals()[_shape_helper_name],
        Fidelity.APPROXIMATE,
        _SHAPE_HELPER_FIDELITY_DETAIL,
    )

del _shape_helper_name

_ELEMENTWISE_FIDELITY_DETAIL = (
    "matches Torch values for supported real tensor inputs but omits Torch "
    "device, layout, and out keyword semantics"
)

from .elementwise import _copysign_impl

from .elementwise import copysign

from .elementwise import _xlogy_impl

from .elementwise import xlogy

from .elementwise import _heaviside_impl

from .elementwise import heaviside

from .elementwise import _signbit_impl

from .elementwise import signbit

for _elementwise_name in ("copysign", "xlogy", "heaviside", "signbit"):
    register_fidelity(
        "torch." + _elementwise_name,
        globals()[_elementwise_name],
        Fidelity.APPROXIMATE,
        _ELEMENTWISE_FIDELITY_DETAIL,
    )

del _elementwise_name

_FLOAT_POWER_FIDELITY_DETAIL = (
    "computes supported real inputs in float64 like Torch float_power but "
    "omits device, layout, and out keyword semantics"
)

from .elementwise import _float_power_impl

from .elementwise import float_power

register_fidelity(
    "torch.float_power",
    float_power,
    Fidelity.APPROXIMATE,
    _FLOAT_POWER_FIDELITY_DETAIL,
)

_MATRIX_FIDELITY_DETAIL = (
    "matches Torch values for supported real tensor inputs but omits Torch "
    "offset, dimension, device, layout, and out keyword semantics"
)

from .linalg import _trace_impl

from .linalg import trace

from .linalg import _diag_embed_impl

from .linalg import diag_embed

from .linalg import _diagflat_impl

from .linalg import diagflat

for _matrix_name in ("trace", "diag_embed", "diagflat"):
    register_fidelity(
        "torch." + _matrix_name,
        globals()[_matrix_name],
        Fidelity.APPROXIMATE,
        _MATRIX_FIDELITY_DETAIL,
    )

del _matrix_name

_CLOSE_FIDELITY_DETAIL = (
    "matches Torch finite-value comparisons and equal_nan on CPU-backed "
    "tensors but omits device, layout, and out keyword semantics"
)

from .elementwise import _isclose_impl

from .elementwise import isclose

from .elementwise import _allclose_impl

from .elementwise import allclose

for _close_name in ("isclose", "allclose"):
    register_fidelity(
        "torch." + _close_name,
        globals()[_close_name],
        Fidelity.APPROXIMATE,
        _CLOSE_FIDELITY_DETAIL,
    )

del _close_name

_PAIRWISE_SEARCH_FIDELITY_DETAIL = (
    "matches Torch values for supported tensor inputs but omits compute-mode, "
    "device, layout, and out keyword semantics"
)

from .distance import _cdist_impl

from .distance import cdist

from .distance import _bucketize_impl

from .distance import bucketize

for _pairwise_search_name in ("cdist", "bucketize"):
    register_fidelity(
        "torch." + _pairwise_search_name,
        globals()[_pairwise_search_name],
        Fidelity.APPROXIMATE,
        _PAIRWISE_SEARCH_FIDELITY_DETAIL,
    )

del _pairwise_search_name

_NAN_REDUCTION_FIDELITY_DETAIL = (
    "matches Torch NaN-ignoring reductions and NaN counts for supported real "
    "tensor inputs but omits device, layout, and out keyword semantics"
)

from .reductions import _nansum_impl

from .reductions import nansum

from .reductions import _nanmean_impl

from .reductions import nanmean

for _nan_reduction_name in ("nansum", "nanmean"):
    register_fidelity(
        "torch." + _nan_reduction_name,
        globals()[_nan_reduction_name],
        Fidelity.APPROXIMATE,
        _NAN_REDUCTION_FIDELITY_DETAIL,
    )

del _nan_reduction_name

_QUANTILE_FIDELITY_DETAIL = (
    "uses a NumPy CPU fallback for supported real tensors; dtype is returned "
    "as float32 and device, layout, interpolation, and out semantics are not "
    "implemented"
)

from .reductions import _quantile_impl

from .reductions import quantile

register_fidelity(
    "torch.quantile",
    quantile,
    Fidelity.APPROXIMATE,
    _QUANTILE_FIDELITY_DETAIL,
)

_NANQUANTILE_FIDELITY_DETAIL = (
    "uses a NumPy CPU fallback for supported real tensors; dtype is returned "
    "as float32 and device, layout, interpolation, and out semantics are not "
    "implemented"
)

from .reductions import _nanquantile_impl

from .reductions import nanquantile

register_fidelity(
    "torch.nanquantile",
    nanquantile,
    Fidelity.APPROXIMATE,
    _NANQUANTILE_FIDELITY_DETAIL,
)

_STD_MEAN_FIDELITY_DETAIL = (
    "matches the current Jittor mean/std values for supported real tensors; "
    "correction is ignored and dim std does not preserve keepdim, while "
    "device, layout, and out semantics are omitted"
)

from .reductions import _std_mean_impl

from .reductions import std_mean

from .reductions import _var_mean_impl

from .reductions import var_mean

for _std_mean_name in ("std_mean", "var_mean"):
    register_fidelity(
        "torch." + _std_mean_name,
        globals()[_std_mean_name],
        Fidelity.APPROXIMATE,
        _STD_MEAN_FIDELITY_DETAIL,
    )

del _std_mean_name

_AminMax = _namedtuple("aminmax", ["min", "max"])

_AMINMAX_FIDELITY_DETAIL = (
    "matches Torch min/max values for supported real tensor inputs but omits "
    "device, layout, and out keyword semantics"
)

from .reductions import _aminmax_impl

from .reductions import aminmax

register_fidelity(
    "torch.aminmax",
    aminmax,
    Fidelity.APPROXIMATE,
    _AMINMAX_FIDELITY_DETAIL,
)

_PDIST_FIDELITY_DETAIL = (
    "matches Torch pairwise distances for supported real tensor inputs but "
    "omits device, layout, and out keyword semantics"
)

from .distance import _pdist_impl

from .distance import pdist

register_fidelity(
    "torch.pdist",
    pdist,
    Fidelity.APPROXIMATE,
    _PDIST_FIDELITY_DETAIL,
)

_LOGCUMSUMEXP_FIDELITY_DETAIL = (
    "matches Torch cumulative log-sum-exp values for supported real tensors "
    "but omits device, layout, and out keyword semantics"
)

from .reductions import _logcumsumexp_impl

from .reductions import logcumsumexp

register_fidelity(
    "torch.logcumsumexp",
    logcumsumexp,
    Fidelity.APPROXIMATE,
    _LOGCUMSUMEXP_FIDELITY_DETAIL,
)

_MV_FIDELITY_DETAIL = (
    "matches Torch matrix-vector values, shape checks, and out identity for "
    "supported real tensors but omits device, layout, and dtype keyword semantics"
)

from .linalg import _mv_impl

from .linalg import mv

register_fidelity(
    "torch.mv",
    mv,
    Fidelity.APPROXIMATE,
    _MV_FIDELITY_DETAIL,
)

_ADDMM_FIDELITY_DETAIL = (
    "matches Torch alpha/beta matrix addition for supported real tensors but "
    "omits device, layout, dtype, and out keyword semantics"
)

from .linalg import _addmm_impl

from .linalg import addmm

register_fidelity(
    "torch.addmm",
    addmm,
    Fidelity.APPROXIMATE,
    _ADDMM_FIDELITY_DETAIL,
)

_MM_FIDELITY_DETAIL = (
    "matches Torch 2-D matrix multiplication values for supported real tensors "
    "but omits out, device, layout, and dtype keyword semantics"
)

from .linalg import _mm_impl

from .linalg import mm

register_fidelity(
    "torch.mm",
    mm,
    Fidelity.APPROXIMATE,
    _MM_FIDELITY_DETAIL,
)

_TRAPZ_FIDELITY_DETAIL = (
    "matches Torch composite trapezoidal integration values and out identity "
    "for supported real tensors but omits device, layout, and dtype keyword semantics"
)

from .integration import trapz

from .integration import trapezoid

register_fidelity(
    "torch.trapz",
    trapz,
    Fidelity.APPROXIMATE,
    _TRAPZ_FIDELITY_DETAIL,
)

register_fidelity(
    "torch.trapezoid",
    trapezoid,
    Fidelity.APPROXIMATE,
    _TRAPZ_FIDELITY_DETAIL,
)

_MASKED_SELECT_FIDELITY_DETAIL = (
    "matches Torch boolean selection values and flattened shape for supported "
    "real tensors but omits out, device, layout, and dtype keyword semantics; "
    "the same object is bound as the Tensor method, which therefore also "
    "accepts the out keyword that Torch's method rejects"
)

from .indexing import _masked_select_impl

from .indexing import masked_select

register_fidelity(
    "torch.masked_select",
    masked_select,
    Fidelity.APPROXIMATE,
    _MASKED_SELECT_FIDELITY_DETAIL,
)

_NARROW_FIDELITY_DETAIL = (
    "matches Torch contiguous slice values and shape for supported real tensors "
    "but omits device, layout, and dtype keyword semantics"
)

from .shape import _narrow_impl

from .shape import narrow

register_fidelity(
    "torch.narrow",
    narrow,
    Fidelity.APPROXIMATE,
    _NARROW_FIDELITY_DETAIL,
)

_TILE_FIDELITY_DETAIL = (
    "matches Torch repetition values and shape for supported real tensors but "
    "omits device, layout, and dtype keyword semantics"
)

from .shape import _tile_impl

from .shape import tile

register_fidelity(
    "torch.tile",
    tile,
    Fidelity.APPROXIMATE,
    _TILE_FIDELITY_DETAIL,
)

_DIFF_FIDELITY_DETAIL = (
    "matches Torch finite differences and prepend/append concatenation for "
    "supported real tensors but omits device, layout, and dtype keyword semantics"
)

from .elementwise import _diff_impl

from .elementwise import diff

register_fidelity(
    "torch.diff",
    diff,
    Fidelity.APPROXIMATE,
    _DIFF_FIDELITY_DETAIL,
)

_SQUARE_FIDELITY_DETAIL = (
    "matches Torch elementwise square values for supported real tensors but "
    "omits device, layout, and dtype keyword semantics"
)

from .elementwise import square

register_fidelity(
    "torch.square",
    square,
    Fidelity.APPROXIMATE,
    _SQUARE_FIDELITY_DETAIL,
)

_SPLIT_WITH_SIZES_FIDELITY_DETAIL = (
    "matches Torch split sizes and values for supported real tensors but omits "
    "device, layout, and dtype keyword semantics"
)

from .indexing import split_with_sizes

register_fidelity(
    "torch.split_with_sizes",
    split_with_sizes,
    Fidelity.APPROXIMATE,
    _SPLIT_WITH_SIZES_FIDELITY_DETAIL,
)

from .indexing import equal

register_fidelity(
    "torch.equal",
    equal,
    Fidelity.APPROXIMATE,
    "matches Torch Python-bool shape/value equality for CPU tensors; device, "
    "layout, and named-dimension semantics are not implemented",
)

from .indexing import tensor_split

register_fidelity(
    "torch.tensor_split",
    tensor_split,
    Fidelity.APPROXIMATE,
    "matches Torch split shapes and values for CPU tensors; device, layout, "
    "named-dimension, and out semantics are not implemented",
)

from .indexing import take

register_fidelity(
    "torch.take",
    take,
    Fidelity.APPROXIMATE,
    "matches Torch flattened indexing values for CPU tensors; device, layout, "
    "dtype, and out semantics are not implemented",
)

from .indexing import index_copy

register_fidelity(
    "torch.index_copy",
    index_copy,
    Fidelity.APPROXIMATE,
    "matches Torch non-inplace index-copy values for CPU tensors; device, "
    "layout, dtype, and out semantics are not implemented",
)

from .indexing import index_copy_

register_fidelity(
    "torch.index_copy_",
    index_copy_,
    Fidelity.APPROXIMATE,
    "matches Torch in-place indexed assignment for CPU tensors; device, "
    "layout, dtype, and out semantics are not implemented",
)

from .indexing import index_put

register_fidelity(
    "torch.index_put",
    index_put,
    Fidelity.APPROXIMATE,
    "matches Torch non-inplace indexed assignment for CPU tensors; duplicate "
    "multi-dimensional accumulation, device, layout, and out semantics are "
    "not implemented",
)

from .indexing import index_put_

register_fidelity(
    "torch.index_put_",
    index_put_,
    Fidelity.APPROXIMATE,
    "matches Torch in-place indexed assignment and duplicate accumulation for "
    "CPU tensors; device, layout, dtype, and out semantics are not implemented",
)

from .linalg import kron

register_fidelity(
    "torch.kron",
    kron,
    Fidelity.APPROXIMATE,
    "matches Torch Kronecker shape and values for CPU tensors; device, "
    "layout, dtype, and out semantics are not implemented",
)

from .reductions import logsumexp

register_fidelity(
    "torch.logsumexp",
    logsumexp,
    Fidelity.APPROXIMATE,
    "matches Torch reduction values and keepdim shape for CPU tensors; device, "
    "dtype, named-dimension, and out semantics are not implemented",
)

from .signal import hann_window

register_fidelity(
    "torch.hann_window",
    hann_window,
    Fidelity.APPROXIMATE,
    "matches Torch periodic and symmetric CPU window values; device, dtype, "
    "layout, and requires_grad semantics are not implemented",
)

from .signal import stft

register_fidelity(
    "torch.stft",
    stft,
    Fidelity.APPROXIMATE,
    "matches Torch CPU NumPy STFT values for supported real waveforms; "
    "gradient, device, window dtype, and return_complex=False semantics are "
    "not implemented",
)

def _bind_missing(target, name, implementation):
    if not hasattr(target, name):
        setattr(target, name, implementation)


def _tensor_abs(value):
    # Native module-level C functions do not implement Python's descriptor
    # binding protocol; a Tensor method must supply its argument explicitly.
    return _native_abs(value)


def _sparse_sum(x, dim=None):
    d = x._dense if isinstance(x, _SparseCOO) else x
    return _SparseCOO(d.sum(dim) if dim is not None else d.sum())


def _vdet(self):
    import jittor.linalg as _la; return _la.det(self)


def _vinv(self):
    import jittor.linalg as _la; return _la.inv(self)


def install(ctx):
    _modules = ctx.registry.module_map
    g = ctx.jittor_module
    Var = ctx.state["Var"]
    _DTYPE_OBJS = ctx.state["dtypes"]
    import collections as _collections
    # complex-dtype API (#3): jittor represents complex via nn.ComplexNumber (real/imag
    # pair); wire the torch entry points onto it. torch.complex(re,im), view_as_complex
    # (last dim of 2 -> complex), view_as_real (complex -> last dim of 2), polar, real/
    # imag/conj/is_complex. The arithmetic (* / + matmul exp conj) is on ComplexNumber.
    _CN = jt.nn.ComplexNumber
    # A complex value is either the legacy ComplexNumber (still produced by torch.complex and
    # consumed by torch.fft.* -- migrated in P3) OR the native complex64 dtype (Phase 6). The
    # accessors below handle both; Var.real/imag/angle are patched in jittor.nn. We force-set
    # (not _alias) the accessors because _alias skips names that already exist as native ops --
    # that is why torch.conj(ComplexNumber) used to fall through to the native conj op and crash.
    _bind_missing(g, "complex", complex)  # native complex64
    _bind_missing(g, "view_as_complex", view_as_complex)   # -> native complex64
    _bind_missing(g, "view_as_real", view_as_real)         # polymorphic
    g.is_complex = is_complex
    g.real = real
    g.imag = imag
    _bind_missing(g, "polar", polar)                                        # -> native complex64
    g.conj = conj
    g.angle = angle
    # torch.abs of a complex tensor is its magnitude; jittor's abs only takes real Vars.
    g.abs = abs
    Var.abs = _tensor_abs

    # ``jittor.fft`` is the native owner. Torch mode publishes that same module
    # object under its historical namespace instead of carrying a duplicate DFT.
    from jittor import fft as _fft_ns
    from ...namespace import native_module_facade
    _fft_ns = native_module_facade(_fft_ns, "torch.fft")
    g.fft = _fft_ns
    _modules["torch.fft"] = _fft_ns
    # torch.softmax / log_softmax / relu top-level function forms (convbert calls
    # torch.softmax(x, dim=...)). jittor exposes these via nn, not the top level.
    _bind_missing(g, "softmax", softmax)
    _bind_missing(g, "log_softmax", log_softmax)
    _bind_missing(g, "relu", relu)
    # elementwise / functional top-level forms missing from jittor's top level
    _bind_missing(g, "log1p", log1p)
    _bind_missing(g, "reciprocal", reciprocal)
    _bind_missing(g, "lerp", lerp)
    _bind_missing(g, "isclose", isclose)
    _bind_missing(g, "allclose", allclose)
    _bind_missing(g, "cosine_similarity", cosine_similarity)
    _bind_missing(g, "pairwise_distance", pairwise_distance)
    # torch.take_along_dim(input, indices, dim): like gather, but torch BROADCASTS
    # indices against input on every dim except `dim` first. transformers' beam search
    # _gather_beams passes indices of shape (batch, k, 1) to gather full sequences of
    # shape (batch, beams, seq_len) along dim=1 -> expects (batch, k, seq_len). A plain
    # jt.gather returns the index's shape (batch, k, 1), collapsing seq_len -> beam
    # search crashed on the next `seq[:, :, cur_len] = ...` setitem. Broadcast first.
    _bind_missing(g, "take_along_dim", take_along_dim)
    _orig_all = getattr(g, "all", None)
    _orig_any = getattr(g, "any", None)
    if callable(_orig_all):
        g.all = all
    if callable(_orig_any):
        g.any = any
    _bind_missing(g, "movedim", movedim)
    _bind_missing(g, "moveaxis", moveaxis)
    # Var.movedim/moveaxis (the functions exist but weren't bound as methods), plus
    # index_put_/index_put (scatter-style assignment), tensor_split (uneven split), take.
    Var.movedim = _movedim_impl
    Var.moveaxis = _movedim_impl
    Var.index_put_ = index_put_
    Var.index_put = index_put
    # index_copy_(dim, index, source): self[..,index[i],..] = source[i,..] along dim
    # (overwrite, NOT accumulate -- cf. index_add).
    Var.index_copy_ = index_copy_
    Var.index_copy = index_copy
    g.index_copy = index_copy
    g.index_copy_ = index_copy_
    g.index_put = index_put
    g.index_put_ = index_put_
    Var.tensor_split = tensor_split
    g.tensor_split = tensor_split
    Var.take = take
    g.take = take
    _bind_missing(g, "eye", eye)
    register_fidelity(
        "torch.eye", eye, Fidelity.APPROXIMATE,
        "Values and dtype are supported; layout, device, out, and pin_memory "
        "arguments are not implemented.")
    # torch.narrow(input, dim, start, length) / torch.tile(input, dims) --
    # function forms mirroring the Var methods (added in _install_tensor_methods).
    _bind_missing(g, "narrow", narrow)
    _bind_missing(g, "tile", tile)
    # torch.equal returns a Python bool (True iff same shape & all elements
    # equal). jittor's native `equal` is elementwise, so force-override.
    g.equal = equal
    Var.equal = equal
    _bind_missing(g, "diff", diff)
    _bind_missing(g, "trapz", trapz)
    _bind_missing(g, "trapezoid", trapezoid)
    g.repeat_interleave = repeat_interleave
    _bind_missing(g, "autocast", autocast)
    # Real loop-based torch.vmap. The old no-op stub (`lambda fn,*a,**k: fn`)
    # ignored in_dims/out_dims, so transformers' vmap-based causal-mask builder
    # (taken when a model passes and_mask/or_mask -- e.g. falcon) collapsed to a
    # single direct call and produced a wrong all-True (seq,) mask instead of the
    # (b,1,q,kv) causal triangle -> bidirectional attention -> ~79% forward error.
    # Batching is owned by the module-level implementation in batching.py.

    g.vmap = vmap
    g.outer = outer
    g.isin = isin
    # Pairwise distances and sorted-boundary insertion indices.
    _bind_missing(g, "cdist", cdist)
    _bind_missing(g, "bucketize", bucketize)
    # trace / diag_embed / diagflat / kron / logcumsumexp / tensordot / pdist.
    _bind_missing(g, "trace", trace); Var.trace = _trace_impl
    _bind_missing(g, "diag_embed", diag_embed); Var.diag_embed = _diag_embed_impl
    _bind_missing(g, "diagflat", diagflat)
    g.kron = kron; Var.kron = kron
    _bind_missing(g, "logcumsumexp", logcumsumexp); Var.logcumsumexp = _logcumsumexp_impl
    g.tensordot = tensordot
    _bind_missing(g, "pdist", pdist); Var.pdist = _pdist_impl
    # shape ops: unflatten / swapaxes / swapdims / ravel + numpy-style stacking helpers.
    _bind_missing(g, "unflatten", unflatten); Var.unflatten = _unflatten_impl
    _bind_missing(g, "swapaxes", swapaxes); _bind_missing(g, "swapdims", swapdims)
    Var.swapaxes = _swapaxes_impl; Var.swapdims = _swapaxes_impl
    _bind_missing(g, "ravel", ravel); Var.ravel = _ravel_impl
    _bind_missing(g, "vstack", vstack)
    _bind_missing(g, "row_stack", row_stack)
    _bind_missing(g, "hstack", hstack)
    _bind_missing(g, "dstack", dstack)
    _bind_missing(g, "column_stack", column_stack)
    # element-wise ops: copysign / xlogy / heaviside / float_power / signbit.
    _bind_missing(g, "copysign", copysign); Var.copysign = _copysign_impl
    _bind_missing(g, "xlogy", xlogy); Var.xlogy = _xlogy_impl
    _bind_missing(g, "heaviside", heaviside); Var.heaviside = _heaviside_impl
    _bind_missing(g, "float_power", float_power); Var.float_power = _float_power_impl
    _bind_missing(g, "signbit", signbit); Var.signbit = _signbit_impl
    # reductions: logsumexp (attention/MoE/loss/beam), nansum/nanmean, std_mean/var_mean,
    # aminmax, quantile. NaN handling uses nan_to_num plus an explicit isnan mask.
    g.logsumexp = logsumexp; Var.logsumexp = logsumexp
    _bind_missing(g, "nansum", nansum); Var.nansum = _nansum_impl
    _bind_missing(g, "nanmean", nanmean); Var.nanmean = _nanmean_impl
    _bind_missing(g, "std_mean", std_mean)
    _bind_missing(g, "var_mean", var_mean)
    _bind_missing(g, "aminmax", aminmax); Var.aminmax = _aminmax_impl
    _bind_missing(g, "quantile", quantile)
    _bind_missing(g, "nanquantile", nanquantile)
    # Keep the Tensor methods on the same numerical owners as the top-level
    # functions.  The owners intentionally use the documented CPU NumPy
    # fallback, so method and function forms share the same fidelity limits.
    Var.quantile = quantile
    Var.nanquantile = nanquantile
    _bind_missing(g, "square", square)
    _bind_missing(g, "addmm", addmm)

    # ---- torch.* ops used by mmdetection (additive aliases) ----
    _bind_missing(g, "mm", mm)
    _bind_missing(g, "mv", mv)
    _bind_missing(g, "masked_select", masked_select)
    _bind_missing(g, "split_with_sizes", split_with_sizes)
    _bind_missing(g, "_shape_as_tensor", _shape_as_tensor)
    _bind_missing(g, "nan_to_num_", nan_to_num_)
    # torch.randint_like(input, low, high=None, *, dtype=...): jittor's native lacks
    # the dtype kwarg (DINO's denoising uses it). Force-override with torch semantics.
    g.randint_like = randint_like

    _bind_missing(g, "sparse_coo_tensor", sparse_coo_tensor)
    import jittor.sparse as _jt_sparse
    from ...namespace import native_module_facade
    sparse_namespace = native_module_facade(_jt_sparse, "torch.sparse")
    g.sparse = sparse_namespace
    if not hasattr(sparse_namespace, "sum"):
        sparse_namespace.sum = _sparse_sum

    # det/inverse on (batched) square matrices (mmrotate GWD/KLD/KFIoU Gaussian losses)
    if not hasattr(Var, "det"):       Var.det = _vdet
    if not hasattr(Var, "inverse"):   Var.inverse = _vinv
    g.det = det
    g.inverse = inverse

    # ---- linalg (peft / lora init need svd_lowrank, svd) ----
    _bind_missing(g, "svd", svd)
    _bind_missing(g, "svd_lowrank", svd_lowrank)
    _bind_missing(g, "pca_lowrank", pca_lowrank)
    register_api_bindings(Var, "torch.Tensor", (
        "abs", "movedim", "moveaxis", "index_put", "index_copy",
        "tensor_split", "take", "equal", "quantile", "nanquantile",
        "det", "inverse",
    ), Fidelity.APPROXIMATE,
        "Shares the numerical module's implementation; existing dtype, layout and fallback limitations remain")
    register_api_bindings(sparse_namespace, "torch.sparse", ("sum",),
        Fidelity.APPROXIMATE, "Existing dense-backed sparse reduction fallback")

def install_parity(ctx):
    g = ctx.jittor_module
    registry = ctx.registry
    import jittor.linalg as linalg
    from ...namespace import native_module_facade
    linalg = native_module_facade(linalg, "torch.linalg")
    registry.publish("torch.linalg", linalg)
    g.linalg = linalg

    sparse = g.sparse
    registry.publish("torch.sparse", sparse)
    g.sparse = sparse

    special = registry.ensure("torch.special")
    for name in ("erf", "erfc", "exp", "expm1", "log1p", "sinc"):
        value = getattr(g, name, None)
        if value is not None:
            setattr(special, name, value)
    special.expit = getattr(g, "sigmoid")
    g.special = special

def install_signal(ctx):
    """Window and short-time Fourier transform.

    Whisper-style mel feature extraction (Qwen2-Audio/Omni and any other audio
    front-end) calls these two directly. NumPy carries the arithmetic: the inputs
    are single waveforms, so the transform is not on a throughput path, and an
    exact match with torch's definition matters more than speed here.
    """
    g = ctx.jittor_module
    if not hasattr(g, "hann_window"):
        g.hann_window = hann_window
    if not hasattr(g, "stft"):
        g.stft = stft

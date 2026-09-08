"""Native tensor factories, operations and Var protocol bindings."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from jittor._core.dtypes import dtype_for_compute as _dtype_for_compute
from jittor._core.dtypes import is_dtype as _is_dtype

import functools as _functools
import numbers
from collections.abc import Sequence
from builtins import bool as ori_bool, float as ori_float, int as ori_int

import numpy as np
import jittor_core as core
from jittor_core import NanoString, NanoVector, Var, ops
from .flags import flag_scope
from .._runtime.acl_clamp import dispatch_acl_clamp
from .._runtime.backend_libraries import get_library as _get_library
from .._runtime.dispatch import register_kernel as _register_kernel, try_dispatch as _try_dispatch

unary = ops.unary
int32 = ops.int32
float16 = ops.float16
float32 = ops.float32
float64 = ops.float64
bool = ops.bool
reshape = ops.reshape
transpose = ops.transpose
index = ops.index
arg_reduce = ops.arg_reduce
where = ops.where
floor_int = ops.floor_int
sigmoid = ops.sigmoid

# `bool` above is the native cast op. `int` in annotations below is still the
# Python builtin until the historical dtype aliases are installed at the end.

cast = unary

Var.cast = Var.cast

def array(data, dtype=None):
    ''' Constructs a jittor Var from a number, List, numpy array or another jittor Var.

    :param data: The data to initialize the Var.
    :type data: number, list, numpy.ndarray, or jittor.Var.
    :param dtype: The data type of the Var. If None, the data type will be inferred from the data.
    :type dtype: str, jittor type-cast function, or None.

    ----------------

    Example::

        >>> jt.array(1)
        jt.Var([1], dtype=int32)
        >>> jt.array([0, 2.71, 3.14])
        jt.Var([0.   2.71 3.14], dtype=float32)
        >>> jt.array(np.arange(4, dtype=np.uint8))
        jt.Var([0 1 2 3], dtype=uint8)
    '''
    # torch accepts a range/generator in tensor(...); jittor's core array op
    # rejects them -> materialise to a list first (mmdet pisa_loss: jt.array(range(...))).
    import types as _types_arr
    if isinstance(data, (range, _types_arr.GeneratorType, map, filter, zip)):
        data = list(data)
    elif isinstance(data, core.NanoVector):     # e.g. jt.array(some_var.shape)
        data = list(data)                       # NB: `int` is shadowed by the dtype here
    if isinstance(data, core.Var):
        if dtype is None:
            ret = data.clone()
        else:
            ret = cast(data, dtype)
    elif dtype is not None:
        dtype = _dtype_for_compute(dtype)
        with flag_scope(auto_convert_64_to_32=0):
            if dtype == "bfloat16":
                ret = ops.array(np.array(data, "float32")).cast(dtype)
            else:
                ret = ops.array(np.array(data, dtype))
    else:
        ret = ops.array(data)
    # TODO: move those code to core
    return _amp_array_preference(ret)

class amp_flags:
    """Named bits of ``jt.flags.amp_reg``, the auto-mixed-precision register.

    The register is a bit field, and every reader in the tree used to spell its
    bits as bare integers -- ``amp_reg & 16``, ``amp_reg | 36``, ``amp_reg=4``
    -- across six files, with the meaning written down only in the flag's
    description string in ``src/var.cc``. ``36`` is ``keep_reduce |
    array_prefer``; you had to know that.

    These MUST match ``amp_prefer32`` .. ``amp_array_prefer`` in
    ``src/type/nano_string.h`` and the ``auto_mixed_precision_level`` setter in
    ``src/var.cc``; ``tests/core/test_amp_reg_bits.py`` reads the header and
    fails if they drift.
    """

    #: force float32 for ops whose inputs are all non-scalar floats
    prefer32 = 1
    #: force float16 (bfloat16 if an input is bfloat16) for the same
    prefer16 = 2
    #: let a reduce keep its input's float type instead of accumulating in f32
    keep_reduce = 4
    #: let "white list" ops (exp, log, pow, ...) follow the preference too,
    #: instead of always computing in float32
    keep_white = 8
    #: apply the preference to array-like producers (jt.array, jt.random) too
    array_prefer = 16
    #: a float16 sum/mean does NOT use a float32 intermediate accumulator
    #: (read directly as ``amp_reg & 32`` in src/ops/reduce_op.cc)
    reduce16_no_fp32_acc = 32

def _amp_array_preference(ret):
    """Apply the array-like AMP preference to a freshly produced Var.

    ``array()`` and ``random()`` each carried their own copy of this, and the
    copies had drifted: ``array()`` skipped one-element and non-float results,
    ``random()`` did not. So under ``auto_mixed_precision_level=5``,
    ``jt.array([1.0])`` stayed float32 while ``jt.random((1,))`` came back
    float16 -- the same value, produced two ways, with two dtypes.

    ``array()``'s guards are the ones that survive. A Var of one element is a
    scalar as far as jittor's dtype inference is concerned (``dtype_infer``
    passes ``has_scalar`` and then skips the preference entirely), so casting
    one down here would make ``jt.array(1e-8) * x`` disagree with
    ``1e-8 * x``; and a non-float result has no float preference to apply.
    """
    import jittor as jt
    amp_reg = jt.flags.amp_reg
    if not (amp_reg & amp_flags.array_prefer):
        return ret
    if ret.numel() == 1 or not ret.dtype.is_float():
        return ret
    if amp_reg & amp_flags.prefer32:
        return ret if _jittor_dtype_name(ret.dtype) == "float32" else ret.float32()
    if amp_reg & amp_flags.prefer16:
        return ret if _jittor_dtype_name(ret.dtype) == "float16" else ret.float16()
    return ret

def random(shape, dtype="float32", type="uniform"):
    ''' Constructs a random jittor Var.

    :param shape: The shape of the random Var.
    :type shape: list or tuple.
    :param dtype: The data type of the random Var.
    :type dtype: str, jittor type-cast function, or None.
    :param type: The random distribution, can be 'uniform' or 'normal'.
    :type type: str

    ----------------

    Example::

        >>> jt.random((2, 3))
        jt.Var([[0.96788853 0.28334728 0.30482838]
                [0.46107793 0.62798643 0.03457401]], dtype=float32)
    '''
    for dim in shape:
        if dim < 0:
            raise RuntimeError(f"Trying to create tensor with negative dimension {dim}: {shape}")
    dtype = _dtype_for_compute(dtype)
    if _jittor_dtype_name(dtype) in ("float16", "bfloat16"):
        # The CPU and accelerator random engines generate standard floating
        # types; low-precision outputs use their regular cast kernels.
        ret = ops.random(shape, "float32", type).cast(dtype)
    else:
        ret = ops.random(shape, dtype, type)
    return _amp_array_preference(ret)

_core_to_device = Var.to_device

def to_device(self, device):
    ''' Return this Var on CUDA device ``device`` -- an index, or anything
    with an ``index`` attribute such as a torch device. A Var already on that
    device is returned unchanged; otherwise the data is copied there by the
    ``device_copy`` op, whose gradient is a copy back.

    Devices are independent: an op takes its inputs' device and mixing two
    devices in one op is an error, as in torch. This is the only way data
    changes device. '''
    # `from jittor import *` shadows the builtin int with the cast op.
    device = ori_int(getattr(device, "index", device))
    if device < 0:
        raise RuntimeError("CUDA device index must be non-negative")
    # A host-resident Var remembers the CUDA device it belongs to.  Returning
    # it unchanged merely because the index matches would make ``x.cpu().cuda()``
    # stay on the host.  DeviceCopyOp has an explicit host-to-device path.
    if device == self.device_id and self.location() != "cpu":
        return self
    return _core_to_device(self, device)

Var.to_device = to_device

def _copy_to_cpu(self):
    '''Return a differentiable, independently allocated host copy.'''
    return _core_to_device(self, -1)

Var._copy_to_cpu = _copy_to_cpu

def float_auto(x):
    import jittor as jt
    if jt.flags.amp_reg & amp_flags.prefer16:
        return x.float16()
    return x.float32()

Var.float_auto = float_auto

def array64(data, dtype=None):
    with flag_scope(auto_convert_64_to_32=0):
        return array(data, dtype)

def grad(loss, targets, retain_graph=True):
    if isinstance(targets, core.Var):
        return core.grad(loss, [targets], retain_graph)[0]
    return core.grad(loss, targets, retain_graph)


def submit_pending(*vars, device_sync=False):
    """Submit exactly the pending subgraphs rooted at ``vars``.

    This is the explicit partial-graph boundary used by Function callbacks and
    custom execution bridges. It does not flush unrelated holder roots or alter
    the normal lazy/auto-flush policy. Set ``device_sync`` only when the caller
    immediately consumes host-visible data.
    """
    if not vars:
        raise ValueError("submit_pending requires at least one Var")
    for var in vars:
        if not isinstance(var, Var):
            raise TypeError("submit_pending expects Var arguments")
        submit = getattr(var, "submit_pending", None)
        # Older generated bindings expose only the C++ executor entry point on
        # the holder; keep the public API compatible with that shape.
        if submit is None:
            submit = lambda: core.submit_pending(var)
        if not callable(submit):
            raise RuntimeError("partial graph submission is unavailable in this core")
        submit()
    if device_sync:
        for var in vars:
            var.sync()
    return vars[0] if len(vars) == 1 else tuple(vars)

def ones(*shape, dtype="float32"):
    ''' Constructs a jittor Var with all elements set to 1.

    :param shape: The shape of the output Var.
    :type shape: list or tuple.
    :param dtype: The data type of the output Var.
    :type dtype: str, jittor type-cast function, or None.
    :return: The output Var.
    :rtype: jittor.Var
    '''
    if isinstance(shape, tuple) and _is_dtype(shape[-1]):
        dtype = shape[-1]
        shape = shape[:-1]
    if isinstance(shape, tuple) and isinstance(shape[0], (Sequence, NanoVector)):
        shape = shape[0]
    for dim in shape:
        if dim < 0:
            raise RuntimeError(f"Trying to create tensor with negative dimension {dim}: {shape}")
    return unary(1, dtype).broadcast(shape)

def new_ones(x, size):
    return ones(size, x.dtype)

Var.new_ones = new_ones

def ones_like(x):
    ''' Constructs a jittor Var with all elements set to 1 and shape same with x.

    :param x: The reference jittor Var.
    :type x: jt.Var
    :return: The output Var.
    :rtype: jittor.Var
    '''
    return ones(x.shape,x.dtype)

def zeros(*shape, dtype="float32"):
    ''' Constructs a jittor Var with all elements set to 0.

    :param shape: The shape of the output Var.
    :type shape: list or tuple.
    :param dtype: The data type of the output Var.
    :type dtype: str, jittor type-cast function, or None.
    :return: The output Var.
    :rtype: jittor.Var
    '''
    if isinstance(shape, tuple) and _is_dtype(shape[-1]):
        dtype = shape[-1]
        shape = shape[:-1]
    if isinstance(shape, tuple) and isinstance(shape[0], (Sequence, NanoVector)):
        shape = shape[0]
    for dim in shape:
        if dim < 0:
            raise RuntimeError(f"Trying to create tensor with negative dimension {dim}: {shape}")
    return unary(0, dtype).broadcast(shape)

def new_zeros(x, size):
    return zeros(size, x.dtype)

Var.new_zeros = new_zeros

def empty(*shape, dtype="float32"):
    if isinstance(shape, tuple) and _is_dtype(shape[-1]):
        dtype = shape[-1]
        shape = shape[:-1]
    if isinstance(shape, tuple) and isinstance(shape[0], (Sequence, NanoVector)):
        shape = shape[0]
    return ops.empty(shape, dtype)

def new_empty(x, size):
    return empty(size, x.dtype)

Var.new_empty = new_empty

def full(shape,val,dtype="float32"):
    ''' Constructs a jittor Var with all elements set to val.

    :param shape: The shape of the output Var.
    :type shape: list or tuple.
    :param val: The value of the output Var.
    :type val: number.
    :param dtype: The data type of the output Var. Defaults to jt.float32.
    :type dtype: str, jittor type-cast function, or None.
    :return: The output Var.
    :rtype: jittor.Var
    '''
    if not isinstance(shape, (NanoVector, Sequence)):
        shape = (shape,)
    for dim in shape:
        if dim < 0:
            raise RuntimeError(f"Trying to create tensor with negative dimension {dim}: {shape}")
    return unary(val, dtype).broadcast(shape)

def new_full(x, size, val):
    return full(size, val, x.dtype)

Var.new_full = new_full

def ne(x,y):
    return x!=y

Var.ne = ne

def full_like(x, val, dtype=None) -> Var:
    ''' Constructs a jittor Var with all elements set to val and shape same with x.

    :param x: The reference jittor Var.
    :type x: jt.Var.
    :param val: The value of the output Var.
    :type val: number.
    :param dtype: if None, the dtype of the output is the same as x.
        Otherwise, use the specified dtype. Defaults to None.
    :type dtype: str, optional
    :return: The output Var.
    :rtype: jittor.Var
    '''
    if dtype is None: dtype = x.dtype
    return full(x.shape, val, dtype)

def zeros_like(x, dtype=None) -> Var:
    ''' Constructs a jittor Var with all elements set to 0 and shape same with x.

    :param x: The reference jittor Var.
    :type x: jt.Var
    :param dtype: if None, the dtype of the output is the same as x.
        Otherwise, use the specified dtype. Defaults to None.
    :type dtype: str, optional
    :return: The output Var.
    :rtype: jittor.Var
    '''
    if dtype is None: dtype = x.dtype
    return zeros(x.shape, dtype)

def var(x, dim=None, dims=None, unbiased=False, keepdims=False):
    """ return the sample variance. If unbiased is True, Bessel's correction will be used.

    :param x: the input jittor Var.
    :type x: jt.Var.
    :param dim: the dimension to compute the variance. If both dim and dims are None, the variance of the whole tensor will be computed.
    :type dim: int.
    :param dims: the dimensions to compute the variance. If both dim and dims are None, the variance of the whole tensor will be computed.
    :type dims: tuple of int.
    :param unbiased: if True, Bessel's correction will be used.
    :type unbiased: bool.
    :param keepdim: if True, the output shape is same as input shape except for the dimension in dim.
    :type keepdim: bool.

    Example::

        >>> a = jt.rand(3)
        >>> a
        jt.Var([0.79613626 0.29322362 0.19785859], dtype=float32)
        >>> a.var()
        jt.Var([0.06888353], dtype=float32)
        >>> a.var(unbiased=True)
        jt.Var([0.10332529], dtype=float32)
    """
    import jittor as jt
    shape = x.shape
    new_shape = list(x.shape)

    if dim is not None and dims is not None:
        raise ValueError("dim and dims can not be both set")
    if dim is None and dims is None:
        dims = list(range(len(shape)))
    elif dim is not None:
        dims = [dim]

    mean = jt.mean(x, dims, keepdims=True)
    mean = jt.broadcast(mean, shape)

    n = 1
    for d in dims:
        n *= shape[d]
        new_shape[d] = 1

    sqr = (x - mean) ** 2
    sqr = jt.sum(sqr, dims=dims, keepdims=False)
    if unbiased:
        n -= 1
    sqr /= n

    if keepdims:
        sqr = sqr.view(new_shape)
    return sqr

Var.var = var

def std(x, dim=None, keepdim=False):
    import jittor as jt
    if dim is None:
        matsize=1
        for i in x.shape:
            matsize *= i
        out=(x-x.mean()).sqr().sum()
        out=out/(matsize-1)
        out=out.maximum(1e-6).sqrt()
        return out
    else:
        dimsize=x.size(dim)
        mean=jt.mean(x, dim, keepdim=True)
        out=(x - mean).sqr().sum(dim=dim, keepdim=keepdim)
        out=out/(dimsize-1)
        out=out.maximum(1e-6).sqrt()
        return out

Var.std = std

def norm(x, p=2, dim=-1, keepdims=False, eps=1e-30, keepdim=False):
    keepdim = keepdim or keepdims
    if p not in (1, 2):
        raise ValueError("norm: only p=1 and p=2 are supported")
    if p==1:
        return x.abs().sum(dim, keepdim)
    if p==2:
        return (x.sqr()).sum(dim, keepdim).maximum(eps).sqrt()

Var.norm = norm

origin_reshape = reshape

def view(x, *shape):
    if len(shape) == 1 and isinstance(shape[0], (Sequence, NanoVector)):
        shape = shape[0]
    # torch accepts 0-d int tensors / numpy ints as shape elements (e.g. longformer's
    # `_chunk` passes torch.div(size, n) into .view); jittor's core reshape needs plain
    # int64. Coerce only when a non-int element is present — plain-int shapes (the hot
    # path) are untouched, so this can't change existing behavior, only un-break it.
    # (NB: in this namespace `int`/`all`/`any` are shadowed by jittor's dtype/reductions,
    # so use an explicit loop and grab the genuine builtin int via `(0).__class__`.)
    pyint = (0).__class__
    coerce = False
    for s in shape:
        if type(s) is not pyint:
            coerce = True
            break
    if coerce:
        shape = tuple(pyint(s.item()) if isinstance(s, Var) else pyint(s) for s in shape)
    result = origin_reshape(x, shape)
    result._set_storage_view_of(x, False)
    return result

view.__doc__ = origin_reshape.__doc__

def reshape(x, *shape):
    source = x if x._storage_is_contiguous() else x.contiguous()
    return view(source, *shape)

reshape.__doc__ = origin_reshape.__doc__
Var.view = view
Var.reshape = reshape

_accelerator_transpose_tried = False

def _load_accelerator_transpose():
    """Build cuTT, which provides the accelerated CUDA transpose kernel.

    TransposeOp looks the kernel up through OpCapability::Transpose, and that
    capability only exists once the cuTT module has been compiled and loaded.
    Loading downloads and builds cutt-1.2, so it does not happen during import
    (9.01); the first transpose pays for it, the same way the first CPU float32
    batched matmul pays for MKL. setup_cutt() decides whether this
    configuration has CUDA at all, so there is nothing to test for here.

    Failing to build cuTT is not fatal -- TransposeOp has its own kernel -- so
    it is reported once and not retried.
    """
    from jittor.compiler import LOG
    global _accelerator_transpose_tried
    if _accelerator_transpose_tried:
        return
    _accelerator_transpose_tried = True
    try:
        _get_library("cutt", load=True)
    except Exception as e:
        LOG.w("cuTT is unavailable, transposing with the built-in kernel:", e)

def _with_accelerator_kernel_loaded(func):
    """Load cuTT around ``transpose`` rather than inside it.

    ``transpose`` below is the argument adapter -- axis forms, shapes, backend
    dispatch -- and its dependencies are kept to what that needs. Building a
    backend is a different concern and stays outside the function.
    """
    @_functools.wraps(func)
    def transpose_with_accelerator_kernel(x, *dim):
        _load_accelerator_transpose()
        return func(x, *dim)
    return transpose_with_accelerator_kernel

origin_transpose = transpose

def transpose(x, *dim):
    if len(dim) == 1 and isinstance(dim[0], (Sequence, NanoVector)):
        dim = dim[0]
    elif len(dim) == 2:
        axes = list(range(x.ndim))
        a, b = dim
        axes[a], axes[b] = axes[b], axes[a]
        dim = axes
    if not dim:
        dim = tuple(reversed(range(x.ndim)))
    # NumPy helpers such as np.argsort return numpy.integer axis values.  The
    # C++ transpose binding requires exact Python ints, while torch accepts any
    # integral sequence in Tensor.permute().
    pyint = (0).__class__
    coerce = False
    for d in dim:
        if type(d) is not pyint:
            coerce = True
            break
    if coerce:
        dim = tuple(pyint(d.item()) if isinstance(d, Var) else pyint(d) for d in dim)
    out = _try_dispatch("tensor.transpose", x, dim)
    if out is None:
        out = origin_transpose(x, dim)
    out._set_transpose_view_of(x, NanoVector(dim))
    return out

transpose.__doc__ = origin_transpose.__doc__

transpose = _with_accelerator_kernel_loaded(transpose)

Var.transpose = Var.permute = permute = transpose

def _flatten_cpu(input, start_dim=0, end_dim=-1):
    '''flatten dimentions by reshape'''
    in_shape = input.shape
    start_dim = len(in_shape) + start_dim if start_dim < 0 else start_dim
    end_dim = len(in_shape) + end_dim if end_dim < 0 else end_dim
    if end_dim < start_dim:
        raise ValueError("flatten: end_dim must be greater than or equal to start_dim")
    if len(in_shape) <= end_dim:
        raise IndexError(f"Dimension out of range (expected to be in range of [{-len(in_shape)}, {len(in_shape) - 1}], but got {end_dim})")
    out_shape = []
    for i in range(0,start_dim,1): out_shape.append(in_shape[i])
    dims = 1
    for i in range(start_dim, end_dim+1, 1): dims *= in_shape[i]
    out_shape.append(dims)
    for i in range(end_dim+1,len(in_shape),1): out_shape.append(in_shape[i])
    return input.reshape(out_shape)

_register_kernel("flatten", "*", _flatten_cpu)

def flatten(input, start_dim=0, end_dim=-1):
    return _try_dispatch("flatten", input, start_dim, end_dim)

Var.flatten = flatten

Var.detach_inplace = Var.start_grad

def detach(x):
    return x.detach()

def unsqueeze(x, dim):
    shape = list(x.shape)
    if dim < 0: dim += len(shape) + 1
    if dim < 0 or dim > len(shape):
        raise IndexError("unsqueeze: dimension {} out of range".format(dim))
    return x.reshape(shape[:dim] + [1] + shape[dim:])

Var.unsqueeze = unsqueeze

def squeeze(x, dim=None):
    shape = list(x.shape)
    if dim is None:
        # squeeze removes ONLY size-1 dims (size-0 dims must be kept, else an empty
        # tensor like [0,1] reshapes to the wrong size). jittor has no 0-dim tensors,
        # so an all-ones shape collapses to [1] (mmdet: nonzero(...).squeeze()).
        new_shape = [s for s in shape if s != 1]
        return x.reshape(new_shape if new_shape else [1])
    else:
        if dim < 0: dim += len(shape)
        if dim < 0 or dim >= len(shape):
            raise IndexError("squeeze: dimension {} out of range".format(dim))
        # torch (and numpy): squeeze(dim) is a no-op when that dim's size != 1,
        # not an error (canine's _downsample_attention_mask relies on this).
        if shape[dim] != 1:
            return x
        new_shape = shape[:dim] + shape[dim+1:]
        return x.reshape(new_shape if new_shape else [1])

Var.squeeze = squeeze

def _clamp_cpu(x, min_v=None, max_v=None):
    import jittor as jt
    if x.shape[0]==0:
        return x
    # Torch allows tensor bounds and reversed scalar bounds. Applying the lower
    # then upper bound also gives Torch's all-max result when min_v > max_v.

    def prepare_bound(value, bound):
        if isinstance(bound, jt.Var):
            dtype = jt.binary_dtype_infer("add", value.dtype, bound.dtype)
            if _jittor_dtype_name(value.dtype) != _jittor_dtype_name(dtype):
                value = value.cast(dtype)
            if _jittor_dtype_name(bound.dtype) != _jittor_dtype_name(dtype):
                bound = bound.cast(dtype)
        elif "float" in _jittor_dtype_name(value.dtype):
            bound = jt.unary(bound, value.dtype).stop_grad()
        elif isinstance(bound, numbers.Real) \
                and not isinstance(bound, numbers.Integral):
            value = value.cast("float32")
            bound = jt.unary(bound, "float32").stop_grad()
        else:
            bound = jt.unary(bound, value.dtype).stop_grad()
        return value, bound

    scalar_bounds = (
        min_v is not None
        and max_v is not None
        and not isinstance(min_v, jt.Var)
        and not isinstance(max_v, jt.Var)
        and min_v <= max_v
    )
    if min_v is not None:
        x, min_v = prepare_bound(x, min_v)
    if max_v is not None:
        x, max_v = prepare_bound(x, max_v)

    if scalar_bounds:
        result = dispatch_acl_clamp(x, min_v, max_v)
        if result is not None:
            return result

    def select_bound(value, bound, lower):
        keep = value >= bound if lower else value <= bound
        result = jt.ternary(keep, value, bound)
        if "float" in _jittor_dtype_name(value.dtype):
            nan_value = value.clone().stop_grad()
            result = jt.ternary(value != value, nan_value, result)
        return result

    if min_v is not None:
        x = select_bound(x, min_v, True)
    if max_v is not None:
        x = select_bound(x, max_v, False)
    return x

_register_kernel("clamp", "*", _clamp_cpu)

def clamp(x, min_v=None, max_v=None):
    return _try_dispatch("clamp", x, min_v, max_v)

Var.clamp = clamp

def clamp_(x, min_v=None, max_v=None):
    ''' In-place version of clamp().

    Args:
        x (Jittor Var):
            the input var
        min_v ( Number or Var, optional) - lower-bound of clamp range
        max_v ( Number or Var, optional) - upper-bound of clamp range

    Return:
        x itself after clamp.

    '''
    return x.assign(x.clamp(min_v=min_v, max_v=max_v))

Var.clamp_ = clamp_

def _outer_cpu(x, y):
    ''' Returns the outer product of two 1-D vectors.

    :param x: the input Var.
    :type x: jt.Var, numpy array, or python sequence.
    :param y: the input Var.
    :type y: jt.Var, numpy array, or python sequence.


    Example::

    >>> x = jt.arange(3)
    >>> y = jt.arange(4)
    >>> jt.outer(x, y)
    jt.Var([[0 0 0 0]
            [0 1 2 3]
            [0 2 4 6]], dtype=int32)
    >>> x.outer(y)
    jt.Var([[0 0 0 0]
            [0 1 2 3]
            [0 2 4 6]], dtype=int32)
    '''
    import jittor as jt
    return jt.multiply(x.unsqueeze(1), y.unsqueeze(0))

_register_kernel("outer", "*", _outer_cpu)

def outer(x, y):
    return _try_dispatch("outer", x, y)

Var.outer = outer

_native_index = index

_native_arg_reduce = arg_reduce

_native_where = where

_native_floor_int = floor_int

_native_sigmoid = sigmoid

def index(inshape=None, dim=None, dtype="int32", **kwargs):
    if "shape" in kwargs or "a" in kwargs:
        if inshape is not None or ("shape" in kwargs and "a" in kwargs):
            raise TypeError("index received its input twice")
        inshape = kwargs.pop("shape") if "shape" in kwargs else kwargs.pop("a")
    if kwargs:
        raise TypeError("index got an unexpected keyword argument: " + next(iter(kwargs)))
    if isinstance(dim, (str, NanoString, np.dtype)) or callable(dim):
        dtype, dim = dim, None
    result = _try_dispatch("tensor.index", inshape, dim, dtype)
    if result is not None:
        return result
    if dim is None:
        return _native_index(inshape, dtype=dtype)
    return _native_index(inshape, dim, dtype)

def arg_reduce(x, op, dim, keepdims=False):
    result = _try_dispatch("tensor.arg_reduce", x, op, dim, keepdims)
    if result is not None:
        return result
    return _native_arg_reduce(x, op, dim, keepdims)

def where(condition=None, x=None, y=None, dtype=None, **kwargs):
    if "cond" in kwargs:
        if condition is not None:
            raise TypeError("where received its condition twice")
        condition = kwargs.pop("cond")
    if kwargs:
        raise TypeError("where got an unexpected keyword argument: " + next(iter(kwargs)))
    if y is None and dtype is None and (isinstance(x, (str, NanoString, np.dtype)) or callable(x)):
        dtype, x = x, None
    if x is None and y is None:
        result = _try_dispatch("tensor.where", condition)
        if result is None:
            return _native_where(condition) if dtype is None else _native_where(condition, dtype)
        if dtype is not None:
            result = [value.cast(dtype) for value in result]
        return result
    if dtype is not None:
        raise TypeError("where dtype is only valid for the coordinate overload")
    if x is None or y is None:
        raise TypeError("where requires both x and y")
    result = _try_dispatch("tensor.where", condition, x, y)
    return _native_where(condition, x, y) if result is None else result

def floor_int(x):
    result = _try_dispatch("tensor.floor_int", x)
    return _native_floor_int(x) if result is None else result

def sigmoid(x):
    result = _try_dispatch("tensor.sigmoid", x)
    return _native_sigmoid(x) if result is None else result

index.__doc__ = _native_index.__doc__

arg_reduce.__doc__ = _native_arg_reduce.__doc__

where.__doc__ = _native_where.__doc__

floor_int.__doc__ = _native_floor_int.__doc__

sigmoid.__doc__ = _native_sigmoid.__doc__

Var.index = index

Var.arg_reduce = arg_reduce

Var.where = where

Var.floor_int = floor_int

Var.sigmoid = sigmoid

def erfinv_(x):
    ''' In-place version of erfinv().
    '''
    return x.assign(x.erfinv())

Var.erfinv_ = erfinv_

def erf_(x):
    ''' In-place version of erf().
    '''
    return x.assign(x.erf())

Var.erf_ = erf_

def abs_(x):
    ''' In-place version of abs().
    '''
    return x.assign(x.abs())

Var.abs_ = abs_

def sigmoid_(x):
    ''' In-place version of sigmoid().
    '''
    return x.assign(x.sigmoid())

Var.sigmoid_ = sigmoid_

def sqrt_(x):
    ''' In-place version of sqrt().
    '''
    return x.assign(x.sqrt())

Var.sqrt_ = sqrt_

def add_(x, y):
    ''' In-place version of add().
    '''
    return x.assign(x.add(y))

Var.add_ = add_

def multiply_(x, y):
    ''' In-place version of multiply().
    '''
    return x.assign(x.multiply(y))

Var.multiply_ = multiply_

def type_as(a, b):
    return a.unary(op=b.dtype)

Var.type_as = type_as

Var.astype = Var.cast

def masked_fill(x, mask, value):
    import jittor as jt
    return jt.ternary(mask, value, x)

Var.masked_fill = masked_fill

def sqr(x): return x*x

Var.sqr = sqr

def pow(x, y):
    ''' computes x^y, element-wise.

    This operation is equivalent to ``x ** y``.

    :param x: the first input.
    :type x: a python number or jt.Var.
    :param y: the second input.
    :type y: a python number or jt.Var.
    '''
    if isinstance(x,Var) and isinstance(y, (ori_int, ori_float)):
        if y == 2:
            return x.sqr()
        if y == 3 and _jittor_dtype_name(x.dtype) == "float32":
            return x*x*x
    return core.ops.pow(x, y)

Var.pow = Var.__pow__ = pow

def argmax(x: Var, dim: int, keepdims:bool=False):
    ''' Returns the indices and values of the maximum elements along the specified dimension.

    :param x: the input Var.
    :type x: jt.Var, numpy array, or python sequence.
    :param dim: the dimension to reduce.
    :type dim: int.
    :param keepdims: whether the output Var has dim retained or not. Defaults to False
    :type keepdims: bool, optional

    Example::

        >>> a = jt.randn((2, 4))
        >>> a
        jt.Var([[-0.33272865 -0.4951588   1.4128606   0.13734372]
                [-1.633469    0.19593953 -0.7803732  -0.5260756 ]], dtype=float32)
        >>> a.argmax(dim=0)
        (jt.Var([0 1 0 0], dtype=int32), jt.Var([-0.33272865  0.19593953  1.4128606   0.13734372], dtype=float32))
        >>> a.argmax(dim=1)
        (jt.Var([2 1], dtype=int32), jt.Var([1.4128606  0.19593953], dtype=float32))
    '''
    import jittor as jt
    if dim is None:
        dim = 0
        x = x.flatten()
    elif hasattr(x, "shape"):
        nd = len(x.shape)
        if not (-nd <= dim < nd):
            # clear error instead of the cryptic cutt_transpose "axes != xdim"
            raise IndexError(f"argmax: dim {dim} out of range for a {nd}-D "
                             f"input (expected dim in [{-nd}, {nd-1}])")
        # normalize negative dim: arg_reduce's internal transpose miscomputes the
        # axes for negative dims other than -1 -> cryptic cutt_transpose crash
        if dim < 0:
            dim += nd
    return jt.arg_reduce(x, "max", dim, keepdims)

Var.argmax = argmax

def argmin(x, dim: int, keepdims:bool=False):
    ''' Returns the indices and values of the minimum elements along the specified dimension.

    :param x: the input Var.
    :type x: jt.Var, numpy array, or python sequence.
    :param dim: the dimension to reduce.
    :type dim: int.
    :param keepdims: whether the output Var has dim retained or not. Defaults to False
    :type keepdims: bool, optional

    Example::

        >>> a = jt.randn((2, 4))
        >>> a
        jt.Var([[-0.33272865 -0.4951588   1.4128606   0.13734372]
                [-1.633469    0.19593953 -0.7803732  -0.5260756 ]], dtype=float32)
        >>> a.argmin(dim=0)
        (jt.Var([1 0 1 1], dtype=int32), jt.Var([-1.633469  -0.4951588 -0.7803732 -0.5260756], dtype=float32))
        >>> a.argmin(dim=1)
        (jt.Var([1 0], dtype=int32), jt.Var([-0.4951588 -1.633469 ], dtype=float32))
    '''
    import jittor as jt
    if dim is not None and hasattr(x, "shape"):
        nd = len(x.shape)
        if not (-nd <= dim < nd):
            raise IndexError(f"argmin: dim {dim} out of range for a {nd}-D "
                             f"input (expected dim in [{-nd}, {nd-1}])")
        if dim < 0:
            dim += nd
    return jt.arg_reduce(x, "min", dim, keepdims)

Var.argmin = argmin

def randn(*size, dtype="float32", requires_grad=True) -> Var:
    ''' samples random numbers from a standard normal distribution.

    :param size: shape of the output.
    :type size: int or a sequence of int

    :param dtype: data type, defaults to "float32".
    :type dtype: str, optional

    :param requires_grad: whether to enable gradient back-propgation, defaults to True.
    :type requires_grad: bool, optional

    Example::

        >>> jt.randn(3)
        jt.Var([-1.019889   -0.30377278 -1.4948598 ], dtype=float32)
        >>> jt.randn(2, 3)
        jt.Var([[-0.15989183 -1.5010914   0.5476955 ]
         [-0.612632   -1.1471151  -1.1879086 ]], dtype=float32)
    '''
    import jittor as jt
    if isinstance(size, tuple) and isinstance(size[0], (tuple, list, NanoVector)): size = size[0]
    for dim in size:
        if dim < 0:
            raise RuntimeError(f"Trying to create tensor with negative dimension {dim}: {size}")
    arr = jt.random(size, dtype, "normal")
    if not requires_grad: return arr.stop_grad()
    return arr

def rand(*size, dtype="float32", requires_grad=True) -> Var:
    ''' samples random numbers from a uniform distribution on the interval [0, 1).

    :param size: shape of the output.
    :type size: int or a sequence of int

    :param dtype: data type, defaults to "float32".
    :type dtype: str, optional

    :param requires_grad: whether to enable gradient back-propgation. defaults to True.
    :type requires_grad: bool, optional

    Example::

        >>> jt.rand(3)
        jt.Var([0.31005102 0.02765604 0.8150749 ], dtype=float32)
        >>> jt.rand(2, 3)
        jt.Var([[0.96414304 0.3519264  0.8268017 ]
         [0.05658621 0.04449705 0.86190987]], dtype=float32)
    '''
    import jittor as jt
    if isinstance(size, tuple) and isinstance(size[0], (tuple, list, NanoVector)): size = size[0]
    arr = jt.random(size, dtype)
    if not requires_grad: return arr.stop_grad()
    return arr

def rand_like(x, dtype=None) -> Var:
    ''' samples random values from standard uniform distribution with the same shape as x.

    :param x: reference variable.
    :type x: jt.Var

    :param dtype: if None, the dtype of the output is the same as x.
        Otherwise, use the specified dtype. Defaults to None.
    :type dtype: str, optional

    Example::

        >>> x = jt.zeros((2, 3))
        >>> jt.rand_like(x)
        jt.Var([[0.6164821  0.21476883 0.61959815]
         [0.58626485 0.35345772 0.5638483 ]], dtype=float32)
    '''
    import jittor as jt
    if dtype is None: dtype = x.dtype
    return jt.random(x.shape, dtype)

def randn_like(x, dtype=None) -> Var:
    ''' samples random values from standard normal distribution with the same shape as x.

    :param x: reference variable.
    :type x: jt.Var

    :param dtype: if None, the dtype of the output is the same as x.
        Otherwise, use the specified dtype. Defaults to None.
    :type dtype: str, optional

    Example::

        >>> x = jt.zeros((2, 3))
        >>> jt.randn_like(x)
        jt.Var([[-1.1647032   0.34847224 -1.3061888 ]
         [ 1.068085   -0.34366122  0.13172573]], dtype=float32)
    '''
    import jittor as jt
    if dtype is None: dtype = x.dtype
    return jt.random(x.shape, dtype, "normal")

def randint(low, high=None, shape=(1,), dtype="int32") -> Var:
    ''' samples random integers from a uniform distribution on the interval [low, high).

    :param low: lowest intergers to be drawn from the distribution, defaults to 0.
    :type low: int, optional

    :param high: One above the highest integer to be drawn from the distribution.
    :type high: int

    :param shape: shape of the output size, defaults to (1,).
    :type shape: tuple, optional

    :param dtype: data type of the output, defaults to "int32".
    :type dtype: str, optional

    Example::

        >>> jt.randint(3, shape=(3, 3))
        jt.Var([[2 0 2]
                [2 1 2]
                [2 0 1]], dtype=int32)
        >>> jt.randint(1, 3, shape=(3, 3))
        jt.Var([[2 2 2]
                [1 1 2]
                [1 1 1]], dtype=int32)
    '''
    import jittor as jt
    if high is None: low, high = 0, low
    for dim in shape:
        if dim < 0:
            raise RuntimeError(f"Trying to create tensor with negative dimension {dim}: {shape}")
    v = (jt.random(shape) * (high - low) + low).clamp(low, high-0.5)
    v = jt.floor_int(v)
    return v.astype(_jittor_dtype_name(dtype))

def randint_like(x, low, high=None) -> Var:
    ''' samples random values from standard normal distribution with the same shape as x.

    :param x: reference variable.
    :type x: jt.Var

    :param low: lowest intergers to be drawn from the distribution, defaults to 0.
    :type low: int, optional

    :param high: One above the highest integer to be drawn from the distribution.
    :type high: int

    Example::

        >>> x = jt.zeros((2, 3))
        >>> jt.randint_like(x, 10)
        jt.Var([[9. 3. 4.]
                [4. 8. 5.]], dtype=float32)
        >>> jt.randint_like(x, 10, 20)
        jt.Var([[17. 11. 18.]
                [14. 17. 15.]], dtype=float32)
     '''

    return randint(low, high, x.shape, x.dtype)

def normal(mean, std, size=None, dtype="float32") -> Var:
    ''' samples random values from a normal distribution.

    :param mean: means of the normal distributions.
    :type mean: int or jt.Var

    :param std: standard deviations of the normal distributions.
    :type std: int or jt.Var

    :param size: shape of the output size. if not specified, the
        shape of the output is determined by mean or std. Exception will be
        raised if mean and std are all integers or have different shape in
        this case. Defaults to None
    :type size: tuple, optional

    :param dtype: data type of the output, defaults to "float32".
    :type dtype: str, optional

    Example::

        >>> jt.normal(5, 3, size=(2,3))
        jt.Var([[ 8.070848   7.654219  10.252696 ]
                [ 6.383718   7.8817277  3.0786133]], dtype=float32)
        >>> mean = jt.randint(low=0, high=10, shape=(10,))
        >>> jt.normal(mean, 0.1)
        jt.Var([1.9524184 1.0749301 7.9864206 5.9407325 8.1596155 4.824019  7.955083
                8.972998  6.0674286 8.88026  ], dtype=float32)
    '''
    import jittor as jt
    if size is None:
        if isinstance(mean, Var) and isinstance(std, Var):
            if mean.shape != std.shape:
                raise ValueError("normal: mean and std tensors must have matching shapes")
            size = mean.shape
        else:
            if isinstance(mean, Var): size = mean.shape
            if isinstance(std, Var): size = std.shape
    return jt.init.gauss(size, dtype, mean, std)

def attrs(var):
    return {
        "is_stop_fuse": var.is_stop_fuse(),
        "is_stop_grad": var.is_stop_grad(),
        "shape": var.shape,
        "dtype": var.dtype,
    }

Var.attrs = attrs

def fetch(*args):
    ''' Async fetch vars with function closure.

Example 1::

    for img,label in enumerate(your_dataset):
        pred = your_model(img)
        loss = critic(pred, label)
        acc = accuracy(pred, label)
        jt.fetch(acc, loss,
            lambda acc, loss:
                print(f"loss:{loss} acc:{acc}"
        )

Example 2::

    for i,(img,label) in enumerate(your_dataset):
        pred = your_model(img)
        loss = critic(pred, label)
        acc = accuracy(pred, label)
        # variable i will be bind into function closure
        jt.fetch(i, acc, loss,
            lambda i, acc, loss:
                print(f"#{i}, loss:{loss} acc:{acc}"
        )
    '''
    if len(args) < 1:
        raise ValueError("fetch requires at least one Var and a callback")
    func = args[-1]
    if not callable(func):
        raise TypeError("fetch callback must be callable")
    args = list(args[:-1])
    if len(args)>0 and isinstance(args[0], Sequence) \
        and len(args[0])>=1 and isinstance(args[0][0], Var):
        raise TypeError("jt.Var should not inside a list or tuple.")

    var_map = []
    variables = []
    for i, v in enumerate(args):
        if isinstance(v, Var):
            variables.append(v)
            var_map.append(i)
            args[i] = None
    def callback(*results):
        for i,v in enumerate(results):
            args[var_map[i]] = v
        func(*args)
    core.ops.fetch(variables, callback)
    core.submit_pending_fetches()

Var.fetch = fetch

def vtos(v):
    data_str = f"jt.Var({v.numpy()}, dtype={_jittor_dtype_name(v.dtype)})"
    data_str = data_str.replace("\n", "\n       ")
    return data_str

Var.__str__ = vtos

Var.__repr__ = vtos

Var.peek = lambda x: f"{_jittor_dtype_name(x.dtype)}{x.shape}"

def size(v, dim=None):
    if dim is None:
        return v.shape
    return v.shape[dim]

Var.size = size

def to_int(v):
    return ori_int(v.item())

def to_float(v):
    return ori_float(v.item())

def to_bool(v):
    if not (v.dtype.is_int() or v.dtype.is_bool()):
        raise TypeError("bool conversion requires an integer or boolean Var")
    return ori_bool(v.item())

Var.__int__ = to_int

Var.__float__ = to_float

Var.__bool__ = to_bool

def format(v, spec):
    return v.item().__format__(spec)

Var.__format__ = format

def get_len(var):
    return var.shape[0]

Var.__len__ = get_len

int = int32

Var.int = Var.int32

Var.long = Var.int32

float = float32

Var.float = Var.float32

double = float64

Var.double = Var.float64

half = float16

Var.half = Var.float16

def is_var(v):
    return isinstance(v, Var)

def _var__array__(self, dtype=None, copy=None):
    a = self.numpy()
    if dtype is not None:
        a = a.astype(_jittor_dtype_name(dtype))
    return a

Var.__array__ = _var__array__

Var.__array_priority__ = 2000

Var.__module__ = "jittor"

Var.__reduce__ = lambda self: (Var, (self.data,))

__all__ = (
    'abs_',
    'add_',
    'amp_flags',
    'arg_reduce',
    'argmax',
    'argmin',
    'array',
    'array64',
    'attrs',
    'cast',
    'clamp',
    'clamp_',
    'detach',
    'double',
    'empty',
    'erf_',
    'erfinv_',
    'fetch',
    'flatten',
    'float',
    'float_auto',
    'floor_int',
    'format',
    'full',
    'full_like',
    'get_len',
    'grad',
    'half',
    'int',
    'index',
    'is_var',
    'masked_fill',
    'multiply_',
    'ne',
    'new_empty',
    'new_full',
    'new_ones',
    'new_zeros',
    'norm',
    'normal',
    'ones',
    'ones_like',
    'origin_reshape',
    'origin_transpose',
    'outer',
    'permute',
    'pow',
    'rand',
    'rand_like',
    'randint',
    'randint_like',
    'randn',
    'randn_like',
    'random',
    'reshape',
    'sigmoid',
    'sigmoid_',
    'size',
    'sqr',
    'sqrt_',
    'squeeze',
    'std',
    'to_bool',
    'to_device',
    'to_float',
    'to_int',
    'transpose',
    'type_as',
    'unsqueeze',
    'var',
    'view',
    'vtos',
    'where',
    'zeros',
    'zeros_like',
)

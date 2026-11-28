"""Recognize the adjacent-pair RoPE expression built from stock Tensor ops."""

import jittor as jt
from jittor._core.dtypes import dtype_name
from jittor._runtime.core_api import _output_requires_grad
from jittor._runtime.dispatch import try_dispatch


def _stamp(value):
    return (value.var_ptr, value._autograd_version())


def _tag(value, stage, *operands):
    # Match only pending expressions with the same holders and write versions.
    attr = "_jt_rope_sin" if stage == "sin" else "_jt_interleaved_rope"
    setattr(value, attr, (stage, _stamp(value), operands, tuple(_stamp(t) for t in operands)))
    return value


def _read_tag(value, stage, tag):
    if tag is not None and tag[0] == stage and not value.is_finished and _stamp(value) == tag[1]:
        if all(_stamp(t) == stamp for t, stamp in zip(tag[2], tag[3])):
            return tag[2]
    return None


def _without_clone(value):
    while value._producer_name() == "clone":
        value = value._input(0)
    return value


def tag_stack(values, dim, out):
    if len(values) != 2 or not all(isinstance(v, jt.Var) for v in values):
        return out
    negative, real = values
    if (dim not in (-1, real.ndim) or not jt.flags.use_cuda or jt.flags.amp_reg
            or _output_requires_grad(negative, real)):
        return out
    negative = _without_clone(negative)
    if negative._producer_unary() != "negative":
        return out
    imag, real = _without_clone(negative._input(0)), _without_clone(real)
    if real._producer_name() != "getitem" or imag._producer_name() != "getitem":
        return out
    packed = real._input(0)
    dims = tuple(packed.shape)
    if not (len(dims) >= 2 and dims[-1] == 2 and dims[-2]
            and packed.var_ptr == imag._input(0).var_ptr
            and tuple(real.shape) == tuple(imag.shape) == dims[:-1]):
        return out
    strides = tuple(packed._storage_strides())
    offset = packed._storage_offset()
    if (tuple(real._storage_strides()) != strides[:-1]
            or tuple(imag._storage_strides()) != strides[:-1]
            or real._storage_offset() != offset
            or imag._storage_offset() != offset + strides[-1]):
        return out
    packed = _without_clone(packed)
    if packed._producer_name() != "reshape":
        return out
    x = packed._input(0)
    if (tuple(x.shape) != dims[:-2] + (dims[-2] * 2,)
            or dtype_name(x.dtype) not in ("float32", "bfloat16", "float16")):
        return out
    return _tag(out, "stack", x)


def flatten(input, start_dim=0, end_dim=-1):
    tagged = _read_tag(input, "stack", getattr(input, "_jt_interleaved_rope", None))
    out = jt.flatten(input, start_dim, end_dim)
    if tagged is not None and out.shape == tagged[0].shape:
        return _tag(out, "rotated", *tagged)
    return out


def tag_float(value, out):
    tagged = _read_tag(value, "rotated", getattr(value, "_jt_interleaved_rope", None))
    return _tag(out, "float", *tagged) if tagged is not None else out


def tag_product(rotated, sin, out, tag):
    tagged = _read_tag(rotated, "float", tag)
    if (tagged is not None and isinstance(sin, jt.Var)
            and dtype_name(sin.dtype) == dtype_name(out.dtype) == "float32"):
        # Keep the graph operand when a mutable Python holder is rebound.
        return _tag(out, "sin", tagged[0], sin._alias())
    return out


def _cos_operand(term, x):
    # Binary codegen may permute a transposed input, then restore its axes.
    axes = tuple(term._producer_transpose_axes())
    if axes:
        term = term._input(0)
    if term._producer_op() != "binary.multiply":
        return None
    operands = term._inputs()
    if len(operands) != 2:
        return None
    inverse = tuple(axes.index(i) for i in range(len(axes)))
    for value, cos in (operands, operands[::-1]):
        if axes and dtype_name(x.dtype) != "float32" and value._producer_unary() == "float32":
            value = value._input(0)
        if axes:
            if (tuple(value._producer_transpose_axes()) != inverse
                    or tuple(cos._producer_transpose_axes()) != inverse):
                continue
            value, cos = value._input(0), cos._input(0)
        if value.var_ptr != x.var_ptr and value._producer_unary() == "float32":
            value = value._input(0)
        if value.var_ptr == x.var_ptr:
            while cos._producer_name() == "broadcast_to":
                parent = cos._input(0)
                # Only remove a right-aligned broadcast, as used by multiply.
                extra = cos.ndim - parent.ndim
                shape = (1,) * extra + tuple(parent.shape)
                strides = (0,) * extra + tuple(parent._storage_strides())
                expected = tuple(s if a == b else 0 for a, b, s in zip(shape, cos.shape, strides))
                if (len(shape) != cos.ndim or any(a not in (1, b) for a, b in zip(shape, cos.shape))
                        or any(s != t for a, s, t in zip(cos.shape, expected, cos._storage_strides()) if a != 1)):
                    break
                cos = parent
            return cos
    return None


def sum_operands(cos_term, sin_term, tag):
    tagged = _read_tag(sin_term, "sin", tag)
    if tagged is not None and isinstance(cos_term, jt.Var):
        x, sin = tagged
        if (cos_term.shape == sin_term.shape == x.shape and dtype_name(cos_term.dtype) == "float32"
                and not _output_requires_grad(cos_term, sin_term)):
            cos = _cos_operand(cos_term, x)
            if cos is not None:
                return x, cos, sin
    return None


def tag_sum(out, operands):
    return _tag(out, "sum", *operands)


def fuse(operands):
    return try_dispatch("nn.interleaved_rope.inference", *operands)


def cast(value, dtype, tag):
    tagged = _read_tag(value, "sum", tag)
    if tagged is not None and dtype_name(tagged[0].dtype) == dtype:
        return fuse(tagged)
    return None

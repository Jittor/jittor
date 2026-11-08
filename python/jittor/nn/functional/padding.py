"""Functional tensor padding."""

import operator

from jittor._core.dtypes import dtype_name

from ..backends import hooks as _backend_hooks


def pad(x, padding=None, mode="constant", value=0, pad=None):
    import jittor as jt

    # Torch spells the amounts argument ``pad``; Jittor historically used
    # ``padding``. Keep both spellings on the canonical function.
    if padding is None:
        padding = pad
    assert mode in ("constant", "replicate", "reflect", "circular"), (
        "only support constant,replicate,reflect,circular pad"
    )
    assert len(padding) % 2 == 0 and len(padding) // 2 <= x.ndim

    # Widths are shape metadata. Normalize before either backend dispatch or
    # reindex expression construction; formatting a Var is not an index, and
    # the ACL constant-pad route deliberately accepts integer widths only.
    widths = []
    for width in padding:
        if isinstance(width, bool):
            raise TypeError("pad widths must be integers, not bool")
        if isinstance(width, jt.Var):
            # Var.__index__ historically truncates floating point values.
            # Guard it here without changing the global tensor protocol.
            integral = dtype_name(width.dtype) in ("int8", "uint8", "int16", "int32", "int64")
            if not integral or width.numel() != 1:
                raise TypeError("pad tensor widths must contain exactly one integer")
        widths.append(operator.index(width))
    padding = widths
    left = [0] * (x.ndim - len(padding) // 2) + padding[::2][::-1]
    right = [0] * (x.ndim - len(padding) // 2) + padding[1::2][::-1]

    if mode == "constant":
        acl_pad = _backend_hooks.acl_constant_pad
        if acl_pad is not None:
            result = acl_pad(x, padding, value)
            if result is not None:
                return result

    out_dims = []
    out_shape = []
    for index, size, before, after in zip(range(x.ndim), x.shape, left, right):
        out_shape.append(int(size) + int(before) + int(after))
        # Widths may be negative (cropping). Parenthesize numeric operands:
        # an unquoted i2-<before=-5> becomes i2--5, a C++ decrement token.
        if mode == "constant":
            out_dims.append("i{}-({})".format(index, before))
        elif mode == "replicate":
            out_dims.append(
                "i{0}<({1}) ? 0 : i{0} > ({2}) ? ({3}) : i{0}-({1})".format(
                    index, before, size + before - 1, size - 1
                )
            )
        elif mode == "reflect":
            out_dims.append(
                "i{0}<({1}) ? ({1})-i{0} : i{0} > ({2}) ? ({3})-i{0} : i{0}-({1})".format(
                    index,
                    before,
                    size + before - 1,
                    2 * (size - 1) + before,
                )
            )
        else:
            out_dims.append(
                "i{0}<({1}) ? ({2})+i{0} : i{0} > ({3}) ? i{0}-({4}) : i{0}-({1})".format(
                    index,
                    before,
                    size - before,
                    size + before - 1,
                    size + before,
                )
            )

    return x.reindex(out_shape, out_dims, overflow_value=float(value))


__all__ = ["pad"]

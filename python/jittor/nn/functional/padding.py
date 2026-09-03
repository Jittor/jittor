"""Functional tensor padding."""

import jittor as jt

from ..backends import hooks as _backend_hooks


def pad(x, padding=None, mode="constant", value=0, pad=None):
    # Torch spells the amounts argument ``pad``; Jittor historically used
    # ``padding``. Keep both spellings on the canonical function.
    if padding is None:
        padding = pad
    assert mode in ("constant", "replicate", "reflect", "circular"), (
        "only support constant,replicate,reflect,circular pad"
    )
    assert len(padding) % 2 == 0 and len(padding) // 2 <= x.ndim

    padding = list(padding)
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
        if mode == "constant":
            out_dims.append("i{}-{}".format(index, before))
        elif mode == "replicate":
            out_dims.append(
                "i{0}<{1} ? 0 : i{0} > {2} ? {3} : i{0}-{1}".format(
                    index, before, size + before - 1, size - 1
                )
            )
        elif mode == "reflect":
            out_dims.append(
                "i{0}<{1} ? {1}-i{0} : i{0} > {2} ? {3}-i{0} : i{0}-{1}".format(
                    index,
                    before,
                    size + before - 1,
                    2 * (size - 1) + before,
                )
            )
        else:
            out_dims.append(
                "i{0}<{1} ? {2}+i{0} : i{0} > {3} ? i{0}-{4} : i{0}-{1}".format(
                    index,
                    before,
                    size - before,
                    size + before - 1,
                    size + before,
                )
            )

    return x.reindex(out_shape, out_dims, overflow_value=float(value))


__all__ = ["pad"]

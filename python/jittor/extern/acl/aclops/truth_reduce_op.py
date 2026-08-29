import jittor as jt


def _normalize_dims(input, dim):
    if dim is None or (isinstance(dim, (list, tuple)) and not dim):
        return list(range(input.ndim))
    if isinstance(dim, int):
        dims = [dim]
    elif isinstance(dim, (list, tuple)):
        dims = list(dim)
    else:
        raise TypeError("dim must be an int, tuple, list, or None")

    normalized = []
    for axis in dims:
        if not isinstance(axis, int):
            raise TypeError("each reduction dimension must be an int")
        if axis < 0:
            axis += input.ndim
        if not 0 <= axis < input.ndim:
            raise ValueError(
                f"dimension {axis} out of range for tensor with {input.ndim} dimensions")
        if axis in normalized:
            raise ValueError(f"dimension {axis} appears more than once")
        normalized.append(axis)
    return normalized


def _truth_reduce_cmd(input, dims, reduce_all):
    output_shape = [
        size for axis, size in enumerate(input.shape) if axis not in dims
    ] or [1]
    output = jt.empty(output_shape, dtype="bool")
    axes = ", ".join(map(str, dims))
    return jt.code(
        outputs=[output],
        inputs=[input],
        cuda_header='#include "acl/aclops/aclops.h"',
        cuda_src=f"""
        // aclop
        TruthReduceOpRunner op({str(reduce_all).lower()});
        op.add(in0, true);
        op.add(out0, false);
        ReduceAttr *attr = new ReduceAttr();
        attr->axes = {{{axes}}};
        attr->keepdims = false;
        op.op_attr.reset(attr);
        op.jt_name = "{'all' if reduce_all else 'any'}";
        op.run();
        """,
    )[0]


def truth_reduce(input, dim, reduce_all):
    dims = _normalize_dims(input, dim)
    truth = input if input.dtype == "bool" else input != 0
    return _truth_reduce_cmd(truth, dims, reduce_all)

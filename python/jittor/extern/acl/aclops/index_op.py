import os
from jittor_utils import env_or_try_find
import jittor_utils
import ctypes
import glob
import jittor.compiler as compiler
import jittor as jt
import math
import numpy as np

from typing import Union
from collections.abc import Sequence, Iterable


def range_forward(name: str,
                  inputs: list,
                  output_dtypes: list = None,
                  output_shapes: list = None,
                  attr_code: str = "",
                  attr_header: str = "",
                  outputs: list = None,
                  extra_data: dict = {}):
    attr_header = "\nnamespace jittor{" + attr_header + "}\n"

    cuda_header = '''
    #include "acl/aclops/aclops.h"
    '''
    outputs_ = []
    if outputs is not None:
        outputs_ = outputs
    else:
        assert output_dtypes is not None
        assert output_shapes is not None
        assert len(output_dtypes) == len(output_shapes)
        for i in range(len(output_shapes)):
            outputs_.append(jt.empty(output_shapes[i], output_dtypes[i]))
    input_code = ''
    for i in range(len(inputs)):
        input_code += f"op.add(in{i}, true);\n"

    output_code = ''
    for i in range(len(outputs_)):
        output_code += f"op.add(out{i}, false);\n"
    return jt.code(outputs=outputs_,
                   inputs=inputs,
                   cuda_header=attr_header + cuda_header,
                   cuda_src=f"""
    // aclop
    {name}OpRunner op;
    {input_code}
    op.add(out0, false);
    {attr_code}
    op.run();""",
                   data=extra_data)


class IndexACL(jt.Function):

    def __init__(self):
        super(IndexACL, self).__init__()

    def execute(self, inshape: list, dim=None, dtype="int32"):
        # zeros a tensor, shape is inshape, dtype is dtype
        dim_input = dim
        if dim == None:
            dim = [i for i in range(len(inshape))]
        elif type(dim) == int:
            dim = [dim]
        results = []
        extra_data = {}
        extra_data["dim_count"] = len(dim)

        for i, d in enumerate(dim):
            max_len = inshape[d]

            extra_data[f"dim_{i}_start"] = 0
            extra_data[f"dim_{i}_end"] = max_len
            extra_data[f"dim_{i}_step"] = 1

            tmp = jt.zeros(max_len, dtype=dtype)
            range_attr_code = f"""
            op.jt_name = "range";
            RangeAttr *attr = new RangeAttr();
            attr->start = data["dim_{i}_start"];
            attr->end = data["dim_{i}_end"];
            attr->step = data["dim_{i}_step"];
            op.op_attr.reset(attr);
            """
            result = range_forward("Range", [],
                                   output_dtypes=[tmp.dtype],
                                   output_shapes=[tmp.shape],
                                   attr_code=range_attr_code,
                                   extra_data=extra_data)[0]
            broadcast_dims = list(range(len(inshape)))
            broadcast_dims.remove(d)
            result = jt.broadcast(result, shape=inshape, dims=broadcast_dims)
            results.append(result)

        if len(results) != 1 or dim_input == None:
            return tuple(results)
        elif len(results) == 1 and dim_input != None:
            return results[0]
        else:
            return results

    def grad(self, grad_output):
        return grad_output

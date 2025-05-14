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


def where_cmd(name: str,
              inputs: list,
              output_dtypes: list = None,
              output_shapes: list = None,
              attr_code: str = "",
              attr_header: str = "",
              outputs: list = None):
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
    {output_code}
    {attr_code}
    op.run();""")


class NonzeroACL(jt.Function):

    def __init__(self):
        super(NonzeroACL, self).__init__()

    def execute(self, x):
        attr_code = f"""
        op.jt_name = "nonzero";
        """
        nonzero_cnt = (x != 0.0).sum().item()

        result = where_cmd("Nonzero", [x],
                           output_dtypes=['int64'],
                           output_shapes=[(nonzero_cnt, x.ndim)],
                           attr_code=attr_code)[0]

        return result

    def grad(self, grad_output):
        return grad_output


class WhereACL(jt.Function):

    def __init__(self):
        super(WhereACL, self).__init__()

    def execute(self, condition, x=None, y=None):
        # case 1 (unary)
        if y is None:
            self.unary = True

            # In this case, `condition` is the input, while `x` is dtype
            result = NonzeroACL()(condition).t()
            result = [result[i] for i in range(result.size(0))]
            return result
            # The return value should be a tuple, but even we set to tuple here, it will be convert to a list in `Function.__call__`.

        # case 2 (cond ? x : y)
        else:
            self.condition = condition

            if x.dtype != y.dtype:
                if x.dtype == jt.float32:
                    y = y.float32()
                elif y.dtype == jt.float32:
                    x = x.float32()
                else:
                    x = x.to(y.dtype)

            self.x = x
            self.y = y

            result = where_cmd("Where", [condition, x, y],
                               output_dtypes=[x.dtype],
                               output_shapes=[x.shape],
                               attr_code="op.jt_name=\"where\";")[0]
            return result

    def grad(self, grad_output):
        if hasattr(self, 'unary') and self.unary:
            return grad_output
        else:
            tmp = jt.zeros(grad_output.shape, dtype=grad_output.dtype)
            grad_x = where_cmd("Where", [self.condition, grad_output, tmp],
                               output_dtypes=[self.x.dtype],
                               output_shapes=[self.x.shape],
                               attr_code="op.jt_name=\"where\";")[0]

            grad_y = where_cmd("Where", [self.condition, tmp, grad_output],
                               output_dtypes=[self.y.dtype],
                               output_shapes=[self.y.shape],
                               attr_code="op.jt_name=\"where\";")[0]
            return grad_output, grad_x, grad_y

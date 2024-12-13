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

def cumsum_cmd(name: str,
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

class CumsumACL(jt.Function):

    def __init__(self):
        super(CumsumACL, self).__init__()

    def execute(self, input, dim=-1):
        self.dim = dim
        attr_code = f"""
        op.jt_name = "cumsum";
        GatherAttr *attr = new GatherAttr();
        attr->dim = {dim};
        op.op_attr.reset(attr);
        """
        result = cumsum_cmd("Cumsum", [input],
                            output_dtypes=[input.dtype],
                            output_shapes=[input.shape],
                            attr_code=attr_code)[0]
        return result

    def grad(self, grad_output):
        cumsum_attr_code = f"""
        op.jt_name = "cumsum";
        GatherAttr *attr = new GatherAttr();
        attr->dim = {self.dim};
        op.op_attr.reset(attr);
        """
        flip_attr_code = f"""
        op.jt_name = "flip";
        ReduceAttr *attr = new ReduceAttr();
        attr->axes = {{{self.dim}}};
        attr->prod_dim = {{{1}}};
        op.op_attr.reset(attr);
        """
        flipped_grad_output = cumsum_cmd("Flip", [grad_output],
                                        output_dtypes=[grad_output.dtype],
                                        output_shapes=[grad_output.shape],
                                        attr_code=flip_attr_code)[0]
        cumulative_grad = cumsum_cmd("Cumsum", [flipped_grad_output],
                                    output_dtypes=[grad_output.dtype],
                                    output_shapes=[grad_output.shape],
                                    attr_code=cumsum_attr_code)[0]
        grad_input = cumsum_cmd("Flip", [cumulative_grad],
                                output_dtypes=[grad_output.dtype],
                                output_shapes=[grad_output.shape],
                                attr_code=flip_attr_code)[0]
        return grad_input
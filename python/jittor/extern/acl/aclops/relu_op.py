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


def relu_cmd(name: str,
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


class ReLUACL(jt.Function):

    def __init__(self):
        super(ReLUACL, self).__init__()

    def execute(self, x):
        x = x.float32()
        self.input = x
        result = relu_cmd("ReLU", [x],
                          output_dtypes=[x.dtype],
                          output_shapes=[x.shape],
                          attr_code="op.jt_name=\"unary\";")[0]
        return result

    def grad(self, grad_output):
        mask = relu_cmd("Greater",
                        [self.input, jt.zeros(self.input.shape)],
                        output_dtypes=[self.input.dtype],
                        output_shapes=[self.input.shape],
                        attr_code="op.jt_name=\"binary\";")[0]
        grad_input = relu_cmd("Mul", [grad_output, mask],
                              output_dtypes=[grad_output.dtype],
                              output_shapes=[grad_output.shape],
                              attr_code="op.jt_name=\"binary\";")[0]
        return grad_input


class LeakyReLUACL(jt.Function):

    def __init__(self):
        super(LeakyReLUACL, self).__init__()

    def execute(self, x, negative_slope=0.01):
        x = x.float32()
        self.input = x
        attr_code = f"""
        op.jt_name = "leakyrelu";
        LeakyReluAttr *attr = new LeakyReluAttr();
        attr->negativeSlope = {negative_slope};
        op.op_attr.reset(attr);
        """
        result = relu_cmd("LeakyReLU", [x],
                          output_dtypes=[x.dtype],
                          output_shapes=[x.shape],
                          attr_code=attr_code)[0]
        self.negative_slope = negative_slope
        return result

    def grad(self, grad_output):
        attr_code = f"""
        op.jt_name = "leakyrelubackward";
        LeakyReluAttr *attr = new LeakyReluAttr();
        attr->negativeSlope = {self.negative_slope};
        attr->selfIsResult = false;
        op.op_attr.reset(attr);
        """
        grad_input = relu_cmd("LeakyReLUBackward", [grad_output, self.input],
                              output_dtypes=[grad_output.dtype],
                              output_shapes=[grad_output.shape],
                              attr_code=attr_code)[0]
        return grad_input

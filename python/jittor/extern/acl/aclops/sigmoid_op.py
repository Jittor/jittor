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


def sigmoid_cmd(name: str,
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


class SigmoidACL(jt.Function):

    def __init__(self):
        super(SigmoidACL, self).__init__()

    def execute(self, x):
        x = x.float32()
        inputs = [x]
        outputs = [jt.empty(x.shape, x.dtype)]
        attr_code = f"""
        op.jt_name = "sigmoid";
        """
        result = sigmoid_cmd("Sigmoid",
                             inputs=inputs,
                             outputs=outputs,
                             attr_code=attr_code)[0]
        self.output = result
        return result

    def grad(self, grad_output):
        attr_code = f"""
        op.jt_name = "sigmoidbackward";
        """
        inputs = [grad_output, self.output]
        outputs = [jt.empty(grad_output.shape, grad_output.dtype)]
        grad_input = sigmoid_cmd("SigmoidBackward",
                                 inputs=inputs,
                                 outputs=outputs,
                                 attr_code=attr_code)[0]
        return grad_input

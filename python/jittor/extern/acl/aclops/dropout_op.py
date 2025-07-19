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


def dropout_cmd(name: str,
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


class DropoutACL(jt.Function):

    def __init__(self):
        super(DropoutACL, self).__init__()

    def execute(self, x, p=0.5, is_train=False):
        self.input = x
        num_elements = x.numel()
        aligned_elements = (num_elements + 127) // 128 * 128
        mask_shape = (aligned_elements // 8, )
        attr_code = f"""
        op.jt_name = "dropout";
        DropoutAttr *attr = new DropoutAttr();
        attr->p = {p};
        attr->train = {"true" if is_train else "false"};
        attr->seed = 0;
        attr->offset = 0;
        op.op_attr.reset(attr);
        """
        result = dropout_cmd("Dropout", [x],
                             output_dtypes=[x.dtype, "uint8"],
                             output_shapes=[x.shape, mask_shape],
                             attr_code=attr_code)
        self.maskout = result[1]
        return result[0]

    def grad(self, grad_output):
        attr_code = f"""
        op.jt_name = "dropoutbackward";
        DropoutAttr *attr = new DropoutAttr();
        attr->scale = 1.0;
        op.op_attr.reset(attr);
        """
        grad_input = dropout_cmd("DropoutBackward",
                                 [grad_output, self.maskout],
                                 output_dtypes=[grad_output.dtype],
                                 output_shapes=[grad_output.shape],
                                 attr_code=attr_code)[0]
        return grad_input

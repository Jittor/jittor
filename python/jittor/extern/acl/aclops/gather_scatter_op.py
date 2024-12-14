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

def gather_scatter_cmd(name: str,
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

class GatherACL(jt.Function):

    def __init__(self):
        super(GatherACL, self).__init__()

    def execute(self, input, dim, index):
        self.dim = dim
        self.index = index
        attr_code = f"""
        op.jt_name = "gather";
        GatherAttr *attr = new GatherAttr();
        attr->dim = {dim};
        op.op_attr.reset(attr);
        """
        result = gather_scatter_cmd("Gather", [input, index],
                            output_dtypes=[input.dtype],
                            output_shapes=[index.shape],
                            attr_code=attr_code)[0]
        return result

    def grad(self, grad_output):
        tmp = jt.zeros(self.index.shape, dtype=grad_output.dtype)
        attr_code = f"""
        op.jt_name = "scatter";
        ScatterAttr *attr = new ScatterAttr();
        attr->axis = {self.dim};
        attr->reduction = {1};
        op.op_attr.reset(attr);
        """
        grad_input = gather_scatter_cmd("Scatter", [tmp, self.index, grad_output],
                                output_dtypes=[grad_output.dtype],
                                output_shapes=[tmp.shape],
                                attr_code=attr_code)[0]
        return grad_input

class ScatterACL(jt.Function):

    def __init__(self):
        super(ScatterACL, self).__init__()

    def execute(self, input, dim, index, src, reduce='void'):
        self.dim = dim
        self.index = index
        self.reduce = reduce
        attr_code = f"""
        op.jt_name = "scatter";
        ScatterAttr *attr = new ScatterAttr();
        attr->axis = {dim};
        attr->reduction = {1 if reduce == 'add' else 2 if reduce == 'mul' else 0};
        op.op_attr.reset(attr);
        """
        result = gather_scatter_cmd("Scatter", [input, self.index, src],
                            output_dtypes=[input.dtype],
                            output_shapes=[input.shape],
                            attr_code=attr_code)[0]
        return result

    def grad(self, grad_output):
        attr_code = f"""
        op.jt_name = "gather";
        GatherAttr *attr = new GatherAttr();
        attr->dim = {self.dim};
        op.op_attr.reset(attr);
        """
        grad_input = gather_scatter_cmd("Gather", [grad_output, self.index],
                                output_dtypes=[grad_output.dtype],
                                output_shapes=[self.index.shape],
                                attr_code=attr_code)[0]
        return grad_output, None, None, grad_input
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


def nantonum_cmd(name: str,
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


class NanToNumACL(jt.Function):

    def __init__(self):
        super(NanToNumACL, self).__init__()

    def execute(self, input, nan_or_inf):
        attr_code = f"""
        op.jt_name = "NanToNum";
        NanToNumAttr *attr = new NanToNumAttr();
        attr->nan = {nan_or_inf};
        attr->posinf = {-nan_or_inf};
        attr->neginf = {-nan_or_inf};
        op.op_attr.reset(attr);
        """
        self.attr_code = attr_code
        result = nantonum_cmd("NanToNum", [input],
                              output_dtypes=[input[0].dtype],
                              output_shapes=[input.shape],
                              attr_code=self.attr_code)[0]
        return result

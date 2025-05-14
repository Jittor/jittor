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


def rope_cmd(name: str,
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


class RopeACL(jt.Function):

    def __init__(self):
        super(RopeACL, self).__init__()

    def execute(self, xq, xk, freqs_cis, freq_cos, freq_sin):
        attr_code = f"""
        op.jt_name = "RotaryPosEmb";
        """
        if freqs_cis is not None:
            freq_cos = freqs_cis[..., 0]
            freq_sin = freqs_cis[..., 1]
        else:
            assert freq_cos is not None and freq_sin is not None
        inputs = [xq, xk, freq_cos, freq_sin]
        results = rope_cmd("RotaryPosEmb",
                           inputs,
                           output_dtypes=[
                               xq.dtype,
                           ],
                           output_shapes=[
                               xq.shape,
                           ],
                           attr_code=attr_code)
        results[0].sync()
        return inputs[0], inputs[1]

    def grad(self, grad_output):
        return grad_output

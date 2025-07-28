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

def embedding_cmd(name: str,
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
    
class EmbeddingACL(jt.Function):

    def __init__(self):
        super(EmbeddingACL, self).__init__()

    def execute(
        self,
        indices,
        weight,
    ):
        inputs = [weight, indices]
        self.indices = indices
        self.weight_shape = weight.shape
        output_shape = list(indices.shape) + list(weight.shape[1:])
        outputs = [jt.empty(output_shape, weight.dtype)]
        attr_code = f"""
        op.jt_name = "embedding";
        """
        result = embedding_cmd("Embedding",
                            inputs=inputs,
                            outputs=outputs,
                            attr_code=attr_code)[0]
        return result

    def grad(self, grad_output):
        inputs = [grad_output, self.indices]
        outputs = [jt.empty(self.weight_shape, grad_output.dtype)]
        attr_code = f"""
        op.jt_name = "embeddingbackward";
        EmbeddingAttr *attr = new EmbeddingAttr();
        attr->numEmbeddings = {self.weight_shape[0]};
        op.op_attr.reset(attr);
        """
        grad_weight = embedding_cmd("EmbeddingBackward",
                                inputs=inputs,
                                outputs=outputs,
                                attr_code=attr_code)[0]
        return None, grad_weight

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

def norms_cmd(name: str,
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
    
class BatchNormACL(jt.Function):

    def __init__(self,
                    num_features,
                    eps=1e-05,
                    momentum=0.1,
                    affine=True,
                    is_train=True,
                    sync=True):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.is_train = is_train
        self.sync = sync
        self.weight = jt.init.constant(
            (num_features, ), "float32", 1.0) if affine else 1.0
        self.bias = jt.init.constant(
            (num_features, ), "float32", 0.0) if affine else 0.0
        self.running_mean = jt.init.constant((num_features, ), "float32",
                                                0.0).stop_grad()
        self.running_var = jt.init.constant((num_features, ), "float32",
                                            1.0).stop_grad()

    def execute(self, x):
        # assert self.num_features == x.shape[-1]
        self.input = x.float32()
        inputs = [
            self.input, self.weight, self.bias, self.running_mean,
            self.running_var
        ]
        outputs = [
            jt.empty(x.shape),
            jt.empty(self.num_features),
            jt.empty(self.num_features)
        ]
        attr_code = f"""
        op.jt_name = "batchnorm";
        BatchNormAttr *attr = new BatchNormAttr();
        attr->is_train = {"true" if self.is_train else "false"};
        attr->momentum = {self.momentum};
        attr->eps = {self.eps};
        op.op_attr.reset(attr);
        """
        result = norms_cmd("BatchNorm",
                            inputs=inputs,
                            outputs=outputs,
                            attr_code=attr_code)
        self.output = result[0]
        self.saveMean = result[1]
        self.saveInvstd = result[2]
        return self.output

    def grad(self, grad_output):
        attr_code = f"""
        op.jt_name = "batchnorm";
        BatchNormAttr *attr = new BatchNormAttr();
        attr->is_train = {"true" if self.is_train else "false"};
        attr->momentum = {self.momentum};
        attr->eps = {self.eps};
        op.op_attr.reset(attr);
        """
        inputs = [
            grad_output, self.input, self.weight, self.running_mean,
            self.running_var, self.saveMean, self.saveInvstd
        ]
        outputs = [
            jt.empty(self.input.shape),
            jt.empty(self.num_features),
            jt.empty(self.num_features)
        ]
        grad_input = norms_cmd("BatchNormBackward",
                                inputs=inputs,
                                outputs=outputs,
                                attr_code=attr_code)[0]
        return grad_input
    

class LayerNormACL(jt.Function):

    def __init__(self,
                    normalized_shape,
                    eps: float = 1e-5,
                    elementwise_affine: bool = True):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = jt.init.constant(normalized_shape, "float32",
                                        1.0) if elementwise_affine else 1.0
        self.bias = jt.init.constant(normalized_shape, "float32",
                                        0.0) if elementwise_affine else 0.0

    def execute(self, x):
        self.input = x.float32()
        inputs = [self.input, self.weight, self.bias]
        outputs = [jt.empty(x.shape), jt.empty(x.shape), jt.empty(x.shape)]
        attr_code = f"""
        op.jt_name = "layernorm";
        LayerNormAttr *attr = new LayerNormAttr();
        attr->eps = {self.eps};
        attr->normalizedShape = {{{', '.join(map(str, (list(self.normalized_shape))))}}};
        attr->size = {x.shape[-1]};
        op.op_attr.reset(attr);
        """
        result = norms_cmd("LayerNorm",
                            inputs=inputs,
                            outputs=outputs,
                            attr_code=attr_code)
        self.output = result[0]
        self.meanout = result[1]
        self.rstdout = result[2]
        return self.output

    def grad(self, grad_output):
        attr_code = f"""
        op.jt_name = "batchnorm";
        BatchNormAttr *attr = new BatchNormAttr();
        attr->is_train = {"true" if self.is_train else "false"};
        attr->momentum = {self.momentum};
        attr->eps = {self.eps};
        op.op_attr.reset(attr);
        """
        inputs = [grad_output, self.input, self.weight, self.running_mean, self.running_var, self.saveMean, self.saveInvstd]
        outputs = [jt.empty(self.input.shape), jt.empty(self.num_features), jt.empty(self.num_features)]
        grad_input = norms_cmd("SoftmaxBackward",
                            inputs=inputs,
                            outputs=outputs,
                            attr_code=attr_code)[0]
        return grad_input

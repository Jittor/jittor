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


def concat_cmd(name: str,
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


class ConcatACL(jt.Function):

    def __init__(self):
        super(ConcatACL, self).__init__()

    def __call__(self, *args):
        assert isinstance(args[0], list)
        assert isinstance(args[1], int)
        if jt.flags.no_grad:
            return self.execute(*args)
        backup = args
        args = list(args)
        taped_inputs = []
        taped_outputs = []
        input_mask = [-1] * (len(args[0]) + 1)
        newargs = [list(), args[1]]
        for i, v in enumerate(args[0]):
            if isinstance(v, jt.Var):
                if v.is_stop_grad():
                    # -2 in input_mask represents it is stop_grad
                    input_mask[i] = -2
                    newargs[0].append(v)
                    continue
                v = v.tape()
                newargs[0].append(v)
                input_mask[i] = len(taped_inputs)
                taped_inputs.append(v)

        ori_res = self.execute(*newargs)
        if not isinstance(ori_res, Sequence):
            res = [ori_res]
        else:
            res = list(ori_res)
        output_mask = [-1] * len(res)
        for i, v in enumerate(res):
            if isinstance(v, jt.Var):
                v = v.tape()
                output_mask[i] = len(taped_outputs)
                res[i] = v
                taped_outputs.append(v)
        self.input_mask = input_mask
        self.output_mask = output_mask
        # tape output and input together so
        # backward treat them as one operator
        jt.tape_together(taped_inputs, taped_outputs, self._grad)
        if isinstance(ori_res, Sequence):
            return res
        else:
            return res[0]

    def execute(self, input_tensors, dim=0):
        for _ in input_tensors:
            if not (-_.ndim <= dim < _.ndim):
                print(_.shape, dim)
                raise ValueError("dim out of range")

        if dim < 0:
            dim += input_tensors[0].ndim

        self.input = input_tensors
        self.dim = dim
        for i in range(len(input_tensors)):
            if input_tensors[i].dtype != input_tensors[0].dtype:
                raise ValueError("All input tensors must have the same dtype")
            if input_tensors[i].shape[:dim] != input_tensors[
                    0].shape[:dim] or input_tensors[i].shape[
                        dim + 1:] != input_tensors[0].shape[dim + 1:]:
                raise ValueError("All input tensors must have the same shape")
        attr_code = f"""
        op.jt_name = "concat";
        ConcatAttr *attr = new ConcatAttr();
        attr->tensorNum = {len(input_tensors)};
        attr->dim = {dim};
        op.op_attr.reset(attr);
        """
        result = concat_cmd(
            "Concat",
            input_tensors,
            output_dtypes=[input_tensors[0].dtype],
            output_shapes=[
                jt.empty(self.calculate_output_shape(input_tensors, dim)).shape
            ],
            attr_code=attr_code)[0]
        return result

    def _grad(self, *args):
        new_args = ((args[i] if i >= 0 else None) for i in self.output_mask)
        ret = self.grad(*new_args)
        new_ret = []
        for i, r in enumerate(ret):
            j = self.input_mask[i]
            if j < 0:
                # -2 in input_mask represents it is stop_grad
                assert r is None or j==-2, f"{type(self)}'s {i}-th returned grad should be None, "\
                    "because the input value is not jittor variable."
            else:
                new_ret.append(r)
        return new_ret

    def grad(self, grad_output):
        grad_inputs = self.split_grad(grad_output, self.input, self.dim)
        return grad_inputs

    def calculate_output_shape(self, input_tensors, axis):
        shape = list(input_tensors[0].shape)
        for tensor in input_tensors[1:]:
            shape[axis] += tensor.shape[axis]
        return tuple(shape)

    def split_grad(self, grad_output, input_tensors, axis):
        offset = []
        shapeVec = []
        dtypeVec = []
        for tensor in input_tensors:
            offset.append(tensor.shape[axis])
            dtypeVec.append(tensor.dtype)
            shapeVec.append(tensor.shape)

        attr_code = f"""
        op.jt_name = "splitwithsize";
        auto *attr = new SplitWithSizeAttr();
        attr->splitSize = {{ {", ".join(map(str, offset))} }};
        attr->dim = {axis};
        op.op_attr.reset(attr);
        """

        result = concat_cmd("SplitWithSize", [grad_output],
                            output_dtypes=dtypeVec,
                            output_shapes=shapeVec,
                            attr_code=attr_code)
        return result

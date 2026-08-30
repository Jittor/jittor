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

from ._code import acl_code as concat_cmd


class ConcatACL:

    def __call__(self, input_tensors, dim=0):
        assert isinstance(input_tensors, (list, tuple))
        assert isinstance(dim, int)
        return self.execute(input_tensors, dim)

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
        split_sizes = [tensor.shape[dim] for tensor in input_tensors]
        grad_attr_code = f"""
        op.jt_name = "splitwithsize";
        auto *attr = new SplitWithSizeAttr();
        attr->splitSize = {{ {", ".join(map(str, split_sizes))} }};
        attr->dim = {dim};
        op.op_attr.reset(attr);
        """
        grad_outputs = "\n".join(
            f"op.add(out{i}, false);" for i in range(len(input_tensors)))
        result = concat_cmd(
            "Concat",
            input_tensors,
            output_dtypes=[input_tensors[0].dtype],
            output_shapes=[self.calculate_output_shape(input_tensors, dim)],
            attr_code=attr_code,
            multi_grad_src=f"""
            // aclop
            SplitWithSizeOpRunner op;
            op.add(dout, true);
            {grad_outputs}
            {grad_attr_code}
            op.run();
            """)[0]
        return result

    def calculate_output_shape(self, input_tensors, axis):
        shape = list(input_tensors[0].shape)
        for tensor in input_tensors[1:]:
            shape[axis] += tensor.shape[axis]
        return tuple(shape)

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

from ._code import acl_code as norms_cmd


class BatchNormACL:

    def __init__(self, eps=1e-5, momentum=0.1, is_train=False):
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.is_train = bool(is_train)

    def _attr_code(self, name):
        return f"""
        op.jt_name = "{name}";
        BatchNormAttr *attr = new BatchNormAttr();
        attr->is_train = {"true" if self.is_train else "false"};
        attr->momentum = {self.momentum};
        attr->eps = {self.eps};
        op.op_attr.reset(attr);
        """

    def __call__(self, x, weight, bias, running_mean, running_var):
        channels = int(x.shape[1])
        result = norms_cmd(
            "BatchNorm",
            inputs=[x, weight, bias, running_mean, running_var],
            output_dtypes=[x.dtype] * 3,
            output_shapes=[x.shape, (channels,), (channels,)],
            attr_code=self._attr_code("batchnorm"),
            multi_grad_input_count=3,
            multi_grad_src=f"""
            // aclop
            BatchNormBackwardOpRunner op;
            op.add(dout, true);
            op.add(in0, true);
            op.add(in1, true);
            op.add(in3, true);
            op.add(in4, true);
            op.add(pout1, true);
            op.add(pout2, true);
            op.add(out0, false);
            op.add(out1, false);
            op.add(out2, false);
            {self._attr_code("batchnormbackward")}
            op.run();
            """,
        )
        return result[0]
    

class LayerNormACL:

    def __init__(self,
                    normalized_shape,
                    eps: float = 1e-5,
                    elementwise_affine: bool = True):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

    def _attr_code(self, name):
        return f"""
        op.jt_name = "{name}";
        LayerNormAttr *attr = new LayerNormAttr();
        attr->eps = {self.eps};
        attr->normalizedShape = {{{', '.join(map(str, self.normalized_shape))}}};
        attr->size = {len(self.normalized_shape)};
        op.op_attr.reset(attr);
        """

    def __call__(self, x, weight, bias):
        input_value = x.float32()
        # aclnnLayerNorm outputs: out (x.shape), mean & rstd (reduced over the
        # normalized dims -> same leading shape with the normalized dims = 1).
        nd = len(self.normalized_shape)
        reduced_shape = list(x.shape[:len(x.shape) - nd]) + [1] * nd
        result = norms_cmd(
            "LayerNorm",
            inputs=[input_value, weight, bias],
            output_dtypes=[input_value.dtype] * 3,
            output_shapes=[input_value.shape, reduced_shape, reduced_shape],
            attr_code=self._attr_code("layernorm"),
            multi_grad_src=f"""
            // aclop
            LayerNormBackwardOpRunner op;
            op.add(dout, true);
            op.add(in0, true);
            op.add(pout1, true);
            op.add(pout2, true);
            op.add(in1, true);
            op.add(in2, true);
            op.add(out0, false);
            op.add(out1, false);
            op.add(out2, false);
            {self._attr_code("layernormbackward")}
            op.run();
            """,
        )
        return result[0]


class GroupNormACL:

    def __init__(self, num_groups, eps):
        self.num_groups = int(num_groups)
        self.eps = float(eps)

    def _attr_code(self):
        return f'''
        op.jt_name = "groupnorm";
        GroupNormAttr *attr = new GroupNormAttr();
        attr->batch = {self.batch};
        attr->channels = {self.channels};
        attr->spatialSize = {self.spatial_size};
        attr->groups = {self.num_groups};
        attr->eps = {self.eps};
        op.op_attr.reset(attr);
        '''

    def __call__(self, x, weight, bias):
        self.batch = int(x.shape[0])
        self.channels = int(x.shape[1])
        self.spatial_size = 1
        for size in x.shape[2:]:
            self.spatial_size *= int(size)
        result = norms_cmd(
            "GroupNorm",
            inputs=[x, weight, bias],
            output_dtypes=[x.dtype, x.dtype, x.dtype],
            output_shapes=[
                x.shape,
                (self.batch, self.num_groups),
                (self.batch, self.num_groups),
            ],
            attr_code=self._attr_code(),
            multi_grad_src=f'''
            // aclop
            GroupNormBackwardOpRunner op;
            op.add(dout, true);
            op.add(in0, true);
            op.add(pout1, true);
            op.add(pout2, true);
            op.add(in1, true);
            op.add(out0, false);
            op.add(out1, false);
            op.add(out2, false);
            {self._attr_code()}
            op.run();
            ''',
        )
        return result[0]


class RmsNormACL(jt.Function):

    def execute(self, x, weight, eps):
        self.input = x
        self.weight = weight
        reduced_shape = list(x.shape[:-1]) + [1]
        outputs = [
            jt.empty(x.shape, x.dtype),
            jt.empty(reduced_shape, "float32"),
        ]
        attr_code = f"""
        op.jt_name = "rmsnorm";
        RmsNormAttr *attr = new RmsNormAttr();
        attr->eps = {eps};
        op.op_attr.reset(attr);
        """
        result = norms_cmd(
            "RmsNorm", inputs=[x, weight], outputs=outputs,
            attr_code=attr_code)
        self.rstd = result[1]
        return result[0]

    def grad(self, grad_output):
        outputs = [
            jt.empty(self.input.shape, self.input.dtype),
            jt.empty(self.weight.shape, "float32"),
        ]
        result = norms_cmd(
            "RmsNormGrad",
            inputs=[grad_output, self.input, self.rstd, self.weight],
            outputs=outputs,
            attr_code='op.jt_name = "rmsnormgrad";',
        )
        grad_weight = result[1]
        if str(self.weight.dtype) != "float32":
            grad_weight = grad_weight.cast(self.weight.dtype)
        return result[0], grad_weight


class GroupedAddRmsNormACL:

    def __call__(self, x, residual, weight, eps):
        reduced_shape = list(x.shape[:-1]) + [1]
        outputs = [
            jt.empty(x.shape, x.dtype),
            jt.empty(x.shape, x.dtype),
            jt.empty(reduced_shape, "float32"),
        ]
        result = jt.code(
            outputs=outputs,
            inputs=[x, residual, weight],
            cuda_header='''
namespace jittor {}
#include "acl/aclops/aclops.h"
''',
            cuda_src=f'''
// aclop
BinaryOpRunner add_op;
add_op.name = "Add";
add_op.add(in0, true);
add_op.add(in1, true);
add_op.add(out1, false);
add_op.jt_name = "grouped_add_rms_norm";
add_op.run();

RmsNormOpRunner norm_op;
norm_op.add(out1, true);
norm_op.add(in2, true);
norm_op.add(out0, false);
norm_op.add(out2, false);
norm_op.jt_name = "grouped_add_rms_norm";
auto *norm_attr = new RmsNormAttr();
norm_attr->eps = {eps};
norm_op.op_attr.reset(norm_attr);
norm_op.run();
''',
        )
        return result[0], result[1]


class GroupedBFloat16RmsNormACL:

    def __call__(self, x, unit_weight, weight, eps):
        reduced_shape = list(x.shape[:-1]) + [1]
        outputs = [
            jt.empty(x.shape, x.dtype),
            jt.empty(x.shape, x.dtype),
            jt.empty(reduced_shape, "float32"),
        ]
        result = jt.code(
            outputs=outputs,
            inputs=[x, unit_weight, weight],
            cuda_header='''
namespace jittor {}
#include "acl/aclops/aclops.h"
''',
            cuda_src=f'''
// aclop
RmsNormOpRunner norm_op;
norm_op.add(in0, true);
norm_op.add(in1, true);
norm_op.add(out1, false);
norm_op.add(out2, false);
norm_op.jt_name = "grouped_bfloat16_rms_norm";
auto *norm_attr = new RmsNormAttr();
norm_attr->eps = {eps};
norm_op.op_attr.reset(norm_attr);
norm_op.run();

BinaryOpRunner multiply_op;
multiply_op.name = "Mul";
multiply_op.add(in2, true);
multiply_op.add(out1, true);
multiply_op.add(out0, false);
multiply_op.jt_name = "grouped_bfloat16_rms_norm";
multiply_op.run();
''',
        )
        return result[0]


class GroupedDualBFloat16RmsNormACL:

    def __call__(
            self, first, second, first_unit, second_unit,
            first_weight, second_weight, eps):
        first_rstd_shape = list(first.shape[:-1]) + [1]
        second_rstd_shape = list(second.shape[:-1]) + [1]
        outputs = [
            jt.empty(first.shape, first.dtype),
            jt.empty(second.shape, second.dtype),
            jt.empty(first.shape, first.dtype),
            jt.empty(second.shape, second.dtype),
            jt.empty(first_rstd_shape, "float32"),
            jt.empty(second_rstd_shape, "float32"),
        ]
        result = jt.code(
            outputs=outputs,
            inputs=[
                first, second, first_unit, second_unit,
                first_weight, second_weight,
            ],
            cuda_header='''
namespace jittor {}
#include "acl/aclops/aclops.h"
''',
            cuda_src=f'''
// aclop
RmsNormOpRunner first_norm;
first_norm.add(in0, true);
first_norm.add(in2, true);
first_norm.add(out2, false);
first_norm.add(out4, false);
first_norm.jt_name = "grouped_dual_bfloat16_rms_norm";
auto *first_attr = new RmsNormAttr();
first_attr->eps = {eps};
first_norm.op_attr.reset(first_attr);
first_norm.run();

BinaryOpRunner first_multiply;
first_multiply.name = "Mul";
first_multiply.add(in4, true);
first_multiply.add(out2, true);
first_multiply.add(out0, false);
first_multiply.jt_name = "grouped_dual_bfloat16_rms_norm";
first_multiply.run();

RmsNormOpRunner second_norm;
second_norm.add(in1, true);
second_norm.add(in3, true);
second_norm.add(out3, false);
second_norm.add(out5, false);
second_norm.jt_name = "grouped_dual_bfloat16_rms_norm";
auto *second_attr = new RmsNormAttr();
second_attr->eps = {eps};
second_norm.op_attr.reset(second_attr);
second_norm.run();

BinaryOpRunner second_multiply;
second_multiply.name = "Mul";
second_multiply.add(in5, true);
second_multiply.add(out3, true);
second_multiply.add(out1, false);
second_multiply.jt_name = "grouped_dual_bfloat16_rms_norm";
second_multiply.run();
''',
        )
        return result[0], result[1]

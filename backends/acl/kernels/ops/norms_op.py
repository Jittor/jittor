from ._code import code_with_attributes
from ._attributes import attribute_program, code_program, runner_for_alias
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
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
from ._code import check_acl_float_dtype


class BatchNormACL:
    def __init__(self, eps=1e-5, momentum=0.1, is_train=False):
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.is_train = bool(is_train)

    def _attributes(self):
        return {"is_train": bool(self.is_train), "momentum": self.momentum, "eps": self.eps}

    def __call__(self, x, weight, bias, running_mean, running_var):
        channels = int(x.shape[1])
        result = norms_cmd(
            "BatchNorm",
            inputs=[x, weight, bias, running_mean, running_var],
            output_dtypes=[x.dtype] * 3,
            output_shapes=[x.shape, (channels,), (channels,)],
            attributes=self._attributes(),
            multi_grad_input_count=3,
            multi_grad_src=code_program(
                [
                    "\n            // aclop\n            BatchNormBackwardOpRunner op;\n            op.add(dout, true);\n            op.add(in0, true);\n            op.add(in1, true);\n            op.add(in3, true);\n            op.add(in4, true);\n            op.add(pout1, true);\n            op.add(pout2, true);\n            op.add(out0, false);\n            op.add(out1, false);\n            op.add(out2, false);\n            ",
                    "\n            op.run();\n            ",
                ]
            ),
            multi_grad_attributes=self._attributes(),
        )
        return result[0]


class LayerNormACL:
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

    def _attributes(self):
        return {
                "eps": self.eps,
                "normalizedShape": list(self.normalized_shape),
            }

    def __call__(self, x, weight, bias):
        input_value = check_acl_float_dtype(x, "layernorm")
        # aclnnLayerNorm outputs: out (x.shape), mean & rstd (reduced over the
        # normalized dims -> same leading shape with the normalized dims = 1).
        nd = len(self.normalized_shape)
        reduced_shape = list(x.shape[: len(x.shape) - nd]) + [1] * nd
        result = norms_cmd(
            "LayerNorm",
            inputs=[input_value, weight, bias],
            output_dtypes=[input_value.dtype] * 3,
            output_shapes=[input_value.shape, reduced_shape, reduced_shape],
            attributes=self._attributes(),
            multi_grad_src=code_program(
                [
                    "\n            // aclop\n            LayerNormBackwardOpRunner op;\n            op.add(dout, true);\n            op.add(in0, true);\n            op.add(pout1, true);\n            op.add(pout2, true);\n            op.add(in1, true);\n            op.add(in2, true);\n            op.add(out0, false);\n            op.add(out1, false);\n            op.add(out2, false);\n            ",
                    "\n            op.run();\n            ",
                ]
            ),
            multi_grad_attributes=self._attributes(),
        )
        return result[0]


class GroupNormACL:
    def __init__(self, num_groups, eps):
        self.num_groups = int(num_groups)
        self.eps = float(eps)

    def _attributes(self):
        return {
                "batch": self.batch,
                "channels": self.channels,
                "spatialSize": self.spatial_size,
                "groups": self.num_groups,
                "eps": self.eps,
            }

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
            attributes=self._attributes(),
            multi_grad_src=code_program(
                [
                    "\n            // aclop\n            GroupNormBackwardOpRunner op;\n            op.add(dout, true);\n            op.add(in0, true);\n            op.add(pout1, true);\n            op.add(pout2, true);\n            op.add(in1, true);\n            op.add(out0, false);\n            op.add(out1, false);\n            op.add(out2, false);\n            ",
                    "\n            op.run();\n            ",
                ]
            ),
            multi_grad_attributes=self._attributes(),
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
        result = norms_cmd("RmsNorm", inputs=[x, weight], outputs=outputs,
                           attributes={"eps": eps})
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
        if _jittor_dtype_name(self.weight.dtype) != "float32":
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
        result = code_with_attributes(
            backend="acl",
            outputs=outputs,
            inputs=[x, residual, weight],
            cuda_header="""
namespace jittor {}
#include "aclops/aclops.h"
""",
            cuda_src=code_program(
                [
                    '\n// aclop\nBinaryOpRunner add_op;\nadd_op.name = "Add";\nadd_op.add(in0, true);\nadd_op.add(in1, true);\nadd_op.add(out1, false);\nadd_op.jt_name = "grouped_add_rms_norm";\nadd_op.run();\n\nRmsNormOpRunner norm_op;\nnorm_op.add(out1, true);\nnorm_op.add(in2, true);\nnorm_op.add(out0, false);\nnorm_op.add(out2, false);\nnorm_op.jt_name = "grouped_add_rms_norm";\n',
                    'apply_acl_code_attributes(norm_op, data, "acl_payload.norm.", "RmsNorm");\n',
                    "\nnorm_op.run();\n",
                ]
            ),
            attribute_sets={"norm": ("RmsNorm", {"eps": eps})},
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
        result = code_with_attributes(
            backend="acl",
            outputs=outputs,
            inputs=[x, unit_weight, weight],
            cuda_header="""
namespace jittor {}
#include "aclops/aclops.h"
""",
            cuda_src=code_program(
                [
                    '\n// aclop\nRmsNormOpRunner norm_op;\nnorm_op.add(in0, true);\nnorm_op.add(in1, true);\nnorm_op.add(out1, false);\nnorm_op.add(out2, false);\nnorm_op.jt_name = "grouped_bfloat16_rms_norm";\n',
                    'apply_acl_code_attributes(norm_op, data, "acl_payload.norm.", "RmsNorm");\n',
                    '\nnorm_op.run();\n\nBinaryOpRunner multiply_op;\nmultiply_op.name = "Mul";\nmultiply_op.add(in2, true);\nmultiply_op.add(out1, true);\nmultiply_op.add(out0, false);\nmultiply_op.jt_name = "grouped_bfloat16_rms_norm";\nmultiply_op.run();\n',
                ]
            ),
            attribute_sets={"norm": ("RmsNorm", {"eps": eps})},
        )
        return result[0]


class GroupedDualBFloat16RmsNormACL:
    def __call__(self, first, second, first_unit, second_unit, first_weight, second_weight, eps):
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
        result = code_with_attributes(
            backend="acl",
            outputs=outputs,
            inputs=[
                first,
                second,
                first_unit,
                second_unit,
                first_weight,
                second_weight,
            ],
            cuda_header="""
namespace jittor {}
#include "aclops/aclops.h"
""",
            cuda_src=code_program(
                [
                    '\n// aclop\nRmsNormOpRunner first_norm;\nfirst_norm.add(in0, true);\nfirst_norm.add(in2, true);\nfirst_norm.add(out2, false);\nfirst_norm.add(out4, false);\nfirst_norm.jt_name = "grouped_dual_bfloat16_rms_norm";\n',
                    'apply_acl_code_attributes(first_norm, data, "acl_payload.first_norm.", "RmsNorm");\n',
                    '\nfirst_norm.run();\n\nBinaryOpRunner first_multiply;\nfirst_multiply.name = "Mul";\nfirst_multiply.add(in4, true);\nfirst_multiply.add(out2, true);\nfirst_multiply.add(out0, false);\nfirst_multiply.jt_name = "grouped_dual_bfloat16_rms_norm";\nfirst_multiply.run();\n\nRmsNormOpRunner second_norm;\nsecond_norm.add(in1, true);\nsecond_norm.add(in3, true);\nsecond_norm.add(out3, false);\nsecond_norm.add(out5, false);\nsecond_norm.jt_name = "grouped_dual_bfloat16_rms_norm";\n',
                    'apply_acl_code_attributes(second_norm, data, "acl_payload.second_norm.", "RmsNorm");\n',
                    '\nsecond_norm.run();\n\nBinaryOpRunner second_multiply;\nsecond_multiply.name = "Mul";\nsecond_multiply.add(in5, true);\nsecond_multiply.add(out3, true);\nsecond_multiply.add(out1, false);\nsecond_multiply.jt_name = "grouped_dual_bfloat16_rms_norm";\nsecond_multiply.run();\n',
                ]
            ),
            attribute_sets={"first_norm": ("RmsNorm", {"eps": eps}), "second_norm": ("RmsNorm", {"eps": eps})},
        )
        return result[0], result[1]

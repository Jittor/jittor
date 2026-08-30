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

    def execute(self, x, weight, bias):
        # weight/bias come from the host nn.LayerNorm module so its parameters
        # stay tracked (grad, DDP broadcast, etc). grad() returns grads for all
        # three inputs (x, weight, bias) in order.
        self.input = x.float32()
        self.weight = weight
        self.bias = bias
        inputs = [self.input, self.weight, self.bias]
        # aclnnLayerNorm outputs: out (x.shape), mean & rstd (reduced over the
        # normalized dims -> same leading shape with the normalized dims = 1).
        nd = len(self.normalized_shape)
        reduced_shape = list(x.shape[:len(x.shape) - nd]) + [1] * nd
        outputs = [jt.empty(x.shape), jt.empty(reduced_shape), jt.empty(reduced_shape)]
        attr_code = f"""
        op.jt_name = "layernorm";
        LayerNormAttr *attr = new LayerNormAttr();
        attr->eps = {self.eps};
        attr->normalizedShape = {{{', '.join(map(str, (list(self.normalized_shape))))}}};
        attr->size = {len(self.normalized_shape)};
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
        # aclnnLayerNormBackward inputs : gradOut, input, mean, rstd, weight, bias
        #                        outputs: gradInput, gradWeight, gradBias
        attr_code = f"""
        op.jt_name = "layernormbackward";
        LayerNormAttr *attr = new LayerNormAttr();
        attr->eps = {self.eps};
        attr->normalizedShape = {{{', '.join(map(str, (list(self.normalized_shape))))}}};
        attr->size = {len(self.normalized_shape)};
        op.op_attr.reset(attr);
        """
        inputs = [grad_output.float32(), self.input, self.meanout, self.rstdout,
                  self.weight, self.bias]
        outputs = [jt.empty(self.input.shape),
                   jt.empty(self.normalized_shape),
                   jt.empty(self.normalized_shape)]
        result = norms_cmd("LayerNormBackward",
                           inputs=inputs,
                           outputs=outputs,
                           attr_code=attr_code)
        # grads for (x, weight, bias) to match execute()'s inputs
        return result[0], result[1], result[2]


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

# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers:
#     Guoye Yang <498731903@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
from jittor import nn
from jittor import Function
from jittor._core.dtypes import dtype_name
from jittor._runtime.dispatch import native_rule, register_kernel, select_kernel
from jittor.nn.functional._amp import bias_for_compute_dtype
from jittor.nn.functional._layout import channels_last_source, channels_last_view, records_no_grad
from jittor.backends.cuda.kernels.nn.depthwise import (
    depthwise_forward, depthwise_forward_nhwc, depthwise_backward,
    _depthwise_forward_rows, _supports_depthwise_rows,
)

class DepthwiseConv(Function):
    def __init__(self, stride=1, padding=0, dilation=1):
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

    def __call__(self, x, weight):
        kernel = select_kernel("depthwise_conv2d", x, weight, self)
        return kernel(x, weight, self)

    def _output_shape(self, x, weight):
        N,C,H,W = x.shape
        o,i,Kh,Kw = weight.shape
        assert(o == C)
        oh = (H+self.padding[0]*2-Kh*self.dilation[0]+self.dilation[0]-1)//self.stride[0]+1
        ow = (W+self.padding[1]*2-Kw*self.dilation[1]+self.dilation[1]-1)//self.stride[1]+1
        assert oh>0 and ow>0
        return [N, C, oh, ow]

    def execute(self, x, weight, bias=None):
        if select_kernel("depthwise_conv2d", x, weight, self) is not _depthwise_cuda:
            return _depthwise_generic(x, weight, self, bias)
        self.save_vars = x, weight
        self.has_bias = bias is not None
        shape = self._output_shape(x, weight)
        if self.has_bias:
            # conv2d only supplies a bias to this Function for the row kernel.
            return _depthwise_forward_rows(x, weight, shape, self.padding, bias)
        N, C, oh, ow = shape
        if records_no_grad(x, weight):
            nhwc_x = channels_last_source(x)
            if nhwc_x is not None:
                out_nhwc = depthwise_forward_nhwc(
                    nhwc_x, weight, [N, oh, ow, C], self.stride, self.padding, self.dilation)
                return channels_last_view(out_nhwc)
        return depthwise_forward(x, weight, shape, self.stride, self.padding, self.dilation)

    def grad(self, grad):
        x, weight = self.save_vars
        gradients = depthwise_backward(
            x, weight, grad, weight.shape[-2:], self.stride, self.padding, self.dilation)
        if self.has_bias:
            return (*gradients, grad.sum((0, 2, 3)))
        return gradients


def _depthwise_generic(x, weight, operator, bias=None):
    return nn.conv2d(x, weight, bias, operator.stride, operator.padding,
                     operator.dilation, x.shape[1], _depthwise_fast_path=False)


def _depthwise_cuda(x, weight, operator):
    return Function.__call__(operator, x, weight)


def _supports_depthwise(x, weight, operator):
    return x.dtype == weight.dtype


@native_rule("depthwise_conv2d")
def _supports_depthwise_conv2d(x, weight, bias, stride, padding, dilation, groups,
                              *, _depthwise_fast_path=True):
    return (_depthwise_fast_path and groups == weight.shape[0] == x.shape[1]
            and x.dtype == weight.dtype)


def _depthwise_conv2d(x, weight, bias, stride, padding, dilation, groups,
                      *, _depthwise_fast_path=True):
    operator = DepthwiseConv(stride, padding, dilation)
    if (isinstance(bias, jt.Var) and not jt.flags.amp_reg and dtype_name(bias.dtype) == "float32"
            and x._storage_is_contiguous()
            and _supports_depthwise_rows(x, weight, operator._output_shape(x, weight),
                                         operator.stride, operator.dilation)):
        return Function.__call__(operator, x, weight, bias)
    y = operator(x, weight)
    if bias is not None:
        y = y + bias_for_compute_dtype(y, bias).broadcast(y.shape, [0, 2, 3])
    return y


register_kernel("depthwise_conv2d", "cuda", _depthwise_cuda,
                supports=_supports_depthwise)
register_kernel("depthwise_conv2d", "*", _depthwise_generic)
register_kernel("conv2d", "cuda", _depthwise_conv2d,
                supports=_supports_depthwise_conv2d, priority=20)

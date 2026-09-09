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
from jittor._runtime.dispatch import register_kernel, select_kernel
from jittor.backends.cuda.kernels.nn.depthwise import depthwise_forward, depthwise_backward

class DepthwiseConv(Function):
    def __init__(self, stride=1, padding=0, dilation=1):
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

    def __call__(self, x, weight):
        kernel = select_kernel("depthwise_conv2d", x, weight, self)
        return kernel(x, weight, self)

    def execute(self, x, weight):
        if select_kernel("depthwise_conv2d", x, weight, self) is not _depthwise_cuda:
            return _depthwise_generic(x, weight, self)
        self.save_vars = x, weight
        N,C,H,W = x.shape
        o,i,Kh,Kw = weight.shape
        assert(o == C)
        oh = (H+self.padding[0]*2-Kh*self.dilation[0]+self.dilation[0]-1)//self.stride[0]+1
        ow = (W+self.padding[1]*2-Kw*self.dilation[1]+self.dilation[1]-1)//self.stride[1]+1
        self.Khw = Kh, Kw
        assert oh>0 and ow>0
        output = depthwise_forward(
            x, weight, [N, C, oh, ow], self.stride, self.padding, self.dilation)
        return output

    def grad(self, grad):
        x, weight = self.save_vars
        Kh, Kw = self.Khw
        return depthwise_backward(
            x, weight, grad, (Kh, Kw), self.stride, self.padding, self.dilation)


def _depthwise_generic(x, weight, operator):
    return nn.conv2d(x, weight, None, operator.stride, operator.padding,
                     operator.dilation, x.shape[1], _depthwise_fast_path=False)


def _depthwise_cuda(x, weight, operator):
    return Function.__call__(operator, x, weight)


def _supports_depthwise(x, weight, operator):
    return x.dtype == weight.dtype


def _supports_depthwise_conv2d(x, weight, bias, stride, padding, dilation, groups,
                              *, _depthwise_fast_path=True):
    return (_depthwise_fast_path and groups == weight.shape[0] == x.shape[1]
            and x.dtype == weight.dtype)


def _depthwise_conv2d(x, weight, bias, stride, padding, dilation, groups,
                      *, _depthwise_fast_path=True):
    y = DepthwiseConv(stride, padding, dilation)(x, weight)
    if bias is not None:
        y = y + bias.broadcast(y.shape, [0, 2, 3])
    return y


register_kernel("depthwise_conv2d", "cuda", _depthwise_cuda,
                supports=_supports_depthwise)
register_kernel("depthwise_conv2d", "*", _depthwise_generic)
register_kernel("conv2d", "cuda", _depthwise_conv2d,
                supports=_supports_depthwise_conv2d, priority=20)

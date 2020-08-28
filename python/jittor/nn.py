# ***************************************************************
# Copyright (c) 2020 Jittor. Authors:
#     Guowei Yang <471184555@qq.com>
#     Guoye Yang <498731903@qq.com>
#     Wenyang Zhou <576825820@qq.com>
#     Meng-Hao Guo <guomenghao1997@gmail.com>
#     Dun Liang <randonlang@gmail.com>.
#
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
from jittor import init, Module
import numpy as np
import math
from jittor.pool import Pool, pool, AdaptiveAvgPool2d
from jittor.optim import *


def matmul_transpose(a, b):
    '''
    returns a * b^T
    '''
    assert len(a.shape) >= 2 and len(b.shape) == 2
    assert a.shape[-1] == b.shape[-1]

    shape = list(a.shape)[:-1] + list(b.shape)
    a = a.broadcast(shape, [len(shape)-2])
    b = b.broadcast(shape)
    return (a*b).sum(len(shape)-1)

def bmm(a, b):
    ''' batch matrix multiply, 
shape of input a is [batch, n, m],
shape of input b is [batch, m, k],
return shape is [batch, n, k]

Example::

    import jittor as jt
    from jittor import nn

    batch, n, m, k = 100, 5, 6, 7

    a = jt.random((batch, n, m))
    b = jt.random((batch, m, k))
    c = nn.bmm(a, b)
    '''
    assert len(a.shape) > 2 and len(b.shape) > 2
    return matmul(a, b)

def matmul(a, b):
    ''' matrix multiply, 

Example::

    a = jt.random([3])
    b = jt.random([3])
    c = jt.matmul(a, b)
    assert c.shape == [1]

    a = jt.random([3, 4])
    b = jt.random([4])
    c = jt.matmul(a, b)
    assert c.shape == [3]

    a = jt.random([10, 3, 4])
    b = jt.random([4])
    c = jt.matmul(a, b)
    assert c.shape == [10, 3]

    a = jt.random([10, 3, 4])
    b = jt.random([4, 5])
    c = jt.matmul(a, b)
    assert c.shape == [10, 3, 5]

    a = jt.random([10, 3, 4])
    b = jt.random([10, 4, 5])
    c = jt.matmul(a, b)
    assert c.shape == [10, 3, 5]

    a = jt.random([8, 1, 3, 4])
    b = jt.random([10, 4, 5])
    c = jt.matmul(a, b)
    assert c.shape == [8, 10, 3, 5]
    '''
    len_a = len(a.shape)
    len_b = len(b.shape)
    if len_b == 1:
        # a: [n, m], b:[m], c:[n]
        return (a*b).sum(-1)
    if len_a == 1:
        # a: [n], b:[n,k], c:[k]
        return (a.broadcast(b, [-1]) * b).sum(0)
    if len_a>=3 and len_a==len_b:
        # bmm
        # a: [..., n, m], b: [..., m, k], c:[..., n, k]
        if jt.flags.use_cuda:
            return jt.compile_extern.cublas_ops.cublas_batched_matmul(a, b, 0, 0)
    shape = []
    len_c = max(len_a, len_b)
    (n, m), (m_, k) = a.shape[-2:], b.shape[-2:]
    assert m == m_, f"dimension not match, a.shape:{a.shape}, b.shape:{a.shape}"
    # a: [..., n, m]
    # b: [..., m, k]
    # cc:[..., n, m, k]
    #     -->
    #     012
    for i in range(len_c-2):
        ai = len_a-(len_c-i)
        bi = len_b-(len_c-i)
        an = a.shape[ai] if ai>=0 else 1
        bn = b.shape[bi] if bi>=0 else 1
        if an!=1 and bn!=1:
            assert an == bn, f"dimension not match, a.shape:{a.shape}, b.shape:{a.shape}"
        cn = max(an, bn)
        shape.append(cn)
    shape.extend([n, m, k])
    a = a.broadcast(shape, [-1])
    b = b.broadcast(shape, [-3])
    return (a*b).sum(-2)
jt.Var.matmul = jt.Var.__matmul__ = matmul
jt.Var.__imatmul__ = lambda a,b: a.assign(matmul(a,b))

def get_init_var_rand(shape, dtype):
    return jt.array(np.random.normal(0.0, 1.0, shape).astype(np.float32))

def relu(x): return jt.maximum(x, 0)
def leaky_relu(x, scale=0.01): return jt.ternary(x>0, x, x*scale)
def relu6(x): return jt.minimum(jt.maximum(x, 0), 6)

class PReLU(Module):
    def __init__(self, num_parameters=1, init_=0.25):
        self.num_parameters = num_parameters
        self.a = init.constant((num_parameters,), "float32", init_)

    def execute(self, x):
        if self.num_parameters != 1:
            assert self.num_parameters == x.size(1), f"num_parameters does not match input channels in PReLU"
            return jt.maximum(0, x) + self.a.broadcast(x, [0,2,3]) * jt.minimum(0, x)
        else:
            return jt.maximum(0, x) + self.a * jt.minimum(0, x)

#TODO dims is 4 will cause slowly execution
def cross_entropy_loss(output, target, ignore_index=None):
    if len(output.shape) == 4:
        c_dim = output.shape[1]
        output = output.transpose((0, 2, 3, 1))
        output = output.reshape((-1, c_dim))
    if ignore_index is not None:
        target = jt.ternary(target==ignore_index,
            jt.array(-1).broadcast(target), target)
        mask = jt.logical_and(target >= 0, target < output.shape[1])
    target = target.reshape((-1, ))
    target = target.broadcast(output, [1])
    target = target.index(1) == target
    
    output = output - output.max([1], keepdims=True)
    loss = output.exp().sum(1).log()
    loss = loss - (output*target).sum(1)
    if ignore_index is None:
        return loss.mean()
    else:
        return loss.sum() / jt.maximum(mask.int().sum(), 1)

def mse_loss(output, target):
    return (output-target).sqr().mean()

def bce_loss(output, target, weight=None, size_average=True):
    loss = - (target * jt.log(jt.maximum(output, 1e-20)) + (1 - target) * jt.log(jt.maximum(1 - output, 1e-20)))

    if weight is not None:
        loss *= weight
    
    if size_average:
        return loss.mean()
    else:
        return loss.sum()

def l1_loss(output, target):
    return (output-target).abs().mean()


def smooth_l1_loss(y_true, y_pred,reduction="mean"):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.

    Args:
         y_true - ground truth 
         y_pred - predictions
         reduction - the mode of cal loss which must be in ['mean','sum','none']
    """
    diff = jt.abs(y_true - y_pred)
    less_than_one = (diff<1.0).float32()
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    if reduction=="mean":
        return loss.mean()
    elif reduction=="sum":
        return loss.sum()
    elif reduction=="none":
        return loss
    else:
        raise ValueError(f'not support {reduction}')

class CrossEntropyLoss(Module):
    def __init__(self,ignore_index=None):
        self.ignore_index = ignore_index
        
    def execute(self, output, target):
        return cross_entropy_loss(output, target,self.ignore_index)

class MSELoss(Module):
    def __init__(self):
        pass
    def execute(self, output, target):
        return mse_loss(output, target)

class BCELoss(Module):
    def __init__(self, weight=None, size_average=True):
        self.weight = weight
        self.size_average = size_average
    def execute(self, output, target):
        return bce_loss(output, target, self.weight, self.size_average)

class L1Loss(Module):
    def __init__(self):
        pass
    def execute(self, output, target):
        return l1_loss(output, target)

class BCEWithLogitsLoss(Module):
    def __init__(self, weight=None, size_average=True):
        self.sigmoid = Sigmoid()
        self.bce = BCELoss(weight, size_average)
    def execute(self, output, target):
        output = self.sigmoid(output)
        output = self.bce(output, target)
        return output

def binary_cross_entropy_with_logits(input, target, weight=None, size_average=True):
    return BCEWithLogitsLoss(weight, size_average)(input, target)

def softmax(x, dim = None):
    if dim is None:
        x = (x - x.max()).exp()
        ret = x / x.sum()
    else:
        x = (x-x.max(dim, keepdims=True)).exp()
        ret = x / x.sum(dim, keepdims=True)
    return ret

class Dropout(Module):
    def __init__(self, p=0.5, is_train=False):
        assert p >= 0 and p <= 1, "dropout probability has to be between 0 and 1, but got {}".format(p)
        self.p = p
        self.is_train = is_train
        #TODO: test model.train() to change self.is_train
    def execute(self, input):
        output = input
        if self.p > 0 and self.is_train:
            if self.p == 1:
                noise = jt.zeros(input.shape)
                output = output * noise
            else:
                noise = jt.random(input.shape)
                noise = (noise > self.p).int()
                output = output * noise / (1.0 - self.p) # div keep prob
        return output

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = init.invariant_uniform((out_features, in_features), "float32")
        bound = 1.0/math.sqrt(in_features)
        self.bias = init.uniform((out_features,), "float32",-bound,bound) if bias else None

    def execute(self, x):
        x = matmul_transpose(x, self.weight)
        if self.bias is not None:
            return x + self.bias
        return x

class BatchNorm(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=None, is_train=True, sync=True):
        assert affine == None

        self.sync = sync
        self.num_features = num_features
        self.is_train = is_train
        self.eps = eps
        self.momentum = momentum
        self.weight = init.constant((num_features,), "float32", 1.0)
        self.bias = init.constant((num_features,), "float32", 0.0)
        self.running_mean = init.constant((num_features,), "float32", 0.0).stop_grad()
        self.running_var = init.constant((num_features,), "float32", 1.0).stop_grad()

    def execute(self, x):
        if self.is_train:
            xmean = jt.mean(x, dims=[0,2,3], keepdims=1)
            x2mean = jt.mean(x*x, dims=[0,2,3], keepdims=1)
            if self.sync and jt.in_mpi:
                xmean = xmean.mpi_all_reduce("mean")
                x2mean = x2mean.mpi_all_reduce("mean")

            xvar = x2mean-xmean*xmean
            norm_x = (x-xmean)/jt.sqrt(xvar+self.eps)
            self.running_mean.update(self.running_mean +
                (xmean.reshape((-1,)) - self.running_mean) * self.momentum)
            self.running_var.update(self.running_var +
                (xvar.reshape((-1,))-self.running_var)*self.momentum)
        else:
            running_mean = self.running_mean.broadcast(x, [0,2,3])
            running_var = self.running_var.broadcast(x, [0,2,3])
            norm_x = (x-running_mean)/jt.sqrt(running_var+self.eps)
        w = self.weight.broadcast(x, [0,2,3])
        b = self.bias.broadcast(x, [0,2,3])
        return norm_x * w + b
        
class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=None, is_train=True, sync=True):
        assert affine == None
        self.sync = sync
        self.num_features = num_features
        self.is_train = is_train
        self.eps = eps
        self.momentum = momentum
        self.weight = init.constant((num_features,), "float32", 1.0)
        self.bias = init.constant((num_features,), "float32", 0.0)
        self.running_mean = init.constant((num_features,), "float32", 0.0).stop_grad()
        self.running_var = init.constant((num_features,), "float32", 1.0).stop_grad()

    def execute(self, x):
        if self.is_train:
            xmean = jt.mean(x, dims=[0], keepdims=1)
            x2mean = jt.mean(x*x, dims=[0], keepdims=1)

            if self.sync and jt.in_mpi:
                xmean = xmean.mpi_all_reduce("mean")
                x2mean = x2mean.mpi_all_reduce("mean")

            xvar = x2mean-xmean*xmean
            norm_x = (x-xmean)/jt.sqrt(xvar+self.eps)
            self.running_mean.update(self.running_mean + 
                (xmean.sum([0])-self.running_mean)*self.momentum)
            self.running_var.update(self.running_var + 
                (xvar.sum([0])-self.running_var)*self.momentum)
        else:
            running_mean = self.running_mean.broadcast(x, [0])
            running_var = self.running_var.broadcast(x, [0])
            norm_x = (x-running_mean)/jt.sqrt(running_var+self.eps)
        w = self.weight.broadcast(x, [0])
        b = self.bias.broadcast(x, [0])
        return norm_x * w + b

class InstanceNorm2d(Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=None, is_train=True, sync=True):
        assert affine == None
        self.sync = sync
        self.num_features = num_features
        self.is_train = is_train
        self.eps = eps
        self.momentum = momentum
        self.weight = init.constant((num_features,), "float32", 1.0)
        self.bias = init.constant((num_features,), "float32", 0.0)

    def execute(self, x):
        xmean = jt.mean(x, dims=[2,3], keepdims=1)
        x2mean = jt.mean(x*x, dims=[2,3], keepdims=1)
        if self.sync and jt.in_mpi:
            xmean = xmean.mpi_all_reduce("mean")
            x2mean = x2mean.mpi_all_reduce("mean")

        xvar = jt.maximum(x2mean-xmean*xmean, 0)
        norm_x = (x-xmean)/jt.sqrt(xvar+self.eps)
        w = self.weight.broadcast(x, [0,2,3])
        b = self.bias.broadcast(x, [0,2,3])
        return norm_x * w + b

class GroupNorm(Module):
    def __init__(self, num_groups, num_channels, eps=1e-05, affine=None, is_train=True):
        assert affine == None
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = init.constant((num_channels,), "float32", 1.0)
        self.bias = init.constant((num_channels,), "float32", 0.0)

    def execute(self, x):
        N,C,H,W = x.shape
        assert C == self.num_channels
        assert C % self.num_groups == 0
        x = x.reshape((N, self.num_groups, int(C/self.num_groups), H*W))
        xmean = jt.mean(x, dims=[2,3], keepdims=1)
        x2mean = jt.mean(x*x, dims=[2,3], keepdims=1)
        xvar = jt.maximum(x2mean-xmean*xmean, 0)
        norm_x = (x-xmean)/jt.sqrt(xvar+self.eps)
        w = self.weight.reshape((1,self.num_groups,C//self.num_groups,1))
        b = self.bias.reshape((1,self.num_groups,C//self.num_groups,1))
        return (norm_x * w + b).reshape((N,C,H,W))

Relu = jt.make_module(relu)
ReLU = Relu
Leaky_relu = jt.make_module(leaky_relu, 2)
LeakyReLU = Leaky_relu
ReLU6 = jt.make_module(relu6)
Softmax = jt.make_module(softmax, 2)

class Conv(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        assert out_channels % groups == 0, 'out_channels must be divisible by groups'
        Kh, Kw = self.kernel_size
        self.groups = groups
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        assert out_channels % groups == 0, 'out_channels must be divisible by groups'

        # self.weight = init.relu_invariant_gauss([out_channels, in_channels//groups, Kh, Kw], dtype="float", mode="fan_out")
        self.weight = init.invariant_uniform([out_channels, in_channels//groups, Kh, Kw], dtype="float")
        if bias:
            fan=1
            for i in self.weight.shape[1:]:
                fan *= i
            bound = 1 / math.sqrt(fan)
            self.bias = init.uniform([out_channels], dtype="float", low=-bound, high=bound)
        else:
            self.bias = None

    def execute(self, x):
        if self.groups == 1:
            N,C,H,W = x.shape
            Kh, Kw = self.kernel_size
            assert C==self.in_channels
            oh = (H+self.padding[0]*2-Kh*self.dilation[0]+self.dilation[0]-1)//self.stride[0]+1
            ow = (W+self.padding[1]*2-Kw*self.dilation[1]+self.dilation[1]-1)//self.stride[1]+1
            xx = x.reindex([N,self.out_channels,C,oh,ow,Kh,Kw], [
                'i0', # Nid
                'i2', # Cid
                f'i3*{self.stride[0]}-{self.padding[0]}+i5*{self.dilation[0]}', # Hid+Khid
                f'i4*{self.stride[1]}-{self.padding[1]}+i6*{self.dilation[1]}', # Wid+KWid
            ])
            ww = self.weight.broadcast(xx.shape, [0,3,4])
            yy = xx*ww
            y = yy.sum([2,5,6]) # Kc, Kh, Kw
            if self.bias is not None:
                b = self.bias.broadcast(y.shape, [0,2,3])
                y = y + b
            return y
        else:
            N,C,H,W = x.shape
            Kh, Kw = self.kernel_size
            G = self.groups
            CpG = C // G # channels per group
            assert C==self.in_channels
            oc = self.out_channels
            oh = (H+self.padding[0]*2-Kh*self.dilation[0]+self.dilation[0]-1)//self.stride[0]+1
            ow = (W+self.padding[1]*2-Kw*self.dilation[1]+self.dilation[1]-1)//self.stride[1]+1
            xx = x.reindex([N,G,oc//G,CpG,oh,ow,Kh,Kw], [
                'i0', # Nid
                f'i1*{CpG}+i3', # Gid
                f'i4*{self.stride[0]}-{self.padding[0]}+i6*{self.dilation[0]}', # Hid+Khid
                f'i5*{self.stride[1]}-{self.padding[1]}+i7*{self.dilation[1]}', # Wid+KWid
            ])
            xx.compile_options = {"G":G}
            # w: [oc, CpG, Kh, Kw]
            ww = self.weight.reindex([N, G, oc//G, CpG, oh, ow, Kh, Kw], [
                f'i1*{oc//G}+i2',
                'i3',
                'i6',
                'i7'
            ])
            yy = xx*ww
            y = yy.reindex_reduce('add', [N, oc, oh, ow], [
                'i0',
                f'i1*{oc//G}+i2',
                'i4',
                'i5'
            ])
            if self.bias is not None:
                b = self.bias.broadcast(y.shape, [0,2,3])
                y = y + b
            return y          


class ConvTranspose(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, \
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        self.in_channels = in_channels
        self.out_channels = out_channels

        # added
        self.dilation = dilation
        self.group = groups
        assert groups==1, "Group conv not supported yet."

        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        # added
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.real_padding = (self.dilation[0] * (self.kernel_size[0] - 1) - self.padding[0],
            self.dilation[1] * (self.kernel_size[1] - 1) - self.padding[1])
        self.output_padding = output_padding if isinstance (output_padding, tuple) else (output_padding, output_padding)

        self.weight = init.invariant_uniform((in_channels, out_channels) + self.kernel_size, dtype="float")
        if bias:
            self.bias = init.uniform([out_channels], dtype="float", low=-1, high=1)
        else:
            self.bias = None

    def execute(self, x):
        N,C,H,W = x.shape
        i,o,h,w = self.weight.shape
        assert C==i
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        dilation_h, dilation_w = self.dilation

        h_out = (H-1) * stride_h + self.output_padding[0] - 2*padding_h + 1 + (h-1)*dilation_h
        w_out = (W-1) * stride_w + self.output_padding[1] - 2*padding_w + 1 + (w-1)*dilation_w
        out_shape = (N, o, h_out, w_out)
        shape = (N, i, o, H, W, h, w)
        xx = x.broadcast(shape, (2, 5, 6)) # i,h,w
        ww = self.weight.broadcast(shape, (0, 3, 4)) # N,H,W
        y = (ww*xx).reindex_reduce("add", out_shape, [
            'i0', # N
            'i2', # o
            f'i3*{stride_h}-{padding_h}+i5*{dilation_h}', # Hid+Khid
            f'i4*{stride_w}-{padding_w}+i6*{dilation_w}', # Wid+KWid
        ])
        if self.bias is not None:
            b = self.bias.broadcast(y.shape, [0,2,3])
            y = y + b
        return y


class ReflectionPad2d(Module):
    def __init__(self, padding):
        self.padding = padding
        if isinstance(self.padding, int):
            self.pl = self.padding
            self.pr = self.padding
            self.pt = self.padding
            self.pb = self.padding
        elif isinstance(self.padding, tuple):
            self.pl, self.pr, self.pt, self.pb = self.padding
        else:
            raise TypeError(f"ReflectionPad2d padding just support int or tuple, but found {type(padding)}")

    def execute(self, x):
        n,c,h,w = x.shape
        assert (self.pl < w and self.pr < w), f"padding_left and padding_right should be smaller than input width"
        assert (self.pt < h and self.pb < h), f"padding_top and padding_bottom should be smaller than input height"
        oh=h+self.pt+self.pb
        ow=w+self.pl+self.pr
        l = self.pl
        r = self.pl + w - 1
        t = self.pt
        b = self.pt + h - 1
        return x.reindex([n,c,oh,ow], ["i0","i1",
            f"i2<{t} ? {t}-i2 : i2 > {b} ? {h-1+b}-i2 : i2-{t}",
            f"i3<{l} ? {l}-i3 : i3 > {r} ? {w-1+r}-i3 : i3-{l}",
        ])

class ZeroPad2d(Module):
    def __init__(self, padding):
        self.padding = padding
        if isinstance(self.padding, int):
            self.pl = self.padding
            self.pr = self.padding
            self.pt = self.padding
            self.pb = self.padding
        elif isinstance(self.padding, tuple):
            self.pl, self.pr, self.pt, self.pb = self.padding
        else:
            raise TypeError(f"ZeroPad2d padding just support int or tuple, but found {type(padding)}")

    def execute(self, x):
        n,c,h,w = x.shape
        return x.reindex([n,c,h+self.pt+self.pb,w+self.pl+self.pr], ["i0","i1",f"i2-{self.pt}",f"i3-{self.pl}"])

class ConstantPad2d(Module):
    def __init__(self, padding, value):
        self.padding = padding
        if isinstance(self.padding, int):
            self.pl = self.padding
            self.pr = self.padding
            self.pt = self.padding
            self.pb = self.padding
        elif isinstance(self.padding, tuple):
            self.pl, self.pr, self.pt, self.pb = self.padding
        else:
            raise TypeError(f"ConstantPad2d padding just support int or tuple, but found {type(padding)}")
        self.value = value

    def execute(self, x):
        assert len(x.shape) >= 2
        shape = x.shape
        tar_shape = shape[0:-2] + [shape[-2]+self.pt+self.pb,shape[-1]+self.pl+self.pr]
        tar_dims = []
        for i in range(len(shape)-2):
            tar_dims.append(f"i{i}")
        tar_dims.append(f"i{i+1}-{self.pt}")
        tar_dims.append(f"i{i+2}-{self.pl}")
        return x.reindex(tar_shape, tar_dims, overflow_value=self.value)

class ReplicationPad2d(Module):
    def __init__(self, padding):
        self.padding = padding
        if isinstance(self.padding, int):
            self.pl = self.padding
            self.pr = self.padding
            self.pt = self.padding
            self.pb = self.padding
        elif isinstance(self.padding, tuple):
            self.pl, self.pr, self.pt, self.pb = self.padding
        else:
            raise TypeError(f"ReplicationPad2d padding just support int or tuple, but found {type(padding)}")

    def execute(self, x):
        n,c,h,w = x.shape
        oh=h+self.pt+self.pb
        ow=w+self.pl+self.pr
        l = self.pl
        r = self.pl + w - 1
        t = self.pt
        b = self.pt + h - 1
        return x.reindex([n,c,oh,ow], ["i0","i1",
            f"i2<{t} ? 0 : i2 > {b} ? {h-1} : i2-{t}",
            f"i3<{l} ? 0 : i3 > {r} ? {w-1} : i3-{l}"
        ])

class Embedding(Module):
    def __init__(self, num, dim):
        self.num = num
        self.dim = dim
        self.weight = jt.init.gauss([num,dim],'float32').stop_grad()

    def execute(self, x):
        res = self.weight[x].reshape([x.shape[0],self.dim])
        return res

class PixelShuffle(Module):
    def __init__(self, upscale_factor):
        self.upscale_factor = upscale_factor

    def execute(self, x):
        n,c,h,w = x.shape
        r = self.upscale_factor
        assert c%(r*r)==0, f"input channel needs to be divided by upscale_factor's square in PixelShuffle"
        return x.reindex([n,int(c/r**2),h*r,w*r], [
            "i0",
            f"i1*{r*r}+i2%{r}*{r}+i3%{r}",
            f"i2/{r}",
            f"i3/{r}"
        ])

class Tanh(Module):
    def __init__(self):
        super().__init__()
    def execute(self, x) :
        return x.tanh()

class Sigmoid(Module):
    def __init__(self):
        super().__init__()
    def execute(self, x) :
        return x.sigmoid()

class Softplus(Module):
    r'''
    SoftPlus is a smooth approximation to the ReLU function and can be used to constrain the output of a machine to always be positive.

    Args:
        
        [in] beta (float): the beta value for the Softplus formulation. Default: 1.

        [in] threshold (float): values above this revert to a linear function. Default: 20.
    '''
    def __init__(self, beta=1, threshold=20):
        self.beta = beta
        self.threshold = threshold

    def execute(self, x):
        return 1 / self.beta * jt.log(1 + (self.beta * x).exp())

class Resize(Module):
    def __init__(self, size, mode="nearest", align_corners=False):
        super().__init__()
        self.size = size
        self.mode = mode
        self.align_corners = align_corners
    def execute(self, x):
        return resize(x, self.size, self.mode, self.align_corners)

def _interpolate(img, x, y, ids, mode):
    if mode=="nearest":
        return img.reindex([*ids, x.floor(), y.floor()])
    if mode=="bilinear":
        fx, fy = x.floor(), y.floor()
        cx, cy = fx+1, fy+1
        dx, dy = x-fx, y-fy
        a = img.reindex_var([*ids, fx, fy])
        b = img.reindex_var([*ids, cx, fy])
        c = img.reindex_var([*ids, fx, cy])
        d = img.reindex_var([*ids, cx, cy])
        dnx, dny = 1-dx, 1-dy
        ab = dx*b + dnx*a
        cd = dx*d + dnx*c
        o = ab*dny + cd*dy
        return o
    raise(f"Not support interpolation mode: {mode}")
    
def resize(img, size, mode="nearest", align_corners=False):
    n,c,h,w = img.shape
    H,W = size
    nid, cid, hid, wid = jt.index((n,c,H,W))
    if align_corners:
        x = hid * ((h-1) / max(1, H-1))
        y = wid * ((w-1) / max(1, W-1))
    else:
        x = hid * (h / H) + (h/H*0.5 - 0.5)
        if H>h: x = x.clamp(0, h-1)
        y = wid * (w / W) + (w/W*0.5 - 0.5)
        if W>w: y = y.clamp(0, w-1)
    return _interpolate(img, x, y, (nid,cid), mode)

def upsample(img, size, mode="nearest", align_corners=False):
    n,c,h,w = img.shape
    H,W = size
    nid, cid, hid, wid = jt.index((n,c,H,W))
    if align_corners:
        x = hid * ((h-1) / max(1, H-1))
        y = wid * ((w-1) / max(1, W-1))
    else:
        x = hid * (h / H)
        y = wid * (w / W)
    return _interpolate(img, x, y, (nid,cid), mode)

def grid_sample_v0(input, grid, mode='bilinear', padding_mode='zeros'):
    r'''
    Given an input and a flow-field grid, computes the output using input values and pixel locations from grid.

    grid specifies the sampling pixel locations normalized by the input spatial dimensions. Therefore, it should have most values in the range of [-1, 1]. For example, values x = -1, y = -1 is the left-top pixel of input, and values x = 1, y = 1 is the right-bottom pixel of input.

    Args:

        [in] input (var): the source input var, whose shape is (N, C, Hi, Wi)

        [in] grid (var): the pixel locations, whose shape is (N, Ho, Wo, 2)

        [in] mode (string): the interpolate way, default: bilinear.

        [in] padding_mode (string): the padding way, default: zeros.
        
        [out] output (var): the output var, whose shape is (N, C, Ho, Wo)

    Example:

        >>> x = jt.array([[[[1,2],[3,4]]]])
        >>> print(x)
        [[[[1 2]
        [3 4]]]] 

        >>> grid = jt.array([[[[0.5, 0.5]]]])
        >>> print(x.shape, grid.shape)
        [1,1,2,2,], [1,1,2,2,]

        >>> nn.grid_sample(x, grid)
        [[[[3.25]]]]
    '''
    assert padding_mode == 'zeros'
    Ni, Ci, Hi, Wi = input.shape
    No, Ho, Wo, D = grid.shape
    assert D == 2
    assert Ni == No
    assert len(input.shape) == 4 and len(grid.shape)

    nid, cid, hid, wid = jt.index((Ni,Ci,Ho,Wo))
    x = ((grid[:,:,:,1].unsqueeze(1).repeat([1,Ci,1,1]) + 1) / 2) * (Hi - 1)
    y = ((grid[:,:,:,0].unsqueeze(1).repeat([1,Ci,1,1]) + 1) / 2) * (Wi - 1)
    return _interpolate(input, x, y, (nid,cid), mode)


def linspace_from_neg_one(grid,num_steps,align_corners):
    if  num_steps <= 1:
        return jt.array([],dtype=grid.dtype)
    ra = np.linspace(-1,1,num_steps)
    if not align_corners:
        ra = ra*(num_steps-1)/num_steps
    return jt.array(ra,dtype=grid.dtype)

def make_base_grid_4D(theta,N,C,H,W,align_corners):
    base_grid = jt.zeros((N, H, W, 3), dtype=theta.dtype);
    base_grid[:,:,:,0] = linspace_from_neg_one(theta, W, align_corners)
    base_grid[:,:,:,1] = jt.unsqueeze(linspace_from_neg_one(theta, H, align_corners),-1)
    base_grid[:,:,:,2] = 1
    return base_grid

def make_base_grid_5D(theta,N,C,D,H,W,align_corners):
    base_grid = jt.zeros((N, D, H, W, 4), dtype=theta.dtype)

    base_grid[:,:,:,:,0] = linspace_from_neg_one(theta, W, align_corners)
    base_grid[:,:,:,:,1] = jt.unsqueeze(linspace_from_neg_one(theta, H, align_corners),-1)
    base_grid[:,:,:,:,2] = jt.unsqueeze(jt.unsqueeze(linspace_from_neg_one(theta, D, align_corners),-1),-1)
    base_grid[:,:,:,:,3] = 1
    return base_grid

def affine_grid_generator_4D(theta,N,C,H,W,align_corners):
     base_grid = make_base_grid_4D(theta, N, C, H, W, align_corners)
     grid = jt.nn.bmm(base_grid.reshape(N, H * W, 3),theta.transpose(0,2,1))
     return grid.reshape(N, H, W, 2)

def affine_grid_generator_5D(theta,N,C,D,H,W,align_corners):
    base_grid = make_base_grid_5D(theta, N, C, D, H, W, align_corners)
    grid = jt.nn.bmm(base_grid.reshape(N, D * H * W, 4),theta.transpose(0,2,1))
    return grid.reshape(N, D, H, W, 3)


def affine_grid_generator(theta, size,align_corners):
    if len(size)== 4:
        return affine_grid_generator_4D(theta, size[0], size[1], size[2], size[3], align_corners)
    elif len(size)==5:
        return affine_grid_generator_5D(theta, size[0], size[1], size[2], size[3], size[4], align_corners)
    else:
        raise('AffineGridGenerator needs 4d (spatial) or 5d (volumetric) inputs.')


def affine_grid(theta, size, align_corners=None):
    # type: (Tensor, List[int], Optional[bool]) -> Tensor
    # Convert from pytorch
    r"""Generates a 2D or 3D flow field (sampling grid), given a batch of
    affine matrices :attr:`theta`.
    .. note::
        This function is often used in conjunction with :func:`grid_sample`
        to build `Spatial Transformer Networks`_ .
    Args:
        theta (Tensor): input batch of affine matrices with shape
            (:math:`N \times 2 \times 3`) for 2D or
            (:math:`N \times 3 \times 4`) for 3D
        size (torch.Size): the target output image size.
            (:math:`N \times C \times H \times W` for 2D or
            :math:`N \times C \times D \times H \times W` for 3D)
            Example: torch.Size((32, 3, 24, 24))
        align_corners (bool, optional): if ``True``, consider ``-1`` and ``1``
            to refer to the centers of the corner pixels rather than the image corners.
            Refer to :func:`grid_sample` for a more complete description.
            A grid generated by :func:`affine_grid` should be passed to :func:`grid_sample`
            with the same setting for this option.
            Default: ``False``
    Returns:
        output (Tensor): output Tensor of size (:math:`N \times H \times W \times 2`)
    .. _`Spatial Transformer Networks`:
        https://arxiv.org/abs/1506.02025
    .. warning::
        When ``align_corners = True``, the grid positions depend on the pixel
        size relative to the input image size, and so the locations sampled by
        :func:`grid_sample` will differ for the same input given at different
        resolutions (that is, after being upsampled or downsampled).
        The default behavior up to version 1.2.0 was ``align_corners = True``.
        Since then, the default behavior has been changed to ``align_corners = False``,
        in order to bring it in line with the default for :func:`interpolate`.
    .. warning::
        When ``align_corners = True``, 2D affine transforms on 1D data and
        3D affine transforms on 2D data (that is, when one of the spatial
        dimensions has unit size) are ill-defined, and not an intended use case.
        This is not a problem when ``align_corners = False``.
        Up to version 1.2.0, all grid points along a unit dimension were
        considered arbitrarily to be at ``-1``.
        From version 1.3.0, under ``align_corners = True`` all grid points
        along a unit dimension are considered to be at ```0``
        (the center of the input image).
    """
    if align_corners is None:
        align_corners = False

    # enforce floating point dtype on theta
    if theta.dtype!='float' and theta.dtype!='float32' and theta.dtype!='float64':
        raise ValueError("Expected theta to have floating point type, but got {}"
                         .format(theta.dtype))
    # check that shapes and sizes match
    if len(size) == 4:
        if theta.ndim != 3 or theta.shape[-2] != 2 or theta.shape[-1] != 3:
            raise ValueError("Expected a batch of 2D affine matrices of shape Nx2x3 "
                             "for size {}. Got {}.".format(size, theta.shape))
        spatial_size = size[-2:]  # spatial dimension sizes
    elif len(size) == 5:
        if theta.ndim != 3 or theta.shape[-2] != 3 or theta.shape[-1] != 4:
            raise ValueError("Expected a batch of 3D affine matrices of shape Nx3x4 "
                             "for size {}. Got {}.".format(size, theta.shape))
        spatial_size = size[-3:]  # spatial dimension sizes
    else:
        raise NotImplementedError("affine_grid only supports 4D and 5D sizes, "
                                  "for 2D and 3D affine transforms, respectively. "
                                  "Got size {}.".format(size))
    # check for empty span
    if align_corners and min(spatial_size) == 1:
        warnings.warn("Since version 1.3.0, affine_grid behavior has changed "
                      "for unit-size grids when align_corners=True. "
                      "This is not an intended use case of affine_grid. "
                      "See the documentation of affine_grid for details.")
    elif min(size) <= 0:
        raise ValueError("Expected non-zero, positive output size. Got {}"
                         .format(size))

    return affine_grid_generator(theta, size, align_corners)



def grid_sampler_unnormalize(coord,size,align_corners):
    if align_corners:
        #unnormalize coord from [-1, 1] to [0, size - 1]
        return ((coord + 1) / 2) * (size - 1)
    else:
        #unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
        return ((coord + 1) * size - 1) / 2


def clip_coordinates(x,clip_limit):
    return jt.minimum(clip_limit - 1, jt.maximum(x, 0))

def reflect_coordinates(x,twice_low,twice_high):
    if twice_low == twice_high:
        return jt.zeros(x.shape,dtype=x.dtype)
    m = twice_low / 2
    span = (twice_high - twice_low) / 2
    x = (x - m).abs()
    #`fmod` returns same sign as `in`, which is positive after the `fabs` above.
    extra = x.mod(span)

    flips = (x / span).floor()
    result1 = extra+m
    result2 = span-extra+m
    con = flips%2==0
    not_con = flips%2!=0
    result1[not_con]=0.0
    result2[con]=0.0
    return result1+result2


def grid_sampler_compute_source_index(coord,size,padding_mode,align_corners):
    coord = grid_sampler_unnormalize(coord, size, align_corners)
    if padding_mode == 'border':
        #clip coordinates to image borders
        coord = clip_coordinates(coord, size)
    elif padding_mode == 'reflection':
        #reflect coordinates by image borders
        if align_corners:
           coord = reflect_coordinates(coord, 0, 2*(size - 1))
        else:
           coord = reflect_coordinates(coord, -1, 2*size - 1)
        print('coord',coord)
        #clip coordinates to image borders
        coord = clip_coordinates(coord, size)
    return coord



def grid_sampler_3d(X,grid,mode,padding_mode,align_corners):
    N = X.shape[0]
    C = X.shape[1]
    inp_D = X.shape[2]
    inp_H = X.shape[3]
    inp_W = X.shape[4]

    D  = grid.shape[1]
    H = grid.shape[2]
    W = grid.shape[3]
    x = grid[:,:,:,:,0]
    y = grid[:,:,:,:,1]
    z = grid[:,:,:,:,2]
    shape = [N,C,D,H,W]
    cid = jt.index(shape, dim=1)
    nid = jt.index(shape, dim=0)

    x = grid_sampler_compute_source_index(x,inp_W,padding_mode,align_corners)
    y = grid_sampler_compute_source_index(y,inp_H,padding_mode,align_corners)
    z = grid_sampler_compute_source_index(z,inp_D,padding_mode,align_corners)
    xid = x.reindex(shape,['i0','i2','i3','i4'])
    yid = y.reindex(shape,['i0','i2','i3','i4'])
    zid = z.reindex(shape,['i0','i2','i3','i4'])

    if mode=='nearest':
        return X.reindex([nid,cid,zid.round(),yid.round(),xid.round()])
    elif mode=='bilinear':
        fx,fy,fz = xid.floor(),yid.floor(),zid.floor()
        cx,cy,cz = fx+1,fy+1,fz+1
        dx,dy,dz = xid-fx,yid-fy,zid-fz
        dnx,dny,dnz = cx-xid,cy-yid,cz-zid
        a = X.reindex([nid,cid,fz,fy,fx])
        b = X.reindex([nid,cid,cz,fy,fx])
        c = X.reindex([nid,cid,fz,cy,fx])
        d = X.reindex([nid,cid,fz,fy,cx])
        e = X.reindex([nid,cid,fz,cy,cx])
        f = X.reindex([nid,cid,cz,fy,cx])
        g = X.reindex([nid,cid,cz,cy,fx])
        h = X.reindex([nid,cid,cz,cy,cx])

        o = a*dnx*dny*dnz+b*dnx*dny*dz+c*dnx*dy*dnz+d*dx*dny*dnz+e*dx*dy*dnz+f*dx*dny*dz+g*dnx*dy*dz+h*dx*dy*dz
        return o


    raise ValueError(f"Not support {mode}")

def grid_sampler_2d(X,grid,mode,padding_mode,align_corners):
    N = X.shape[0]
    C = X.shape[1]
    inp_H = X.shape[2]
    inp_W = X.shape[3]

    H  = grid.shape[1]
    W = grid.shape[2]
    x = grid[:,:,:,0]
    y = grid[:,:,:,1]
    shape = [N,C,H,W]
    cid = jt.index(shape, dim=1)
    nid = jt.index(shape, dim=0)

    x = grid_sampler_compute_source_index(x,inp_W,padding_mode,align_corners)
    y = grid_sampler_compute_source_index(y,inp_H,padding_mode,align_corners)
    xid = x.unsqueeze(1).repeat(1,C,1,1)
    yid = y.unsqueeze(1).repeat(1,C,1,1)


    if mode=='nearest':
        return X.reindex([nid,cid,yid.round(),xid.round()])
    elif mode=='bilinear':
        fx,fy = xid.floor(),yid.floor()
        cx,cy = fx+1,fy+1
        dx,dy = xid-fx,yid-fy
        dnx,dny = cx-xid,cy-yid

        a = X.reindex([nid,cid,fy,fx])
        b = X.reindex([nid,cid,cy,fx])
        c = X.reindex([nid,cid,fy,cx])
        d = X.reindex([nid,cid,cy,cx])

        o = a*dnx*dny+b*dnx*dy+c*dx*dny+d*dx*dy
        return o
    raise ValueError(f"Not support {mode}")


def grid_sampler(X, grid, mode, padding_mode, align_corners):
    if X.dtype!=grid.dtype:
        raise(f' "grid_sampler(): expected input and grid to have same dtype, but input has {X.dtype} and grid has {grid.dtype}')
    if not ((X.ndim==4 or X.ndim==5) and X.ndim==grid.ndim):
        raise(f'grid_sampler(): expected 4D or 5D input and grid with same number of dimensions, but got input with sizes {X.ndim}, and grid with sizes {grid.ndim}.')
    if X.shape[0]!=grid.shape[0]:
        raise(f'grid_sampler(): expected grid and input to have same batch size, but got input with batchsize {X.shape[0]}, and grid with batchsize {gird.shape[0]}')
    if grid.shape[-1]!=X.ndim-2:
        raise(f'grid_sampler(): expected grid to have size {X.ndim-2} in last dimension, but got grid with last dimension {grid.shape[-1]}.')
    for i in range(2,X.ndim):
        if X.shape[i]==0:
            raise(f'grid_sampler(): expected input to have non-empty spatial dimensions, but input has size {X.shape[i]} with dimension {i} being empty.')

    if X.ndim == 4:
        return grid_sampler_2d(X, grid, mode, padding_mode, align_corners)
    else:
        return grid_sampler_3d(X, grid, mode, padding_mode, align_corners)


def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None):
    # type: (Tensor, Tensor, str, str, Optional[bool]) -> Tensor
    r"""Given an :attr:`input` and a flow-field :attr:`grid`, computes the
    ``output`` using :attr:`input` values and pixel locations from :attr:`grid`.
    Currently, only spatial (4-D) and volumetric (5-D) :attr:`input` are
    supported.
    In the spatial (4-D) case, for :attr:`input` with shape
    :math:`(N, C, H_\text{in}, W_\text{in})` and :attr:`grid` with shape
    :math:`(N, H_\text{out}, W_\text{out}, 2)`, the output will have shape
    :math:`(N, C, H_\text{out}, W_\text{out})`.
    For each output location ``output[n, :, h, w]``, the size-2 vector
    ``grid[n, h, w]`` specifies :attr:`input` pixel locations ``x`` and ``y``,
    which are used to interpolate the output value ``output[n, :, h, w]``.
    In the case of 5D inputs, ``grid[n, d, h, w]`` specifies the
    ``x``, ``y``, ``z`` pixel locations for interpolating
    ``output[n, :, d, h, w]``. :attr:`mode` argument specifies ``nearest`` or
    ``bilinear`` interpolation method to sample the input pixels.
    :attr:`grid` specifies the sampling pixel locations normalized by the
    :attr:`input` spatial dimensions. Therefore, it should have most values in
    the range of ``[-1, 1]``. For example, values ``x = -1, y = -1`` is the
    left-top pixel of :attr:`input`, and values  ``x = 1, y = 1`` is the
    right-bottom pixel of :attr:`input`.
    If :attr:`grid` has values outside the range of ``[-1, 1]``, the corresponding
    outputs are handled as defined by :attr:`padding_mode`. Options are
        * ``padding_mode="zeros"``: use ``0`` for out-of-bound grid locations,
        * ``padding_mode="border"``: use border values for out-of-bound grid locations,
        * ``padding_mode="reflection"``: use values at locations reflected by
          the border for out-of-bound grid locations. For location far away
          from the border, it will keep being reflected until becoming in bound,
          e.g., (normalized) pixel location ``x = -3.5`` reflects by border ``-1``
          and becomes ``x' = 1.5``, then reflects by border ``1`` and becomes
          ``x'' = -0.5``.
    Note:
        This function is often used in conjunction with :func:`affine_grid`
        to build `Spatial Transformer Networks`_ .
    Note:
        When using the CUDA backend, this operation may induce nondeterministic
        behaviour in its backward pass that is not easily switched off.
        Please see the notes on :doc:`/notes/randomness` for background.
    Note:
        NaN values in :attr:`grid` would be interpreted as ``-1``.
    Args:
        input (Tensor): input of shape :math:`(N, C, H_\text{in}, W_\text{in})` (4-D case)
                        or :math:`(N, C, D_\text{in}, H_\text{in}, W_\text{in})` (5-D case)
        grid (Tensor): flow-field of shape :math:`(N, H_\text{out}, W_\text{out}, 2)` (4-D case)
                       or :math:`(N, D_\text{out}, H_\text{out}, W_\text{out}, 3)` (5-D case)
        mode (str): interpolation mode to calculate output values
            ``'bilinear'`` | ``'nearest'``. Default: ``'bilinear'``
        padding_mode (str): padding mode for outside grid values
            ``'zeros'`` | ``'border'`` | ``'reflection'``. Default: ``'zeros'``
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input  as squares rather than points.
            If set to ``True``, the extrema (``-1`` and ``1``) are considered as referring
            to the center points of the input's corner pixels. If set to ``False``, they
            are instead considered as referring to the corner points of the input's corner
            pixels, making the sampling more resolution agnostic.
            This option parallels the ``align_corners`` option in
            :func:`interpolate`, and so whichever option is used here
            should also be used there to resize the input image before grid sampling.
            Default: ``False``
    Returns:
        output (Tensor): output Tensor
    .. _`Spatial Transformer Networks`:
        https://arxiv.org/abs/1506.02025
    .. warning::
        When ``align_corners = True``, the grid positions depend on the pixel
        size relative to the input image size, and so the locations sampled by
        :func:`grid_sample` will differ for the same input given at different
        resolutions (that is, after being upsampled or downsampled).
        The default behavior up to version 1.2.0 was ``align_corners = True``.
        Since then, the default behavior has been changed to ``align_corners = False``,
        in order to bring it in line with the default for :func:`interpolate`.
    """
    if mode != 'bilinear' and mode != 'nearest':
        raise ValueError("nn.grid_sample(): expected mode to be "
                         "'bilinear' or 'nearest', but got: '{}'".format(mode))
    if padding_mode != 'zeros' and padding_mode != 'border' and padding_mode != 'reflection':
        raise ValueError("nn.grid_sample(): expected padding_mode "
                         "to be 'zeros', 'border', or 'reflection', "
                         "but got: '{}'".format(padding_mode))

    if align_corners is None:
        warnings.warn("Default grid_sample and affine_grid behavior has changed "
                      "to align_corners=False since 1.3.0. Please specify "
                      "align_corners=True if the old behavior is desired. "
                      "See the documentation of grid_sample for details.")
        align_corners = False

    return grid_sampler(input, grid, mode, padding_mode, align_corners)


class Upsample(Module):
    def __init__(self, scale_factor=None, mode='nearest'):
        self.scale_factor = scale_factor if isinstance(scale_factor, tuple) else (scale_factor, scale_factor)
        self.mode = mode
    
    def execute(self, x):
        return upsample(x,
            size=(
                int(x.shape[2]*self.scale_factor[0]), 
                int(x.shape[3]*self.scale_factor[1])),
            mode=self.mode)

class Sequential(Module):
    def __init__(self, *args):
        self.layers = []
        for mod in args:
            self.append(mod)
    def __getitem__(self, idx):
        return self.layers[idx]
    def execute(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def dfs(self, parents, k, callback, callback_leave):
        n_children = len(self.layers)
        ret = callback(parents, k, self, n_children)
        if ret == False:
            return
        for k,v in enumerate(self.layers):
            parents.append(self)
            v.dfs(parents, k, callback, callback_leave)
            parents.pop()
        if callback_leave:
            callback_leave(parents, k, self, n_children)
    def append(self, mod):
        assert callable(mod), f"Module <{type(mod)}> is not callable"
        assert not isinstance(mod, type), f"Module is not a type"
        self.layers.append(mod)

ModuleList = Sequential

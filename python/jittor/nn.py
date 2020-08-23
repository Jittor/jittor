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

class CrossEntropyLoss(Module):
    def __init__(self):
        pass
    def execute(self, output, target):
        return cross_entropy_loss(output, target)

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

def grid_sample(input, grid, mode='bilinear', padding_mode='zeros'):
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

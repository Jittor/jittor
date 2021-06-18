# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers:
#     Guowei Yang <471184555@qq.com>
#     Guoye Yang <498731903@qq.com>
#     Wenyang Zhou <576825820@qq.com>
#     Meng-Hao Guo <guomenghao1997@gmail.com>
#     Dun Liang <randonlang@gmail.com>.
#     Zheng-Ning Liu <lzhengning@gmail.com>
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
from abc import abstractmethod
import jittor as jt
from jittor import init, Module
import numpy as np
import collections
import math
from collections import OrderedDict
from jittor.pool import *
from jittor.optim import *
from jittor.misc import _pair, _triple
from jittor_utils import LOG


def matmul_transpose(a, b):
    '''
    returns a * b^T
    '''
    assert a.shape[-1] == b.shape[-1], (a.shape, b.shape)
    if len(a.shape) != 2:
        aa = a.reshape((-1, a.shape[-1]))
        cc = matmul_transpose(aa, b)
        return cc.reshape(a.shape[:-1]+(-1,))
    assert len(a.shape) == 2 and len(b.shape) == 2

    shape = list(a.shape)[:-1] + list(b.shape)
    a = a.broadcast(shape, [len(shape)-2])
    b = b.broadcast(shape)
    return (a*b).sum(len(shape)-1)


def bmm_transpose(a, b):
    '''
    returns a * b^T
    '''
    if jt.flags.use_cuda:
        return jt.compile_extern.cublas_ops.cublas_batched_matmul(a, b, 0, 1)
    t = list(range(b.ndim))
    t[-1], t[-2] = t[-2], t[-1]
    return bmm(a, b.transpose(t))


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
    if len_b == 2 and len_a>2:
        # TODO:ugly implementation for tuner
        aa = a.reshape((-1, m))
        cc = matmul(aa, b)
        # print(a.shape, b.shape, cc.shape) 
        return cc.reshape(a.shape[:-1] + [k])
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

def relu(x): return jt.ternary((x>0.0), x, jt.broadcast_var(0.0, x))
def leaky_relu(x, scale=0.01): return jt.ternary(x>0, x, x*scale)
def relu6(x): return jt.minimum(jt.maximum(x, 0.0), 6.0)
def elu(x,alpha=1.0):return jt.ternary(x>0,x,alpha*(x.exp()-1))
def sign(x):
    one = jt.ones(x.shape)
    x = jt.ternary(x>0, one, x)
    return jt.ternary(x<0, -one, x)

def gelu(x):
    _sqrt2 = 1.4142135623730951
    erf = jt.erf(x/_sqrt2)+1
    r = erf*x*.5
    return r

class ELU(Module):
    def __init__(self,alpha=1.0):
        self.alpha=alpha
    
    def execute(self,x):
        return elu(x,self.alpha)

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
    loss = (less_than_one * 0.5 * diff.sqr()) + (1 - less_than_one) * (diff - 0.5)
    if reduction=="mean":
        return loss.mean()
    elif reduction=="sum":
        return loss.sum()
    elif reduction=="none":
        return loss
    else:
        raise ValueError(f'not support {reduction}')

def nll_loss(output,target,weight=None,ignore_index=-100,reduction='mean'):
    assert output.ndim<=2 and output.ndim>0 and target.ndim==1
    n_classes = output.shape[-1]
    assert weight is None or weight.numel()==n_classes
    assert ignore_index<0 or ignore_index<n_classes
    if weight is None:
        weight = jt.ones((n_classes,))
    if ignore_index>0:
        weight[ignore_index]=0
    if output.ndim==2:
        index = jt.index((output.shape[0],),dim=0)
        loss = -output[index,target]*weight[target]
    else:
        loss = -output[target[0]]*weight[target[0]]
    if reduction=="mean":
        total_weight  = weight[target].sum() if output.ndim==2 else weight[target[0]].sum()
        return loss.sum()/total_weight
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

def binary_cross_entropy_with_logits(output, target, weight=None, pos_weight=None, size_average=True):
    max_val = jt.clamp(-output,min_v=0)
    if pos_weight is not None:
        log_weight = (pos_weight-1)*target + 1
        loss = (1-target)*output+(log_weight*(((-max_val).exp()+(-output - max_val).exp()).log()+max_val))
    else:
        loss = (1-target)*output+max_val+((-max_val).exp()+(-output -max_val).exp()).log()
    if weight is not None:
        loss *=weight

    if size_average:
        return loss.mean()
    else:
        return loss.sum()

class BCEWithLogitsLoss(Module):
    def __init__(self, weight=None, pos_weight=None, size_average=True):
        self.pos_weight = pos_weight
        self.weight = weight
        self.size_average = size_average

    def execute(self, output, target):
        return binary_cross_entropy_with_logits(output,target,self.weight,self.pos_weight,self.size_average)

def softmax(x, dim = None):
    if dim is None:
        x = (x - x.max()).exp()
        ret = x / x.sum()
    else:
        x = (x-x.max(dim, keepdims=True)).exp()
        ret = x / x.sum(dim, keepdims=True)
    return ret

def log_softmax(x,dim=None):
    x = softmax(x,dim=dim)
    return jt.log(x)

def log_sigmoid(x):
    return jt.log(jt.sigmoid(x))

def logsumexp(x, dim, keepdim=False):
    return x.exp().sum(dim, keepdim).log()

class Identity(Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def execute(self, input):
        return input

def identity(input): return input

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

def dropout(x,p=0.5,is_train=False):
    return Dropout(p=p,is_train=is_train)(x)

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
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, is_train=True, sync=True):
        self.sync = sync
        self.num_features = num_features
        self.is_train = is_train
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.weight = init.constant((num_features,), "float32", 1.0) if affine else 1.0
        self.bias = init.constant((num_features,), "float32", 0.0) if affine else 0.0
        self.running_mean = init.constant((num_features,), "float32", 0.0).stop_grad()
        self.running_var = init.constant((num_features,), "float32", 1.0).stop_grad()

    def execute(self, x):
        dims = [0]+list(range(2,x.ndim))
        if self.is_train:
            xmean = jt.mean(x, dims=dims)
            x2mean = jt.mean(x*x, dims=dims)
            if self.sync and jt.in_mpi:
                xmean = xmean.mpi_all_reduce("mean")
                x2mean = x2mean.mpi_all_reduce("mean")

            xvar = (x2mean-xmean*xmean).maximum(0.0)
            w = self.weight / jt.sqrt(xvar+self.eps)
            b = self.bias - xmean * w
            norm_x = x * w.broadcast(x, dims) + b.broadcast(x, dims)

            self.running_mean.update(self.running_mean +
                (xmean.reshape((-1,)) - self.running_mean) * self.momentum)
            self.running_var.update(self.running_var +
                (xvar.reshape((-1,))-self.running_var)*self.momentum)
            return norm_x
        else:
            w = self.weight / jt.sqrt(self.running_var+self.eps)
            b = self.bias - self.running_mean * w
            norm_x = x * w.broadcast(x, dims) + b.broadcast(x, dims)
            return norm_x

BatchNorm3d = BatchNorm2d = BatchNorm1d = BatchNorm

class InstanceNorm(Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, is_train=True, sync=True):
        self.sync = sync
        self.num_features = num_features
        self.is_train = is_train
        self.eps = eps
        self.momentum = momentum

        self.affine = affine
        self.weight = init.constant((num_features,), "float32", 1.0) if affine else 1.0
        self.bias = init.constant((num_features,), "float32", 0.0) if affine else 0.0

    def execute(self, x):
        dims = list(range(2,x.ndim))
        xmean = jt.mean(x, dims=dims)
        x2mean = jt.mean(x*x, dims=dims)

        xvar = (x2mean-xmean*xmean).maximum(0.0)
        w = self.weight / jt.sqrt(xvar+self.eps)
        b = self.bias - xmean * w
        return x * w.broadcast(x, dims) + b.broadcast(x, dims)

InstanceNorm3d = InstanceNorm2d = InstanceNorm1d = InstanceNorm

class LayerNorm(Module):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = init.constant(normalized_shape, "float32", 1.0) if elementwise_affine else 1.0
        self.bias = init.constant(normalized_shape, "float32", 0.0) if elementwise_affine else 0.0

    def execute(self, x):
        dims = [-i for i in range(len(self.normalized_shape), 0, -1)]
        xmean = jt.mean(x, dims=dims, keepdims=1)
        x2mean = jt.mean(x*x, dims=dims, keepdims=1)

        xvar = (x2mean-xmean*xmean).maximum(0.0)
        w = self.weight / jt.sqrt(xvar+self.eps)
        b = self.bias - xmean * w
        return x * w + b


LayerNorm3d = LayerNorm2d = LayerNorm1d = LayerNorm

class GroupNorm(Module):
    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True, is_train=True):
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps

        self.affine = affine
        self.weight = init.constant((num_channels,), "float32", 1.0) if affine else 1.0
        self.bias = init.constant((num_channels,), "float32", 0.0) if affine else 0.0

    def execute(self, x):
        N = x.shape[0]
        C = self.num_channels
        output_shape = (N,-1)
	    # TODO: 3d group norm
        if x.ndim==4:
            output_shape = x.shape
        assert C % self.num_groups == 0
        x = x.reshape((N, self.num_groups, C//self.num_groups, -1))
        xmean = jt.mean(x, dims=[2,3]).reshape((N, self.num_groups, 1))
        x2mean = jt.mean(x*x, dims=[2,3]).reshape((N, self.num_groups, 1))
        xvar = (x2mean-xmean*xmean).maximum(0.0)

        if self.affine:
            w = self.weight.reshape((1, self.num_groups, -1))
            b = self.bias.reshape((1, self.num_groups, -1))
        else:
            w = 1
            b = 0
        w = w / jt.sqrt(xvar+self.eps)
        b = b - xmean * w
        x = x * w.broadcast(x, [3]) + b.broadcast(x, [3])
        return x.reshape(output_shape)

Relu = jt.make_module(relu)
ReLU = Relu
Leaky_relu = jt.make_module(leaky_relu, 2)
LeakyReLU = Leaky_relu
ReLU6 = jt.make_module(relu6)
Softmax = jt.make_module(softmax, 2)
GELU = jt.make_module(gelu)

from jittor.depthwise_conv import DepthwiseConv

class Conv(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.is_depthwise_conv = self.groups == self.out_channels and self.groups == self.in_channels
        if self.is_depthwise_conv and jt.flags.use_cuda:
            self.depthwise_conv = DepthwiseConv(stride, padding, dilation)
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
        if self.is_depthwise_conv and jt.flags.use_cuda:
            y = self.depthwise_conv(x, self.weight)
            if self.bias is not None:
                b = self.bias.broadcast(y.shape, [0,2,3])
                y = y + b
            return y
        elif self.groups == 1:
            N,C,H,W = x.shape
            Kh, Kw = self.kernel_size
            assert C==self.in_channels
            oh = (H+self.padding[0]*2-Kh*self.dilation[0]+self.dilation[0]-1)//self.stride[0]+1
            ow = (W+self.padding[1]*2-Kw*self.dilation[1]+self.dilation[1]-1)//self.stride[1]+1
            assert oh>0 and ow>0
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
            assert oh>0 and ow>0
            xx = x.reindex([N,G,oc//G,CpG,oh,ow,Kh,Kw], [
                'i0', # Nid
                f'i1*{CpG}+i3', # Gid
                f'i4*{self.stride[0]}-{self.padding[0]}+i6*{self.dilation[0]}', # Hid+Khid
                f'i5*{self.stride[1]}-{self.padding[1]}+i7*{self.dilation[1]}', # Wid+KWid
            ])
            # w: [oc, CpG, Kh, Kw]
            ww = self.weight.reindex([N, G, oc//G, CpG, oh, ow, Kh, Kw], [
                f'i1*{oc//G}+i2',
                'i3',
                'i6',
                'i7'
            ])
            ww.compile_options = xx.compile_options = {"G":G,"C":C}
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

Conv2d = Conv

class Conv1d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, 1)
        self.stride = (stride, 1)
        self.padding = (padding, 0)
        self.dilation = (dilation, 1)
        self.groups = groups
        self.bias = bias
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        assert out_channels % groups == 0, 'out_channels must be divisible by groups'
        # using list to escape module dfs
        self._conv = [Conv(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)]
        self.weight = self._conv[0].weight.squeeze(-1)
        self.bias = self._conv[0].bias

    def execute(self, x):
        N,C,D = x.shape
        assert C==self.in_channels
        self._conv[0].weight = self.weight.unsqueeze(-1)
        x = x.unsqueeze(-1)
        x = self._conv[0](x)
        y = x.squeeze(-1)
        return y

class Conv3d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        assert out_channels % groups == 0, 'out_channels must be divisible by groups'
        Kh, Kw, Kd = self.kernel_size
        self.groups = groups
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        assert out_channels % groups == 0, 'out_channels must be divisible by groups'

        self.weight = init.invariant_uniform([out_channels, in_channels//groups, Kh, Kw, Kd], dtype="float")
        if bias:
            fan=1
            for i in self.weight.shape[1:]:
                fan *= i
            bound = 1 / math.sqrt(fan)
            self.bias = init.uniform([out_channels], dtype="float", low=-bound, high=bound)
        else:
            self.bias = None

    def execute(self, x):
        return conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    padding = _pair(padding)
    stride = _pair(stride)
    dilation = _pair(dilation)
    out_channels = weight.shape[0]

    if groups == 1:
        N,C,H,W = x.shape
        Kh, Kw = weight.shape[-2:]
        oh = (H+padding[0]*2-Kh*dilation[0]+dilation[0]-1)//stride[0]+1
        ow = (W+padding[1]*2-Kw*dilation[1]+dilation[1]-1)//stride[1]+1
        xx = x.reindex([N,out_channels,C,oh,ow,Kh,Kw], [
                'i0', # Nid
                'i2', # Cid
                f'i3*{stride[0]}-{padding[0]}+i5*{dilation[0]}', # Hid+Khid
                f'i4*{stride[1]}-{padding[1]}+i6*{dilation[1]}', # Wid+KWid
            ])
        ww = weight.broadcast(xx.shape, [0,3,4])
        yy = xx*ww
        y = yy.sum([2,5,6]) # Kc, Kh, Kw
        if bias is not None:
            b = bias.broadcast(y.shape, [0,2,3])
            y = y + b
        return y
    else:
        N,C,H,W = x.shape
        Kh, Kw = weight.shape[-2:]
        G = groups
        CpG = C // G # channels per group
        oc = out_channels
        oh = (H+padding[0]*2-Kh*dilation[0]+dilation[0]-1)//stride[0]+1
        ow = (W+padding[1]*2-Kw*dilation[1]+dilation[1]-1)//stride[1]+1
        xx = x.reindex([N,G,oc//G,CpG,oh,ow,Kh,Kw], [
                'i0', # Nid
                f'i1*{CpG}+i3', # Gid
                f'i4*{stride[0]}-{padding[0]}+i6*{dilation[0]}', # Hid+Khid
                f'i5*{stride[1]}-{padding[1]}+i7*{dilation[1]}', # Wid+KWid
            ])
        xx.compile_options = {"G":G}
        # w: [oc, CpG, Kh, Kw]
        ww = weight.reindex([N, G, oc//G, CpG, oh, ow, Kh, Kw], [
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
        if bias is not None:
            b = bias.broadcast(y.shape, [0,2,3])
            y = y + b
        return y

def conv3d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    padding = _triple(padding)
    stride = _triple(stride)
    dilation = _triple(dilation)
    out_channels = weight.shape[0]

    if jt.flags.use_cuda and jt.cudnn:
        return jt.cudnn.ops.cudnn_conv3d(x, weight, *stride, *padding, *dilation, groups)

    if groups == 1:
        N,C,D,H,W = x.shape
        Kd, Kh, Kw = weight.shape[-3:]
        od = (D+padding[0]*2-Kd*dilation[0]+dilation[0]-1)//stride[0]+1
        oh = (H+padding[1]*2-Kh*dilation[1]+dilation[1]-1)//stride[1]+1
        ow = (W+padding[2]*2-Kw*dilation[2]+dilation[2]-1)//stride[2]+1
        xx = x.reindex([N,out_channels,C,od,oh,ow,Kd,Kh,Kw], [
                'i0', # Nid
                'i2', # Cid
                f'i3*{stride[0]}-{padding[0]}+i6*{dilation[0]}', # Hid+Khid
                f'i4*{stride[1]}-{padding[1]}+i7*{dilation[1]}', # Wid+KWid
                f'i5*{stride[2]}-{padding[2]}+i8*{dilation[2]}', # Did+KDid
            ])
        ww = weight.broadcast(xx.shape, [0,3,4,5])
        yy = xx*ww
        y = yy.sum([2,6,7,8]) # Kc, Kh, Kw,Kd
        if bias is not None:
            b = bias.broadcast(y.shape, [0,2,3,4])
            y = y + b
        return y
    else:
        N,C,D,H,W = x.shape
        Kd, Kh, Kw = weight.shape[-3:]
        G = groups
        CpG = C // G # channels per group
        oc = out_channels
        od = (D+padding[0]*2-Kd*dilation[0]+dilation[0]-1)//stride[0]+1
        oh = (H+padding[1]*2-Kh*dilation[1]+dilation[1]-1)//stride[1]+1
        ow = (W+padding[2]*2-Kw*dilation[2]+dilation[2]-1)//stride[2]+1
        xx = x.reindex([N,G,oc//G,CpG,od,oh,ow,Kd,Kh,Kw], [
                'i0', # Nid
                f'i1*{CpG}+i3', # Gid
                f'i4*{stride[0]}-{padding[0]}+i7*{dilation[0]}', # Hid+Khid
                f'i5*{stride[1]}-{padding[1]}+i8*{dilation[1]}', # Wid+KWid
                f'i6*{stride[2]}-{padding[2]}+i9*{dilation[2]}', # Did+KDid
            ])
        xx.compile_options = {"G":G}
        # w: [oc, CpG, Kh, Kw, Kd]
        ww = weight.reindex([N, G, oc//G, CpG, oh, ow, od, Kh, Kw, Kd], [
                f'i1*{oc//G}+i2',
                'i3',
                'i7',
                'i8',
                'i9'
            ])
        yy = xx*ww
        y = yy.reindex_reduce('add', [N, oc, od, oh, ow], [
                'i0',
                f'i1*{oc//G}+i2',
                'i4',
                'i5',
                'i6'
            ])

        if bias is not None:
            b = bias.broadcast(y.shape, [0,2,3,4])
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
        assert self.output_padding[0] < max(self.stride[0], self.dilation[0]) and \
            self.output_padding[1] < max(self.stride[1], self.dilation[1]), \
            "output padding must be smaller than max(stride, dilation)"

        self.weight = init.invariant_uniform((in_channels, out_channels) + self.kernel_size, dtype="float")
        if bias:
            fan=1
            for i in self.weight.shape[1:]:
                fan *= i
            bound = 1 / math.sqrt(fan)
            self.bias = init.uniform([out_channels], dtype="float", low=-bound, high=bound)
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

class ConvTranspose3d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, \
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        self.in_channels = in_channels
        self.out_channels = out_channels

        # added
        self.dilation = dilation
        self.group = groups
        assert groups==1, "Group conv not supported yet."

        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        # added
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.real_padding = (
            self.dilation[0] * (self.kernel_size[0] - 1) - self.padding[0],
            self.dilation[1] * (self.kernel_size[1] - 1) - self.padding[1],
            self.dilation[2] * (self.kernel_size[2] - 1) - self.padding[2])
        self.output_padding = output_padding if isinstance (output_padding, tuple) else (output_padding, output_padding, output_padding)
        assert self.output_padding[0] < max(self.stride[0], self.dilation[0]) and \
            self.output_padding[1] < max(self.stride[1], self.dilation[1]) and \
            self.output_padding[2] < max(self.stride[2], self.dilation[2]), \
            "output padding must be smaller than max(stride, dilation)"

        self.weight = init.invariant_uniform((in_channels, out_channels) + self.kernel_size, dtype="float")
        if bias:
            fan=1
            for i in self.weight.shape[1:]:
                fan *= i
            bound = 1 / math.sqrt(fan)
            self.bias = init.uniform([out_channels], dtype="float", low=-bound, high=bound)
        else:
            self.bias = None

    def execute(self, x):
        return conv_transpose3d(x, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.group, self.dilation)

def conv_transpose(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    x = input
    N,C,H,W = x.shape
    i,o,h,w = weight.shape
    assert C==i
    assert groups==1, "Group conv not supported yet."
    stride = stride if isinstance(stride, tuple) else (stride, stride)
    dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
    # added
    padding = padding if isinstance(padding, tuple) else (padding, padding)
    output_padding = output_padding if isinstance (output_padding, tuple) else (output_padding, output_padding)
    assert output_padding[0] < max(stride[0], dilation[0]) and \
        output_padding[1] < max(stride[1], dilation[1]), \
        "output padding must be smaller than max(stride, dilation)"

    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation

    h_out = (H-1) * stride_h + output_padding[0] - 2*padding_h + 1 + (h-1)*dilation_h
    w_out = (W-1) * stride_w + output_padding[1] - 2*padding_w + 1 + (w-1)*dilation_w
    out_shape = (N, o, h_out, w_out)
    shape = (N, i, o, H, W, h, w)
    xx = x.broadcast(shape, (2, 5, 6)) # i,h,w
    ww = weight.broadcast(shape, (0, 3, 4)) # N,H,W
    y = (ww*xx).reindex_reduce("add", out_shape, [
        'i0', # N
        'i2', # o
        f'i3*{stride_h}-{padding_h}+i5*{dilation_h}', # Hid+Khid
        f'i4*{stride_w}-{padding_w}+i6*{dilation_w}', # Wid+KWid
    ])
    if isinstance(bias, jt.Var):
        b = bias.broadcast(y.shape, [0,2,3])
        y = y + b
    else:
        assert not bias, "Bias should be none or jittor var"
    return y

def conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    x = input
    N,C,D,H,W = x.shape
    i,o,d,h,w = weight.shape
    assert C==i
    assert groups==1, "Group conv not supported yet."
    stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
    dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
    # added
    padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
    output_padding = output_padding if isinstance (output_padding, tuple) else (output_padding, output_padding, output_padding)
    assert output_padding[0] < max(stride[0], dilation[0]) and \
        output_padding[1] < max(stride[1], dilation[1]) and \
        output_padding[2] < max(stride[2], dilation[2]), \
        "output padding must be smaller than max(stride, dilation)"

    stride_d, stride_h, stride_w = stride
    padding_d, padding_h, padding_w = padding
    dilation_d, dilation_h, dilation_w = dilation

    d_out = (D-1) * stride_d + output_padding[0] - 2*padding_d + 1 + (d-1)*dilation_d
    h_out = (H-1) * stride_h + output_padding[1] - 2*padding_h + 1 + (h-1)*dilation_h
    w_out = (W-1) * stride_w + output_padding[2] - 2*padding_w + 1 + (w-1)*dilation_w
    out_shape = (N, o, d_out, h_out, w_out)
    if jt.flags.use_cuda and jt.cudnn:
        return jt.cudnn.ops.cudnn_conv3d_backward_x(weight, x, *out_shape[2:], *stride, *padding, *dilation, groups)
    shape = (N, i, o, D, H, W, d, h, w)
    xx = x.broadcast(shape, (2, 6, 7, 8)) # i,h,w
    ww = weight.broadcast(shape, (0, 3, 4, 5)) # N,H,W
    y = (ww*xx).reindex_reduce("add", out_shape, [
        'i0', # N
        'i2', # o
        f'i3*{stride_d}-{padding_d}+i6*{dilation_d}', # Did+Kdid
        f'i4*{stride_h}-{padding_h}+i7*{dilation_h}', # Hid+Khid
        f'i5*{stride_w}-{padding_w}+i8*{dilation_w}', # Wid+KWid
    ])
    if isinstance(bias, jt.Var):
        b = bias.broadcast(y.shape, [0,2,3,4])
        y = y + b
    else:
        assert not bias, "Bias should be none or jittor var"
    return y

conv_transpose2d = conv_transpose

def pad(x,padding, mode='constant', value=0):
    assert mode in ['constant','replicate','reflect','circular'],'only support constant,replicate,reflect,circular pad'
    assert len(padding)%2==0 and len(padding)//2<=x.ndim

    padding = list(padding)
    left = [0]*(x.ndim-len(padding)//2)+padding[::2][::-1]
    right = [0]*(x.ndim-len(padding)//2)+padding[1::2][::-1]

    out_dims = []
    out_shape = []
    for i,n,l,r in zip(range(x.ndim),x.shape,left,right):
        out_shape.append(n+l+r)
        if mode == 'constant':
            out_dims.append(f'i{i}-{l}')
        elif mode == 'replicate':
            out_dims.append(f"i{i}<{l} ? 0 : i{i} > {n+l-1} ? {n-1} : i{i}-{l}")
        elif mode == 'reflect':
            out_dims.append(f"i{i}<{l} ? {l}-i{i} : i{i} > {n+l-1} ? {2*(n-1)+l}-i{i} : i{i}-{l}")
        elif mode == 'circular':
            out_dims.append(f"i{i}<{l} ? {n-l}+i{i} : i{i} > {n+l-1} ? i{i}-{n+l} : i{i}-{l}")

    return x.reindex(out_shape,out_dims,overflow_value=value)


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
        elif isinstance(self.padding, (tuple,list)):
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

def softplus(x,beta=1.0,threshold=20.0):
    return 1 / beta * jt.log(1 + (beta * x).minimum(threshold).exp()) + \
        (x - threshold/beta).maximum(0.0)

def hardtanh(x,min_val=-1,max_val=1):
    return jt.clamp(x,min_v=min_val,max_v=max_val)


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
        return softplus(x, self.beta, self.threshold)

class Resize(Module):
    def __init__(self, size, mode="nearest", align_corners=False):
        super().__init__()
        self.size = size
        self.mode = mode
        self.align_corners = align_corners
    def execute(self, x):
        return resize(x, self.size, self.mode, self.align_corners)


def _bicubic(x, a, func):
    # normal ver
    if func == 1:
        return (a+2)*(jt.abs(x)**3)-(a+3)*(x**2)+1
    if func == 2:
        return a*(jt.abs(x)**3)-5*a*(x**2)+8*a*(jt.abs(x))-4*a
    return 0


def _interpolate(img, x, y, ids, mode):
    if mode == "nearest":
        return img.reindex([*ids, x.floor(), y.floor()])
    if mode == "bilinear":
        fx, fy = x.floor(), y.floor()
        cx, cy = fx + 1, fy + 1
        dx, dy = x - fx, y - fy
        a = img.reindex_var([*ids, fx, fy])
        b = img.reindex_var([*ids, cx, fy])
        c = img.reindex_var([*ids, fx, cy])
        d = img.reindex_var([*ids, cx, cy])
        dnx, dny = 1 - dx, 1 - dy
        ab = dx * b + dnx * a
        cd = dx * d + dnx * c
        o = ab * dny + cd * dy
        return o
    if mode=="bicubic": # ugly ver.
        n,c,h,w = img.shape
        fx, fy = x.floor(), y.floor()
        dix, diy = x - fx, y - fy
        ax, ay = _bicubic(dix+1,-0.75,2), _bicubic(diy+1,-0.75,2)
        bx, by = _bicubic(dix,-0.75,1), _bicubic(diy,-0.75,1)
        cx, cy = _bicubic(1-dix,-0.75,1), _bicubic(1-diy,-0.75,1)
        dx, dy = _bicubic(2-dix,-0.75,2), _bicubic(2-diy,-0.75,2)
        afx, afy = jt.maximum(jt.minimum(fx-1,h-1),0), jt.maximum(jt.minimum(fy-1,w-1),0)
        bfx, bfy = jt.maximum(jt.minimum(fx,h-1),0), jt.maximum(jt.minimum(fy,w-1),0)
        cfx, cfy = jt.maximum(jt.minimum(fx+1,h-1),0), jt.maximum(jt.minimum(fy+1,w-1),0)
        dfx, dfy = jt.maximum(jt.minimum(fx+2,h-1),0), jt.maximum(jt.minimum(fy+2,w-1),0)
        a = ax*(img.reindex_var([*ids,afx,afy])*ay+img.reindex_var([*ids,afx,bfy])*by+img.reindex_var([*ids,afx,cfy])*cy+img.reindex_var([*ids,afx,dfy])*dy)
        b = bx*(img.reindex_var([*ids,bfx,afy])*ay+img.reindex_var([*ids,bfx,bfy])*by+img.reindex_var([*ids,bfx,cfy])*cy+img.reindex_var([*ids,bfx,dfy])*dy)
        c = cx*(img.reindex_var([*ids,cfx,afy])*ay+img.reindex_var([*ids,cfx,bfy])*by+img.reindex_var([*ids,cfx,cfy])*cy+img.reindex_var([*ids,cfx,dfy])*dy)
        d = dx*(img.reindex_var([*ids,dfx,afy])*ay+img.reindex_var([*ids,dfx,bfy])*by+img.reindex_var([*ids,dfx,cfy])*cy+img.reindex_var([*ids,dfx,dfy])*dy)
        o = a + b + c + d
        return o
    raise (f"Not support interpolation mode: {mode}")


def resize(img, size, mode="nearest", align_corners=False):
    n, c, h, w = img.shape
    H, W = size
    nid, cid, hid, wid = jt.index((n, c, H, W))
    if align_corners:
        x = hid * ((h - 1) / max(1, H - 1))
        y = wid * ((w - 1) / max(1, W - 1))
    else:
        x = hid * (h / H) + (h / H * 0.5 - 0.5)
        if H > h: x = x.clamp(0, h - 1)
        y = wid * (w / W) + (w / W * 0.5 - 0.5)
        if W > w: y = y.clamp(0, w - 1)
    return _interpolate(img, x, y, (nid, cid), mode)


def upsample(img, size, mode="nearest", align_corners=False):
    n, c, h, w = img.shape
    H, W = size
    nid, cid, hid, wid = jt.index((n, c, H, W))
    if align_corners:
        x = hid * ((h - 1) / max(1, H - 1))
        y = wid * ((w - 1) / max(1, W - 1))
    elif mode == "bicubic":
        x = (hid + 0.5) * (h / H) - 0.5
        y = (wid + 0.5) * (w / W) - 0.5
    elif mode == 'nearest':
        x = hid * (h / H)
        y = wid * (w / W)
    else:
        x = hid * (h / H) + (h / H * 0.5 - 0.5)
        if H > h: x = x.clamp(0, h - 1)
        y = wid * (w / W) + (w / W * 0.5 - 0.5)
        if W > w: y = y.clamp(0, w - 1)
    return _interpolate(img, x, y, (nid, cid), mode)


def interpolate(X, size=None, scale_factor=None, mode='bilinear', align_corners=False):
    if scale_factor is not None:
        size = [X.shape[-2] * scale_factor, X.shape[-1] * scale_factor]
    if isinstance(size, int):
        size = (size, size)
    if scale_factor is not None and scale_factor > 1:
        return upsample(X, size, mode, align_corners)
    else:
        return resize(X, size, mode, align_corners)


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

    nid, cid, hid, wid = jt.index((Ni, Ci, Ho, Wo))
    x = ((grid[:, :, :, 1].unsqueeze(1).repeat([1, Ci, 1, 1]) + 1) / 2) * (Hi - 1)
    y = ((grid[:, :, :, 0].unsqueeze(1).repeat([1, Ci, 1, 1]) + 1) / 2) * (Wi - 1)
    return _interpolate(input, x, y, (nid, cid), mode)


def linspace_from_neg_one(grid,num_steps,align_corners):
    if  num_steps <= 1:
        return jt.array([],dtype=grid.dtype)
    # TODO: use jt.index
    ra = np.linspace(-1,1,num_steps)
    if not align_corners:
        ra = ra*(num_steps-1)/num_steps
    return jt.array(ra,dtype=grid.dtype)

def make_base_grid_4D(theta,N,C,H,W,align_corners):
    base_grid = jt.zeros((N, H, W, 3), dtype=theta.dtype)
    base_grid[...,0] = linspace_from_neg_one(theta, W, align_corners)
    base_grid[...,1] = jt.unsqueeze(linspace_from_neg_one(theta, H, align_corners),-1)
    base_grid[...,-1] = 1
    return base_grid

def make_base_grid_5D(theta,N,C,D,H,W,align_corners):
    base_grid = jt.zeros((N, D, H, W, 4), dtype=theta.dtype)
    base_grid[...,0] = linspace_from_neg_one(theta, W, align_corners)
    base_grid[...,1] = jt.unsqueeze(linspace_from_neg_one(theta, H, align_corners),-1)
    base_grid[...,2] = jt.unsqueeze(jt.unsqueeze(linspace_from_neg_one(theta, D, align_corners),-1),-1)
    base_grid[...,-1] = 1
    return base_grid

def affine_grid_generator_4D(theta,N,C,H,W,align_corners):
     base_grid = make_base_grid_4D(theta, N, C, H, W, align_corners)
     grid = jt.nn.bmm(base_grid.reshape(N, H * W, 3),theta.transpose(0,2,1))
     return grid.reshape(N, H, W, 2)

def affine_grid_generator_5D(theta,N,C,D,H,W,align_corners):
    base_grid = make_base_grid_5D(theta, N, C, D, H, W, align_corners)
    grid = jt.nn.bmm(base_grid.reshape(N, D * H * W, 4),theta.transpose(0,2,1))
    return grid.reshape(N, D, H, W, 3)

def affine_grid(theta, size, align_corners=False):
    assert str(theta.dtype) in ['float','float32','float64']
    assert min(size)>0
    assert len(size) in [4,5]    
    if len(size)== 4:
        assert theta.ndim == 3 and theta.shape[-2] == 2 and theta.shape[-1] == 3
        return affine_grid_generator_4D(theta, size[0], size[1], size[2], size[3], align_corners)
    elif len(size)==5:
        assert theta.ndim == 3 and theta.shape[-2] == 3 and theta.shape[-1] == 4
        return affine_grid_generator_5D(theta, size[0], size[1], size[2], size[3], size[4], align_corners)


def grid_sampler_unnormalize(coord,size,align_corners):
    if align_corners:
        #unnormalize coord from [-1, 1] to [0, size - 1]
        return ((coord + 1) / 2) * (size - 1)
    else:
        #unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
        return ((coord + 1) * size - 1) / 2


def clip_coordinates(x,clip_limit):
    return jt.clamp(x,min_v=0,max_v=clip_limit-1)

def reflect_coordinates(x,twice_low,twice_high):
    if twice_low == twice_high:
        return jt.zeros_like(x)
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
    xid = x.reindex(shape,['i0','i2','i3'])
    yid = y.reindex(shape,['i0','i2','i3'])

    if mode=='nearest':
        return X.reindex([nid,cid,yid.round(),xid.round()])
    elif mode=='bilinear':
        #xid,yid = (xid+0.00001),(yid+0.00001)
        fx,fy = (xid).floor(),(yid).floor()
        cx,cy = fx+1,fy+1
        dx,dy = xid-fx,yid-fy
        dnx,dny = cx-xid,cy-yid

        a = X.reindex([nid,cid,fy,fx],overflow_value=0.0)
        b = X.reindex([nid,cid,cy,fx],overflow_value=0.0)
        c = X.reindex([nid,cid,fy,cx],overflow_value=0.0)
        d = X.reindex([nid,cid,cy,cx],overflow_value=0.0)
        o = a*dnx*dny+b*dnx*dy+c*dx*dny+d*dx*dy
        return o


def grid_sampler(X, grid, mode, padding_mode, align_corners):
    assert X.dtype==grid.dtype
    assert ((X.ndim==4 or X.ndim==5) and X.ndim==grid.ndim)
    assert X.shape[0]==grid.shape[0] and grid.shape[-1]==X.ndim-2
    assert X.numel()>0
    if X.ndim == 4:
        return grid_sampler_2d(X, grid, mode, padding_mode, align_corners)
    else:
        return grid_sampler_3d(X, grid, mode, padding_mode, align_corners)


def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    assert mode in ['bilinear','nearest']
    assert padding_mode in ['zeros','border','reflection']
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

class UpsamplingBilinear2d(Upsample):
    def __init__(self, scale_factor=None):
        Upsample.__init__(self, scale_factor, 'bilinear')

class UpsamplingNearest2d(Upsample):
    def __init__(self, scale_factor=None):
        Upsample.__init__(self, scale_factor, 'nearest')

class Sequential(Module):
    def __init__(self, *args):
        self.layers = collections.OrderedDict()
        for mod in args:
            if isinstance(mod, collections.OrderedDict):
                for k, m in mod.items():
                    self.add_module(k, m)
            elif isinstance(mod,list):
                for m in mod:
                    self.append(m)
            else:
                self.append(mod)
    def __getitem__(self, idx):
        if idx not in self.layers:
            return list(self.layers.values())[idx]

        return self.layers[idx]
    def __iter__(self):
        return self.layers.values().__iter__()
    def keys(self):
        return self.layers.keys()
    def values(self):
        return self.layers.values()
    def items(self):
        return self.layers.items()
    def execute(self, x):
        for k, layer in self.layers.items():
            x = layer(x)
        return x
    def dfs(self, parents, k, callback, callback_leave):
        n_children = len(self.layers)
        ret = callback(parents, k, self, n_children)
        if ret == False:
            return
        parents.append(self)
        for k,v in self.layers.items():
            v.dfs(parents, k, callback, callback_leave)
        parents.pop()
        if callback_leave:
            callback_leave(parents, k, self, n_children)
    def append(self, mod):
        assert callable(mod), f"Module <{type(mod)}> is not callable"
        assert not isinstance(mod, type), f"Module is not a type"
        self.layers[len(self.layers)]=mod
    def add_module(self, name, mod):
        assert callable(mod), f"Module <{type(mod)}> is not callable"
        assert not isinstance(mod, type), f"Module is not a type"
        self.layers[name]=mod

    def __len__(self):
        return len(self.layers)


def unfold(X, kernel_size, dilation=1, padding=0, stride=1):
    assert X.ndim == 4
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    if not isinstance(dilation, tuple):
        dilation = (dilation, dilation)
    if not isinstance(padding, tuple):
        padding = (padding, padding)
    if not isinstance(stride, tuple):
        stride = (stride, stride)
    n, c, h, w = X.shape
    shape = X.shape
    area = kernel_size[0] * kernel_size[1]
    block_nums = []
    for i in range(2, 4):
        block_nums.append(
            (shape[i] + 2 * padding[i - 2] - dilation[i - 2] * (kernel_size[i - 2] - 1) - 1) // stride[i - 2] + 1)
    if padding[0] != 0 or padding[1] != 0:
        X = X.reindex([n, c, h + padding[0] * 2, w + padding[1] * 2],
                      ["i0", "i1", f"i2-{padding[0]}", f"i3-{padding[1]}"])
    output = X.reindex([n, c * area, block_nums[0] * block_nums[1]], ["i0", f"i1/{area}",
                                                                      f"i2/{block_nums[1]}*{stride[0]}+(i1%{area})/{kernel_size[1]}*{dilation[0]}",
                                                                      f"i2%{block_nums[1]}*{stride[1]}+(i1%{area})%{kernel_size[1]}*{dilation[1]}"])
    return output


def fold(X,output_size,kernel_size,dilation=1,padding=0,stride=1):
    assert X.ndim==3
    if not isinstance(kernel_size,tuple):
        kernel_size = (kernel_size,kernel_size)
    if not isinstance(dilation,tuple):
        dilation = (dilation,dilation)
    if not isinstance(padding,tuple):
        padding = (padding,padding)
    if not isinstance(stride,tuple):
        stride = (stride,stride)
    n,cl,num = X.shape
    area = kernel_size[0] * kernel_size[1]
    block_nums = []
    for i in range(2,4):
        block_nums.append((output_size[i-2]+2*padding[i-2]-dilation[i-2]*(kernel_size[i-2]-1)-1) // stride[i-2]+1)
    output = X.reindex_reduce("add",[n,cl // area,output_size[0]+2*padding[0],output_size[1]+2*padding[1]],["i0",f"i1/{area}",f"i2/{block_nums[1]}*{stride[0]}+(i1%{area})/{kernel_size[1]}*{dilation[0]}",f"i2%{block_nums[1]}*{stride[1]}+(i1%{area})%{kernel_size[1]}*{dilation[1]}"])
    return output[:,:,padding[0]:padding[0]+output_size[0],padding[1]:padding[1]+output_size[1]]

ModuleList = Sequential


class LSTMCell(jt.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        ''' A long short-term memory (LSTM) cell.

        :param input_size: The number of expected features in the input
        :type input_size: int

        :param hidden_size: The number of features in the hidden state
        :type hidden_size: int

        :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True.
        :type bias: bool, optional

        Example:

        >>> rnn = nn.LSTMCell(10, 20) # (input_size, hidden_size)
        >>> input = jt.randn(2, 3, 10) # (time_steps, batch, input_size)
        >>> hx = jt.randn(3, 20) # (batch, hidden_size)
        >>> cx = jt.randn(3, 20)
        >>> output = []
        >>> for i in range(input.shape[0]):
                hx, cx = rnn(input[i], (hx, cx))
                output.append(hx)
        >>> output = jt.stack(output, dim=0)
        '''
        super().__init__()

        self.hidden_size = hidden_size
        self.bias = bias

        k = math.sqrt(1 / hidden_size)
        self.weight_ih = init.uniform((4 * hidden_size, input_size), 'float32', -k, k)
        self.weight_hh = init.uniform((4 * hidden_size, hidden_size), 'float32', -k, k)

        if bias:
            self.bias_ih = init.uniform((4 * hidden_size,), 'float32', -k, k)
            self.bias_hh = init.uniform((4 * hidden_size,), 'float32', -k, k)

    def execute(self, input, hx = None):
        if hx is None:
            zeros = jt.zeros((input.shape[0], self.hidden_size), dtype=input.dtype)
            h, c = zeros, zeros
        else:
            h, c = hx

        y = matmul_transpose(input, self.weight_ih) + matmul_transpose(h, self.weight_hh)

        if self.bias:
            y = y + self.bias_ih + self.bias_hh
        
        i = y[:, :self.hidden_size].sigmoid()
        f = y[:, self.hidden_size : 2 * self.hidden_size].sigmoid()
        g = y[:, 2 * self.hidden_size : 3 * self.hidden_size].tanh()
        o = y[:, 3 * self.hidden_size:].sigmoid()

        c = f * c + i * g
        h = o * c.tanh()

        return h, c


class RNNCell(jt.Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity = "tanh"):
        ''' An Elman RNN cell with tanh or ReLU non-linearity.

        :param input_size: The number of expected features in the input
        :type input_size: int

        :param hidden_size: The number of features in the hidden state
        :type hidden_size: int

        :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True.
        :type bias: bool, optional

        :param nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'.
        :type nonlinearity: str, optional

        Example:

        >>> rnn = nn.RNNCell(10, 20)
        >>> input = jt.randn((6, 3, 10))
        >>> hx = jt.randn((3, 20))
        >>> output = []
        >>> for i in range(6):
                hx = rnn(input[i], hx)
                output.append(hx)
        '''
        super().__init__()

        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity

        k = math.sqrt(1 / hidden_size)
        self.weight_ih = init.uniform((hidden_size, input_size), 'float32', -k, k)
        self.weight_hh = init.uniform((hidden_size, hidden_size), 'float32', -k, k)

        if bias:
            self.bias_ih = init.uniform((hidden_size,), 'float32', -k, k)
            self.bias_hh = init.uniform((hidden_size,), 'float32', -k, k)

    def execute(self, input, hx = None):
        if hx is None:
            hx = jt.zeros((input.shape[0], self.hidden_size), dtype=input.dtype)

        y = matmul_transpose(input, self.weight_ih)+matmul_transpose(hx, self.weight_hh)

        if self.bias:
            y= y + self.bias_ih + self.bias_hh

        if self.nonlinearity == 'tanh':
            y = y.tanh()
        elif self.nonlinearity == 'relu':
            y = relu(y) 
        else:
            raise RuntimeError("Unknown nonlinearity: {}".format(self.nonlinearity))

        return y

class GRUCell(jt.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        ''' A gated recurrent unit (GRU) cell.

        :param input_size: The number of expected features in the input
        :type input_size: int

        :param hidden_size: The number of features in the hidden state
        :type hidden_size: int

        :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True.
        :type bias: bool, optional

        Example:

        >>> rnn = nn.GRUCell(10, 20)
        >>> input = jt.randn((6, 3, 10))
        >>> hx = jt.randn((3, 20))
        >>> output = []
        >>> for i in range(6):
                hx = rnn(input[i], hx)
                output.append(hx)
        '''
        super().__init__()

        self.hidden_size = hidden_size
        self.bias = bias

        k = math.sqrt(1 / hidden_size)
        self.weight_ih = init.uniform((3*hidden_size, input_size), 'float32', -k, k)
        self.weight_hh = init.uniform((3*hidden_size, hidden_size), 'float32', -k, k)

        if bias:
            self.bias_ih = init.uniform((3*hidden_size,), 'float32', -k, k)
            self.bias_hh = init.uniform((3*hidden_size,), 'float32', -k, k)

    def execute(self, input, hx = None):
        if hx is None:
            hx = jt.zeros((input.shape[0], self.hidden_size), dtype=input.dtype)

        gi = matmul_transpose(input, self.weight_ih)
        gh = matmul_transpose(hx, self.weight_hh)

        if self.bias:
            gi += self.bias_ih
            gh += self.bias_hh
            
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)
        
        resetgate = jt.sigmoid(i_r + h_r)
        inputgate = jt.sigmoid(i_i + h_i)
        newgate = jt.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hx - newgate)
        return hy

class RNNBase(Module):
    def __init__(self, mode: str, input_size: int, hidden_size: int, 
            num_layers: int = 1, bias: bool = True, batch_first: bool = False, 
            dropout: float = 0, bidirectional: bool = False, 
            proj_size: int = 0) -> None:
        super().__init__()

        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.proj_size = proj_size

        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        elif mode == 'GRU':
            gate_size = 3 * hidden_size
        elif mode == 'RNN':
            gate_size = hidden_size
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)

        num_directions = 1 + bidirectional
        k = math.sqrt(1 / hidden_size)

        def build_unit(name, in_channels, out_channels=None):
            if out_channels is not None:
                shape = (in_channels, out_channels)
            else:
                shape = (in_channels,)
            setattr(self, name, init.uniform(shape, 'float32', -k, k))
            if self.bidirectional:
                setattr(self, name + '_reverse', init.uniform(shape, 'float32', -k, k))

        for layer in range(num_layers):
            if layer == 0:
                build_unit(f'weight_ih_l{layer}', gate_size, input_size)
            else:
                if proj_size > 0:
                    build_unit(f'weight_ih_l{layer}', gate_size, num_directions * proj_size)
                else:
                    build_unit(f'weight_ih_l{layer}', gate_size, num_directions * hidden_size)

            if proj_size > 0:
                build_unit(f'weight_hh_l{layer}', gate_size, proj_size)
                build_unit(f'weight_hr_l{layer}', proj_size, hidden_size)
            else:
                build_unit(f'weight_hh_l{layer}', gate_size, hidden_size)

            if bias:
                build_unit(f'bias_ih_l{layer}', gate_size)
                build_unit(f'bias_hh_l{layer}', gate_size)

    @abstractmethod
    def call_rnn_cell(self, input, hidden, suffix):
        pass

    def call_rnn_sequence(self, input, hidden, suffix):
        if 'reverse' in suffix:
            input = input[::-1]

        output = []
        for s in range(input.shape[0]):
            out, hidden = self.call_rnn_cell(input[s], hidden, suffix)
            output.append(out)

        if 'reverse' in suffix:
            output = output[::-1]
        output = jt.stack(output, dim=0)

        return output, hidden

    def execute(self, input, hx):
        if self.batch_first:
            input = input.permute(1, 0, 2)

        num_directions = 2 if self.bidirectional else 1

        if hx is None:
            hx = self.default_init_state()

        hidden_n = []

        for l in range(self.num_layers):
            output = []

            if isinstance(hx, tuple):
                hidden = [h[l * num_directions] for h in hx]
            else:
                hidden = hx[l * num_directions]

            output, _hidden = self.call_rnn_sequence(input, hidden, f'l{l}')
            hidden_n.append(_hidden)

            if self.bidirectional:
                if isinstance(hx, tuple):
                    hidden = [h[l * num_directions + 1] for h in hx]
                else:
                    hidden = hx[l * num_directions + 1]

                output_b, _hidden = self.call_rnn_sequence(input, hidden, f'l{l}_reverse')
                output = jt.concat([output, output_b], dim=-1)
                hidden_n.append(_hidden)

            if self.dropout > 0:
                input = dropout(output, p=self.dropout)
            else:
                input = output

        if isinstance(hx, tuple):
            hidden_n = tuple(jt.stack(hn, dim=0) for hn in zip(*hidden_n))
        else:
            hidden_n = jt.stack(hidden_n, dim=0)

        return output, hidden_n


class RNN(RNNBase):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
        nonlinearity: str = 'tanh', bias: bool = True, batch_first: bool = False, 
        dropout: float = 0, bidirectional: bool = False) -> None:
        ''' Applies a multi-layer Elman RNN with tanh ReLU non-linearity to an input sequence.

        :param input_size: The number of expected features in the input.
        :type input_size: int

        :param hidden_size: The number of features in the hidden state.
        :type hidden_size: int

        :param num_layers: Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together to form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results. Default: 1
        :type num_layers: int, optinal

        :param nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'
        :type nonlinearity: str, optional

        :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True.
        :type bias: bool, optional

        :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
        :type bias: bool, optional

        :param dropout: If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0
        :type dropout: float, optional

        :param bidirectional: If True, becomes a bidirectional RNN. Default: False
        :type bidirectional: bool, optional

        Example:
            >>> rnn = nn.RNN(10, 20, 2)
            >>> input = jt.randn(5, 3, 10)
            >>> h0 = jt.randn(2, 3, 20)
            >>> output, hn = rnn(input, h0)
        '''
        super().__init__('RNN', input_size, hidden_size, num_layers=num_layers, 
            bias=bias, batch_first=batch_first, dropout=dropout, 
            bidirectional=bidirectional)

        if not nonlinearity in ['tanh', 'relu']:
            raise ValueError('Unrecognized nonlinearity: ' + nonlinearity)
        self.nonlinearity = nonlinearity

    def call_rnn_cell(self, input, hidden, suffix):
        y = matmul_transpose(input, getattr(self, f'weight_ih_{suffix}')) 
        y = y + matmul_transpose(hidden, getattr(self, f'weight_hh_{suffix}'))
        
        if self.bias:
            y = y + getattr(self, f'bias_ih_{suffix}') + getattr(self, f'bias_hh_{suffix}')

        if self.nonlinearity == 'tanh':
            h = jt.tanh(y)
        else:
            h = jt.relu(y)

        return h, h


class LSTM(RNNBase):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, 
            batch_first=False, dropout=0, bidirectional=False, proj_size=0):
        ''' Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        :param input_size: The number of expected features in the input.
        :type input_size: int

        :param hidden_size: The number of features in the hidden state.
        :type hidden_size: int

        :param num_layers: Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1
        :type num_layers: int, optinal

        :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True.
        :type bias: bool, optional

        :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
        :type bias: bool, optional

        :param dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. Default: 0
        :type dropout: float, optional

        :param bidirectional: If True, becomes a bidirectional LSTM. Default: False
        :type bidirectional: bool, optional

        :param proj_size: If > 0, will use LSTM with projections of corresponding size. Default: 0
        :type proj_size: int, optional

        Example:
            >>> rnn = nn.LSTM(10, 20, 2)
            >>> input = jt.randn(5, 3, 10)
            >>> h0 = jt.randn(2, 3, 20)
            >>> c0 = jt.randn(2, 3, 20)
            >>> output, (hn, cn) = rnn(input, (h0, c0))
        '''
        super().__init__('LSTM', input_size, hidden_size, num_layers=num_layers, 
            bias=bias, batch_first=batch_first, dropout=dropout, 
            bidirectional=bidirectional, proj_size=proj_size)

    def call_rnn_cell(self, input, hidden, suffix):
        h, c = hidden
        y = matmul_transpose(input, getattr(self, f'weight_ih_{suffix}')) 
        y = y + matmul_transpose(h, getattr(self, f'weight_hh_{suffix}'))
        
        if self.bias:
            y = y + getattr(self, f'bias_ih_{suffix}') + getattr(self, f'bias_hh_{suffix}')

        i = y[:, :self.hidden_size].sigmoid()
        f = y[:, self.hidden_size : 2 * self.hidden_size].sigmoid()
        g = y[:, 2 * self.hidden_size : 3 * self.hidden_size].tanh()
        o = y[:, 3 * self.hidden_size:].sigmoid()
        c = f * c + i * g
        h = o * c.tanh()

        if self.proj_size > 0:
            h = matmul_transpose(h, getattr(self, f'weight_hr_{suffix}'))

        return h, (h, c)


class GRU(RNNBase):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
        bias: bool = True, batch_first: bool = False, dropout: float = 0, 
        bidirectional: bool = False) -> None:
        ''' Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.

        :param input_size: The number of expected features in the input.
        :type input_size: int

        :param hidden_size: The number of features in the hidden state.
        :type hidden_size: int

        :param num_layers: Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two GRUs together to form a stacked GRU, with the second GRU taking in outputs of the first GRU and computing the final results. Default: 1
        :type num_layers: int, optinal

        :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True.
        :type bias: bool, optional

        :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
        :type bias: bool, optional

        :param dropout: If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer, with dropout probability equal to dropout. Default: 0
        :type dropout: float, optional

        :param bidirectional: If True, becomes a bidirectional GRU. Default: False
        :type bidirectional: bool, optional

        Example:
            >>> rnn = nn.GRU(10, 20, 2)
            >>> input = jt.randn(5, 3, 10)
            >>> h0 = jt.randn(2, 3, 20)
            >>> output, hn = rnn(input, h0)
        '''
        super().__init__('GRU', input_size, hidden_size, num_layers=num_layers, 
            bias=bias, batch_first=batch_first, dropout=dropout, 
            bidirectional=bidirectional)

    def call_rnn_cell(self, input, hidden, suffix):
        ih = matmul_transpose(input, getattr(self, f'weight_ih_{suffix}')) 
        hh = matmul_transpose(hidden, getattr(self, f'weight_hh_{suffix}'))
        
        if self.bias:
            ih = ih + getattr(self, f'bias_ih_{suffix}')
            hh = hh + getattr(self, f'bias_hh_{suffix}')

        hs = self.hidden_size
        r = (ih[:, :hs] + hh[:, :hs]).sigmoid()
        z = (ih[:, hs: 2 * hs] + hh[:, hs: 2 * hs]).sigmoid()
        n = (ih[:, 2 * hs:] + r * hh[:, 2 * hs:]).tanh()
        h = (1 - z) * n + z * hidden

        return h, h

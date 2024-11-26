# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
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
from jittor import flatten, init, Module
import numpy as np
import collections
import math
from collections import OrderedDict
from jittor.pool import *
from jittor.optim import *
from jittor.misc import _pair, _triple
from jittor_utils import LOG
from functools import partial


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
    with jt.flag_scope(amp_reg = jt.flags.amp_reg | 36):
        a = a.broadcast(shape, [len(shape)-2])
        b = b.broadcast(shape)
        return (a*b).sum(len(shape)-1)


def bmm_transpose(a, b):
    '''
    returns a * b^T
    '''
    if jt.flags.use_cuda and jt.compile_extern.cublas_ops:
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

def baddbmm(input, batch1, batch2, beta=1, alpha=1):
    res = bmm(batch1, batch2)
    if alpha != 1: res = res * alpha
    if beta == 0: return res
    return beta * input + res

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
    with jt.flag_scope(amp_reg = jt.flags.amp_reg | 36):
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
            if jt.flags.use_cuda and jt.compile_extern.cublas_ops:
                return jt.compile_extern.cublas_ops.cublas_batched_matmul(a, b, 0, 0)
        shape = []
        len_c = max(len_a, len_b)
        (n, m), (m_, k) = a.shape[-2:], b.shape[-2:]
        assert m == m_, f"dimension not match, a.shape:{a.shape}, b.shape:{b.shape}"
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
                assert an == bn, f"dimension not match, a.shape:{a.shape}, b.shape:{b.shape}"
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

def relu(x): 
    r''' Applies the element-wise function:

    .. math::
        \text{ReLU6}(x) = \max(0,x)

    :param x: the input var
    :type x: jt.Var

    Example:
        >>> a = jt.randn(3)
        >>> a
        jt.Var([-0.38380373 1.1338731   6.128115  ], dtype=float32)
        >>> nn.relu(a)
        jt.Var([0.        1.1338731 6.128115 ], dtype=float32)
    '''
    cond = x>0.0
    return jt.ternary_out_hint(cond, x, 0.0)


def leaky_relu(x, scale=0.01): 
    r''' Applies the element-wise function:

    .. math::
        \text{LeakyRELU}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \text{scale} \times x, & \text{ otherwise }
        \end{cases}

    :param x: the input var
    :type x: jt.Var

    :param scale: the :math:`\scale` value for the leaky relu formulation. Default: 0.01
    :param scale: float, optional

    Example:
        >>> a = jt.randn(3)
        >>> a
        jt.Var([-0.38380373 1.1338731   6.128115  ], dtype=float32)
        >>> nn.leaky_relu(a)
        jt.Var([-3.8380371e-03  1.1338731e+00  6.1281152e+00], dtype=float32)
    '''
    return jt.ternary(x>0, x, x*scale)

def relu6(x): 
    r''' Applies the element-wise function:

    .. math::
        \text{ReLU6}(x) = \min(\max(0,x), 6)

    :param x: the input var
    :type x: jt.Var

    Example:
        >>> a = jt.randn(3)
        >>> a
        jt.Var([-0.38380373 1.1338731   6.128115  ], dtype=float32)
        >>> nn.relu6(a)
        jt.Var([0.        1.1338731 6.       ], dtype=float32)
    '''
    return jt.minimum(jt.maximum(x, 0.0), 6.0)

def elu(x: jt.Var, alpha: float = 1.0) -> jt.Var:
    r''' Applies the element-wise function:

    .. math::
        \text{ELU}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
        \end{cases}

    :param x: the input var
    :type x: jt.Var

    :param alpha: the :math:`\alpha` value for the ELU formulation. Default: 1.0
    :param alpha: float, optional

    Example:
        >>> a = jt.randn(3)
        >>> a
        jt.Var([-0.38380373 -1.1338731   2.128115  ], dtype=float32)
        >>> nn.elu(a)
        jt.Var([-0.31873488 -0.6782155   2.128115  ], dtype=float32)
    '''
    return jt.ternary(x>0,x,alpha*(x.exp()-1))

def sign(x: jt.Var) -> jt.Var:
    ''' returns the signs of elements of x

    :param x: the input Var
    :type x: jt.Var

    Example:
        >>> a = jt.float32([0.99, 0, -0.99])
        >>> nn.sign(a)
        jt.Var([ 1.  0. -1.], dtype=float32)
    '''
    one = jt.ones(x.shape)
    x = jt.ternary(x>0, one, x)
    return jt.ternary(x<0, -one, x)

def gelu(x):
    r''' Applies the element-wise function:

    .. math::
        \text{GELU}(x) = x * \Phi(x)

    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

    :param x: the input var
    :type x: jt.Var

    Example:
        >>> a = jt.randn(3)
        >>> a
        jt.Var([-0.38380373 -1.1338731   2.128115  ], dtype=float32)
        >>> nn.gelu(a)
        jt.Var([-0.134547   0.9882567  6.128115 ], dtype=float32)
    '''
    _sqrt2 = 1.4142135623730951
    erf = jt.erf(x/_sqrt2)+1
    r = erf*x*.5
    return r

def silu(x):
    r''' Applies the element-wise function:

    .. math::
        \text{SILU}(x) = x * Sigmoid(x)
    
    :param x: the input var
    :type x: jt.Var

    Example:
        >>> a = jt.randn(3)
        >>> a
        jt.Var([-0.38380373 -1.1338731   2.128115  ], dtype=float32)
        >>> nn.silu(a)
        jt.Var([-0.15552104 -0.27603802  1.9016962 ], dtype=float32)
    '''
    return x * x.sigmoid()

class ELU(Module):
    r''' Applies the element-wise function:

    .. math::
        \text{ELU}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
        \end{cases}

    :param x: the input var
    :type x: jt.Var

    :param alpha: the :math:`\alpha` value for the ELU formulation. Default: 1.0
    :param alpha: float, optional

    Example:
        >>> a = jt.randn(3)
        >>> a
        jt.Var([-0.38380373 -1.1338731   2.128115  ], dtype=float32)
        >>> nn.elu(a)
        jt.Var([-0.31873488 -0.6782155   2.128115  ], dtype=float32)
    '''
    def __init__(self,alpha=1.0):
        self.alpha=alpha
    
    def execute(self,x):
        return elu(x,self.alpha)

class PReLU(Module):
    r''' Applies the element-wise function:

    .. math::
        \text{PReLU}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        ax, & \text{ otherwise }
        \end{cases}

    :param x: the input var
    :type x: jt.Var

    :param num_parameters: number of :math:`a` to learn, can be either 1 or the number of channels at input. Default: 1
    :type num_parameters: int, optional

    :param init: the initial value of :math:`a`. Default: 0.25
    :param init: float, optional

    Example:
        >>> a = jt.randn(3)
        >>> prelu = nn.PReLU()
        >>> prelu(a)
        jt.Var([-0.09595093  1.1338731   6.128115  ], dtype=float32)
    '''

    def __init__(self, num_parameters=1, init_=0.25):
        self.num_parameters = num_parameters
        self.weight = init.constant((num_parameters,), "float32", init_)

    def execute(self, x):
        if self.num_parameters != 1:
            assert self.num_parameters == x.size(1), f"num_parameters does not match input channels in PReLU"
            return jt.maximum(0, x) + self.weight.broadcast(x, [0,2,3]) * jt.minimum(0, x)
        else:
            return jt.maximum(0, x) + self.weight * jt.minimum(0, x)

#TODO dims is 4 will cause slowly execution
def cross_entropy_loss(output, target, weight=None, ignore_index=None,reduction='mean'):
    target_shape = target.shape
    if len(output.shape) == 4:
        c_dim = output.shape[1]
        output = output.transpose((0, 2, 3, 1))
        output = output.reshape((-1, c_dim))

    target = target.reshape((-1, ))
    target_weight = ((target >= 0) & (target < output.shape[1])).float32() 
    if weight is not None:
        target_weight = weight[target]
    if ignore_index is not None:
        target_weight = jt.ternary(
            target==ignore_index,
            jt.array(0).broadcast(target_weight).type_as(target_weight),
            target_weight
        )
    
    target = target.broadcast(output, [1])
    target = target.index(1) == target
    
    output = output - output.max([1], keepdims=True)
    logsum = output.exp().sum(1).log()
    loss = (logsum - (output*target).sum(1)) * target_weight
    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean() / target_weight.mean()
    else:
        return loss.reshape(target_shape) 

def mse_loss(output, target, reduction="mean"):
    return (output-target).sqr().reduce(reduction)

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
    def __init__(self, weight=None, ignore_index=None):
        self.weight = weight
        self.ignore_index = ignore_index
        
    def execute(self, output, target):
        return cross_entropy_loss(output, target, self.weight, self.ignore_index)

class MSELoss(Module):
    def __init__(self, reduction='mean'):
        self.reduction = reduction
    def execute(self, output, target):
        return mse_loss(output, target, self.reduction)

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
    if not (target.shape == output.shape):
        raise ValueError(f"Target size ({target.shape}) must be the same as output size ({output.shape})")
    
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

def softmax(x, dim=None, log=False):
    import jittor.other.code_softmax as code_softmax
    if code_softmax.can_softmax_v1(x, dim) and jt.compiler.is_cuda:
        return code_softmax.softmax_v1(x, log)
    if dim is None: dim = ()
    dtype, x = x.dtype, x._to_float()
    if log:
        a = x - jt.max(x, dim, keepdims=True)
        ret = a - a.exp().sum(dim, keepdims=True).log()
    else:
        x = (x - jt.max(x, dim, keepdims=True)).exp()
        ret = x / x.sum(dim, keepdims=True)
    return ret.cast(dtype)
jt.Var.softmax = softmax

def log_softmax(x,dim=None):
    return softmax(x,dim=dim, log=True)
jt.Var.log_softmax = log_softmax

def log_sigmoid(x):
    return jt.log(jt.sigmoid(x))
jt.Var.log_sigmoid = log_sigmoid

def logsumexp(x, dim, keepdims=False, keepdim=False):
    return x.exp().sum(dim, keepdim or keepdims).log()
jt.Var.logsumexp = logsumexp

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
        output = output.to(input.dtype)
        return output

def dropout(x,p=0.5,is_train=False):
    return Dropout(p=p,is_train=is_train)(x)

class Dropout2d(Module):
    def __init__(self, p=0.5, is_train=False):
        '''
        Randomly zero out entire channels, from "Efficient Object Localization Using Convolutional Networks"
        input:
            x: [N,C,H,W] or [N,C,L]
        output:
            y: same shape as x
        '''
        assert p >= 0 and p <= 1, "dropout probability has to be between 0 and 1, but got {}".format(p)
        self.p = p
        self.is_train = is_train
        #TODO: test model.train() to change self.is_train
    def execute(self, input):
        output = input
        if (input.dim() != 4) and (input.dim() != 3):
            raise RuntimeError(f'Expected 3D (unbatched) or 4D (batched) input to Dropout2d, but got input of size: {input.shape}')
        shape = input.shape[:-2]
        if self.p > 0 and self.is_train:
            if self.p == 1:
                output = jt.zeros(input.shape)
            else:
                noise = jt.random(shape)
                noise = (noise > self.p).int()
                output = output * noise.broadcast(input.shape, dims=[-2,-1]) / (1.0 - self.p) # div keep prob
        return output

def dropout2d(x,p=0.5,is_train=False):
    return Dropout2d(p=p,is_train=is_train)(x)

class DropPath(Module):
    '''Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    '''
    def __init__(self, p=0.5, is_train=False):
        '''
            :param p: Specifies the probability of each batch retention. Defaults to 0.5.
            :type p: float dtype
            :param is_train: Specify whether it is a training model. Defaults to False.
            :type is_train: bool
        '''
        self.p = p
        self.is_train = is_train
        #TODO: test model.train() to change self.is_train
    def execute(self, x):
        if self.p == 0. or not self.is_train:
            return x
        keep_prob = 1 - self.p
        shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
        random_tensor = keep_prob + jt.rand(shape, dtype=x.dtype)
        output = x.divide(keep_prob) * random_tensor.floor()
        return output

def droppath(x,p=0.5,is_train=False):
    return DropPath(p=p,is_train=is_train)(x)

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
    
def linear(x, weight, bias=None):
    ''' Returns x * weight^T
    '''
    x = matmul_transpose(x, weight)
    if bias is not None:
        return x + bias
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

def batch_norm(x, running_mean, running_var, weight=1, bias=0, training=False, momentum=0.1, eps=1e-05):
    dims = [0]+list(range(2,x.ndim))
    assert not training
    w = weight / jt.sqrt(running_var+eps)
    b = bias - running_mean * w
    norm_x = x * w.broadcast(x, dims) + b.broadcast(x, dims)
    return norm_x


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

def fp32_guard(func):
    def wrapper(*args, **kw):
        if jt.flags.amp_level == 0:
            return func(*args, **kw)
        new_args = []
        need_cast = False
        dtype = None
        for a in args:
            if isinstance(a, jt.Var) and (a.dtype == "float16" or a.dtype == "bfloat16"):
                dtype = a.dtype
                new_args.append(a.float32())
                need_cast = True
            else:
                new_args.append(a)
        with jt.flag_scope(amp_level=0):
            a = func(*new_args, **kw)
            if need_cast and isinstance(a, jt.Var) and a.dtype == "float32":
                a = a.cast(dtype)
        return a
    return wrapper

def instance_norm(x, 
    running_mean = None,
    running_var = None,
    weight = 1,
    bias = 0,
    momentum = 0.1,
    eps = 1e-5):
    dims = list(range(2,x.ndim))
    xmean = jt.mean(x, dims=dims)
    x2mean = jt.mean(x*x, dims=dims)

    xvar = (x2mean-xmean*xmean).maximum(0.0)
    w = weight / jt.sqrt(xvar+eps)
    b = bias - xmean * w
    return x * w.broadcast(x, dims) + b.broadcast(x, dims)

class LayerNorm(Module):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = init.constant(normalized_shape, "float32", 1.0) if elementwise_affine else 1.0
        self.bias = init.constant(normalized_shape, "float32", 0.0) if elementwise_affine else 0.0

    @fp32_guard
    def execute(self, x):
        dims = [-i for i in range(len(self.normalized_shape), 0, -1)]
        xmean = jt.mean(x, dims=dims, keepdims=1)
        x2mean = jt.mean(x*x, dims=dims, keepdims=1)

        xvar = (x2mean-xmean*xmean).maximum(0.0)
        w = self.weight / jt.sqrt(xvar+self.eps)
        b = self.bias - xmean * w
        return x * w + b


LayerNorm3d = LayerNorm2d = LayerNorm1d = LayerNorm

@fp32_guard
def layer_norm(x, 
    normalized_shape, 
    weight = 1,
    bias = 0,
    eps: float = 1e-5, 
    elementwise_affine: bool = True):
    dims = [-i for i in range(len(normalized_shape), 0, -1)]
    xmean = jt.mean(x, dims=dims, keepdims=1)
    x2mean = jt.mean(x*x, dims=dims, keepdims=1)

    xvar = (x2mean-xmean*xmean).maximum(0.0)
    w = weight / jt.sqrt(xvar+eps)
    b = bias - xmean * w
    return x * w + b

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
        # output_shape = (N,-1)
	    # TODO: 3d group norm
        # if x.ndim==4:
            # output_shape = x.shape
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

def group_norm(x, 
    num_groups, 
    weight = 1,
    bias = 0,
    eps=1e-05):
    N = x.shape[0]
    C = x.shape[1]
    output_shape = (N,-1)
    # TODO: 3d group norm
    if x.ndim==4:
        output_shape = x.shape
    assert C % num_groups == 0
    x = x.reshape((N, num_groups, C//num_groups, -1))
    xmean = jt.mean(x, dims=[2,3]).reshape((N, num_groups, 1))
    x2mean = jt.mean(x*x, dims=[2,3]).reshape((N, num_groups, 1))
    xvar = (x2mean-xmean*xmean).maximum(0.0)

    if isinstance(weight, jt.Var):
        weight = weight.reshape((1, num_groups, -1))
    if isinstance(bias, jt.Var):
        bias = bias.reshape((1, num_groups, -1))
    weight = weight / jt.sqrt(xvar+eps)
    bias = bias - xmean * weight
    x = x * weight.broadcast(x, [3]) + bias.broadcast(x, [3])
    return x.reshape(output_shape)


Relu = jt.make_module(relu)
ReLU = Relu
Leaky_relu = jt.make_module(leaky_relu, 2)
LeakyReLU = Leaky_relu
ReLU6 = jt.make_module(relu6)
Softmax = jt.make_module(softmax, 2)
GELU = jt.make_module(gelu)
SiLU = jt.make_module(silu)

class Flatten(Module):
    ''' Flattens the contiguous range of dimensions in a Var.

    :param start_dim: the first dimension to be flattened. Defaults: 1.
    :type start_dim: int

    :param end_dim: the last dimension to be flattened. Defaults: -1.
    :type end_dim: int
    '''
    def __init__(self, start_dim=1, end_dim=-1):
        self.start_dim = start_dim
        self.end_dim = end_dim

    def execute(self, x) -> jt.Var:
        return x.flatten(self.start_dim, self.end_dim)


from jittor.depthwise_conv import DepthwiseConv

class Conv(Module):
    ''' Applies a 2D convolution over an input signal composed of several input planes.

    :param in_channels: Number of channels in the input feature map
    :type in_channels: int

    :param out_channels: Number of channels in the output feature map
    :type out_channels: int

    :param kernel_size: Size of the convolving kernel
    :type kernel_size: int or tuple

    :param stride: Stride of the convolution. Default: 1
    :type stride: int or tuple, optional

    :param padding: Padding added to all four sides of the input. Default: 0
    :type padding: int or tuple, optional

    :param dilation: Spacing between kernel elements. Default: 1
    :type dilation: int or tuple, optional

    :param groups: Number of blocked connections from input channels to output channels. Default: 1
    :type groups: int, optional

    :param bias: If True, adds a learnable bias to the output. Default: True
    :type bias: bool, optional

    Example:

    >>> conv = nn.Conv2d(24, 32, 3)
    >>> conv = nn.Conv2d(24, 32, (3,3))
    >>> conv = nn.Conv2d(24, 32, 3, stride=2, padding=1)
    >>> conv = nn.Conv2d(24, 32, 3, dilation=(3, 1))
    >>> input = jt.randn(4, 24, 100, 100)
    >>> output = conv(input)
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        if in_channels <= 0:
            raise ValueError(f"in_channels must be greater than zero, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be greater than zero, got {out_channels}")
        if groups <= 0:
            raise ValueError(f"groups must must be greater than zero, got {groups}")
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        assert out_channels % groups == 0, 'out_channels must be divisible by groups'
        if isinstance(kernel_size, tuple):
            for size in kernel_size:
                if size <= 0:
                    raise ValueError(f"kernel_size must be greater than zero, got {kernel_size}")
        else:
            if kernel_size <= 0:
                raise ValueError(f"kernel_size must be greater than zero, got {kernel_size}")
        if isinstance(stride, tuple):
            for size in stride:
                if size <= 0:
                    raise ValueError(f"stride must be greater than zero, got {stride}")
        else:
            if stride <= 0:
                raise ValueError(f"stride must be greater than zero, got {stride}")
        if isinstance(padding, tuple):
            for size in padding:
                if size < 0:
                    raise ValueError(f"padding must be nonnegative, got {padding}")
        else:
            if padding < 0:
                raise ValueError(f"padding must be nonnegative, got {padding}")
        if isinstance(dilation, tuple):
            for size in dilation:
                if size <= 0:
                    raise ValueError(f"dilation must be greater than zero, got {dilation}")
        else:
            if dilation <= 0:
                raise ValueError(f"dilation must be greater than zero, got {dilation}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.is_depthwise_conv = self.groups == self.out_channels and self.groups == self.in_channels
        if self.is_depthwise_conv and jt.flags.use_cuda and jt.compiler.is_cuda:
            self.depthwise_conv = DepthwiseConv(stride, padding, dilation)
        Kh, Kw = self.kernel_size

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
        if hasattr(self, 'depthwise_conv'):
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
            with jt.flag_scope(amp_reg = jt.flags.amp_reg | 36):
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
    ''' Applies a 1D convolution over an input signal composed of several input planes.

    :param in_channels: Number of channels in the input feature map
    :type in_channels: int

    :param out_channels: Number of channels in the output feature map
    :type out_channels: int

    :param kernel_size: Size of the convolving kernel
    :type kernel_size: int or tuple

    :param stride: Stride of the convolution. Default: 1
    :type stride: int or tuple, optional

    :param padding: Padding added to all four sides of the input. Default: 0
    :type padding: int or tuple, optional

    :param dilation: Spacing between kernel elements. Default: 1
    :type dilation: int or tuple, optional

    :param groups: Number of blocked connections from input channels to output channels. Default: 1
    :type groups: int, optional

    :param bias: If True, adds a learnable bias to the output. Default: True
    :type bias: bool, optional

    Example:

    >>> conv = nn.Conv1d(24, 32, 3)
    >>> conv = nn.Conv1d(24, 32, (3,3))
    >>> conv = nn.Conv1d(24, 32, 3, stride=2, padding=1)
    >>> conv = nn.Conv1d(24, 32, 3, dilation=(3, 1))
    >>> input = jt.randn(4, 24, 100)
    >>> output = conv(input)
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        assert in_channels > 0, 'in_channels must be positive'
        assert out_channels > 0, 'out_channels must be positive'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, 1)
        self.stride = (stride, 1)
        self.padding = (padding, 0)
        self.dilation = (dilation, 1)
        self.groups = groups
        self.bias = bias
        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        assert out_channels % groups == 0, 'out_channels must be divisible by groups'
        # using list to escape module dfs
        self._conv = [Conv(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)]
        self.weight = self._conv[0].weight.squeeze(-1)
        self.bias = self._conv[0].bias

    def execute(self, x):
        if x.dim() != 3:
            raise ValueError("Input shape must be `(N, C, L)`!")
        N,C,D = x.shape
        assert C==self.in_channels
        self._conv[0].weight = self.weight.unsqueeze(-1)
        x = x.unsqueeze(-1)
        x = self._conv[0](x)
        y = x.squeeze(-1)
        return y

class Conv3d(Module):
    ''' Applies a 3D convolution over an input signal composed of several input planes.

    :param in_channels: Number of channels in the input feature map
    :type in_channels: int

    :param out_channels: Number of channels in the output feature map
    :type out_channels: int

    :param kernel_size: Size of the convolving kernel
    :type kernel_size: int or tuple

    :param stride: Stride of the convolution. Default: 1
    :type stride: int or tuple, optional

    :param padding: Padding added to all four sides of the input. Default: 0
    :type padding: int or tuple, optional

    :param dilation: Spacing between kernel elements. Default: 1
    :type dilation: int or tuple, optional

    :param groups: Number of blocked connections from input channels to output channels. Default: 1
    :type groups: int, optional

    :param bias: If True, adds a learnable bias to the output. Default: True
    :type bias: bool, optional

    Example:

    >>> conv = nn.Conv3d(24, 32, 3)
    >>> conv = nn.Conv3d(24, 32, (3,3))
    >>> conv = nn.Conv3d(24, 32, 3, stride=2, padding=1)
    >>> conv = nn.Conv3d(24, 32, 3, dilation=(3, 1))
    >>> input = jt.randn(4, 24, 50, 50, 50)
    >>> output = conv(input)
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        if groups <= 0:
            raise ValueError("groups must be a positive integer")
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


class Conv1d_sp(Linear):
    def __init__(self, inchannels, outchannels, kernel_size=1, bias=True):
        assert inchannels > 0, 'in_channels must be positive'
        assert outchannels > 0, 'out_channels must be positive'
        super().__init__(inchannels, outchannels, bias=bias)
        assert kernel_size == 1

    def execute(self, x):
        if x.dim() != 3:
            raise ValueError("Input shape must be `(N, C, L)`!")
        x = x.transpose(0, 2, 1)
        x = super().execute(x)
        x = x.transpose(0, 2, 1)
        return x

def conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    ''' Applies a 2D convolution over an input signal composed of several input planes.

    :param x: the input image
    :type x: jt.Var

    :param weight: the convolution kernel
    :type weight: jt.Var

    :param bias: the bias after convolution
    :type bias: jt,Var, optional

    :param stride: Stride of the convolution. Default: 1
    :type stride: int or tuple, optional

    :param padding: Padding added to all four sides of the input. Default: 0
    :type padding: int or tuple, optional

    :param dilation: Spacing between kernel elements. Default: 1
    :type dilation: int or tuple, optional

    :param groups: Number of blocked connections from input channels to output channels. Default: 1
    :type groups: int, optional

    Example:

    >>> x = jt.randn(4, 24, 100, 100)
    >>> w = jt.randn(32, 24, 3, 3)
    >>> y = nn.conv2d(x, w)
    '''
    padding = _pair(padding)
    stride = _pair(stride)
    dilation = _pair(dilation)
    out_channels = weight.shape[0]
    if groups <= 0:
        raise ValueError("groups must be a positive integer")
    if groups == 1:
        N,C,H,W = x.shape
        Kh, Kw = weight.shape[-2:]
        oh = (H+padding[0]*2-Kh*dilation[0]+dilation[0]-1)//stride[0]+1
        ow = (W+padding[1]*2-Kw*dilation[1]+dilation[1]-1)//stride[1]+1
        with jt.flag_scope(amp_reg = jt.flags.amp_reg | 36):
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
conv = conv2d

def conv3d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    ''' Applies a 3D convolution over an input signal composed of several input planes.

    :param x: the input volume
    :type x: jt.Var

    :param weight: the convolution kernel
    :type weight: jt.Var

    :param bias: the bias after convolution
    :type bias: jt,Var, optional

    :param stride: Stride of the convolution. Default: 1
    :type stride: int or tuple, optional

    :param padding: Padding added to all four sides of the input. Default: 0
    :type padding: int or tuple, optional

    :param dilation: Spacing between kernel elements. Default: 1
    :type dilation: int or tuple, optional

    :param groups: Number of blocked connections from input channels to output channels. Default: 1
    :type groups: int, optional

    Example:

    >>> x = jt.randn(4, 24, 50, 50, 50)
    >>> w = jt.randn(32, 24, 3, 3, 3)
    >>> y = nn.conv2d(x, w)
    '''
    padding = _triple(padding)
    stride = _triple(stride)
    dilation = _triple(dilation)
    out_channels = weight.shape[0]
    if groups <= 0:
        raise ValueError("groups must be a positive integer")
    if jt.flags.use_cuda and jt.cudnn:
        y = jt.cudnn.ops.cudnn_conv3d(x, weight, *stride, *padding, *dilation, groups)
    elif groups == 1:
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
        self.groups = groups

        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        # added
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.real_padding = (self.dilation[0] * (self.kernel_size[0] - 1) - self.padding[0],
            self.dilation[1] * (self.kernel_size[1] - 1) - self.padding[1])
        self.output_padding = output_padding if isinstance (output_padding, tuple) else (output_padding, output_padding)
        assert self.stride[0] > 0 and self.stride[1] > 0,"stride must be positive"
        assert self.padding[0] >= 0 and self.padding[1] >= 0,"padding must be non-negative"
        assert self.output_padding[0] < max(self.stride[0], self.dilation[0]) and \
            self.output_padding[1] < max(self.stride[1], self.dilation[1]), \
            "output padding must be smaller than max(stride, dilation)"
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        assert out_channels % groups == 0, 'out_channels must be divisible by groups'

        self.weight = init.invariant_uniform((in_channels, out_channels//groups) + self.kernel_size, dtype="float")
        if bias:
            fan=1
            for i in self.weight.shape[1:]:
                fan *= i
            bound = 1 / math.sqrt(fan)
            self.bias = init.uniform([out_channels], dtype="float", low=-bound, high=bound)
        else:
            self.bias = None

    def execute(self, x):
        if x.dim() != 4:
            raise RuntimeError(f'Expected 4D (batched) input to conv_transpose2d, but got input of size: {x.shape}')
        if self.groups == 1:
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
        else:
            N,C,H,W = x.shape
            Kh, Kw = self.kernel_size
            i,o,h,w = self.weight.shape
            oc = self.out_channels
            G = self.groups
            CpG = C // G # channels per group
            assert C==self.in_channels
            stride_h, stride_w = self.stride
            padding_h, padding_w = self.padding
            dilation_h, dilation_w = self.dilation

            oh = (H-1) * stride_h + self.output_padding[0] - 2*padding_h + 1 + (h-1)*dilation_h
            ow = (W-1) * stride_w + self.output_padding[1] - 2*padding_w + 1 + (w-1)*dilation_w
            out_shape = (N, oc, oh, ow)
            shape = [N,G,oc//G,CpG,oh,ow,Kh,Kw]
            xx = x.reindex(shape, [
                'i0',
                f'i1*{oc//G}+i2',
                'i4',
                'i5'
            ])
            ww = self.weight.reindex(shape, [
                f'i1*{oc//G}+i2',
                'i3',
                'i6',
                'i7'
            ])
            ww.compile_options = xx.compile_options = {"G":G,"C":C}
            y = (ww*xx).reindex_reduce("add", out_shape, [
                'i0', # Nid
                f'i1*{CpG}+i3', # Gid
                f'i4*{self.stride[0]}-{self.padding[0]}+i6*{self.dilation[0]}', # Hid+Khid
                f'i5*{self.stride[1]}-{self.padding[1]}+i7*{self.dilation[1]}', # Wid+KWid
            ])
            if self.bias is not None:
                b = self.bias.broadcast(y.shape, [0,2,3])
                y = y + b
            return y
ConvTranspose2d = ConvTranspose

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
    if groups == 1:
        x = input
        if x.dim() != 4:
            raise RuntimeError(f'Expected 4D input to conv_transpose, but got input of size: {x.shape}')
        N,C,H,W = x.shape
        i,o,h,w = weight.shape
        assert C==i
        stride = stride if isinstance(stride, tuple) else (stride, stride)
        if stride[0] <= 0 or stride[1] <= 0:
            raise RuntimeError("non-positive stride is not supported")
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
    else:
        if input.dim() != 4:
            raise RuntimeError(f'Expected 4D input to conv_transpose, but got input of size: {input.shape}')
        N,C,H,W = input.shape
        i,o,h,w = weight.shape
        G = groups
        oc = o * G
        CpG = C // G # channels per group
        assert C % G == 0
        assert C==i, (C, i)
        stride = stride if isinstance(stride, tuple) else (stride, stride)
        if stride[0] <= 0 or stride[1] <= 0:
            raise RuntimeError("non-positive stride is not supported")
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

        oh = (H-1) * stride_h + output_padding[0] - 2*padding_h + 1 + (h-1)*dilation_h
        ow = (W-1) * stride_w + output_padding[1] - 2*padding_w + 1 + (w-1)*dilation_w
        out_shape = (N, oc, oh, ow)
        shape = [N,G,oc//G,CpG,oh,ow,h,w]
        xx = input.reindex(shape, [
            'i0',
            f'i1*{oc//G}+i2',
            'i4',
            'i5'
        ])
        ww = weight.reindex(shape, [
            f'i1*{oc//G}+i2',
            'i3',
            'i6',
            'i7'
        ])
        ww.compile_options = xx.compile_options = {"G":G,"C":C}
        y = (ww*xx).reindex_reduce("add", out_shape, [
            'i0', # Nid
            f'i1*{CpG}+i3', # Gid
            f'i4*{stride[0]}-{padding[0]}+i6*{dilation[0]}', # Hid+Khid
            f'i5*{stride[1]}-{padding[1]}+i7*{dilation[1]}', # Wid+KWid
        ])
        if bias is not None:
            b = bias.broadcast(y.shape, [0,2,3])
            y = y + b
        return y
conv_transpose2d = conv_transpose

def conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    x = input
    if x.dim() != 5:
        raise RuntimeError(f'Expected 5D input to conv_transpose3d, but got input of size: {x.shape}')
    N,C,D,H,W = x.shape
    i,o,d,h,w = weight.shape
    assert C==i
    assert groups==1, "Group conv not supported yet."
    stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
    if stride[0] <= 0 or stride[1] <= 0 or stride[2] <= 0:
        raise RuntimeError("non-positive stride is not supported")
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
        if padding < 0:
            raise RuntimeError(f"padding must be > 0, but got {padding}")
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
        if self.pl < 0 or self.pr < 0 or self.pt < 0 or self.pb < 0:
            raise ValueError(f"padding must be non-negative")

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
        if self.pl < 0 or self.pr < 0 or self.pt < 0 or self.pb < 0:
            raise ValueError(f"padding must be non-negative")

    def execute(self, x):
        if x.dim() != 4:
            raise RuntimeError("Input shape must be `(N, C, H, W)`!")
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
        if self.pl < 0 or self.pr < 0 or self.pt < 0 or self.pb < 0:
            raise ValueError(f"padding must be non-negative")

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
        if padding < 0:
            raise RuntimeError(f"padding must be > 0, but got {padding}")
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
        if self.pl < 0 or self.pr < 0 or self.pt < 0 or self.pb < 0:
            raise ValueError(f"padding must be non-negative")

    def execute(self, x):
        if x.dim() != 4:
            raise RuntimeError("Input shape must be `(N, C, H, W)`!")
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
    ''' A simple lookup table that stores embeddings of a fixed dictionary and size.

        :param num: size of the dictionary of embeddings
        :type num: int

        :param dim: the size of each embedding vector
        :type dim: int

        Example:
            >>> embedding = nn.Embedding(10, 3)
            >>> x = jt.int32([1, 2, 3, 3])
            >>> embedding(x)
            jt.Var([[ 1.1128596   0.19169547  0.706642]
             [ 1.2047412   1.9668795   0.9932192]
             [ 0.14941819  0.57047683 -1.3217674]
             [ 0.14941819  0.57047683 -1.3217674]], dtype=float32)
    '''
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, dtype="float32"):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = jt.init.gauss([self.num_embeddings, self.embedding_dim], dtype)
        if padding_idx is not None:
            self.weight[padding_idx] = 0

    def execute(self, x):
        res = self.weight[x]
        return res

def embedding(input, weight):
    return weight[input]

class PixelShuffle(Module):
    def __init__(self, upscale_factor):
        assert upscale_factor > 0,f"upscale_factor must be greater than zero,got {upscale_factor}"
        self.upscale_factor = upscale_factor

    def execute(self, x):
        n,c,h,w = x.shape
        r = self.upscale_factor
        assert c%(r*r)==0, f"input channel needs to be divided by upscale_factor's square in PixelShuffle"
        if r<=0:
            raise RuntimeError(f"pixel_shuffle expects a positive upscale_factor, but got {r}")
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
        if isinstance(size,int):
            if size <= 0:
                raise ValueError(f"sizes must be positive, got {size}")
        elif isinstance(size,tuple) or isinstance(size,list):
            for item in size:
                if item <= 0:
                    raise ValueError(f"sizes must be positive, got {item}")
        else:
            raise ValueError(f"size must be int or tuple")
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
        return img.reindex([*ids, x.floor_int(), y.floor_int()])
    if mode == "bilinear":
        fx, fy = x.floor_int(), y.floor_int()
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
        fx, fy = x.floor_int(), y.floor_int()
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

# TODO: tf_mode to another function
def resize(img, size, mode="nearest", align_corners=False, tf_mode=False):
    if img.dim() != 4:
        raise ValueError("Input shape must be `(N, C, H, W)`!")
    n, c, h, w = img.shape
    H, W = size
    if h <= 0 or w <= 0 or H <= 0 or W <= 0:
        raise RuntimeError(f"Input and output sizes should be greater than 0, but got input (H: {h}, W: {w}) output (H: {H}, W: {W})")
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
    elif mode == "area":
        '''
        Area interpolation uses AdaptivePool2D to resize origin images.
        '''
        stride = (h // H, w // W)
        assert stride[0] > 0 and stride[1] > 0
        x, y = jt.meshgrid(jt.arange(0, H, 1), jt.arange(0, W, 1))
        startH = jt.floor(x*h/H).int32()
        endH = jt.ceil((x+1)*h/H).int32()
        maxH = int(jt.max(endH - startH).data)
        startW = jt.floor(y*w/W).int32()
        endW = jt.ceil((y+1)*w/W).int32()
        maxW = int(jt.max(endW - startW).data)
        pixel_count = (endH - startH) * (endW - startW)
        adaptive_output = img.reindex([img.shape[0], img.shape[1], H, W, maxH, maxW], ["i0", "i1", "@e0(i2, i3) + i4", "@e2(i2, i3) + i5"], extras=[startH, endH, startW, endW], overflow_conditions=["i4 >= @e1(i2, i3) - @e0(i2, i3)", "i5 >= @e3(i2, i3) - @e2(i2, i3)"], overflow_value=0)
        adaptive_output = adaptive_output.reduce("sum", [4,5]) / pixel_count[None, None, ...]
        return adaptive_output
    else:
        if (tf_mode):
            x = hid * (h / H)
            if H > h: x = x.clamp(0, h - 1)
            y = wid * (w / W)
            if W > w: y = y.clamp(0, w - 1)
        else:
            x = hid * (h / H) + (h / H * 0.5 - 0.5)
            if H > h: x = x.clamp(0, h - 1)
            y = wid * (w / W) + (w / W * 0.5 - 0.5)
            if W > w: y = y.clamp(0, w - 1)
    return _interpolate(img, x, y, (nid, cid), mode)

upsample = resize


def interpolate(X, size=None, scale_factor=None, mode='bilinear', align_corners=False, tf_mode=False):
    if scale_factor is not None:
        size = [int(X.shape[-2] * scale_factor), int(X.shape[-1] * scale_factor)]
    if isinstance(size, int):
        size = (size, size)
    if scale_factor is not None and scale_factor > 1:
        return upsample(X, size, mode, align_corners, tf_mode)
    else:
        return resize(X, size, mode, align_corners, tf_mode)


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
    flips = (x / span).floor_int()
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
        return X.reindex([nid,cid,zid.round_int(),yid.round_int(),xid.round_int()])
    elif mode=='bilinear':
        fx,fy,fz = xid.floor_int(),yid.floor_int(),zid.floor_int()
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
        return X.reindex([nid,cid,yid.round_int(),xid.round_int()])
    elif mode=='bilinear':
        #xid,yid = (xid+0.00001),(yid+0.00001)
        fx,fy = (xid).floor_int(),(yid).floor_int()
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
    def __init__(self, scale_factor=None, mode='nearest', align_corners=False):
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def execute(self, x):
        if self.scale_factor is None:
            raise ValueError("scale_factor should be defined")
        elif isinstance(self.scale_factor, float):
            return upsample(x, 
                size=(int(x.shape[2]*self.scale_factor),
                      int(x.shape[3]*self.scale_factor)),
                mode=self.mode,
                align_corners=self.align_corners)
        else:
            return upsample(x,
                size=(
                    int(x.shape[2]*self.scale_factor[0]), 
                    int(x.shape[3]*self.scale_factor[1])),
                mode=self.mode,
                align_corners=self.align_cornerss)

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
        if isinstance(idx, slice) or idx not in self.layers:
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
    def dfs(self, parents, k, callback, callback_leave, recurse=True):
        n_children = len(self.layers)
        ret = callback(parents, k, self, n_children)
        if ret == False:
            return
        parents.append(self)
        if recurse:
            for k,v in self.layers.items():
                if isinstance(v, Module):
                    v.dfs(parents, k, callback, callback_leave)
        parents.pop()
        if callback_leave:
            callback_leave(parents, k, self, n_children)
    def append(self, mod):
        assert callable(mod), f"Module <{type(mod)}> is not callable"
        assert not isinstance(mod, type), f"Module is not a type"
        self.layers[str(len(self.layers))]=mod
    def add_module(self, name, mod):
        assert callable(mod), f"Module <{type(mod)}> is not callable"
        assert not isinstance(mod, type), f"Module is not a type"
        self.layers[str(name)]=mod

    def __len__(self):
        return len(self.layers)
    
    def named_children(self,):
        return list(self.layers.items())

    def __setattr__(self, key, value) -> None:
        if isinstance(key, str) and key.isdigit():
            if int(key)<len(self.layers):
                self.add_module(key, value)
            else:
                super().__setattr__(key, value)
        else:
            super().__setattr__(key, value)
    

    def __getattr__(self, key):
        if 'layers' in self.__dict__ and key in self.__dict__['layers']:
            return self.__dict__['layers'][key]
        return super().__getattr__(key)


class ParameterList(Module):
    def __init__(self, *args):
        self.params = collections.OrderedDict()
        for var in args:
            if isinstance(var, (collections.OrderedDict, dict)):
                for k, v in var.items():
                    self.add_param(k, v)
            elif isinstance(var, list):
                for v in var:
                    self.append(v)
            else:
                self.append(var)
    def __getitem__(self, idx):
        if idx not in self.params:
            return list(self.params.values())[idx]

        return self.params[idx]
    def __iter__(self):
        return self.params.values().__iter__()
    def keys(self):
        return self.params.keys()
    def values(self):
        return self.params.values()
    def items(self):
        return self.params.items()
    def execute(self, x):
        raise NotImplementedError("Parameters is not executable")
    def append(self, var):
        assert isinstance(var, jt.Var), f"argument <{type(var)}> is not jittor var"
        self.params[len(self.params)] = var
    def add_param(self, name, var):
        assert isinstance(var, jt.Var), f"argument <{type(var)}> is not jittor var"
        self.params[name]=var
    def __setitem__(self, name, var):
        self.add_param(name, var)

    def __len__(self):
        return len(self.params)

ParameterDict = ParameterList

def Parameter(data, requires_grad=True):
    ''' The `Parameter` interface isn't needed in Jittor, this interface
does nothings and it is just used for compatible.
    
A Jittor Var is a Parameter
when it is a member of Module, if you don't want a Jittor
Var menber is treated as a Parameter, just name it startswith
underscore `_`.
    '''
    LOG.w(Parameter.__doc__)
    data = data.clone()
    data.requires_grad = requires_grad
    return data

def backward(v, *args, **kw):
    ''' The `backward` variable interface doesn't exist in Jittor.
please use `optimizer.backward(loss)` or 
`optimizer.step(loss)` instead.
For example, if your code looks like this::

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

It can be changed to this::

    optimizer.zero_grad()
    optimizer.backward(loss)
    optimizer.step()

Or more concise::

    optimizer.step(loss)

The step function will automatically zero grad and backward.
    '''
    LOG.f(backward.__doc__)

jt.Var.backward = backward

def unfold(X, kernel_size, dilation=1, padding=0, stride=1):
    assert X.ndim == 4
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    assert kernel_size[0] > 0 and kernel_size[1] > 0, "kernel size must be positive"
    if not isinstance(dilation, tuple):
        dilation = (dilation, dilation)
    assert dilation[0] > 0 and dilation[1] > 0, "dilation must be positive"
    if not isinstance(padding, tuple):
        padding = (padding, padding)
    assert padding[0] >= 0 and padding[1] >= 0, "padding must be non-negative"
    if not isinstance(stride, tuple):
        stride = (stride, stride)
    assert stride[0] > 0 and stride[1] > 0, "stride must be positive"
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
    assert output_size[0] > 0 and output_size[1] > 0, "output size must be positive."
    if not isinstance(kernel_size,tuple):
        kernel_size = (kernel_size,kernel_size)
    assert kernel_size[0] > 0 and kernel_size[1] > 0, "kernel size must be positive"
    if not isinstance(dilation,tuple):
        dilation = (dilation,dilation)
    assert dilation[0] > 0 and dilation[1] > 0, "dilation must be positive"
    if not isinstance(padding,tuple):
        padding = (padding,padding)
    assert padding[0] >= 0 and padding[1] >= 0, "padding must be non-negative"
    if not isinstance(stride,tuple):
        stride = (stride,stride)
    assert stride[0] > 0 and stride[1] > 0, "stride must be positive"
    n,cl,num = X.shape
    area = kernel_size[0] * kernel_size[1]
    block_nums = []
    for i in range(2,4):
        block_nums.append((output_size[i-2]+2*padding[i-2]-dilation[i-2]*(kernel_size[i-2]-1)-1) // stride[i-2]+1)
    output = X.reindex_reduce("add",[n,cl // area,output_size[0]+2*padding[0],output_size[1]+2*padding[1]],["i0",f"i1/{area}",f"i2/{block_nums[1]}*{stride[0]}+(i1%{area})/{kernel_size[1]}*{dilation[0]}",f"i2%{block_nums[1]}*{stride[1]}+(i1%{area})%{kernel_size[1]}*{dilation[1]}"])
    return output[:,:,padding[0]:padding[0]+output_size[0],padding[1]:padding[1]+output_size[1]]

ModuleList = Sequential


class LSTMCell(jt.Module):
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
    def __init__(self, input_size, hidden_size, bias=True):
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
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity = "tanh"):
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
    def __init__(self, input_size, hidden_size, bias=True):
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
            proj_size: int = 0, nonlinearity: str = None) -> None:
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
        self.nonlinearity = nonlinearity

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

    def _cudnn_flatten_weights(self, cudnn_mode):
        def copy_to_flatten_weight(param_name, offset_idx, num_gates):
            def copy_to(param_name, offset_idx, idx):
                cur_offset = self._cudnn_weight_offset[offset_idx]
                param = getattr(self, param_name)
                param = param[self.hidden_size * idx: self.hidden_size * (idx + 1)]
                ft_weight[cur_offset:cur_offset + param.numel()] = param.flatten()
                
            if self.bias:
                for idx in range(num_gates):
                    copy_to('weight' + param_name, offset_idx + idx * 2, idx)
                    copy_to('bias' + param_name, offset_idx + idx * 2 + 1, idx)
                return num_gates * 2
            else:
                for idx in range(num_gates):
                    copy_to('weight' + param_name, offset_idx + idx, idx)
                return num_gates

        if jt.flags.use_cuda and jt.cudnn and jt.compiler.is_cuda:
            if getattr(self, '_cudnn_weight_size', None) is None:                
                offset_array = jt.cudnn.cudnn_rnn_weight_offset(
                    cudnn_mode,
                    self.input_size,
                    self.hidden_size, 
                    self.num_layers,
                    self.proj_size,
                    self.bias,
                    self.bidirectional
                )
                self._cudnn_weight_size = offset_array[0]
                self._cudnn_weight_offset = offset_array[1:]
            
            num_gates = {
                "RNN": 1, "LSTM": 4, "GRU": 3
            }[self.mode]
            ft_weight = jt.zeros(self._cudnn_weight_size, dtype=jt.float32)

            cnt = 0
            for layer in range(self.num_layers):
                suffix = ''
                cnt += copy_to_flatten_weight(f'_ih_l{layer}' + suffix, cnt, num_gates)
                cnt += copy_to_flatten_weight(f'_hh_l{layer}' + suffix, cnt, num_gates)
                if self.bidirectional:
                    suffix = '_reverse'
                    cnt += copy_to_flatten_weight(f'_ih_l{layer}' + suffix, cnt, num_gates)
                    cnt += copy_to_flatten_weight(f'_hh_l{layer}' + suffix, cnt, num_gates)
            return ft_weight
        else:
            raise RuntimeError("Not Cudnn found")

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

    def _execute_cudnn_rnn(self, input, hx):
        cudnn_mode = {
            ('RNN', 'tanh'): 'tanh',
            ('RNN', 'relu'): 'relu',
            ('LSTM', None): 'lstm',
            ('GRU', None): 'gru'
        }[(self.mode, self.nonlinearity)]
        ft_weight = self._cudnn_flatten_weights(cudnn_mode)

        if self.mode == 'LSTM':
            ret = jt.cudnn.ops.cudnn_rnn(input, hx[0], hx[1], ft_weight,
                cudnn_mode, self.input_size, self.hidden_size, self.num_layers, 0,
                self.dropout, self.bias, self.bidirectional, self.is_training()
            )
            return ret[0], (ret[1], ret[2])
        else:
            ret = jt.cudnn.ops.cudnn_rnn(input, hx, ft_weight,
                cudnn_mode, self.input_size, self.hidden_size, self.num_layers, 0,
                self.dropout, self.bias, self.bidirectional, self.is_training()
            )
            return ret[0], ret[1]

    def execute(self, input, hx=None):
        if self.batch_first:
            input = input.permute(1, 0, 2)

        num_directions = 2 if self.bidirectional else 1

        if hx is None:
            if self.mode in ['RNN', 'GRU']:
                hx = jt.zeros((num_directions * self.num_layers, input.shape[1], self.hidden_size), dtype=input.dtype)
            elif self.mode == 'LSTM':
                hx = (jt.zeros((num_directions * self.num_layers, input.shape[1], self.hidden_size), dtype=input.dtype),
                      jt.zeros((num_directions * self.num_layers, input.shape[1], self.hidden_size), dtype=input.dtype))

        if jt.flags.use_cuda and jt.cudnn and self.proj_size == 0 and jt.compiler.is_cuda:
            return self._execute_cudnn_rnn(input, hx)
        else:
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
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
        nonlinearity: str = 'tanh', bias: bool = True, batch_first: bool = False, 
        dropout: float = 0, bidirectional: bool = False) -> None:
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
            h = jt.nn.relu(y)

        return h, h


class LSTM(RNNBase):
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

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, 
            batch_first=False, dropout=0, bidirectional=False, proj_size=0):
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

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
        bias: bool = True, batch_first: bool = False, dropout: float = 0, 
        bidirectional: bool = False) -> None:
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

def bilinear(in1, in2, weight, bias):
    if weight.shape[1] != in1.shape[1]:
        raise RuntimeError(f"bilinear(): input1 size deos not match weight size: got {in1.shape[1]} but expected {weight.shape[1]}")
    if weight.shape[2] != in2.shape[1]:
        raise RuntimeError(f"bilinear(): input2 size deos not match weight size: got {in2.shape[1]} but expected {weight.shape[2]}")
    w = weight.transpose((1,0,2))
    w = w.reshape((w.shape[0], -1))
    x = jt.matmul(in1, w)
    x = x.reshape(x.shape[:-1]+[weight.shape[0], weight.shape[2]])
    y = in2.broadcast(x, (-2,))
    z = (x*y).sum(-1)
    if bias is not None:
        z += bias
    return z


class Bilinear(Module):
    ''' bilinear transformation $out = in1^T W in2 + bias$, Example::

    m = nn.Bilinear(20, 30, 40)
    input1 = jt.randn(128, 20)
    input2 = jt.randn(128, 30)
    output = m(input1, input2)
    print(output.shape)
    # [128, 40]

    '''
    def __init__(self, in1_features, in2_features, out_features, bias=True, dtype="float32"):
        bound = 1 / math.sqrt(in1_features)
        self.weight = jt.init.uniform([out_features, in1_features, in2_features], dtype, -bound, bound)
        self.bias = bias
        if bias:
            self.bias = jt.init.uniform([out_features], dtype, -bound, bound)

    def execute(self, in1, in2):
        return bilinear(in1, in2, self.weight, self.bias)

#TODO: support FFT2D only now.
def _fft2(x, inverse=False):
    assert(jt.flags.use_cuda==1)
    assert(len(x.shape) == 4)
    assert(x.shape[3] == 2)
    y = jt.compile_extern.cufft_ops.cufft_fft(x, inverse)
    if inverse:
        y /= x.shape[1] * x.shape[2]
    return y

class ComplexNumber:
    ''' Applys Complex number class.

        It's saved as jt.stack(real, imag, dim=-1)

        You can construct ComplexNumber with real part and imaginary part like ComplexNumber(real, imag)
        or real part only with ComplexNumber(real)
        or value after jt.stack with ComplexNumber(value, is_concat_value=True)

        add, sub, mul and truediv between ComplexNumber and ComplexNumber, jt.Var, int, float are implemented

        You can use 'shape', 'reshape' etc. as jt.Var

    Example:
        >>> real = jt.array([[[1., -2., 3.]]])
        >>> imag = jt.array([[[0., 1., 6.]]])
        >>> a = ComplexNumber(real, imag)
        >>> a + a
        >>> a / a
        >>> a.norm()                # sqrt(real^2+imag^2)
        >>> a.exp()                 # e^real(cos(imag)+isin(imag))
        >>> a.conj()                # ComplexNumber(real, -imag)
        >>> a.fft2()                # cuda only now. len(real.shape) equals 3
        >>> a.ifft2()               # cuda only now. len(real.shape) equals 3

        >>> a = jt.array([[1,1],[1,-1]])
        >>> b = jt.array([[0,-1],[1,0]])
        >>> c = ComplexNumber(a,b) / jt.sqrt(3)
        >>> c @ c.transpose().conj()
        ComplexNumber(real=jt.Var([[0.99999994 0.        ]
                [0.         0.99999994]], dtype=float32), imag=jt.Var([[0. 0.]
                [0. 0.]], dtype=float32))
    '''
    def __init__(self, real: jt.Var, imag: jt.Var=None, is_concat_value=False):
        if is_concat_value:
            assert real.shape[-1] == 2
            assert imag is None
            self.value = real
        elif imag is None:
            self.value = jt.stack([real, jt.zeros_like(real)], dim=-1)
        else:
            assert real.shape == imag.shape
            assert real.dtype == imag.dtype
            self.value = jt.stack([real, imag], dim=-1)

    @property
    def requires_grad(self):
        return self.value.requires_grad

    @property
    def real(self):
        return self.value[..., 0]

    @property
    def imag(self):
        return self.value[..., 1]

    @property
    def shape(self):
        return self.value.shape[:-1]

    @property
    def dtype(self):
        return "complex64"

    def norm(self):
        return jt.sqrt(jt.sqr(self.real) + jt.sqr(self.imag))

    def stop_grad(self):
        return ComplexNumber(self.value.stop_grad(), is_concat_value=True)

    def start_grad(self):
        return ComplexNumber(self.value.start_grad(), is_concat_value=True)
    
    def detach(self):
        return ComplexNumber(self.value.detach(), is_concat_value=True)

    def unsqueeze(self, dim=0):
        return ComplexNumber(jt.unsqueeze(self.real, dim=dim), jt.unsqueeze(self.imag, dim=dim))

    def squeeze(self, dim=0):
        return ComplexNumber(jt.squeeze(self.real, dim=dim), jt.squeeze(self.imag, dim=dim))

    def reshape(self, shape):
        return ComplexNumber(jt.reshape(self.real, shape), jt.reshape(self.imag, shape))
    
    def permute(self, *axes):
        return ComplexNumber(jt.permute(self.real, *axes), jt.permute(self.imag, *axes))

    def transpose(self, *axes):
        return ComplexNumber(jt.transpose(self.real, *axes), jt.transpose(self.imag, *axes))

    def broadcast(self, shape, dims):
       return ComplexNumber(self.real.broadcast(shape, dims), self.imag.broadcast(shape, dims))

    def sum(self, dims, keepdims: bool=False):
        return ComplexNumber(self.real.sum(dims, keepdims=keepdims), self.imag.sum(dims, keepdims=keepdims))

    def exp(self):
        er = jt.exp(self.real)
        return ComplexNumber(er * jt.cos(self.imag), er * jt.sin(self.imag))

    def conj(self):
        return ComplexNumber(self.real, -self.imag)

    def __add__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(self.real + other.real, self.imag + other.imag)
        elif isinstance(other, (int, float)):
            return ComplexNumber(self.real + other, self.imag)
        else:
            raise NotImplementedError

    def __radd__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(other.real + self.real, other.imag + self.imag)
        elif isinstance(other, (int, float)):
            return ComplexNumber(other + self.real, self.imag)
        else:
            raise NotImplementedError

    def __sub__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(self.real - other.real, self.imag - other.imag)
        elif isinstance(other, (int, float)):
            return ComplexNumber(self.real - other, self.imag)
        else:
            raise NotImplementedError

    def __rsub__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(other.real - self.real, other.imag - self.imag)
        elif isinstance(other, (int, float)):
            return ComplexNumber(other - self.real, self.imag)
        else:
            raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(self.real * other.real - self.imag * other.imag,
                                 self.real * other.imag + self.imag * other.real)
        elif isinstance(other, (int, float)):
            return ComplexNumber(self.value * other, is_concat_value=True)
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(other.real * self.real - other.imag * self.imag,
                                 other.imag * self.real + other.real * self.imag)
        elif isinstance(other, (int, float)):
            return ComplexNumber(other * self.value, is_concat_value=True)
        else:
            raise NotImplementedError

    def __truediv__(self, other):
        if isinstance(other, ComplexNumber):
            norm = jt.sqr(other.real) + jt.sqr(other.imag)
            return ComplexNumber((self.real * other.real + self.imag * other.imag) / norm,
                                 (self.imag * other.real - self.real * other.imag) / norm)
        elif isinstance(other, (int, float)):
            return ComplexNumber(self.value / other, is_concat_value=True)
        else:
            raise NotImplementedError

    def __rtruediv__(self, other):
        norm = jt.sqr(self.real) + jt.sqr(self.imag)
        if isinstance(other, ComplexNumber):
            return ComplexNumber((other.real * self.real + other.imag * self.imag) / norm,
                                 (other.imag * self.real - other.real * self.imag) / norm)
        elif isinstance(other, (int, float)):
            return ComplexNumber(other * self.real / norm, - other * self.imag / norm)
        else:
            raise NotImplementedError

    def __matmul__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(self.real @ other.real - self.imag @ other.imag,
                                 self.real @ other.imag + self.imag @ other.real)
        else:
            raise NotImplementedError

    def __imatmul__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(other.real @ self.real - other.imag @ self.imag,
                                 other.imag @ self.real + other.real @ self.imag)
        else:
            raise NotImplementedError

    def __repr__(self) -> str:
        return f'ComplexNumber(real={self.real.__repr__()}, imag={self.imag.__repr__()})'
    
    def fft2(self):
        return ComplexNumber(_fft2(self.value, inverse=False), is_concat_value=True)

    def ifft2(self):
        return ComplexNumber(_fft2(self.value, inverse=True), is_concat_value=True)


def polar(abs:jt.Var, angle: jt.Var) -> ComplexNumber:
    assert abs.shape == angle.shape
    return ComplexNumber(abs * angle.cos(),abs * angle.sin())

def view_as_complex(x: jt.Var) -> ComplexNumber:
    assert x.shape[-1] == 2
    return ComplexNumber(x[...,0],x[...,1])

def view_as_real(x: ComplexNumber) -> jt.Var:
    return jt.stack([x.value[...,0],x.value[...,1]],dim=-1)

# reference: https://github.com/pytorch/pytorch/blob/8ea5b572a63b1acc538a9fc8d3862c73739116e8/torch/functional.py#L1258
def tensordot(a, b, dims=2):
    r"""Returns a contraction of a and b over multiple dimensions.

    :attr:`tensordot` implements a generalized matrix product.

    Args:
      a (Tensor): Left tensor to contract
      b (Tensor): Right tensor to contract
      dims (int or Tuple[List[int], List[int]] or List[List[int]] containing two lists or Tensor): number of dimensions to
         contract or explicit lists of dimensions for :attr:`a` and
         :attr:`b` respectively

    When called with a non-negative integer argument :attr:`dims` = :math:`d`, and
    the number of dimensions of :attr:`a` and :attr:`b` is :math:`m` and :math:`n`,
    respectively, :func:`tensordot` computes

    .. math::
        r_{i_0,...,i_{m-d}, i_d,...,i_n}
          = \sum_{k_0,...,k_{d-1}} a_{i_0,...,i_{m-d},k_0,...,k_{d-1}} \times b_{k_0,...,k_{d-1}, i_d,...,i_n}.

    When called with :attr:`dims` of the list form, the given dimensions will be contracted
    in place of the last :math:`d` of :attr:`a` and the first :math:`d` of :math:`b`. The sizes
    in these dimensions must match.

    """
    if not isinstance(dims, (tuple, list, int)):
        raise RuntimeError(
            "tensordot expects dims to be int or "
            + "Tuple[List[int], List[int]] or "
            + "List[List[int]] containing two lists, but got "
            + f"dims={dims}"
        )

    dims_a, dims_b = [], []

    if isinstance(dims, (tuple, list)):
        dims_a, dims_b = dims

    if isinstance(dims, (int)):
        if dims < 0:
            raise RuntimeError(f"tensordot expects dims >= 0, but got dims={dims}")
        if dims > min(len(a.shape), len(b.shape)):
            raise RuntimeError(
                f"tensordot expects dims < ndim_a or ndim_b, but got dims={dims}"
            )
        dims_a = list(range(len(a.shape)-dims, len(a.shape)))
        dims_b = list(range(dims))

    # reference: https://github.com/pytorch/pytorch/blob/8ea5b572a63b1acc538a9fc8d3862c73739116e8/aten/src/ATen/native/Linear.cpp#L769
    def __tensordot_native(input1:jt.Var, input2:jt.Var, dims1, dims2):
        if not isinstance(dims1, (list, tuple)):
            raise RuntimeError("tensordot expects dims1 to be List[Int], but got dims={}".format(dims1))
        if not isinstance(dims2, (list, tuple)):
            raise RuntimeError("tensordot expects dims2 to be List[Int], but got dims={}".format(dims2))
        dims1 = list(dims1)
        dims2 = list(dims2)
        if len(dims1) != len(dims2):
            raise RuntimeError("both dimension lists should have the same length")
        if input1.dtype != input2.dtype:
            raise RuntimeError("both inputs should have the same dtype")
        t1 = input1
        t2 = input2
        csize = 1
        input1_bitmap = np.zeros(len(input1.shape), dtype='bool')
        input2_bitmap = np.zeros(len(input2.shape), dtype='bool')
        for i in range(len(dims1)):
            s1 = input1.shape[dims1[i]]
            s2 = input2.shape[dims2[i]]
            input1_bitmap[dims1] = True
            input2_bitmap[dims2] = True
            if s2 == 1:     #broadcasted dimensions can be summed right away
                t1 = t1.sum(dims1[i], keepdims=True)
            elif s1 == 1:
                t2 = t2.sum(dims2[i], keepdims=True)
            else:
                if s1 != s2:
                    raise RuntimeError("contracted dimensions need to match, but first has size {}, in dim {}, and second has size {}".format(s1, i, s2))
                csize *= s1

        p1, p2 = [], []     # p1, p2: input permutations
        rsizes = []
        size1, size2 = 1, 1     #  number of non-contracted elements
        for i in range(len(input1.shape)):
            if not input1_bitmap[i]:
                p1.append(i)
                size1 *= t1.shape[i]
                rsizes.append(t1.shape[i])
        p1 += dims1
        p2 += dims2
        for i in range(len(input2.shape)):
            if not input2_bitmap[i]:
                p2.append(i)
                size2 *= t2.shape[i]
                rsizes.append(t2.shape[i])

        # permute and reshape for matrix multiplication
        t1 = t1.permute(p1).reshape((size1, csize))
        t2 = t2.permute(p2).reshape((csize, size2))
        # multiply and reshape to target size
        return jt.matmul(t1, t2).reshape(rsizes)

    return __tensordot_native(a, b, dims_a, dims_b)

# reference: https://github.com/pytorch/pytorch/blob/5ed3b70d09a4ab2a5be4becfda9dd0d3e3227c39/aten/src/ATen/native/LinearAlgebra.cpp#L3375
def kron(a:jt.Var, b:jt.Var):
    a_dim, b_dim = len(a.shape), len(b.shape)
    max_dim = max(a_dim, b_dim)
    pad_a, pad_b = max_dim-a_dim, max_dim-b_dim
    a_reshape, b_reshape = [], []
    result_reshape = []
    for i in range(max_dim):
        a_2i_shape = a.shape[i - pad_a] if i >= pad_a else 1
        b_2i1_shape = b.shape[i - pad_b] if i >= pad_b else 1
        a_reshape.append(a_2i_shape)
        a_reshape.append(1)
        b_reshape.append(1)
        b_reshape.append(b_2i1_shape)
        result_reshape.append(a_2i_shape * b_2i1_shape)
    a = a.reshape(a_reshape)
    b = b.reshape(b_reshape)
    return (a * b).reshape(result_reshape)

def one_hot(x: jt.Var, num_classes: int=-1) -> jt.Var:
    ''' Returns the one_hot encoding of inputs.

    :param x: class values of any shape
    :type x: jt.Var with bool or integer dtype

    :param num_classes: Total number of classes. If set to -1, the number of classes will be inferred as one greater than the largest class value in the input tensor.
    :type num_classes: int, optional

    :return: a Var with one more dimension with 1 values at the index 
    of last dimension indicated by the input, and 0 everywhere else.
    :rtype: jt.Var

    .. note::
        if the values in x are greater than num_class or less than 0, 
        the returned one_hot will be all zeros.

    Example:
        >>> jt.nn.one_hot(jt.arange(5) % 3)
            jt.Var([[1 0 0]
                [0 1 0]
                [0 0 1]
                [1 0 0]
                [0 1 0]], dtype=int32)
        >>> jt.nn.one_hot(jt.arange(5) % 3, num_classes=5)
            jt.Var([[1 0 0 0 0]
                [0 1 0 0 0]
                [0 0 1 0 0]
                [1 0 0 0 0]
                [0 1 0 0 0]], dtype=int32)
        >>> jt.nn.one_hot(jt.arange(6).reshape(3,2) % 3)
            jt.Var([[[1 0 0]
                [0 1 0]]

                [[0 0 1]
                [1 0 0]]

                [[0 1 0]
                [0 0 1]]], dtype=int32)
    '''

    assert x.dtype in [jt.bool, jt.int8, jt.int16, jt.int32, jt.int64, jt.uint8, jt.uint16, jt.uint32, jt.uint64]
    if num_classes == -1:
        num_classes = x.max().item() + 1

    N = len(x.shape)
    indices = ["i"+str(i) for i in range(N)]
    y = jt.ones_like(x).reindex(
        x.shape + [num_classes],
        indices, 
        extras=[x],
        overflow_conditions=[f"i{N} != @e0({','.join(indices)})"],
        overflow_value=0)
    return y


class KLDivLoss(Module):
    ''' Computes the Kullback-Leibler divergence loss.
    '''

    def __init__(self, reduction: str = 'mean', log_target: bool = False):
        '''
            :param reduction: Specifies the reduction to apply to the output. Can be 'mean', 'sum', 'batchmean', or 'none'. Defaults to 'mean'.
            :type reduction: str, optional
            :param log_target: Specifies whether target is the log space. Defaults to False.
            :type log_target: bool, optional
        '''
        self.reduction = reduction
        self.log_target = log_target

    def execute(self, input: jt.Var, target: jt.Var) -> jt.Var:
        if not self.log_target:
            loss_pointwise = target * (target.log() - input)
        else:
            loss_pointwise = target.exp() * (target - input)

        if self.reduction == "mean":
            loss = loss_pointwise.mean()
        elif self.reduction == "batchmean":
            loss = loss_pointwise.sum() / input.size(0)
        elif self.reduction == "sum":
            loss = loss_pointwise.sum()
        else:
            loss = loss_pointwise
        return loss

class Mish(Module):
    def __init__(self, inplace=False):
        '''
Applies the Mish function, element-wise.
reference: Mish - A Self Regularized Non-Monotonic Neural Activation Function.
        '''
        pass
    def execute(self, x):
        return x * jt.tanh(jt.softplus(x))

def mish(x, inplace=False):
    return x * jt.tanh(jt.nn.softplus(x))

def skip_init(module_cls, *args, **kw):
    return module_cls(*args, **kw)

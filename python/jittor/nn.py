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
import os
from collections import OrderedDict
from jittor.pool import *
from jittor.optim import *
from jittor.misc import _pair, _triple
# torch exposes the CTCLoss module under torch.nn; jittor defines it in
# jittor.misc, so re-export the CLASS for `nn.CTCLoss` / the shim's
# `torch.nn.CTCLoss` to resolve. We deliberately do NOT re-export the functional
# `ctc_loss`: torch_compat installs its own torch-faithful F.ctc_loss only when
# F lacks one, and misc.ctc_loss's reduction='mean' differs from torch (torch
# divides each sample by its target_length) — re-exporting it would shadow the
# correct F.ctc_loss.
from jittor.misc import CTCLoss
from jittor_utils import LOG
from functools import partial

from ._nn.runtime import bind_runtime as _bind_nn_runtime

_bind_nn_runtime(jt)
del _bind_nn_runtime

from ._nn.activations import (
    elu, gelu, hardsigmoid, hardswish, leaky_relu, prelu, relu, relu6,
    rrelu, sigmoid, sign, silu,
)
from ._nn.convolution import conv1d, conv2d, conv3d
from ._nn.convolution_3d_layers import Conv3d
from ._nn.convolution_cudnn import (
    _CUDNN_3D_HALF_DTYPES, _CudnnConv2d, _CudnnConvT2d,
    _cudnn_conv3d_fp16_safe, _try_cudnn_conv2d,
    _try_cudnn_conv_transpose2d,
)
from ._nn.convolution_layers import Conv, Conv1d
from ._nn.convolution_transpose import (
    conv_transpose, conv_transpose1d, conv_transpose3d,
)
from ._nn.convolution_transpose_layers import ConvTranspose, ConvTranspose3d
from ._nn.losses import (
    bce_loss, binary_cross_entropy_with_logits, cross_entropy_loss, l1_loss,
    mse_loss, nll_loss, smooth_l1_loss,
)
from ._nn.layer_norm_cuda import _layer_norm_no_grad_cuda
from ._nn.normalization import (
    _ln_function_cls, _ln_normalize, batch_norm, group_norm, instance_norm,
)
from ._nn.padding import (
    ConstantPad1d, ConstantPad2d, ConstantPad3d, ReflectionPad2d,
    ReplicationPad2d, ZeroPad2d, pad,
)
from ._nn.recurrent_base import RNNBase
from ._nn.recurrent_cells import GRUCell, LSTMCell, RNNCell
from ._nn.recurrent_layers import GRU, LSTM, RNN
from ._nn.softmax import (
    _get_softmax_dim, log_sigmoid, log_softmax, logsumexp, softmax,
)
from ._nn.vector import (
    cosine_similarity, glu, normalize, pairwise_distance, softsign,
)


def _broadcast_batch_dims(a, b):
    ''' Broadcast the leading batch dims of two tensors with equal ndim>=3 to a
    common shape (torch matmul/bmm semantics), leaving the trailing two (matrix)
    dims untouched. cublasGemmStridedBatchedEx only supports a single batch
    stride per operand, so a batch dim of size 1 broadcast against >1 (e.g.
    Falcon multi-query attention: [b,nh,q,d] @ [b,1,d,k]) must be materialized
    here before dispatch. '''
    if a.ndim != b.ndim or a.ndim < 3:
        return a, b
    bshape = []
    need = False
    for i in range(a.ndim - 2):
        an, bn = a.shape[i], b.shape[i]
        if an != bn:
            assert an == 1 or bn == 1, \
                f"dimension not match, a.shape:{a.shape}, b.shape:{b.shape}"
            need = True
        bshape.append(max(an, bn))
    if not need:
        return a, b
    if list(a.shape[:-2]) != bshape:
        a = a.expand(bshape + list(a.shape[-2:]))
    if list(b.shape[:-2]) != bshape:
        b = b.expand(bshape + list(b.shape[-2:]))
    return a, b


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
    a_dtype, b_dtype = str(a.dtype), str(b.dtype)
    if (jt.flags.use_cuda and jt.compile_extern.cublas_ops
            and a_dtype == b_dtype and "float" in a_dtype
            and "complex" not in a_dtype and "complex" not in b_dtype):
        if a_dtype == "float64":
            r = jt.compile_extern.cublas_ops.cublas_matmul(a.float32(), b.float32(), 0, 1)
            return r.cast("float64")
        return jt.compile_extern.cublas_ops.cublas_matmul(a, b, 0, 1)

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
        a, b = _broadcast_batch_dims(a, b)
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

def _matmul_2d_cublas(a, b, trans_a=0, trans_b=0):
    a_dtype, b_dtype = str(a.dtype), str(b.dtype)
    if (jt.flags.use_cuda and jt.compile_extern.cublas_ops
            and a_dtype == b_dtype and "float" in a_dtype
            and "complex" not in a_dtype and "complex" not in b_dtype):
        if a_dtype == "float64":
            r = jt.compile_extern.cublas_ops.cublas_matmul(
                a.float32(), b.float32(), trans_a, trans_b)
            return r.cast("float64")
        return jt.compile_extern.cublas_ops.cublas_matmul(a, b, trans_a, trans_b)
    return None

def _transpose_base_last2(x):
    try:
        base = getattr(x, "_jittor_transpose_base", None)
        if base is not None and getattr(x, "_jittor_transpose_last2", False):
            return base
    except Exception:
        pass
    return None

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
        if len_a == 2 and len_b == 2:
            # a: [n, m], b: [m, k], c: [n, k]
            a_base = _transpose_base_last2(a)
            b_base = _transpose_base_last2(b)
            aa = a_base if a_base is not None else a
            bb = b_base if b_base is not None else b
            fast = _matmul_2d_cublas(aa, bb, 1 if a_base is not None else 0,
                                     1 if b_base is not None else 0)
            if fast is not None:
                return fast
        if len_a>=3 and len_a==len_b:
            # bmm
            # a: [..., n, m], b: [..., m, k], c:[..., n, k]
            # cublas_batched_matmul only supports float dtypes; complex64 falls through to
            # the reindex path below (broadcast * multiply + sum-reduce), which the native
            # complex kernels support on both CPU and CUDA.
            if jt.flags.use_cuda and jt.compile_extern.cublas_ops and "complex" not in str(a.dtype):
                a_base = _transpose_base_last2(a)
                b_base = _transpose_base_last2(b)
                if a_base is not None:
                    a = a_base
                if b_base is not None:
                    b = b_base
                a, b = _broadcast_batch_dims(a, b)
                # cuBLAS strided-batched gemm rejects float64 (CUBLAS_STATUS_NOT_SUPPORTED)
                # on many GPUs; compute in float32 and cast back (rare path, e.g. a float64
                # attention mask contaminating a transformer's batched matmul).
                if str(a.dtype) == "float64" or str(b.dtype) == "float64":
                    r = jt.compile_extern.cublas_ops.cublas_batched_matmul(
                        a.float32(), b.float32(),
                        1 if a_base is not None else 0,
                        1 if b_base is not None else 0)
                    return r.cast("float64") if (str(a.dtype) == "float64" and str(b.dtype) == "float64") else r
                return jt.compile_extern.cublas_ops.cublas_batched_matmul(
                    a, b,
                    1 if a_base is not None else 0,
                    1 if b_base is not None else 0)
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

jt.Var.prelu = prelu

jt.Var.hardswish = hardswish

jt.Var.hardsigmoid = hardsigmoid

jt.Var.rrelu = rrelu

class RReLU(Module):
    ''' Applies the randomized leaky rectified linear unit function,
    element-wise. See :func:`rrelu`.

    :param lower: lower bound of the uniform slope. Default: 1/8
    :param upper: upper bound of the uniform slope. Default: 1/3
    '''
    def __init__(self, lower=1./8, upper=1./3):
        self.lower = lower
        self.upper = upper
        self.is_train = True

    def execute(self, x):
        return rrelu(x, self.lower, self.upper, getattr(self, "is_train", True))

class Hardswish(Module):
    ''' Applies the element-wise Hardswish function. See :func:`hardswish`. '''
    def execute(self, x):
        return hardswish(x)

class Hardsigmoid(Module):
    ''' Applies the element-wise Hardsigmoid function. See :func:`hardsigmoid`. '''
    def execute(self, x):
        return hardsigmoid(x)

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
    
class CrossEntropyLoss(Module):
    def __init__(self, weight=None, ignore_index=None, reduction='mean'):
        # torch.nn.CrossEntropyLoss takes a `reduction` arg ('mean'/'sum'/'none');
        # it was silently dropped before, so reduction='sum'/'none' had no effect.
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def execute(self, output, target):
        return cross_entropy_loss(output, target, self.weight, self.ignore_index,
                                  reduction=self.reduction)

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


class BCEWithLogitsLoss(Module):
    def __init__(self, weight=None, pos_weight=None, size_average=True, reduction=None):
        self.pos_weight = pos_weight
        self.weight = weight
        self.size_average = size_average
        self.reduction = reduction

    def execute(self, output, target):
        return binary_cross_entropy_with_logits(output,target,self.weight,self.pos_weight,self.size_average,self.reduction)

jt.Var.softmax = softmax

jt.Var.log_softmax = log_softmax

jt.Var.log_sigmoid = log_sigmoid

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

def dropout(x,p=0.5,is_train=False,training=None):
    if training is not None:
        is_train = training
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


def multi_head_attention_forward(query, key, value, embed_dim_to_check, num_heads,
        in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn, dropout_p,
        out_proj_weight, out_proj_bias, training=True, key_padding_mask=None,
        need_weights=True, attn_mask=None, use_separate_proj_weight=False,
        q_proj_weight=None, k_proj_weight=None, v_proj_weight=None, static_k=None,
        static_v=None, average_attn_weights=True, is_causal=False):
    ''' torch's F.multi_head_attention_forward (the functional behind nn.MultiheadAttention
    and used directly by fairseq-style models, e.g. wavlm). query/key/value are
    (L, N, E) = (seq, batch, embed). Returns (attn_output (L,N,E), attn_weights or None).
    Masked positions use a large finite negative (not -inf) to avoid jittor's inf/nan
    JIT codegen segfault; softmax drives them to ~0 identically to torch. '''
    tgt_len, bsz, embed_dim = query.shape
    head_dim = embed_dim // num_heads
    scaling = float(head_dim) ** -0.5
    NEG = -1e30                                                # finite "-inf" for masks

    # q/k/v projections (separate weights or a fused in_proj_weight)
    if use_separate_proj_weight:
        b = in_proj_bias
        bq = b[:embed_dim] if b is not None else None
        bk = b[embed_dim:embed_dim*2] if b is not None else None
        bv = b[embed_dim*2:] if b is not None else None
        q = linear(query, q_proj_weight, bq)
        k = linear(key,   k_proj_weight, bk)
        v = linear(value, v_proj_weight, bv)
    else:
        w_q = in_proj_weight[:embed_dim]
        w_k = in_proj_weight[embed_dim:embed_dim*2]
        w_v = in_proj_weight[embed_dim*2:]
        if in_proj_bias is not None:
            bq = in_proj_bias[:embed_dim]; bk = in_proj_bias[embed_dim:embed_dim*2]; bv = in_proj_bias[embed_dim*2:]
        else:
            bq = bk = bv = None
        q = linear(query, w_q, bq); k = linear(key, w_k, bk); v = linear(value, w_v, bv)
    q = q * scaling

    if static_k is not None: k = static_k
    if static_v is not None: v = static_v

    # optional bias_k / bias_v: append a learned key/value
    if bias_k is not None and bias_v is not None:
        k = jt.concat([k, bias_k.repeat(1, bsz, 1)], dim=0)
        v = jt.concat([v, bias_v.repeat(1, bsz, 1)], dim=0)
        if attn_mask is not None:
            attn_mask = jt.concat([attn_mask, jt.zeros((*attn_mask.shape[:-1], 1), attn_mask.dtype)], dim=-1)
        if key_padding_mask is not None:
            key_padding_mask = jt.concat([key_padding_mask, jt.zeros((key_padding_mask.shape[0], 1), key_padding_mask.dtype)], dim=1)

    # (L, N, E) -> (N*H, L, head_dim)
    q = q.reshape(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.reshape(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.reshape(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if add_zero_attn:
        z = jt.zeros((k.shape[0], 1, k.shape[2]), k.dtype)
        k = jt.concat([k, z], dim=1)
        v = jt.concat([v, z], dim=1)
        if attn_mask is not None:
            attn_mask = jt.concat([attn_mask, jt.zeros((*attn_mask.shape[:-1], 1), attn_mask.dtype)], dim=-1)
        if key_padding_mask is not None:
            key_padding_mask = jt.concat([key_padding_mask, jt.zeros((key_padding_mask.shape[0], 1), key_padding_mask.dtype)], dim=1)

    src_len = k.shape[1]
    attn = jt.matmul(q, k.transpose(1, 2))                    # (N*H, L, S)

    if attn_mask is not None:
        # float mask -> additive bias; bool mask -> fill masked with NEG
        if str(attn_mask.dtype) == "bool":
            attn = attn + jt.ternary(attn_mask, jt.array(NEG).cast(attn.dtype).broadcast(attn_mask.shape), jt.zeros(attn_mask.shape, attn.dtype))
        else:
            attn = attn + attn_mask                           # broadcasts over the N*H dim

    if key_padding_mask is not None:
        attn = attn.reshape(bsz, num_heads, tgt_len, src_len)
        kpm = (key_padding_mask != 0).reshape(bsz, 1, 1, src_len).broadcast([bsz, num_heads, tgt_len, src_len])
        attn = jt.ternary(kpm, jt.array(NEG).cast(attn.dtype).broadcast(attn.shape), attn)
        attn = attn.reshape(bsz * num_heads, tgt_len, src_len)

    attn = softmax(attn, dim=-1)
    if dropout_p > 0.0 and training:
        attn = dropout(attn, p=dropout_p, is_train=True)
    if str(attn.dtype) != str(v.dtype):       # a float64 attn_mask can promote attn
        attn = attn.cast(v.dtype)
    out = jt.matmul(attn, v)                                  # (N*H, L, head_dim)
    out = out.transpose(0, 1).reshape(tgt_len, bsz, embed_dim)
    out = linear(out, out_proj_weight, out_proj_bias)

    attn_weights = None
    if need_weights:
        attn_weights = attn.reshape(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_weights = attn_weights.mean(dim=1)
    return out, attn_weights

class BatchNorm(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, is_train=True, sync=True,
                 track_running_stats=True, device=None, dtype=None):
        # track_running_stats/device/dtype accepted for torch.nn.BatchNorm* compat.
        self.sync = sync
        self.num_features = num_features
        self.is_train = is_train
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.weight = init.constant((num_features,), "float32", 1.0) if affine else 1.0
        self.bias = init.constant((num_features,), "float32", 0.0) if affine else 0.0
        self.running_mean = init.constant((num_features,), "float32", 0.0).stop_grad()
        self.running_var = init.constant((num_features,), "float32", 1.0).stop_grad()
        # torch keeps num_batches_tracked as a buffer; some models read it.
        self.num_batches_tracked = init.constant((1,), "int32", 0.0).stop_grad()
        # running stats are BUFFERS, not trainable params (torch semantics):
        # exclude them from parameters()/named_parameters() so optimizers and the
        # autograd bridge don't treat them as trainable.
        for _b in (self.running_mean, self.running_var, self.num_batches_tracked):
            object.__setattr__(_b, "is_buffer", True)
        # num_batches_tracked is a non-numeric counter; torch stores it as a 0-d
        # scalar but jittor has no 0-d tensors (it's (1,)), which mismatches on a
        # state_dict roundtrip. Keep it non-persistent so it isn't serialized.
        object.__setattr__(self.num_batches_tracked, "persistent", False)

    def execute(self, x):
        dims = [0]+list(range(2,x.ndim))
        if self.is_train:
            xmean = jt.mean(x, dims=dims)
            x2mean = jt.mean(x*x, dims=dims)
            sync = self.sync and jt.in_mpi
            if sync:
                xmean = xmean.mpi_all_reduce("mean")
                x2mean = x2mean.mpi_all_reduce("mean")

            xvar = (x2mean-xmean*xmean).maximum(0.0)
            if sync:
                # SyncBatchNorm: stats are cross-rank, so normalize with the composite
                # form (the stable _ln_normalize helper only sees local data and would
                # break sync semantics). Precision cost is the small-variance backround
                # cancellation, accepted to preserve correctness across ranks.
                w = self.weight / jt.sqrt(xvar+self.eps)
                b = self.bias - xmean * w
                norm_x = x * w.broadcast(x, dims) + b.broadcast(x, dims)
            else:
                # local stats: use the numerically-stable custom-backward normalization
                # (see _ln_normalize) — avoids the E[x^2]-E[x]^2 fp32 cancellation that
                # corrupts the backward for small-variance batches; affine applied after.
                xhat = _ln_normalize(x, dims, self.eps)
                if self.affine:
                    sh = [1, self.num_features] + [1]*(x.ndim-2)
                    norm_x = xhat * self.weight.reshape(sh) + self.bias.reshape(sh)
                else:
                    norm_x = xhat

            self.running_mean.update(self.running_mean +
                (xmean.reshape((-1,)) - self.running_mean) * self.momentum)
            # torch updates running_var with the UNBIASED (Bessel-corrected) batch
            # variance (var * n/(n-1)) while normalizing with the biased one; match it
            # so running stats (hence eval-mode outputs) align with torch. n = count
            # reduced per channel (global across ranks in sync mode).
            n = 1
            for _d in dims:
                n *= x.shape[_d]
            if sync:
                n *= jt.world_size
            run_var = xvar * (n / (n - 1)) if n > 1 else xvar
            self.running_var.update(self.running_var +
                (run_var.reshape((-1,))-self.running_var)*self.momentum)
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
        # Per-(N,C) normalization over spatial dims with a numerically-stable custom
        # backward (see _ln_normalize) — the composite E[x^2]-E[x]^2 form loses float32
        # precision in backward for small-variance inputs (same cancellation as LayerNorm).
        dims = list(range(2,x.ndim))
        xhat = _ln_normalize(x, dims, self.eps)
        if not self.affine:
            return xhat
        sh = [1, self.num_features] + [1]*len(dims)
        return xhat * self.weight.reshape(sh) + self.bias.reshape(sh)

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

class LayerNorm(Module):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True, bias: bool = True, device=None, dtype=None) -> None:
        # device/dtype: torch's LayerNorm accepts them (factory kwargs); jittor places
        # params on the active device and uses float32, so they're accepted and ignored
        # (nemotron passes them positionally).
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = init.constant(normalized_shape, "float32", 1.0) if elementwise_affine else 1.0
        # torch 2.1+ adds `bias`: a learnable bias only when both affine AND bias are on
        # (e.g. dbrx uses LayerNorm(..., bias=False) -> scale only, no shift param).
        self.bias = init.constant(normalized_shape, "float32", 0.0) if (elementwise_affine and bias) else 0.0

    @fp32_guard
    def execute(self, x):
        dims = [-i for i in range(len(self.normalized_shape), 0, -1)]
        # out = weight*(x-mean)/sqrt(var+eps) + bias. Normalization has a stable custom
        # backward (see _ln_normalize); the affine stays composite (no cancellation).
        # torch's LayerNorm/F.layer_norm accept weight=None / bias=None (MPT sets
        # norm.bias = None for Hub-weight compat). Treat None as identity/zero.
        weight = 1.0 if self.weight is None else self.weight
        bias = 0.0 if self.bias is None else self.bias
        fast = _layer_norm_no_grad_cuda(x, self.normalized_shape, weight, bias, self.eps)
        if fast is not None:
            return fast
        xhat = _ln_normalize(x, dims, self.eps)
        return xhat * weight + bias


LayerNorm3d = LayerNorm2d = LayerNorm1d = LayerNorm

@fp32_guard
def layer_norm(x, 
    normalized_shape, 
    weight = 1,
    bias = 0,
    eps: float = 1e-5, 
    elementwise_affine: bool = True):
    dims = [-i for i in range(len(normalized_shape), 0, -1)]
    weight = 1.0 if weight is None else weight
    bias = 0.0 if bias is None else bias
    fast = _layer_norm_no_grad_cuda(x, tuple(normalized_shape), weight, bias, eps)
    if fast is not None:
        return fast
    xhat = _ln_normalize(x, dims, eps)   # stable custom backward, see LayerNorm.execute
    return xhat * weight + bias

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
        assert C % self.num_groups == 0, \
            f"GroupNorm: num_channels ({C}) must be divisible by num_groups ({self.num_groups})"
        # Per-(N,group) normalization with a numerically-stable custom backward (see
        # _ln_normalize) — the composite E[x^2]-E[x]^2 form loses float32 precision in
        # backward for small-variance inputs (same cancellation as LayerNorm). Affine is
        # per-channel and applied after restoring shape (no cancellation).
        xg = x.reshape((N, self.num_groups, C//self.num_groups, -1))
        xhat = _ln_normalize(xg, [2,3], self.eps).reshape(output_shape)
        if not self.affine:
            return xhat
        sh = [1, C] + [1]*(x.ndim-2)
        return xhat * self.weight.reshape(sh) + self.bias.reshape(sh)

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

Conv2d = Conv

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

conv = conv2d

ConvTranspose2d = ConvTranspose

conv_transpose2d = conv_transpose

conv_transpose2d = conv_transpose

def adaptive_avg_pool2d(input, output_size):
    ''' Applies a 2D adaptive average pooling over an input signal composed of
    several input planes. Torch-compatible functional interface that reuses the
    :class:`AdaptiveAvgPool2d` module implementation.

    :param input: the input var of shape ``(N, C, H, W)``
    :type input: jt.Var

    :param output_size: the target output size ``(H_out, W_out)``. A single int
        ``H_out`` is interpreted as ``(H_out, H_out)``; ``None`` keeps that
        dimension unchanged.
    :type output_size: int or tuple

    Example:
        >>> x = jt.randn(2, 3, 10, 12)
        >>> y = nn.adaptive_avg_pool2d(x, (5, 6))
    '''
    return AdaptiveAvgPool2d(output_size)(input)


# ---------------------------------------------------------------------------
# torch-grade overrides for the average-pooling family.
#
# ``jittor.pool`` (imported above via ``from jittor.pool import *``) ships
# correct ``MaxPool*`` and an ``AvgPool``/``AdaptiveAvgPool2d`` that match
# PyTorch only in the easy cases.  Two documented torch behaviours were missing:
#
#   1. ``avg_pool2d(..., count_include_pad=False)`` -- pool.py's mean path divides
#      every window by ``kernel_size`` regardless of the flag (verified: incl == excl
#      bit-for-bit), so padded borders use the wrong denominator.  torch divides by
#      the count of *real* (in-bounds) input elements when count_include_pad=False.
#   2. ``AdaptiveAvgPool2d`` with a non-divisor output (e.g. 8 -> 3) -- pool.py uses a
#      single uniform stride/kernel, whereas torch uses variable-width overlapping
#      bins ``[floor(i*H/O), ceil((i+1)*H/O))``.  These agree only when O | H.
#
# Both are fixed below in pure jittor (reindex + reduce), so forward AND backward
# stay differentiable and run identically on CPU and CUDA.  The implementations are
# numpy/torch-formula validated in test_torch_compat_pool_parity.py.  The same gaps
# remain in jittor.pool for the top-level ``jt.AvgPool2d`` / ``jt.AdaptiveAvgPool2d``
# symbols; only the ``nn``-surface names are corrected here.
# ---------------------------------------------------------------------------
class AvgPool2d(Module):
    '''2D average pooling, torch-compatible (N,C,H,W) -> (N,C,Hout,Wout).

    Unlike ``jittor.pool.AvgPool2d`` this honours ``count_include_pad`` exactly as
    PyTorch documents it: when ``True`` (default) padded zeros are counted in the
    averaging denominator; when ``False`` only real input elements are.  ``ceil_mode``
    overshoot beyond the input is never counted as padding (matches torch).
    '''
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def execute(self, x):
        kh, kw = _pair(self.kernel_size)
        sh, sw = _pair(self.stride)
        ph, pw = _pair(self.padding)
        N, C, H, W = x.shape
        if self.ceil_mode:
            Ho = (H + 2 * ph - kh + sh - 1) // sh + 1
            Wo = (W + 2 * pw - kw + sw - 1) // sw + 1
            # torch drops the last window if it would start in the right padding.
            if (Ho - 1) * sh >= H + ph:
                Ho -= 1
            if (Wo - 1) * sw >= W + pw:
                Wo -= 1
        else:
            Ho = (H + 2 * ph - kh) // sh + 1
            Wo = (W + 2 * pw - kw) // sw + 1
        idx = ["i0", "i1", f"i2*{sh}+i4-{ph}", f"i3*{sw}+i5-{pw}"]
        summed = x.reindex([N, C, Ho, Wo, kh, kw], idx,
                           overflow_value=0.0).reduce("add", [4, 5])
        # Fast path: no padding and no ceil overshoot -> every window is full kh*kw.
        if self.count_include_pad and ph == 0 and pw == 0 and not self.ceil_mode:
            return summed / (kh * kw)
        i2 = jt.index((Ho,), dim=0).reshape(Ho, 1).float32()
        i3 = jt.index((Wo,), dim=0).reshape(1, Wo).float32()
        if self.count_include_pad:
            # Divisor = window area clamped to the *padded* input [-pad, dim+pad);
            # ceil_mode overshoot past dim+pad is excluded (torch semantics).
            h_lo = (i2 * sh - ph).maximum(-float(ph))
            h_hi = (i2 * sh - ph + kh).minimum(float(H + ph))
            w_lo = (i3 * sw - pw).maximum(-float(pw))
            w_hi = (i3 * sw - pw + kw).minimum(float(W + pw))
        else:
            # Divisor = window area clamped to the *real* input [0, dim).
            h_lo = (i2 * sh - ph).maximum(0.0)
            h_hi = (i2 * sh - ph + kh).minimum(float(H))
            w_lo = (i3 * sw - pw).maximum(0.0)
            w_hi = (i3 * sw - pw + kw).minimum(float(W))
        denom = ((h_hi - h_lo) * (w_hi - w_lo)).reshape(1, 1, Ho, Wo)
        return summed / denom


def avg_pool2d(x, kernel_size, stride=None, padding=0, ceil_mode=False,
               count_include_pad=True):
    '''Functional 2D average pooling, torch-compatible (see :class:`AvgPool2d`).'''
    return AvgPool2d(kernel_size, stride, padding, ceil_mode, count_include_pad)(x)


class AdaptiveAvgPool2d(Module):
    '''2D adaptive average pooling, torch-compatible (N,C,H,W) -> (N,C,Oh,Ow).

    Uses torch's variable-width overlapping bins
    ``hstart=floor(i*H/Oh)``, ``hend=ceil((i+1)*H/Oh)`` (and likewise for W) and
    divides by the real bin size, so it matches PyTorch even when the output size
    does not divide the input size (the common diffusers / classifier-head case).
    '''
    def __init__(self, output_size):
        self.output_size = output_size

    def execute(self, x):
        if isinstance(self.output_size, int):
            oh = ow = self.output_size
        elif hasattr(self.output_size, "__len__") and not isinstance(self.output_size, str):
            # tuple / list / jittor NanoVector (e.g. x.shape[2:] from a semantic head)
            oh = x.shape[2] if self.output_size[0] is None else int(self.output_size[0])
            ow = x.shape[3] if self.output_size[1] is None else int(self.output_size[1])
        else:
            raise TypeError(f"AdaptiveAvgPool2d only support int, tuple or list "
                            f"input. Not support {type(self.output_size)} yet.")
        N, C, H, W = x.shape
        if oh == 1 and ow == 1:
            return x.reduce("mean", [2, 3], keepdims=True)
        yy, xx = jt.meshgrid(jt.arange(0, oh, 1), jt.arange(0, ow, 1))   # (oh, ow)
        startH = jt.floor(yy * H / oh).int32()
        endH = jt.ceil((yy + 1) * H / oh).int32()
        startW = jt.floor(xx * W / ow).int32()
        endW = jt.ceil((xx + 1) * W / ow).int32()
        maxH = int(jt.max(endH - startH).data)
        maxW = int(jt.max(endW - startW).data)
        pixel_count = (endH - startH) * (endW - startW)
        out = x.reindex(
            [N, C, oh, ow, maxH, maxW],
            ["i0", "i1", "@e0(i2, i3) + i4", "@e2(i2, i3) + i5"],
            extras=[startH, endH, startW, endW],
            overflow_conditions=["i4 >= @e1(i2, i3) - @e0(i2, i3)",
                                 "i5 >= @e3(i2, i3) - @e2(i2, i3)"],
            overflow_value=0)
        return out.reduce("sum", [4, 5]) / pixel_count[None, None, ...]



class GLU(Module):
    r''' Applies the gated linear unit function. See :func:`glu`.

    :param dim: the dimension on which to split the input. Default: -1
    :type dim: int

    Example:
        >>> m = nn.GLU()
        >>> x = jt.randn(4, 6)
        >>> y = m(x)   # y.shape == [4, 3]
    '''
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def execute(self, x):
        return glu(x, self.dim)

class Softsign(Module):
    r''' Applies the element-wise SoftSign function. See :func:`softsign`.

    Example:
        >>> m = nn.Softsign()
        >>> x = jt.randn(3)
        >>> y = m(x)
    '''
    def __init__(self):
        super().__init__()

    def execute(self, x):
        return softsign(x)

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
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 dtype="float32", max_norm=None, norm_type=2.0,
                 scale_grad_by_freq=False, sparse=False, _weight=None,
                 _freeze=False, device=None):
        # torch.nn.Embedding-compatible signature. max_norm/norm_type/
        # scale_grad_by_freq/sparse/device are accepted for API parity (they
        # don't affect forward numerics here); _weight provides an initial weight;
        # _freeze makes the embedding non-trainable (e.g. Pegasus sinusoidal
        # positions). `dtype` stays the 4th positional for jittor backward-compat.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        if dtype is None:
            dtype = "float32"
        elif not isinstance(dtype, str):
            dtype = str(dtype).replace("torch.", "") or "float32"
        if _weight is not None:
            self.weight = _weight if isinstance(_weight, jt.Var) else jt.array(_weight)
        else:
            self.weight = jt.init.gauss([self.num_embeddings, self.embedding_dim], dtype)
            if padding_idx is not None:
                self.weight[padding_idx] = 0
        if _freeze:
            self.weight = self.weight.stop_grad()

    def execute(self, x):
        res = self.weight[x]
        if self.padding_idx is not None:
            # torch parity: the padding_idx row is frozen (its gradient is zeroed,
            # so it never trains). The padding row only receives gradient from
            # positions where x==padding_idx, so block the gradient there while
            # keeping forward values intact. Multiply-mask (NOT ternary: jittor's
            # ternary with a stop_grad branch zeroes the whole tensor's grad).
            keep = (x != self.padding_idx).unsqueeze(-1).float32()
            res = res * keep + (res * (1.0 - keep)).stop_grad()
        return res

def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0,
              scale_grad_by_freq=False, sparse=False):
    # Full torch F.embedding signature (ibert's quantized embedding passes all 7
    # positionally). scale_grad_by_freq / sparse only affect gradient bookkeeping,
    # not forward values, so they're accepted and ignored. max_norm renormalizes
    # rows whose p-norm exceeds the bound (rare; None is the hot path). padding_idx
    # freezes the padding row's gradient (torch parity) -- see Embedding.execute.
    if max_norm is not None:
        pn = (weight.abs() ** norm_type).sum(dim=-1, keepdims=True) ** (1.0 / norm_type)
        weight = weight * (jt.minimum(pn, max_norm) / (pn + 1e-12))
    res = weight[input]
    if padding_idx is not None:
        keep = (input != padding_idx).unsqueeze(-1).float32()
        res = res * keep + (res * (1.0 - keep)).stop_grad()
    return res

def embedding_bag(input, weight, offsets=None, mode="mean", per_sample_weights=None):
    ''' Computes sums, means or maxes of "bags" of embeddings, without
    instantiating the intermediate embeddings. Torch-compatible
    (functional form of :class:`EmbeddingBag`).

    :param input: indices into ``weight``. Either a 2-D var where every row is
        a bag of fixed length, or a 1-D var of concatenated bags together with
        ``offsets``.
    :param weight: the embedding matrix of shape ``(num_embeddings, embedding_dim)``.
    :param offsets: only used when ``input`` is 1-D. ``offsets[i]`` is the start
        index of the ``i``-th bag in ``input``.
    :param mode: one of ``"sum"``, ``"mean"`` or ``"max"``. Default: ``"mean"``.
    :param per_sample_weights: optional weights for a weighted ``"sum"`` (only
        valid when ``mode == "sum"``), same shape as ``input``.
    '''
    assert mode in ("sum", "mean", "max"), f"unsupported mode {mode} in embedding_bag"
    input = input if isinstance(input, jt.Var) else jt.array(input)
    if input.ndim == 1:
        assert offsets is not None, \
            "offsets has to be provided when input is 1-D in embedding_bag"
        offsets = offsets if isinstance(offsets, jt.Var) else jt.array(offsets)
        ends = jt.concat([offsets[1:], jt.array([input.shape[0]]).cast(offsets.dtype)], dim=0)
        bags = []
        n = offsets.shape[0]
        for i in range(n):
            s = int(offsets[i].item())
            e = int(ends[i].item())
            emb = weight[input[s:e]]
            if per_sample_weights is not None and mode == "sum":
                psw = per_sample_weights if isinstance(per_sample_weights, jt.Var) \
                    else jt.array(per_sample_weights)
                emb = emb * psw[s:e].reshape((-1, 1))
            if mode == "max":
                bag = emb.max(dim=0)
            elif mode == "mean":
                bag = emb.mean(dim=0)
            else:
                bag = emb.sum(dim=0)
            bags.append(bag.reshape((1, -1)))
        return jt.concat(bags, dim=0)
    else:
        assert input.ndim == 2, "input must be 1-D or 2-D in embedding_bag"
        emb = weight[input]  # (B, L, D)
        if per_sample_weights is not None and mode == "sum":
            psw = per_sample_weights if isinstance(per_sample_weights, jt.Var) \
                else jt.array(per_sample_weights)
            emb = emb * psw.reshape(psw.shape + (1,))
        if mode == "max":
            return emb.max(dim=1)
        elif mode == "mean":
            return emb.mean(dim=1)
        else:
            return emb.sum(dim=1)

class EmbeddingBag(Module):
    ''' Computes sums, means or maxes of "bags" of embeddings. See
    :func:`embedding_bag`.

    :param num_embeddings: size of the dictionary of embeddings.
    :param embedding_dim: the size of each embedding vector.
    :param mode: one of ``"sum"``, ``"mean"`` or ``"max"``. Default: ``"mean"``.
    '''
    def __init__(self, num_embeddings, embedding_dim, mode="mean", dtype="float32"):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.weight = jt.init.gauss([num_embeddings, embedding_dim], dtype)

    def execute(self, input, offsets=None, per_sample_weights=None):
        return embedding_bag(input, self.weight, offsets, self.mode, per_sample_weights)

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
                align_corners=self.align_corners)

class UpsamplingBilinear2d(Upsample):
    def __init__(self, scale_factor=None):
        # torch.nn.UpsamplingBilinear2d is documented as equivalent to
        # Upsample(mode='bilinear', align_corners=True) (it predates the 0.3.1
        # default flip to align_corners=False). The base Upsample defaults
        # align_corners=False, so it must be set True here for torch parity.
        Upsample.__init__(self, scale_factor, 'bilinear', align_corners=True)

class UpsamplingNearest2d(Upsample):
    def __init__(self, scale_factor=None):
        Upsample.__init__(self, scale_factor, 'nearest')

class Sequential(Module):
    def __init__(self, *args):
        self.layers = collections.OrderedDict()
        import types as _types_ml
        for mod in args:
            if mod is None:
                continue                       # torch: ModuleList(None) -> empty
            if isinstance(mod, collections.OrderedDict):
                for k, m in mod.items():
                    self.add_module(k, m)
            elif isinstance(mod, (list, tuple, _types_ml.GeneratorType)) or \
                    (hasattr(mod, "__iter__") and not isinstance(mod, Module)):
                # torch's ModuleList accepts ANY iterable of modules (incl. a
                # generator, e.g. DINO: ModuleList(build(l) for l in layers)).
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
        # torch's ModuleList stores None children (e.g. HRNet's _make_fuse_layers
        # appends None for the identity/same-resolution path and checks `is not None`
        # in forward). Accept None as a placeholder instead of asserting.
        if mod is None:
            self.layers[str(len(self.layers))] = None
            return self
        assert callable(mod), f"Module <{type(mod)}> is not callable"
        assert not isinstance(mod, type), f"Module is not a type"
        self.layers[str(len(self.layers))]=mod
        return self
    def extend(self, mods):
        # torch.nn.ModuleList.extend: append every module from an iterable
        # (mmdet PVT does `layers.extend([...])` when assembling backbone stages).
        for m in mods:
            self.append(m)
        return self
    def insert(self, index, mod):
        # torch.nn.ModuleList.insert: insert before `index`, shifting the
        # (string-keyed) tail. Rebuild the OrderedDict with contiguous int keys.
        assert callable(mod) and not isinstance(mod, type)
        vals = list(self.layers.values())
        n = len(vals)
        if index < 0:
            index += n
        index = max(0, min(index, n))
        vals.insert(index, mod)
        self.layers = collections.OrderedDict((str(i), v) for i, v in enumerate(vals))
        return self
    def add_module(self, name, mod):
        assert callable(mod), f"Module <{type(mod)}> is not callable"
        assert not isinstance(mod, type), f"Module is not a type"
        self.layers[str(name)]=mod

    def __len__(self):
        return len(self.layers)
    
    def named_children(self,):
        return list(self.layers.items())

    @property
    def _modules(self):
        return self.layers

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
    '''Torch-compatible Parameter wrapper.

    Jittor treats a Var assigned to a Module as a parameter, so wrapping an
    existing Var only needs to set the trainable flag. Do not clone here:
    PyTorch's Parameter is a lightweight wrapper over the supplied tensor data,
    while cloning can force materialization/JIT work and makes large pretrained
    model construction unnecessarily slow.
    '''
    if not isinstance(data, jt.Var):
        data = jt.array(data)
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
    # accept int OR (tuple/list) pairs -- torch passes lists, e.g. convbert's
    # nn.functional.unfold(kernel_size=[k, 1], padding=[(k-1)//2, 0]).
    _pair = lambda v: tuple(v) if isinstance(v, (tuple, list)) else (v, v)
    kernel_size = _pair(kernel_size)
    assert kernel_size[0] > 0 and kernel_size[1] > 0, "kernel size must be positive"
    dilation = _pair(dilation)
    assert dilation[0] > 0 and dilation[1] > 0, "dilation must be positive"
    padding = _pair(padding)
    assert padding[0] >= 0 and padding[1] >= 0, "padding must be non-negative"
    stride = _pair(stride)
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
    _pair = lambda v: tuple(v) if isinstance(v, (tuple, list)) else (v, v)
    kernel_size = _pair(kernel_size)
    assert kernel_size[0] > 0 and kernel_size[1] > 0, "kernel size must be positive"
    dilation = _pair(dilation)
    assert dilation[0] > 0 and dilation[1] > 0, "dilation must be positive"
    padding = _pair(padding)
    assert padding[0] >= 0 and padding[1] >= 0, "padding must be non-negative"
    stride = _pair(stride)
    assert stride[0] > 0 and stride[1] > 0, "stride must be positive"
    n,cl,num = X.shape
    area = kernel_size[0] * kernel_size[1]
    block_nums = []
    for i in range(2,4):
        block_nums.append((output_size[i-2]+2*padding[i-2]-dilation[i-2]*(kernel_size[i-2]-1)-1) // stride[i-2]+1)
    output = X.reindex_reduce("add",[n,cl // area,output_size[0]+2*padding[0],output_size[1]+2*padding[1]],["i0",f"i1/{area}",f"i2/{block_nums[1]}*{stride[0]}+(i1%{area})/{kernel_size[1]}*{dilation[0]}",f"i2%{block_nums[1]}*{stride[1]}+(i1%{area})%{kernel_size[1]}*{dilation[1]}"])
    return output[:,:,padding[0]:padding[0]+output_size[0],padding[1]:padding[1]+output_size[1]]


class Unfold(Module):
    ''' torch's nn.Unfold (im2col): extract sliding local blocks from a batched
    (N, C, H, W) input into (N, C*prod(kernel_size), L). Wraps the functional unfold.
    (convbert builds its span-based conv with nn.Unfold.) '''
    def __init__(self, kernel_size, dilation=1, padding=0, stride=1):
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def execute(self, x):
        return unfold(x, self.kernel_size, self.dilation, self.padding, self.stride)

class Fold(Module):
    ''' torch's nn.Fold: the inverse of Unfold, combining sliding local blocks back
    into (N, C, output_size). Wraps the functional fold. '''
    def __init__(self, output_size, kernel_size, dilation=1, padding=0, stride=1):
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def execute(self, x):
        return fold(x, self.output_size, self.kernel_size, self.dilation, self.padding, self.stride)

ModuleList = Sequential


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
    ''' Complex number helper (real/imag float pair).

        .. deprecated::
            Prefer the native ``complex64`` dtype. ``jt.array(complex_ndarray)``,
            ``torch.complex(re, im)``, ``torch.view_as_complex``, ``torch.polar`` and the
            ``torch.fft.*`` / ``jt.linalg`` complex paths now all produce and consume the native
            complex64 Var directly. ComplexNumber is kept only as an INTERNAL bridge substrate
            for the complex linalg kernels and is no longer returned by any public jittor / torch
            API.

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

    def abs(self):
        # magnitude |a+bi| = sqrt(a^2+b^2)  (torch.abs of a complex tensor)
        return self.norm()

    def __abs__(self):
        return self.norm()

    def angle(self):
        # phase atan2(imag, real)  (torch.angle)
        return jt.atan2(self.imag, self.real)

    def __getitem__(self, idx):
        return ComplexNumber(self.real[idx], self.imag[idx])

    def __neg__(self):
        return ComplexNumber(-self.real, -self.imag)

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


# ---------------------------------------------------------------------------
# Native complex64 <-> float32[..., 2] bridge. This lets FFT / linalg use the native
# complex64 dtype while the internal kernels still consume a real/imag float pair:
#   _complex64_to_real2 : complex64[...]   -> float32[..., 2]   (torch.view_as_real)
#   _real2_to_complex64 : float32[..., 2]  -> complex64[...]     (torch.view_as_complex)
# view_as_real/view_as_complex prefer the zero-copy reinterpret_view core op when available,
# and fall back to isolated jt.code kernels otherwise. Both are wrapped as jt.Function with
# each other as the adjoint backward, so the bridge is autograd-transparent on CPU+CUDA.
_complex64_imag_unit_cache = None
def _complex64_imag_unit():
    global _complex64_imag_unit_cache
    if _complex64_imag_unit_cache is None:
        _complex64_imag_unit_cache = jt.array(np.array(1j, dtype="complex64"))
    return _complex64_imag_unit_cache

def _complex64_to_real2_raw(z):
    reinterpret_view = getattr(jt, "reinterpret_view", None)
    if reinterpret_view is not None:
        return reinterpret_view(z, list(z.shape) + [2], "float32")
    # flatten to 1-D so the jt.code kernel is shape-agnostic, then restore the [..., 2] tail.
    n = 1
    for s in z.shape:
        n *= s
    flat = jt.code([n, 2], "float32", [z.reshape([n])],
        cpu_src="""
        for (int i=0; i<in0_shape0; i++) {
            @out(i,0) = @in0(i).real;
            @out(i,1) = @in0(i).imag;
        }""",
        cuda_src="""
        __global__ void k(@ARGS_DEF) {
            @PRECALC
            int i = blockIdx.x*blockDim.x + threadIdx.x;
            if (i < in0_shape0) { @out(i,0) = @in0(i).real; @out(i,1) = @in0(i).imag; }
        }
        int n = in0_shape0; k<<<(n+63)/64, 64>>>(@ARGS);""")
    return flat.reshape(list(z.shape) + [2])

def _real2_to_complex64_raw(x):
    assert x.shape[-1] == 2, f"view_as_complex expects last dim 2, got shape {x.shape}"
    reinterpret_view = getattr(jt, "reinterpret_view", None)
    if reinterpret_view is not None:
        return reinterpret_view(x, list(x.shape[:-1]) or [1], "complex64")
    # real[..., 2] -> native complex64. Use one code kernel instead of two getitem ops
    # plus mixed complex arithmetic; this is the hot path for RoPE view_as_complex.
    n = 1
    for s in x.shape[:-1]:
        n *= s
    out_shape = list(x.shape[:-1]) or [1]
    flat = jt.code([n], "complex64", [x.reshape([n, 2])],
        cpu_src="""
        for (int i=0; i<in0_shape0; i++) {
            @out(i) = complex64(float(@in0(i,0)), float(@in0(i,1)));
        }""",
        cuda_src="""
        __global__ void k(@ARGS_DEF) {
            @PRECALC
            int i = blockIdx.x*blockDim.x + threadIdx.x;
            if (i < in0_shape0) {
                @out(i) = complex64(float(@in0(i,0)), float(@in0(i,1)));
            }
        }
        int n = in0_shape0; k<<<(n+63)/64, 64>>>(@ARGS);""")
    return flat.reshape(out_shape)

class _Complex64ToReal2(jt.Function):
    def execute(self, z):
        return _complex64_to_real2_raw(z)
    def grad(self, g):                       # adjoint of view_as_real is view_as_complex
        return _real2_to_complex64_raw(g)

class _Real2ToComplex64(jt.Function):
    def execute(self, x):
        return _real2_to_complex64_raw(x)
    def grad(self, g):                       # adjoint of view_as_complex is view_as_real
        return _complex64_to_real2_raw(g)

def _complex64_to_real2(z):
    return _Complex64ToReal2.apply(z)

def _real2_to_complex64(x):
    return _Real2ToComplex64.apply(x)


def polar(abs:jt.Var, angle: jt.Var) -> jt.Var:
    # torch.polar: magnitude `abs`, phase `angle` -> native complex64 (Phase 6 migration off
    # ComplexNumber). Differentiable through the P1 bridge.
    assert abs.shape == angle.shape
    return _real2_to_complex64(jt.stack([abs * angle.cos(), abs * angle.sin()], dim=-1))

def view_as_complex(x: jt.Var) -> jt.Var:
    # torch.view_as_complex: real [..., 2] -> native complex64 (Phase 6 migration). Callers that
    # still need the legacy pair use nn.ComplexNumber(...) directly.
    assert x.shape[-1] == 2, f"view_as_complex expects last dim 2, got shape {x.shape}"
    return _real2_to_complex64(x)

def view_as_real(x) -> jt.Var:
    # torch.view_as_real: complex -> real [..., 2]. Polymorphic across the native complex64
    # dtype (Phase 6 bridge, differentiable) and the legacy nn.ComplexNumber (real/imag pair).
    if isinstance(x, ComplexNumber):
        return jt.stack([x.value[...,0],x.value[...,1]],dim=-1)
    assert "complex" in str(x.dtype), \
        f"view_as_real expects a complex64 Var or ComplexNumber, got dtype {x.dtype}"
    return _complex64_to_real2(x)


# Native complex64 accessors (torch parity), patched onto Var so they are available globally
# after `import jittor` (which imports jittor.nn). dtype-aware: complex64 slices the P1
# view_as_real bridge; real-dtype Vars match torch (real->self, imag->zeros, angle->0 or pi).
def _var_real(self):
    if "complex" in str(self.dtype):
        return view_as_real(self)[..., 0]
    return self
def _var_imag(self):
    if "complex" in str(self.dtype):
        return view_as_real(self)[..., 1]
    return jt.zeros_like(self)
def _var_angle(self):
    return jt.atan2(self.imag, self.real)
jt.Var.real = property(_var_real)
jt.Var.imag = property(_var_imag)
jt.Var.angle = _var_angle

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

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

def matmul(a, b):
    assert len(a.shape) >= 2 and len(b.shape) == 2
    assert a.shape[-1] == b.shape[-2]

    shape = list(a.shape) + [b.shape[-1]]
    a = a.broadcast(shape, [len(shape)-1])
    b = b.broadcast(shape)
    return (a*b).sum(len(shape)-2)
jt.Var.matmul = jt.Var.__matmul__ = matmul
jt.Var.__imatmul__ = lambda a,b: a.assign(matmul(a,b))

def get_init_var_rand(shape, dtype):
    return jt.array(np.random.normal(0.0, 1.0, shape).astype(np.float32))

@jt.var_scope('conv')
def conv(x, in_planes, out_planes, kernel_size, padding, stride = 1, init_method=None):
    Kw = kernel_size
    Kh = kernel_size
    _C = in_planes
    Kc = out_planes
    N,C,H,W = x.shape

    assert C==_C
    if init_method==None:
        w = jt.make_var([Kc, _C, Kh, Kw], init=lambda *a: init.relu_invariant_gauss(*a, mode="fan_out"))
    else:
        w = jt.make_var([Kc, _C, Kh, Kw], init=init_method)
    xx = x.reindex([N,Kc,C,(H+padding*2-kernel_size)//stride+1,(W+padding*2-kernel_size)//stride+1,Kh,Kw], [
        'i0', # Nid
        'i2', # Cid
        f'i3*{stride}-{padding}+i5', # Hid+Khid
        f'i4*{stride}-{padding}+i6', # Wid+KWid
    ])
    ww = w.broadcast(xx.shape, [0,3,4])
    yy = xx*ww
    y = yy.sum([2,5,6]) # C, Kh, Kw
    return y

@jt.var_scope('linear')
def linear(x, n):
    w = jt.make_var([n, x.shape[-1]], init=lambda *a: init.invariant_uniform(*a))
    w = w.reindex([w.shape[1], w.shape[0]],["i1","i0"])
    bound = 1.0/math.sqrt(w.shape[0])
    b = jt.make_var([n], init=lambda *a: init.uniform(*a,-bound,bound))
    return jt.matmul(x, w) + b

def relu(x): return jt.maximum(x, 0)
def leaky_relu(x, scale): return jt.ternary(x>0, x, x*scale)
def relu6(x): return jt.minimum(jt.maximum(x, 0), 6)

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

class SGD(object):
    """ Usage:
    optimizer = nn.SGD(model.parameters(), lr)
    optimizer.step(loss)
    """
    def __init__(self, parameters, lr, momentum=0, weight_decay=0, dampening=0, nesterov=False):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov

        self.no_grad_parameters = []
        self.parameters = []
        self.values = []
        for p in parameters:
            # broadcast parameter from 0 node when init
            if jt.mpi:
                p.assign(p.mpi_broadcast().detach())
            if p.is_stop_grad():
                self.no_grad_parameters.append(p)
                continue
            self.parameters.append(p)
            self.values.append(jt.zeros(p.shape, p.dtype).stop_fuse().stop_grad())

    def step(self, loss):
        ps = self.parameters
        gs = jt.grad(loss, ps)
        if jt.mpi:
            for g in gs:
                g.assign(g.mpi_all_reduce("mean"))
        for p, g, v in zip(ps, gs, self.values):
            dp = p * self.weight_decay + g
            v.assign(self.momentum * v + dp * (1 - self.dampening))
            if self.nesterov:
                p -= (dp + self.momentum * v) * self.lr
            else:
                p -= v * self.lr
            # detach with the prev graph to reduce memory consumption
            p.detach_inplace()
        # sync all no grad parameters, such as
        # moving_mean and moving_var in batch_norm
        # sync such parameters to reduce memory consumption
        jt.sync(self.no_grad_parameters)

class Adam(object):
    """ Usage:
    optimizer = nn.Adam(model.parameters(), lr)
    optimizer.step(loss)
    """
    def __init__(self, parameters, lr, eps=1e-8, betas=(0.9, 0.999), weight_decay=0):
        self.lr = lr
        self.eps = eps
        self.betas = betas
        # self.weight_decay = weight_decay
        assert weight_decay==0, "weight_decay is not supported yet"
        self.adam_step = 0
        
        self.no_grad_parameters = []
        self.parameters = []
        self.values = []
        self.m = []
        for p in parameters:
            if jt.mpi:
                p.assign(p.mpi_broadcast().detach())
            if p.is_stop_grad():
                self.no_grad_parameters.append(p)
                continue
            self.parameters.append(p)
            self.values.append(jt.zeros(p.shape, p.dtype).stop_fuse().stop_grad())
            self.m.append(jt.zeros(p.shape, p.dtype).stop_fuse().stop_grad())

    def step(self, loss):
        ps = self.parameters
        gs = jt.grad(loss, ps)
        if jt.mpi:
            for g in gs:
                g.assign(g.mpi_all_reduce("mean"))
        self.adam_step += 1
        n, (b0, b1) = float(self.adam_step), self.betas
        for p, g, v, m in zip(ps, gs, self.values, self.m):
            m.assign(b0 * m + (1-b0) * g)
            v.assign(b1 * v + (1-b1) * g * g)
            step_size = self.lr * jt.sqrt(1-b1**n) / (1-b0 ** n)
            p -= m * step_size / (jt.sqrt(v) + self.eps)
            p.detach_inplace()
        jt.sync(self.no_grad_parameters)

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
            if self.sync and jt.mpi:
                xmean = xmean.mpi_all_reduce("mean")
                x2mean = x2mean.mpi_all_reduce("mean")

            xvar = x2mean-xmean*xmean
            norm_x = (x-xmean)/jt.sqrt(xvar+self.eps)
            self.running_mean += (xmean.sum([0,2,3])-self.running_mean)*self.momentum
            self.running_var += (xvar.sum([0,2,3])-self.running_var)*self.momentum
        else:
            running_mean = self.running_mean.broadcast(x, [0,2,3])
            running_var = self.running_var.broadcast(x, [0,2,3])
            norm_x = (x-running_mean)/jt.sqrt(running_var+self.eps)
        w = self.weight.broadcast(x, [0,2,3])
        b = self.bias.broadcast(x, [0,2,3])
        return norm_x * w + b

Relu = jt.make_module(relu)
ReLU = Relu
Leaky_relu = jt.make_module(leaky_relu, 0.01)
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
        self.weight = init.relu_invariant_gauss([out_channels, in_channels // groups, Kh, Kw], dtype="float", mode="fan_out")
        if bias:
            self.bias = init.uniform([out_channels], dtype="float", low=-1, high=1)
        else:
            self.bias = None

    def execute(self, x):
        N,C,H,W = x.shape
        Kh, Kw = self.kernel_size
        assert C==self.in_channels
        if self.groups == 1:
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
        else:
            G = self.groups
            oc = self.out_channels
            oh = (H+self.padding[0]*2-Kh*self.dilation[0]+self.dilation[0]-1)//self.stride[0]+1
            ow = (W+self.padding[1]*2-Kw*self.dilation[1]+self.dilation[1]-1)//self.stride[1]+1
            xx = x.reshape((N, G, C//G, H, W))
            xx = xx.reindex([N,G,oc//G,C//G,oh,ow,Kh,Kw], [
                'i0', # Nid
                'i1', # Gid
                'i3', # C//G id
                f'i4*{self.stride[0]}-{self.padding[0]}+i6*{self.dilation[0]}', # Hid+Khid
                f'i5*{self.stride[1]}-{self.padding[1]}+i7*{self.dilation[1]}', # Wid+KWid
            ])
            ww = self.weight.reshape((G, oc//G, C//G, Kh, Kw))
            ww = ww.broadcast(xx.shape, [0,4,5])
            yy = xx*ww
            yy = yy.sum([3,6,7]) # oc//G, Kh, Kw
            y = yy.reshape((N, oc, oh, ow))
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

        self.weight = init.relu_invariant_gauss((in_channels, out_channels) + self.kernel_size, dtype="float", mode="fan_out")
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


class Tanh(Module):
    def __init__(self):
        super().__init__()
    def execute(self, x) :
        return x.tanh()

class Sigmoid(Module):
    def __init__(self):
        super().__init__()
    def execute(self, x) :
        return 1 / (1 + jt.exp(-x))

def resize(x, size, mode="nearest"):
    img = x
    n,c,h,w = x.shape
    H,W = size
    new_size = [n,c,H,W]
    nid, cid, hid, wid = jt.index(new_size)
    x = hid * ((h-1)/(H-1))
    y = wid * ((w-1)/(W-1))
    if mode=="nearest":
        return img.reindex([nid, cid, x.floor(), y.floor()])
    if mode=="bilinear":
        fx, fy = x.floor(), y.floor()
        cx, cy = fx+1, fy+1
        dx, dy = x-fx, y-fy
        a = img.reindex_var([nid, cid, fx, fy])
        b = img.reindex_var([nid, cid, cx, fy])
        c = img.reindex_var([nid, cid, fx, cy])
        d = img.reindex_var([nid, cid, cx, cy])
        dnx, dny = 1-dx, 1-dy
        ab = dx*b + dnx*a
        cd = dx*d + dnx*c
        o = ab*dny + cd*dy
        return o
    raise(f"Not support {interpolation}")



class Sequential(Module):
    def __init__(self, *args):
        self.layers = args
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

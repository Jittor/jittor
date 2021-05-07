# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
import numpy as np
import math

def eye(shape, dtype):
    return jt.array(np.identity(shape[0])).unary(dtype)

def eye_(var):
    var.assign(eye(var.shape, var.dtype))

def constant(shape, dtype, value=0.0):
    return jt.array(value).unary(dtype).broadcast(shape)

def constant_(var, value=0.0):
    var.assign(constant(var.shape, var.dtype, value))

def uniform(shape, dtype, low, high):
    return jt.random(shape, dtype) * (low - high) + high

def uniform_(var, low, high):
    var.assign(uniform(var.shape, var.dtype, low, high))

def gauss(shape, dtype, mean=0.0, std=1.0):
    return jt.random(shape, dtype, "normal") * std + mean

def gauss_(var, mean=0.0, std=1.0):
    var.assign(gauss(var.shape, var.dtype, mean, std))

def invariant_uniform(shape, dtype, mode="fan_in"):
    assert len(shape)>1
    assert mode=="fan_in" or mode=="fan_out"

    matsize=1
    for i in shape[2:]:
        matsize *= i
    fan = (shape[1] * matsize) if mode=="fan_in" else (shape[0] * matsize)
    bound = math.sqrt(1.0/fan)
    return uniform(shape, dtype, -bound, bound)

def invariant_uniform_(var, mode="fan_in"):
    var.assign(invariant_uniform(tuple(var.shape), var.dtype, mode))

def relu_invariant_gauss(shape, dtype, mode="fan_in"):
    assert len(shape)>1
    assert mode=="fan_in" or mode=="fan_out"
    
    matsize=1
    for i in shape[2:]:
        matsize *= i
    fan = (shape[1] * matsize) if mode=="fan_in" else (shape[0] * matsize)
    std = math.sqrt(2.0/fan)
    return gauss(shape, dtype, 0, std)

def relu_invariant_gauss_(var, mode="fan_in"):
    var.assign(relu_invariant_gauss(tuple(var.shape), var.dtype, mode))

def calculate_std(var,mode,nonlinearity,param=0.01):
    mode = mode.lower()
    assert isinstance(param,(int,float))
    assert var.ndim>=2
    assert mode in ['fan_in', 'fan_out']

    fan = var.shape[1] if mode == 'fan_in' else var.shape[0]
    fan *= var[0][0].numel()

    gains = {
        'linear':1,
        'conv1d':1,
        'conv2d':1,
        'conv3d':1,
        'conv_transpose1d':1,
        'conv_transpose2d':1,
        'conv_transpose3d':1,
        'sigmoid':1,
        'tanh':5.0/3,
        'relu':math.sqrt(2.0),
        'leaky_relu':math.sqrt(2.0 / (1 + param ** 2)),
    }
    gain = gains[nonlinearity]
    std = gain/math.sqrt(fan)
    return std


def kaiming_uniform_(var, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    std = calculate_std(var,mode,nonlinearity,a)
    bound = math.sqrt(3.0) * std
    return uniform_(var,-bound, bound)

def kaiming_normal_(var, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    std = calculate_std(var,mode,nonlinearity,a)
    return gauss_(var,0, std)


#TODO: bound = gain * math.sqrt(6.0/fan) ??
def xavier_uniform(shape, dtype, gain=1.0):
    assert len(shape)>1

    matsize=1
    for i in shape[2:]:
        matsize *= i
    fan = (shape[1] * matsize) + (shape[0] * matsize)
    bound = gain * math.sqrt(1.0/fan)
    return uniform(shape, dtype, -bound, bound)

def xavier_uniform_(var, gain=1.0):
    var.assign(xavier_uniform(tuple(var.shape), var.dtype, gain))

def xavier_gauss(shape, dtype, gain=1.0):
    assert len(shape)>1
    
    matsize=1
    for i in shape[2:]:
        matsize *= i
    fan = (shape[1] * matsize) + (shape[0] * matsize)
    std = gain * math.sqrt(2.0/fan)
    return gauss(shape, dtype, 0, std)

def xavier_gauss_(var, gain=1.0):
    var.assign(xavier_gauss(tuple(var.shape), var.dtype, gain))

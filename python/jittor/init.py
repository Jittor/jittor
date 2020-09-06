# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
import numpy as np
import math

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
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


def calculate_gain(nonlinearity, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.
    The values are as follows:
    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    ================= ====================================================
    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function
    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.ndim
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.ndim > 2:
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out

def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where
    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}
    Also known as He initialization.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).
    Examples:
        >>> init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with jt.no_grad():
        return uniform_(tensor,-bound, bound)

def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    normal distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where
    .. math::
        \text{std} = \frac{\text{gain}}{\sqrt{\text{fan\_mode}}}
    Also known as He initialization.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    with jt.no_grad():
        return gauss_(tensor,0, std)
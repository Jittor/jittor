# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
from jittor import NanoVector, Var
import numpy as np
import math
import warnings

def eye(shape, dtype="float32"):
    ''' Generate 2-D identity matrix.

    Args:
        shape (int or tuple of int):
            shape of the output matrix
        dtype (string):
            dtype of the output matrix, default float32
    
    Return:
        A Jittor Var of identity matrix.

    Example::

        from jittor import init
        print(init.eye(2))
        # output: [[1.,0.],[0.,1.]]
        print(init.eye((2,3), "float32"))
        # output: [[1.,0.,0.],[0.,1.,0.]]

    '''
    if isinstance(shape, int):
        shape = (shape,shape)
    assert len(shape)==2, f"len of shape should be 2, but got {shape}"
    index = jt.index(shape)
    return (index[0]==index[1]).unary(dtype)

def eye_(var):
    ''' Inplace initialize variable with identity matrix.

    Args:
        var (Jittor Var):
            Var to initialize with identity matrix.
    
    Return:
        var itself.
    
    Example::

        from jittor import init
        from jittor import nn
        linear = nn.Linear(2,2)
        init.eye_(linear.weight)
        print(linear.weight)
        # output: [[1.,0.],[0.,1.]]
        linear.weight.eye_() # This is ok too

    '''
    return var.assign(eye(var.shape, var.dtype))
Var.eye_ = eye_

def constant(shape, dtype="float32", value=0.0):
    '''Generate constant Jittor Var.

    Args:
        shape (int or tuple of int):
            shape of the output Var
        dtype (string):
            dtype of the output Var, default float32
        value (int or float):
            value to be filled in output Var
    
    Return:
        A Jittor Var which filled by constant value.

    Example::

        from jittor import init
        print(init.constant(2))
        # output: [0.,0.]
        print(init.constant((2,3), value=1.))
        # output: [[1.,1.,1.],[1.,1.,1.]]

    '''
    return jt.array(value).unary(dtype).broadcast(NanoVector(shape))

def constant_(var, value=0.0):
    ''' Inplace initialize variable with constant value.

    Args:
        var (Jittor Var):
            Var to initialize with constant value.
    
    Return:
        var itself.
    
    Example::

        from jittor import init
        from jittor import nn
        linear = nn.Linear(2,2)
        init.constant_(linear.weight)
        print(linear.weight)
        # output: [[0.,0.],[0.,0.]]
        linear.weight.constant_() # This is ok too

    '''
    return var.assign(constant(var.shape, var.dtype, value))
Var.constant_ = constant_

def zero(shape, dtype="float32"):
    '''Generate zero Jittor Var.

    Args:
        shape (int or tuple of int):
            shape of the output Var
        dtype (string):
            dtype of the output Var, default float32
    
    Return:
        A Jittor Var which filled by constant value.

    Example::

        from jittor import init
        print(init.zero(2))
        # output: [0.,0.]
        print(init.zero((2,3)))
        # output: [[0.,0.,0.],[0.,0.,0.]]

    '''
    return constant(shape, dtype, 0)
def zero_(var):
    ''' Inplace initialize variable with zero.

    Args:
        var (Jittor Var):
            Var to initialize with zero.
    
    Return:
        var itself.
    
    Example::

        from jittor import init
        from jittor import nn
        linear = nn.Linear(2,2)
        init.zero_(linear.weight)
        print(linear.weight)
        # output: [[0.,0.],[0.,0.]]
        linear.weight.zero_() # This is ok too

    '''
    return var.assign(zero(var.shape, var.dtype))
Var.zero_ = zero_
def one(shape, dtype="float32"):
    '''Generate Jittor Var filled by one.

    Args:
        shape (int or tuple of int):
            shape of the output Var
        dtype (string):
            dtype of the output Var, default float32
    
    Return:
        A Jittor Var which filled by one.

    Example::

        from jittor import init
        print(init.one(2))
        # output: [1.,1.]
        print(init.one((2,3)))
        # output: [[1.,1.,1.],[1.,1.,1.]]

    '''
    return constant(shape, dtype, 1)
def one_(var):
    ''' Inplace initialize variable with one.

    Args:
        var (Jittor Var):
            Var to initialize with one.
    
    Return:
        var itself.
    
    Example::

        from jittor import init
        from jittor import nn
        linear = nn.Linear(2,2)
        init.one_(linear.weight)
        print(linear.weight)
        # output: [[1.,1.],[1.,1.]]
        linear.weight.one_() # This is ok too

    '''
    return var.assign(one(var.shape, var.dtype))
Var.one_ = one_

def uniform(shape, dtype="float32", low=0, high=1):
    '''Generate random uniform Jittor Var.

    Args:
        shape (int or tuple of int):
            shape of the output Var
        dtype (string):
            dtype of the output Var, default float32
        low (int or float or Var):
            lower bound value of the random uniform
        high (int or float or Var):
            upper bound value of the random uniform

    Return:
        A Jittor Var which filled by random uniform.

    Example::

        from jittor import init
        print(init.uniform(5))
        # output: [0.202268, 0.518688, 0.595274, 0.777354, 0.981979]
        print(init.uniform((2,3), low=-1, high=1))
        # output: [[ 0.6647397   0.2801202  -0.01981187]
        #          [-0.9779438  -0.30149996  0.69056886]]

    '''
    return jt.random(NanoVector(shape), dtype) * (low - high) + high

def uniform_(var, low=0, high=1):
    ''' Inplace initialize Jittor Var by random uniform.

    Args:
        var (Jittor Var):
            Var to be initialized by random uniform
        low (int or float or Var):
            lower bound value of the random uniform
        high (int or float or Var):
            upper bound value of the random uniform

    Example::

        from jittor import init
        from jittor import nn
        linear = nn.Linear(2,2)
        init.uniform_(linear.weight, -1.0, 1.0)
        print(linear.weight)
        # output: [[ 0.6647397   0.2801202], [-0.9779438  -0.30149996]]
        linear.weight.uniform_(-1.0, 1.0) # This is ok too

    '''
    return var.assign(uniform(var.shape, var.dtype, low, high))
Var.uniform_ = uniform_

def gauss(shape, dtype="float32", mean=0.0, std=1.0):
    ''' Return Jittor Var initialize by random gauss.

    Args:
        shape (int or tuple of int):
            shape of the output Var
        dtype (string):
            dtype of the output Var, default float32
        mean (int or float or Var):
            mean value of the random gauss
        std (int or float or Var):
            std value of the random gauss

    Example::

        from jittor import init
        from jittor import nn
        a = init.gauss((2,2), "float32", 0.0, 1.0)
        print(a)

    '''
    return jt.random(NanoVector(shape), dtype, "normal") * std + mean

def gauss_(var, mean=0.0, std=1.0):
    ''' Inplace initialize Jittor Var by random gauss.

    Args:
        var (Jittor Var):
            Var to be initialized by random gauss
        mean (int or float or Var):
            mean value of the random gauss
        std (int or float or Var):
            std value of the random gauss

    Example::

        from jittor import init
        from jittor import nn
        linear = nn.Linear(2,2)
        init.gauss_(linear.weight, 0.0, 1.0)
        print(linear.weight)
        linear.weight.gauss_(0.0, 1.0) # This is ok too

    '''
    return var.assign(gauss(var.shape, var.dtype, mean, std))
Var.gauss_ = gauss_

def invariant_uniform(shape, dtype="float32", mode="fan_in"):
    ''' Return Jittor initialized Var by invariant_uniform.

    Args:
        shape (int or tuple of int):
            shape of the output Var
        dtype (string):
            dtype of the output Var, default float32
        mode (string):
            mode selection, should be fan_in or fan_out.
            Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.

    Example::

        from jittor import init
        from jittor import nn
        a = init.invariant_uniform_((2,2))
        print(a)

    '''
    assert len(shape)>1
    assert mode=="fan_in" or mode=="fan_out", \
        f"mode not supported, should be fan_in or fan_out, but got {mode}"

    matsize=1
    for i in shape[2:]:
        matsize *= i
    fan = (shape[1] * matsize) if mode=="fan_in" else (shape[0] * matsize)
    bound = math.sqrt(1.0/fan)
    return uniform(shape, dtype, -bound, bound)

def invariant_uniform_(var, mode="fan_in"):
    ''' Inplace initialize Jittor Var by invariant_uniform.

    Args:
        var (Jittor Var):
            Var to be initialized by random invariant_uniform
        mode (string):
            mode selection, should be fan_in or fan_out.
            Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.

    Example::

        from jittor import init
        from jittor import nn
        linear = nn.Linear(2,2)
        init.invariant_uniform_(linear.weight)
        print(linear.weight)
        linear.weight.invariant_uniform_() # This is ok too

    '''
    var.assign(invariant_uniform(tuple(var.shape), var.dtype, mode))
Var.invariant_uniform_ = invariant_uniform_

def relu_invariant_gauss(shape, dtype="float32", mode="fan_in"):
    ''' Return Jittor Var initialized by relu_invariant_gauss.

    Args:
        shape (int or tuple of int):
            shape of the output Var
        dtype (string):
            dtype of the output Var, default float32
        mode (string):
            mode selection, should be fan_in or fan_out.
            Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.

    Example::

        from jittor import init
        from jittor import nn
        a = init.relu_invariant_gauss((2,2))
        print(a)
    
    '''
    assert len(shape)>1
    assert mode=="fan_in" or mode=="fan_out"
    
    matsize=1
    for i in shape[2:]:
        matsize *= i
    fan = (shape[1] * matsize) if mode=="fan_in" else (shape[0] * matsize)
    std = math.sqrt(2.0/fan)
    return gauss(shape, dtype, 0, std)

def relu_invariant_gauss_(var, mode="fan_in"):
    ''' Inplace initialize Jittor Var by relu_invariant_gauss.

    Args:
        var (Jittor Var):
            Var to be initialized by random relu_invariant_gauss
        mode (string):
            mode selection, should be fan_in or fan_out.
            Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.

    Example::

        from jittor import init
        from jittor import nn
        linear = nn.Linear(2,2)
        init.relu_invariant_gauss_(linear.weight)
        print(linear.weight)
        linear.weight.relu_invariant_gauss_() # This is ok too

    '''
    return var.assign(relu_invariant_gauss(tuple(var.shape), var.dtype, mode))
Var.relu_invariant_gauss_ = relu_invariant_gauss_

def calculate_std(var, mode, nonlinearity, param=0.01):
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
    ''' Inplace initialize Jittor Var by kaiming_uniform.

    Args:
        var (Jittor Var):
            Var to be initialized by random kaiming_uniform
        a (float):
            the negative slope of the rectifier used after this layer (only used with 'leaky_relu')
        mode (string):
            mode selection, should be fan_in or fan_out.
            Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.
        nonlinearity (string):
            nonlinearity used after this layer. 
            It can be one of [linear, conv*, sigmoid, tanh, relu, leaky_relu].
            leaky_relu is used by default.

    Example::

        from jittor import init
        from jittor import nn
        linear = nn.Linear(2,2)
        init.kaiming_uniform_(linear.weight)
        print(linear.weight)
        linear.weight.kaiming_uniform_() # This is ok too

    '''
    std = calculate_std(var,mode,nonlinearity,a)
    bound = math.sqrt(3.0) * std
    return uniform_(var,-bound, bound)
Var.kaiming_uniform_ = kaiming_uniform_

def kaiming_normal_(var, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    ''' Inplace initialize Jittor Var by kaiming_normal.

    Args:
        var (Jittor Var):
            Var to be initialized by random kaiming_normal
        a (float):
            the negative slope of the rectifier used after this layer (only used with 'leaky_relu')
        mode (string):
            mode selection, should be fan_in or fan_out.
            Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.
        nonlinearity (string):
            nonlinearity used after this layer. 
            It can be one of [linear, conv*, sigmoid, tanh, relu, leaky_relu].
            leaky_relu is used by default.

    Example::

        from jittor import init
        from jittor import nn
        linear = nn.Linear(2,2)
        init.kaiming_normal_(linear.weight)
        print(linear.weight)
        linear.weight.kaiming_normal_() # This is ok too

    '''
    std = calculate_std(var,mode,nonlinearity,a)
    return gauss_(var,0, std)
Var.kaiming_normal_ = kaiming_normal_


def xavier_uniform(shape, dtype="float32", gain=1.0):
    ''' Inplace initialize Jittor Var by xavier_uniform.
    The resulting var will have values sampled from
    :math:`uniform(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Args:
        shape (int or tuple of int):
            shape of the return Var.
        dtype (string):
            dtype of the return Var, default float32.
        gain (float):
            an optional scaling factor.

    Example::

        from jittor import init
        from jittor import nn
        a = init.xavier_uniform((2,2), gain=init.calculate_gain('relu'))
        print(a)
    '''
    assert len(shape)>1

    matsize=1
    for i in shape[2:]:
        matsize *= i
    fan = (shape[1] * matsize) + (shape[0] * matsize)
    bound = gain * math.sqrt(6.0/fan)
    return uniform(shape, dtype, -bound, bound)

def xavier_uniform_(var, gain=1.0):
    ''' Inplace initialize Jittor Var by xavier_uniform.
    The resulting var will have values sampled from
    :math:`uniform(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Args:
        var (Jittor Var):
            Var to be initialized by random xavier_uniform
        gain (float):
            an optional scaling factor.

    Example::

        from jittor import init
        from jittor import nn
        linear = nn.Linear(2,2)
        init.xavier_uniform_(linear.weight, init.calculate_gain('relu'))
        print(linear.weight)
        linear.weight.xavier_uniform_() # This is ok too

    '''
    return var.assign(xavier_uniform(tuple(var.shape), var.dtype, gain))
Var.xavier_uniform_ = xavier_uniform_

def xavier_gauss(shape, dtype="float32", gain=1.0):
    ''' Return Jittor Var initialized by xavier_gauss, a.k.a xavier_normal.
    The resulting var will have values sampled from
    :math:`gauss(-a, a)` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}

    Args:
        shape (int or tuple of int):
            shape of the return Var.
        dtype (string):
            dtype of the return Var, default float32.
        gain (float):
            an optional scaling factor.

    Example::

        from jittor import init
        from jittor import nn
        linear = nn.Linear(2,2)
        init.xavier_gauss_(linear.weight, init.calculate_gain('relu'))
        print(linear.weight)
        linear.weight.xavier_gauss_() # This is ok too

    '''
    assert len(shape)>1
    
    matsize=1
    for i in shape[2:]:
        matsize *= i
    fan = (shape[1] * matsize) + (shape[0] * matsize)
    std = gain * math.sqrt(2.0/fan)
    return gauss(shape, dtype, 0, std)

def xavier_gauss_(var, gain=1.0):
    ''' Inplace initialize Jittor Var by xavier_gauss, a.k.a xavier_normal.
    The resulting var will have values sampled from
    :math:`gauss(-a, a)` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}

    Args:
        var (Jittor Var):
            Var to be initialized by random xavier_gauss
        gain (float):
            an optional scaling factor.

    Example::

        from jittor import init
        from jittor import nn
        linear = nn.Linear(2,2)
        init.xavier_gauss_(linear.weight, init.calculate_gain('relu'))
        print(linear.weight)
        linear.weight.xavier_gauss_() # This is ok too

    '''
    return var.assign(xavier_gauss(tuple(var.shape), var.dtype, gain))
Var.xavier_gauss_ = xavier_gauss_

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
    SELU              :math:`\frac{3}{4}`
    ================= ====================================================

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function

    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2

    .. _Self-Normalizing Neural Networks: https://papers.nips.cc/paper/2017/hash/5d44ee6f2c3f71b73125876103c8f6c4-Abstract.html
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
    elif nonlinearity == 'selu':
        return 3.0 / 4 
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def trunc_normal_(var, mean=0., std=1., a=-2., b=2.):
    # type: (jt.jittor_core.Var, float, float, float, float) -> jt.jittor_core.Var
    r"""Fills the input jt.jittor_core.Var with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        var: an n-dimensional `jt.jittor_core.Var` 
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:

        from jittor import init
        from jittor import nn
        linear = nn.Linear(2,2)
        init.trunc_normal_(linear.weight, std=.02)
        print(linear.weight)
        linear.weight.trunc_normal_(std=.02) # This is ok too
    """
    return _no_grad_trunc_normal_(var, mean, std, a, b)
Var.trunc_normal_ = trunc_normal_

def _no_grad_trunc_normal_(var, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)


    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    # var.uniform(2 * l - 1, 2 * u - 1)
    var.uniform_(low=2 * l - 1, high=2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    var = var.erfinv()

    # Transform to proper mean, std
    var = var.multiply(std * math.sqrt(2.))
    var = var.add(mean)

    # Clamp to ensure it's in the proper range
    var = var.clamp(min_v=a, max_v=b)
    return var
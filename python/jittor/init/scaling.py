# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import math
from jittor import _arg_policy
from .basic import uniform, uniform_, gauss, gauss_
from ._fan import _calculate_fan_in_and_fan_out, _fan_for_mode, calculate_std


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
    fan = _fan_for_mode(shape, mode)
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
    fan = _fan_for_mode(shape, mode)
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


def kaiming_uniform_(var, a=0, mode='fan_in', nonlinearity='leaky_relu', generator=None):
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
    return _kaiming_uniform_(var, uniform_, a, mode, nonlinearity, generator)


def _kaiming_uniform_(var, uniform_impl, a=0, mode='fan_in',
                      nonlinearity='leaky_relu', generator=None):
    """Shared Kaiming math with an explicitly selected random/writeback owner."""
    if generator is not None:
        # `unsupported`, not `ignored`: a seeded generator asks for one specific
        # tensor, and jittor draws a different one from its global RNG. That is
        # a changed observable value, not a missed optimisation.
        #
        # Not implemented by reseeding the global RNG from the generator (the
        # shortcut jittor.compat.torch.installers.factories uses for randn):
        # torch ADVANCES a generator per draw, so reseeding on every call would
        # make N layers initialised from the same generator come out with
        # IDENTICAL weights wherever their shapes match -- trading a visible
        # error for a much worse silent one. Real support needs a per-generator
        # RNG stream in the core, which jittor does not have (only the process-
        # wide jt.set_global_seed).
        _arg_policy.unsupported(
            "jittor.init.kaiming_uniform_", "generator", generator,
            "jittor has only a process-wide RNG, so the draw cannot come from "
            "the supplied generator's stream: the values differ from torch's "
            "and the generator is neither read nor advanced")
    std = calculate_std(var,mode,nonlinearity,a)
    bound = math.sqrt(3.0) * std
    return uniform_impl(var,-bound, bound)


def kaiming_normal_(var, a=0, mode='fan_in', nonlinearity='leaky_relu', generator=None):
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
    if generator is not None:
        # `unsupported`, not `ignored`: a seeded generator asks for one specific
        # tensor, and jittor draws a different one from its global RNG. That is
        # a changed observable value, not a missed optimisation.
        #
        # Not implemented by reseeding the global RNG from the generator (the
        # shortcut jittor.compat.torch.installers.factories uses for randn):
        # torch ADVANCES a generator per draw, so reseeding on every call would
        # make N layers initialised from the same generator come out with
        # IDENTICAL weights wherever their shapes match -- trading a visible
        # error for a much worse silent one. Real support needs a per-generator
        # RNG stream in the core, which jittor does not have (only the process-
        # wide jt.set_global_seed).
        _arg_policy.unsupported(
            "jittor.init.kaiming_normal_", "generator", generator,
            "jittor has only a process-wide RNG, so the draw cannot come from "
            "the supplied generator's stream: the values differ from torch's "
            "and the generator is neither read nor advanced")
    std = calculate_std(var,mode,nonlinearity,a)
    return gauss_(var,0, std)


def xavier_uniform(shape, dtype="float32", gain=1.0):
    r''' Inplace initialize Jittor Var by xavier_uniform.
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
    fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
    fan = fan_in + fan_out
    bound = gain * math.sqrt(6.0/fan)
    return uniform(shape, dtype, -bound, bound)


def xavier_uniform_(var, gain=1.0):
    r''' Inplace initialize Jittor Var by xavier_uniform.
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


def xavier_gauss(shape, dtype="float32", gain=1.0):
    r''' Return Jittor Var initialized by xavier_gauss, a.k.a xavier_normal.
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
    fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
    fan = fan_in + fan_out
    std = gain * math.sqrt(2.0/fan)
    return gauss(shape, dtype, 0, std)


def xavier_gauss_(var, gain=1.0):
    r''' Inplace initialize Jittor Var by xavier_gauss, a.k.a xavier_normal.
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

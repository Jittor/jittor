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


def _calculate_fan_in_and_fan_out(shape):
    """``(fan_in, fan_out)`` for a weight of this shape.

    The single definition used by every initializer here. Matches
    ``torch.nn.init._calculate_fan_in_and_fan_out``: dim 1 is the number of
    input feature maps, dim 0 the number of output ones, and the trailing dims
    are the receptive field that both get multiplied by.

    There used to be two spellings of this -- a ``matsize`` loop in the
    ``invariant_*``/``xavier_*`` family and ``fan *= var[0][0].numel()`` in
    ``calculate_std``. They agreed numerically, but the second one indexes the
    Var (real work, and it needs shape[0] and shape[1] to be non-empty) just to
    read a number that is already in the shape.
    """
    if len(shape) < 2:
        raise ValueError(
            "fan in and fan out cannot be computed for a var with fewer than "
            "2 dimensions, got shape %s" % (tuple(shape),))
    receptive_field_size = 1
    for i in shape[2:]:
        receptive_field_size *= i
    return shape[1] * receptive_field_size, shape[0] * receptive_field_size


def _fan_for_mode(shape, mode):
    """The fan selected by ``mode``, which must be 'fan_in' or 'fan_out'."""
    mode = mode.lower()
    if mode not in ("fan_in", "fan_out"):
        raise ValueError(
            "mode not supported, should be fan_in or fan_out, but got %r" % (mode,))
    fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
    return fan_in if mode == "fan_in" else fan_out


def calculate_std(var, mode, nonlinearity, param=0.01):
    """The kaiming standard deviation ``gain / sqrt(fan)``.

    Reads its gain from :func:`calculate_gain`, which is the module's one gain
    table. There used to be a second, private table inline here that disagreed
    with it: it had no ``'selu'`` entry, so ``kaiming_uniform_(w,
    nonlinearity='selu')`` died with a bare ``KeyError: 'selu'`` while
    ``calculate_gain('selu')`` happily returned 3/4. An unsupported name now
    raises the same ValueError from the same place no matter which way in you
    came.
    """
    assert isinstance(param,(int,float))
    assert var.ndim>=2
    fan = _fan_for_mode(var.shape, mode)
    gain = calculate_gain(nonlinearity, param)
    std = gain/math.sqrt(fan)
    return std


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

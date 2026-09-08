# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************


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
    if len(shape) != 2:
        raise ValueError("eye: shape must have two dimensions, got {}".format(shape))
    import jittor as jt
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
    import jittor as jt
    return jt.array(value).unary(dtype).broadcast(jt.NanoVector(shape))


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


def random_(var):
    import jittor as jt
    return var.assign(jt.rand(var.shape, var.dtype))


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
    import jittor as jt
    return jt.random(jt.NanoVector(shape), dtype) * (low - high) + high


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
    import jittor as jt
    return jt.random(jt.NanoVector(shape), dtype, "normal") * std + mean


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

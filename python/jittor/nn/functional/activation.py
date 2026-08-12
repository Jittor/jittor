"""Functional activation implementations exposed through :mod:`jittor.nn`."""

import numpy as np

import jittor as jt


def relu(x, inplace=False):
    r''' Applies the element-wise function:

    .. math::
        \text{ReLU}(x) = \max(0,x)

    :param x: the input var
    :type x: jt.Var

    :param inplace: can optionally do the operation in-place (accepted for
        torch compatibility; Jittor computes a new var). Default: ``False``
    :type inplace: bool

    Example:
        >>> a = jt.randn(3)
        >>> a
        jt.Var([-0.38380373 1.1338731   6.128115  ], dtype=float32)
        >>> nn.relu(a)
        jt.Var([0.        1.1338731 6.128115 ], dtype=float32)
    '''
    cond = x>0.0
    return jt.ternary_out_hint(cond, x, 0.0)


def leaky_relu(x, scale=0.01, negative_slope=None, inplace=False):
    # torch spells the slope `negative_slope` (+ an `inplace` flag); accept both.
    if negative_slope is not None:
        scale = negative_slope
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


def gelu(x, approximate='none'):
    r''' Applies the element-wise function:

    .. math::
        \text{GELU}(x) = x * \Phi(x)

    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

    When ``approximate='tanh'``, GELU is estimated with:

    .. math::
        \text{GELU}(x) = 0.5 * x * (1 + \tanh(\sqrt{2/\pi} * (x + 0.044715 * x^3)))

    :param x: the input var
    :type x: jt.Var
    :param approximate: the gelu approximation algorithm to use, either ``'none'``
        (exact, erf-based) or ``'tanh'``. Default: ``'none'``.
    :type approximate: str

    Example:
        >>> a = jt.randn(3)
        >>> a
        jt.Var([-0.38380373 -1.1338731   2.128115  ], dtype=float32)
        >>> nn.gelu(a)
        jt.Var([-0.134547   0.9882567  6.128115 ], dtype=float32)
    '''
    if approximate == 'tanh':
        _sqrt_2_over_pi = 0.7978845608028654
        return 0.5*x*(1+jt.tanh(_sqrt_2_over_pi*(x+0.044715*(x*x*x))))
    elif approximate == 'none':
        # Keep the exact GELU kernel in the tensor's compute dtype. Dividing a
        # float32 Var by a Python float intentionally uses a float64 intermediate
        # in torch_compat (to match scalar division to the last bit), which made
        # this elementwise hot path execute a double-precision divide per value.
        # PyTorch's GELU kernel uses a typed 1/sqrt(2) constant instead. Low
        # precision inputs compute in fp32 and cast back, matching torch's output
        # dtype while retaining the existing elementwise fusion opportunity.
        input_dtype = str(x.dtype)
        low_precision = input_dtype in ('float16', 'bfloat16')
        compute_x = x.float32() if low_precision else x
        scalar_type = np.float64 if input_dtype == 'float64' else np.float32
        inv_sqrt2 = scalar_type(0.7071067811865476)
        half = scalar_type(0.5)
        one = scalar_type(1.0)
        result = half * compute_x * (one + jt.erf(compute_x * inv_sqrt2))
        return result.cast(input_dtype) if low_precision else result
    else:
        raise ValueError(f"approximate argument must be either 'none' or 'tanh', got {approximate}")


def sigmoid(x):
    ''' Element-wise sigmoid. Exposed as a function (torch.nn.functional.sigmoid /
    nn.functional.sigmoid) -- jittor only had jt.sigmoid / Var.sigmoid before, so
    `F.sigmoid(x)` (used by qwen2_moe and others) raised AttributeError.'''
    return jt.sigmoid(x)


def silu(x, inplace=False):     # inplace: accepted for torch/mmcv compat, ignored
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


def prelu(x, weight):
    ''' Applies the element-wise PReLU function (functional form):

    .. math::
        \\text{PReLU}(x) = \\max(0, x) + weight * \\min(0, x)

    :param x: the input var
    :type x: jt.Var
    :param weight: the (learnable) slope, either a scalar or a 1-D var with one
        value per input channel (broadcast over dim 1).
    :type weight: jt.Var or float
    '''
    if isinstance(weight, jt.Var) and weight.numel() != 1:
        assert weight.numel() == x.size(1), \
            "weight (number of parameters) does not match input channels in prelu"
        dims = [i for i in range(x.ndim) if i != 1]
        w = weight.broadcast(x, dims)
    else:
        w = weight
    return jt.maximum(0, x) + w * jt.minimum(0, x)


def hardswish(x):
    ''' Applies the element-wise Hardswish function:

    .. math::
        \\text{Hardswish}(x) = \\begin{cases}
        0, & x \\le -3 \\\\
        x, & x \\ge +3 \\\\
        x \\cdot (x + 3) / 6, & \\text{otherwise}
        \\end{cases}
    '''
    return x * jt.clamp(x + 3, min_v=0, max_v=6) / 6


def hardsigmoid(x):
    ''' Applies the element-wise Hardsigmoid function:

    .. math::
        \\text{Hardsigmoid}(x) = \\begin{cases}
        0, & x \\le -3 \\\\
        1, & x \\ge +3 \\\\
        x / 6 + 1/2, & \\text{otherwise}
        \\end{cases}
    '''
    return jt.clamp(x / 6 + 0.5, min_v=0.0, max_v=1.0)


def rrelu(x, lower=1./8, upper=1./3, training=False):
    ''' Applies the randomized leaky rectified linear unit function,
    element-wise, as described in `Empirical Evaluation of Rectified
    Activations in Convolutional Network`.

    During training the negative slope ``a`` is sampled uniformly from
    ``[lower, upper]``; during evaluation the fixed slope
    ``(lower + upper) / 2`` is used (matching torch).

    :param x: the input var
    :param lower: lower bound of the uniform slope. Default: 1/8
    :param upper: upper bound of the uniform slope. Default: 1/3
    :param training: whether to sample the slope (train) or use its mean (eval).
    '''
    if training:
        a = jt.random(x.shape, x.dtype) * (upper - lower) + lower
    else:
        a = (lower + upper) / 2
    return jt.ternary(x >= 0, x, a * x)


def get_init_var_rand(shape, dtype):
    return jt.array(np.random.normal(0.0, 1.0, shape).astype(np.float32))


def softplus(x, beta=1.0, threshold=20.0):
    return 1 / beta * jt.log(1 + (beta * x).minimum(threshold).exp()) + \
        (x - threshold / beta).maximum(0.0)


def hardtanh(x, min_val=-1, max_val=1):
    return jt.clamp(x, min_v=min_val, max_v=max_val)


def mish(x, inplace=False):
    return x * jt.tanh(jt.nn.softplus(x))

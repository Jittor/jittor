"""Vector functional implementations exposed through :mod:`jittor.nn`."""

import jittor as jt


def glu(input, dim=-1):
    r''' Applies the gated linear unit function

    .. math::
        \text{GLU}(a, b) = a \otimes \sigma(b)

    where ``input`` is split in half along ``dim`` to form ``a`` and ``b``,
    ``a`` is the first half and ``b`` the second half, and :math:`\sigma` is the
    sigmoid function. Torch-compatible.

    :param input: the input var
    :type input: jt.Var

    :param dim: the dimension on which to split the input. Default: -1
    :type dim: int

    Example:
        >>> x = jt.randn(4, 6)
        >>> y = nn.glu(x)   # y.shape == [4, 3]
    '''
    ndim = input.ndim
    if ndim == 0:
        raise RuntimeError("glu does not support scalars because halving size must be even")
    if dim < 0:
        dim += ndim
    size = input.shape[dim]
    if size % 2 != 0:
        raise RuntimeError(f"Halving dimension must be even, but dimension {dim} is size {size}")
    half = size // 2
    a, b = input.split([half, half], dim=dim)
    return a * b.sigmoid()


def normalize(input, p=2, dim=1, eps=1e-12):
    r''' Performs :math:`L_p` normalization of inputs over a specified dimension.

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}

    Torch-compatible functional interface. Note the torch-compatible default of
    ``eps=1e-12`` (clamping the denominator), as opposed to the additive ``eps``
    used by :func:`jittor.normalize`.

    :param input: input var of any shape
    :type input: jt.Var

    :param p: the exponent value in the norm formulation. Default: 2
    :type p: float

    :param dim: the dimension to reduce. Default: 1
    :type dim: int

    :param eps: small value to avoid division by zero. Default: 1e-12
    :type eps: float

    Example:
        >>> x = jt.randn(3, 4)
        >>> y = nn.normalize(x, dim=1)
    '''
    if p == 2:
        norm = (input * input).sum(dim, keepdims=True).sqrt()
    elif p == 1:
        norm = input.abs().sum(dim, keepdims=True)
    elif p == float("inf"):
        norm = input.abs().max(dim, keepdims=True)
    else:
        norm = (input.abs() ** p).sum(dim, keepdims=True) ** (1.0 / p)
    return input / norm.maximum(eps)


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    r''' Returns the cosine similarity between ``x1`` and ``x2`` along ``dim``.

    .. math::
        \text{similarity} = \frac{x_1 \cdot x_2}
            {\max(\lVert x_1 \rVert_2, \epsilon) \cdot \max(\lVert x_2 \rVert_2, \epsilon)}

    Torch-compatible.

    :param x1: first input var
    :type x1: jt.Var

    :param x2: second input var
    :type x2: jt.Var

    :param dim: dimension along which cosine similarity is computed. Default: 1
    :type dim: int

    :param eps: small value to avoid division by zero. Default: 1e-8
    :type eps: float

    Example:
        >>> a = jt.randn(4, 8)
        >>> b = jt.randn(4, 8)
        >>> sim = nn.cosine_similarity(a, b)   # sim.shape == [4]
    '''
    w12 = (x1 * x2).sum(dim)
    w1 = (x1 * x1).sum(dim)
    w2 = (x2 * x2).sum(dim)
    n12 = (w1 * w2).maximum(eps * eps).sqrt()
    return w12 / n12


def pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    r''' Computes the batchwise :math:`p`-norm distance between vectors.

    .. math::
        \lVert x_1 - x_2 + \epsilon \rVert_p

    Torch-compatible.

    :param x1: first input var
    :type x1: jt.Var

    :param x2: second input var
    :type x2: jt.Var

    :param p: the norm degree. Default: 2.0
    :type p: float

    :param eps: small value added to avoid division by zero. Default: 1e-6
    :type eps: float

    :param keepdim: whether to keep the reduced (vector) dimension. Default: False
    :type keepdim: bool

    Example:
        >>> a = jt.randn(4, 8)
        >>> b = jt.randn(4, 8)
        >>> d = nn.pairwise_distance(a, b)   # d.shape == [4]
    '''
    diff = (x1 - x2) + eps
    adiff = diff.abs()
    if p == 2:
        out = (diff * diff).sum(-1, keepdims=keepdim).sqrt()
    elif p == 1:
        out = adiff.sum(-1, keepdims=keepdim)
    elif p == float("inf"):
        out = adiff.max(-1, keepdims=keepdim)
    else:
        out = (adiff ** p).sum(-1, keepdims=keepdim) ** (1.0 / p)
    return out


def softsign(x):
    r''' Applies the element-wise function

    .. math::
        \text{SoftSign}(x) = \frac{x}{1 + |x|}

    Torch-compatible.

    :param x: the input var
    :type x: jt.Var

    Example:
        >>> a = jt.randn(3)
        >>> nn.softsign(a)
    '''
    return x / (1 + x.abs())

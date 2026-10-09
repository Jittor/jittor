"""Functional activation implementations exposed through :mod:`jittor.nn`."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import numpy as np

import jittor as jt
from jittor._runtime.dispatch import try_dispatch
from jittor.nn.functional._layout import channels_last_source

from ... import _arg_policy


_INPLACE_CONSEQUENCE = (
    "the input var is left untouched and a new one is returned, so none of the "
    "memory the flag asks for is saved"
)



#: Offers to take a residual add into the pass as well: var id -> (build,
#: whether the add's operands are in storage order). The add is a new Var
#: whose operands come back as new Python objects, so the offer cannot live on
#: the offering object alone; an entry lives exactly as long as that object
#: does (`_OfferLease`), since it holds what its pass reads.
_RESIDUAL_OFFERS = {}


class _OfferLease:
    """Removes residual offers when the object they were made on goes."""
    __slots__ = ("keys",)

    def __init__(self, keys):
        self.keys = keys

    def __del__(self):
        for key in self.keys:
            _RESIDUAL_OFFERS.pop(key, None)


def offer_activation(y, build, residual=None, storage=None):
    """Say that ``build(act)`` computes ``act(y)`` in the pass that makes ``y``.

    The offer describes the Var ``y`` holds now. An in-place op rebinds the
    same Python object to a new Var -- ``out = bn(out); out += identity;
    relu(out)``, every torchvision bottleneck -- and an offer read through the
    object afterwards applied the activation to the normalization alone,
    dropping the residual.

    ``residual(act, r)``, when given, computes ``act(y + r)`` in that pass: an
    activation of ``y`` plus a residual found through the add's operands,
    however the add was spelled. With ``storage`` -- the dense tensor ``y``
    is a channels-last view of -- the offer is found through it as well: an
    add of channels-last operands runs on their storage
    (`propagate_storage_layout`), and ``residual(act, r, True)`` then gets
    ``r`` in that storage order too.
    """
    y.__dict__["_fuse_activation"] = (y.id, build)
    if residual is not None:
        keys = [y.id]
        _RESIDUAL_OFFERS[y.id] = (residual, False)
        if storage is not None:
            _RESIDUAL_OFFERS[storage.id] = (residual, True)
            keys.append(storage.id)
        y.__dict__["_residual_offer"] = _OfferLease(keys)


def _fused_activation(x, act):
    """``act(x)`` from the pass that makes ``x``, if that pass offered it."""
    entry = getattr(x, "__dict__", {}).get("_fuse_activation")
    if entry is not None and entry[0] == x.id and not x.is_finished:
        _RESIDUAL_OFFERS.pop(entry[0], None)
        return entry[1](act)
    if not _RESIDUAL_OFFERS or x.is_finished:
        return None
    add, storage = x, False
    if x._producer_op() == "transpose" and channels_last_source(x) is not None:
        # An add kept channels-last hands out the NCHW view of a result it
        # computed on its operands' storage.
        add, storage = x._input(0), True
    if add._producer_op() != "binary.add":
        return None
    a, b = add._input(0), add._input(1)
    for y, r in ((a, b), (b, a)):
        offer = _RESIDUAL_OFFERS.get(y.id)
        if offer is None or offer[1] != storage:
            continue
        del _RESIDUAL_OFFERS[y.id]
        fused = offer[0](act, r, True) if storage else offer[0](act, r)
        if fused is not None:
            return fused
    return None


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
    if inplace:
        _arg_policy.ignored("jittor.nn.relu", "inplace", inplace,
                            _INPLACE_CONSEQUENCE)
    # A normalization that can apply the activation in its own last pass
    # (the training batch norm) says so on its unexecuted output.
    fused = _fused_activation(x, "relu")
    if fused is not None:
        return fused
    fast = try_dispatch("nn.relu", x, inplace=inplace)
    if fast is not None:
        return fast
    # One elementwise operator: it fuses into whatever produced `x` (a batch
    # norm, a residual add) and differentiates from its own output, so neither
    # the input nor a sign mask is kept for the backward. See `UnaryOp::grad`.
    return jt.unary(x, "relu")


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
    if inplace:
        _arg_policy.ignored("jittor.nn.leaky_relu", "inplace", inplace,
                            _INPLACE_CONSEQUENCE)
    fast = try_dispatch("nn.leaky_relu", x, scale=scale, inplace=inplace)
    if fast is not None:
        return fast
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


#: (half, one, 1/sqrt(2)) for the exact GELU, indexed by "is the input
#: float64". Python floats: against a floating tensor they keep its dtype, both
#: natively and under torch_compat, and take the frontend's native binary path.
#: Numpy scalars -- this used to hold them, against a float64 promotion the
#: frontend no longer does -- missed it, and three of a GELU's four operators
#: went through the Python promotion instead: 18.7 us to build one, against
#: 6.4 us for PyTorch to run it. Same bits either way.
_GELU_CONSTANTS = (
    (0.5, 1.0, 0.7071067811865476),
    (0.5, 1.0, 0.7071067811865476),
)


#: `src/bindings/pyjt/py_compat_fast.h`'s `_fast_gelu`.
_FAST_GELU = getattr(jt.core, "_fast_gelu", None)


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
    if approximate == 'none' and _FAST_GELU is not None:
        # The body below, built natively when the frontend's binary operators
        # are bound and no "nn.gelu" kernel is registered; None otherwise.
        fast = _FAST_GELU(x)
        if fast is not None:
            return fast
    fast = try_dispatch("nn.gelu", x, approximate=approximate)
    if fast is not None:
        return fast
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
        # `_jittor_dtype_name` is idempotent, so it ran three times per call --
        # twice on the *string* its own first call returned. The three typed
        # constants were rebuilt per call too; they depend on nothing but the
        # compute dtype, so there are exactly two sets of them.
        input_dtype = _jittor_dtype_name(x.dtype)
        low_precision = input_dtype in ('float16', 'bfloat16')
        compute_x = x.float32() if low_precision else x
        half, one, inv_sqrt2 = _GELU_CONSTANTS[input_dtype == 'float64']
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
    if inplace:
        _arg_policy.ignored("jittor.nn.silu", "inplace", inplace,
                            _INPLACE_CONSEQUENCE)
    # A normalization that can apply the activation in its own last pass
    # (group norm, see `group_norm_cuda.py`) says so on its unexecuted output.
    fused = _fused_activation(x, "silu")
    if fused is not None:
        return fused
    fast = try_dispatch("nn.silu", x, inplace=inplace)
    if fast is not None:
        return fast
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
    if inplace:
        _arg_policy.ignored("jittor.nn.mish", "inplace", inplace,
                            _INPLACE_CONSEQUENCE)
    return x * jt.tanh(jt.nn.softplus(x))

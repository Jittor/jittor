"""Torch numerical elementwise operations."""

def log1p(x):
    """Compute ``log(1 + x)`` elementwise."""
    from . import (
        jt,
    )
    return jt.log(1.0 + x)


def reciprocal(x):
    """Compute the elementwise multiplicative reciprocal."""
    return 1.0 / x


def lerp(input, end, weight):
    """Linearly interpolate between ``input`` and ``end``."""
    return input + weight * (end - input)


def softmax(input, dim=None, dtype=None, **kwargs):
    """Compute softmax values through Jittor's native nn owner.

    ``dtype=`` casts the input before the op, as Torch's does. It used to be
    swallowed here while ``Tensor.softmax`` (a second, separate closure) applied
    it, so ``torch.softmax(logits, -1, dtype=torch.float32)`` -- vLLM's sampler
    written in functional form -- silently computed in the input's narrow dtype.
    """
    from . import (
        _dtype_to_str,
        jt,
    )
    if dtype is not None:
        input = input.cast(_dtype_to_str(dtype))
    return jt.nn.softmax(input, dim=dim)


def log_softmax(input, dim=None, dtype=None, **kwargs):
    """Compute log-softmax values through Jittor's native nn owner.

    ``dtype=`` casts the input before the op; see ``softmax`` for why the
    functional spelling used to drop it.
    """
    from . import (
        _dtype_to_str,
        jt,
    )
    if dtype is not None:
        input = input.cast(_dtype_to_str(dtype))
    return jt.nn.log_softmax(input, dim=dim)


def relu(input, **kwargs):
    """Compute elementwise rectified linear activation."""
    from . import (
        jt,
    )
    return jt.nn.relu(input)


def _shape_as_tensor(input):
    """Return the static tensor shape as an int64 Jittor array."""
    from . import (
        jt,
        np,
    )
    return jt.array(np.asarray(input.shape, dtype=np.int64))


def isin(elements, test_elements, assume_unique=False, invert=False, **kwargs):
    """Test element membership using the captured native owner."""
    from . import (
        _NATIVE_ISIN,
    )
    return _NATIVE_ISIN(
        elements, test_elements, assume_unique=assume_unique, invert=invert)


def nan_to_num_(input, nan=0.0, posinf=None, neginf=None):
    """Replace non-finite values in-place and return the input tensor."""
    from . import (
        EXPECTED,
        swallowed,
    )
    result = input.nan_to_num(nan=nan, posinf=posinf, neginf=neginf)
    try:
        input.assign(result)
        return input
    except EXPECTED as exc:
        swallowed(
            "torch/installers/numerical.py nan_to_num_: input.assign(result); return input",
            exc,
        )
        return result


def _copysign_impl(input, other):
    from . import (
        jt,
    )
    sign = (other >= 0).float32() * 2 - 1
    return jt.abs(input) * sign


def copysign(input, other):
    """Copy the sign of ``other`` onto the magnitude of ``input``."""
    from . import (
        _copysign_impl,
    )
    return _copysign_impl(input, other)


def _xlogy_impl(input, other):
    from . import (
        jt,
    )
    return jt.ternary(
        input == 0, jt.zeros_like(input), input * jt.log(other))


def xlogy(input, other):
    """Return ``input * log(other)`` with the Torch ``xlogy(0, y) == 0`` rule."""
    from . import (
        _xlogy_impl,
    )
    return _xlogy_impl(input, other)


def _heaviside_impl(input, values):
    return (input > 0).float32() + (input == 0).float32() * values


def heaviside(input, values):
    """Return the elementwise Heaviside step function."""
    from . import (
        _heaviside_impl,
    )
    return _heaviside_impl(input, values)


def _signbit_impl(input):
    return input < 0


def signbit(input):
    """Return a boolean tensor identifying negative values."""
    from . import (
        _signbit_impl,
    )
    return _signbit_impl(input)


def _float_power_impl(input, exponent):
    from . import (
        jt,
    )
    if isinstance(exponent, jt.Var):
        exponent = exponent.float64()
    return input.float64() ** exponent


def float_power(input, exponent):
    """Raise tensors to a power using Torch's float64 computation policy."""
    from . import (
        _float_power_impl,
    )
    return _float_power_impl(input, exponent)


def _isclose_impl(a, b, rtol=1e-5, atol=1e-8, equal_nan=False, **kwargs):
    from . import (
        jt,
    )
    out = jt.abs(a - b) <= (atol + rtol * jt.abs(b))
    if equal_nan:
        out = out | (jt.isnan(a) & jt.isnan(b))
    return out


def isclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False, **kwargs):
    """Return an elementwise tensor indicating whether values are close."""
    from . import (
        _isclose_impl,
    )
    return _isclose_impl(
        a, b, rtol=rtol, atol=atol, equal_nan=equal_nan, **kwargs)


def _allclose_impl(a, b, rtol=1e-5, atol=1e-8, equal_nan=False, **kwargs):
    from . import (
        _isclose_impl,
    )
    return bool(_isclose_impl(
        a, b, rtol=rtol, atol=atol, equal_nan=equal_nan, **kwargs
    ).all().item())


def allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False, **kwargs):
    """Return a Python bool indicating whether all values are close."""
    from . import (
        _allclose_impl,
    )
    return _allclose_impl(
        a, b, rtol=rtol, atol=atol, equal_nan=equal_nan, **kwargs)


def _diff_impl(x, n=1, dim=-1, prepend=None, append=None):
    from . import (
        _diff,
    )
    return _diff(x, n=n, dim=dim, prepend=prepend, append=append)


def diff(x, n=1, dim=-1, prepend=None, append=None):
    """Compute consecutive differences along a tensor dimension."""
    from . import (
        _diff_impl,
    )
    return _diff_impl(x, n=n, dim=dim, prepend=prepend, append=append)


def square(x):
    """Return the elementwise square of a tensor."""
    return x * x


def _integer_result(dtype):
    """Whether a result of ``dtype`` is an integral one."""
    from jittor._core.dtypes import dtype_name as _jittor_dtype_name
    return _jittor_dtype_name(dtype).startswith(("int", "uint"))


def div(input, other, *, rounding_mode=None, out=None):
    """``torch.div``, including its one Torch-only keyword, ``rounding_mode``.

    The three modes are three different operations rather than three spellings
    of one: ``None`` is true division, ``"floor"`` is :func:`jt.floor_divide`,
    and ``"trunc"`` rounds **toward zero** -- which jittor has no primitive for,
    and which differs from ``"floor"`` exactly when the quotient is negative and
    inexact (``trunc(-7/3) = -2``, ``floor(-7/3) = -3``).

    Values and output dtypes measured against real torch 2.13 on CPU, over
    ``7,-7`` divided by ``3,3,-3,-3``:

        int32    None   -> float32  2.33, -2.33, -2.33,  2.33
        int32    trunc  -> int32    2,    -2,    -2,     2
        int32    floor  -> int32    2,    -3,    -3,     2
        float32  trunc  -> float32  2.0,  -2.0,  -2.0,   2.0
        float32  floor  -> float32  2.0,  -3.0,  -3.0,   2.0

    An integral pair therefore gives an integral result under either rounded
    mode and true division gives a float, which is what ``jt.floor_divide`` and
    ``jt.div`` already do. ``"trunc"`` on integers is computed in integer
    arithmetic (floor, corrected by one where the remainder is non-zero and the
    signs differ) so it stays exact past the float32 mantissa; on floats it is
    the quotient rounded toward zero.

    ``out`` is accepted for API shape compatibility and is not populated, the
    same boundary the other approximate fallbacks in this package keep.
    """
    from . import (
        _native_div,
        jt,
    )

    if rounding_mode not in (None, "trunc", "floor"):
        raise RuntimeError(
            "div expected rounding_mode to be one of None, 'trunc', or 'floor' "
            "but found %r" % (rounding_mode,))
    if rounding_mode is None:
        return _native_div(input, other)
    floored = jt.floor_divide(input, other)
    if rounding_mode == "floor":
        return floored
    if _integer_result(floored.dtype):
        # `%` follows the floor convention, so a zero remainder is exactly
        # "divisible", and the quotient needs correcting only when the two
        # operands have opposite signs.
        remainder = input % other
        opposite_signs = (input < 0) != (other < 0)
        return floored + (remainder != 0) * opposite_signs
    quotient = _native_div(input, other)
    return jt.ternary(quotient < 0, jt.ceil(quotient), jt.floor(quotient))


#: ``torch.divide`` is documented as an alias of ``torch.div`` and accepts the
#: same ``rounding_mode``; published as the same object so the two spellings
#: cannot drift apart.
divide = div


def _tensor_div(self, other, *, rounding_mode=None):
    """``Tensor.div`` -- the method spelling of the same operation.

    Torch's method form takes the same ``rounding_mode`` as the function form
    (measured against real torch 2.13: ``x.div(y, rounding_mode="trunc")`` gives
    the values *and* the dtype of ``torch.div(x, y, rounding_mode="trunc")``),
    so it has to reach this adapter instead of the native binary op that jittor
    generates from the ``div`` op name. ``div_``/``divide_`` keep the native path,
    and ``true_divide`` takes no ``rounding_mode`` in torch either.
    """
    return div(self, other, rounding_mode=rounding_mode)

"""Torch numerical complex operations."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

def complex(real, imag, **kwargs):
    """Construct a native complex tensor from real and imaginary parts."""
    from . import (
        jt,
    )
    return jt.nn.view_as_complex(jt.stack([real, imag], dim=-1))


def view_as_complex(input):
    """Interpret the trailing size-two dimension as a complex tensor."""
    from . import (
        jt,
    )
    return jt.nn.view_as_complex(input)


def view_as_real(input):
    """Expose native complex values as a trailing size-two real dimension."""
    from . import (
        jt,
    )
    return jt.nn.view_as_real(input)


def _is_complex_value(value):
    from . import (
        jt,
    )
    complex_type = jt.nn.ComplexNumber
    return isinstance(value, complex_type) or (
        isinstance(value, jt.Var) and "complex" in _jittor_dtype_name(value.dtype)
    )


def is_complex(input):
    from . import (
        _is_complex_value,
    )
    return _is_complex_value(input)


def real(input):
    from . import (
        jt,
    )
    return input.real if isinstance(input, (jt.nn.ComplexNumber, jt.Var)) else input


def imag(input):
    from . import (
        jt,
    )
    return input.imag if isinstance(input, (jt.nn.ComplexNumber, jt.Var)) else jt.zeros_like(input)


def conj(input):
    from . import (
        jt,
    )
    return input.conj() if isinstance(input, (jt.nn.ComplexNumber, jt.Var)) else input


def angle(input):
    from . import (
        jt,
    )
    return input.angle() if isinstance(input, (jt.nn.ComplexNumber, jt.Var)) else jt.zeros_like(input)


def abs(input):
    from . import (
        _native_abs,
        jt,
    )
    return input.abs() if isinstance(input, jt.nn.ComplexNumber) else _native_abs(input)


def polar(abs, angle, **kwargs):
    """Construct a native complex tensor from magnitude and phase."""
    from . import (
        jt,
    )
    return jt.nn.polar(abs, angle)

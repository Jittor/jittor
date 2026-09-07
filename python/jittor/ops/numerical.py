"""Numerical tensor operations."""

import numpy as np
import math
from jittor_core import Var
from .._runtime.dispatch import select_kernel, try_dispatch

def all(x, dim=()):
    import jittor as jt
    result = try_dispatch("tensor.all", x, dim)
    if result is not None:
        return result
    return jt.ops.all_(x, dim).bool()


def any(x,dim=()):
    import jittor as jt
    result = try_dispatch("tensor.any", x, dim)
    if result is not None:
        return result
    return jt.ops.any_(x, dim).bool()


def normalize(input, p=2, dim=1, eps=1e-12):
    r'''
    Performs L_p normalization of inputs over specified dimension.

    Args:

        input – input array of any shape

        p (float) – the exponent value in the norm formulation. Default: 2

        dim (int) – the dimension to reduce. Default: 1

        eps (float) – small value to avoid division by zero. Default: 1e-12

    .. note::
        This is now a thin alias for :func:`jittor.nn.normalize`; the two used
        to be separate implementations of the same name with **different
        semantics**, and this one was the odd one out. What changed here:

        * ``eps`` clamps the norm (``v / max(||v||_p, eps)``, torch's rule)
          instead of clamping the *sum of squares*, whose effective floor was
          ``sqrt(eps)``. With the old default of ``eps=1e-30`` the floor was
          ``1e-15``, so ``normalize([1e-20, 0])`` returned ``1e-5`` where torch
          returns ``1e-8``.
        * The default ``eps`` is torch's ``1e-12`` (was ``1e-30``).
        * ``p=1`` is protected at all. It used to divide by an unclamped sum of
          absolute values, so **a zero vector produced NaN**.
        * ``p=inf`` and other values of ``p`` work (used to hit an ``assert``).

    Example:

        >>> x = jt.random((6,3))
        [[0.18777736 0.9739261  0.77647036]
        [0.13710196 0.27282116 0.30533272]
        [0.7272278  0.5174613  0.9719775 ]
        [0.02566639 0.37504175 0.32676998]
        [0.0231761  0.5207773  0.70337296]
        [0.58966476 0.49547017 0.36724383]]

        >>> jt.normalize(x)
        [[0.14907198 0.7731768  0.61642134]
        [0.31750825 0.63181424 0.7071063 ]
        [0.5510936  0.39213243 0.736565  ]
        [0.05152962 0.7529597  0.656046  ]
        [0.02647221 0.59484214 0.80340654]
        [0.6910677  0.58067477 0.4303977 ]]
    '''
    from jittor.nn.functional.vector import normalize as _normalize
    return _normalize(input, p=p, dim=dim, eps=eps)


def hypot(a,b):
    import jittor as jt
    return jt.sqrt(a.sqr()+b.sqr())


_DEGREES_PER_RADIAN = 180.0 / np.pi

_RADIANS_PER_DEGREE = np.pi / 180.0

def rad2deg(x):
    return x * _DEGREES_PER_RADIAN


def deg2rad(x):
    return x * _RADIANS_PER_DEGREE


def arctan2(y,x):
    import jittor as jt
    angle = jt.zeros(x.shape,dtype=x.dtype)
    x = (x!=0.0).ternary(x, 1e-30)
    angle = (y/x).arctan()
    mask = (x<0)&(y<0)
    # mask is bool; `bool * python-float` promotes to float16 under the torch
    # dtype lattice, rounding pi to 3.14 (3.140625 after upcast) and breaking
    # the atol=1e-6 contract. Cast the mask to the angle's float dtype first so
    # pi keeps full precision (float32 -> 3.1415927) while still honoring a
    # genuinely float16/float64 input.
    angle = angle - mask.cast(angle.dtype)*np.pi
    mask = (x<0)&(y>=0)
    angle = angle + mask.cast(angle.dtype)*np.pi
    return angle


atan2 = arctan2

def log2(x):
    import jittor as jt
    return jt.log(x)/math.log(2.0)


def safe_log(x):
    import jittor as jt
    return jt.safe_clip(x, 1e-30, 1e30).log()


def _simple_for(x, func):
    import jittor as jt
    with jt.flag_scope(compile_options={"FLAGS: -O2 ":1}):
        src = f'''
        __inline_static__
        @python.jittor.auto_parallel(1)
        void kernel(int n0, int i0, in0_type* _x, out0_type* y) {{
            using namespace std;
            auto x = _x[i0];
            y[i0] = {func};
        }}
        kernel(in0->num, 0, in0_p, out0_p);
        '''
        return jt.code(x.shape, "bool", [x], cpu_src=src, cuda_src=src)


def _isnan_acl(x):
    import jittor as jt
    x = x if isinstance(x, Var) else jt.array(x)
    if not x.dtype.is_float(): return jt.zeros(x.shape, "bool")
    return jt.logical_not((x >= 0) | (x <= 0))


def _isinf_acl(x):
    import jittor as jt
    x = x if isinstance(x, Var) else jt.array(x)
    if not x.dtype.is_float(): return jt.zeros(x.shape, "bool")
    return x.abs() == float("inf")


def _isfinite_acl(x):
    import jittor as jt
    x = x if isinstance(x, Var) else jt.array(x)
    if not x.dtype.is_float(): return jt.ones(x.shape, "bool")
    return x.abs() < float("inf")


def _classify_value(dtype):
    """The C++ expression these kernels must test, per input dtype.

    Every one of them used to test ``float(x)`` unconditionally. That is a
    *narrowing* cast for float64: 1e300 is an ordinary finite double and becomes
    inf as a float, so ``jt.isinf`` said True for it on CPU and CUDA while the
    ACL path -- which never narrows -- said False. Same public API, different
    answer per backend.

    float16/bfloat16 still widen to float. That direction is lossless, and
    neither type has a std::isnan overload to call instead.
    """
    return "x" if dtype in ("float32", "float64") else "float(x)"


def _classify(x, expr, acl_body, integral):
    """One body for isnan/isinf/isfinite and the two signed-infinity variants.

    ``integral`` is the answer for a dtype that has neither nan nor infinity --
    torch's answer too: isnan/isinf are all-False over an integer tensor and
    isfinite is all-True. That used to fall out of casting the integer to float;
    saying it directly is what lets the float kernel keep the input's own type.
    """
    import jittor as jt
    x = x if isinstance(x, Var) else jt.array(x)
    if not x.dtype.is_float():
        return (jt.ones if integral else jt.zeros)(x.shape, "bool")
    return select_kernel("misc.classify", x)(x, expr, acl_body)


def _classify_acl(x, expr, acl_body):
    return acl_body(x)


def _classify_code(x, expr, acl_body):
    import jittor as jt
    return jt.misc._simple_for(x, expr(jt.misc._classify_value(str(x.dtype))))


def isnan(x):
    import jittor as jt
    return jt.misc._classify(
        x, lambda v: f"isnan({v})", jt.misc._isnan_acl, False)


def isfinite(x):
    import jittor as jt
    return jt.misc._classify(
        x, lambda v: f"!isnan({v}) && !isinf({v})", jt.misc._isfinite_acl, True)


def isinf(x):
    import jittor as jt
    return jt.misc._classify(
        x, lambda v: f"isinf({v})", jt.misc._isinf_acl, False)


def isneginf(x):
    import jittor as jt
    return jt.misc._classify(
        x, lambda v: f"x<0 && isinf({v})",
        lambda v: (v < 0) & jt.misc._isinf_acl(v), False)


def isposinf(x):
    import jittor as jt
    return jt.misc._classify(
        x, lambda v: f"x>0 && isinf({v})",
        lambda v: (v > 0) & jt.misc._isinf_acl(v), False)


def rsqrt(x):
    import jittor as jt
    return 1/jt.sqrt(x)


def all_equal(a: Var, b: Var) -> bool:
    return (a == b).all().item()


def _to_float(x: Var) -> Var:
    if x.dtype != "float64": x = x.float()
    return x


class Finfo:
    pass


bfloat16_finfo = Finfo()

def finfo(dtype):
    import jittor as jt
    if dtype == "bfloat16":
        return jt.misc.bfloat16_finfo
    if callable(dtype) and hasattr(dtype, "__name__"):
        dtype = dtype.__name__.split('.')[-1]
    else:
        dtype = str(dtype).split('.')[-1]
    return np.finfo(dtype)


def iinfo(dtype):
    if callable(dtype) and hasattr(dtype, "__name__"):
        dtype = dtype.__name__.split('.')[-1]
    else:
        dtype = str(dtype).split('.')[-1]
    return np.iinfo(dtype)


def expm1(x):
    import jittor as jt
    return jt.exp(x) - 1

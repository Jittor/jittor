# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Special-function unary OpInfos -- the differentiable special funcs jittor ships
in core (not the torch-compat shim): ``erfinv``, ``lgamma``, ``digamma`` and the
linear angle converters ``deg2rad`` / ``rad2deg``.

These matter because each has a NON-TRIVIAL hand-written backward kernel:
  * ``erfinv'(x) = sqrt(pi)/2 * exp(erfinv(x)^2)``  (a closed form, easy to stub wrong),
  * ``lgamma'(x) = digamma(x)`` and ``digamma'(x) = polygamma(1, x)``  (the very
    derivatives the Gamma/Dirichlet distributions rely on for their KL/entropy grads).
A forward-only check would pass a backward that returns the wrong special function.

The forward oracle is SciPy (``scipy.special.{erfinv,gammaln,digamma}``) -- a fully
independent implementation -- so ``test_ops`` pins the value, and ``gradcheck`` (float64,
CPU) pins the analytic derivative against finite differences. ``lgamma``/``digamma`` are
``jt.Function`` objects, invoked via ``.apply`` (their public call form, see
``jittor/distributions.py``). Domains stay on the smooth, finite region (``erfinv`` off
the +/-1 poles; ``lgamma``/``digamma`` strictly positive, off the non-positive-integer
poles).
"""
import scipy.special as _sp

from ._refs import *  # noqa: F401,F403  (make_tensor, SampleInput, np, jt, nn, F, cu)
from ..core import UnaryUfuncInfo, skip


# ----------------------------------------------------------------- numpy/scipy refs
def erfinv_ref(x):   return _sp.erfinv(x)
def lgamma_ref(x):   return _sp.gammaln(x)
def digamma_ref(x):  return _sp.digamma(x)
def deg2rad_ref(x):  return np.deg2rad(x)
def rad2deg_ref(x):  return np.rad2deg(x)


# ----------------------------------------------------------------- jittor callables
def _lgamma(x):
    return jt.lgamma.apply(x)


def _digamma(x):
    return jt.digamma.apply(x)


op_db = [
    # erfinv: smooth on (-1, 1); stay off the +/-1 poles. 2nd derivative exists
    # (it is 2x*erfinv'(x) composed of exp/erfinv), so gradgrad stays on.
    UnaryUfuncInfo("erfinv", ref=erfinv_ref, domain=(-0.9, 0.9), op=jt.erfinv),

    # lgamma / digamma: strictly positive domain (poles at 0, -1, -2, ...). Their
    # backward emits digamma / polygamma(1); gradgrad differentiates THAT again.
    # digamma's own backward (trigamma) is not provided as a separate Function, so
    # the 2nd derivative of lgamma / digamma is not represented -> gradgrad off.
    #
    # lgamma's backward (digamma) is accurate enough that float64 finite-difference
    # gradcheck passes. digamma's backward (trigamma) rides a ~1e-7-accurate kernel:
    # gradcheck divides the forward's ~1e-7 absolute error by eps~1e-6, so the FD
    # *numerical* gradient is round-off-dominated and spuriously diverges from the
    # CORRECT analytic one (verified == scipy.special.polygamma(1) in
    # test_special_grad.py). So skip digamma's FD gradcheck -- but keep
    # supports_autograd=True: the backward kernel is still checked CPU-vs-CUDA by
    # test_device_parity, and against scipy by test_special_grad.
    UnaryUfuncInfo("lgamma", ref=lgamma_ref, domain=(0.3, 4.0), op=_lgamma,
                   supports_gradgrad=False),
    UnaryUfuncInfo("digamma", ref=digamma_ref, domain=(0.3, 4.0), op=_digamma,
                   supports_gradgrad=False,
                   skips=[skip("test_gradcheck",
                               reason="float32-precision kernel: float64 FD gradcheck is "
                                      "round-off dominated; backward verified analytically "
                                      "vs scipy polygamma(1) in test_special_grad.py")]),

    # linear angle converters: derivative is the constant pi/180 (resp. 180/pi);
    # trivially smooth, gradgrad fine.
    UnaryUfuncInfo("deg2rad", ref=deg2rad_ref, op=jt.deg2rad),
    UnaryUfuncInfo("rad2deg", ref=rad2deg_ref, op=jt.rad2deg),
]

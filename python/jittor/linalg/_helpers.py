# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Haoyang Peng <2247838039@qq.com>
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Shared array algebra and native-complex bridges."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
import numpy as np


def _linalg_errors_raise(array_module):
    """Make a numpy_code callback's array module raise on singular input.

    On CUDA the callback receives CuPy, whose ``linalg.solve``/``inv``/
    ``cholesky`` return NaN for a singular (or non-positive-definite) matrix
    where NumPy raises ``LinAlgError`` -- so the same call raised on CPU and
    went silently wrong on CUDA. CuPy raises the same error under
    ``cupyx.errstate(linalg="raise")``.
    """
    import contextlib
    if getattr(array_module, "__name__", "") != "cupy":
        return contextlib.nullcontext()
    import cupyx
    return cupyx.errstate(linalg="raise")


def _is_batched(value):
    # vmap's BatchedVar, recognised without importing jittor.vmap on the hot
    # path (and without touching BatchedVar.__getattr__, which raises for
    # unknown names).
    return type(value).__name__ == "BatchedVar" and type(value).__module__ == "jittor.vmap"


def _batching_aware(rule):
    """Route a BatchedVar argument to vmap's rule for this function.

    vmap installs its batching patches on first use, so a reference saved
    before that (``solve = jt.linalg.solve``) used to call the plain function
    with BatchedVar arguments. The check lives in the function itself; the
    rule receives the undecorated function as its base case.
    """
    import functools

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if any(_is_batched(arg) for arg in args):
                # The submodule, not the `jittor.vmap` function attribute.
                from jittor.vmap import dispatch_linalg
                return dispatch_linalg(rule, fn, *args, **kwargs)
            return fn(*args, **kwargs)
        wrapper.__wrapped_linalg__ = fn
        return wrapper
    return decorate


def _reconnect(cached, live):
    """Give a stop-grad'd numpy_code result a real, higher-order-differentiable
    gradient path, without changing its value.

    ``cached`` is a value already computed once inside a ``jt.Function``'s
    ``execute()`` -- ``jt.Function`` tapes and stop-grads its own execute()
    inputs/outputs, so a value cached there (e.g. ``self.mx = inv_via_numpy(x)``)
    is permanently detached from ``x``'s autograd graph. ``live`` is the SAME
    mathematical quantity recomputed via a live, autograd-connected recursive
    call (e.g. calling the public ``inv(x)`` again from inside ``inv``'s own
    ``grad()``, closing over the still-live outer ``x``).

    Returns a Var whose VALUE is exactly ``cached`` (bit-identical --
    ``live - live.detach()`` is a subtraction of a tensor from its own detached
    copy, so it is exactly zero regardless of ``live``'s own numerical
    precision) but whose GRADIENT w.r.t. ``live``'s inputs matches ``live``'s.
    This keeps first-order precision identical to a plain numpy_code op (no
    duplicated rounding from an independent recompute) while still making
    second/third-order derivatives available through the ``live`` path -- the
    standard "stop-gradient trick".
    """
    return cached + (live - live.detach())


def _conj_T(v):
    """Conjugate transpose of the last two axes of a Var (a plain transpose
    for real dtypes)."""
    return v.transpose(-1, -2).conj()


def _eye_like(ref, k):
    """A (k, k) identity Var in `ref`'s (real) dtype, broadcastable over batches."""
    import jittor as jt
    return jt.array(np.eye(k), dtype=_jittor_dtype_name(ref.dtype))


def _diag_of(m):
    """The diagonal of the last two axes of a square Var, differentiably."""
    k = m.shape[-1]
    return (m * _eye_like(m.real if "complex" in _jittor_dtype_name(m.dtype) else m, k)).sum(-1)


def _diag_embed(vec, k):
    """(..., k) -> (..., k, k) with `vec` on the diagonal, differentiably."""
    real = vec.real if "complex" in _jittor_dtype_name(vec.dtype) else vec
    return vec.unsqueeze(-1) * _eye_like(real, k)


def _transpose(x):
    """Batched transpose: swap the last two axes."""
    return np.swapaxes(x, -1, -2)


def _conj_transpose(x):
    """Batched conjugate transpose (the Hermitian adjoint)."""
    return np.conj(np.swapaxes(x, -1, -2))


def _matmul(a, b):
    """Batched matrix product over the last two axes."""
    return np.einsum('...ij,...jk->...ik', a, b)


def _stack_to_complex(x):
    """float ``[..., 2]`` (real, imag) stack -> complex array."""
    return x[..., 0] + 1j * x[..., 1]


def _complex_to_stack(x):
    """complex array -> float ``[..., 2]`` (real, imag) stack."""
    return np.stack([np.real(x), np.imag(x)], axis=-1)


def _is_native_complex(x):
    # A native complex64 Var (NOT an nn.ComplexNumber, which is a plain object).
    import jittor as jt
    return isinstance(x, jt.Var) and "complex" in _jittor_dtype_name(x.dtype)


def _native_to_cn(z):
    # native complex64 [...]  ->  nn.ComplexNumber (differentiable, via P1 bridge)
    import jittor as jt
    from ..nn import ComplexNumber
    return ComplexNumber(jt.nn.view_as_real(z), is_concat_value=True)


def _cn_to_native(cn):
    # nn.ComplexNumber  ->  native complex64/complex128 [...]  (differentiable,
    # via P1 bridge). cn.value is a float32-or-float64 [..., 2] stack;
    # _real2_to_complex rebuilds whichever native complex width matches --
    # NOT the width-fixed _real2_to_complex64, which would silently downcast
    # a complex128 result (e.g. from svd()/qr() on a native complex128 input)
    # back to complex64 on the way out of this bridge.
    from ..nn.functional.complex import _real2_to_complex
    return _real2_to_complex(cn.value)

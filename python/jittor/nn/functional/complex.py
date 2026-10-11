"""Native complex tensor bridge operations."""

from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import numpy as np

import jittor as jt
from jittor.backends.cuda.kernels.nn.complex_views import (
    COMPLEX64_TO_REAL2_CUDA_SOURCE,
    REAL2_TO_COMPLEX64_CUDA_SOURCE,
    COMPLEX128_TO_REAL2_CUDA_SOURCE,
    REAL2_TO_COMPLEX128_CUDA_SOURCE,
)

# Native complex64 <-> float32[..., 2] bridge. This lets FFT / linalg use the native
# complex64 dtype while the internal kernels still consume a real/imag float pair:
#   _complex64_to_real2 : complex64[...]   -> float32[..., 2]   (torch.view_as_real)
#   _real2_to_complex64 : float32[..., 2]  -> complex64[...]     (torch.view_as_complex)
# view_as_real/view_as_complex prefer the zero-copy reinterpret_view core op when available,
# and fall back to isolated jt.code kernels otherwise. Both are wrapped as jt.Function with
# each other as the adjoint backward, so the bridge is autograd-transparent on CPU+CUDA.
_complex64_imag_unit_cache = None


def _complex64_imag_unit():
    global _complex64_imag_unit_cache
    if _complex64_imag_unit_cache is None:
        _complex64_imag_unit_cache = jt.array(np.array(1j, dtype="complex64"))
    return _complex64_imag_unit_cache


def _complex64_to_real2_raw(z):
    reinterpret_view = getattr(jt, "reinterpret_view", None)
    if reinterpret_view is not None:
        return reinterpret_view(z, list(z.shape) + [2], "float32")
    # flatten to 1-D so the jt.code kernel is shape-agnostic, then restore the [..., 2] tail.
    n = 1
    for s in z.shape:
        n *= s
    flat = jt.code(
        [n, 2],
        "float32",
        [z.reshape([n])],
        cpu_src="""
        for (int i=0; i<in0_shape0; i++) {
            @out(i,0) = @in0(i).real;
            @out(i,1) = @in0(i).imag;
        }""",
        cuda_src=COMPLEX64_TO_REAL2_CUDA_SOURCE,
    )
    return flat.reshape(list(z.shape) + [2])


def _real2_to_complex64_raw(x):
    assert x.shape[-1] == 2, f"view_as_complex expects last dim 2, got shape {x.shape}"
    # The pair this reads is float32, because the complex dtype it builds is
    # complex64 and jittor has no complex128 (KI-COMPLEX-001, and
    # docs/notes/complex-dtype.md for what registering one would take). A
    # float64 pair therefore has nowhere to go, and saying so here names the
    # dtype and the operation -- `reinterpret_view` further down could only
    # report the arithmetic: "byte size mismatch, input [8,2] float64 target
    # [8] complex64".
    # `_jittor_dtype_name`, not `str(x.dtype)`: in Torch-compatibility mode the
    # Var that arrives here is a `torch.Tensor`, whose `dtype` stringifies as
    # "torch.float32". The check was reporting a float32 pair as the wrong
    # dtype -- "needs a float32 pair, got torch.float32" -- and refusing every
    # `view_as_complex` under the shim, which is the whole FFT surface.
    if _jittor_dtype_name(x.dtype) != "float32":
        raise NotImplementedError(
            "view_as_complex builds complex64 and needs a float32 pair, got %s. "
            "jittor has no complex128 (KI-COMPLEX-001); cast the input with "
            ".float32() if the precision is not needed." % x.dtype
        )
    reinterpret_view = getattr(jt, "reinterpret_view", None)
    if reinterpret_view is not None:
        return reinterpret_view(x, list(x.shape[:-1]), "complex64")
    # real[..., 2] -> native complex64. Use one code kernel instead of two getitem ops
    # plus mixed complex arithmetic; this is the hot path for RoPE view_as_complex.
    n = 1
    for s in x.shape[:-1]:
        n *= s
    out_shape = list(x.shape[:-1])
    flat = jt.code(
        [n],
        "complex64",
        [x.reshape([n, 2])],
        cpu_src="""
        for (int i=0; i<in0_shape0; i++) {
            @out(i) = complex64(float(@in0(i,0)), float(@in0(i,1)));
        }""",
        cuda_src=REAL2_TO_COMPLEX64_CUDA_SOURCE,
    )
    return flat.reshape(out_shape)


class _Complex64ToReal2(jt.Function):
    def execute(self, z):
        return _complex64_to_real2_raw(z)

    def grad(self, g):  # adjoint of view_as_real is view_as_complex
        return _real2_to_complex64_raw(g)


class _Real2ToComplex64(jt.Function):
    def execute(self, x):
        return _real2_to_complex64_raw(x)

    def grad(self, g):  # adjoint of view_as_complex is view_as_real
        return _complex64_to_real2_raw(g)


def _complex64_to_real2(z):
    return _Complex64ToReal2.apply(z)


def _real2_to_complex64(x):
    return _Real2ToComplex64.apply(x)


# Width-generic versions of the two bridges above: float32 pair <-> complex64
# *or* float64 pair <-> complex128, picking the target/expected width from
# whichever side is already concrete (the component dtype on the way in,
# the complex dtype on the way out). `_real2_to_complex64`/`_complex64_to_real2`
# stay float32-only above; the public view and linalg bridges use the
# width-generic helpers below to preserve float64 components as complex128.
_COMPLEX_REAL_DTYPE = {"complex64": "float32", "complex128": "float64"}
_REAL_COMPLEX_DTYPE = {"float32": "complex64", "float64": "complex128"}


def _complex_to_real2_raw(z):
    real_dtype = _COMPLEX_REAL_DTYPE[_jittor_dtype_name(z.dtype)]
    reinterpret_view = getattr(jt, "reinterpret_view", None)
    if reinterpret_view is not None:
        return reinterpret_view(z, list(z.shape) + [2], real_dtype)
    n = 1
    for s in z.shape:
        n *= s
    if real_dtype == "float32":
        cpu_src = """
        for (int i=0; i<in0_shape0; i++) {
            @out(i,0) = @in0(i).real;
            @out(i,1) = @in0(i).imag;
        }"""
        cuda_src = COMPLEX64_TO_REAL2_CUDA_SOURCE
    else:
        cpu_src = """
        for (int i=0; i<in0_shape0; i++) {
            @out(i,0) = @in0(i).real;
            @out(i,1) = @in0(i).imag;
        }"""
        cuda_src = COMPLEX128_TO_REAL2_CUDA_SOURCE
    flat = jt.code([n, 2], real_dtype, [z.reshape([n])], cpu_src=cpu_src, cuda_src=cuda_src)
    return flat.reshape(list(z.shape) + [2])


def _real2_to_complex_raw(x):
    assert x.shape[-1] == 2, f"view_as_complex expects last dim 2, got shape {x.shape}"
    real_name = _jittor_dtype_name(x.dtype)
    if real_name not in _REAL_COMPLEX_DTYPE:
        raise NotImplementedError(
            "view_as_complex needs a float32 or float64 pair, got %s." % x.dtype
        )
    complex_dtype = _REAL_COMPLEX_DTYPE[real_name]
    reinterpret_view = getattr(jt, "reinterpret_view", None)
    if reinterpret_view is not None:
        return reinterpret_view(x, list(x.shape[:-1]), complex_dtype)
    n = 1
    for s in x.shape[:-1]:
        n *= s
    out_shape = list(x.shape[:-1])
    if complex_dtype == "complex64":
        cpu_src = """
        for (int i=0; i<in0_shape0; i++) {
            @out(i) = complex64(float(@in0(i,0)), float(@in0(i,1)));
        }"""
        cuda_src = REAL2_TO_COMPLEX64_CUDA_SOURCE
    else:
        cpu_src = """
        for (int i=0; i<in0_shape0; i++) {
            @out(i) = complex128(double(@in0(i,0)), double(@in0(i,1)));
        }"""
        cuda_src = REAL2_TO_COMPLEX128_CUDA_SOURCE
    flat = jt.code([n], complex_dtype, [x.reshape([n, 2])], cpu_src=cpu_src, cuda_src=cuda_src)
    return flat.reshape(out_shape)


class _ComplexToReal2(jt.Function):
    def execute(self, z):
        return _complex_to_real2_raw(z)

    def grad(self, g):  # adjoint of view_as_real is view_as_complex
        # Preserve the AD chain even when execute uses the opaque code fallback.
        return None if g is None else _real2_to_complex(g)


class _Real2ToComplex(jt.Function):
    def execute(self, x):
        return _real2_to_complex_raw(x)

    def grad(self, g):  # adjoint of view_as_complex is view_as_real
        return None if g is None else _complex_to_real2(g)


def _complex_to_real2(z):
    return _ComplexToReal2.apply(z)


def _real2_to_complex(x):
    return _Real2ToComplex.apply(x)


def polar(abs: jt.Var, angle: jt.Var) -> jt.Var:
    # torch.polar: magnitude `abs`, phase `angle` -> native complex (complex64
    # for a float32 pair, complex128 for a float64 pair). Differentiable
    # through the P1 bridge.
    assert abs.shape == angle.shape
    return _real2_to_complex(jt.stack([abs * angle.cos(), abs * angle.sin()], dim=-1))


def view_as_complex(x: jt.Var) -> jt.Var:
    # torch.view_as_complex: real [..., 2] -> native complex64/complex128
    # (matching the input's own float32/float64 width). Callers that still
    # need the legacy pair use nn.ComplexNumber(...) directly.
    assert x.shape[-1] == 2, f"view_as_complex expects last dim 2, got shape {x.shape}"
    return _real2_to_complex(x)


def view_as_real(x) -> jt.Var:
    # torch.view_as_real: complex -> real [..., 2]. Polymorphic across the native complex64/
    # complex128 dtypes (Phase 6 bridge, differentiable) and the legacy nn.ComplexNumber
    # (real/imag pair, always float32).
    if isinstance(x, jt.nn.ComplexNumber):
        return jt.stack([x.value[..., 0], x.value[..., 1]], dim=-1)
    assert "complex" in _jittor_dtype_name(x.dtype), (
        f"view_as_real expects a complex Var or ComplexNumber, got dtype {_jittor_dtype_name(x.dtype)}"
    )
    return _complex_to_real2(x)


def _var_real(self):
    if "complex" in _jittor_dtype_name(self.dtype):
        return jt.nn.view_as_real(self)[..., 0]
    return self


def _var_imag(self):
    if "complex" in _jittor_dtype_name(self.dtype):
        return jt.nn.view_as_real(self)[..., 1]
    return jt.zeros_like(self)


def _var_angle(self):
    return jt.atan2(self.imag, self.real)


__all__ = ["polar", "view_as_complex", "view_as_real"]

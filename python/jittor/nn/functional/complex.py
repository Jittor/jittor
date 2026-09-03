"""Native complex tensor bridge operations."""

import numpy as np

import jittor as jt

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
        # The legacy facade exposed this private cache directly. Keep that
        # observable alias synchronized while the implementation lives here.
        jt.nn._complex64_imag_unit_cache = _complex64_imag_unit_cache
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
        cuda_src="""
        __global__ void k(@ARGS_DEF) {
            @PRECALC
            int i = blockIdx.x*blockDim.x + threadIdx.x;
            if (i < in0_shape0) { @out(i,0) = @in0(i).real; @out(i,1) = @in0(i).imag; }
        }
        int n = in0_shape0; k<<<(n+63)/64, 64>>>(@ARGS);""",
    )
    return flat.reshape(list(z.shape) + [2])


def _real2_to_complex64_raw(x):
    assert x.shape[-1] == 2, f"view_as_complex expects last dim 2, got shape {x.shape}"
    reinterpret_view = getattr(jt, "reinterpret_view", None)
    if reinterpret_view is not None:
        return reinterpret_view(x, list(x.shape[:-1]) or [1], "complex64")
    # real[..., 2] -> native complex64. Use one code kernel instead of two getitem ops
    # plus mixed complex arithmetic; this is the hot path for RoPE view_as_complex.
    n = 1
    for s in x.shape[:-1]:
        n *= s
    out_shape = list(x.shape[:-1]) or [1]
    flat = jt.code(
        [n],
        "complex64",
        [x.reshape([n, 2])],
        cpu_src="""
        for (int i=0; i<in0_shape0; i++) {
            @out(i) = complex64(float(@in0(i,0)), float(@in0(i,1)));
        }""",
        cuda_src="""
        __global__ void k(@ARGS_DEF) {
            @PRECALC
            int i = blockIdx.x*blockDim.x + threadIdx.x;
            if (i < in0_shape0) {
                @out(i) = complex64(float(@in0(i,0)), float(@in0(i,1)));
            }
        }
        int n = in0_shape0; k<<<(n+63)/64, 64>>>(@ARGS);""",
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


def polar(abs: jt.Var, angle: jt.Var) -> jt.Var:
    # torch.polar: magnitude `abs`, phase `angle` -> native complex64 (Phase 6 migration off
    # ComplexNumber). Differentiable through the P1 bridge.
    assert abs.shape == angle.shape
    return _real2_to_complex64(jt.stack([abs * angle.cos(), abs * angle.sin()], dim=-1))


def view_as_complex(x: jt.Var) -> jt.Var:
    # torch.view_as_complex: real [..., 2] -> native complex64 (Phase 6 migration). Callers that
    # still need the legacy pair use nn.ComplexNumber(...) directly.
    assert x.shape[-1] == 2, f"view_as_complex expects last dim 2, got shape {x.shape}"
    return _real2_to_complex64(x)


def view_as_real(x) -> jt.Var:
    # torch.view_as_real: complex -> real [..., 2]. Polymorphic across the native complex64
    # dtype (Phase 6 bridge, differentiable) and the legacy nn.ComplexNumber (real/imag pair).
    if isinstance(x, jt.nn.ComplexNumber):
        return jt.stack([x.value[..., 0], x.value[..., 1]], dim=-1)
    assert "complex" in str(x.dtype), (
        f"view_as_real expects a complex64 Var or ComplexNumber, got dtype {x.dtype}"
    )
    return _complex64_to_real2(x)


def _var_real(self):
    if "complex" in str(self.dtype):
        return jt.nn.view_as_real(self)[..., 0]
    return self


def _var_imag(self):
    if "complex" in str(self.dtype):
        return jt.nn.view_as_real(self)[..., 1]
    return jt.zeros_like(self)


def _var_angle(self):
    return jt.atan2(self.imag, self.real)


__all__ = ["polar", "view_as_complex", "view_as_real"]

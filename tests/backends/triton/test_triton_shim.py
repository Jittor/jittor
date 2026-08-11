"""Torch-grade tests for the jittor ``triton`` compatibility shim.

The shim (``jittor.triton_shim``) makes ``import triton`` / ``@triton.jit`` work
on a jittor environment without the real ``triton`` PyPI package, and its naive
executor (``triton_shim/launch.py``) can actually *run* a narrow but useful
class of fused kernels by tracing the kernel body once and lowering ``tl.*`` to
jittor ops over whole Vars:

  * 1-D elementwise kernels   (vector add, fused ``x*y + z``, ``tl.where``)
  * single-row / single-program reductions  (fused softmax: rowmax/exp/rowsum)

This module locks that behaviour the same way the other torch-grade suites do:
every kernel result is compared against an INDEPENDENT numpy reference and run on
BOTH CPU and CUDA (when the build has it). It also locks the shim's safety
contract ("clear error beats silent wrong"): a kernel the executor cannot run
faithfully (a *multi-row* 2-D softmax) must raise, not return garbage.

Run:  python -m pytest tests/backends/triton/test_triton_shim.py
      python -m pytest tests/backends/triton/test_triton_shim.py
"""
import unittest
import numpy as np
import jittor as jt

triton = None
tl = None

# Exercise CPU always; add CUDA when the build has it (mirrors the other suites).
_DEVICES = [("cpu", 0)] + ([("cuda", 1)] if jt.has_cuda else [])


def both_devices(fn):
    """Run ``fn(device_name)`` once per available device under the right flag."""
    for name, use_cuda in _DEVICES:
        with jt.flag_scope(use_cuda=use_cuda):
            fn(name)


# --------------------------------------------------------------------------- #
#  @triton.jit kernels under test
# --------------------------------------------------------------------------- #
def _define_kernels():
    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        """Canonical 1-D elementwise vector add with a tail mask."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    @triton.jit
    def fma_kernel(x_ptr, y_ptr, z_ptr, out_ptr, n, BLOCK: tl.constexpr):
        """Genuinely fused elementwise: out = x * y + z (one traced pass)."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        z = tl.load(z_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x * y + z, mask=mask)

    @triton.jit
    def relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        """Fused elementwise exercising tl.where: relu(x) = where(x > 0, x, 0)."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.where(x > 0.0, x, 0.0)
        tl.store(out_ptr + offs, y, mask=mask)

    @triton.jit
    def gelu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        """Fused elementwise chain: sigmoid-approx GELU = x * sigmoid(1.702 * x)."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = x * tl.sigmoid(1.702 * x)
        tl.store(out_ptr + offs, y, mask=mask)

    @triton.jit
    def softmax_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        """Fused softmax over a single row with reduction and broadcast."""
        offs = tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=-float("inf"))
        xmax = tl.max(x, axis=0)
        num = tl.exp(x - xmax)
        den = tl.sum(num, axis=0)
        tl.store(out_ptr + offs, num / den, mask=mask)

    @triton.jit
    def softmax_2d_kernel(x_ptr, out_ptr, row_stride, n_cols, BLOCK: tl.constexpr):
        """Unsupported multi-row softmax used to verify a clear failure."""
        row = tl.program_id(0)
        row_start = x_ptr + row * row_stride
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(row_start + cols, mask=mask, other=-float("inf"))
        xmax = tl.max(x, axis=0)
        num = tl.exp(x - xmax)
        den = tl.sum(num, axis=0)
        tl.store(out_ptr + row * row_stride + cols, num / den, mask=mask)

    return add_kernel, fma_kernel, relu_kernel, gelu_kernel, softmax_kernel, softmax_2d_kernel


def setUpModule():
    global triton, tl
    global add_kernel, fma_kernel, relu_kernel, gelu_kernel
    global softmax_kernel, softmax_2d_kernel
    from jittor import triton_shim as shim_module
    from jittor.triton_shim import language as language_module

    triton = shim_module.triton
    tl = language_module
    (
        add_kernel,
        fma_kernel,
        relu_kernel,
        gelu_kernel,
        softmax_kernel,
        softmax_2d_kernel,
    ) = _define_kernels()


# --------------------------------------------------------------------------- #
class Base(unittest.TestCase):
    def ac(self, got, ref, atol=1e-5, rtol=1e-5, msg=""):
        g = np.asarray(got)
        r = np.asarray(ref)
        self.assertEqual(tuple(g.shape), tuple(r.shape),
                         f"shape {g.shape}!={r.shape}; {msg}")
        np.testing.assert_allclose(g, r, atol=atol, rtol=rtol, err_msg=msg)


class TestHostUtils(Base):
    """Host-callable triton utilities are real implementations."""

    def test_cdiv(self):
        for x, d in [(10, 3), (1024, 256), (1000, 256), (7, 1), (0, 4)]:
            self.assertEqual(triton.cdiv(x, d), -(-x // d), f"cdiv({x},{d})")
            self.assertEqual(tl.cdiv(x, d), -(-x // d), f"tl.cdiv({x},{d})")

    def test_next_power_of_2(self):
        cases = {0: 1, 1: 1, 2: 2, 3: 4, 5: 8, 1000: 1024, 1024: 1024, 1025: 2048}
        for n, want in cases.items():
            self.assertEqual(triton.next_power_of_2(n), want, f"npo2({n})")

    def test_is_shim(self):
        # advertise itself as the shim so callers can branch on it
        self.assertTrue(getattr(triton, "__triton_shim__", False))
        self.assertTrue(triton.is_shim())


class TestElementwiseKernels(Base):
    """1-D elementwise @triton.jit kernels run end-to-end through the shim."""

    def test_vector_add(self):
        # n NOT a multiple of BLOCK -> exercises the tail-mask path.
        for n, BLOCK in [(1024, 256), (1000, 256), (1, 64), (333, 128)]:
            rs = np.random.RandomState(n)
            xn = rs.randn(n).astype("float32")
            yn = rs.randn(n).astype("float32")
            ref = xn + yn

            def body(dev, n=n, BLOCK=BLOCK, xn=xn, yn=yn, ref=ref):
                x = jt.array(xn)
                y = jt.array(yn)
                out = jt.zeros(n, dtype="float32")
                grid = (triton.cdiv(n, BLOCK),)
                add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
                self.ac(out.numpy(), ref, atol=1e-6,
                        msg=f"vector_add n={n} BLOCK={BLOCK} {dev}")
            both_devices(body)

    def test_fused_mul_add(self):
        n, BLOCK = 777, 128
        rs = np.random.RandomState(1)
        xn = rs.randn(n).astype("float32")
        yn = rs.randn(n).astype("float32")
        zn = rs.randn(n).astype("float32")
        ref = xn * yn + zn

        def body(dev):
            x, y, z = jt.array(xn), jt.array(yn), jt.array(zn)
            out = jt.zeros(n, dtype="float32")
            grid = (triton.cdiv(n, BLOCK),)
            fma_kernel[grid](x, y, z, out, n, BLOCK=BLOCK)
            self.ac(out.numpy(), ref, atol=1e-5, msg=f"fused x*y+z {dev}")
        both_devices(body)

    def test_relu_where(self):
        n, BLOCK = 512, 128
        xn = np.random.RandomState(3).randn(n).astype("float32")
        ref = np.maximum(xn, 0.0)

        def body(dev):
            x = jt.array(xn)
            out = jt.zeros(n, dtype="float32")
            grid = (triton.cdiv(n, BLOCK),)
            relu_kernel[grid](x, out, n, BLOCK=BLOCK)
            self.ac(out.numpy(), ref, atol=1e-6, msg=f"relu/where {dev}")
        both_devices(body)

    def test_gelu_fused_chain(self):
        n, BLOCK = 640, 256
        xn = np.random.RandomState(4).randn(n).astype("float32")
        ref = xn * (1.0 / (1.0 + np.exp(-1.702 * xn)))

        def body(dev):
            x = jt.array(xn)
            out = jt.zeros(n, dtype="float32")
            grid = (triton.cdiv(n, BLOCK),)
            gelu_kernel[grid](x, out, n, BLOCK=BLOCK)
            self.ac(out.numpy(), ref, atol=1e-5, msg=f"gelu fused {dev}")
        both_devices(body)


class TestFusedReduction(Base):
    """Fused softmax: reduction (rowmax/rowsum) + broadcast, single program."""

    def test_softmax_1d(self):
        for n in [16, 128, 257]:
            xn = (np.random.RandomState(n).randn(n).astype("float32")) * 3.0
            e = np.exp(xn - xn.max())
            ref = e / e.sum()

            def body(dev, n=n, xn=xn, ref=ref):
                x = jt.array(xn)
                out = jt.zeros(n, dtype="float32")
                BLOCK = triton.next_power_of_2(n)
                softmax_kernel[(1,)](x, out, n, BLOCK=BLOCK)
                got = out.numpy()
                self.ac(got, ref, atol=1e-6, msg=f"softmax n={n} {dev}")
                # softmax is a probability distribution: rows sum to 1.
                self.assertAlmostEqual(float(got.sum()), 1.0, places=5,
                                       msg=f"softmax sum!=1 n={n} {dev}")
            both_devices(body)


class TestSafetyContract(Base):
    """Lock 'clear error beats silent wrong': kernels the executor cannot run
    faithfully must RAISE NotImplementedError, never return a wrong result."""

    def test_multirow_softmax_raises(self):
        # A multi-row (per-program-row) softmax is NOT whole-array-safe; the
        # executor must refuse it loudly rather than silently reduce wrong cols.
        M, N = 4, 8
        xn = np.random.RandomState(0).randn(M, N).astype("float32")

        def body(dev):
            x = jt.array(xn)
            out = jt.zeros((M, N), dtype="float32")
            with self.assertRaises(NotImplementedError):
                softmax_2d_kernel[(M,)](x, out, N, N, BLOCK=N)
        both_devices(body)

    def test_direct_call_raises(self):
        # Calling a @triton.jit kernel directly (not kernel[grid](...)) is an
        # error in real triton too; the shim must raise, not no-op.
        with self.assertRaises(NotImplementedError):
            add_kernel(1, 2, 3, 4, BLOCK=8)


if __name__ == "__main__":
    unittest.main(verbosity=2)

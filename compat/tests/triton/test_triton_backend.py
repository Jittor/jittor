"""Torch-grade tests for the jittor *real* triton backend (bridge mode).

Unlike the naive tracer exercised by ``test_triton_shim.py`` (which lowers a
narrow 1-D-elementwise subset of ``tl.*`` to whole-Var jittor ops), this suite
runs **real** ``@triton.jit`` kernels: upstream triton compiles them to
PTX/cubin and jittor launches the cubin on its own ``Var`` device pointers via
the CUDA driver API (see :mod:`jittor.compat.triton.backend`). That makes it able
to run kernels the tracer cannot — ``tl.dot`` matmul, 2-D row softmax, fused
layernorm — which is the whole point of "real triton support".

The suite is skipped unless BOTH are true:
  * a genuine upstream ``triton`` is importable (not the jittor shim), and
  * jittor was built with CUDA (``jt.has_cuda``).

Every kernel result is checked against an INDEPENDENT numpy reference.

Run:  python -m pytest compat/tests/triton/test_triton_backend.py
      python -m pytest compat/tests/triton/test_triton_backend.py
"""

from _helpers import capability as _test_capability
import ctypes
import importlib.util
import os
import unittest
from unittest import mock
import numpy as np

import jittor as jt


_HAVE = bool(_test_capability.check_accelerator('cuda', backend=jt).enabled and importlib.util.find_spec("triton") is not None)
_shim = None
triton = None
tl = None


# --------------------------------------------------------------------------- #
#  kernels (defined only when the test module starts executing)
# --------------------------------------------------------------------------- #
def _define_kernels():

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    @triton.jit
    def fma_kernel(x_ptr, y_ptr, z_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        z = tl.load(z_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x * y + z, mask=mask)

    @triton.jit
    def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                      stride_am, stride_ak, stride_bk, stride_bn,
                      stride_cm, stride_cn,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
            # allow_tf32=False -> true IEEE fp32 so we can check against numpy
            # tightly (TF32, triton's default on Ampere+, is ~1e-3 relative).
            acc += tl.dot(a, b, allow_tf32=False)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=mask)

    @triton.jit
    def softmax_kernel(out_ptr, in_ptr, in_stride, out_stride, n_cols,
                       BLOCK: tl.constexpr):
        row = tl.program_id(0)
        in_row = in_ptr + row * in_stride
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(in_row + cols, mask=mask, other=-float("inf"))
        x = x - tl.max(x, axis=0)
        num = tl.exp(x)
        den = tl.sum(num, axis=0)
        tl.store(out_ptr + row * out_stride + cols, num / den, mask=mask)

    @triton.jit
    def mul2_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) * 2.0, mask=m)

    @triton.autotune(configs=[triton.Config({'BLOCK': 128}, num_warps=4),
                              triton.Config({'BLOCK': 1024}, num_warps=8)],
                     key=['n'])
    @triton.jit
    def add_autotuned(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(o_ptr + offs,
                 tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)

    @triton.jit
    def attn_fwd(Q, K, V, Out,
                 stride_qm, stride_qd, stride_kn, stride_kd,
                 stride_vn, stride_vd, stride_om, stride_od,
                 M, N, sm_scale,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
        # Flash-attention forward (online softmax): two tl.dot, tl.trans,
        # masking, and >48KB dynamic shared memory — the canonical hard kernel.
        pid_m = tl.program_id(0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        q = tl.load(Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                    mask=offs_m[:, None] < M, other=0.0)
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for start_n in range(0, N, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k = tl.load(K + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                        mask=offs_n[:, None] < N, other=0.0)
            qk = tl.dot(q, tl.trans(k), allow_tf32=False) * sm_scale
            qk = tl.where(offs_n[None, :] < N, qk, -float("inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_ij[:, None])
            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v = tl.load(V + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                        mask=offs_n[:, None] < N, other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p, v, allow_tf32=False)
            m_i = m_ij
        acc = acc / l_i[:, None]
        tl.store(Out + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
                 acc, mask=offs_m[:, None] < M)

    @triton.jit
    def layernorm_kernel(X, Y, W, B, stride, N, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        X += row * stride
        Y += row * stride
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / N
        xmean = tl.where(mask, x - mean, 0.0)
        var = tl.sum(xmean * xmean, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        y = xmean * rstd * w + b
        tl.store(Y + cols, y, mask=mask)

    return (
        add_kernel,
        fma_kernel,
        matmul_kernel,
        softmax_kernel,
        mul2_kernel,
        add_autotuned,
        attn_fwd,
        layernorm_kernel,
    )


def setUpModule():
    global _shim, triton, tl
    global add_kernel, fma_kernel, matmul_kernel, softmax_kernel
    global mul2_kernel, add_autotuned, attn_fwd, layernorm_kernel
    if not _HAVE:
        return
    try:
        import jittor.compat.triton as shim_module
        import triton as triton_module
        import triton.language as language_module

        if not shim_module.backend.is_available():
            raise RuntimeError("real Triton backend is unavailable")
    except Exception as error:
        raise unittest.SkipTest("real Triton backend is unavailable: {}".format(error))
    _shim = shim_module
    triton = triton_module
    tl = language_module
    (
        add_kernel,
        fma_kernel,
        matmul_kernel,
        softmax_kernel,
        mul2_kernel,
        add_autotuned,
        attn_fwd,
        layernorm_kernel,
    ) = _define_kernels()


@unittest.skipUnless(_HAVE, "real upstream triton + CUDA not available")
class TestTritonBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from _helpers.runtime_policy import fixture_stack
        _test_policy_stack = fixture_stack(cls, class_scope=True)
        try:
            super().setUpClass()
            cls._previous_use_cuda = jt.introspection.policy.runtime.use_cuda
            try:
                _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=1))
                if not _shim.activate_bridge():
                    raise unittest.SkipTest("jittor Triton bridge is unavailable")
            except BaseException:
                _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=cls._previous_use_cuda))
                raise
        except BaseException:
            _test_policy_stack.close()
            raise

    @classmethod
    def tearDownClass(cls):
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            try:
                _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=cls._previous_use_cuda))
            finally:
                super().tearDownClass()

    def ac(self, got, ref, atol=1e-5, rtol=1e-5, msg=""):
        g, r = np.asarray(got), np.asarray(ref)
        self.assertEqual(tuple(g.shape), tuple(r.shape), f"shape {g.shape}!={r.shape}; {msg}")
        np.testing.assert_allclose(g, r, atol=atol, rtol=rtol, err_msg=msg)

    def test_vector_add(self):
        for n, BLOCK in [(4096, 1024), (1000, 256), (1, 64), (333, 128)]:
            rs = np.random.RandomState(n)
            xn, yn = rs.randn(n).astype("float32"), rs.randn(n).astype("float32")
            x, y = jt.array(xn), jt.array(yn)
            out = jt.zeros(n, dtype="float32")
            add_kernel[(triton.cdiv(n, BLOCK),)](x, y, out, n, BLOCK=BLOCK)
            self.ac(out.numpy(), xn + yn, atol=1e-6, msg=f"add n={n}")

    def test_fused_mul_add(self):
        n, BLOCK = 777, 128
        rs = np.random.RandomState(1)
        xn, yn, zn = (rs.randn(n).astype("float32") for _ in range(3))
        x, y, z = jt.array(xn), jt.array(yn), jt.array(zn)
        out = jt.zeros(n, dtype="float32")
        fma_kernel[(triton.cdiv(n, BLOCK),)](x, y, z, out, n, BLOCK=BLOCK)
        self.ac(out.numpy(), xn * yn + zn, atol=1e-5, msg="x*y+z")

    def test_matmul_tl_dot(self):
        # The defining test: a real tl.dot matmul the naive tracer cannot run.
        for (M, N, K) in [(256, 256, 256), (128, 64, 192)]:
            BM = BN = BK = 32
            rs = np.random.RandomState(M + N + K)
            an = rs.randn(M, K).astype("float32")
            bn = rs.randn(K, N).astype("float32")
            a, b = jt.array(an), jt.array(bn)
            c = jt.zeros((M, N), dtype="float32")
            grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
            matmul_kernel[grid](a, b, c, M, N, K, K, 1, N, 1, N, 1,
                                BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
            self.ac(c.numpy(), an @ bn, atol=1e-2, rtol=1e-3, msg=f"matmul {M}x{N}x{K}")

    def test_softmax_2d(self):
        # Multi-row softmax (one program per row) — the tracer REFUSES this; the
        # real backend runs it.
        M, Ncols = 64, 781
        xn = (np.random.RandomState(7).randn(M, Ncols).astype("float32")) * 3.0
        e = np.exp(xn - xn.max(axis=1, keepdims=True))
        ref = e / e.sum(axis=1, keepdims=True)
        x = jt.array(xn)
        out = jt.zeros((M, Ncols), dtype="float32")
        BLOCK = triton.next_power_of_2(Ncols)
        softmax_kernel[(M,)](out, x, Ncols, Ncols, Ncols, BLOCK=BLOCK)
        got = out.numpy()
        self.ac(got, ref, atol=1e-5, msg="softmax_2d")
        np.testing.assert_allclose(got.sum(axis=1), np.ones(M), atol=1e-4)

    def test_dtype_fp16(self):
        n, BLOCK = 2048, 256
        rs = np.random.RandomState(21)
        xn, yn = rs.randn(n).astype("float16"), rs.randn(n).astype("float16")
        x, y = jt.array(xn), jt.array(yn)
        out = jt.zeros(n, dtype="float16")
        add_kernel[(triton.cdiv(n, BLOCK),)](x, y, out, n, BLOCK=BLOCK)
        self.ac(out.numpy().astype("f4"),
                (xn.astype("f4") + yn.astype("f4")), atol=1e-2, msg="fp16 add")

    def test_dtype_bf16(self):
        n, BLOCK = 2048, 256
        xn = np.random.RandomState(22).randn(n).astype("float32")
        x = jt.array(xn).cast("bfloat16")
        out = jt.zeros(n, dtype="bfloat16")
        mul2_kernel[(triton.cdiv(n, BLOCK),)](x, out, n, BLOCK=BLOCK)
        self.ac(out.float32().numpy(), xn * 2.0, atol=5e-2, msg="bf16 mul2")

    def test_autotune(self):
        n = 4096
        rs = np.random.RandomState(23)
        xn, yn = rs.randn(n).astype("float32"), rs.randn(n).astype("float32")
        x, y = jt.array(xn), jt.array(yn)
        out = jt.zeros(n, dtype="float32")
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
        add_autotuned[grid](x, y, out, n)
        self.ac(out.numpy(), xn + yn, atol=1e-6, msg="autotuned add")

    def test_flash_attention(self):
        # Flagship: a real flash-attention forward kernel run on jittor Vars,
        # checked against a numpy softmax-attention reference.
        M = N = 128
        D = 64
        scale = 1.0 / np.sqrt(D)
        rs = np.random.RandomState(0)
        Qn = rs.randn(M, D).astype("float32")
        Kn = rs.randn(N, D).astype("float32")
        Vn = rs.randn(N, D).astype("float32")
        s = (Qn @ Kn.T) * scale
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s)
        ref = (e / e.sum(axis=1, keepdims=True)) @ Vn
        Q, K, V = jt.array(Qn), jt.array(Kn), jt.array(Vn)
        O = jt.zeros((M, D), dtype="float32")
        BM = BN = 64
        attn_fwd[(triton.cdiv(M, BM),)](Q, K, V, O, D, 1, D, 1, D, 1, D, 1,
                                        M, N, scale, BLOCK_M=BM, BLOCK_N=BN, BLOCK_D=D)
        self.ac(O.numpy(), ref, atol=1e-4, rtol=1e-3, msg="flash-attention fwd")

    def test_layernorm(self):
        M, N = 32, 512
        rs = np.random.RandomState(11)
        xn = rs.randn(M, N).astype("float32")
        wn = rs.randn(N).astype("float32")
        bn = rs.randn(N).astype("float32")
        eps = 1e-5
        mean = xn.mean(axis=1, keepdims=True)
        var = xn.var(axis=1, keepdims=True)
        ref = (xn - mean) / np.sqrt(var + eps) * wn + bn
        X, W, B = jt.array(xn), jt.array(wn), jt.array(bn)
        Y = jt.zeros((M, N), dtype="float32")
        BLOCK = triton.next_power_of_2(N)
        layernorm_kernel[(M,)](X, Y, W, B, N, N, eps, BLOCK=BLOCK)
        self.ac(Y.numpy(), ref, atol=1e-3, rtol=1e-3, msg="layernorm")


@unittest.skipUnless(_HAVE, "real upstream triton + CUDA not available")
class TestBridgeLaunchStream(unittest.TestCase):
    """The bridge has to launch inside jittor's own stream order.

    jittor runs its kernels, its copies and its library calls on
    ``cudaStreamPerThread`` (``compute_stream`` in
    ``backends/cuda/runtime/driver.cc``). The legacy default stream does not
    order against it, so a bridge kernel left on the legacy stream is an
    unordered race with jittor's scheduler *and* its allocator: jittor can hand
    an operand or output block to something else while the kernel is still
    reading it, which surfaces as a racy ``cudaErrorIllegalAddress``.

    That is not hypothetical -- it is the hazard the ``jt.sync_all(True)`` after
    every launch in ``run`` exists to paper over, and the fast-sync path (on by
    default whenever ``JITTOR_TORCH_SHIM`` is set) skips it.
    """

    def _recorded_stream(self):
        """The stream value handed to cuLaunchKernel, without launching."""
        from jittor.compat.triton import backend as tb

        driver = tb._Driver.get(0)
        seen = []

        class _Recorder:
            def __call__(self, func, gx, gy, gz, bx, by, bz, shared,
                         stream, params, extra):
                seen.append(stream.value)
                # Stop here: the point is the argument, not a real launch.
                raise RuntimeError("recorded")

        original = driver.lib.cuLaunchKernel
        driver.lib.cuLaunchKernel = _Recorder()
        try:
            with self.assertRaises(RuntimeError):
                driver.launch(ctypes.c_void_p(1), (1, 1, 1), (1, 1, 1), 0,
                              ctypes.c_void_p(0))
        finally:
            driver.lib.cuLaunchKernel = original
        return seen

    def test_launch_targets_the_per_thread_stream(self):
        # 0x2 is `cudaStreamPerThread`, what `compute_stream` returns, and the
        # same value as the driver API's `CU_STREAM_PER_THREAD`.
        self.assertEqual(self._recorded_stream(), [0x2])

    def test_the_legacy_stream_is_still_reachable_by_env(self):
        with mock.patch.dict(os.environ,
                             {"JITTOR_TRITON_LEGACY_STREAM": "1"}):
            self.assertEqual(self._recorded_stream(), [None])


@unittest.skipUnless(_HAVE, "real upstream triton + CUDA not available")
class TestDriverIsPerDevice(unittest.TestCase):
    """The bridge's driver has to follow the operands' device.

    A ``CUmodule``/``CUfunction`` belongs to the context it was loaded into and a
    primary context is per device, so a rank whose operands live on CUDA device 1
    cannot be served by device 0's handles. Pinning device 0 made the first
    triton launch of a TP2 rank-1 request run on the wrong device against
    device-1 pointers: a sticky ``cudaErrorIllegalAddress`` reported later, at
    whatever CUDA call came next (measured on the H3 modulation kernel, whose
    operands were all in bounds and identical to rank 0's).
    """

    def test_a_driver_is_cached_per_cuda_device(self):
        from jittor.compat.triton import backend as tb

        count = int(getattr(jt, "device_count", lambda: 1)())
        if count < 2:
            self.skipTest("needs at least two visible CUDA devices")
        d0 = tb._Driver.get(0)
        d1 = tb._Driver.get(1)
        self.assertIs(d0, tb._Driver.get(0))
        self.assertIsNot(d0, d1)
        self.assertEqual((d0.ordinal, d1.ordinal), (0, 1))
        self.assertNotEqual(d0.ctx.value, d1.ctx.value)


@unittest.skipUnless(_HAVE, "real upstream triton + CUDA not available")
class TestGuardedBounceRequiresContiguous(unittest.TestCase):
    """A strided operand must not be bounced.

    The over-read guard copies a small operand into a guarded buffer with one
    flat ``copy_dtod`` of ``numel * elsize`` bytes from ``data_ptr()`` and hands
    the kernel the bounce pointer -- while the caller's own stride arguments
    still describe the original layout. A contiguous operand survives that; a
    strided one does not: the copy takes the wrong elements and the kernel then
    walks off the copied payload reading the zeroed guard, so the launch
    succeeds and returns silently wrong numbers.

    MiniMax-H3 hit this on its AdaLN modulation, whose ``chunk`` views have row
    stride ``6 * hidden``: the fused kernel diverged from its own eager
    reference by 40 (cos 0.985) on the real tensors while agreeing bitwise on
    contiguous ones, and the DiT produced title cards instead of the prompted
    scene. This drives the same shape through ``matmul_kernel``'s explicit
    strides, with a contiguous control.
    """

    def _prepare(self):
        # Done per test rather than in `setUp` so a by-path runner that calls the
        # test method directly (no `setUpModule`/`setUpClass`/`setUp`) still
        # exercises the kernels.
        global triton, tl
        if triton is None or tl is None:
            setUpModule()
        if not _shim.activate_bridge():
            self.skipTest("jittor Triton bridge is unavailable")
        # The model code passes the shim's torch-shaped tensors, which is the
        # operand flavour this regression is about. `jittor.compat.torch`
        # installs itself as `torch` for the process.
        import jittor.compat.torch  # noqa: F401
        import torch as _torch
        if not callable(getattr(_torch, "tensor", None)) or not callable(getattr(_torch, "empty", None)):
            self.skipTest("no torch-shaped tensor namespace for the strided operand")
        return _torch

    def _run_matmul(self, torch_ns, a, b, M, N, K, stride_am):
        c = torch_ns.empty((M, N), dtype=torch_ns.float32, device="cuda:0")
        BM = BN = BK = 32
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        matmul_kernel[grid](a, b, c, M, N, K, stride_am, 1, N, 1, N, 1,
                            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
        jt.sync_all(True)
        return np.asarray(c.float().cpu().numpy(), dtype=np.float64)

    def test_strided_operand_survives_the_guard(self):
        """The failing case: the operand is a row-strided view.

        This is MiniMax-H3's AdaLN modulation shape -- a `chunk` of a
        `[M, k*H]` projection, i.e. row stride `k*H` -- driven through
        ``matmul_kernel``'s explicit strides. Before the fix the guard bounced
        it and the result was silently wrong.
        """
        torch_ns = self._prepare()
        M, N, K = 64, 64, 64
        rs = np.random.RandomState(7)
        wide = rs.randn(M, 2 * K).astype("float32")
        bn = rs.randn(K, N).astype("float32")

        a_view = torch_ns.tensor(wide, device="cuda:0")[:, :K]
        b = torch_ns.tensor(bn, device="cuda:0")
        jt.sync_all(True)
        self.assertFalse(bool(a_view.is_contiguous()),
                         "the view must be strided for this test to mean anything")

        got = self._run_matmul(torch_ns, a_view, b, M, N, K, 2 * K)
        np.testing.assert_allclose(got, wide[:, :K] @ bn, atol=1e-2, rtol=1e-3,
                                   err_msg="matmul with a row-strided operand")

    def test_contiguous_control_still_matches(self):
        # The same values and kernel with a contiguous operand: the guard may
        # bounce this one, and it must stay correct.
        torch_ns = self._prepare()
        M, N, K = 64, 64, 64
        rs = np.random.RandomState(8)
        an = rs.randn(M, K).astype("float32")
        bn = rs.randn(K, N).astype("float32")
        a = torch_ns.tensor(an, device="cuda:0")
        b = torch_ns.tensor(bn, device="cuda:0")
        jt.sync_all(True)
        self.assertTrue(bool(a.is_contiguous()))
        got = self._run_matmul(torch_ns, a, b, M, N, K, K)
        np.testing.assert_allclose(got, an @ bn, atol=1e-2, rtol=1e-3,
                                   err_msg="matmul with a contiguous operand")

    def test_the_helper_reports_layout(self):
        from jittor.compat.triton import backend as tb

        torch_ns = self._prepare()
        M, K = 8, 6
        wide = torch_ns.tensor(np.arange(M * 2 * K, dtype="float32").reshape(M, 2 * K),
                               device="cuda:0")
        jt.sync_all(True)
        self.assertTrue(tb._tensor_is_contiguous(wide))
        self.assertFalse(tb._tensor_is_contiguous(wide[:, :K]))
        self.assertTrue(tb._tensor_is_contiguous(wide[:, :K].contiguous()))

        # jittor Vars reach the bridge too, where `_storage_is_contiguous` is the
        # authority. Whether a slice is a view or a copy is jittor's business, so
        # only assert agreement with that authority when it yields a bool.
        def authority(value):
            flag = getattr(value, "_storage_is_contiguous", None)
            flag = flag() if callable(flag) else flag
            return flag if isinstance(flag, bool) else None

        var = jt.array(np.arange(M * 2 * K, dtype="float32").reshape(M, 2 * K))
        for candidate in (var, var[:, :K]):
            expected = authority(candidate)
            if expected is not None:
                self.assertEqual(tb._tensor_is_contiguous(candidate), expected)


class TestLaunchFollowsItsProducers(unittest.TestCase):
    """Operands produced immediately before a launch must be the ones it reads.

    The barrier ``run`` puts before packing submits the operand graph without
    waiting for it (`jt.sync_all(device_sync=False)`, which still plans,
    allocates and enqueues). Correctness therefore rests on *stream order*:
    jittor schedules its ops on ``cudaStreamPerThread`` and the bridge launches
    on that same stream (``_launch_stream``), so the kernel cannot start before
    the ops that write its operands have.

    If that stops holding -- a launch sent to another stream, or a barrier that
    stops submitting -- the kernel reads whatever the operand's buffer held
    before, which here is the *previous* iteration's value, and the launch still
    returns cleanly with a plausible-looking answer.
    """

    def _prepare(self):
        global triton, tl
        if triton is None or tl is None:
            setUpModule()
        if not _shim.activate_bridge():
            self.skipTest("jittor Triton bridge is unavailable")

    def test_each_launch_reads_the_value_its_producer_just_wrote(self):
        self._prepare()
        n, BLOCK = 4096, 1024
        rs = np.random.RandomState(7)
        xn, yn = rs.randn(n).astype("float32"), rs.randn(n).astype("float32")
        y = jt.array(yn)
        grid = (triton.cdiv(n, BLOCK),)

        # `x` is a fresh pending op every step, so nothing else materialises it;
        # a kernel that ran before its producer would sum the *previous* step's
        # `x`. The output is fresh per step too: this test is about operand
        # freshness, and re-reading one output Var across launches mixes in a
        # separate (pre-existing) staleness of jittor's view of a buffer a
        # kernel wrote through its own pointer.
        for step in range(1, 6):
            x = jt.array(xn) * float(step)
            out = jt.empty(n, dtype="float32")
            add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
            jt.sync_all(True)
            got = np.asarray(out.numpy(), dtype=np.float64)
            np.testing.assert_allclose(
                got, xn.astype(np.float64) * step + yn, atol=1e-5,
                err_msg="step %d: the launch read a stale operand" % step)


class TestTheLaunchBarrierNamesItsOperands(unittest.TestCase):
    """The barrier before packing must name the operands, not the process.

    ``jt.sync_all(False)`` collects *every* leaf var alive -- it walks
    ``runtime_holder_state().holders()`` and keeps each one with no consumers --
    so what it costs is the size of the live holder set rather than the work a
    launch needs. Measured on an idle graph: 1.2 us with nothing alive, 138 us at
    10,000 live holders, 752 us at 50,000, against a flat ~25 us for
    ``jt.sync([o], False)`` at any of those sizes. One MiniMax-H3 autocast VAE
    decode reaches this barrier 4,536 times.

    A broad sweep is not *wrong*, which is why no correctness test catches it:
    it materialises at least as much as the operands need. So pin the scope.
    Only ``sync_all(False)`` -- the sweep that skips the device wait -- is
    asserted against; ``sync_all(True)`` is the conservative mode's own
    before/after barrier and is a different thing.
    """

    def _prepare(self):
        global triton, tl
        if triton is None or tl is None:
            setUpModule()
        if not _shim.activate_bridge():
            self.skipTest("jittor Triton bridge is unavailable")

    def test_the_barrier_takes_operands_rather_than_the_whole_process(self):
        self._prepare()
        n, BLOCK = 1024, 256
        grid = (triton.cdiv(n, BLOCK),)
        xn = np.arange(n, dtype="float32")

        def launch():
            # `x` has a pending producer, so a barrier that does not submit it
            # would leave the kernel reading an unbacked buffer.
            x = jt.array(xn) * 2.0 + 1.0
            y = jt.array(xn)
            out = jt.empty(n, dtype="float32")
            add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
            return out

        launch()                       # warm up: compile the cubin
        jt.sync_all(True)

        seen = []
        real_sync, real_sync_all = jt.sync, jt.sync_all

        def recording(name, real):
            def call(*args, **kwargs):
                seen.append((name, args))
                return real(*args, **kwargs)
            return call

        jt.sync = recording("sync", real_sync)
        jt.sync_all = recording("sync_all", real_sync_all)
        try:
            out = launch()
        finally:
            jt.sync, jt.sync_all = real_sync, real_sync_all

        np.testing.assert_allclose(
            np.asarray(out.numpy(), dtype=np.float64),
            xn.astype(np.float64) * 3.0 + 1.0, atol=1e-5,
            err_msg="the launch did not read the operand its producer just wrote")

        named = [a for (name, a) in seen if name == "sync" and a and a[0]]
        self.assertTrue(named, "the launch never named its operands to jt.sync")
        self.assertNotIn(
            ("sync_all", (False,)), seen,
            "the launch swept the whole process with jt.sync_all(False); name the "
            "operand Vars instead, or every launch pays for every live holder")


if __name__ == "__main__":
    unittest.main(verbosity=2)

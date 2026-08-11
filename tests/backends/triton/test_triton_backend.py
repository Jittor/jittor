"""Torch-grade tests for the jittor *real* triton backend (bridge mode).

Unlike the naive tracer exercised by ``test_triton_shim.py`` (which lowers a
narrow 1-D-elementwise subset of ``tl.*`` to whole-Var jittor ops), this suite
runs **real** ``@triton.jit`` kernels: upstream triton compiles them to
PTX/cubin and jittor launches the cubin on its own ``Var`` device pointers via
the CUDA driver API (see :mod:`jittor.triton_shim.backend`). That makes it able
to run kernels the tracer cannot — ``tl.dot`` matmul, 2-D row softmax, fused
layernorm — which is the whole point of "real triton support".

The suite is skipped unless BOTH are true:
  * a genuine upstream ``triton`` is importable (not the jittor shim), and
  * jittor was built with CUDA (``jt.has_cuda``).

Every kernel result is checked against an INDEPENDENT numpy reference.

Run:  python -m pytest tests/backends/triton/test_triton_backend.py
      python -m pytest tests/backends/triton/test_triton_backend.py
"""
import importlib.util
import unittest
import numpy as np

import jittor as jt


_HAVE = bool(jt.has_cuda and importlib.util.find_spec("triton") is not None)
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
        import jittor.triton_shim as shim_module
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
        super().setUpClass()
        cls._previous_use_cuda = jt.flags.use_cuda
        try:
            jt.flags.use_cuda = 1
            if not _shim.activate_bridge():
                raise unittest.SkipTest("jittor Triton bridge is unavailable")
        except BaseException:
            jt.flags.use_cuda = cls._previous_use_cuda
            raise

    @classmethod
    def tearDownClass(cls):
        try:
            jt.flags.use_cuda = cls._previous_use_cuda
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


if __name__ == "__main__":
    unittest.main(verbosity=2)

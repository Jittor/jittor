"""Torch-grade tests for *torch-compatible* triton on jittor.

These lock the second half of "triton support": code written against ``torch``
(jittor's torch shim, i.e. ``import jittor as torch`` / a deployed ``torch``
redirect) that launches real ``@triton.jit`` kernels must run on jittor. Under
the shim a "torch tensor" *is* a jittor ``Var``; the bridge
(:mod:`jittor.triton_shim.backend`) also duck-types any tensor with
``data_ptr``/``dtype``/``shape`` so a genuine ``torch.Tensor`` works too.

Each kernel is created with the ``torch`` API and checked against an independent
numpy reference. Skipped unless real upstream triton + CUDA + an importable
``torch`` are all present.

Run:  python -m jittor.test.test_triton_torch_compat
"""
import unittest
import numpy as np

import jittor as jt
import jittor.triton_shim as _shim


def _available():
    if not jt.has_cuda:
        return False
    try:
        if not _shim.backend.is_available():
            return False
        import torch  # noqa: F401
        return True
    except Exception:
        return False


_HAVE = _available()
if _HAVE:
    jt.flags.use_cuda = 1
    _shim.activate_bridge()
    import torch
    import triton
    import triton.language as tl


if _HAVE:

    @triton.jit
    def add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        tl.store(o_ptr + i, tl.load(x_ptr + i, mask=m) + tl.load(y_ptr + i, mask=m), mask=m)

    @triton.jit
    def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                      sam, sak, sbk, sbn, scm, scn,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak
        b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_n[None, :] * sbn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
            acc += tl.dot(a, b, allow_tf32=False)
            a_ptrs += BLOCK_K * sak
            b_ptrs += BLOCK_K * sbk
        c_ptrs = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    @triton.jit
    def softmax_kernel(out_ptr, in_ptr, in_stride, out_stride, n_cols, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(in_ptr + row * in_stride + cols, mask=mask, other=-float("inf"))
        x = x - tl.max(x, axis=0)
        num = tl.exp(x)
        tl.store(out_ptr + row * out_stride + cols, num / tl.sum(num, axis=0), mask=mask)


def _cuda(t):
    """Best-effort: ensure a torch-shim tensor is on CUDA (it is, under use_cuda)."""
    return t.cuda() if hasattr(t, "cuda") and not bool(getattr(t, "is_cuda", True)) else t


@unittest.skipUnless(_HAVE, "real triton + CUDA + torch (shim) not available")
class TestTritonTorchCompat(unittest.TestCase):
    def ac(self, got, ref, atol=1e-5, rtol=1e-5, msg=""):
        g, r = np.asarray(got), np.asarray(ref)
        np.testing.assert_allclose(g, r, atol=atol, rtol=rtol, err_msg=msg)

    def test_torch_tensor_is_bridged(self):
        # A 'torch' tensor must be recognised as a launchable device tensor.
        t = _cuda(torch.zeros(8))
        self.assertTrue(_shim.backend._is_tensor(t))
        self.assertTrue(_shim.backend._tensor_is_cuda(t),
                        "torch tensor should be on CUDA under use_cuda=1")

    def test_torch_vector_add(self):
        n = 4096
        xn = np.random.RandomState(0).randn(n).astype("float32")
        yn = np.random.RandomState(1).randn(n).astype("float32")
        x = _cuda(torch.tensor(xn)); y = _cuda(torch.tensor(yn))
        o = _cuda(torch.zeros(n))
        add_kernel[(triton.cdiv(n, 256),)](x, y, o, n, BLOCK=256)
        self.ac(o.numpy(), xn + yn, atol=1e-6, msg="torch add")

    def test_torch_empty_like_output(self):
        # idiomatic torch+triton: allocate the output with torch.empty_like
        n = 1000
        xn = np.random.RandomState(2).randn(n).astype("float32")
        yn = np.random.RandomState(3).randn(n).astype("float32")
        x = _cuda(torch.tensor(xn)); y = _cuda(torch.tensor(yn))
        o = torch.empty_like(x)
        add_kernel[(triton.cdiv(n, 128),)](x, y, o, n, BLOCK=128)
        self.ac(o.numpy(), xn + yn, atol=1e-6, msg="torch empty_like out")

    def test_torch_matmul(self):
        M = N = K = 192
        BM = BN = BK = 32
        an = np.random.RandomState(4).randn(M, K).astype("float32")
        bn = np.random.RandomState(5).randn(K, N).astype("float32")
        a = _cuda(torch.tensor(an)); b = _cuda(torch.tensor(bn))
        c = _cuda(torch.zeros(M, N))
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        matmul_kernel[grid](a, b, c, M, N, K, K, 1, N, 1, N, 1,
                            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
        self.ac(c.numpy(), an @ bn, atol=1e-2, rtol=1e-3, msg="torch matmul")

    def test_torch_softmax_2d(self):
        M, Ncols = 48, 333
        xn = (np.random.RandomState(6).randn(M, Ncols).astype("float32")) * 2.0
        e = np.exp(xn - xn.max(1, keepdims=True)); ref = e / e.sum(1, keepdims=True)
        x = _cuda(torch.tensor(xn)); o = _cuda(torch.zeros(M, Ncols))
        BLOCK = triton.next_power_of_2(Ncols)
        softmax_kernel[(M,)](o, x, Ncols, Ncols, Ncols, BLOCK=BLOCK)
        self.ac(o.numpy(), ref, atol=1e-5, msg="torch softmax 2d")


if __name__ == "__main__":
    unittest.main(verbosity=2)

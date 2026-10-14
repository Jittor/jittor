
from _helpers import capability as _test_capability
from _helpers.cupy_bridge import cuda_numpy_code_available
# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guoye Yang <498731903@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from jittor import compile_extern
from _helpers.assertions import expect_error
from jittor._runtime import forbid_backend_fallbacks
if _test_capability.check_accelerator('cuda', backend=jt).enabled:
    from jittor.compile_extern import cublas_ops, cudnn_ops, cub_ops
else:
    cublas_ops = cudnn_ops = cub_ops = None


def check_forward(shape, dim=None):
    x = jt.random(shape)
    y = jt.numpy_cumsum(x, dim=dim)
    y_ = jt.cub_cumsum(x, dim=dim)
    assert(np.allclose(y.data, y_.data))

def check_backward(shape, dim=None):
    x = jt.random(shape)
    z = jt.random(shape)

    y = jt.numpy_cumsum(x, dim=dim)
    loss = (y * z).sum()
    grad = jt.grad(loss, x)
    
    y_ = jt.cub_cumsum(x, dim=dim)
    loss_ = (y_ * z).sum()
    grad_ = jt.grad(loss_, x)
    assert(np.allclose(grad.data, grad_.data))

# The reference these cases compare against is computed through a numpy-code
# operator, and py_converter hands that callback `cupy` instead of `numpy`
# when use_cuda is on. Without CuPy every case here raises from inside
# execution -- and leaves the CUDA work pending for an unrelated later test.
@unittest.skipIf(not cuda_numpy_code_available(),
                 "CUDA numpy-code operators need CuPy; it is not installed")
class TestCubCumsumOp(unittest.TestCase):
    def setUp(self):
        self.is_reversed = False

    @unittest.skipIf(cub_ops==None, "CUDA cub is unavailable in this build")
    @jt.flag_scope(use_cuda=1)
    def test_rank_three_is_rejected_clearly(self):
        expect_error(
            lambda: cub_ops.cub_cumsum(jt.ones((2, 2, 2)), False),
            exc_type=RuntimeError,
            match="shape.size",
        )

    @unittest.skipIf(cub_ops==None, "CUDA cub is unavailable in this build")
    @jt.flag_scope(use_cuda=1)
    def test_1d(self):
        check_forward([20])
        check_forward([6007])
        check_forward([6007], 0)
        check_forward([6007], -1)

    @unittest.skipIf(cub_ops==None, "CUDA cub is unavailable in this build")
    @jt.flag_scope(use_cuda=1)
    def test_1d_backward(self):
        check_backward([20])
        check_backward([6007])
        check_backward([6007], 0)
        check_backward([6007], -1)

    @unittest.skipIf(cub_ops==None, "CUDA cub is unavailable in this build")
    @jt.flag_scope(use_cuda=1)
    def test_2d(self):
        check_forward([5,5])
        check_forward([2000, 6007])
        check_forward([2000, 6007], 1)
        check_forward([2000, 6007], -1)

    @unittest.skipIf(cub_ops==None, "CUDA cub is unavailable in this build")
    @jt.flag_scope(use_cuda=1)
    def test_2d_backward(self):
        check_backward([5,5])
        check_backward([2000, 6007])
        check_backward([2000, 6007], 1)
        check_backward([2000, 6007], -1)

    @unittest.skipIf(cub_ops==None, "CUDA cub is unavailable in this build")
    @jt.flag_scope(use_cuda=1)
    def test_nd(self):
        check_forward([5,6,7,8], 0)
        check_forward([5,6,7,8], 1)
        check_forward([5,6,7,8], 2)
        check_forward([5,6,7,8], 3)
        check_forward([5,6,7,8], -1)
        check_forward([16,14,14,4096], 0)
        check_forward([16,14,14,4096], 1)
        check_forward([16,14,14,4096], 2)
        check_forward([16,14,14,4096], 3)
        check_forward([16,14,14,4096], -1)
        check_forward([16,14,14,4095], 3)

    @unittest.skipIf(cub_ops==None, "CUDA cub is unavailable in this build")
    @jt.flag_scope(use_cuda=1)
    def test_nd_backward(self):
        check_backward([5,6,7,8], 0)
        check_backward([5,6,7,8], 1)
        check_backward([5,6,7,8], 2)
        check_backward([5,6,7,8], 3)
        check_backward([5,6,7,8], -1)
        check_backward([16,14,14,4096], 0)
        check_backward([16,14,14,4096], 1)
        check_backward([16,14,14,4096], 2)
        check_backward([16,14,14,4096], 3)
        check_backward([16,14,14,4096], -1)
        check_backward([16,14,14,4095], 3)

@unittest.skipUnless(_test_capability.check_accelerator('cuda', backend=jt).enabled,
                     "requires a real CUDA device")
class TestCubToolchain(unittest.TestCase):
    """No CuPy dependency: validate the compiled CCCL scan and its gradient."""

    @jt.flag_scope(use_cuda=1)
    def _check(self, shape, dim, dtype):
        self.assertIsNotNone(cub_ops, "CUDA must compile and register CUB")
        rng = np.random.RandomState(71)
        values = (rng.randn(*shape) * 0.125).astype(dtype)
        upstream = (rng.randn(*shape) * 0.125).astype(dtype)
        before = jt.core.backend_fallback_count()
        with forbid_backend_fallbacks():
            x = jt.array(values, dtype=dtype)
            dout = jt.array(upstream, dtype=dtype)
            y = jt.cub_cumsum(x, dim)
            grad = jt.grad((y * dout).sum(), x)
            jt.sync_all()
            for tensor in (x, dout):
                self.assertEqual(tensor.location(), "device")
                self.assertEqual(str(tensor.dtype), dtype)
            want = np.cumsum(values.astype("float64"), axis=dim)
            want_grad = np.flip(np.cumsum(np.flip(upstream.astype("float64"), axis=dim), axis=dim), axis=dim)
            for tensor, expected in ((y, want), (grad, want_grad)):
                tensor.sync()
                self.assertEqual(tensor.location(), "device")
                np.testing.assert_allclose(tensor.numpy(), expected,
                                           rtol=2e-5 if dtype == "float32" else 1e-12,
                                           atol=2e-6 if dtype == "float32" else 1e-12)
        self.assertEqual(jt.core.backend_fallback_count(), before)

    def test_non_tile_aligned_float32_scan_and_gradient(self):
        self._check((6007,), 0, "float32")

    def test_batched_float64_scan_and_gradient(self):
        self._check((5, 1031), 1, "float64")

    def test_non_last_axis_scan_and_gradient(self):
        self._check((5, 6, 7), 1, "float32")


if __name__ == "__main__":
    unittest.main()

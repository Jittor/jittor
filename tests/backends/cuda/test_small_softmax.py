"""Real CUDA functional/domain regression for bounded FP32 warp softmax.

CPU NumPy is a reference calculator, never a tensor-framework execution.
Bitwise Native Torch CUDA confirmation is a separate full-array pair.
"""
from _helpers import capability as _test_capability
import unittest
import numpy as np
import jittor as jt
from jittor.backends.cuda.kernels.nn.softmax_cuda import _supports_warp_softmax_f32


@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled,
                 'CUDA is required; skipping is not CUDA acceptance')
class TestSmallSoftmaxCUDA(unittest.TestCase):
    def snapshot(self, value):
        value.sync()
        self.assertEqual(jt.core.dispatch_context([value]), ('cuda', 0))
        self.assertEqual(value.location(), 'device')
        before = (tuple(value.shape), str(value.dtype), value.is_stop_grad(),
                  value.device_id, value.placement_backend, value.location())
        with jt.no_grad():
            host = jt.device_copy(value, -1)
            host.sync()
            result = np.array(host.numpy(), copy=True, order='C')
        self.assertEqual(before, (tuple(value.shape), str(value.dtype), value.is_stop_grad(),
                                 value.device_id, value.placement_backend, value.location()))
        self.assertEqual(result.dtype, np.dtype('float32'))
        self.assertTrue(bool(np.isfinite(result).all()))
        return result

    def close(self, actual, expected):
        result = self.snapshot(actual)
        self.assertEqual(result.shape, expected.shape)
        delta = np.abs(result.astype(np.float64) - expected.astype(np.float64))
        self.assertEqual(int(np.count_nonzero(delta > 1e-5 + 1e-5 * np.abs(expected))), 0)

    def test_public_forward_and_complete_vjp(self):
        with jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error', cuda_kernel_math='strict'):
            before = jt.core.backend_fallback_count()
            for columns in (32, 38, 64, 128, 256, 512, 1024):
                for rows in (1, 3, 17):
                    with self.subTest(columns=columns, rows=rows):
                        index = np.arange(rows * columns, dtype=np.uint64).reshape(rows, columns)
                        fixture = (((index * 73 + 19) % 257).astype(np.int32) - 128).astype(np.float32) / np.float32(32)
                        gradient = (((index * 53 + 7) % 251).astype(np.int32) - 125).astype(np.float32) / np.float32(128)
                        fixture = np.where(np.arange(columns)[None, :] > np.arange(rows)[:, None] + columns // 2,
                                           np.float32(-np.inf), fixture)
                        wide = fixture.astype(np.float64)
                        expected = np.exp(wide - wide.max(axis=-1, keepdims=True))
                        expected /= expected.sum(axis=-1, keepdims=True)
                        vjp = expected * (gradient - (expected * gradient).sum(axis=-1, keepdims=True))
                        x = jt.array(fixture).start_grad()
                        self.assertTrue(_supports_warp_softmax_f32(x))
                        y = jt.nn.softmax(x, dim=-1)
                        dx = jt.grad((y * jt.array(gradient)).sum(), x)
                        self.close(y, expected.astype(np.float32))
                        self.close(dx, vjp.astype(np.float32))
            self.assertEqual(jt.core.backend_fallback_count(), before)

    def test_uniform_rows(self):
        with jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error', cuda_kernel_math='strict'):
            for columns in (32, 38, 128, 256, 1024):
                with self.subTest(columns=columns):
                    x = jt.array(np.zeros((3, columns), dtype=np.float32)).start_grad()
                    y = jt.nn.softmax(x, dim=-1)
                    self.close(y, np.full((3, columns), 1.0 / columns, dtype=np.float32))
                    self.close(jt.grad(y.sum(), x), np.zeros((3, columns), dtype=np.float32))

    def test_declines_outside_contract(self):
        with jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error', cuda_kernel_math='strict'):
            for shape in ((3, 31), (3, 1025), (0, 38), ()):
                self.assertFalse(_supports_warp_softmax_f32(jt.array(np.zeros(shape, dtype=np.float32))))
            for dtype in ('float16', 'float64', 'int32'):
                self.assertFalse(_supports_warp_softmax_f32(jt.ones((3, 38), dtype=dtype)))
            with jt.flag_scope(transpose_storage_view=1):
                transposed = jt.ones((38, 32)).transpose((1, 0))
            self.assertFalse(transposed._storage_is_contiguous())
            self.assertFalse(_supports_warp_softmax_f32(transposed))


if __name__ == '__main__':
    unittest.main()

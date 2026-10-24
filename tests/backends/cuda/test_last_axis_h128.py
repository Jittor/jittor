"""CUDA-only H128 native routing regression; independent Torch replay is separate."""
from _helpers import capability as _test_capability
import unittest
import numpy as np
import jittor as jt
from jittor.backends.cuda.kernels.nn.last_axis_mean_cuda import (
    _last_axis_mean_cuda, _last_axis_sum_cuda,
)


@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled,
                 'CUDA is required')
class TestLastAxisH128CUDA(unittest.TestCase):
    def _fixture(self, shape):
        index = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
        q = ((index % 127) + 1).astype(np.float32) / np.float32(263)
        dy = ((index % 19) - 9).astype(np.float32) / np.float32(29)
        return q, dy[..., :1]

    def _resident(self, value):
        value.sync()
        self.assertEqual(jt.core.dispatch_context([value]), ('cuda', 0))
        self.assertEqual(value.location(), 'device')

    def _state(self, value):
        return (tuple(value.shape), str(value.dtype), value.is_stop_grad(),
                value.device_id, value.placement_backend, value.location())

    def _snapshot(self, value):
        # Synchronize results before entry, never recover the ORIGINAL after D2H.
        before = self._state(value)
        self.assertEqual(before[-1], 'device')
        with jt.no_grad():
            host = jt.device_copy(value, -1)
            self.assertTrue(host is not value, 'independent host holder')
            host.sync()
            self.assertEqual(self._state(value), before)
            self.assertEqual(host.location(), 'cpu')
            array = np.array(host.numpy(), copy=True, order='C')
        self.assertEqual(self._state(value), before)
        self.assertTrue(bool(np.isfinite(array).all()), 'complete finite snapshot')
        return array

    def _close(self, actual, expected):
        self._resident(actual)
        self._resident(expected)
        a, b = self._snapshot(actual), self._snapshot(expected)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(bool((np.abs(a.astype(np.float64) - b.astype(np.float64))
                              <= 1e-5 + 1e-5 * np.abs(b.astype(np.float64))).all()),
                        'complete fixed1e-5 comparison')

    def test_public_full_forward_and_vjp(self):
        # First seven shapes lock every source-derived H128 block geometry.
        shapes = tuple((n, 128) for n in (1, 2, 4, 8, 16, 17, 608)) + ((1, 38, 16, 128),)
        with jt.runtime.scope(use_cuda=1, backend_fallback='error', cuda_kernel_math='strict'):
            before_fallbacks = jt.core.backend_fallback_count()
            for shape in shapes:
                for mean in (False, True):
                    for keepdims in (False, True):
                        with self.subTest(shape=shape, mean=mean, keepdims=keepdims):
                            source, dy = self._fixture(shape)
                            x = jt.array(source)
                            self._resident(x)
                            initial = self._state(x)
                            cotangent = jt.array(dy if keepdims else dy.reshape(shape[:-1]))
                            self._resident(cotangent)
                            # Test-only mathematical reference is computed on CUDA
                            # using the existing generic FLOAT64 reduction, not CPU.
                            reference = x.float64().sum(dims=(-1,), keepdims=keepdims)
                            if mean:
                                reference = reference / 128.0
                            if mean:
                                y = x.mean(dims=(-1,), keepdims=keepdims)
                                direct = _last_axis_mean_cuda(x, keepdims=keepdims)
                                root = jt.mean(x, dims=(-1,), keepdims=keepdims)
                            else:
                                y = x.sum(dims=(-1,), keepdims=keepdims)
                                direct = _last_axis_sum_cuda(x, keepdims=keepdims)
                                root = jt.sum(x, dims=(-1,), keepdims=keepdims)
                            self.assertIsNotNone(direct)
                            self._close(y, reference)
                            self._resident(direct)
                            self._resident(root)
                            self.assertTrue(self._snapshot(y).tobytes() == self._snapshot(direct).tobytes(),
                                            'public method reaches the shared owner')
                            self.assertTrue(self._snapshot(y).tobytes() == self._snapshot(root).tobytes(),
                                            'public root reaches the shared owner')
                            gradient = jt.grad((y * cotangent).sum(), x)
                            self.assertIsNotNone(gradient)
                            expected = cotangent.reshape(shape[:-1] + (1,)).broadcast(shape)
                            if mean:
                                expected = expected / 128.0
                            self._close(gradient, expected)
                            self.assertEqual(self._state(x), initial)
                            self.assertTrue(self._snapshot(x).tobytes() == source.tobytes(),
                                            'complete original input bytes unchanged')
            self.assertEqual(jt.core.backend_fallback_count(), before_fallbacks)

    def test_sum_declines_outside_h128(self):
        with jt.runtime.scope(use_cuda=1, backend_fallback='error', cuda_kernel_math='strict'):
            initial = jt.core.backend_fallback_count()
            for shape in ((6, 127), (6, 129), (6, 1024), (0, 128)):
                self.assertTrue(_last_axis_sum_cuda(jt.array(np.zeros(shape, dtype=np.float32))) is None,
                                'sum outside H128 declined')
            for dtype in ('float16', 'float64', 'int32'):
                self.assertTrue(_last_axis_sum_cuda(jt.ones((6, 128), dtype=dtype)) is None,
                                'non-FP32 sum declined')
            with jt.flag_scope(transpose_storage_view=1):
                value = jt.ones((128, 6)).transpose((1, 0))
            self.assertFalse(value._storage_is_contiguous())
            self.assertTrue(_last_axis_sum_cuda(value) is None, 'strided storage declined')
            self.assertEqual(jt.core.backend_fallback_count(), initial)


if __name__ == '__main__':
    unittest.main()

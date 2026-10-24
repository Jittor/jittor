"""Real CUDA regression for bounded FP32 penultimate-axis sum rounding.

The host expectation is a fixed float32 addition tree, not a CPU framework
execution. Independent Native Torch CUDA confirmation is recorded separately.
"""
from _helpers import capability as _test_capability
import unittest
import numpy as np
import jittor as jt
from jittor.backends.cuda.kernels.nn.full_reduce_cuda import _small_nonlast_sum_cuda


def _fixture(rows, columns, outer=16):
    index = np.arange(outer * rows * columns, dtype=np.uint64)
    mixed = ((index * np.uint64(1664525) + np.uint64(1013904223)) & np.uint64(0xffffffff)).astype(np.uint32)
    bits = ((mixed & np.uint32(0x007fffff)) |
            (((mixed >> np.uint32(20)) % np.uint32(12) + np.uint32(120)) << np.uint32(23)) |
            ((mixed & np.uint32(1)) << np.uint32(31)))
    values = bits.view(np.float32).reshape(1, outer, 1, rows, columns)
    return values, _reference_sum(values)


def _reference_sum(values):
    rows, columns = values.shape[-2:]
    accumulators = [np.zeros(values.shape[:-2] + (columns,), dtype=np.float32) for _ in range(4)]
    for row in range(rows):
        slot = row % 4
        accumulators[slot] = np.add(accumulators[slot], values[..., row, :], dtype=np.float32)
    return np.add(np.add(np.add(accumulators[0], accumulators[1], dtype=np.float32),
                         accumulators[2], dtype=np.float32), accumulators[3], dtype=np.float32)



@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled,
                 'CUDA is required; a skip is not CUDA acceptance')
class TestSmallNonlastSumCUDA(unittest.TestCase):
    def _resident(self, value):
        value.sync()
        self.assertEqual(jt.core.dispatch_context([value]), ('cuda', 0))
        self.assertEqual(value.location(), 'device')

    def _snapshot(self, value):
        self._resident(value)
        before = (tuple(value.shape), str(value.dtype), value.is_stop_grad(),
                  value.device_id, value.placement_backend, value.location())
        with jt.no_grad():
            host = jt.device_copy(value, -1)
            self.assertIsNot(host, value)
            host.sync()
            self.assertEqual(host.location(), 'cpu')
            result = np.array(host.numpy(), copy=True, order='C')
        after = (tuple(value.shape), str(value.dtype), value.is_stop_grad(),
                 value.device_id, value.placement_backend, value.location())
        self.assertEqual(before, after)
        return result

    def _exact(self, actual, expected):
        array = self._snapshot(actual)
        self.assertEqual(array.shape, expected.shape)
        self.assertEqual(array.dtype, np.dtype('float32'))
        self.assertTrue(bool(np.isfinite(array).all()))
        self.assertTrue(bool(np.array_equal(array.view(np.uint32),
                                           np.ascontiguousarray(expected).view(np.uint32))),
                        'complete CUDA rounded values differ')

    def test_public_forward_and_vjp(self):
        with jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error', cuda_kernel_math='strict'):
            before = jt.core.backend_fallback_count()
            for rows in range(1, 128):
                for keepdims in (False, True):
                    with self.subTest(rows=rows, keepdims=keepdims):
                        fixture, expected = _fixture(rows, min(rows, 64))
                        x = jt.array(fixture).start_grad()
                        want = expected[..., None, :] if keepdims else expected
                        y = x.sum(dims=(-2,), keepdims=keepdims)
                        root = jt.sum(x, dims=(len(fixture.shape) - 2,), keepdims=keepdims)
                        direct = _small_nonlast_sum_cuda(x, keepdims=keepdims)
                        self.assertIsNotNone(direct)
                        for output in (y, root, direct):
                            self._exact(output, want)
                        dy = jt.array(np.full(want.shape, 0.125, dtype=np.float32))
                        dx = jt.grad((y * dy).sum(), x)
                        self._exact(dx, np.full(fixture.shape, 0.125, dtype=np.float32))
                        self._exact(x, fixture)
            self.assertEqual(jt.core.backend_fallback_count(), before)

    def test_columns_and_outer_geometry(self):
        with jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error', cuda_kernel_math='strict'):
            for outer, rows, columns in ((1, 31, 2), (3, 63, 64),
                                         (65, 31, 7), (129, 7, 32)):
                with self.subTest(outer=outer, rows=rows, columns=columns):
                    fixture, expected = _fixture(rows, columns, outer)
                    self._exact(jt.array(fixture).sum(dims=(-2,)), expected)

    def test_implicit_broadcast_gradient_uses_native_order(self):
        with jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error', cuda_kernel_math='strict'):
            self.assertIn('ordered_sum', jt.core.backend_supported_capabilities('cuda'))
            self.assertEqual(jt.core.backend_capability_dtypes('cuda', 'ordered_sum'), ['float32'])
            for rows in (38, 64, 65, 76, 127):
                for columns in (16, 64):
                    for scale in (1, 1024):
                        with self.subTest(rows=rows, columns=columns, scale=scale):
                            incoming = _fixture(rows, columns, outer=1)[0].reshape(1, rows, columns)
                            incoming[0, :, 0] = np.array([float(1 << 24), 1., -float(1 << 24), 1.], dtype=np.float32)[np.arange(rows) % 4]
                            incoming *= np.float32(scale)
                            parameter = jt.ones((columns,)).start_grad()
                            expanded = parameter.broadcast((1, rows, columns), (0, 1))
                            dy = jt.array(incoming).stop_grad()
                            gradient = jt.grad((expanded * dy).sum(), parameter)
                            self._exact(gradient, _reference_sum(incoming).reshape(columns))
                            self._exact(dy, incoming)
            self.assertEqual(jt.core.backend_fallback_count(), 0)

    def test_second_gradient_remains_native(self):
        with jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error', cuda_kernel_math='strict'):
            x = jt.ones((76, 16)).start_grad()
            y = x.sum(dims=(0,))
            first = jt.grad((y * y).sum(), x)
            second = jt.grad(first.sum(), x)
            expected = np.full((76, 16), 152., dtype=np.float32)
            self._exact(first, expected)
            self._exact(second, expected)
            self.assertEqual(jt.core.backend_fallback_count(), 0)

    def test_strided_broadcast_gradient_is_materialized_on_cuda(self):
        with jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error', cuda_kernel_math='strict'):
            for rows in (38, 76):
                with self.subTest(rows=rows):
                    incoming = _fixture(rows, 16, outer=1)[0].reshape(rows, 16)
                    with jt.flag_scope(transpose_storage_view=1):
                        dy = jt.array(np.array(incoming.T, copy=True, order='C')).transpose((1, 0)).stop_grad()
                    self.assertFalse(dy._storage_is_contiguous())
                    parameter = jt.zeros((16,)).start_grad()
                    expanded = parameter.broadcast((rows, 16), (0,))
                    gradient = jt.grad((expanded * dy).sum(), parameter)
                    self._exact(gradient, _reference_sum(incoming))
                    self._exact(dy, incoming)
                    self.assertFalse(dy._storage_is_contiguous())
            self.assertEqual(jt.core.backend_fallback_count(), 0)

    def test_declines_outside_contract(self):
        with jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error', cuda_kernel_math='strict'):
            for shape in ((128, 63), (31, 65), (63, 1), (7, 1), (0, 7), (7, 0), (7,)):
                self.assertIsNone(_small_nonlast_sum_cuda(jt.array(np.zeros(shape, dtype=np.float32))))
            for dtype in ('float16', 'float64', 'int32'):
                self.assertIsNone(_small_nonlast_sum_cuda(jt.ones((7, 31), dtype=dtype)))
            with jt.flag_scope(transpose_storage_view=1):
                value = jt.ones((7, 31)).transpose((1, 0))
            self.assertFalse(value._storage_is_contiguous())
            self.assertIsNone(_small_nonlast_sum_cuda(value))


if __name__ == '__main__':
    unittest.main()

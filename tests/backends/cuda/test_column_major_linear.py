"""CUDA linear column-major operands retain current-leaf AD and public shapes."""
from _helpers import capability
import unittest
import numpy as np
import jittor as jt
from jittor.nn.functional.matrix import _column_major_flattened_matrix


@unittest.skipUnless(capability.check_accelerator('cuda', backend=jt).enabled,
                     'CUDA hardware required; a skip is not acceptance')
class TestColumnMajorLinearCUDA(unittest.TestCase):
    def _snapshot(self, value):
        value.sync()
        self.assertEqual(jt.core.dispatch_context([value]), ('cuda', 0))
        self.assertEqual(value.location(), 'device')
        with jt.no_grad():
            host = jt.device_copy(value, -1)
            host.sync()
            return np.array(host.numpy(), copy=True, order='C')

    def _close(self, value, expected):
        actual = self._snapshot(value)
        self.assertEqual(actual.shape, expected.shape)
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)

    def _leaf(self, raw, transposed):
        if transposed:
            base = jt.array(np.ascontiguousarray(raw.swapaxes(-1, -2)))*1.0
            with jt.flag_scope(transpose_storage_view=1):
                view = base.transpose(tuple(range(raw.ndim-2))+(raw.ndim-1, raw.ndim-2))
            leaf = view.detach().start_grad()
            self.assertFalse(leaf._storage_is_contiguous())
        else:
            base = jt.array(raw)*1.0
            leaf = base.detach().start_grad()
        return leaf, base

    def test_forward_vjp_and_detached_leaf_boundary(self):
        with jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error', cuda_kernel_math='strict'):
            for shape in ((1, 3, 8), (2, 3, 8)):
                for layout in ('dense', 'input', 'weight', 'both'):
                    for biased in (False, True):
                        with self.subTest(shape=shape, layout=layout, biased=biased):
                            raw = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape)%13-6)/8
                            weight = (np.arange(48, dtype=np.float32).reshape(6, 8)%11-5)/16
                            bias = np.arange(6, dtype=np.float32)/32
                            dy = (np.arange(np.prod(shape[:-1])*6, dtype=np.float32).reshape(shape[:-1]+(6,))%7-3)/32
                            x, base_x = self._leaf(raw, layout in ('input', 'both'))
                            w, base_w = self._leaf(weight, layout in ('weight', 'both'))
                            before = (tuple(x._storage_strides()), tuple(w._storage_strides()))
                            y = jt.nn.linear(x, w, jt.array(bias) if biased else None)
                            expected = np.matmul(raw, weight.T)+(bias if biased else 0)
                            self._close(y, expected)
                            gx, gw, old_x, old_w = jt.grad((y*jt.array(dy)).sum(), (x, w, base_x, base_w))
                            self._close(gx, np.matmul(dy, weight))
                            self._close(gw, np.matmul(dy.reshape(-1, 6).T, raw.reshape(-1, 8)))
                            self._close(old_x, np.zeros(tuple(base_x.shape), dtype=np.float32))
                            self._close(old_w, np.zeros(tuple(base_w.shape), dtype=np.float32))
                            self.assertEqual(before, (tuple(x._storage_strides()), tuple(w._storage_strides())))
                            self._close(x, raw); self._close(w, weight)
            self.assertEqual(jt.core.backend_fallback_count(), 0)

    def test_second_derivative_through_current_input_leaf(self):
        with jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error', cuda_kernel_math='strict'):
            for shape in ((1, 3, 8), (2, 3, 8)):
                for weight_view in (False, True):
                    with self.subTest(shape=shape, weight_view=weight_view):
                        raw = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)/16
                        weight = (np.arange(48, dtype=np.float32).reshape(6, 8)%5-2)/8
                        x, _ = self._leaf(raw, True)
                        w, _ = self._leaf(weight, weight_view)
                        w.stop_grad()
                        y = jt.nn.linear(x, w)
                        first = jt.grad((y*y).sum(), x)
                        second = jt.grad(first.sum(), x)
                        self._close(first, 2*np.matmul(np.matmul(raw, weight.T), weight))
                        row = 2*np.matmul(weight.sum(axis=1), weight)
                        self._close(second, np.broadcast_to(row, shape).copy())
            self.assertEqual(jt.core.backend_fallback_count(), 0)

    def test_layout_admission_uses_actual_collapsible_strides(self):
        with jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error', cuda_kernel_math='strict'):
            for shape, accepted in (((1, 3, 8), True), ((2, 3, 8), False), ((3, 8), True)):
                with self.subTest(shape=shape):
                    x, _ = self._leaf(np.zeros(shape, dtype=np.float32), True)
                    matrix = _column_major_flattened_matrix(x)
                    self.assertEqual(matrix is not None, accepted)
                    if matrix is not None:
                        self.assertEqual(tuple(matrix.shape), (8, int(np.prod(shape[:-1]))))
                        self.assertTrue(matrix._storage_is_contiguous())
                        self._close(matrix, np.zeros(tuple(matrix.shape), dtype=np.float32))
            for shape in ((3, 8), (0, 8), (8,)):
                with self.subTest(declined=shape):
                    self.assertIsNone(_column_major_flattened_matrix(jt.zeros(shape)))
            self.assertEqual(jt.core.backend_fallback_count(), 0)


if __name__ == '__main__':
    unittest.main()

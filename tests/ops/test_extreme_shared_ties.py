"""Native extrema retain right ties by default; sharing is an explicit option."""
import unittest
import numpy as np
import jittor as jt


class TestExtremeSharedTies(unittest.TestCase):
    def _check(self, cuda):
        if cuda and not jt.has_cuda:
            self.skipTest('CUDA unavailable; no CUDA claim')
        with jt.runtime.scope(use_cuda=int(cuda), device_id=0 if cuda else -1,
                              backend_fallback='error', auto_convert_64_to_32=0):
            before = jt.core.backend_fallback_count()
            for name in ('maximum', 'minimum'):
                for dtype in ('float32', 'float64'):
                    for tied in (True, False):
                        with self.subTest(name=name, dtype=dtype, tied=tied):
                            a = np.array([[-2., 0., 3.], [4., -5., 0.]], dtype=dtype)
                            b = a.copy() if tied else -a
                            cot = np.array([[.125, -.25, .5], [-1., 2., -4.]], dtype=dtype)
                            x, y = jt.array(a), jt.array(b)
                            dy = jt.array(cot).stop_grad()
                            operation = getattr(jt, name)
                            shared = operation(x, y, tie_grad='split')
                            legacy = operation(x, y)
                            gx, gy = jt.grad((shared * dy).sum(), (x, y))
                            lx, ly = jt.grad((legacy * dy).sum(), (x, y))
                            chosen = np.maximum(a, b) if name == 'maximum' else np.minimum(a, b)
                            active_x = a > b if name == 'maximum' else a < b
                            active_y = a < b if name == 'maximum' else a > b
                            expected_x = np.where(a == b, cot * .5, np.where(active_x, cot, 0))
                            expected_y = np.where(a == b, cot * .5, np.where(active_y, cot, 0))
                            for result, expected in ((shared, chosen), (legacy, chosen),
                                    (gx, expected_x), (gy, expected_y),
                                    (lx, np.where(active_x, cot, 0)),
                                    (ly, np.where(active_y | (a == b), cot, 0))):
                                result.sync()
                                self.assertEqual(jt.core.dispatch_context([result]), ('cuda', 0) if cuda else ('cpu', -1))
                                np.testing.assert_array_equal(result.numpy(), expected)
            self.assertEqual(jt.core.backend_fallback_count(), before)

    def test_cpu(self):
        self._check(False)

    def test_cuda(self):
        self._check(True)


if __name__ == '__main__':
    unittest.main()

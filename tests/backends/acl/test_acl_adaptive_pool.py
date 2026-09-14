"""Adaptive CANN windows and their gradients against independent NumPy bins."""
import unittest

import numpy as np
import jittor as jt
from _helpers import capability


def reference_pool(source, size, weight):
    height, width = source.shape[-2:]
    output = np.empty(source.shape[:-2] + size, dtype=np.float64)
    gradient = np.zeros_like(source, dtype=np.float64)
    for y in range(size[0]):
        y0, y1 = y * height // size[0], ((y + 1) * height + size[0] - 1) // size[0]
        for x in range(size[1]):
            x0, x1 = x * width // size[1], ((x + 1) * width + size[1] - 1) // size[1]
            output[..., y, x] = source[..., y0:y1, x0:x1].mean(axis=(-2, -1))
            gradient[..., y0:y1, x0:x1] += weight[..., y, x, None, None] / ((y1-y0)*(x1-x0))
    return output, gradient


@unittest.skipIf(not capability.check_accelerator("acl", backend=jt).enabled, "No ACL found")
class TestACLAdaptiveAveragePool(unittest.TestCase):
    @jt.flag_scope(use_cuda=1)
    def test_windows_and_weighted_backward(self):
        cases = [((2, 2, 5, 7), (3, 4)), ((1, 2, 2, 3), (4, 5)),
                 ((2, 5, 7), (None, 3)), ((1, 2, 5, 7), (2, None)),
                 ((1, 2, 4, 4), 2), ((1, 2, 5, 7), 1)]
        for shape, requested_size in cases:
            size = (requested_size, requested_size) if isinstance(requested_size, int) else requested_size
            size = tuple(shape[-2+i] if value is None else value for i, value in enumerate(size))
            source = ((np.arange(np.prod(shape)).reshape(shape) % 17) - 8).astype(np.float64) / 16
            out_shape = shape[:-2] + size
            weight = ((np.arange(np.prod(out_shape)).reshape(out_shape) % 11) - 5).astype(np.float64) / 16
            expected, grad_expected = reference_pool(source, size, weight)
            for dtype, tolerance in (("float32", 2e-6), ("float16", 5e-4), ("bfloat16", 4e-3)):
                with self.subTest(shape=shape, output_size=requested_size, dtype=dtype):
                    x = jt.array(source.astype(np.float32)).cast(dtype)
                    out = jt.nn.adaptive_avg_pool2d(x, requested_size)
                    grad = jt.grad((out * jt.array(weight.astype(np.float32)).cast(dtype)).sum(), x)
                    for value, reference in ((out, expected), (grad, grad_expected)):
                        value.sync()
                        self.assertEqual(str(value.dtype), dtype)
                        self.assertEqual(tuple(value.shape), reference.shape)
                        self.assertTrue(jt.compiler.has_acl)
                        self.assertEqual(jt.runtime.use_cuda, 1)
                        self.assertEqual(value.location(), "device")
                        self.assertGreaterEqual(value.device_id, 0)
                        self.assertIn(value.placement_backend, (-1, 2))
                        np.testing.assert_allclose(value.float32().numpy(), reference,
                                                   atol=tolerance, rtol=tolerance)

    @jt.flag_scope(use_cuda=1)
    def test_invalid_shape_and_dtype_are_rejected(self):
        x = jt.ones((1, 2, 5, 7))
        for size in ((2,), (2, 3, 4), (0, 2), (-1, 2), (1.5, 2)):
            with self.subTest(size=size), self.assertRaises(ValueError):
                jt.nn.adaptive_avg_pool2d(x, size)
        with self.assertRaises(ValueError):
            jt.nn.adaptive_avg_pool2d(jt.ones((5, 7)), 2)
        with self.assertRaises(TypeError):
            jt.nn.adaptive_avg_pool2d(x.int32(), 2)

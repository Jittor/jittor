"""Hardware acceptance for ACL CodeOp gradients with NumPy formula references.

These cases require a real ACL runtime. CPU collection or a skip does not
validate the kernels; the backend-gradient inventory records that boundary.
"""

import math
import unittest

import jittor as jt
import numpy as np

from _helpers import capability


@unittest.skipIf(not capability.check_accelerator('acl', backend=jt).enabled,
                 "No ACL found")
class TestACLCodeOpGradients(unittest.TestCase):
    def _compare(self, actual, expected, fallback_count, tolerance=2e-4):
        self.assertTrue(jt.compiler.has_acl)
        self.assertEqual(jt.flags.use_acl, 1)
        self.assertEqual(jt.flags.use_cuda, 1)
        for tensor, reference in zip(actual, expected):
            tensor.sync()
            self.assertEqual(tensor.location(), "device")
            self.assertGreaterEqual(tensor.device_id, 0)
            self.assertIn(tensor.placement_backend, (-1, 2))
            np.testing.assert_allclose(tensor.float32().numpy(), reference,
                                       rtol=tolerance, atol=tolerance)
        self.assertEqual(jt.core.backend_fallback_count(), fallback_count)

    @jt.flag_scope(use_acl=1, use_cuda=1, backend_fallback="error")
    def test_gelu_forward_and_gradient(self):
        count = jt.core.backend_fallback_count()
        source = np.array([-2., -1., 0., 1., 2.], dtype=np.float32)
        weight = np.array([1., -2., 3., -4., 5.], dtype=np.float32)
        cdf = .5 * (1 + np.array([math.erf(float(x) / math.sqrt(2)) for x in source]))
        derivative = cdf + source * np.exp(-source ** 2 / 2) / math.sqrt(2 * math.pi)
        x = jt.array(source)
        output = jt.nn.gelu(x, approximate="none")
        gradient = jt.grad((output * jt.array(weight)).sum(), x)
        self._compare((output, gradient), (source * cdf, derivative * weight), count)

    @jt.flag_scope(use_acl=1, use_cuda=1, backend_fallback="error")
    def test_bfloat16_swish_forward_and_gradient(self):
        count = jt.core.backend_fallback_count()
        source = np.array([-2., -1., 0., 1., 2.], dtype=np.float32)
        sigmoid = 1 / (1 + np.exp(-source))
        x = jt.array(source).bfloat16()
        output = jt.nn.silu(x)
        gradient = jt.grad(output.sum(), x)
        self.assertEqual(str(output.dtype), "bfloat16")
        self._compare((output, gradient),
                      (source * sigmoid, sigmoid * (1 + source * (1 - sigmoid))),
                      count, tolerance=2e-2)

    @jt.flag_scope(use_acl=1, use_cuda=1, backend_fallback="error")
    def test_cross_entropy_forward_and_gradient(self):
        count = jt.core.backend_fallback_count()
        source = np.array([[1., -1., 2.], [-2., 3., 0.]], dtype=np.float32)
        labels = np.array([2, 0], dtype=np.int32)
        weights = np.array([2., -3.], dtype=np.float32)
        shifted = source - source.max(axis=1, keepdims=True)
        log_prob = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        expected_gradient = np.exp(log_prob)
        expected_gradient[np.arange(2), labels] -= 1
        x = jt.array(source)
        output = jt.nn.cross_entropy_loss(x, jt.array(labels), reduction="none")
        gradient = jt.grad((output * jt.array(weights)).sum(), x)
        self._compare((output, gradient),
                      (-log_prob[np.arange(2), labels], expected_gradient * weights[:, None]),
                      count)

    @jt.flag_scope(use_acl=1, use_cuda=1, backend_fallback="error")
    def test_nearest_upsample_forward_and_gradient(self):
        count = jt.core.backend_fallback_count()
        source = np.arange(6, dtype=np.float32).reshape(1, 1, 2, 3)
        weights = np.arange(24, dtype=np.float32).reshape(1, 1, 4, 6) / 8
        expected = source.repeat(2, axis=2).repeat(2, axis=3)
        expected_gradient = weights.reshape(1, 1, 2, 2, 3, 2).sum(axis=(3, 5))
        x = jt.array(source)
        output = jt.nn.interpolate(x, size=(4, 6), mode="nearest")
        gradient = jt.grad((output * jt.array(weights)).sum(), x)
        self._compare((output, gradient), (expected, expected_gradient), count)

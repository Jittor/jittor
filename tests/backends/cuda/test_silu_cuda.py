"""Native SiLU CUDA gradients, higher-order inputs and layout boundaries."""
import unittest

import jittor as jt
import numpy as np

from _helpers import capability
from jittor._runtime.dispatch import select_kernel
from jittor._runtime.fallback import forbid_backend_fallbacks


@unittest.skipUnless(capability.check_accelerator("cuda", backend=jt).enabled,
                     "SiLU CUDA validation needs an allocated real CUDA device")
class TestSiLUCuda(unittest.TestCase):
    def test_input_and_cotangent_second_derivatives(self):
        rng = np.random.RandomState(23)
        raw = rng.uniform(-4, 4, (5, 7)).astype("float32")
        weights = rng.uniform(-2, 2, raw.shape).astype("float32")
        direction = rng.uniform(-1, 1, raw.shape).astype("float32")
        wide = raw.astype("float64")
        sigmoid = 1.0 / (1.0 + np.exp(-wide))
        first = sigmoid * (1.0 + wide * (1.0 - sigmoid))
        second = sigmoid * (1.0 - sigmoid) * (2.0 + wide * (1.0 - 2.0 * sigmoid))
        with jt.flag_scope(use_cuda=1, amp_reg=0), forbid_backend_fallbacks():
            x, dy, vector = jt.array(raw), jt.array(weights), jt.array(direction)
            self.assertIsNotNone(select_kernel("nn.silu", x))
            y = jt.nn.silu(x)
            gradient = jt.grad((y * dy).sum(), x)
            gx, gdy = jt.grad((gradient * vector).sum(), (x, dy))
            for value, expected in (
                (y, wide * sigmoid), (gradient, weights * first),
                (gx, direction * weights * second), (gdy, direction * first),
            ):
                value.sync()
                self.assertEqual(value.location(), "device")
                self.assertEqual(str(value.dtype), "float32")
                np.testing.assert_allclose(value.numpy(), expected, atol=3e-6, rtol=3e-6)

    def test_scalar_empty_transpose_and_stopped_inputs(self):
        with jt.flag_scope(use_cuda=1, amp_reg=0), forbid_backend_fallbacks():
            scalar = np.array(-.5, dtype="float32")
            empty = np.empty((0, 3), dtype="float32")
            matrix = np.arange(15, dtype="float32").reshape(3, 5) / 4 - 2
            cases = [(jt.array(scalar), scalar), (jt.array(empty), empty),
                     (jt.array(matrix).transpose(1, 0), matrix.T)]
            for x, original in cases:
                with self.subTest(shape=tuple(x.shape)):
                    raw = original.astype("float64")
                    before = x.is_stop_grad()
                    y = jt.nn.silu(x)
                    gradient = jt.grad(y.sum(), x)
                    sigmoid = 1.0 / (1.0 + np.exp(-raw))
                    for value, expected in ((y, raw * sigmoid),
                                            (gradient, sigmoid * (1.0 + raw * (1.0 - sigmoid)))):
                        value.sync()
                        self.assertEqual(value.location(), "device")
                        self.assertEqual(tuple(value.shape), tuple(x.shape))
                        np.testing.assert_allclose(value.numpy(), expected, atol=3e-6, rtol=3e-6)
                    self.assertEqual(x.is_stop_grad(), before)
            frozen = jt.array([-.5, .25, 1.0]).stop_grad()
            self.assertTrue(jt.nn.silu(frozen).is_stop_grad())
            trainable = jt.array([-.5, .25, 1.0])
            with jt.no_grad():
                self.assertTrue(jt.nn.silu(trainable).is_stop_grad())
            self.assertFalse(trainable.is_stop_grad())


if __name__ == "__main__":
    unittest.main()

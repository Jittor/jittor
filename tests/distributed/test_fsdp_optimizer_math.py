"""Independent arithmetic references for shared native/FSDP update kernels."""

import unittest

import numpy as np


class TestSharedOptimizerMath(unittest.TestCase):
    def test_adam_epsilon_policies_and_weight_decay(self):
        import jittor as jt
        from jittor.optim.algorithms.adam import adam_update

        for torch_math, decoupled in ((False, False), (True, False), (True, True), (False, True)):
            with self.subTest(torch_math=torch_math, decoupled=decoupled):
                p = np.array([1.0, -2.0], dtype=np.float32)
                g = np.array([0.1, -0.3], dtype=np.float32)
                v = np.array([0.2, 0.4], dtype=np.float32)
                m = np.array([0.3, -0.1], dtype=np.float32)
                pg = p * 0.05 + g if not decoupled else g
                expected_m = 0.9 * m + 0.1 * pg
                expected_v = 0.99 * v + 0.01 * pg * pg
                base = p * (1 - 0.02 * 0.05) if decoupled else p
                scale = 0.02 / (1 - 0.9 ** 3)
                denominator = np.sqrt(expected_v)
                if torch_math or decoupled:
                    denominator = denominator / np.sqrt(1 - 0.99 ** 3) + 0.1
                else:
                    denominator = (denominator + 0.1) / np.sqrt(1 - 0.99 ** 3)
                expected = base - expected_m * scale / denominator
                value, momentum = jt.array(v).stop_grad(), jt.array(m).stop_grad()
                result = adam_update(jt.array(p), jt.array(g), value, momentum,
                                     lr=0.02, eps=0.1, weight_decay=0.05,
                                     betas=(0.9, 0.99), step=3,
                                     decoupled_weight_decay=decoupled, torch_math=torch_math)
                np.testing.assert_allclose(result.numpy(), expected, rtol=2e-6, atol=2e-7)
                np.testing.assert_allclose(value.numpy(), expected_v, rtol=2e-6)
                np.testing.assert_allclose(momentum.numpy(), expected_m, rtol=2e-6)

    def test_sgd_momentum_and_nesterov(self):
        import jittor as jt
        from jittor.optim.algorithms.sgd import sgd_update

        for momentum, dampening, nesterov in ((0, 0, False), (0.9, 0.1, False), (0.9, 0, True)):
            p = np.array([1, -2], dtype=np.float32)
            g = np.array([0.1, -0.3], dtype=np.float32)
            v = np.array([0.2, 0.4], dtype=np.float32)
            direction = g + 0.05 * p
            new_v = momentum * v + (1 - dampening) * direction
            update = direction if momentum == 0 else direction + momentum * new_v if nesterov else new_v
            value = jt.array(v).stop_grad()
            result = sgd_update(jt.array(p), jt.array(g), value, lr=0.02,
                                momentum=momentum, dampening=dampening,
                                nesterov=nesterov, weight_decay=0.05)
            np.testing.assert_allclose(result.numpy(), p - 0.02 * update, rtol=2e-6, atol=2e-7)


if __name__ == "__main__":
    unittest.main()

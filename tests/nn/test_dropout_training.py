"""Training dropout: ``x * mask / (1 - p)`` as plain operators on a one-byte mask.

What the backward keeps is the mask and nothing else, and the product fuses
with what comes before and after it -- the bias add of the layer in front, the
residual add behind -- which a `jt.Function` wrapper prevented.
"""

import unittest

import numpy as np

import jittor as jt
from _helpers import capability as _test_capability
from _helpers.common import JittorTestCase
from _helpers.device_types import instantiate_device_type_tests


class TestTrainingDropout(JittorTestCase):

    def test_the_gradient_is_the_kept_mask_scaled(self, device):
        rng = np.random.RandomState(0)
        x = jt.array(rng.rand(64, 96).astype("float32") + 0.5)
        w = jt.array(rng.randn(64, 96).astype("float32"))
        y = jt.nn.dropout(x, 0.3, is_train=True)
        gx = jt.grad((y * w).sum(), x)
        kept = y.numpy() != 0
        self.assertTrue(0.6 < kept.mean() < 0.8, kept.mean())
        np.testing.assert_allclose(y.numpy(), kept * x.numpy() / 0.7, rtol=1e-5)
        np.testing.assert_allclose(gx.numpy(), kept * w.numpy() / 0.7, rtol=1e-5)

    def test_a_gradient_free_input_still_drops(self, device):
        x = jt.ones((128, 128)).stop_grad()
        y = jt.nn.dropout(x, 0.5, is_train=True).numpy()
        self.assertTrue(0.4 < (y == 0).mean() < 0.6)
        np.testing.assert_allclose(y[y != 0], 2.0)


instantiate_device_type_tests(TestTrainingDropout, globals())


@unittest.skipUnless(_test_capability.check_accelerator('cuda', backend=jt).enabled,
                     "counts device kernels")
class TestTrainingDropoutFuses(JittorTestCase):

    def test_bias_dropout_and_residual_run_as_one_kernel(self):
        rng = np.random.RandomState(1)
        with jt.flag_scope(use_cuda=1):
            h = jt.array(rng.randn(256, 512).astype("float32"))
            w = jt.array(rng.randn(512, 512).astype("float32") * 0.05)
            b = jt.array(rng.randn(512).astype("float32"))
            r = jt.array(rng.randn(256, 512).astype("float32"))
            jt.sync([h, w, b, r])

            def build():
                y = jt.nn.dropout(jt.matmul(h, w) + b, 0.1, is_train=True) + r
                return [y]
            jt.sync(build())
            jt.sync_all(True)
            with jt.profile() as p:
                jt.sync(build())
                jt.sync_all(True)
        names = [dict(k)["name"] if not isinstance(k, dict) else k["name"]
                 for k in p.result.kernel_records]
        # the comparison that makes the mask, and one pass for the rest
        self.assertLessEqual(len([n for n in names if "func_" in n]), 2, names)


if __name__ == "__main__":
    unittest.main()

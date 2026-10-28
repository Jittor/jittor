"""CUDA debug-level launches honor the configured kernel block width."""
import unittest
import numpy as np
import jittor as jt
from _helpers import capability as _test_capability


@unittest.skipUnless(_test_capability.check_accelerator('cuda', backend=jt).enabled,
                     'CUDA launch geometry requires a device')
class TestParallelLevelZero(unittest.TestCase):
    def test_elementwise_at_default_and_custom_block_width(self):
        for options in ({}, {'cuda_thread_num': 128}):
            for shape in ((65537,), (128, 256)):
                with self.subTest(options=options, shape=shape), jt.flag_scope(
                        use_cuda=1, para_opt_level=0, compile_options=options):
                    data = np.arange(np.prod(shape), dtype='float32').reshape(shape)
                    value = (jt.array(data) * 2 + 3).numpy()
                    jt.sync_all(True)
                    np.testing.assert_array_equal(value, data * 2 + 3)

    def test_reduction_backward_at_default_and_custom_block_width(self):
        for options in ({}, {'cuda_thread_num': 128}):
            with self.subTest(options=options), jt.flag_scope(
                    use_cuda=1, para_opt_level=0, compile_options=options):
                data = np.random.RandomState(712).randn(128, 256).astype('float32')
                x = jt.array(data)
                result = (x * x).sum()
                np.testing.assert_allclose(float(result.numpy()),
                                           np.square(data.astype('float64')).sum(), rtol=1e-5)
                np.testing.assert_allclose(jt.grad(result, x).numpy(), 2 * data,
                                           rtol=1e-5, atol=1e-6)
                jt.sync_all(True)

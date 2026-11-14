"""Gradient bucket readiness follows dependencies instead of creation IDs."""
import unittest
import numpy as np
import jittor as jt
from _helpers import capability


class TestGradientReadiness(unittest.TestCase):
    def _check(self, use_cuda):
        with jt.flag_scope(use_cuda=use_cuda, keep_graph=0, auto_graph_replay=0,
                           auto_flush_ops=0, auto_flush_bytes=0):
            leaf = jt.array([1.]).stop_grad()
            leaf.sync()
            deep = leaf + 2
            expected = np.asarray([3.], dtype='float32')
            for _ in range(5):
                deep = deep.sin()
                expected = np.sin(expected)
            shallow = leaf + 3  # Created later, but ready much earlier.
            view = deep.reshape((1, 1))
            self.assertFalse(deep.is_finished)
            orders = jt.core._gradient_producer_orders([deep, shallow, view])
            self.assertGreater(orders[0], orders[1])
            self.assertEqual(orders[0], orders[2])
            self.assertFalse(deep.is_finished, 'metadata must not execute the graph')
            reordered = jt.core._gradient_producer_orders([view, shallow, deep])
            self.assertEqual(reordered, [orders[2], orders[1], orders[0]])
            np.testing.assert_allclose(deep.numpy(), expected, atol=1e-6, rtol=1e-6)
            self.assertEqual(deep.location(), 'device' if use_cuda else 'cpu')
            self.assertEqual(jt.core._gradient_producer_orders([deep])[0], 0)

    def test_cpu_dependency_readiness(self):
        self._check(0)

    @unittest.skipUnless(capability.check_accelerator('cuda', backend=jt).enabled,
                         'requires a real CUDA runtime')
    def test_cuda_dependency_readiness(self):
        self._check(1)

"""`torch.backends.cudnn.flags(...)`: set the switches, then put them back.

Missing entirely, so MiniMax-H3's reference path died with
`module 'torch.backends.cudnn' has no attribute 'flags'`.

What it does and does not do: it restores the attributes, it does **not**
change which kernels run. cudnn's four switches here are already settings
nothing acts on -- convolution picks its own path -- and this does not make
them act. What it buys is that the value read back inside the block is the one
that was set, and that the block is not an AttributeError.
"""
import unittest

import jittor as jt
import torch


class TestCudnnFlags(unittest.TestCase):
    def setUp(self):
        self.cudnn = torch.backends.cudnn
        self.before = (self.cudnn.enabled, self.cudnn.benchmark,
                       self.cudnn.deterministic)

    def tearDown(self):
        (self.cudnn.enabled, self.cudnn.benchmark,
         self.cudnn.deterministic) = self.before

    def test_it_exists(self):
        self.assertTrue(callable(self.cudnn.flags))

    def test_values_apply_inside_the_block(self):
        with self.cudnn.flags(enabled=False, benchmark=True, deterministic=True):
            self.assertFalse(self.cudnn.enabled)
            self.assertTrue(self.cudnn.benchmark)
            self.assertTrue(self.cudnn.deterministic)

    def test_values_are_restored_after(self):
        with self.cudnn.flags(enabled=False, benchmark=True, deterministic=True):
            pass
        self.assertEqual(
            (self.cudnn.enabled, self.cudnn.benchmark, self.cudnn.deterministic),
            self.before)

    def test_an_exception_still_restores(self):
        with self.assertRaises(ValueError):
            with self.cudnn.flags(enabled=False, benchmark=True):
                raise ValueError("boom")
        self.assertEqual(
            (self.cudnn.enabled, self.cudnn.benchmark, self.cudnn.deterministic),
            self.before)

    def test_benchmark_limit_is_accepted(self):
        # torch's signature carries it; nothing here reads it, but a caller
        # passing it must not get a TypeError.
        with self.cudnn.flags(enabled=True, benchmark_limit=4):
            pass

    def test_nesting_unwinds_in_order(self):
        with self.cudnn.flags(enabled=False):
            self.assertFalse(self.cudnn.enabled)
            with self.cudnn.flags(enabled=True):
                self.assertTrue(self.cudnn.enabled)
            self.assertFalse(self.cudnn.enabled)
        self.assertEqual(self.cudnn.enabled, self.before[0])


if __name__ == "__main__":
    unittest.main()

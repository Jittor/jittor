"""cuDNN contexts restore settings; determinism reaches native convolution policy."""
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

    def test_deterministic_write_reaches_native_policy(self):
        native = getattr(jt, "cudnn", None)
        if native is None:
            self.skipTest("cuDNN unavailable")
        before = native.get_deterministic()
        self.addCleanup(native.set_deterministic, before)
        for enabled in (True, False, True):
            self.cudnn.deterministic = enabled
            self.assertEqual(native.get_deterministic(), int(enabled))
            self.assertIs(self.cudnn.deterministic, enabled)

    def test_native_deterministic_changes_are_visible(self):
        native = getattr(jt, "cudnn", None)
        if native is None:
            self.skipTest("cuDNN unavailable")
        before = native.get_deterministic()
        self.addCleanup(native.set_deterministic, before)
        for enabled in (1, 0, 1):
            native.set_deterministic(enabled)
            self.assertEqual(self.cudnn.deterministic, bool(enabled))

    def test_flags_restore_native_deterministic_policy(self):
        native = getattr(jt, "cudnn", None)
        if native is None:
            self.skipTest("cuDNN unavailable")
        before = native.get_deterministic()
        with self.cudnn.flags(deterministic=not bool(before), benchmark=False):
            self.assertEqual(native.get_deterministic(), int(not bool(before)))
        self.assertEqual(native.get_deterministic(), before)

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

"""Routing follows a Var's target device, not its allocation's location."""

from _helpers import capability as _test_capability

import unittest

import numpy as np

import jittor as jt


class TestCudaDispatchContext(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _test_capability.check_accelerator('cuda', backend=jt).enabled or _test_capability.device_count('cuda', backend=jt) < 1:
            raise unittest.SkipTest("CUDA dispatch context requires a CUDA device")

    def test_pending_and_host_staged_follow_accelerator_policy(self):
        with jt.flag_scope(use_cuda=1, device_id=0, lazy_execution=1, auto_flush_ops=0):
            x = jt.array(np.arange(8, dtype=np.float32))
            y = x + 1
            before = (x.location(), y.location(), jt.introspection.counters.live_ops,
                      jt.introspection.counters.held_vars)
            self.assertEqual(jt.core.dispatch_context([x, y]), ("cuda", 0))
            self.assertEqual(jt.core.dispatch_context([]), ("cuda", 0))
            self.assertEqual((x.location(), y.location(), jt.introspection.counters.live_ops,
                              jt.introspection.counters.held_vars), before)
            host = y.cpu()
            host.sync()
            self.assertEqual(host.location(), "cpu")
            self.assertEqual(jt.core.dispatch_context([host]), ("cuda", 0))
            np.testing.assert_array_equal(host.numpy(), np.arange(8, dtype=np.float32) + 1)

    def test_input_device_wins_without_mutating_pending_constants(self):
        if _test_capability.device_count('cuda', backend=jt) < 2:
            self.skipTest("mixed-device dispatch context requires two CUDA devices")
        with jt.flag_scope(use_cuda=1, device_id=0, lazy_execution=1, auto_flush_ops=0):
            x = jt.array(np.arange(8, dtype=np.float32))
            x.sync()
            with jt.flag_scope(device_id=1):
                scalar = jt.array(2.0)
                before = (scalar.device_id, scalar.location(), jt.introspection.counters.live_ops)
                self.assertEqual(jt.core.dispatch_context([scalar, x]), ("cuda", 0))
                self.assertEqual((scalar.device_id, scalar.location(),
                                  jt.introspection.counters.live_ops), before)
                y = x + scalar
                self.assertEqual(y.device_id, 0)
                np.testing.assert_array_equal(y.numpy(), np.arange(8, dtype=np.float32) + 2)

    def test_real_mixed_device_inputs_are_rejected_before_execution(self):
        if _test_capability.device_count('cuda', backend=jt) < 2:
            self.skipTest("mixed-device dispatch context requires two CUDA devices")
        with jt.flag_scope(use_cuda=1, device_id=0, lazy_execution=1, auto_flush_ops=0):
            x = jt.array(np.arange(8, dtype=np.float32))
            with jt.flag_scope(device_id=1):
                y = jt.array(np.arange(8, dtype=np.float32))
                before = (x.location(), y.location(), jt.introspection.counters.live_ops)
                with self.assertRaisesRegex(RuntimeError, "same device"):
                    jt.core.dispatch_context([x, y])
                self.assertEqual((x.location(), y.location(), jt.introspection.counters.live_ops), before)

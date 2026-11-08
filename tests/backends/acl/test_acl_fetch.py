"""Startup flag synchronization and lazy fetch queue lifetime on native ACL."""
import unittest

import numpy as np
import jittor as jt


@unittest.skipUnless(jt.compiler.has_acl, "requires the ACL provider")
class TestAclFetch(unittest.TestCase):
    @jt.flag_scope(use_acl=1, use_cuda=1, lazy_execution=1, auto_flush_ops=0)
    def test_pending_fetch_is_delivered_once_after_sync(self):
        expected = np.arange(8, dtype=np.float32) + 7
        value = jt.array(expected - 7) + 7
        seen = []
        jt.fetch(value, lambda array: seen.append(np.array(array, copy=True)))
        self.assertEqual(seen, [])
        jt.sync_all(True)
        self.assertEqual(value.location(), "device")
        self.assertEqual(value.device_id, jt.current_device())
        self.assertEqual(len(seen), 1)
        np.testing.assert_array_equal(seen[0], expected)
        jt.sync_all(True)
        self.assertEqual(len(seen), 1)

    @jt.flag_scope(use_acl=1, use_cuda=1, lazy_execution=1, auto_flush_ops=0)
    def test_backpressure_drains_and_queue_can_be_reused(self):
        value = jt.array([3.0], dtype="float32")
        seen = []
        for index in range(25):
            pending = value + index
            jt.fetch(pending, lambda array: seen.append(float(array[0])))
        jt.sync_all(True)
        self.assertEqual(pending.location(), "device")
        self.assertEqual(sorted(seen), list(range(3, 28)))
        jt.fetch(value - 5, lambda array: seen.append(float(array[0])))
        jt.sync_all(True)
        self.assertEqual(seen[-1], -2.0)
        self.assertEqual(len(seen), 26)
        jt.sync_all(True)
        self.assertEqual(len(seen), 26)

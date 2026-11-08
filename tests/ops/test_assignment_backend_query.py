from _helpers import capability as _test_capability
import unittest
from unittest import mock

import jittor as jt
import numpy as np
from jittor.ops import indexing
from jittor._runtime import dispatch


class TestAssignmentBackendQuery(unittest.TestCase):
    """Assignment asks only which backend `x` is on, through a native query.

    `indexing._assignment_backend` answers `dispatch_context(x).backend` without
    building the rest of the context. It must give the same answer as the full
    context on every device, and a replaced `dispatch_context` -- how a caller
    states the device -- must still be the one asked.
    """

    def _check_matches_context(self):
        for value in (jt.array(np.zeros((3, 2), np.float32)), jt.array(np.arange(4))):
            self.assertEqual(indexing._assignment_backend(value),
                             dispatch.dispatch_context(value).backend)

    def test_matches_dispatch_context_cpu(self):
        with jt.flag_scope(use_cuda=0):
            self._check_matches_context()

    @unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No cuda found")
    def test_matches_dispatch_context_cuda(self):
        with jt.flag_scope(use_cuda=1):
            self._check_matches_context()

    def test_replaced_dispatch_context_is_asked(self):
        x = jt.array(np.zeros(3, np.float32))
        acl = dispatch.DispatchContext("acl", 0, ("float32",))
        with mock.patch.object(indexing, "dispatch_context", lambda *a, **k: acl):
            value = indexing._acl_assignment_value(x, jt.array(np.arange(3)))
        # Under ACL the value is cast to x's dtype; elsewhere it is passed through.
        self.assertEqual(str(value.dtype), "float32")
        passed = jt.array(np.arange(3))
        self.assertIs(indexing._acl_assignment_value(x, passed), passed)

    def test_assignment_still_writes(self):
        x = jt.array(np.zeros(5, np.float32))
        x[1:3] = jt.array(np.array([7, 8], np.float32))
        x[jt.array(np.array([4]))] = 9.0
        np.testing.assert_array_equal(x.numpy(), [0, 7, 8, 0, 9])


if __name__ == "__main__":
    unittest.main()

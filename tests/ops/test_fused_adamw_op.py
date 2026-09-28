"""`fused_adamw` validates its four tensor lists before it does anything else.

The op is CUDA-only, so the test has to ask for a device -- and that is the very
first thing it does, before the code under test runs. On a machine without one,
`jt.flag_scope(use_cuda=1)` is what raises ("device.cc:61: No CUDA found"), and
`assertRaisesRegex` then fails against a message that has nothing to do with the
argument it wanted to check. The guard below is what keeps that a skip instead
of a failure; the ordering of the check inside the op was not the problem (it
runs before `set_flag(OpFlags::_cuda)`), and `flag_scope` restores the flag on
this path -- measured, not assumed.
"""

import unittest

import jittor as jt

from _helpers.capability import require_accelerator


class TestFusedAdamwInputErrors(unittest.TestCase):
    def test_mismatched_tensor_lists_fail_at_construction(self):
        require_accelerator("cuda")
        parameter = jt.ones((1,), dtype="float32")
        step = jt.array(1.0)
        with jt.flag_scope(use_cuda=1):
            with self.assertRaisesRegex(RuntimeError, r"parameters.size\(\)"):
                jt.fused_adamw(
                    [parameter], [], [parameter], [parameter], step,
                    0.001, 0.9, 0.999, 0.0, 1e-8)


if __name__ == "__main__":
    unittest.main()

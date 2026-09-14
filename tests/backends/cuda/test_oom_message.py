# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Running out of accelerator memory says how much, and how much there was.

The message used to be the whole of ``cudaMalloc failed``. Nothing in it is
actionable: not the size that was refused, not what the device had, not which
device. The commonest cause in practice is another process holding the card --
half a training run was lost to it on 2026-09-11, because the bare message
cannot be told apart from a model that is genuinely too large for the GPU.

By the time this is thrown the caching allocators above have already released
their cached blocks and retried, so the free figure it reports is the real one
rather than a snapshot taken before the pool gave anything back.

The allocation asked for here is 4 TiB, which no accelerator has, so the case
does not depend on what else is running on the box.
"""

import unittest

import jittor as jt

from _helpers import capability as _test_capability
from _helpers.child_process import run_python_child


def _has_cuda():
    return bool(_test_capability.check_accelerator('cuda', backend=jt).enabled)


#: Run in a child: an accelerator OOM can leave the allocator pools in a state
#: later cases in the same process would inherit.
_CHILD = """
import jittor as jt
jt.flags.use_cuda = 1
probe = jt.array([1.0])
probe.sync()
assert probe.location() == "device", probe.location()
try:
    big = jt.empty((1 << 20, 1 << 20), "float32")
    big.sync()
    print("NO_ERROR")
except Exception as exc:
    print("MESSAGE", repr(str(exc)))
"""


@unittest.skipIf(not _has_cuda(), "no CUDA device")
class TestOutOfMemoryMessage(unittest.TestCase):

    def test_the_refusal_names_the_size_and_what_was_free(self):
        completed = run_python_child(["-c", _CHILD], merge_stderr=True, timeout=0)
        self.assertNotIn("NO_ERROR", completed.stdout,
                         "a 4 TiB allocation succeeded; the case measures nothing")
        message = completed.stdout
        # Asserting only that it failed would hold nothing down: the old bare
        # message failed too. These are the four facts a reader needs.
        self.assertIn("out of memory on the accelerator", message)
        self.assertIn("MiB", message, "the refused size is not reported")
        self.assertIn("free of", message, "what the device had is not reported")
        self.assertIn("device", message, "the device is not named")
        self.assertNotEqual(
            "cudaMalloc failed", message.strip(),
            "the message regressed to the bare form")


if __name__ == "__main__":
    unittest.main()

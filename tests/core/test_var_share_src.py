# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``Var::share_with`` stores a Var, not an Allocator pretending to be one.

``share_with`` used to write its argument into ``Var::allocator``::

    inline void share_with(Var* x, size_t offset = 0) {
        allocator = (Allocator*)x; allocation = offset;
    }

and ``Var::alloc`` cast it back.  The two meanings of that one field were told
apart by ``mem_ptr == nullptr`` -- an unwritten invariant, not a check -- so
between ``share_with`` and ``alloc`` it held a ``Var*`` while a dozen call sites
read it as an ``Allocator*``.  Anything reaching ``var->allocator->is_cuda()``
in that window issued a virtual call on a Var.

The reachable consequence, and what this file pins, is that ``alloc`` could not
ask whether the var it was told to alias had any memory to alias: it went
straight to ``x->allocator->share_with(...)``, a virtual call through a null
pointer when the source has not been executed yet.  Nothing orders the source
before the sharer, so that is a plain segfault -- and it takes the whole process
down, hence the subprocess.

``share_src`` / ``share_offset`` carry the request instead, so the source's
state can be asked about: an unusable source falls back to a private allocation
rather than crashing.

Run::  python -m pytest tests/core/test_var_share_src.py
"""

import unittest

import numpy as np

import jittor as jt

from _helpers.child_process import run_child_script

# `source` is created *after* `sharer` on purpose: top_weak_sync only sweeps in
# holders older than the var being synced, so a newer one stays unexecuted and
# the share request meets a source with no allocator.
SHARE_WITH_UNALLOCATED_SOURCE = r'''
import numpy as np
import jittor as jt

sharer = jt.array(np.zeros(4, "float32"))
source = jt.array(np.ones(4, "float32"))
sharer.share_with(source)
print("VALUE", sharer.numpy().tolist())
print("DONE")
'''


def run_probe(source):
    """Run a probe in a child process pinned to this worktree."""
    done = run_child_script(source, text=True, merge_stderr=True,
                            name="var_share_src")
    return done.returncode, done.stdout


class TestVarShareSrc(unittest.TestCase):
    def test_sharing_with_an_unallocated_source_does_not_crash(self):
        code, output = run_probe(SHARE_WITH_UNALLOCATED_SOURCE)
        self.assertEqual(code, 0, output[-4000:])
        self.assertIn("DONE", output, output[-4000:])
        # Falling back to a private allocation is the defined outcome: the
        # values are the sharer's own, no aliasing happened.
        self.assertIn("VALUE [0.0, 0.0, 0.0, 0.0]", output, output[-4000:])

    def test_share_with_an_executed_source_still_aliases(self):
        # Not jt.array for the sharer: ArrayOp::run swaps its own staging
        # allocation into the output and would undo the share.
        source = jt.array(np.arange(4, dtype="float32")) * 1.0
        source.sync()
        sharer = jt.array(np.zeros(4, "float32")) + 5.0
        sharer.share_with(source)
        sharer.sync()
        # One buffer, two vars: what the sharer's op wrote is what the source
        # reads back.
        self.assertEqual(sharer.raw_ptr, source.raw_ptr)
        np.testing.assert_array_equal(
            source.numpy(), np.full(4, 5.0, dtype="float32"))

    def test_share_with_rejects_a_var_that_already_shares(self):
        first = jt.array(np.zeros(4, "float32")) * 1.0
        first.sync()
        second = jt.array(np.zeros(4, "float32")) * 1.0
        second.sync()
        sharer = jt.array(np.zeros(4, "float32")) + 1.0
        sharer.share_with(first)
        # The guard used to read the same field the request was parked in. It
        # has to keep rejecting a second request now that they are separate.
        with self.assertRaises(Exception) as caught:
            sharer.share_with(second)
        self.assertIn("already executed or shared", str(caught.exception))

    def test_inplace_getitem_and_setitem_still_copy_correctly(self):
        # getitem_inplace / setitem_inplace are the busiest share_with callers,
        # and their "is this already shared?" test read the punned field.
        # Values, not pointers, are what must survive.
        a = jt.array(np.arange(32, dtype="float32").reshape(4, 8)) * 1.0
        a.sync()
        np.testing.assert_array_equal(
            a[0].numpy(), np.arange(32, dtype="float32").reshape(4, 8)[0])
        b = jt.array(np.zeros((4, 8), "float32")) * 1.0
        b.sync()
        b[1] = jt.array(np.arange(8, dtype="float32") + 1)
        want = np.zeros((4, 8), "float32")
        want[1] = np.arange(8, dtype="float32") + 1
        np.testing.assert_array_equal(b.numpy(), want)


if __name__ == "__main__":
    unittest.main()

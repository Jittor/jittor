# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Destroying a released VarHolder must not silently switch weak sync off.

``sync_ptr`` (``var_holder.cc``) is a global iterator into ``hold_vars`` marking
how far ``top_weak_sync`` (``executor.cc``) has already walked.  Both
``release_from_holders()`` and ``~VarHolder()`` moved it on with::

    if (iter == sync_ptr)
        sync_ptr = std::next(sync_ptr);

``release_from_holders()`` leaves ``iter == hold_vars.end()``, so a destructor
running after it evaluated ``std::next(end())`` whenever ``sync_ptr`` was also
``end()`` -- the state the process starts in.  That is undefined behaviour, and
libstdc++'s circular list quietly returns ``begin()``.  ``top_weak_sync`` then
breaks on its very first line::

    if (sync_ptr == hold_vars.begin()) break;

for the rest of the process.  Nothing raises, nothing is logged and no value is
wrong; held vars simply stop being swept in, and the lazy graph grows until
something else forces it out.

The probe runs in two child processes -- one that triggers the destructor path,
one that does not -- and compares.  Comparing the two runs rather than asserting
an absolute op count is what makes this stable: the number of lived ops depends
on which ops the probe builds, the equality does not.

Run::  python -m pytest tests/core/test_hold_vars_sync_ptr.py
"""

import unittest

from _helpers.child_process import run_child_script


PROBE = r'''
import os
import numpy as np
import jittor as jt

release_a_holder = os.environ["PROBE_MODE"] == "release"

def pending():
    """A var whose op has not run yet."""
    return jt.array(np.zeros(4, "float32")) * 2.0

# hold_vars has to be non-empty when the destructor runs: next(end()) on an
# empty list is end() again and nothing is corrupted.
keeper = jt.array(np.zeros(4, "float32"))

if release_a_holder:
    victim = jt.array(np.zeros(4, "float32"))
    victim.release_from_holders()
    del victim

a, b, c = pending(), pending(), pending()
before = jt.number_of_lived_ops()
# Weak sync is on by default: syncing the newest var must also sweep in the
# older pending ones.
c.sync()
print("RESULT before=%d after=%d" % (before, jt.number_of_lived_ops()))
'''


def run_probe(mode):
    done = run_child_script(PROBE, env={"PROBE_MODE": mode},
                            text=True, merge_stderr=True, name="sync_ptr")
    output = done.stdout
    assert done.returncode == 0, output[-4000:]
    lines = [l for l in output.splitlines() if l.startswith("RESULT")]
    assert len(lines) == 1, output[-4000:]
    fields = dict(part.split("=") for part in lines[0].split()[1:])
    return int(fields["before"]), int(fields["after"]), output


class TestHoldVarsSyncPtr(unittest.TestCase):
    def test_weak_sync_sweeps_older_pending_vars(self):
        before, after, output = run_probe("keep")
        # Guards the comparison in the next case against being vacuously true.
        self.assertLess(after, before, output[-4000:])

    def test_weak_sync_survives_a_released_holders_destructor(self):
        clean_before, clean_after, _ = run_probe("keep")
        released_before, released_after, output = run_probe("release")
        self.assertEqual(released_before, clean_before, output[-4000:])
        self.assertEqual(
            released_after, clean_after,
            "weak sync stopped sweeping after a released holder was destroyed\n"
            + output[-4000:])


if __name__ == "__main__":
    unittest.main()

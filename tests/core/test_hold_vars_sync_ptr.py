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
    def test_graph_and_memory_diagnostics_use_the_runtime_roots(self):
        probe = r'''
import jittor as jt
import numpy as np
with jt.flag_scope(use_cuda=0, lazy_execution=1, auto_flush_ops=0,
                   profile_memory_enable=1, trace_py_var=1):
    x = jt.array([1., 2., 3.]).name("runtime_root_input")
    y = (x * x).sum()
    gradient = jt.grad(y, x)
    np.testing.assert_allclose(gradient.numpy(), [2., 4., 6.])
    assert jt.liveness_info()["hold_vars"] >= 3
    assert any("runtime_root_input" in name for name in jt.dump_all_graphs().hold_vars)
    assert "root()" in jt.get_max_memory_treemap()[1]
    jt.core.display_memory_info("runtime-root-test", True)
print("ROOT_DIAGNOSTICS_OK")
'''
        done = run_child_script(probe, text=True, merge_stderr=True,
                                name="runtime_root_diagnostics")
        self.assertEqual(done.returncode, 0, done.stdout[-4000:])
        self.assertIn("ROOT_DIAGNOSTICS_OK", done.stdout)

    def test_sync_cutoff_does_not_consume_newer_pending_holders(self):
        probe = r'''
import jittor as jt
with jt.flag_scope(use_cuda=0, lazy_execution=1, auto_flush_ops=0, use_threading=0):
    def pending(label):
        return jt.code([1], "float32", [], cpu_header="#include <stdio.h>",
                       cpu_src='@out0(0)=1; printf("%s\\n"); fflush(stdout);' % label)
    oldest = pending("OLDEST_EXECUTED")
    middle = pending("MIDDLE_EXECUTED")
    newest = pending("NEWEST_EXECUTED")
    oldest.sync()
    print("OLD_SYNC_DONE", flush=True)
    newest.sync()
    print("NEW_SYNC_DONE", flush=True)
'''
        done = run_child_script(probe, text=True, merge_stderr=True,
                                name="holder_sync_cutoff")
        self.assertEqual(done.returncode, 0, done.stdout[-4000:])
        output = done.stdout
        self.assertEqual(output.count("MIDDLE_EXECUTED"), 1, output[-4000:])
        self.assertLess(output.index("OLD_SYNC_DONE"), output.index("MIDDLE_EXECUTED"))
        self.assertLess(output.index("MIDDLE_EXECUTED"), output.index("NEW_SYNC_DONE"))

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

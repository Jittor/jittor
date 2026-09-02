# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Dun Liang <randonlang@gmail.com>.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""What a process does when a child dies, and when it exits mid-import.

Both cases used to end the same way: the process disappeared without printing
anything, which is the least useful outcome available and hides precisely the
failures that need diagnosing.  Every case here therefore asserts on *readable*
output and on an exit status, never only on "it did not hang".
"""

import os
import unittest

import jittor as jt

from _helpers.child_process import run_child_script


class TestChildKilledBySignal(unittest.TestCase):
    """A child dying by a signal must not take this process with it.

    Jittor installs a process-wide SIGCHLD action.  It used to quick_exit the
    whole process whenever *any* child died by a signal -- including children
    jittor never started -- and the explanation went to a buffered LOGe that
    std::quick_exit then discarded.  A test whose child segfaults on purpose
    killed the entire pytest session with no output at all.

    The child here is launched *without* ``crash_isolated``: shielding it would
    hide the very behaviour under test.
    """

    def test_aborting_child_leaves_this_process_alive(self):
        child = run_child_script(
            "import os\n"
            "import jittor as jt\n"
            "print('CHILD-READY', flush=True)\n"
            "os.abort()\n",
            merge_stderr=True,
        )
        output = child.stdout.decode("utf8", "replace")
        # 1. we are still here to make an assertion at all
        # 2. the child's death is visible as a signal, not swallowed
        self.assertLess(child.returncode, 0, output)
        self.assertEqual(-child.returncode, 6, output)
        # 3. the child's own output survived
        self.assertIn("CHILD-READY", output)

    def test_the_parent_reports_the_signal_it_saw(self):
        # The diagnosis has to reach fd 2 directly: composing it into a
        # buffered stream and then quick_exit-ing threw it away every time.
        child = run_child_script(
            "import os\n"
            "import subprocess\n"
            "import sys\n"
            "import jittor as jt\n"
            "grandchild = subprocess.run([sys.executable, '-c', 'import os; os.abort()'])\n"
            "print('SIGNAL', grandchild.returncode, flush=True)\n"
            "print('PARENT-SURVIVED', flush=True)\n",
            merge_stderr=True,
        )
        output = child.stdout.decode("utf8", "replace")
        self.assertEqual(child.returncode, 0, output)
        self.assertIn("PARENT-SURVIVED", output)
        self.assertIn("SIGNAL -6", output)
        self.assertIn("killed by signal", output)


@unittest.skipIf(not jt.has_cuda, "EventQueue only exists in CUDA builds")
class TestExitWithoutCleanup(unittest.TestCase):
    """Exiting without ``core.cleanup()`` must not abort.

    The global ``EventQueue`` starts a worker thread from its constructor and
    relies on ``cleanup_callback`` to stop it.  That list is drained by
    ``core.cleanup()``, which python's atexit calls -- and which never runs when
    ``import jittor`` raises partway, because the registration is at the end of
    the module that raised.  Static destruction then reached ``~std::thread``
    on a joinable thread and called ``std::terminate``: a failed import ended
    in SIGABRT, and a parent watching that import saw nothing.

    The child below reproduces the *state* that causes it -- worker running,
    cleanup not registered -- by unregistering the hook, which is deterministic
    and does not depend on where inside the import the failure happens.
    """

    def test_exit_without_the_atexit_hook(self):
        child = run_child_script(
            "import atexit\n"
            "import jittor as jt\n"
            "from jittor._runtime import core_api\n"
            "jt.array([1.0, 2.0]).sync()\n"
            "atexit.unregister(core_api.jittor_exit)\n"
            "print('EXITING-WITHOUT-CLEANUP', flush=True)\n",
            merge_stderr=True,
        )
        output = child.stdout.decode("utf8", "replace")
        self.assertIn("EXITING-WITHOUT-CLEANUP", output)
        self.assertNotIn("terminate called", output)
        self.assertEqual(child.returncode, 0, output)


if __name__ == "__main__":
    unittest.main()

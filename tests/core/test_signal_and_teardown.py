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



class TestSegfaultReport(unittest.TestCase):
    """A crashing process must produce a report, promptly, and then stop.

    Unlike the cases above, the crash here is the *means*, not the subject: what
    is under test is what the handler prints and how it leaves. So this one does
    use ``crash_isolated`` -- shielding it hides nothing that matters.
    """

    def test_segfault_is_reported_and_the_process_leaves(self):
        child = run_child_script(
            "import ctypes\n"
            "import sys\n"
            "import jittor as jt\n"
            "print('READY', flush=True)\n"
            "sys.stderr.flush()\n"
            "ctypes.string_at(0)\n"
            "print('NOT-REACHED', flush=True)\n",
            merge_stderr=True,
            crash_isolated=True,
            timeout=300,
        )
        output = child.stdout.decode("utf8", "replace")
        self.assertIn("READY", output)
        self.assertNotIn("NOT-REACHED", output)
        # The report reaches fd 2 through write(2), so it survives however the
        # handler leaves -- including quick_exit, which discards buffered stdio.
        self.assertIn("Caught segfault at address 0x", output)
        self.assertIn("Segfault, exit", output)
        self.assertNotEqual(child.returncode, 0, output)


class TestSignalHandlerStaysAsyncSignalSafe(unittest.TestCase):
    """The handler must not reach anything that can allocate, lock or throw.

    This is a source check because the property is a latent one: a handler that
    calls malloc is correct on every run that did not interrupt malloc.  The
    failure it guards against -- crashing inside the allocator, re-entering it
    from the handler, and deadlocking -- produces a hung process with no report,
    which is indistinguishable from a slow one.
    """

    def _handler_body(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))),
            "python", "jittor", "src", "utils", "log.cc")
        with open(path, encoding="utf8") as handle:
            source = handle.read()
        start = source.index("void segfault_sigaction(")
        end = source.index("\n}", start)
        body = source[start:end]
        # Drop comments: the explanations of why these calls are gone name them.
        return "\n".join(
            line.split("//", 1)[0] for line in body.splitlines())

    def test_no_stdio_no_logging_no_exit(self):
        body = self._handler_body()
        offenders = []
        for banned, why in (
                ("std::cerr", "ostream: locks and can allocate"),
                ("LOGe", "builds an ostringstream"),
                ("LOGf", "builds an ostringstream and then throws"),
                ("LOGw", "builds an ostringstream"),
                ("LOGi", "builds an ostringstream"),
        ):
            if banned in body:
                offenders.append("%s (%s)" % (banned, why))
        # exit() runs atexit handlers and static destructors while the other
        # threads of a just-faulted process are still running; _exit does not.
        for line in body.splitlines():
            stripped = line.strip()
            if "exit(1)" in stripped and "_exit(1)" not in stripped \
                    and "quick_exit" not in stripped:
                offenders.append("exit(1) instead of _exit(1): " + stripped)
        self.assertEqual(offenders, [])

    def test_flags_the_handler_touches_are_sig_atomic(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))),
            "python", "jittor", "src", "utils", "log.cc")
        with open(path, encoding="utf8") as handle:
            source = handle.read()
        # Plain bool/int lets the compiler cache or tear a value the handler can
        # change at any instruction. Asserted on a boolean rather than with
        # assertIn so a failure does not print the whole translation unit.
        missing = [
            declaration
            for declaration in ("volatile sig_atomic_t exited",
                                "volatile sig_atomic_t segfault_happen")
            if declaration not in source
        ]
        self.assertEqual(missing, [], "log.cc is missing: %s" % missing)

if __name__ == "__main__":
    unittest.main()

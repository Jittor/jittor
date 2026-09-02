# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Wenyang Zhou <576825820@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import os, subprocess, tempfile
import jittor as jt
import jittor_utils as jit_utils
from jittor_utils import lock as jit_lock

from _helpers.child_process import PYTHON, child_env, run_python_child, shell_status

class TestLock(unittest.TestCase):
    def test(self):
        if os.environ.get('lock_full_test', '0') == '1':
            cache_path = os.path.join(jit_utils.home(), ".cache", "jittor", "lock")
            assert os.system(f"rm -rf {cache_path}") == 0
            cmd = f"cache_name=lock {PYTHON} -m jittor.selftest"
        else:
            cmd = f"{PYTHON} -m jittor.selftest"
        print("run cmd twice", cmd)
        assert shell_status(f"{cmd} & {cmd} & wait %1 && wait %2") == 0


# A bare Python child: it must not import jittor, or it would block on the very
# lock these tests hold.
_PROBE = """
import fcntl, sys
f = open(sys.argv[1], 'r+')
try:
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print('free')
except OSError:
    print('busy')
"""

# The holder exits on a line from stdin rather than being killed: Jittor
# installs a SIGCHLD handler that treats a killed child as an out-of-memory
# event and quick-exits the whole process, so a test that kills a subprocess
# takes the test runner down with it.
_HOLDER = """
import fcntl, json, os, sys, time
f = open(sys.argv[1], 'r+')
fcntl.flock(f, fcntl.LOCK_EX)
f.seek(0); f.truncate()
f.write(json.dumps({"pid": os.getpid(), "time": time.time(), "cmd": "holder"}))
f.flush()
sys.stdout.write("locked\\n"); sys.stdout.flush()
sys.stdin.readline()
"""


def _lock_state(path):
    out = run_python_child(["-c", _PROBE, path], text=False)
    return out.stdout.decode().strip()


@unittest.skipIf(os.name == 'nt', "flock semantics are POSIX only")
@unittest.skipIf(os.environ.get("disable_lock", "0") == "1",
                 "the build lock is disabled in this environment")
class TestBuildLockIsShared(unittest.TestCase):
    """The Python and the C++ side must hold *one* lock, of one kind, on one fd.

    They used to hold two: ``fcntl.flock`` (BSD) from Python and
    ``fcntl(F_SETLKW)`` (POSIX record lock) from C++, on two descriptors. On
    Linux those two families are independent, so both sides could be inside the
    "exclusive" section at once and compile into the same cache directory.
    Every assertion below is about a *third* process observing what we hold:
    that is the only way to tell which lock family is actually in use.
    """

    def setUp(self):
        self.lock = jit_lock.jittor_lock
        if self.lock.core is None:
            self.skipTest("jittor_core was not handed the lock descriptor")

    def test_core_lock_is_visible_to_other_processes(self):
        self.assertEqual(_lock_state(self.lock.filename), "free")
        jt.core.lock_acquire()
        try:
            self.assertTrue(jt.core.lock_is_held())
            # One flag, not one per language.
            self.assertTrue(self.lock.is_locked)
            self.assertEqual(_lock_state(self.lock.filename), "busy")
        finally:
            jt.core.lock_release()
        self.assertFalse(self.lock.is_locked)
        self.assertEqual(_lock_state(self.lock.filename), "free")

    def test_closing_another_descriptor_does_not_release_the_lock(self):
        """A POSIX record lock dies when *any* fd for the file is closed."""
        jt.core.lock_acquire()
        try:
            extra = os.open(self.lock.filename, os.O_RDWR)
            os.close(extra)
            self.assertEqual(_lock_state(self.lock.filename), "busy")
        finally:
            jt.core.lock_release()

    def test_python_lock_scope_is_the_same_lock(self):
        with jit_lock.lock_scope():
            self.assertTrue(jt.core.lock_is_held())
            self.assertEqual(_lock_state(self.lock.filename), "busy")
        self.assertFalse(jt.core.lock_is_held())
        self.assertEqual(_lock_state(self.lock.filename), "free")

    def test_nested_scopes_do_not_release_early(self):
        with jit_lock.lock_scope():
            with jit_lock.lock_scope():
                pass
            self.assertEqual(_lock_state(self.lock.filename), "busy")


@unittest.skipIf(os.name == 'nt', "flock semantics are POSIX only")
class TestBuildLockTimeout(unittest.TestCase):
    """``F_SETLKW`` had no timeout: a wedged holder produced no output, ever."""

    def test_wait_times_out_and_names_the_holder(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "jittor.lock")
            open(path, "w").close()
            # Popen, not a helper runner: the test needs the process alive
            # while it takes the lock, so it still pins PYTHONPATH itself.
            holder = subprocess.Popen([PYTHON, "-c", _HOLDER, path],
                                      env=child_env(),
                                      stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE)
            try:
                self.assertEqual(holder.stdout.readline().strip(), b"locked")
                fd = os.open(path, os.O_RDWR)
                old = (jit_lock.lock_timeout, jit_lock.lock_report_after)
                jit_lock.lock_timeout, jit_lock.lock_report_after = 2.0, 0.5
                try:
                    with self.assertRaises(RuntimeError) as caught:
                        jit_lock._acquire(fd, path)
                finally:
                    jit_lock.lock_timeout, jit_lock.lock_report_after = old
                    os.close(fd)
            finally:
                holder.stdin.write(b"\n")
                holder.stdin.close()
                holder.wait(timeout=30)
        message = str(caught.exception)
        self.assertIn("timed out", message)
        # The point of the timeout is the diagnostic, not the exception.
        self.assertIn(str(holder.pid), message)


if __name__ == "__main__":
    unittest.main()

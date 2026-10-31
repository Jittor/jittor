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
import select
import jittor as jt
import jittor_utils as jit_utils
from jittor_utils import lock as jit_lock

from _helpers.child_process import PYTHON, child_env, run_python_child, shell_status


class TestDisableLockOptIn(unittest.TestCase):
    def test_disabling_the_lock_warns_in_a_fresh_process(self):
        result = run_python_child(
            ["-c", "import jittor_utils.lock"],
            env={"JT_BUILD_DISABLE_LOCK": "1"},
            text=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertIn(
            "JT_BUILD_DISABLE_LOCK=1: Jittor build locking is disabled",
            result.stdout.decode(),
        )

    def test_the_deprecated_unprefixed_name_still_disables_the_lock(self):
        # 2.22 gave every setting a namespace; the old name has to keep working
        # or every existing script and gate that uses it breaks at once.
        result = run_python_child(
            ["-c", "import jittor_utils.lock"],
            env={"disable_lock": "1"},
            text=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertIn(
            "JT_BUILD_DISABLE_LOCK=1: Jittor build locking is disabled",
            result.stdout.decode(),
        )

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

    Nothing here asserts that the file is *idle*, because it is the session's
    lock and the gate runs several workers against one ``JITTOR_HOME``; an
    "idle" assertion would be a statement about the whole session that this
    process cannot make. The two that existed were removed for that reason --
    see the comment in the first case.
    """

    def setUp(self):
        self.lock = jit_lock.jittor_lock
        if self.lock.core is None:
            self.skipTest("jittor_core was not handed the lock descriptor")

    def test_core_lock_is_visible_to_other_processes(self):
        # Deliberately *not* `assertEqual(_lock_state(...), "free")` first, and
        # not at the end either. The file is the session's build lock, shared
        # with every other worker, and the gate runs four of them against one
        # `JITTOR_HOME`: "nobody holds it" is a property of the whole session,
        # not something one process can establish. Measured: with any concurrent
        # holder the probe answers "busy", which is exactly how this failed in
        # the native half of the pull-request gate (`'busy' != 'free'`, while the
        # same file passed serially). What the case is about is the class
        # docstring's claim -- a third process observes *our* hold, and our own
        # release is asserted through `is_locked` below.
        jt.core.lock_acquire()
        try:
            self.assertTrue(jt.core.lock_is_held())
            # One flag, not one per language.
            self.assertTrue(self.lock.is_locked)
            self.assertEqual(_lock_state(self.lock.filename), "busy")
        finally:
            jt.core.lock_release()
        self.assertFalse(self.lock.is_locked)

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

    def test_nested_scopes_do_not_release_early(self):
        with jit_lock.lock_scope():
            with jit_lock.lock_scope():
                pass
            self.assertEqual(_lock_state(self.lock.filename), "busy")

    @unittest.skipUnless(hasattr(os, "fork"), "fork is required")
    def test_fork_child_cannot_enter_while_parent_holds_the_build_lock(self):
        """The child must get a new flock description before C++ lock calls."""
        if self.lock.is_locked:
            self.skipTest("the parent already holds the session build lock")

        attempting_r, attempting_w = os.pipe()
        acquired_r, acquired_w = os.pipe()
        release_r, release_w = os.pipe()
        child = None
        parent_holds_lock = False
        try:
            jt.core.lock_acquire()
            parent_holds_lock = True
            child = os.fork()
            if child == 0:
                try:
                    os.close(attempting_r)
                    os.close(acquired_r)
                    os.close(release_w)
                    os.write(attempting_w, b"A")
                    jt.core.lock_acquire()
                    os.write(acquired_w, b"L")
                    os.read(release_r, 1)
                    jt.core.lock_release()
                except BaseException as error:
                    try:
                        os.write(acquired_w, b"E" + repr(error).encode("utf8"))
                    except OSError:
                        pass
                os._exit(0)

            os.close(attempting_w)
            attempting_w = -1
            os.close(acquired_w)
            acquired_w = -1
            os.close(release_r)
            release_r = -1

            ready, _, _ = select.select([attempting_r], [], [], 10)
            self.assertTrue(ready, "fork child did not reach lock acquisition")
            self.assertEqual(os.read(attempting_r, 1), b"A")

            # Keep the parent lock held for a bounded observation window. An
            # inherited OFD or stale C++ _has_lock lets the child through here.
            ready, _, _ = select.select([acquired_r], [], [], 0.25)
            entered_before_release = os.read(acquired_r, 4096) if ready else b""

            jt.core.lock_release()
            parent_holds_lock = False
            if entered_before_release:
                entered_after_release = b""
            else:
                ready, _, _ = select.select([acquired_r], [], [], 10)
                self.assertTrue(ready, "fork child did not acquire after parent release")
                entered_after_release = os.read(acquired_r, 4096)
            os.write(release_w, b"R")
            _, status = os.waitpid(child, 0)
            child = None
            entered = entered_before_release or entered_after_release
            self.assertFalse(
                entered_before_release,
                "fork child entered the build critical section before parent release: "
                + repr(entered_before_release),
            )
            self.assertEqual(entered, b"L")
            self.assertTrue(os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, status)
        finally:
            if parent_holds_lock:
                jt.core.lock_release()
            if child is not None:
                try:
                    os.write(release_w, b"R")
                except OSError:
                    pass
                try:
                    os.waitpid(child, 0)
                except ChildProcessError:
                    pass
            for fd in (attempting_r, attempting_w, acquired_r, acquired_w, release_r, release_w):
                if fd >= 0:
                    try:
                        os.close(fd)
                    except OSError:
                        pass

    @unittest.skipUnless(hasattr(os, "fork"), "fork is required")
    def test_fork_child_lock_reopen_failure_is_reported_without_releasing_parent(self):
        if self.lock.is_locked:
            self.skipTest("the parent already holds the session build lock")

        original_open = os.open
        original_fd = self.lock.fd
        original_path = self.lock.filename
        read_fd, write_fd = os.pipe()
        child = None
        new_fd = None
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "jittor.lock")
        try:
            new_fd = original_open(path, os.O_RDWR | os.O_CREAT, 0o600)
            self.lock.fd = new_fd
            self.lock.filename = path
            self.lock.core.set_lock_fd(new_fd, False)
            self.lock.lock()

            parent_pid = os.getpid()

            def fail_child_reopen(filename, flags, *args, **kwargs):
                if os.getpid() != parent_pid and os.fspath(filename) == path:
                    raise OSError(24, "injected fork-child lock reopen failure")
                return original_open(filename, flags, *args, **kwargs)

            os.open = fail_child_reopen
            child = os.fork()
            if child == 0:
                try:
                    os.close(read_fd)
                    try:
                        self.lock.lock()
                    except RuntimeError as error:
                        os.write(write_fd, b"E" + str(error).encode("utf8"))
                    else:
                        os.write(write_fd, b"L")
                finally:
                    os._exit(0)

            os.open = original_open
            os.close(write_fd)
            write_fd = -1
            ready, _, _ = select.select([read_fd], [], [], 10)
            self.assertTrue(ready, "fork child did not report its lock result")
            result = os.read(read_fd, 4096)
            self.assertTrue(result.startswith(b"E"), result)
            self.assertIn(b"could not reopen the build lock after fork", result)
            self.assertTrue(self.lock.is_locked, "child failure changed parent lock state")
            self.assertEqual(_lock_state(path), "busy")
            _, status = os.waitpid(child, 0)
            child = None
            self.assertTrue(os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, status)
        finally:
            os.open = original_open
            if child is not None:
                try:
                    os.waitpid(child, 0)
                except ChildProcessError:
                    pass
            if self.lock.is_locked:
                self.lock.unlock()
            self.lock.fd = original_fd
            self.lock.filename = original_path
            self.lock.core.set_lock_fd(original_fd, False)
            if new_fd is not None:
                os.close(new_fd)
            if os.path.exists(path):
                os.unlink(path)
            os.rmdir(directory)
            for fd in (read_fd, write_fd):
                if fd >= 0:
                    try:
                        os.close(fd)
                    except OSError:
                        pass


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

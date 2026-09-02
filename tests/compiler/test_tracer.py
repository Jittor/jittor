# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import os
from pathlib import Path
import subprocess as sp
import sys

REPO_PYTHON = str(Path(__file__).resolve().parents[2] / "python")


def _child_env():
    """Environment for a child that has to import *this* checkout's jittor.

    pytest puts the checkout on sys.path for its own process only; it does not
    export PYTHONPATH, so a bare `python child.py` imports whatever jittor is
    installed. The parent does export `cache_name`, though, so such a child
    loads the core this checkout just built while running some *other*
    checkout's Python layer -- a new core against an old compiler.py. That
    combination fails on whatever the two disagree about (it surfaced as
    `jittor_core has no attribute set_lock_path` when the lock binding moved to
    set_lock_fd) and says nothing about the code under test.
    """
    environment = dict(os.environ)
    environment["PYTHONPATH"] = \
        REPO_PYTHON + os.pathsep + environment.get("PYTHONPATH", "")
    return environment

class TestTracer(unittest.TestCase):
    def test_print_trace(self):
        jt.print_trace()

        if os.name != 'nt':
            # force use addr2line
            with jt.flag_scope(gdb_path=""):
                jt.print_trace()

    @unittest.skipUnless(jt.flags.gdb_path, "GDB is disabled in this test environment")
    def test_breakpoint(self):
        fname = os.path.join(jt.flags.cache_path, "test_breakpoint.py")
        with open(fname, 'w') as f:
            f.write("""
import jittor as jt
with jt.flag_scope(extra_gdb_cmd="c;q"):
    jt.flags.gdb_attach = 1
""")
        completed = sp.run(
            (sys.executable, fname), env=_child_env(),
            stdout=sp.PIPE, stderr=sp.STDOUT, universal_newlines=True)
        out = completed.stdout
        print(out)
        assert "Attaching to" in out

    @unittest.skipIf(os.name == 'nt', "POSIX fork/waitpid contract")
    def test_gdb_backtrace_wait_is_bounded(self):
        """A wedged debugger must not block the traced process forever.

        ``print_trace`` forks GDB and waits for it.  When that wait was
        unbounded, a GDB that hung -- or that a crash reporter intercepted --
        froze the whole process, so a single crashing test could stall an
        entire suite.  Stand in a debugger that never exits and require the
        call to return.
        """
        fake_gdb = os.path.join(jt.flags.cache_path, "hanging_gdb.sh")
        with open(fake_gdb, "w") as f:
            # exec, so the stand-in debugger *is* the forked child. A plain
            # "sleep 600" would leave a grandchild holding the inherited stdout
            # after print_trace kills the child, and the reader below would then
            # block for the full 600s waiting for that pipe to close.
            f.write("#!/bin/sh\nexec sleep 600\n")
        os.chmod(fake_gdb, 0o755)

        fname = os.path.join(jt.flags.cache_path, "test_gdb_timeout.py")
        with open(fname, 'w') as f:
            f.write("""
import time
import jittor as jt
with jt.flag_scope(gdb_path=%r, gdb_trace_timeout=2):
    start = time.time()
    jt.print_trace()
    print("ELAPSED", time.time() - start)
""" % fake_gdb)
        completed = sp.run(
            (sys.executable, fname), env=_child_env(),
            stdout=sp.PIPE, stderr=sp.STDOUT, universal_newlines=True, timeout=300,
        )
        out = completed.stdout
        assert "ELAPSED" in out, out
        elapsed = float(out.split("ELAPSED")[1].split()[0])
        assert elapsed < 30, "print_trace waited {}s for a hung debugger".format(elapsed)

    def test_segfault(self):
        if os.name == 'nt':
            a = jt.array([1,2,3])
            b = jt.array([1,2,300000000])
            c = a[b]
            try:
                c.sync()
            except Exception as e:
                assert "access violation reading" in str(e)
        


if __name__ == "__main__":
    unittest.main()

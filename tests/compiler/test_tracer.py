# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import os
import subprocess as sp
import sys

class TestTracer(unittest.TestCase):
    def test_print_trace(self):
        jt.print_trace()

        if os.name != 'nt':
            # force use addr2line
            with jt.flag_scope(gdb_path=""):
                jt.print_trace()

    def test_breakpoint(self):
        fname = os.path.join(jt.flags.cache_path, "test_breakpoint.py")
        with open(fname, 'w') as f:
            f.write("""
import jittor as jt
with jt.flag_scope(extra_gdb_cmd="c;q"):
    jt.flags.gdb_attach = 1
""")
        out = sp.getoutput(sys.executable+' '+fname)
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
            f.write("#!/bin/sh\nsleep 600\n")
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
        out = sp.getoutput(sys.executable + ' ' + fname)
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
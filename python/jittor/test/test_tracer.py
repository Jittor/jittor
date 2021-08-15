# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
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


if __name__ == "__main__":
    unittest.main()
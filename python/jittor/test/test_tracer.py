# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt

class TestTracer(unittest.TestCase):
    def test_print_trace(self):
        jt.print_trace()

        # force use addr2line
        jt.flags.gdb_path = ""
        with jt.flag_scope(gdb_path=""):
            jt.print_trace()


if __name__ == "__main__":
    unittest.main()
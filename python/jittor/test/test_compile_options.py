# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import os
from .test_log import find_log_with_re
from .test_fused_op import retry

class TestCompileOptions(unittest.TestCase):
    def test(self):
        a = jt.array([1,2,3])
        a.sync()
        assert a.compile_options=={}
        a.compile_options = {"compile_shapes":1}
        assert a.compile_options=={"compile_shapes":1}
        b = a+a
        assert b.compile_options=={}
        with jt.flag_scope(compile_options={"compile_shapes":1}):
            c = a+b
        assert c.compile_options=={"compile_shapes":1}
        with jt.profile_scope() as report:
            c.sync()
        assert len(report)==2 and "compile_shapes:1" in report[1][0]


if __name__ == "__main__":
    unittest.main()
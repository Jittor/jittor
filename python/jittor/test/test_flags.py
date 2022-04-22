# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
from .test_core import expect_error

class TestFlags(unittest.TestCase):
    def test_error(self):
        def check(): jt.flags.asdasd=1
        expect_error(check)
    
    def test_get_set(self):
        prev = jt.flags.log_v
        jt.flags.log_v=1
        assert jt.flags.log_v == 1
        jt.flags.log_v=prev
        assert jt.flags.log_v == prev
    
    def test_scope(self):
        prev = jt.flags.log_v
        with jt.flag_scope(log_v=1):
            assert jt.flags.log_v == 1
        assert jt.flags.log_v == prev


if __name__ == "__main__":
    unittest.main()

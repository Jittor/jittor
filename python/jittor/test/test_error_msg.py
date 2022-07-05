# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: 
#    Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np

class TestErrorMsg(unittest.TestCase):

    def test_error_msg(self):
        a = jt.array([3,2,1])
        b = jt.code(a.shape, a.dtype, [a],
            cpu_header="""
                #include <algorithm>
                @alias(a, in0)
                @alias(b, out)
            """,
            cpu_src="""
                for (int i=0; i<a_shape0; i++)
                    @b(i) = @a(i);
                std::sort(&@b(0), &@b(in0_shape0));
                throw std::runtime_error("???");
            """
        )
        msg = ""
        try:
            print(b)
        except Exception as e:
            msg = str(e)
        assert "[Reason]: ???" in msg
        assert "[Input]: int32[3,]" in msg
        assert "[OP TYPE]: code" in msg
        assert "[Async Backtrace]:" in msg

    @jt.flag_scope(trace_py_var=3)
    def test_error_msg_trace_py_var(self):
        a = jt.array([3,2,1])
        b = jt.code(a.shape, a.dtype, [a],
            cpu_header="""
                #include <algorithm>
                @alias(a, in0)
                @alias(b, out)
            """,
            cpu_src="""
                for (int i=0; i<a_shape0; i++)
                    @b(i) = @a(i);
                std::sort(&@b(0), &@b(in0_shape0));
                throw std::runtime_error("???");
            """
        )
        msg = ""
        try:
            print(b)
        except Exception as e:
            msg = str(e)
        print(msg)
        assert "[Reason]: ???" in msg
        assert "[Input]: int32[3,]" in msg
        assert "[OP TYPE]: code" in msg
        assert "[Async Backtrace]:" in msg
        assert "test_error_msg.py:" in msg



if __name__ == "__main__":
    unittest.main()
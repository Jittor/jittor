# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
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
        # The labels moved when the report was reordered to put the reason
        # first (2026-09-10); what each of them asserted still holds.
        assert "???" in msg, msg
        assert "in:  int32[3,]" in msg, msg
        assert "op: code" in msg, msg
        assert "[Async Backtrace]:" in msg, msg
        # The reason now precedes the machinery rather than following it.
        assert msg.index("???") < msg.index("op: code"), msg

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
        assert "???" in msg, msg
        assert "in:  int32[3,]" in msg, msg
        assert "op: code" in msg, msg
        assert "[Async Backtrace]:" in msg, msg
        assert "test_error_msg.py:" in msg, msg
        assert msg.index("???") < msg.index("op: code"), msg



if __name__ == "__main__":
    unittest.main()
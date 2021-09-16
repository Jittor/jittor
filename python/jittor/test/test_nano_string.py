# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import time
from .test_core import expect_error
import os

mid = 0
if hasattr(os, "uname") and "jittor" in os.uname()[1]:
    mid = 1

class TestNanoString(unittest.TestCase):
    def test(self):
        dtype = jt.NanoString
        t = time.time()
        n = 1000000
        for i in range(n):
            dtype("float")
        t = (time.time() - t)/n
        # t is about 0.01 for 100w loop
        # 92ns one loop
        print("nanostring time", t)
        assert t < [1.5e-7, 1.9e-7][mid], t

        assert (jt.hash("asdasd") == 4152566416)
        assert str(jt.NanoString("float"))=="float32"
        assert jt.NanoString("float")=="float32"
        # py_bind11: 7
        # Tuple call: 1.3
        # fast call (with or with not): 0.9
        # init call 1.5
        # int init: 1.2
        # dtype init(cache): 0.75
        # final: 1.0
    
    def test_type(self):
        import numpy as np
        assert str(jt.NanoString(float)) == "float32"
        assert str(jt.NanoString(np.float)) == "float32"
        assert str(jt.NanoString(np.float32)) == "float32"
        assert str(jt.NanoString(np.float64)) == "float64"
        assert str(jt.NanoString(np.int8)) == "int8"
        assert str(jt.NanoString(np.array([1,2,3]).dtype)) == "int64"

        assert str(jt.NanoString(jt.float)) == "float32"
        assert str(jt.NanoString(jt.float32)) == "float32"
        assert str(jt.NanoString(jt.float64)) == "float64"
        assert str(jt.NanoString(jt.int8)) == "int8"
        assert str(jt.NanoString(jt.array([1,2,3]).dtype)) == "int32"
        assert str(jt.NanoString(jt.sum)) == "add"

        def get_error_str(call):
            es = ""
            try:
                call()
            except Exception as e:
                es = str(e)
            return es
            
        e = get_error_str(lambda: jt.code([1,], {}, [1], cpu_header=""))
        assert "help(jt.ops.code)" in e
        assert "cpu_header=str" in e
        e = get_error_str(lambda: jt.NanoString([1,2,3], fuck=1))
        assert "fuck=int" in str(e)
        assert "(list, )" in str(e)
        


if __name__ == "__main__":
    unittest.main()
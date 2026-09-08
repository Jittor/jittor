# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import time

class TestNanoString(unittest.TestCase):
    def test(self):
        dtype = jt.NanoString

        def best_call_time(call, argument):
            n = 250000
            timings = []
            for _ in range(5):
                start = time.perf_counter()
                for _ in range(n):
                    call(argument)
                timings.append((time.perf_counter() - start) / n)
            return min(timings)

        nano_time = best_call_time(dtype, "float")
        builtin_time = best_call_time(int, "1")
        print("nanostring time", nano_time, "builtin int time", builtin_time)
        assert nano_time < builtin_time * 1.25, (nano_time, builtin_time)

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
        assert str(jt.binary_dtype_infer("add", "int8", "int8")) == "int8"
        assert str(jt.binary_dtype_infer("add", "uint8", "uint8")) == "uint8"

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

# ***************************************************************
# Copyright (c) 2020 Jittor. Authors:
#   Meng-Hao Guo <guomenghao1997@gmail.com>
#   Dun Liang <randonlang@gmail.com>.
#
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
import unittest
import sys, os
from subprocess import getoutput

class TestLazyExecution(unittest.TestCase):
    @unittest.skipIf(not jt.has_cuda, "No cuda found")
    def test_lazy_execution(self):
        code = """
import jittor as jt
jt.flags.use_cuda = 1

a = jt.zeros(1)
b = jt.code([1], a.dtype, [a],
cuda_header='''
#include <assert.h>
''',
cuda_src='''
__global__ void kernel(float32* a, float32* b) {
    b[0] = a[0];
    assert(a[0] == 1);
}
kernel<<<1,1>>>(in0_p, out0_p);
''')
c = a+b
print(c)
"""
        fpath = os.path.join(jt.flags.cache_path, "lazy_error.py")
        with open(fpath, 'w') as f:
            f.write(code)
        res = getoutput(f"{sys.executable} {fpath}")
        assert 'print(c)' in res
        res = getoutput(f"lazy_execution=0 {sys.executable} {fpath}")
        assert "''')" in res
        


if __name__ == "__main__":
    unittest.main()

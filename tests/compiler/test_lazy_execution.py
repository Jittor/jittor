# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers:
#   Meng-Hao Guo <guomenghao1997@gmail.com>
#   Dun Liang <randonlang@gmail.com>.
#
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
import unittest
import os

from _helpers.child_process import python_executable, shell

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
        # shell(), not run_python_child(): this child is *meant* to abort --
        # the assertion inside the kernel is the thing being tested. Reaped
        # directly, jittor's own SIGCHLD handler in this process reads a child
        # that dumped core as "maybe out of memory" and quick-exits pytest:
        #
        #   [e log.cc:250] Caught SIGCHLD. Maybe out of memory ... quick exit
        #
        # The whole session then disappears with exit status 1 and no output
        # at all -- not even a summary. Keeping /bin/sh in between leaves the
        # shell as the reaped child, and it exits normally.
        command = "%s %s" % (python_executable(), fpath)
        res = shell(command, merge_stderr=True).stdout
        # Lazy execution attributes the failure to the point the graph is
        # forced, which is `print(c)` rather than the statement that built
        # the operator.
        assert 'print(c)' in res, res

        res = shell(command, env={"lazy_execution": "0"},
                    merge_stderr=True).stdout
        # With it off, the same failure is attributed to the statement that
        # built the operator. This used to look for the quotes that close
        # cuda_src, which stopped being reachable when CPython began printing
        # only the first physical line of a multi-line call with a `^^^^`
        # marker under it (3.11) -- so the assertion had become about
        # traceback formatting rather than about lazy execution.
        assert 'jt.code(' in res, res
        assert 'print(c)' not in res, res
        


if __name__ == "__main__":
    unittest.main()

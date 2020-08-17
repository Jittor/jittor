# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import os
import numpy as np

class SimpleAsmParser:
    def __init__(self, src):
        funcs = []
        for s in src.split(".globl"):
            funcs.append(s.splitlines())
        self.funcs = funcs

    def count_instructions(self, func_name, ins_name):
        f = None
        for func in self.funcs:
            if func_name in func[0]:
                assert f is None, f"Duplicate func name {func_name}"
                f = func
        assert not (f is None), f"function {func_name} not found"
        count = 0
        for ins in f:
            if ins_name in ins:
                count += 1
        return count


class TestParallelPass(unittest.TestCase):
    def check(self, use_int32):
        n = 1024
        a = jt.random((n, n))
        b = jt.random((n, n))
        a.data, b.data
        with jt.profile_scope(compile_options = {
            "compile_shapes":1, "parallel":2, "try_use_32bit_index":use_int32
        }, try_use_32bit_index = use_int32) as rep:
            c = a + b
            nc = c.data
        assert len(rep) == 2
        assert (a.data+b.data==nc).all()
        fname = rep[1][1]
        with open(fname) as f:
            src = f.read()
            assert "thread_id" in src
        with open(fname.replace(".cc", ".s")) as f:
            asm = SimpleAsmParser(f.read())
        func_name = "run"
        ca = asm.count_instructions(func_name, "vmova")
        cu = asm.count_instructions(func_name, "vmovu")
        return ca, cu

    def test_int32_align(self):
        ca, cu = self.check(1)
        if jt.flags.cc_type=="clang":
            assert ca>1 and cu<=1, (ca, cu)
    
    def test_int64_align(self):
        ca, cu = self.check(0)
        if jt.flags.cc_type=="clang":
            assert ca>1 and cu<=1, (ca, cu)

class TestParallelPass2(TestParallelPass):
    def check(self, use_int32):
        n = 1024
        a = jt.random((n, n*8))
        b = jt.random((n*8,))
        a.data, b.data
        with jt.profile_scope(compile_options = {
            "compile_shapes":1, "parallel":1, "split1":n, "order1":1
        }, try_use_32bit_index = use_int32) as rep:
            c = a - b
            # def func(a, b, c, tid, num):
            #     for i in range(tid*1024, 1024*8, num*1024):
            #         for j in range(n):
            #              for k in range(n):
            #                  c[j*1024*8 + i+k] = a[j*1024*8 + i+k] - b[i+k]
            nc = c.data
        assert len(rep) == 2
        assert (a.data-b.data==nc).all()
        fname = rep[1][1]
        with open(fname) as f:
            src = f.read()
            assert "thread_id" in src
        with open(fname.replace(".cc", ".s")) as f:
            asm = SimpleAsmParser(f.read())
        func_name = "run"
        ca = asm.count_instructions(func_name, "vmova")
        cu = asm.count_instructions(func_name, "vmovu")
        return ca, cu

class TestParallelPass3(unittest.TestCase):
    def test(self):
        def check(ndim, depth, tdim):
            a = jt.random([16]*ndim)
            a.sync()
            compile_options = {"parallel":1}
            if depth is not None:
                compile_options["max_parallel_depth"] = depth
            with jt.profile_scope(compile_options=compile_options) as rep:
                b = (a+a).data
            assert np.allclose(a.data*2, b)
            assert len(rep) == 2
            fname = rep[1][1]
            with open(fname) as f:
                src = f.read()
                for i in range(tdim):
                    assert f"tnum{i}" in src
                assert f"tnum{tdim}" not in src
        check(1, None, 0)
        check(2, None, 1)
        check(3, None, 2)
        check(4, None, 2)
        check(5, None, 2)
        check(5, 3, 3)
        check(5, 4, 4)
        check(5, 5, 5)
        if jt.compiler.has_cuda:
            with jt.flag_scope(use_cuda=1):
                check(1, 2, 1)
                check(2, 2, 2)
                check(3, 2, 2)
                check(4, 2, 2)
                check(5, 2, 2)
                check(5, 3, 3)
                check(5, 4, 4)
                check(5, 5, 5)

    def reduce_check(self, ndim, depth, tdim, rdim, has_atomic, order=[], split=[], **args):
        shape = [8]*ndim
        a = jt.random(shape)
        a.sync()
        config = {
            "parallel":1, "max_parallel_depth":depth
        }
        for k in args:
            config[k] = args[k]
        if not isinstance(rdim, list):
            rdim = [rdim]
        rdim = tuple(rdim)
        nshape = [1024, 256, 128][len(rdim)]
        for d in rdim: shape[d] = nshape
        for i,o in enumerate(order):
            config[f"order{i}"] = o
        for i,o in enumerate(split):
            config[f"split{i}"] = o
        with jt.profile_scope(
            compile_options = config,
            enable_tuner = 0
        ) as rep:
            b = a.sum(rdim).data
        assert len(rep) == 2
        fname = rep[1][1]
        with open(fname) as f:
            src = f.read()
            for i in range(tdim):
                assert f"tnum{i}" in src
            assert f"tnum{tdim}" not in src, f"tnum{tdim}"
            src_has_atomic = "atomic_add" in src or "atomicAdd" in src
            assert has_atomic == src_has_atomic
        assert np.allclose(a.data.sum(rdim), b), (b.sum(), a.data.sum())

    def test_reduce(self):
        check = lambda *a, **kw: self.reduce_check(*a, **kw)
        check(1, 2, 1, 0, 1)
        check(2, 1, 1, 1, 0)
        check(2, 1, 1, 0, 1)
        check(2, 1, 1, 0, 1, [0,0])
        check(2, 1, 1, 0, 0, [0,1])
        check(2, 1, 1, 0, 0, [0,1], [0,64])
        check(2, 1, 1, [0,1], 1, [0,1])
        check(3, 1, 1, [1,2], 0)
        check(3, 1, 1, [0,1], 1)
        check(3, 1, 1, [0,1], 0, [0,0,2])
        check(3, 2, 2, [2], 0)
        if jt.flags.use_cuda:
            # loop is not merged so parallel depth 2
            check(3, 2, 2, [1], 1)
        else:
            check(3, 2, 1, [1], 0)
        check(3, 2, 2, [1], 1, merge=0)
        check(4, 2, 2, [2,3], 0)
        check(4, 2, 2, [0,3], 1)

    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    def test_reduce_cuda(self):
        with jt.flag_scope(use_cuda=1):
            self.test_reduce()

if __name__ == "__main__":
    unittest.main()

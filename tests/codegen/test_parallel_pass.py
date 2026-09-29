
from _helpers import capability as _test_capability
# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import os
import numpy as np
import re

class SimpleAsmParser:
    def __init__(self, src):
        funcs = []
        # GCC emits ``.global`` on AArch64 and ``.globl`` on x86.
        for s in re.split(r"^\s*\.(?:global|globl)\s+", src,
                          flags=re.MULTILINE):
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


def _assembly_for(source):
    """Compile a cached kernel source to assembly.

    The tree no longer keeps a ``.s`` beside the ``.cc`` (the assembly-text
    rewriter went in acfed956), so run the exact command the cache recorded
    for the kernel -- the first line of its ``.so.key`` -- with ``-S``.
    """
    import subprocess
    import tempfile
    command = open(source[:-len(".cc")] + ".so.key", encoding="utf8").readline().strip()
    command = re.sub(r'\s-o\s+"[^"]*"\s*$', "", command).replace(" -shared ", " ")
    with tempfile.TemporaryDirectory() as scratch:
        listing = os.path.join(scratch, "kernel.s")
        subprocess.run(command + ' -S -o "%s"' % listing, shell=True, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
        with open(listing, encoding="utf8") as f:
            return f.read()


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
        asm = SimpleAsmParser(_assembly_for(fname))
        func_name = "run"
        ca = asm.count_instructions(func_name, "vmova")
        cu = asm.count_instructions(func_name, "vmovu")
        return ca, cu

    def test_int32_align(self):
        ca, cu = self.check(1)
        if jt.introspection.policy.startup.cc_type=="clang":
            assert ca>1 and cu<=1, (ca, cu)
    
    def test_int64_align(self):
        ca, cu = self.check(0)
        if jt.introspection.policy.startup.cc_type=="clang":
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
        asm = SimpleAsmParser(_assembly_for(fname))
        func_name = "run"
        ca = asm.count_instructions(func_name, "vmova")
        cu = asm.count_instructions(func_name, "vmovu")
        return ca, cu

class TestParallelPass3(unittest.TestCase):
    def test(self):
        def check(ndim, depth, tdim):
            a = jt.random([16]*ndim)
            a.sync()
            compile_options = {"parallel":1, "merge_loop_var": self.merge_loop_var}
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
                # ParallelPass emits one get_thread_range_log per parallel
                # dimension -- that is this dimension's own bit count -- and
                # then accumulates them into the cumulative boundaries the
                # kernel decodes tid{i} from. Both backends get the same shape;
                # the CPU side used to have the accumulation spliced into these
                # lines afterwards by a regex over the finished source
                # (op_compiler.cc), which is why it used to be asserted in a
                # different form here.
                # except a CUDA kernel over one flat loop, whose grid is
                # sized to the elements (see the test below)
                flat = jt.flags.use_cuda and tdim == 1
                for i in range(tdim):
                    if flat:
                        assert "int tn0 = std::min(NanoVector::get_nbits(" in src, src
                    else:
                        assert f"int tn{i} = get_thread_range_log" in src, src
                for i in range(tdim-1):
                    assert f"tn{i}=tn{i}+tn{i+1};" in src, src
                assert "thread_num /= thread_num_left;" not in src
                if tdim:
                    # threads actually handed out, i.e. what the omp region is
                    # entered with on CPU and what sizes the grid on CUDA
                    assert "thread_num=1<<tn0;" in src, src
        self.merge_loop_var = 0
        check(1, None, 0)
        check(2, None, 1)
        check(3, None, 2)
        check(4, None, 2)
        check(5, None, 2)
        check(5, 3, 3)
        check(5, 4, 4)
        check(5, 5, 5)
        if _test_capability.check_accelerator('cuda', backend=jt).enabled:
            with jt.flag_scope(use_cuda=1):
                check(1, 2, 1)
                check(2, 2, 2)
                check(3, 2, 2)
                check(4, 2, 2)
                check(5, 2, 2)
                check(5, 3, 3)
                check(5, 4, 4)
                check(5, 5, 5)

    @unittest.skipUnless(_test_capability.check_accelerator('cuda', backend=jt).enabled,
                         "the thread count is a CUDA launch shape")
    def test_a_flat_elementwise_kernel_gets_a_thread_per_element(self):
        """One flat loop gets a grid sized to its elements; a nest keeps block_num blocks.

        Over a 12.6 M-element GELU backward on a 4090, 2^23 threads walking
        the range took 168 us, 2^25 (mostly idle) 189, a thread per element 153.
        """
        def launch(shape, **options):
            a = jt.random(shape)
            a.sync()
            with jt.flag_scope(use_cuda=1), jt.profile_scope(
                    compile_options=dict(options, parallel=1)) as rep:
                b = (a + a).data
            np.testing.assert_allclose(b, a.data * 2)
            with open(rep[1][1]) as f:
                return f.read()
        src = launch([64, 1024])
        self.assertRegex(src, r"int tn0 = std::min\(NanoVector::get_nbits\(")
        self.assertRegex(src, r"int p1 = \(int\)std::max\(std::min\(")
        nest = launch([64, 1024], merge_loop_var=0, max_parallel_depth=2)
        self.assertNotRegex(nest, r"int p1 = \(int\)std::max\(std::min\(")
        # Every element exactly once, around the powers of two the grid is
        # rounded against.
        for n in (1, 31, 256, 257, 4095, 65537, (1 << 20) + 3):
            with self.subTest(n=n), jt.flag_scope(use_cuda=1):
                a = np.arange(n, dtype="float32")
                np.testing.assert_array_equal((jt.array(a) * 2 + 1).numpy(), a * 2 + 1)

    @unittest.skipUnless(_test_capability.check_accelerator('cuda', backend=jt).enabled,
                         "the vector loads are a CUDA kernel's")
    def test_a_flat_kernel_moves_its_tensors_in_vectors(self):
        """Tensors read or written at the loop index move 16 bytes at a time.

        A per-channel operand read through a broadcast stays per element; a
        view that is not aligned for the vectors takes the scalar loop, as
        does whatever is left past the last whole vector.
        """
        rng = np.random.RandomState(3)
        for dtype, lanes in (("float16", 8), ("float32", 4)):
            x_np = rng.rand(4, 7, 5, 64).astype(dtype)
            s_np = rng.rand(64).astype(dtype)
            with self.subTest(dtype=dtype), jt.flag_scope(use_cuda=1):
                x, sc = jt.array(x_np), jt.array(s_np)
                jt.sync([x, sc])
                with jt.profile_scope() as rep:
                    y = (x * sc + 1).maximum(0)
                    y.sync()
                with open(rep[1][1]) as f:
                    src = f.read()
                self.assertIn("jt_vecs", src)
                self.assertIn("for (int jt_k = 0; jt_k < %d; jt_k++)" % lanes, src)
                ref = np.maximum(x_np.astype("float32") * s_np.astype("float32") + 1, 0)
                np.testing.assert_allclose(y.numpy().astype("float32"), ref, rtol=1e-2, atol=1e-2)
                flat = x_np.reshape(-1)
                for start, n in ((1, 1001), (0, 1001), (3, 5), (0, lanes), (2, 2 * lanes + 1)):
                    got = (jt.array(flat)[start:start + n] * 2 + 1).numpy().astype("float32")
                    np.testing.assert_allclose(got, flat[start:start + n].astype("float32") * 2 + 1,
                                               rtol=1e-2, atol=1e-2)

    def reduce_check(self, ndim, depth, tdim, rdim, has_atomic, order=[], split=[], **args):
        shape = [8]*ndim
        a = jt.random(shape)
        a.sync()
        config = {
            "parallel":1, "max_parallel_depth":depth, "merge_loop_var": self.merge_loop_var
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
            # cumulative thread-range boundaries come from ParallelPass on
            # both backends now, not from a regex over the CPU source
            assert "thread_num /= thread_num_left;" not in src
            for i in range(tdim-1):
                assert f"tn{i}=tn{i}+tn{i+1};" in src, src
            src_has_atomic = "atomic_add" in src or "atomicAdd" in src
            assert has_atomic == src_has_atomic
        assert np.allclose(a.data.sum(rdim), b), (b.sum(), a.data.sum())

    def test_reduce(self):
        self.merge_loop_var = 0
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
        if jt.introspection.policy.runtime.use_cuda:
            # loop is not merged so parallel depth 2
            check(3, 2, 2, [1], 1)
        else:
            check(3, 2, 1, [1], 0)
        check(3, 2, 2, [1], 1, merge=0)
        check(4, 2, 2, [2,3], 0)
        check(4, 2, 2, [0,3], 1)

    def test_reduce_with_merge_loop_var(self):
        self.merge_loop_var = 1
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
        check(3, 2, 1, [2], 0)
        if jt.introspection.policy.runtime.use_cuda:
            # loop is not merged so parallel depth 2
            check(3, 2, 2, [1], 1)
        else:
            check(3, 2, 1, [1], 0)
        check(3, 2, 2, [1], 1, merge=0)
        check(4, 2, 1, [2,3], 0)
        check(4, 2, 2, [0,3], 1)

    @unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No CUDA found")
    def test_reduce_cuda(self):
        with jt.flag_scope(use_cuda=1):
            self.test_reduce()

if __name__ == "__main__":
    unittest.main()

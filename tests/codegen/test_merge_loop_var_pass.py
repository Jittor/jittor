# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""MergeLoopVarPass: fusing two nested loops into one.

The merged loop is named after the ranges it covers, so the name has to say
which ranges those are without ambiguity -- see parse_loop_id in kernel_ir.h.
"range0_1" is the merge of ranges 0 and 1; "range10" is range number 10.
"""
import itertools
import re
import unittest

import numpy as np

import jittor as jt

class TestMergeLoopVarPass(unittest.TestCase):
    def test(self):
        a = jt.ones([10,10,10,10])
        a.sync()
        with jt.profile_scope() as rep:
            b = a.sum([2,3])
            b.sync()
        with open(rep[1][1]) as f:
            src = f.read()
            assert "range0_1" in src
            assert "range2_3" in src

    def test2(self):
        a = jt.ones([10,10,10,10])
        a.sync()
        with jt.profile_scope() as rep:
            b = a + 1
            b.sync()
        with open(rep[1][1]) as f:
            src = f.read()
            assert "range0_1_2_3" in src

    def test3(self):
        a = jt.ones([10,10,10,10])
        x = jt.ones([1,10,1,1])
        a.sync(), x.sync()
        with jt.profile_scope() as rep:
            b = a + x
            b.sync()
        with open(rep[1][1]) as f:
            src = f.read()
            assert "range2_3" in src

    def test4(self):
        # don't optimize reindex like op yet
        a = jt.ones([10,10,10,10])
        a.sync()
        with jt.profile_scope() as rep:
            b = a.reindex_reduce("add", [10,10], ["i0","i1"])
            b.sync()
        with open(rep[1][1]) as f:
            src = f.read()
            assert "range2_3" not in src

    def test5(self):
        a = jt.ones([10,10,10,10])
        a.sync()
        with jt.profile_scope() as rep:
            b = a.sum([1])
            b.sync()
        with open(rep[1][1]) as f:
            src = f.read()
            assert "range0_1" not in src
            assert "range2_3" in src

    def test_merged_range_name_says_which_ranges(self):
        """A merged range's name must not also read as a single range.

        The name used to be the plain concatenation of the two loop ids, so the
        merge of ranges 0 and 1 was called "range01" and the merge of 1 and 0
        would be called "range10" -- the name of range number 10.  The merged
        range is only defined ``if (!find_define(...))``, so a name that already
        exists is reused rather than defined and the merged loop silently runs
        the wrong number of iterations.
        """
        a = jt.ones([4] * 10)
        a.sync()
        with jt.profile_scope(compile_options={"_mlv_name": 1}) as rep:
            b = a + 1
            b.sync()
        with open(rep[1][1]) as f:
            src = f.read()
        # a merged range is defined as a product of the ranges it covers
        merged = re.findall(r"\brange([0-9_]+) = ([^;]*\*[^;]*);", src)
        assert merged, "expected MergeLoopVarPass to merge something:\n" + src
        for name, rhs in merged:
            assert "_" in name, (
                "merged range 'range%s' is spelled in plain digits, so it also "
                "reads as range number %s: %s" % (name, name, rhs))

    def test_many_ranges_still_compute_the_right_values(self):
        """10 dimensions (the NanoVector limit) plus splits, which push the
        range count past 10 and so past the point where a name is one digit.

        No splits on CUDA: ParallelPass always runs there and cannot resolve the
        inner range a split produces (``::min(range{i}-id{i}, stride{i})``, which
        is defined in the outer loop and varies with it), so every CUDA kernel
        with a split fails to compile on ``Check failed: def``. That is the
        pre-existing split/parallel incompatibility recorded under 1.04, not
        something this pass can do anything about.
        """
        splits = (0,) if jt.flags.use_cuda else (0, 1, 2, 3)
        for nd, nsplit in itertools.product((7, 8, 9, 10), splits):
            shape = [2] * nd
            a = jt.random(shape)
            a.sync()
            co = {"_mlv_dims": nd * 10 + nsplit}
            for k in range(nsplit):
                co["split%d" % k] = 2
            with jt.flag_scope(compile_options=co):
                got = (a + a).numpy()
                red = jt.reduce(a, "add", (nd - 1,)).numpy()
            ref = a.numpy()
            np.testing.assert_allclose(got, ref * 2, rtol=1e-5, atol=1e-5,
                                       err_msg="ndim=%d splits=%d" % (nd, nsplit))
            np.testing.assert_allclose(red, ref.sum(axis=nd - 1), rtol=1e-4,
                                       atol=1e-4,
                                       err_msg="ndim=%d splits=%d" % (nd, nsplit))


@unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
class TestMergeLoopVarPassCuda(TestMergeLoopVarPass):
    def setUp(self):
        self._previous_use_cuda = jt.flags.use_cuda
        jt.flags.use_cuda = 1
    def tearDown(self):
        jt.sync_all()
        jt.flags.use_cuda = self._previous_use_cuda

if __name__ == "__main__":
    unittest.main()

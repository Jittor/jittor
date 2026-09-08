# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Dun Liang <randonlang@gmail.com>.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import re
import unittest
import numpy as np
import jittor as jt

PRAGMA = "#pragma omp parallel for"


def kernel_source(build):
    """Run `build` in a profile scope and return the generated source."""
    with jt.profile_scope() as rep:
        out = build()
        out.sync()
    with open(rep[1][1]) as f:
        return f.read(), out


class TestCpuParallelPass(unittest.TestCase):
    def setUp(self):
        self.saved = jt.flags.use_cuda
        jt.flags.use_cuda = 0

    def tearDown(self):
        jt.flags.use_cuda = self.saved

    def test_elementwise_is_parallelised(self):
        a = jt.random((512, 1024))
        b = jt.random((512, 1024))
        a.sync(), b.sync()
        src, out = kernel_source(lambda: a + b)
        assert PRAGMA in src
        np.testing.assert_allclose(out.numpy(), a.numpy() + b.numpy(),
                                   rtol=1e-6, atol=1e-6)

    def test_both_branches_are_emitted(self):
        # One copy takes the threads, the other keeps the vector width the
        # parallel region would have halved. The serial copy carries no pragma
        # and sits under the negated guard.
        a = jt.random((512, 1024))
        b = jt.random((512, 1024))
        a.sync(), b.sync()
        src, _ = kernel_source(lambda: a + b)
        assert src.count(PRAGMA) == src.count("#pragma omp")
        assert "if (!(" in src

    def test_full_reduction_is_left_alone(self):
        # Every iteration accumulates into the one output element, so no loop
        # variable scales the store index and the disjointness test must reject
        # all of them. Threading this would be a race, not a slow kernel.
        a = jt.random((512, 1024))
        a.sync()
        src, out = kernel_source(lambda: a.sum())
        accumulate = re.compile(r"(\w+)\[(\w+)\]\s*=\s*\(*\(\1\[\2\]\)")
        at = src.find(PRAGMA)
        while at >= 0:
            assert not accumulate.search(src[at:at + 2000]), \
                "a full reduction's read-modify-write was parallelised"
            at = src.find(PRAGMA, at + 1)
        np.testing.assert_allclose(out.numpy(),
                                   a.numpy().astype("float64").sum(),
                                   rtol=1e-4, atol=1e-4)

    def test_reduction_over_a_split_output_dimension(self):
        # Summing over dimension 0 leaves the output dimension outermost, and
        # the loop over it is split into tiles. Distinct tiles accumulate into
        # distinct output elements, so this one is legitimately threaded -- the
        # check that matters is that the numbers survive it.
        a = jt.random((512, 1024))
        a.sync()
        _, out = kernel_source(lambda: a.sum(0))
        np.testing.assert_allclose(out.numpy(),
                                   a.numpy().astype("float64").sum(0),
                                   rtol=1e-5, atol=1e-5)

    def test_row_reduction_matches_numpy(self):
        # Reducing over the innermost dimension leaves the outer loop free, and
        # this is the shape the accumulator pass rewrites first.
        a = jt.random((256, 768))
        a.sync()
        src, out = kernel_source(lambda: a.sum(1))
        assert PRAGMA in src
        np.testing.assert_allclose(out.numpy(),
                                   a.numpy().astype("float64").sum(1),
                                   rtol=1e-5, atol=1e-5)

    def test_fused_softmax_backward_matches_numpy(self):
        x = np.random.randn(2, 3, 64, 128).astype("float32")
        v = jt.array(x)
        y = jt.nn.softmax(v, dim=-1)
        got = jt.grad((y * y).sum(), v).numpy()

        d = x.astype("float64")
        e = np.exp(d - d.max(-1, keepdims=True))
        p = e / e.sum(-1, keepdims=True)
        g = 2 * p
        expect = p * (g - (g * p).sum(-1, keepdims=True))
        np.testing.assert_allclose(got, expect, rtol=1e-4, atol=1e-6)

    def test_odd_trailing_dimension(self):
        # Exercises the vectorised loop's tail, which the two branches must
        # handle identically.
        a = jt.random((97, 131))
        a.sync()
        out = (a * a + 1).numpy()
        ref = a.numpy()
        np.testing.assert_allclose(out, ref * ref + 1, rtol=1e-6, atol=1e-6)

    def test_check_cache_mode_is_untouched(self):
        # That mode pairs every access with a memory_checker call; an
        # uninstrumented copy of the loop would skew the figures it reports.
        a = jt.random((100, 10000))
        a.sync()
        with jt.profile_scope(compile_options={
            "check_cache": 1, "replace_strategy": 1, "page_size": 4 << 10,
            "vtop": 0,
            "tlb_size": 64, "tlb_ways": 4, "tlb_line_size": 1,
            "L1_size": 32 << 10, "L1_ways": 8, "L1_line_size": 64,
            "L2_size": 256 << 10, "L2_ways": 8, "L2_line_size": 64,
            "L3_size": 15 << 20, "L3_ways": 20, "L3_line_size": 64,
        }, enable_tuner=0) as rep:
            c = a.sum(1)
            c.sync()
        with open(rep[1][1]) as f:
            assert PRAGMA not in f.read()
        np.testing.assert_allclose(c.numpy(),
                                   a.numpy().astype("float64").sum(1),
                                   rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()

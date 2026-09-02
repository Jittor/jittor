# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""SharedReducePass: block-wide shared-memory reduction, off by default.

The pass rewrites the per-thread ``atomicAdd`` that ends a CUDA reduction into

    acc = shared_reduce<T, shared_reduce_add>(acc);
    if (threadIdx.x == 0) atomicAdd(&yp[yid], acc);

so one block writes its output once instead of once per thread. It first
re-plans the thread ranges (``apply_reduce_thread_order``) so that a block
covers whole reduced dimensions, which is what makes the block-wide fold legal.

``SharedReducePass::run`` returns immediately unless ``para_opt_level >= 4`` and
the default is 3, so it never runs in a stock build -- that is why no generated
kernel in a normal workload contains ``shared_reduce``. The tests below pin both
halves of that: it stays off by default, and it still produces correct code when
turned on. Measurements that say why the default is 3 live in
agent/skills/cuda-reduction-strategy-comparison/.
"""
import unittest

import numpy as np

import jittor as jt


@unittest.skipIf(not jt.has_cuda, "No cuda found")
class TestSharedReduce(unittest.TestCase):
    def setUp(self):
        self._use_cuda = jt.flags.use_cuda
        self._level = jt.flags.para_opt_level
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.flags.para_opt_level = self._level
        jt.flags.use_cuda = self._use_cuda

    def _reduce(self, shape, dims, tag):
        """Run one reduction and return (generated source, relative error)."""
        value = np.random.RandomState(abs(hash((shape, dims))) % 2**31)
        value = value.randn(*shape).astype("float32")
        x = jt.array(value, dtype="float32")
        x.sync()
        # a compile option nothing reads, so that each case gets its own kernel
        # instead of the one an earlier case with another para_opt_level left in
        # the cache
        with jt.profile_scope(compile_options={"test_shared_reduce": tag}) as rep:
            got = jt.reduce(x, "add", dims).data
        expected = value.sum(axis=tuple(dims))
        scale = max(1.0, float(np.abs(expected).max()))
        error = float(np.abs(got.reshape(expected.shape) - expected).max()) / scale
        source = open(rep[1][1]).read()
        return source, error

    def test_off_at_the_default_level(self):
        self.assertEqual(jt.flags.para_opt_level, 3)
        source, error = self._reduce((8, 96, 32, 32), (0, 2, 3), 1)
        self.assertLess(error, 1e-5)
        self.assertNotIn("shared_reduce<", source)

    def test_on_at_level_4(self):
        jt.flags.para_opt_level = 4
        source, error = self._reduce((8, 96, 32, 32), (0, 2, 3), 2)
        self.assertLess(error, 1e-5)
        self.assertIn("shared_reduce<", source)
        # one write per block, not one per thread
        self.assertIn("if (threadIdx.x == 0)", source)

    def test_warp_pass_leaves_the_guarded_atomic_alone(self):
        # WarpReducePass runs after SharedReducePass and matches the same
        # atomicAdd. Inside "if (threadIdx.x == 0)" one lane is active, so its
        # shuffle path could never be taken; it must not be emitted at all.
        jt.flags.para_opt_level = 4
        source, error = self._reduce((8, 128, 32, 32), (0, 2, 3), 3)
        self.assertLess(error, 1e-5)
        self.assertIn("shared_reduce<", source)
        self.assertNotIn("_wr_mask", source)

    def test_values_match_over_several_shapes(self):
        jt.flags.para_opt_level = 4
        for index, (shape, dims) in enumerate((
            ((8, 384, 32, 32), (0, 2, 3)),
            ((8, 128, 64, 64), (0, 2, 3)),
            ((4, 32, 64, 64), (2, 3)),
            ((16, 8, 4, 4), (0, 2, 3)),
            ((129, 37), (0,)),
        )):
            with self.subTest(shape=shape, dims=dims):
                source, error = self._reduce(shape, dims, 10 + index)
                self.assertLess(error, 1e-5)

    def test_gradient_through_the_block_reduction(self):
        jt.flags.para_opt_level = 4
        value = np.random.RandomState(3).randn(4, 8, 16, 16).astype("float32")
        x = jt.array(value, dtype="float32")
        x.start_grad()
        loss = (x * 2).sum([2, 3]).sum()
        grad = jt.grad(loss, x).numpy()
        np.testing.assert_allclose(grad, np.full_like(value, 2.0), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()

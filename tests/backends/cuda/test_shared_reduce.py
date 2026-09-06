# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""SharedReducePass: optional block-wide warp/shared-memory hybrid reduction.

The pass rewrites the per-thread ``atomicAdd`` that ends a CUDA reduction into

    acc = shared_reduce<T, shared_reduce_add>(acc);
    if (threadIdx.x == 0) atomicAdd(&yp[yid], acc);

so one block writes its output once instead of once per thread. It first
re-plans the thread ranges (``apply_reduce_thread_order``) so that a block
covers whole reduced dimensions, which is what makes the block-wide fold legal.

Level 4 uses warp shuffles for local folds and shared memory only for one partial
per warp. Tests pin the default warp path, the opt-in block path, and its
one-global-write contract. Measurements live in
agent/skills/cuda-reduction-strategy-comparison/.
"""
import os
import unittest

import numpy as np

import jittor as jt
from jittor_utils.backend_resources import backend_root


@unittest.skipIf(not jt.has_cuda, "No cuda found")
class TestSharedReduce(unittest.TestCase):
    def setUp(self):
        self._use_cuda = jt.flags.use_cuda
        self._level = jt.flags.para_opt_level
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.flags.para_opt_level = self._level
        jt.flags.use_cuda = self._use_cuda

    def _reduce(self, shape, dims, tag, **options):
        """Run one reduction and return (generated source, relative error)."""
        value = np.random.RandomState(abs(hash((shape, dims))) % 2**31)
        value = value.randn(*shape).astype("float32")
        x = jt.array(value, dtype="float32")
        x.sync()
        # a compile option nothing reads, so that each case gets its own kernel
        # instead of the one an earlier case with another para_opt_level left in
        # the cache
        compile_options = {"test_shared_reduce": tag}
        compile_options.update(options)
        with jt.profile_scope(compile_options=compile_options) as rep:
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
        self.assertIn("_wr_mask", source)

    def test_level_4_uses_one_block_write(self):
        jt.flags.para_opt_level = 4
        source, error = self._reduce((8, 96, 32, 32), (0, 2, 3), 2)
        self.assertLess(error, 1e-5)
        self.assertIn("shared_reduce<", source)
        self.assertIn("if (threadIdx.x == 0)", source)
        self.assertNotIn("_wr_mask", source)

    def test_shared_reduce_helper_is_two_stage(self):
        # The helper lives with the CUDA backend's kernel sources, not under
        # python/jittor/src: `fd4d8820d` moved it there and left this assertion
        # pointing at a src/type/cuda_atomic.h that still exists but no longer
        # defines shared_reduce. Ask the build for the directory it compiles
        # rather than spelling a path, so the next move fails loudly instead.
        path = os.path.join(backend_root(jt.compiler.jittor_path, "cuda"),
                            "kernels", "core", "cuda_atomic.h")
        source = open(path).read()
        self.assertIn("inline static T shared_reduce(T u)", source)
        body = source.split("inline static T shared_reduce(T u)", 1)[1]
        body = body.split("\n}\n", 1)[0]
        cuda_body = body.split("#else", 1)[1].split("#endif", 1)[0]
        self.assertIn("__shfl_down_sync", cuda_body)
        self.assertIn("warp_values[32]", cuda_body)
        self.assertEqual(cuda_body.count("__syncthreads()"), 2)

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

    def test_values_match_on_the_unet_reduction_shapes(self):
        # Every reduction the code generator emits for one step of
        # large_diffusers_unet2d, read off the profiler with
        # profiler_record_shape=1 (see the caliber section of
        # cuda-reduction-strategy-comparison). These are the shapes 3.22's
        # acceptance is measured on, so the block path has to be right on them
        # and not only on round synthetic ones.
        jt.flags.para_opt_level = 4
        for index, (shape, dims) in enumerate((
            ((4, 256, 384), (0, 1)),      # linear bias gradients, 24 per step
            ((4, 128, 64, 64), (2, 3)),   # time-embedding broadcast gradients
            ((4, 384, 16, 16), (2, 3)),
            ((4, 256, 32, 32), (2, 3)),
            ((4, 384), (0,)),             # time-embedding linear bias gradient
            ((4, 384, 256), (0, 2)),
            ((4, 32, 12, 256), (2, 3)),   # the six attention GroupNorms that
                                          # fall back to the code generator
        )):
            with self.subTest(shape=shape, dims=dims):
                source, error = self._reduce(shape, dims, 30 + index)
                self.assertIn("shared_reduce<", source)
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

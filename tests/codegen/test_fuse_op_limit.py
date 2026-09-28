# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``fuse_op_limit`` bounds a half-precision fused kernel, and only that.

``count_fuse``'s pass 3 unions neighbouring operators at the same fuse level
with no bound on the group, so one fused kernel can hold an arbitrarily long
chain. In float32 the chain stops itself -- the dtype casts that break it are
already in the graph -- but under ``autocast`` the chain is uniformly half
precision and nothing breaks it. The MiniMax-H3 video VAE decode built kernels
of 361 operators that way and took 15.97 s against the same decode's 8.32 s in
float32 (section 47 of ``docs/results/2026-09-14-vllm-omni-h3-enablement.md``).

Three things have to hold, and the third is why the flag is dtype-guarded:

* the bound does not change results -- fusion is an optimisation;
* it does bound, so a half-precision chain lands in more, narrower kernels;
* a float32 chain is untouched, because bounding every dtype cost 8.1% on a
  40-operator float32 elementwise chain for no benefit.

The width is read back out of the JIT key: ``__get_fused_src`` splices one
``opkey<N>:`` per operator into the kernel name, so counting them counts the
operators in that kernel.
"""
import unittest

import numpy as np

import jittor as jt

from _helpers import capability as _test_capability


def _widths(build, **flags):
    """The fusion widths of one graph, by kernel, under ``flags``."""
    with jt.profile_scope(auto_flush_ops=0, **flags) as report:
        build().sync()
    header, rows = report[0], report[1:]
    name = header.index("Name")
    return [str(row[name]).count("opkey") or 1 for row in rows]


def _chain(dtype, length=60, shape=(64, 256)):
    """A long fusable elementwise chain -- the shape the bound acts on."""
    def build():
        v = jt.array(np.random.RandomState(0).randn(*shape).astype("float32"))
        if dtype != "float32":
            v = v.cast(dtype)
        for _ in range(length):
            v = v * 1.0001 + 0.001
        return v
    return build


@unittest.skipUnless(
    _test_capability.check_accelerator("cuda", backend=jt).enabled,
    "fusion width is a CUDA code-generation property")
class TestFuseOpLimit(unittest.TestCase):

    def setUp(self):
        self.saved = jt.flags.fuse_op_limit

    def tearDown(self):
        jt.flags.fuse_op_limit = self.saved

    def test_the_bound_does_not_change_the_answer(self):
        """Bit-identical, every dtype. A fusion decision is not a numerics one."""
        for dtype in ("float32", "float16", "bfloat16"):
            with self.subTest(dtype=dtype), jt.flag_scope(use_cuda=1):
                build = _chain(dtype)
                jt.flags.fuse_op_limit = 0
                unbounded = build().float32().numpy()
                jt.flags.fuse_op_limit = 4
                bounded = build().float32().numpy()
                np.testing.assert_array_equal(
                    unbounded, bounded,
                    "fuse_op_limit changed the result in %s" % dtype)

    def test_a_half_chain_is_bounded(self):
        """The bound is on the union-find group, and the width is not only that.

        A kernel's reported width is its group *plus* the producers the planner
        shares into it (`exec_plan.cc`'s sharegraph), which are recomputed
        rather than fused. So `fuse_op_limit=8` legitimately leaves kernels
        wider than 8 -- measured, 12 on this chain -- and the two numbers being
        different is the whole reason bounding the sharegraph did nothing for
        the H3 decode while bounding the group worked. Assert the reduction,
        not the literal limit.
        """
        for dtype in ("float16", "bfloat16"):
            with self.subTest(dtype=dtype), jt.flag_scope(use_cuda=1):
                build = _chain(dtype)
                unbounded = max(_widths(build, fuse_op_limit=0))
                bounded = max(_widths(build, fuse_op_limit=8))
                self.assertGreater(
                    unbounded, 30,
                    "the unbounded %s chain did not fuse wide enough to test "
                    "the bound" % dtype)
                self.assertLess(
                    bounded, unbounded // 2,
                    "fuse_op_limit=8 barely moved the %s width: %d -> %d"
                    % (dtype, unbounded, bounded))

    def test_a_float32_chain_is_not_bounded(self):
        """The guard is `var->dtype()` half, and this is what it buys.

        Applied to every dtype the same bound cost 8.1% on this chain in
        float32 (3.649 -> 3.944 ms) while fixing nothing: a float32 graph is
        already broken up by its own casts.

        The limit here is 1 on purpose, and it is the whole strength of the
        case. A bound of 1 refuses *every* merge it is allowed to see, so a
        float32 chain coming through it at full width cannot be explained by
        the chain being short or by the bound never being reached -- the only
        thing that can explain it is the dtype guard. At limit 8 the same
        assertion would pass on a chain that simply never got wide enough.
        """
        with jt.flag_scope(use_cuda=1):
            build = _chain("float32")
            unbounded = max(_widths(build, fuse_op_limit=0))
            bounded = max(_widths(build, fuse_op_limit=1))
        self.assertGreater(unbounded, 8, "float32 chain did not fuse wide")
        self.assertEqual(
            bounded, unbounded,
            "the bound reached a float32 chain: %d -> %d" % (unbounded, bounded))

    def test_a_refused_sibling_edge_on_a_finished_input(self):
        """The refusal path must not ask a batch index of an off-batch var.

        `for_each_neighbor`'s sibling walk (`relation == 0`) is the one branch
        that does not filter on `var->tflag == tt`, and its `var` is an *input*
        of the op. `build_exec_plan` enqueues an input node only when it is
        unfinished, so a var that is already computed -- here, one that has
        been `sync()`ed before the batch that reads it -- is never stamped, and
        `Node::batch_index_at` asserts on an unstamped node rather than
        returning a stale index.

        So: one finished float16 var, two consumers of it (adjacent in its
        consumer table, which mirrors creation order), and a limit low enough
        that the sibling merge between those two consumers is certain to be
        refused. Before the `relation == 1` guard this raised out of the
        planner; the assertion is simply that it plans, and agrees.
        """
        for dtype in ("float16", "bfloat16"):
            with self.subTest(dtype=dtype), jt.flag_scope(use_cuda=1):
                def build():
                    shared = jt.array(
                        np.random.RandomState(1).randn(64, 256).astype("float32")
                    ).cast(dtype)
                    shared.sync()          # finished: outside the batch below
                    a, b = shared * 1.5, shared + 0.25
                    for _ in range(20):
                        a = a * 1.0001 + 0.001
                        b = b * 1.0001 + 0.001
                    return a + b

                jt.flags.fuse_op_limit = 0
                unbounded = build().float32().numpy()
                jt.flags.fuse_op_limit = 2      # refuse nearly every merge
                bounded = build().float32().numpy()
                np.testing.assert_array_equal(
                    unbounded, bounded,
                    "a refused sibling edge changed the result in %s" % dtype)

    def test_zero_is_unbounded(self):
        """0 is the documented escape hatch, and the behaviour before the flag."""
        with jt.flag_scope(use_cuda=1):
            build = _chain("float16")
            self.assertEqual(max(_widths(build, fuse_op_limit=0)),
                             max(_widths(build, fuse_op_limit=0)))
            self.assertGreater(max(_widths(build, fuse_op_limit=0)),
                               max(_widths(build, fuse_op_limit=8)))


if __name__ == "__main__":
    unittest.main()

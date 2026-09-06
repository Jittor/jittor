# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The jit key must identify the kernel it selects, and say so out loud.

Three properties, each of which used to fail in silence -- the key *is* the
cache lookup key for a compiled kernel, so anything it fails to distinguish
runs somebody else's compiled code and returns a wrong answer with nothing
printed.

1. Two fusions of different sizes get different keys.  The edge encoding used
   to be fixed width (``hex2 hex1 hex2 hex1``), so producer ids wrapped at 256
   and a 300-op fusion could land on a 200-op fusion's key.
2. A ``compile_options`` entry whose name starts with ``_`` reaches the key.
   The key used to skip those names, so two configurations differing only
   there shared one compiled product.
3. A key that will not fit raises a catchable error.  Overflowing the buffer
   used to reach an mprotect'ed guard page and kill the process from the
   signal handler, where by definition nothing can be caught.

The keys are read out of the executor's own log rather than recomputed here,
so what gets compared is what the cache is actually keyed on.

``auto_flush_ops=0`` everywhere below: on an accelerator the executor launches
whatever is pending once that many operators have been built, which would cut
these chains into segments of its own choosing and make "the 300-op fusion"
mean something different on CPU and on CUDA.

Run::  python -m pytest tests/compiler/test_jit_key_structure.py
"""

import unittest

import numpy as np

import jittor as jt

from _helpers.assertions import expect_error


def chain_sum(n):
    """Left-leaning chain: n external input vars and n-1 element-wise adds,
    which fuse into one segment."""
    xs = [jt.array(np.full((4,), i + 1, dtype="float32"), dtype="float32")
          for i in range(n)]
    total = xs[0]
    for x in xs[1:]:
        total = total + x
    return total


def tree_sum(n):
    """Balanced tree over the same n inputs: the same op count and the same
    var count as `chain_sum(n)`, a different edge structure."""
    xs = [jt.array(np.full((4,), i + 1, dtype="float32"), dtype="float32")
          for i in range(n)]
    level = xs
    while len(level) > 1:
        nxt = [level[i] + level[i + 1] for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def fused_keys(build):
    """The jit keys of every fused segment `build` executes.

    ``fused_op.cc`` logs the key on both the cache-hit and the cache-miss
    path, so this sees it whether the kernel was already compiled or not.
    """
    with jt.log_capture_scope(log_v=0, log_vprefix="fused_op=1000",
                              auto_flush_ops=0) as logs:
        build()
    keys = []
    for log in logs:
        message = log["msg"]
        for marker in ("Jit fused op key found:", "Jit op key not found:"):
            if marker in message:
                # The cache-hit line continues with the entry pointer, which is
                # not part of the key.
                key = message.split(marker, 1)[1]
                keys.append(key.split("jit op entry:", 1)[0].strip())
    return keys


class TestJitKeyIdentifiesTheKernel(unittest.TestCase):
    def test_a_300_op_fusion_does_not_share_a_key_with_a_200_op_one(self):
        keys_200 = fused_keys(lambda: chain_sum(200).sync())
        keys_300 = fused_keys(lambda: chain_sum(300).sync())
        self.assertTrue(keys_200, "no fused segment ran for n=200")
        self.assertTrue(keys_300, "no fused segment ran for n=300")
        # No key from either run may appear in the other: the two fusions
        # differ in op count, var count and edge structure alike.
        self.assertTrue(
            set(keys_200).isdisjoint(set(keys_300)),
            "a 200-op fusion and a 300-op fusion share a jit key, so the edge "
            "encoding is not injective")
        # And the arithmetic is right, which is what a collision would break.
        np.testing.assert_allclose(chain_sum(200).numpy(),
                                   np.full((4,), 200 * 201 // 2, "float32"))
        np.testing.assert_allclose(chain_sum(300).numpy(),
                                   np.full((4,), 300 * 301 // 2, "float32"))

    def test_two_edge_structures_of_the_same_size_get_different_keys(self):
        # The stronger form of the case above, and the one the fixed-width
        # encoding was actually lossy about: same op count, same var count,
        # different edges. Sizes differing in op count are told apart by the
        # per-op `«opkey<i>` entries whatever the edge encoding does, so a
        # 200-vs-300 comparison alone would not notice ids wrapping at 256.
        for n in (300, 512):
            chain = fused_keys(lambda: chain_sum(n).sync())
            tree = fused_keys(lambda: tree_sum(n).sync())
            self.assertTrue(chain, "no fused segment ran for chain n=%d" % n)
            self.assertTrue(tree, "no fused segment ran for tree n=%d" % n)
            self.assertTrue(
                set(chain).isdisjoint(set(tree)),
                "at n=%d a chain and a balanced tree over the same inputs "
                "share a jit key, so the second one runs the first one's "
                "compiled kernel" % n)
            np.testing.assert_allclose(
                tree_sum(n).numpy(), np.full((4,), n * (n + 1) // 2, "float32"))

    def test_underscore_compile_options_reach_the_key(self):
        def keys_for(value):
            def build():
                a = jt.array(np.arange(8, dtype="float32"), dtype="float32")
                b = a + a
                b.compile_options = {"_probe_3_02": value}
                (b + b).sync()
            return fused_keys(build)

        one, two = keys_for(1), keys_for(2)
        self.assertTrue(one)
        self.assertTrue(two)
        self.assertTrue(
            any("_probe_3_02" in key for key in one),
            "an underscore-prefixed compile_option is missing from the jit "
            "key: %r" % (one,))
        self.assertTrue(
            set(one).isdisjoint(set(two)),
            "two fusions differing only in an underscore-prefixed "
            "compile_option share a jit key")

    def test_an_over_long_key_raises_instead_of_killing_the_process(self):
        # A four-op fusion's key is a few hundred bytes and 300 chained adds
        # is tens of kilobytes, so lowering the limit to just above the former
        # costs a fraction of a second where building a 2 MB key would not.
        small = fused_keys(lambda: chain_sum(4).sync())
        self.assertTrue(small)
        headroom = max(len(key) for key in small) + 512
        with jt.flag_scope(jit_key_max_size=headroom, auto_flush_ops=0):
            error = expect_error(lambda: chain_sum(300).sync(),
                                 exc_type=RuntimeError,
                                 match="jit key too long")
        self.assertIn("jit_key_max_size", str(error))
        # The process is still usable afterwards, which is the whole point of
        # raising rather than faulting.
        np.testing.assert_allclose(chain_sum(4).numpy(),
                                   np.full((4,), 10, "float32"))


if __name__ == "__main__":
    unittest.main()

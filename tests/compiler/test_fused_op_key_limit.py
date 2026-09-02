# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Large fusions must get their own jit key, not another fusion's.

``FusedOp::do_jit_prepare`` encodes every fusion edge as four numbers: producer
op, producer output slot, consumer op, consumer input slot.  These used to be
written fixed width -- ``hex2 hex1 hex2 hex1``, 8 bits for the op ids and 4 for
the slots -- while the producer id runs up to ``ops.size() + var_index``
(``executor.cc``).  A fusion holding more than 255 ops plus external input vars
therefore wrapped: two structurally different fusions produced the same jit key,
the kernel cache lookup hit, and an unrelated compiled kernel ran, silently
returning a wrong result.

This is not a theoretical bound.  ``F.interpolate(mode="bicubic")`` fuses 292
ops over 296 vars and reaches edge id 462, so that path was already relying on
keys that could alias -- "it works today" only meant no colliding fusion
happened to be in the same cache.

The fields are now variable-length hex separated by ``.`` and ``,``, which are
outside the hex alphabet, so the encoding is injective for any width.

Run::  python -m pytest tests/compiler/test_fused_op_key_limit.py
"""

import unittest

import numpy as np

import jittor as jt


def chain_sum(n):
    """Left-leaning chain: n external input vars and n-1 element-wise adds."""
    xs = [jt.array(np.full((4,), i + 1, dtype="float32"), dtype="float32")
          for i in range(n)]
    total = xs[0]
    for x in xs[1:]:
        total = total + x
    return total


def tree_sum(n):
    """Balanced tree over the same n inputs: same op and var counts as
    chain_sum(n), a different edge structure, therefore a different key."""
    xs = [jt.array(np.full((4,), i + 1, dtype="float32"), dtype="float32")
          for i in range(n)]
    level = xs
    while len(level) > 1:
        nxt = [level[i] + level[i + 1] for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def expected(n):
    return np.full((4,), n * (n + 1) // 2, "float32")


class TestFusedOpKeyEncoding(unittest.TestCase):
    def test_chain_sum_is_exact_at_every_size(self):
        # 300 inputs + 299 adds pushes the producer id far past 255.
        for n in (2, 8, 64, 128, 200, 300, 400, 600):
            np.testing.assert_allclose(chain_sum(n).numpy(), expected(n),
                                       err_msg="chain n=%d" % n)

    def test_tree_sum_is_exact_at_every_size(self):
        for n in (2, 8, 64, 128, 200, 300, 400, 600):
            np.testing.assert_allclose(tree_sum(n).numpy(), expected(n),
                                       err_msg="tree n=%d" % n)

    def test_two_shapes_of_the_same_size_do_not_share_a_kernel(self):
        # Same op count, same var count, different edges: if the two keys
        # collided, the second would run the first's compiled kernel.
        for n in (300, 512):
            chain = chain_sum(n).numpy()
            tree = tree_sum(n).numpy()
            np.testing.assert_allclose(chain, expected(n), err_msg="chain %d" % n)
            np.testing.assert_allclose(tree, expected(n), err_msg="tree %d" % n)

    def test_large_fusion_matches_an_unfused_reference(self):
        # Cross-check against the same arithmetic with fusion disabled, so the
        # reference does not come from the same code path under test.
        n = 300
        fused = chain_sum(n).numpy()
        with jt.flag_scope(no_fuse=1):
            reference = chain_sum(n).numpy()
        np.testing.assert_allclose(fused, reference)
        np.testing.assert_allclose(fused, expected(n))


if __name__ == "__main__":
    unittest.main()

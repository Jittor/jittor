# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The kernel caches are bounded, and evicting from them stays correct.

The three tables (``jit_ops``, ``jit_key_mapper``, ``jit_fused_ops``) had no
``erase`` and no capacity: a workload whose shapes keep changing produced a new
key per shape and grew them -- plus, for fused kernels, one never-freed
``FusedOpContext`` per entry -- for the life of the process.  They are bounded
now, with least-recently-used eviction (``utils/jit_cache_map.h``), and
``jit_cache_size`` sets the bound.

Bounding them is what makes the *rest* of 3.03 necessary, so this exercises the
eviction path rather than the table sizes, which Python cannot see:

* ``t[a] = t[b] = v``, which every write site used to be, reads a reference the
  other subscript may already have erased.  Both keys of every product are now
  assigned in separate statements.
* one ``FusedOpContext`` is reachable from two keys, so dropping either of them
  must not free it and dropping both must -- it is a ``shared_ptr`` in the
  table and the executing ``FusedOp`` holds a reference of its own.
* a cached context must not point at the ``FusedOp`` it was compiled from.  It
  used to, and what hid that was the executor overwriting the pointer on every
  cache hit.

A capacity of a handful with dozens of distinct shapes makes every lookup after
the first few a miss, so this walks that path hundreds of times in a second.

Run::  python -m pytest tests/codegen/test_jit_cache_bound.py
"""

import unittest

import numpy as np

import jittor as jt


def elementwise(n):
    """One fused segment whose jit key depends on the shape."""
    a = jt.array(np.arange(n, dtype="float32"), dtype="float32")
    b = jt.array(np.arange(n, dtype="float32") * 2, dtype="float32")
    return (a * b + a - b).numpy()


def expected(n):
    a = np.arange(n, dtype="float32")
    b = a * 2
    return a * b + a - b


class TestJitCacheIsBounded(unittest.TestCase):
    def test_the_bound_is_a_flag_with_a_sane_default(self):
        self.assertGreater(jt.introspection.policy.runtime.jit_cache_size, 0)

    def test_a_changing_shape_workload_stays_correct_with_a_tiny_cache(self):
        shapes = list(range(3, 40))
        with jt.flag_scope(jit_cache_size=4):
            # Three passes: the first fills and evicts, the later ones come
            # back to keys that have been evicted and re-inserted.
            for _ in range(3):
                for n in shapes:
                    np.testing.assert_allclose(elementwise(n), expected(n),
                                               err_msg="n=%d" % n)
        # And the same answers once the cache is back to its normal size.
        for n in shapes[:5]:
            np.testing.assert_allclose(elementwise(n), expected(n))

    def test_eviction_is_correct_under_the_parallel_compiler_too(self):
        # The serial path (fused_op.cc) and the parallel path
        # (parallel_compiler.cc) fill the tables from different code, and both
        # used to do it with `t[a] = t[b] = v`.
        with jt.flag_scope(jit_cache_size=4, use_parallel_op_compiler=16):
            for n in range(41, 60):
                np.testing.assert_allclose(elementwise(n), expected(n),
                                           err_msg="n=%d" % n)

    def test_a_relayed_fusion_survives_eviction(self):
        # matmul reaches the relay machinery, which is what holds the
        # `FusedOpContext` the generated kernel reads through `context->vrm`.
        with jt.flag_scope(jit_cache_size=4):
            for n in (8, 12, 16, 20, 8, 12):
                x = jt.array(np.arange(n * n, dtype="float32").reshape(n, n),
                             dtype="float32")
                y = jt.array(np.eye(n, dtype="float32"), dtype="float32")
                np.testing.assert_allclose(jt.matmul(x, y).numpy(),
                                           x.numpy(), atol=1e-3,
                                           err_msg="n=%d" % n)


if __name__ == "__main__":
    unittest.main()

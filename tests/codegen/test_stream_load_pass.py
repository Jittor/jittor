# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""StreamLoadPass: a CUDA kernel loads an input it reads for the last time
evict-first (`stream_dying_inputs`).

The pass rewrites every load of a fused input into `jt_stream_ld`, which picks
the plain load or `ld.global.cs` by one bit of an argument the executor fills
per run. Two things can go wrong silently: the rewrite can change what a load
reads -- a vector load of a vectorised loop, an element of a type whose width
is not its alignment -- and the bit can be set on an input something still
reads. The first is checked by value in every element type, through the
scalar, vectorised and reduction kernels, with the streaming on and off; the
second by a held input reading back what it was.
"""

from _helpers.runtime_policy import preserve_policy as _test_preserve_policy
from _helpers import capability as _test_capability

import unittest

import numpy as np

import jittor as jt

N = 1 << 20


def _inputs(rng, dtype, shape):
    if dtype == "bool":
        return rng.random(shape) > 0.5, rng.random(shape) > 0.5
    if dtype in ("int8", "uint8", "int32", "int64"):
        return (rng.integers(0, 20, shape).astype(dtype),
                rng.integers(0, 20, shape).astype(dtype))
    return (rng.standard_normal(shape).astype("float32"),
            rng.standard_normal(shape).astype("float32"))


def _run(dtype, shape, stream):
    """`x * 2 + y` and its column sums, with both inputs dying in the kernel."""
    rng = np.random.default_rng(len(shape) + shape[-1])
    a, b = _inputs(rng, dtype, shape)
    with jt.flag_scope(stream_dying_inputs=stream):
        x, y = jt.array(a).cast(dtype), jt.array(b).cast(dtype)
        x.sync(), y.sync()
        if dtype == "bool":
            z = x ^ y
            del x, y
            return [z.numpy()], [a ^ b]
        want_x, want_y = x.float64().numpy(), y.float64().numpy()
        z = x * 2 + y
        s = z.float32().sum(0) if len(shape) == 2 else None
        del x, y
        got = [z.float64().numpy()] + ([s.numpy()] if s is not None else [])
        want = (want_x * 2 + want_y)
        return got, [want] + ([want.sum(0)] if s is not None else [])


@_test_preserve_policy(jt, "use_cuda")
@unittest.skipIf(not _test_capability.check_accelerator("cuda", backend=jt).enabled,
                 "no usable CUDA in this build")
class TestStreamLoadPass(unittest.TestCase):
    def setUp(self):
        jt.flags.use_cuda = 1

    def test_the_loads_of_a_fused_input_are_rewritten(self):
        x = jt.array(np.arange(N, dtype="float32"))
        x.sync()
        with jt.profile_scope() as rep:
            z = x * 3 + 1
            z.sync()
        sources = [open(row[1]).read() for row in rep[1:] if row[1].endswith(".cc")]
        self.assertTrue(any("jt_stream_ld(" in src and "streamed_inputs" in src
                            for src in sources), "no kernel loads through jt_stream_ld")

    def test_streamed_and_plain_loads_answer_the_same_in_every_type(self):
        types = ("float32", "float16", "bfloat16", "float64", "int8", "uint8",
                 "int32", "int64", "bool")
        # flat and vectorised, flat with a scalar tail, and a column reduction
        shapes = ((N,), (N + 3,), (1024, 1027))
        for dtype in types:
            for shape in shapes:
                with self.subTest(dtype=dtype, shape=shape):
                    streamed, want = _run(dtype, shape, 1 << 20)
                    plain, _ = _run(dtype, shape, 0)
                    # The elementwise result is the same bits; the column
                    # sums are atomics, whose order no run fixes.
                    np.testing.assert_array_equal(streamed[0], plain[0])
                    for s, p in zip(streamed[1:], plain[1:]):
                        np.testing.assert_allclose(s, p, rtol=1e-5, atol=1e-3)
                    tol = 2e-2 if dtype in ("float16", "bfloat16") else 1e-6
                    if dtype in ("int8", "uint8", "bool"):
                        # Narrow integers wrap; the comparison is with the
                        # unstreamed kernel above.
                        continue
                    for g, w in zip(streamed, want):
                        np.testing.assert_allclose(g, w, rtol=tol, atol=tol * max(1, np.abs(w).max()))

    def test_a_held_input_still_reads_back(self):
        a = np.random.default_rng(3).standard_normal(N).astype("float32")
        x = jt.array(a)
        x.sync()
        z = x * 2 + 1
        z.sync()
        np.testing.assert_array_equal(x.numpy(), a)
        np.testing.assert_allclose(z.numpy(), a * 2 + 1, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()

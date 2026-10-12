# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Dun Liang <randonlang@gmail.com>.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""BlockedReductionPass: max/min get the blocked shape, with their own combiner.

A blocked reduction reassociates, so two things can go wrong *silently*: which
operation folds the partial results, and what a partial starts from.  A `+` fold
on the partials of a `max` compiles, runs, and computes a sum; a partial seeded
with the numeric zero is only the identity for `+`, so an all-negative `max`
answers `0`.  Both are asserted below -- the combiner by reading the generated
kernel, the identity by value on data where a wrong one is visible.

The pass itself is CPU-only, so there is no CUDA variant of these cases.
"""

from _helpers.runtime_policy import preserve_policy as _test_preserve_policy

import unittest

import numpy as np

import jittor as jt

#: Above the pass's cutoff (``LEAST`` = 16), so the blocked branch is taken.
N = 1 << 17
#: Below it, so the straight-accumulation branch is taken.
SMALL = 8


def kernel_source(op, n=N):
    """Return the generated source of one reduce kernel and its result.

    The scope pins the CPU on purpose: this pass returns immediately for an
    accelerator op, so the CPU path is the only one these cases can say
    anything about.
    """
    with jt.flag_scope(use_cuda=0):
        a = jt.ones([n])
        a.sync()
        with jt.profile_scope() as rep:
            out = jt.reduce(a, op, (0,))
            out.sync()
        assert len(rep) == 2, rep
        with open(rep[1][1]) as f:
            return f.read(), out.numpy()


def fold_lines(src):
    """The lines that assign to a partial, i.e. everything the combiner folds.

    The unrolled body copies assign to a partial too, so they are included --
    they are written by ``@expand_op`` and are not the pass's combiner.
    """
    return [ln.strip() for ln in src.splitlines()
            if "_a0_" in ln and " = " in ln]


@_test_preserve_policy(jt, "use_cuda")
class TestBlockedReductionPass(unittest.TestCase):
    def test_max_is_blocked_and_folds_with_max(self):
        src, out = kernel_source("maximum")

        self.assertIn("jt_blk_", src, "maximum was not given the blocked shape")
        # The pass's own folds, which are the ones that could be a `+`.
        self.assertIn("jittor::_max<decltype(", src)
        for line in fold_lines(src):
            self.assertNotIn("+", line,
                             "a partial of a max is folded with '+': %s" % line)
        # A partial must start from the identity the accumulator holds, not
        # from a numeric zero: `max(0, -3.5)` is 0.
        self.assertRegex(src, r"jt_blk_\w*a0_0 = jt_reduce_acc_\w+;")
        self.assertEqual(float(out), 1.0)

    def test_min_is_blocked_and_folds_with_min(self):
        src, out = kernel_source("minimum")

        self.assertIn("jt_blk_", src, "minimum was not given the blocked shape")
        self.assertIn("jittor::_min<decltype(", src)
        for line in fold_lines(src):
            self.assertNotIn("+", line,
                             "a partial of a min is folded with '+': %s" % line)
        self.assertRegex(src, r"jt_blk_\w*a0_0 = jt_reduce_acc_\w+;")
        self.assertEqual(float(out), 1.0)

    def test_sum_keeps_its_plus(self):
        """The control: `sum` folded with a `+` before this and still does."""
        src, out = kernel_source("add")
        self.assertIn("jt_blk_", src)
        self.assertRegex(src, r"jt_blk_\w*a0_0 = \(jt_blk_\w*stack0\["
                               r"jt_blk_\w*top\]\) \+ \(jt_blk_\w*a0_0\);")
        self.assertEqual(float(out), float(N))

    def test_the_identity_is_visible_in_the_value(self):
        """All-negative `max` and all-positive `min`, above and below the
        blocked cutoff. A partial seeded with zero answers 0 for both."""
        for n in (N, SMALL):
            with self.subTest(n=n), jt.flag_scope(use_cuda=0):
                negative = np.arange(n, dtype=np.float32) - n      # max is -1
                positive = np.arange(n, dtype=np.float32) + 1      # min is 1
                a = jt.array(negative)
                self.assertEqual(float(jt.reduce(a, "maximum", (0,)).numpy()), -1.0)
                b = jt.array(positive)
                self.assertEqual(float(jt.reduce(b, "minimum", (0,)).numpy()), 1.0)

    def test_blocked_and_unblocked_max_agree_exactly(self):
        """Both shapes must agree bit for bit: max is exactly associative, so
        the blocked form is only a different order of the same comparisons."""
        for name, data in (
            ("all-negative", np.arange(N, dtype=np.float32) - N),
            ("mixed", np.random.default_rng(1).standard_normal(N).astype(np.float32)),
        ):
            with self.subTest(data=name), jt.flag_scope(use_cuda=0):
                a = jt.array(data)
                with jt.flag_scope(exclude_pass="blocked_reduction",
                                   compile_options={"_blocked_off": 1}):
                    off = jt.reduce(a, "maximum", (0,)).numpy()
                on = jt.reduce(a, "maximum", (0,)).numpy()
                self.assertEqual(float(off), float(on))
                self.assertEqual(float(on), float(data.max()))

    def test_nan_survives_either_fold(self):
        data = np.arange(N, dtype=np.float32)
        data[N // 2] = np.nan
        with jt.flag_scope(use_cuda=0):
            a = jt.array(data)
            self.assertTrue(np.isnan(jt.reduce(a, "maximum", (0,)).numpy()))
            self.assertTrue(np.isnan(jt.reduce(a, "minimum", (0,)).numpy()))


if __name__ == "__main__":
    unittest.main()

# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""cuDNN RNN dropout must advance between calls, and rewind on set_seed.

Also covers the reserve-space query that shape inference needs: it used to run
on every inference (i.e. every training step), creating ``seq_length`` tensor
descriptors it never destroyed.

The dropout state is cuDNN's RNG state, and cuDNN advances it in place.  It
used to be a member of a descriptor built inside ``jit_run``, so every call
constructed a fresh state and seeded it from the global seed -- every training
step drew the *identical* mask sequence.  Nothing about the outputs looks
wrong: the loss still falls, on a model regularized far less than it asked for.

Two things make a test of this real:

* ``num_layers >= 2``.  cuDNN applies RNN dropout *between* layers, so with a
  single layer the flag does nothing and "two calls agree" is true for reasons
  that have nothing to do with the bug.
* the ``dropout=0`` control.  Without it, "two calls disagree" would also pass
  for an implementation that returned noise.
"""
import unittest

import numpy as np

import jittor as jt


def _lstm(dropout):
    jt.set_seed(1234)
    # 3 layers: dropout is applied between layers 0/1 and 1/2.
    rnn = jt.nn.LSTM(8, 8, num_layers=3, dropout=dropout)
    rnn.train()
    return rnn


@unittest.skipIf(not jt.has_cuda, "No CUDA found")
class TestCudnnRnnDropout(unittest.TestCase):
    def setUp(self):
        self.flags = jt.flag_scope(use_cuda=1)
        self.flags.__enter__()
        jt.set_seed(0)
        self.x = jt.randn(5, 4, 8)

    def tearDown(self):
        self.flags.__exit__(None, None, None)

    def test_zero_dropout_repeats(self):
        """Control: with dropout off, the same input gives the same output."""
        rnn = _lstm(0.0)
        a = rnn(self.x)[0].numpy()
        b = rnn(self.x)[0].numpy()
        np.testing.assert_allclose(a, b, rtol=0, atol=0)

    def test_dropout_mask_advances_between_calls(self):
        rnn = _lstm(0.5)
        outs = [rnn(self.x)[0].numpy() for _ in range(3)]
        # Before the fix all three were bit-identical.
        for i, j in ((0, 1), (1, 2), (0, 2)):
            self.assertFalse(
                np.allclose(outs[i], outs[j]),
                "call %d and %d drew the same dropout mask" % (i, j))

    def test_set_seed_rewinds_the_mask_sequence(self):
        """Reuse across calls must not cost reproducibility.

        The states are cached, so seeding has to drop them; otherwise the
        second pass continues an advanced state and jt.set_seed() stops
        meaning anything for a dropout RNN.
        """
        rnn = _lstm(0.5)

        jt.set_seed(7)
        first = [rnn(self.x)[0].numpy() for _ in range(3)]
        jt.set_seed(7)
        again = [rnn(self.x)[0].numpy() for _ in range(3)]

        for k, (a, b) in enumerate(zip(first, again)):
            np.testing.assert_allclose(a, b, rtol=0, atol=0,
                err_msg="call %d differed after re-seeding" % k)
        # ... and the run being reproducible is not the same as it being
        # constant: within a pass the mask still advances.
        self.assertFalse(np.allclose(first[0], first[1]))


@unittest.skipIf(not jt.has_cuda, "No CUDA found")
class TestCudnnRnnReserveSpace(unittest.TestCase):
    def test_reserve_space_is_queried_once_per_configuration(self):
        """Shape inference must not re-ask cuDNN on every step.

        The query is the thing that used to leak seq_length descriptors per
        call, so counting the queries counts the leaks. The op logs one line
        per cache miss; several forward passes of one shape must produce
        exactly one.
        """
        from _helpers.logs import find_log_with_re

        with jt.flag_scope(use_cuda=1):
            jt.set_seed(0)
            rnn = jt.nn.LSTM(8, 8, num_layers=2)
            rnn.train()
            x = jt.randn(5, 4, 8)
            rnn(x)[0].sync()          # warm up: first pass takes the miss

            with jt.log_capture_scope(
                    log_silent=1, log_v=0,
                    log_vprefix="cudnn_rnn_descriptor=100") as raw_log:
                for _ in range(5):
                    rnn(x)[0].sync()
            repeats = find_log_with_re(raw_log, r"reserve space query")
            self.assertEqual(repeats, [], "cuDNN was re-queried on a warm shape")

            # ... and the counter is not stuck at zero: a new shape does query.
            with jt.log_capture_scope(
                    log_silent=1, log_v=0,
                    log_vprefix="cudnn_rnn_descriptor=100") as raw_log:
                rnn(jt.randn(6, 4, 8))[0].sync()
            self.assertEqual(
                len(find_log_with_re(raw_log, r"reserve space query")), 1)


if __name__ == "__main__":
    unittest.main()

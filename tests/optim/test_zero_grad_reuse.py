# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``zero_grad`` reuses one zeros Var per gradient instead of building one
每 step.

`zeros_like` carries no storage, so a fresh one per gradient per step costs
host time rather than memory -- 96 of them a step on an 8-layer transformer,
1.03 ms of an 8.49 ms training step. Reuse is only safe while nothing writes
a gradient buffer in place, so what is pinned here is the observable
contract: the gradients read as zero after every step, and a later step's
gradients never leak into the zeros a reader sees.
"""

import unittest

import numpy as np

import jittor as jt
from jittor import nn


class TestZeroGradReuse(unittest.TestCase):

    def setUp(self):
        jt.set_global_seed(0)
        self.model = nn.Linear(4, 3)
        self.opt = nn.SGD(self.model.parameters(), lr=0.1)
        self.x = jt.array(np.arange(8, dtype="float32").reshape(2, 4))

    def _grads(self):
        return [g.numpy().copy() for pg in self.opt.param_groups for g in pg["grads"]]

    def test_gradients_read_as_zero_after_every_step(self):
        for step in range(4):
            self.opt.step((self.model(self.x) ** 2).mean())
            for g in self._grads():
                np.testing.assert_array_equal(g, np.zeros_like(g),
                                              err_msg=f"after step {step}")

    def test_a_step_still_sees_its_own_gradient(self):
        # The reused zero must not be what the update reads: the parameters
        # have to keep moving.
        before = [p.numpy().copy() for p in self.model.parameters()]
        seen = []
        for _ in range(3):
            self.opt.step((self.model(self.x) ** 2).mean())
            seen.append([p.numpy().copy() for p in self.model.parameters()])
        for i, now in enumerate(seen):
            for a, b in zip(now, before):
                self.assertFalse(np.array_equal(a, b), f"parameters froze at step {i}")
            before = now

    def test_the_trajectory_matches_sgd_done_by_hand(self):
        model = nn.Linear(4, 3)
        opt = nn.SGD(model.parameters(), lr=0.1)
        hand = [p.numpy().copy() for p in model.parameters()]
        for _ in range(3):
            loss = (model(self.x) ** 2).mean()
            grads = [g.numpy().copy() for g in jt.grad(loss, list(model.parameters()))]
            opt.step(loss)
            hand = [h - 0.1 * g for h, g in zip(hand, grads)]
            for got, want in zip(model.parameters(), hand):
                np.testing.assert_allclose(got.numpy(), want, rtol=1e-4, atol=1e-5)

    def test_two_optimizers_do_not_share_the_cache(self):
        # The cache is per instance; a class attribute would be shared by every
        # optimizer in the process and would never be released.
        other = nn.Linear(4, 3)
        opt2 = nn.SGD(other.parameters(), lr=0.1)
        self.opt.step((self.model(self.x) ** 2).mean())
        opt2.step((other(self.x) ** 2).mean())
        a = self.opt.__dict__.get("_zero_grad_cache")
        b = opt2.__dict__.get("_zero_grad_cache")
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertIsNot(a, b)


if __name__ == "__main__":
    unittest.main()

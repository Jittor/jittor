# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``Module.train()`` / ``Module.eval()``: what they are allowed to change.

Task 5.05. torch's ``eval()`` flips one flag; BatchNorm switches to its running
statistics and Dropout becomes a no-op. jittor's also called ``stop_grad()`` on
every parameter and remembered, in a dict keyed by ``id(p)``, which ones
``train()`` should later ``start_grad()``. Every consequence below was silent:

* ``model.eval()`` then differentiating gave no gradient at all -- the pattern
  every "evaluate, then backprop" script uses (adversarial examples, Grad-CAM,
  meta-learning inner loops).
* ``child.eval()`` then ``parent.train()`` left the child frozen forever: the
  backup dict lives on whichever module ``eval()`` was called on, and the
  parent has none.
* an eval/train round trip UNFROZE a parameter the caller had deliberately
  frozen with ``requires_grad = False``.

Run::  python -m pytest tests/core/test_module_train_eval.py
"""

import unittest

import numpy as np

import jittor as jt
from jittor import nn


class Child(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(4, 3)

    def execute(self, x):
        return self.linear(x)


class Parent(nn.Module):
    def __init__(self):
        self.child = Child()
        self.head = nn.Linear(3, 2)

    def execute(self, x):
        return self.head(self.child(x))


def _loss(model, x):
    return model(x).sum()


class TestEvalDoesNotFreeze(unittest.TestCase):
    def setUp(self):
        self.x = jt.array(np.ones((5, 4), dtype="float32"))

    def test_a_gradient_still_flows_after_eval(self):
        model = Parent()
        model.eval()
        params = list(model.parameters())
        grads = jt.grad(_loss(model, self.x), params)
        self.assertEqual(len(grads), len(params))
        self.assertTrue(
            any(np.abs(g.numpy()).sum() > 0 for g in grads),
            "eval() must not stop the gradient; torch's eval only affects "
            "BatchNorm and Dropout")

    def test_eval_leaves_requires_grad_alone(self):
        model = Parent()
        before = [p.requires_grad for p in model.parameters()]
        model.eval()
        self.assertEqual([p.requires_grad for p in model.parameters()], before)

    def test_a_child_eval_then_a_parent_train_leaves_it_trainable(self):
        # The acceptance criterion for this task: the backup dict used to live
        # on the module eval() was called on, so the parent's train() could not
        # find it and the child stayed frozen for the rest of the process.
        model = Parent()
        model.child.eval()
        model.train()
        params = list(model.child.parameters())
        self.assertTrue(all(p.requires_grad for p in params))
        grads = jt.grad(_loss(model, self.x), params)
        self.assertTrue(any(np.abs(g.numpy()).sum() > 0 for g in grads),
                        "a sub-module frozen by eval() never came back")

    def test_a_train_does_not_unfreeze_a_deliberately_frozen_parameter(self):
        model = Parent()
        frozen = model.head.weight
        frozen.requires_grad = False
        model.eval()
        model.train()
        self.assertFalse(frozen.requires_grad,
                         "train() must not undo requires_grad_(False)")


class TestTrainEvalToggleTheFlag(unittest.TestCase):
    def test_the_flag_reaches_every_sub_module(self):
        model = Parent()
        model.eval()
        for module in (model, model.child, model.child.linear, model.head):
            self.assertFalse(module.is_train, type(module).__name__)
        model.train()
        for module in (model, model.child, model.child.linear, model.head):
            self.assertTrue(module.is_train, type(module).__name__)

    def test_training_property_follows(self):
        model = Parent()
        model.eval()
        self.assertFalse(model.training)
        self.assertFalse(model.is_training())
        model.train()
        self.assertTrue(model.training)
        self.assertTrue(model.is_training())

    def test_both_return_self_for_chaining(self):
        model = Parent()
        self.assertIs(model.eval(), model)
        self.assertIs(model.train(), model)


class TestTheFlagStillDoesItsJob(unittest.TestCase):
    """eval() must keep doing the ONE thing torch's eval() does."""

    def test_dropout_is_a_noop_in_eval_and_drops_in_train(self):
        # NB: inverted dropout scales the survivors by 1/(1-p), so the MEAN is
        # preserved and says nothing. Count the zeros instead.
        drop = nn.Dropout(p=0.9)
        x = jt.array(np.ones((64, 64), dtype="float32"))
        drop.eval()
        out = drop(x).numpy()
        np.testing.assert_allclose(out, x.numpy())
        self.assertEqual((out == 0).mean(), 0.0)
        drop.train()
        out = drop(x).numpy()
        self.assertGreater((out == 0).mean(), 0.5,
                           "train() must put dropout back")

    def test_batchnorm_uses_running_stats_in_eval(self):
        bn = nn.BatchNorm(3)
        rng = np.random.RandomState(0)
        bn.train()
        for _ in range(5):
            bn(jt.array((rng.randn(16, 3) * 4 + 7).astype("float32")))
        bn.eval()
        probe = jt.array(np.full((16, 3), 7.0, dtype="float32"))
        first = bn(probe).numpy()
        second = bn(probe).numpy()
        # in eval the running stats are frozen, so the same input twice gives
        # the same answer and the batch's own mean is not used
        np.testing.assert_allclose(first, second)
        self.assertGreater(np.abs(first).max(), 1e-6,
                           "eval() normalized by the batch's own mean")


if __name__ == "__main__":
    unittest.main()

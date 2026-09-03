# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""One definition of "what is a parameter", and five views over it.

``parameters``, ``named_parameters``, ``state_dict``, ``named_buffers`` and
``_buffers`` were five transcriptions of the same depth-first walk, and they had
drifted:

* ``parameters`` de-duplicated by ``id`` while ``named_parameters``
  de-duplicated by NAME -- which does not de-duplicate a tied weight at all,
  because its two names differ. So a model whose two layers share a weight told
  the optimizer one parameter count and transformers/peft another.
* ``state_dict`` de-duplicated by ``id``, so only one of a tied weight's names
  reached the checkpoint, and which one depended on ``__dict__`` order.
* ``_parameters`` and ``_buffers`` both returned every Var the module owned, so
  neither of them actually said anything.
* ``parameters`` and ``state_dict`` were queries that RENAMED the Vars they
  returned, guarded by a string-length test, so a parameter's name -- and every
  key of a checkpoint -- depended on which level of the tree someone had called
  ``parameters()`` from first.

The reference for all of it is torch, whose answers are quoted in each test.
"""

import unittest

import numpy as np

import jittor as jt


class _Tied(jt.Module):
    """``b`` shares ``a``'s weight, the tied-embedding shape of every LM."""

    def __init__(self):
        self.a = jt.nn.Linear(2, 2, bias=False)
        self.b = jt.nn.Linear(2, 2, bias=False)
        self.b.weight = self.a.weight
        self.c = jt.nn.Linear(2, 2, bias=False)

    def execute(self, x):
        return self.c(self.b(self.a(x)))


class _Inner(jt.Module):
    def __init__(self):
        self.w = jt.ones((2,))

    def execute(self, x):
        return x * self.w


class _Outer(jt.Module):
    def __init__(self):
        self.inner = _Inner()

    def execute(self, x):
        return self.inner(x)


class TestTiedWeights(unittest.TestCase):

    def test_parameters_and_named_parameters_agree(self):
        m = _Tied()
        ps = m.parameters()
        nps = m.named_parameters()
        self.assertEqual(len(ps), len(nps))
        self.assertEqual([id(p) for p in ps], [id(p) for _, p in nps])
        # torch: ['a.weight', 'c.weight'] -- the tied Var under its first name.
        self.assertEqual([n for n, _ in nps], ["a.weight", "c.weight"])

    def test_the_optimizer_and_the_checkpoint_see_the_same_vars(self):
        """The set a training step updates is the set a checkpoint carries.

        Not the same *counts*: state_dict has a key per name, so a tied weight
        has two of them. The Vars behind them have to be one set.
        """
        m = _Tied()
        opt = jt.optim.SGD(m.parameters(), lr=0.1)
        optimized = {id(p) for pg in opt.param_groups for p in pg["params"]}
        checkpointed = {id(v) for v in m.state_dict().values()}
        from_named = {id(p) for _, p in m.named_parameters()}
        self.assertEqual(optimized, checkpointed)
        self.assertEqual(optimized, from_named)

    def test_an_aggregate_over_named_parameters_does_not_double_count(self):
        """``sum(p.numel() for _, p in model.named_parameters())``.

        That is how transformers reports a model's size and how a trainer builds
        its parameter list. With name-keyed de-duplication the tied weight was
        counted once per name: 8 for a model holding 4 numbers.
        """
        m = _Tied()
        self.assertEqual(sum(p.numel() for _, p in m.named_parameters()), 8)

    def test_state_dict_keeps_every_name_of_a_tied_weight(self):
        """torch's keys here are a.weight, b.weight, c.weight."""
        m = _Tied()
        keys = list(m.state_dict().keys())
        self.assertEqual(sorted(keys), ["a.weight", "b.weight", "c.weight"])
        sd = m.state_dict()
        self.assertIs(sd["a.weight"], sd["b.weight"])

    def test_a_reloaded_tied_checkpoint_still_ties(self):
        m = _Tied()
        m.a.weight.assign(jt.ones((2, 2)) * 3)
        state = {k: v.numpy() for k, v in m.state_dict().items()}
        m2 = _Tied()
        m2.load_state_dict(state)
        np.testing.assert_allclose(m2.a.weight.numpy(), 3.0)
        self.assertIs(m2.a.weight, m2.b.weight)


class TestQueriesDoNotRename(unittest.TestCase):
    """A query must not mutate the model it is asked about."""

    def _names(self, model):
        return [v.name() for v in [model.inner.w]]

    def test_parameters_leaves_the_var_name_alone(self):
        a = _Outer()
        before = a.inner.w.name()
        a.parameters()
        a.inner.parameters()
        a.state_dict()
        self.assertEqual(a.inner.w.name(), before)

    def test_the_name_does_not_depend_on_where_the_query_started(self):
        """It used to: the write-back fired only when the new path was LONGER.

        Calling ``outer.parameters()`` first stamped ``inner.w`` with
        ``"inner.w"``; calling ``outer.inner.parameters()`` first stamped it with
        ``"w"``, and the later, longer path then overwrote it while the shorter
        one could not. So the name -- and every checkpoint key derived from it --
        was a function of call order.
        """
        root_first = _Outer()
        root_first.parameters()
        root_first.inner.parameters()

        leaf_first = _Outer()
        leaf_first.inner.parameters()
        leaf_first.parameters()

        self.assertEqual(root_first.inner.w.name(), leaf_first.inner.w.name())

    def test_names_come_from_the_traversal_path(self):
        m = _Outer()
        self.assertEqual([n for n, _ in m.named_parameters()], ["inner.w"])
        self.assertEqual([n for n, _ in m.inner.named_parameters()], ["w"])


class TestBatchNormBuffers(unittest.TestCase):
    """jittor's own BatchNorm went around ``register_buffer``.

    It tagged the Var (``object.__setattr__(buf, "is_buffer", True)``) instead of
    recording the NAME on the module -- and the tag does not survive the Var being
    replaced, which is what a checkpoint load, a dtype cast or a hand-written
    reset does. That is precisely the failure mode the name set exists to prevent.
    """

    def test_the_buffers_are_registered_by_name(self):
        bn = jt.nn.BatchNorm(4)
        self.assertEqual(
            sorted(bn.__dict__.get("_buffer_names", ())),
            ["num_batches_tracked", "running_mean", "running_var"])

    def test_reassigning_running_mean_does_not_make_it_trainable(self):
        bn = jt.nn.BatchNorm(4)
        bn.running_mean = jt.zeros(4)
        names = [n for n, _ in bn.named_parameters()]
        self.assertEqual(names, ["weight", "bias"])
        self.assertNotIn(id(bn.running_mean), {id(p) for p in bn.parameters()})
        self.assertIn("running_mean", [n for n, _ in bn.named_buffers()])

    def test_a_reassigned_buffer_stays_out_of_the_optimizer(self):
        """The reachable harm: weight decay on the running statistics."""
        bn = jt.nn.BatchNorm(4)
        bn.running_mean = jt.ones(4)
        opt = jt.optim.SGD(bn.parameters(), lr=0.1, weight_decay=0.5)
        touched = {id(p) for pg in opt.param_groups for p in pg["params"]}
        self.assertNotIn(id(bn.running_mean), touched)

    def test_buffers_and_parameters_partition_the_vars(self):
        bn = jt.nn.BatchNorm(4)
        params = set(bn._parameters.keys())
        buffers = set(bn._buffers.keys())
        self.assertEqual(params, {"weight", "bias"})
        self.assertEqual(buffers,
                         {"running_mean", "running_var", "num_batches_tracked"})
        self.assertEqual(params & buffers, set())

    def test_non_persistent_buffers_are_buffers_but_not_state(self):
        bn = jt.nn.BatchNorm(4)
        self.assertIn("num_batches_tracked", [n for n, _ in bn.named_buffers()])
        self.assertNotIn("num_batches_tracked", bn.state_dict())


class TestViewsAgree(unittest.TestCase):
    """Every view is the same traversal filtered differently."""

    class _Mixed(jt.Module):
        def __init__(self):
            self.weight = jt.ones((2, 2))
            self.register_buffer("kept", jt.zeros(2))
            self.register_buffer("temp", jt.zeros(2), persistent=False)
            self.child = jt.nn.BatchNorm(3)

        def execute(self, x):
            return x

    def test_state_is_parameters_plus_persistent_buffers(self):
        m = self._Mixed()
        state = set(m.state_dict().keys())
        params = {n for n, _ in m.named_parameters()}
        buffers = {n for n, _ in m.named_buffers()}
        self.assertEqual(state - params, {"kept", "child.running_mean",
                                          "child.running_var"})
        self.assertTrue(state - params <= buffers)
        self.assertEqual(params & buffers, set())

    def test_a_plain_var_attribute_is_neither(self):
        m = self._Mixed()
        every = {n for n, _ in m._named_vars("parameters")} | \
                {n for n, _ in m._named_vars("buffers")}
        self.assertEqual(
            every,
            {"weight", "kept", "temp",
             "child.weight", "child.bias", "child.running_mean",
             "child.running_var", "child.num_batches_tracked"})

    def test_the_private_traversal_backs_all_of_them(self):
        m = self._Mixed()
        self.assertEqual(m.named_parameters(), m._named_vars("parameters"))
        self.assertEqual(m.named_buffers(), m._named_vars("buffers"))
        self.assertEqual([v for _, v in m._named_vars("parameters")],
                         m.parameters())


if __name__ == "__main__":
    unittest.main()

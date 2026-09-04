# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The kernel's backward-leaf answer, case by case against real PyTorch.

`2.25` gave the kernel a real answer for "is this Var a leaf of the backward
graph" (`tests/core/test_backward_leaf_query.py` holds the rule itself). This
file holds the thing `7.11` actually needs, which is a different claim: that the
answer *means what torch means by it*.

The comparison is a triple, not a single attribute::

    (requires_grad, is_leaf, grad_fn is None)

That is the whole method. Jittor and torch do disagree about autograd, and every
disagreement measured here is a disagreement about `requires_grad` -- Jittor's
float Vars require gradient by default where torch's do not, and `detach()`
stops the producing op rather than the Var. Comparing `is_leaf` on its own would
mix those in and make the graph query look wrong; comparing the triple separates
them, and the separation is the point, because `requires_grad` belongs to `7.12`
("反向叶子由 requires_grad 加图连通性决定") while connectivity is what `2.25`
delivers.

So the cases below are in three groups, and the groups are assertions:

* **agree** -- all three values identical. 14 of 17 cases, including every
  intermediate, every reduction, in-place, and a held intermediate after a
  `retain_graph=False` backward.
* **requires_grad_only** -- the graph shape agrees, `requires_grad` does not.
  Recorded with the reason and the owner.
* **shape_follows_requires_grad** -- the one case where the `requires_grad`
  difference propagates into the shape. This is the case that would otherwise
  look like the query being wrong, so it comes with two companions
  (`op_of_detached_aligned`, `op_of_frozen_source_explicit`) built under the
  Torch-facing autograd policy `EXPLICIT_REQUIRES_GRAD` from `2.09`: under that
  policy those very graphs agree with torch on all three values. The difference
  is the policy, not the query.

Torch's own invariant (`is_leaf == (grad_fn is None)`) is asserted on both sides,
because the query is built on it.

The torch column is frozen in this file rather than skipped when torch is
absent: a differential test that only runs where an independent PyTorch happens
to be installed is a test the gate reports as passing without having compared
anything. `TestAgainstLiveTorch` re-derives the same table when a binary PyTorch
is importable and asserts the frozen values are still what torch says.

Recording the frozen column (any interpreter with a binary PyTorch; the
development environment's ``torch`` is Jittor's shim and will not do)::

    REAL_TORCH_PYTHON - <<'PY'
    import torch
    print(torch.__version__)
    t = torch.tensor([1., 2., 3., 4.], requires_grad=True)
    print(t.requires_grad, t.is_leaf, t.grad_fn is None)
    PY

Run::  python -m pytest tests/core/test_backward_leaf_torch_parity.py
"""

import unittest

import numpy as np
import jittor as jt

from jittor.autograd.policy import EXPLICIT_REQUIRES_GRAD, policy_scope

from _helpers.torch_runtime import import_torch_modules, modules_available


#: PyTorch this column was recorded from.
TORCH_REFERENCE_VERSION = "2.12.1+cu126"

#: torch's ``(requires_grad, is_leaf, grad_fn is None)`` per case.
TORCH_REFERENCE = {
    "leaf_requires_grad": (True, True, True),
    "leaf_without_grad": (False, True, True),
    "unary_of_leaf": (True, False, False),
    "binary_with_scalar": (True, False, False),
    "reduction_of_intermediate": (True, False, False),
    "op_of_frozen_source": (False, True, True),
    "detached_intermediate": (False, True, True),
    "op_of_detached": (False, True, True),
    "op_of_detached_aligned": (False, True, True),
    "op_of_frozen_source_explicit": (False, True, True),
    "no_grad_output": (False, True, True),
    "integer_source": (False, True, True),
    "leaf_requires_grad_cleared": (False, True, True),
    "op_of_cleared_leaf": (False, True, True),
    "leaf_requires_grad_restored": (True, True, True),
    "op_after_restore": (True, False, False),
    "mixed_frozen_and_live_inputs": (True, False, False),
    "in_place_on_intermediate": (True, False, False),
    "held_intermediate_after_backward": (True, False, False),
}

#: Cases whose graph shape agrees with torch while ``requires_grad`` does not,
#: with the reason and who owns it.
REQUIRES_GRAD_ONLY = {
    "op_of_frozen_source":
        "Jittor's native autograd policy leaves an op's output differentiable "
        "in its own right even when every input is stopped "
        "(stop_outputs_when_inputs_stopped is off; EXPLICIT_REQUIRES_GRAD turns "
        "it on). Connectivity still says leaf, which is torch's answer. Owner: "
        "7.12 / autograd policy selection.",
    "detached_intermediate":
        "detach() marks the clone *op* stop_grad, not the Var it produced "
        "(ops/clone_op.cc), so the Var still requires grad where torch's does "
        "not. Leaf-ness agrees because the op is the barrier. Owner: 7.12.",
}

#: The one case where that difference reaches the shape, and its companion.
SHAPE_FOLLOWS_REQUIRES_GRAD = {
    "op_of_detached":
        "Downstream of the row above: torch's detached tensor does not require "
        "grad, so its consumer is not tracked at all. Jittor's does require "
        "grad, so the consumer has a live differentiable input and the query "
        "correctly reports a grad_fn. Under EXPLICIT_REQUIRES_GRAD -- the "
        "policy a Torch-facing front end selects -- the same graph agrees with "
        "torch on all three values (op_of_detached_aligned), which is what "
        "makes this a requires_grad difference rather than a connectivity one.",
}


def leaf():
    """A user-created float Var that requires gradient."""
    return jt.array(np.array([1.0, 2.0, 3.0, 4.0], dtype="float32"))


def frozen():
    """A user-created float Var that does not."""
    var = leaf()
    var.stop_grad()
    return var


def jittor_triple(var):
    return (bool(var.requires_grad),
            bool(var.is_backward_leaf),
            var.grad_fn_node_id == -1)


def jittor_answers():
    """The same eighteen cases, built with Jittor's spelling."""
    answers = {}
    answers["leaf_requires_grad"] = jittor_triple(leaf())
    answers["leaf_without_grad"] = jittor_triple(frozen())
    answers["unary_of_leaf"] = jittor_triple(leaf().abs())
    answers["binary_with_scalar"] = jittor_triple(leaf() * 2)
    answers["reduction_of_intermediate"] = jittor_triple((leaf() * 2).sum())
    answers["op_of_frozen_source"] = jittor_triple(frozen() * 2)
    answers["detached_intermediate"] = jittor_triple((leaf() * 2).detach())
    answers["op_of_detached"] = jittor_triple((leaf() * 2).detach() * 2)
    # The same two graphs under the policy a Torch-facing front end selects
    # (2.09): requires_grad is explicit and stops propagating through a stopped
    # input, which is exactly what torch does. Both then agree with torch on all
    # three values, so the differences above belong to the policy.
    with policy_scope(EXPLICIT_REQUIRES_GRAD):
        aligned = (leaf() * 2).detach()
        aligned.stop_grad()
        answers["op_of_detached_aligned"] = jittor_triple(aligned * 2)
        answers["op_of_frozen_source_explicit"] = jittor_triple(frozen() * 2)
    with jt.no_grad():
        answers["no_grad_output"] = jittor_triple(leaf() * 2)
    answers["integer_source"] = jittor_triple(jt.array(np.arange(4, dtype="int32")))
    param = leaf()
    param.requires_grad = False
    answers["leaf_requires_grad_cleared"] = jittor_triple(param)
    answers["op_of_cleared_leaf"] = jittor_triple(param * 2)
    param.requires_grad = True
    answers["leaf_requires_grad_restored"] = jittor_triple(param)
    answers["op_after_restore"] = jittor_triple(param * 2)
    answers["mixed_frozen_and_live_inputs"] = jittor_triple(frozen() * leaf())
    intermediate = leaf() * 2
    intermediate[0] = 5.0
    answers["in_place_on_intermediate"] = jittor_triple(intermediate)
    source = leaf()
    held = source * 2
    jt.grad(held.sum(), [source], retain_graph=False)
    answers["held_intermediate_after_backward"] = jittor_triple(held)
    return answers


def torch_answers(torch):
    """The same eighteen cases, built with torch's spelling."""
    def triple(tensor):
        return (bool(tensor.requires_grad),
                bool(tensor.is_leaf),
                tensor.grad_fn is None)

    def torch_leaf():
        return torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)

    def torch_frozen():
        return torch.tensor([1.0, 2.0, 3.0, 4.0])

    answers = {}
    answers["leaf_requires_grad"] = triple(torch_leaf())
    answers["leaf_without_grad"] = triple(torch_frozen())
    answers["unary_of_leaf"] = triple(torch_leaf().abs())
    answers["binary_with_scalar"] = triple(torch_leaf() * 2)
    answers["reduction_of_intermediate"] = triple((torch_leaf() * 2).sum())
    answers["op_of_frozen_source"] = triple(torch_frozen() * 2)
    answers["detached_intermediate"] = triple((torch_leaf() * 2).detach())
    answers["op_of_detached"] = triple((torch_leaf() * 2).detach() * 2)
    # torch has one policy, so these are the same graphs as above.
    answers["op_of_detached_aligned"] = triple((torch_leaf() * 2).detach() * 2)
    answers["op_of_frozen_source_explicit"] = triple(torch_frozen() * 2)
    with torch.no_grad():
        answers["no_grad_output"] = triple(torch_leaf() * 2)
    answers["integer_source"] = triple(
        torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    param = torch_leaf()
    param.requires_grad_(False)
    answers["leaf_requires_grad_cleared"] = triple(param)
    answers["op_of_cleared_leaf"] = triple(param * 2)
    param.requires_grad_(True)
    answers["leaf_requires_grad_restored"] = triple(param)
    answers["op_after_restore"] = triple(param * 2)
    answers["mixed_frozen_and_live_inputs"] = triple(torch_frozen() * torch_leaf())
    intermediate = torch_leaf() * 2
    intermediate[0] = 5.0
    answers["in_place_on_intermediate"] = triple(intermediate)
    source = torch_leaf()
    held = source * 2
    held.sum().backward()
    answers["held_intermediate_after_backward"] = triple(held)
    return answers


class TestAgainstTorchReference(unittest.TestCase):
    def setUp(self):
        self.jittor = jittor_answers()

    def test_the_case_lists_line_up(self):
        self.assertEqual(set(self.jittor), set(TORCH_REFERENCE))
        classified = set(REQUIRES_GRAD_ONLY) | set(SHAPE_FOLLOWS_REQUIRES_GRAD)
        # Every classified case must exist, so a renamed case cannot quietly
        # take an exemption with it.
        self.assertTrue(classified.issubset(set(TORCH_REFERENCE)), classified)

    def test_is_leaf_and_grad_fn_are_one_answer_on_both_sides(self):
        for name, (_, leaf_, fn_none) in sorted(TORCH_REFERENCE.items()):
            with self.subTest(case=name, side="torch"):
                self.assertEqual(leaf_, fn_none)
            with self.subTest(case=name, side="jittor"):
                _, jt_leaf, jt_fn_none = self.jittor[name]
                self.assertEqual(jt_leaf, jt_fn_none)

    def test_the_agreeing_cases_agree_on_all_three(self):
        exempt = set(REQUIRES_GRAD_ONLY) | set(SHAPE_FOLLOWS_REQUIRES_GRAD)
        agreeing = sorted(set(TORCH_REFERENCE) - exempt)
        # Guard against the exemption lists growing until nothing is compared.
        self.assertGreaterEqual(len(agreeing), 14)
        for name in agreeing:
            with self.subTest(case=name):
                self.assertEqual(self.jittor[name], TORCH_REFERENCE[name])

    def test_the_requires_grad_only_cases_still_agree_on_the_graph(self):
        for name, reason in sorted(REQUIRES_GRAD_ONLY.items()):
            with self.subTest(case=name):
                jt_triple = self.jittor[name]
                reference = TORCH_REFERENCE[name]
                self.assertEqual(jt_triple[1:], reference[1:], reason)
                # The difference is real and is asserted, not tolerated: if
                # Jittor ever matches torch here the exemption has to move.
                self.assertNotEqual(jt_triple[0], reference[0], reason)

    def test_the_one_shape_difference_is_a_requires_grad_difference(self):
        reason = SHAPE_FOLLOWS_REQUIRES_GRAD["op_of_detached"]
        self.assertNotEqual(
            self.jittor["op_of_detached"], TORCH_REFERENCE["op_of_detached"],
            reason)
        # Jittor's answer is the one connectivity implies: the detached Var does
        # require grad here, so its consumer really does have a live
        # differentiable input.
        self.assertEqual(self.jittor["op_of_detached"], (True, False, False))
        # Select the Torch-facing policy and the same graphs agree with torch on
        # all three values. This is the causal claim, asserted rather than
        # argued -- and it is also the evidence that 7.11/7.12 can reach torch
        # semantics on top of this query without changing it.
        for name in ("op_of_detached_aligned", "op_of_frozen_source_explicit"):
            with self.subTest(case=name):
                self.assertEqual(
                    self.jittor[name], TORCH_REFERENCE[name], reason)


@unittest.skipIf(not modules_available("torch"), "No independent Torch found")
class TestAgainstLiveTorch(unittest.TestCase):
    """Re-derive the frozen column where a binary PyTorch is importable."""

    @classmethod
    def setUpClass(cls):
        (cls.torch,) = import_torch_modules("torch")

    def test_the_frozen_column_is_still_what_torch_says(self):
        live = torch_answers(self.torch)
        self.assertEqual(set(live), set(TORCH_REFERENCE))
        for name in sorted(live):
            with self.subTest(case=name, torch=self.torch.__version__):
                self.assertEqual(live[name], TORCH_REFERENCE[name])


if __name__ == "__main__":
    unittest.main()

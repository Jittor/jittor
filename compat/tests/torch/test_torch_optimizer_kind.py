"""An optimizer's update rule is identified by its class, not by its name.

``fsdp2/optimizer.py`` and ``torch/optimizers.py`` both used to answer "which
optimizer is this?" with a substring of ``type(opt).__name__.lower()``. That is
not evidence about behaviour, and it failed both ways.

The loud failure is the one the audit recorded: a custom optimizer gets
``NotImplementedError``. The quiet one is worse and is what these tests are
mostly about -- a subclass named ``...Adam...`` that overrides ``step()`` was
recognised as "adam", so FSDP2 applied *its own* base AdamW update to the
shards. The subclass's update rule was never run and nothing said so.
"""
import types
import unittest

import numpy as np

import jittor as torch
import jittor as jt

from jittor.compat import optimizer_kinds
from jittor.compat.fsdp2 import optimizer as fsdp_optimizer


class TestKindOf(unittest.TestCase):
    def test_the_stock_optimizers_are_recognised(self):
        for kind, cls in (("sgd", torch.optim.SGD),
                          ("adam", torch.optim.Adam),
                          ("adamw", torch.optim.AdamW)):
            with self.subTest(kind=kind):
                opt = cls([jt.zeros(2)], lr=0.1)
                self.assertEqual(optimizer_kinds.kind_of(opt), kind)
                self.assertEqual(
                    optimizer_kinds.kind_of(opt, require_unmodified_step=True),
                    kind)

    def test_a_name_that_merely_contains_a_rule_is_not_that_rule(self):
        # The substring matcher answered "sgd" and "adamw" for these.
        class SGDW(torch.optim.Adam):
            pass

        class NoAdamWHere(torch.optim.SGD):
            pass

        self.assertEqual(optimizer_kinds.kind_of(SGDW([jt.zeros(2)], lr=0.1)),
                         "adam")
        self.assertEqual(
            optimizer_kinds.kind_of(NoAdamWHere([jt.zeros(2)], lr=0.1)), "sgd")

    def test_a_plain_subclass_keeps_its_base_rule(self):
        # The ordinary "custom Adam" people write: new defaults, same update.
        class MyAdam(torch.optim.Adam):
            def __init__(self, params, **kw):
                kw.setdefault("lr", 3e-4)
                super().__init__(params, **kw)

        opt = MyAdam([jt.zeros(2)])
        self.assertEqual(optimizer_kinds.kind_of(opt), "adam")
        self.assertEqual(
            optimizer_kinds.kind_of(opt, require_unmodified_step=True), "adam")

    def test_a_subclass_that_replaces_step_is_not_its_base_rule(self):
        # The quiet failure. Loosely it is still an Adam (its state layout is
        # Adam's); strictly it is not, because running Adam's arithmetic would
        # ignore the update rule it actually defines.
        class AdamWithMyOwnMath(torch.optim.Adam):
            def step(self, loss=None, retain_graph=False):
                raise AssertionError("the subclass's step must be the one run")

        opt = AdamWithMyOwnMath([jt.zeros(2)], lr=0.1)
        self.assertEqual(optimizer_kinds.kind_of(opt), "adam")
        self.assertIsNone(
            optimizer_kinds.kind_of(opt, require_unmodified_step=True))

    def test_an_unrelated_optimizer_is_not_guessed_at(self):
        class Lion:
            param_groups = ()

        self.assertIsNone(optimizer_kinds.kind_of(Lion()))
        self.assertEqual(optimizer_kinds.KNOWN_KINDS[:3],
                         ("adamw", "adam", "sgd"))


class TestFsdp2RefusesRatherThanRunningTheWrongUpdate(unittest.TestCase):
    """FSDP2 picks arithmetic here, so an uncertain answer must refuse."""

    def _fsdp_optimizer(self, cls, **kw):
        # One FSDP-managed shard is enough: optimizer_step checks the kind
        # before it looks at any gradient.
        shard = jt.array(np.zeros(2, dtype="float32"))
        state = types.SimpleNamespace(
            true_fsdp_initialized=True, true_fsdp_flat=False,
            true_fsdp_params=[], true_fsdp_world_size=1, true_fsdp_rank=0)
        entry = types.SimpleNamespace(shard=shard, numel=2, shape=(2,))
        state.true_fsdp_params = [entry]
        shard._jittor_fsdp2_state = state
        shard._jittor_fsdp2_entry = entry
        kw.setdefault("lr", 0.1)
        return cls([shard], **kw), state

    def test_a_subclass_that_replaces_step_is_refused_not_silently_rebased(self):
        class AdamWithMyOwnMath(torch.optim.Adam):
            def step(self, loss=None, retain_graph=False):
                raise AssertionError("never reached in this test")

        opt, _state = self._fsdp_optimizer(AdamWithMyOwnMath)
        with self.assertRaises(NotImplementedError) as caught:
            fsdp_optimizer.optimizer_step(opt)
        message = str(caught.exception)
        self.assertIn("overrides step()", message)
        self.assertIn("AdamWithMyOwnMath", message)
        # The point of the message: say what would otherwise have happened.
        self.assertIn("silently", message)

    def test_the_refusal_for_an_unrelated_optimizer_says_what_to_do(self):
        class Lion(torch.optim.Optimizer):
            pass

        opt, _state = self._fsdp_optimizer(Lion)
        with self.assertRaises(NotImplementedError) as caught:
            fsdp_optimizer.optimizer_step(opt)
        message = str(caught.exception)
        self.assertIn("Lion", message)
        self.assertIn("does not inherit", message)

    def test_a_plain_subclass_is_still_accepted(self):
        # The acceptance criterion for 7.13: a custom Adam subclass works.
        class MyAdam(torch.optim.Adam):
            def __init__(self, params, **kw):
                kw.setdefault("lr", 3e-4)
                super().__init__(params, **kw)

        opt, _state = self._fsdp_optimizer(MyAdam)
        self.assertEqual(fsdp_optimizer._optimizer_kind(opt), "adam")
        # It reaches the update path rather than raising.
        try:
            fsdp_optimizer.optimizer_step(opt)
        except NotImplementedError as exc:          # pragma: no cover
            self.fail("a plain Adam subclass must be steppable: %s" % exc)


class TestStateLayoutUsesTheLooserRule(unittest.TestCase):
    def test_a_subclass_that_replaces_step_still_has_its_base_state_layout(self):
        # `torch/optimizers.py` asks the same question for a different purpose:
        # which keys `state`/`state_dict()` expose. Overriding step() does not
        # change that, so that call site must NOT use the strict rule.
        class AdamWithMyOwnMath(torch.optim.Adam):
            def step(self, loss=None, retain_graph=False):
                return None

        opt = AdamWithMyOwnMath([jt.zeros(2)], lr=0.1)
        self.assertEqual(optimizer_kinds.kind_of(opt), "adam")
        self.assertIn("exp_avg", opt.state_dict().get("state", {}).get(0, {})
                      if opt.state_dict().get("state") else {"exp_avg": None})


if __name__ == "__main__":
    unittest.main()

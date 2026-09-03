# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``Module`` hooks: how many, in what order, and who else pays for them.

Task 5.06. A module used to hold exactly ONE hook of each kind, in a plain
attribute, and installing one swapped ``__call__``/``__hooked_call__`` **on the
class**:

* a second ``register_forward_hook`` replaced the first, silently -- while
  accelerate, peft and transformers all register several on one module and rely
  on torch's ordered dict;
* ``prepend`` and ``always_call`` were accepted and ignored;
* the swap was class-level and permanent, so hooking one ``Linear`` put every
  ``Linear`` in the process on the hook path for the rest of the run, and
  ``handle.remove()`` could not undo it;
* several ``register_*`` entry points returned nothing at all, so there was no
  handle to remove.

Run::  python -m pytest tests/core/test_module_hooks.py
"""

import unittest

import numpy as np

import jittor as jt
from jittor import nn


class Twice(nn.Module):
    def execute(self, x):
        return x * 2


class Boom(nn.Module):
    def execute(self, x):
        raise ValueError("boom")


def _x():
    return jt.array(np.ones(3, dtype="float32"))


class TestSeveralHooksAllRun(unittest.TestCase):
    def test_two_forward_hooks_both_fire_in_order(self):
        seen = []
        m = Twice()
        m.register_forward_hook(lambda mod, a, o: seen.append("first"))
        m.register_forward_hook(lambda mod, a, o: seen.append("second"))
        m(_x())
        self.assertEqual(seen, ["first", "second"])

    def test_prepend_puts_a_forward_hook_first(self):
        seen = []
        m = Twice()
        m.register_forward_hook(lambda mod, a, o: seen.append("first"))
        m.register_forward_hook(lambda mod, a, o: seen.append("early"),
                                prepend=True)
        m(_x())
        self.assertEqual(seen, ["early", "first"])

    def test_each_forward_hook_sees_the_previous_ones_replacement(self):
        m = Twice()
        m.register_forward_hook(lambda mod, a, o: o + 1)
        m.register_forward_hook(lambda mod, a, o: o * 10)
        np.testing.assert_allclose(m(_x()).numpy(), np.full(3, 30.0))

    def test_two_pre_hooks_both_fire_and_chain_their_arguments(self):
        m = Twice()
        m.register_forward_pre_hook(lambda mod, a: (a[0] + 1,))
        m.register_forward_pre_hook(lambda mod, a: (a[0] * 3,))
        # ((1 + 1) * 3) * 2
        np.testing.assert_allclose(m(_x()).numpy(), np.full(3, 12.0))

    def test_prepend_puts_a_pre_hook_first(self):
        seen = []
        m = Twice()
        m.register_forward_pre_hook(lambda mod, a: seen.append("first"))
        m.register_forward_pre_hook(lambda mod, a: seen.append("early"),
                                    prepend=True)
        m(_x())
        self.assertEqual(seen, ["early", "first"])


class TestHandlesRemoveExactlyOneHook(unittest.TestCase):
    def test_a_handle_removes_only_its_own(self):
        seen = []
        m = Twice()
        first = m.register_forward_hook(lambda mod, a, o: seen.append("first"))
        m.register_forward_hook(lambda mod, a, o: seen.append("second"))
        first.remove()
        m(_x())
        self.assertEqual(seen, ["second"])

    def test_remove_is_idempotent(self):
        m = Twice()
        handle = m.register_forward_hook(lambda mod, a, o: None)
        handle.remove()
        handle.remove()
        m(_x())

    def test_every_registrar_returns_a_handle(self):
        m = Twice()
        for register, args in (
                (m.register_forward_hook, (lambda mod, a, o: None,)),
                (m.register_forward_pre_hook, (lambda mod, a: None,)),
                (m.register_pre_forward_hook, (lambda mod, a: None,)),
                (m.register_input_backward_hook, (lambda g: None,)),
                (m.register_output_backward_hook, (lambda g: None,)),
                (m.register_backward_hook, (lambda mod, gi, go: None,)),
        ):
            handle = register(*args)
            self.assertTrue(hasattr(handle, "remove"),
                            f"{register.__name__} returned {handle!r}")
            handle.remove()

    def test_removing_the_last_hook_puts_the_module_back_on_the_plain_path(self):
        m = Twice()
        handle = m.register_forward_hook(lambda mod, a, o: o * 0)
        np.testing.assert_allclose(m(_x()).numpy(), np.zeros(3))
        handle.remove()
        np.testing.assert_allclose(m(_x()).numpy(), np.full(3, 2.0))


class TestHooksAreInstanceLevel(unittest.TestCase):
    def test_registering_a_hook_does_not_rewrite_the_class(self):
        class Tiny(nn.Module):
            def execute(self, x):
                return x * 2

        before = set(vars(Tiny))
        module = Tiny()
        handle = module.register_forward_hook(lambda mod, a, o: o)
        module(_x())
        self.assertEqual(
            set(vars(Tiny)) - before, set(),
            "installing a hook rewrote the CLASS -- every instance of it, for "
            "the rest of the process, with no way back")
        handle.remove()

    def test_another_instance_of_the_same_class_is_untouched(self):
        class Tiny(nn.Module):
            def execute(self, x):
                return x * 2

        hooked, plain = Tiny(), Tiny()
        seen = []
        hooked.register_forward_hook(lambda mod, a, o: seen.append(mod))
        plain(_x())
        self.assertEqual(seen, [], "a sibling instance ran the hook path")
        hooked(_x())
        self.assertEqual(seen, [hooked])

    def test_a_hook_on_a_base_instance_does_not_reach_a_subclass_instance(self):
        class Base(nn.Module):
            def execute(self, x):
                return x * 2

        class Derived(Base):
            pass

        seen = []
        base = Base()
        base.register_forward_hook(lambda mod, a, o: seen.append("base"))
        derived = Derived()
        derived.register_forward_hook(lambda mod, a, o: seen.append("derived"))
        derived(_x())
        self.assertEqual(seen, ["derived"])
        base(_x())
        self.assertEqual(seen, ["derived", "base"])

    def test_hook_tables_are_not_mistaken_for_parameters(self):
        model = nn.Linear(4, 3)
        before = len(model.parameters())
        model.register_forward_hook(lambda mod, a, o: o)
        model.register_forward_pre_hook(lambda mod, a: None)
        self.assertEqual(len(model.parameters()), before)


class TestAlwaysCall(unittest.TestCase):
    def test_a_normal_hook_is_skipped_when_the_forward_raises(self):
        seen = []
        m = Boom()
        m.register_forward_hook(lambda mod, a, o: seen.append("plain"))
        with self.assertRaises(ValueError):
            m(_x())
        self.assertEqual(seen, [])

    def test_an_always_call_hook_runs_when_the_forward_raises(self):
        seen = []
        m = Boom()
        m.register_forward_hook(lambda mod, a, o: seen.append(o),
                                always_call=True)
        with self.assertRaises(ValueError):
            m(_x())
        self.assertEqual(seen, [None],
                         "always_call=True was accepted and ignored")

    def test_an_always_call_hook_runs_exactly_once_on_success(self):
        seen = []
        m = Twice()
        m.register_forward_hook(lambda mod, a, o: seen.append("once"),
                                always_call=True)
        m(_x())
        self.assertEqual(seen, ["once"])


class TestBackwardHooksStillWork(unittest.TestCase):
    def test_register_backward_hook_replaces_the_input_gradient(self):
        relu = nn.ReLU()
        hooked = []

        def hook(mod, grad_input, grad_output):
            hooked.append(True)
            return (jt.array([-1.0, -2.0]),)

        relu.register_backward_hook(hook)
        x = jt.array([-1.0, 2.0])
        dx = jt.grad(relu(x), x)
        self.assertTrue(hooked)
        np.testing.assert_allclose(dx.numpy(), [-1.0, -2.0])

    def test_the_backward_handle_removes_both_halves(self):
        relu = nn.ReLU()
        hooked = []

        def hook(mod, grad_input, grad_output):
            hooked.append(True)
            return (jt.array([-1.0, -2.0]),)

        handle = relu.register_backward_hook(hook)
        handle.remove()
        x = jt.array([-1.0, 2.0])
        dx = jt.grad(relu(x), x)
        self.assertEqual(hooked, [])
        np.testing.assert_allclose(dx.numpy(), [0.0, 1.0])


class TestVarRegisterHook(unittest.TestCase):
    """``Var.register_hook`` returned the Var, so a hook could never be removed."""

    def _pair(self):
        x = jt.array([0.0, 0.0])
        return x, x * [1.0, 2.0]

    def test_the_hook_still_alters_the_gradient(self):
        x, y = self._pair()
        y.register_hook(lambda g: g * 2)
        np.testing.assert_allclose(jt.grad(y, x).numpy(), [2.0, 4.0])

    def test_it_returns_a_removable_handle(self):
        x, y = self._pair()
        handle = y.register_hook(lambda g: g * 2)
        self.assertTrue(hasattr(handle, "remove"),
                        f"register_hook returned {handle!r}, not a handle")

    def test_removing_it_restores_the_plain_gradient(self):
        x, y = self._pair()
        handle = y.register_hook(lambda g: g * 2)
        handle.remove()
        np.testing.assert_allclose(jt.grad(y, x).numpy(), [1.0, 2.0])

    def test_removing_one_of_two_leaves_the_other(self):
        x, y = self._pair()
        first = y.register_hook(lambda g: g * 2)
        y.register_hook(lambda g: g * 3)
        first.remove()
        np.testing.assert_allclose(jt.grad(y, x).numpy(), [3.0, 6.0])

    def test_the_handle_is_the_same_kind_module_hooks_hand_out(self):
        module_handle = Twice().register_forward_hook(lambda mod, a, o: None)
        _x_, y = self._pair()
        var_handle = y.register_hook(lambda g: g)
        self.assertIs(type(var_handle), type(module_handle))


class TestWithKwargs(unittest.TestCase):
    class Adder(nn.Module):
        def execute(self, x, bias=0):
            return x + bias

    def test_a_with_kwargs_pre_hook_can_replace_the_kwargs(self):
        m = self.Adder()
        m.register_forward_pre_hook(
            lambda mod, a, kw: (a, {"bias": 5}), with_kwargs=True)
        np.testing.assert_allclose(m(_x(), bias=1).numpy(), np.full(3, 6.0))

    def test_a_with_kwargs_forward_hook_sees_kwargs_before_output(self):
        seen = {}
        m = self.Adder()

        def hook(mod, args, kwargs, output):
            seen["kwargs"] = dict(kwargs)
            seen["output"] = output
            return None

        m.register_forward_hook(hook, with_kwargs=True)
        m(_x(), bias=1)
        self.assertEqual(seen["kwargs"], {"bias": 1})
        np.testing.assert_allclose(seen["output"].numpy(), np.full(3, 2.0))


if __name__ == "__main__":
    unittest.main()

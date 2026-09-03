# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""What the star imports publish, and what they must stop publishing.

Task 5.23 (first half). ``jittor/__init__.py`` builds its namespace from five
star imports. None of the modules behind them declared ``__all__``, so each one
republished **its own imports** as well as its API: ``jittor.misc`` exported
``np``, ``math``, ``time``, ``Sequence`` and ``Iterable`` because
``misc/tensor_ops.py`` imports them, and ``from .misc import *`` carried that on
to the root. ``jt.np`` and ``jt.math`` were public API by accident.

The trap is that code then *depends* on the leak. ``misc/shape_transforms.py``
reached for ``jt.misc.Sequence`` and ``jt.misc.np`` -- names it never imported
-- so adding ``__all__`` without looking first would have broken ``repeat()``.
That is why the dependency is fixed in the same commit, and why this file pins
``repeat()`` specifically.

Run::  python -m pytest tests/core/test_namespace_exports.py
"""

import types
import unittest

import numpy as np

import jittor as jt


#: Names that must never be part of ``jittor.misc``: they are its imports, not
#: its API.
_LEAKED_INTO_MISC = ("np", "math", "time", "Sequence", "Iterable", "jt")


class TestMiscStopsRepublishingItsImports(unittest.TestCase):
    def test_the_stdlib_names_are_gone(self):
        leaked = [name for name in _LEAKED_INTO_MISC if hasattr(jt.misc, name)]
        self.assertEqual(
            leaked, [],
            "jittor.misc re-exports its own imports; give the module behind "
            "the star import an __all__")

    def test_misc_declares_what_it_publishes(self):
        from jittor.misc import tensor_ops
        self.assertTrue(hasattr(tensor_ops, "__all__"))
        self.assertNotIn("np", tensor_ops.__all__)
        self.assertNotIn("math", tensor_ops.__all__)

    def test_the_api_it_did_publish_is_still_there(self):
        # A sample across the module's own definitions and the sibling names
        # it deliberately republishes.
        for name in ("unique", "meshgrid", "scatter", "cumsum", "topk",
                     "repeat", "chunk", "expand", "block_diag",
                     "cartesian_prod", "atleast_1d"):
            self.assertTrue(hasattr(jt.misc, name), name)
            self.assertTrue(hasattr(jt, name), "root lost %s" % name)


class TestRepeatNoLongerNeedsTheLeak(unittest.TestCase):
    """``repeat()`` read ``jt.misc.Sequence`` and ``jt.misc.np``."""

    def test_repeat_with_separate_ints(self):
        x = jt.array(np.array([1, 2, 3], dtype="float32"))
        np.testing.assert_allclose(
            x.repeat(4, 2).numpy(),
            np.tile(np.array([1, 2, 3], dtype="float32"), (4, 2)))

    def test_repeat_with_a_single_sequence_argument(self):
        # This is the branch that tested `isinstance(shape[0], jt.misc.Sequence)`.
        x = jt.array(np.array([1, 2, 3], dtype="float32"))
        np.testing.assert_allclose(
            x.repeat((4, 2)).numpy(),
            np.tile(np.array([1, 2, 3], dtype="float32"), (4, 2)))

    def test_repeat_adds_leading_dims(self):
        x = jt.array(np.array([1, 2, 3], dtype="float32"))
        self.assertEqual(tuple(x.repeat(4, 2, 1).shape), (4, 2, 3))

    def test_shape_transforms_imports_what_it_uses(self):
        from jittor.misc import shape_transforms
        self.assertIs(shape_transforms.np, np)
        from collections.abc import Sequence
        self.assertIs(shape_transforms.Sequence, Sequence)


class TestTheRootStillLeaksFromTheOtherStarImports(unittest.TestCase):
    """Half the job. Recorded here so the remainder is a fact, not a memory.

    ``jittor/_runtime/core_api.py`` is the other big star import and has no
    ``__all__``, so ``os``, ``sys``, ``np``, ``Sequence``, ``Mapping``,
    ``OrderedDict``, ``copy``, ``pickle``, ``hashlib``, ``traceback``,
    ``types``, ``contextlib``, ``numbers``, ``itertools`` and ``random`` are
    still root attributes. Fixing that means regenerating ``__init__.pyi`` from
    the declared surface, which is its own task -- so this test states the
    situation instead of pretending it is clean.
    """

    def test_the_remaining_leak_is_confined_to_core_api(self):
        from jittor._runtime import core_api
        still_leaking = [name for name in ("os", "sys", "np", "Sequence",
                                           "OrderedDict", "pickle")
                         if hasattr(jt, name)]
        for name in still_leaking:
            self.assertTrue(
                hasattr(core_api, name),
                "%r is on the root but no longer comes from core_api; find "
                "which star import publishes it now" % name)

    def test_jittor_still_exports_itself_as_jt(self):
        # jt.jt exists because a module named `jt` was star-imported somewhere.
        # Not fixed here; pinned so the day it goes away is a deliberate one.
        self.assertTrue(isinstance(getattr(jt, "jt", None), types.ModuleType))


if __name__ == "__main__":
    unittest.main()

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

import unittest
import ast
from pathlib import Path

import numpy as np

import jittor as jt


ROOT_INIT = Path(jt.__file__).resolve()
ROOT_STUB = ROOT_INIT.with_suffix(".pyi")


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


class TestRootStopsRepublishingImplementationImports(unittest.TestCase):
    def test_runtime_imports_are_not_root_attributes(self):
        leaked = [
            name for name in (
                "os", "sys", "np", "Sequence", "Mapping", "OrderedDict",
                "pickle", "hashlib", "traceback", "types", "contextlib",
                "numbers", "itertools", "jt",
            )
            if hasattr(jt, name)
        ]
        self.assertEqual(leaked, [])


class TestRootExportsAreDeclared(unittest.TestCase):
    def test_root_initializer_has_no_star_imports(self):
        tree = ast.parse(ROOT_INIT.read_text(encoding="utf-8"))
        stars = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and any(alias.name == "*" for alias in node.names)
        ]
        self.assertEqual(stars, [])

    def test_every_declared_export_exists(self):
        self.assertTrue(hasattr(jt, "__all__"))
        missing = [name for name in jt.__all__ if not hasattr(jt, name)]
        self.assertEqual(missing, [])
        for leaked in ("os", "sys", "np", "Sequence", "Mapping", "pickle"):
            self.assertNotIn(leaked, jt.__all__)

    def test_star_import_is_exactly_the_declared_surface(self):
        namespace = {}
        exec("from jittor import *", {}, namespace)
        self.assertEqual(set(namespace) - {"__builtins__"}, set(jt.__all__))

    def test_stub_export_manifest_matches_runtime(self):
        tree = ast.parse(ROOT_STUB.read_text(encoding="utf-8"))
        assignments = [
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "__all__"
                    for target in node.targets)
        ]
        self.assertEqual(len(assignments), 1)
        self.assertEqual(tuple(ast.literal_eval(assignments[0].value)), tuple(jt.__all__))

    def test_stub_top_level_names_match_the_export_surface(self):
        tree = ast.parse(ROOT_STUB.read_text(encoding="utf-8"))
        declared = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not node.name.startswith("_") or node.name == "__version__":
                    declared.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                declared.update(
                    target.id for target in targets
                    if isinstance(target, ast.Name)
                    and (not target.id.startswith("_") or target.id == "__version__")
                )
            elif isinstance(node, ast.ImportFrom):
                if node.module in ("typing", "collections", "collections.abc"):
                    continue
                for alias in node.names:
                    if alias.name != "*":
                        declared.add(alias.asname or alias.name)
                    elif node.module == "jittor_core":
                        declared.update(
                            name for name in dir(jt.jittor_core)
                            if not name.startswith("_"))
                    elif node.module == "jittor_core.ops":
                        declared.update(
                            name for name in dir(jt.ops)
                            if not name.startswith("_"))
        self.assertEqual(declared, set(jt.__all__))


if __name__ == "__main__":
    unittest.main()

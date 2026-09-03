# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""A module must not reach through ``jt.nn`` for its own private names.

Task 5.22. ``jittor/nn/__init__.py`` re-exports ~35 underscore-private names,
and the modules that define them called *back through the facade* to reach
them -- ``nn/backends/cudnn.py`` read ``jt.nn._CUDNN_3D_HALF_DTYPES`` for a
constant defined four lines above it.

Late binding through the facade is a real mechanism here: ACL replaces
``jt.nn._batch_norm_eval_cuda``, ``jt.nn._group_norm_cuda`` and
``jt.nn._rms_norm_cuda`` at runtime, and ``nn/functional/normalization.py``
calls them through ``jt.nn`` **on purpose** so the replacement takes effect.
That is a backend-integration hook, and it is always cross-module: the module
that defines the default and the module that overrides it are different files.

A module reaching through the facade for a name it defines *itself* is never
that. It is a self-reference that any unrelated replacement of that attribute
silently redirects -- so ``_ln_normalize`` could be swapped out from under
``layer_norm`` by code that meant to affect something else entirely.

Rule: **a private name defined in a module is referenced by its local name in
that module.** Cross-module facade reads are untouched by this test; they are
the hook.

Static only.
"""

import ast
import unittest
from pathlib import Path

import jittor


PACKAGE = Path(jittor.__file__).resolve().parent


def _module_level_private_names(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return {name for name in names if name.startswith("_")}


def _is_nn_facade_attribute(node):
    return (isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "nn"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id in ("jt", "jittor"))


def _self_references():
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        private = _module_level_private_names(tree)
        if not private:
            continue
        for node in ast.walk(tree):
            if not _is_nn_facade_attribute(node):
                continue
            if node.attr not in private:
                continue
            # A WRITE publishes the name onto the facade -- that is the export,
            # not a self-reference. Only reads are the defect.
            if not isinstance(node.ctx, ast.Load):
                continue
            offenders.append("%s:%d reads jt.nn.%s, which it defines itself"
                             % (path.relative_to(PACKAGE), node.lineno,
                                node.attr))
    return offenders


class TestNoPrivateSelfReferenceThroughTheFacade(unittest.TestCase):
    def test_modules_use_their_own_private_names_directly(self):
        self.assertEqual(
            _self_references(), [],
            "use the module-local name. Reaching through jt.nn for a private "
            "name the module defines itself means any replacement of that "
            "facade attribute -- by a backend, a test, or an adapter aiming "
            "at something else -- changes this function's internals.")


class TestTestsPatchWhereTheNameLives(unittest.TestCase):
    """A test that patches the facade must be patching a cross-module hook.

    ``test_torch_compat_norm.py`` used to replace ``nn._ln_normalize`` to prove
    the CUDA no-grad LayerNorm never falls back to the composite path. Once
    ``layer_norm`` called its own module-local name, that patch stopped
    intercepting anything and the test would have gone on passing while
    measuring nothing -- the worst outcome available.
    """

    def test_the_layernorm_fallback_probe_patches_the_defining_module(self):
        source = (Path(__file__).resolve().parents[1] / "compat" / "torch"
                  / "test_torch_compat_norm.py").read_text(encoding="utf-8")
        self.assertIn("_normalization._ln_normalize = reject_composite", source)
        self.assertNotIn("nn._ln_normalize = reject_composite", source)


if __name__ == "__main__":
    unittest.main()

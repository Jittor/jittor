# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The declared installer order and the code must not drift apart.

Task 5.21. ``jittor/_install_order.py`` writes down the order the runtime's
monkeypatch installers run in. A written-down order is worth nothing if the code
can move on without it, so: every name in ``SEQUENCE`` must have a ``record()``
call somewhere in the package, and every ``record()`` call must name a declared
step.

Static only -- it reads the source, it does not import the runtime.
"""

import ast
import re
import unittest
from pathlib import Path

import jittor


PACKAGE = Path(jittor.__file__).resolve().parent
_ORDER_MODULE = PACKAGE / "_install_order.py"


def _declared_steps():
    """The names in SEQUENCE, read out of the source rather than imported."""
    tree = ast.parse(_ORDER_MODULE.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "SEQUENCE"
                   for t in node.targets):
            continue
        names = []
        for call in node.value.elts:
            assert isinstance(call, ast.Call), ast.dump(call)
            assert call.args and isinstance(call.args[0], ast.Constant)
            names.append(call.args[0].value)
        return names
    raise AssertionError("SEQUENCE not found in _install_order.py")


def _is_install_record(func):
    """``_record_install(...)`` or ``_install_order.record(...)``, and nothing else.

    Matched by the exact spelling the install sites use, not by the bare name
    ``record``: ``compat/torch/installers/cuda.py`` has CUDA events with a
    ``record()`` method of their own, and a looser match swept those up.
    """
    if isinstance(func, ast.Name):
        return func.id == "_record_install"
    return (isinstance(func, ast.Attribute) and func.attr == "record"
            and isinstance(func.value, ast.Name)
            and func.value.id == "_install_order")


def _recorded_names():
    """Every step name handed to an install-order ``record`` call."""
    found = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts or path == _ORDER_MODULE:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not _is_install_record(node.func):
                continue
            assert node.args, "%s:%d record() with no name" % (path, node.lineno)
            argument = node.args[0]
            assert isinstance(argument, ast.Constant) and \
                isinstance(argument.value, str), \
                "%s:%d record() needs a literal step name" % (path, node.lineno)
            found.setdefault(argument.value, []).append(
                "%s:%d" % (path.relative_to(PACKAGE), node.lineno))
    return found


class TestInstallOrderDeclaration(unittest.TestCase):
    def test_every_declared_step_is_recorded_somewhere(self):
        declared = _declared_steps()
        recorded = _recorded_names()
        missing = [name for name in declared if name not in recorded]
        self.assertEqual(
            missing, [],
            "declared in _install_order.SEQUENCE but nothing calls record() "
            "for it, so verify() would fail the import (or, if it is not "
            "required, the step is fiction)")

    def test_every_recorded_name_is_declared(self):
        declared = set(_declared_steps())
        offenders = []
        for name, sites in _recorded_names().items():
            if name not in declared:
                offenders.extend("%s -> %r" % (site, name) for site in sites)
        self.assertEqual(
            offenders, [],
            "record() names a step that is not in SEQUENCE; add it in the "
            "position its ordering constraints require")

    def test_each_step_is_recorded_exactly_once(self):
        # Two call sites for one step means the order it claims to pin is not
        # the order that actually runs.
        offenders = {name: sites for name, sites in _recorded_names().items()
                     if len(sites) > 1}
        self.assertEqual(offenders, {},
                         "a step is recorded from more than one place")

    def test_every_step_says_why_it_sits_where_it_does(self):
        source = _ORDER_MODULE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "Step"):
                continue
            self.assertEqual(len(node.args), 3, ast.dump(node))
            why = node.args[2]
            self.assertIsInstance(why, ast.Constant, ast.dump(node))
            self.assertGreater(
                len(why.value), 60,
                "Step(%r) needs a real reason, not a label: the list is only "
                "useful if it says what breaks when the order changes"
                % (node.args[0].value,))

    def test_the_root_import_verifies_the_sequence(self):
        source = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("_install_order.verify()", source,
                      "nothing checks the sequence at the end of the import")


class TestFullReduceRoutesBothSpellings(unittest.TestCase):
    """``jt.sum`` and ``Var.sum`` must be routed by the same installer."""

    def test_the_installer_routes_the_root_functions_too(self):
        source = (PACKAGE / "nn" / "backends"
                  / "full_reduce_cuda.py").read_text(encoding="utf-8")
        self.assertRegex(
            source, r"setattr\(jt, name",
            "install_full_reduce_fast_path must route jt.sum/jt.mean as well "
            "as the methods, or one operation keeps two numerics")
        self.assertEqual(
            len(re.findall(r"^def _route\(", source, re.M)), 1,
            "one wrapper factory, so the fast-path decision is made once")


if __name__ == "__main__":
    unittest.main()

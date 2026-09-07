"""The dependency direction between the build tools and the framework.

Three real cycles were named in the architecture audit. Two are C++ include
cycles and already have contracts:

- ``Executor`` ⇄ ``VarHolder`` -- ``test_core_include_direction.py``
- ``Node`` ⇄ pyjt tracer -- ``test_node_lifecycle_layering.py``

The third is Python and had nothing holding it: ``jittor_utils`` is the package
``jittor.compiler`` imports at its top, so a ``jittor`` import anywhere in
``jittor_utils`` makes the lower layer depend on the framework it serves. It is
clean today; this keeps it that way.

Parsed, not grepped, on purpose. Every current textual match for
``import jittor`` under ``jittor_utils`` is inside a docstring ``Example::``
block or a C++ string literal handed to ``console.run``, and a grep-based
version of this rule reports all of them.
"""

from __future__ import print_function

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPO_ROOT / "python" / "jittor" / "build" / "utils"

#: Deferred imports that are allowed to name the framework, with the reason.
#: A function-local import does not create an import-time cycle, but it still
#: means the tool needs the framework, so each one is listed rather than waved
#: through by a blanket "function-local is fine" rule. Empty is the goal.
_ALLOWED_DEFERRED = {}


def _framework_import(node):
    """The module name this node imports from ``jittor``, or None."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name == "jittor" or alias.name.startswith("jittor."):
                return alias.name
    if isinstance(node, ast.ImportFrom) and node.level == 0:
        module = node.module or ""
        if module == "jittor" or module.startswith("jittor."):
            return module
    return None


def _module_level_and_deferred(tree):
    """Split framework imports into module-level and everything-else.

    Module level is exactly the import statements in ``tree.body``; anything
    else sits inside a function, a class body or a conditional block and only
    runs when that code does.
    """
    top = {id(node) for node in tree.body}
    module_level, deferred = [], []
    for node in ast.walk(tree):
        name = _framework_import(node)
        if name is None:
            continue
        target = module_level if id(node) in top else deferred
        target.append((node.lineno, name))
    return module_level, deferred


def _sources():
    return sorted(TOOLS_ROOT.rglob("*.py"))


def test_the_build_tools_do_not_import_the_framework_at_import_time():
    """``jittor_utils`` is below ``jittor``; a top-level import inverts that."""
    offenders = []
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module_level, _ = _module_level_and_deferred(tree)
        for lineno, name in module_level:
            offenders.append("%s:%d imports %s"
                             % (path.relative_to(REPO_ROOT).as_posix(), lineno, name))
    assert offenders == [], "\n".join(offenders)


def test_deferred_framework_imports_stay_on_the_declared_list():
    """A function-local framework import is a listed exception, not a habit."""
    found = {}
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        _, deferred = _module_level_and_deferred(tree)
        for lineno, name in deferred:
            found["%s:%d" % (path.relative_to(REPO_ROOT).as_posix(), lineno)] = name
    unexpected = sorted(set(found) - set(_ALLOWED_DEFERRED))
    assert unexpected == [], (
        "new deferred framework imports; implement the need through an "
        "injected service or add the site to _ALLOWED_DEFERRED with a reason:\n"
        + "\n".join("%s imports %s" % (site, found[site]) for site in unexpected))
    stale = sorted(set(_ALLOWED_DEFERRED) - set(found))
    assert stale == [], "these listed exceptions are gone; drop them: %s" % stale

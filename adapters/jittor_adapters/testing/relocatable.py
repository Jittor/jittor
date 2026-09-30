"""Static public-framework boundary shared by independently shipped adapters.

This module has no Jittor or downstream dependency. Callers supply their own
reviewed public-import allowlist; sharing the scanner must not widen it.
"""

import ast
from pathlib import Path
from typing import NamedTuple


FRAMEWORK_ROOTS = frozenset(("jt", "jittor", "torch"))


class Violation(NamedTuple):
    kind: str
    lineno: int
    detail: str


def adapter_sources(package):
    """Scan nested runtime modules while excluding co-located tests."""
    package = Path(package)
    return sorted(path for path in package.rglob("*.py")
                  if "tests" not in path.relative_to(package).parts)


def framework_root(node, roots=FRAMEWORK_ROOTS):
    """Return the framework alias behind an attribute chain, if any."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) and node.id in roots else None


def _is_framework_module(name):
    return name in ("jittor", "torch") or name.startswith(("jittor.", "torch."))


def _is_jittor_module(name):
    return name == "jittor" or name.startswith("jittor.")


def _private(name):
    return name.startswith("_") and not name.startswith("__")


def relocatable_violations(source, allowed_jittor_imports):
    """Return import/private-read/framework-write violations with locations.

    This is a source contract, not a general Python alias/data-flow proof.
    Direct imported aliases and literal getattr/hasattr spellings are covered.
    """
    tree = ast.parse(source)
    roots = set(FRAMEWORK_ROOTS)
    issues = []

    def report(kind, node, detail):
        issues.append(Violation(kind, node.lineno, detail))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if _is_framework_module(name):
                    roots.add(alias.asname or name.split(".")[0])
                if _is_jittor_module(name) and name not in allowed_jittor_imports:
                    report("private import", node, name)
        elif isinstance(node, ast.ImportFrom) and not node.level:
            name = node.module or ""
            if _is_jittor_module(name) and name not in allowed_jittor_imports:
                report("private import", node, name)
            if _is_framework_module(name):
                for alias in node.names:
                    roots.add(alias.asname or alias.name)
                    if _private(alias.name):
                        report("private attribute", node, name + "." + alias.name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and framework_root(node, roots):
            if _private(node.attr):
                report("private attribute", node, node.attr)
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                report("framework mutation", node, node.attr)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.args and framework_root(node.args[0], roots)):
            if node.func.id in ("setattr", "delattr"):
                report("framework mutation", node, node.func.id)
            if node.func.id in ("getattr", "hasattr") and len(node.args) > 1:
                attribute = node.args[1]
                if (isinstance(attribute, ast.Str)
                        and _private(attribute.s)):
                    report("private attribute", node, attribute.s)
    return issues

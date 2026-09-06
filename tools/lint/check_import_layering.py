"""Import-direction gate: cycle ratchet and tools-below-framework layering.

Why this is hand-rolled instead of ``import-linter``
---------------------------------------------------
``import-linter`` builds its graph with ``grimp``, which locates a root
package through ``importlib`` and then walks that one directory. The backend
packages are not in that directory: ``pyproject.toml`` maps
``jittor.backends.<x>`` onto the top-level ``backends/<x>`` via ``package-dir``,
and in a source checkout ``python/jittor/backends/__init__.py`` splices them in
by appending to ``__path__`` *at runtime*. Static resolution cannot see that.

Measured on this tree: ``grimp.build_graph("jittor", "jittor_utils")`` finds
294 modules and exactly one ``jittor.backends.*`` module (the shim
``__init__``), against 385 modules and 88 ``jittor.backends.*`` modules found
here. The 91 invisible modules include every ACL and CUDA kernel module, and
they are not bystanders: they sit inside the largest import cycle in the tree.
An ``import-linter`` contract over these packages would report "no violations"
while never having read a quarter of the source -- the failure this gate exists
to prevent. Making ``grimp`` see them would mean naming ``jittor.backends.cuda``
as a root package, which makes ``importlib`` import ``jittor`` first, which
compiles the C++ core. A layout check must not need a compiler.

So the scan roots are read from ``package-dir`` in ``pyproject.toml`` rather
than hardcoded: when the ``python/jittor/extern`` -> ``backends`` migration
finishes, this file does not need editing, and if the mapping is edited to
something that resolves to nothing, ``check_coverage`` fails instead of
quietly passing on an empty set.

Run directly (``python tools/lint/check_import_layering.py``) or through
``nox -s imports``; ``tests/structure/test_import_layering.py`` asserts the
same report.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

MODULE = "module"
DEFERRED = "deferred"
TYPING = "typing"

# ---------------------------------------------------------------------------
# Baselines. Every number here is measured, not aspirational: run this file
# with --report to reprint them. They are ceilings and allowlists, so the
# tree can only get better without editing this block.
# ---------------------------------------------------------------------------

#: Floors that make "zero violations" mean something. If a layout change makes
#: a root resolve to nothing, these fail rather than letting the contracts
#: pass over an empty graph.
MIN_MODULES = 360
MIN_IMPORT_EDGES = 900
MIN_BACKEND_MODULES = 80

#: Subpackages that currently sit on an import-time cycle. Frozen so a *new*
#: area of the tree joining a cycle is a failure. Shrinking is always allowed.
#:
#: The bulk of this is one architectural cycle, not many small ones: the
#: ``jittor`` package __init__ imports its submodules while ~150 of those
#: submodules do ``import jittor as jt`` at module scope. Breaking that up is
#: 4.07's job, not this gate's; the gate's job is that it stops growing.
CYCLIC_SUBPACKAGES = frozenset(
    {
        "jittor",
        "jittor._runtime",
        "jittor.autograd",
        "jittor.backends.acl",
        "jittor.backends.cuda",
        "jittor.compat",
        "jittor.compile_extern",
        "jittor.compiler",
        "jittor.dataset",
        "jittor.distributions",
        "jittor.einops",
        "jittor.fft",
        "jittor.init",
        "jittor.linalg",
        "jittor.math_util",
        "jittor.misc",
        "jittor.nn",
        "jittor.optim",
        "jittor.pool",
        "jittor.sparse",
        "jittor.transform",
        "jittor_utils",
    }
)

#: Ceiling on how many modules sit on an import-time cycle. Catches growth
#: inside a subpackage that is already on the list above.
MAX_CYCLIC_MODULES = 164

#: Ceiling on the number of distinct import-time cycles.
MAX_CYCLES = 3

#: Deferred (function-local) framework imports inside the tools layer, with a
#: reason each. A function-local import does not create an import-time cycle,
#: but it still means the lower layer needs the upper one. Empty is the goal.
ALLOWED_DEFERRED_TOOL_IMPORTS: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _load_pyproject(root: Path) -> dict:
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python < 3.11
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            from setuptools._vendor import tomli as tomllib  # type: ignore[no-redef]
    with (root / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def scan_roots(root: Path) -> list[tuple[Path, str]]:
    """(directory, module prefix) pairs taken from ``package-dir``.

    ``{"": "python"}`` means "every importable package under python/"; an
    explicit ``{"jittor.backends.cuda": "backends/cuda"}`` maps one dotted
    name onto a directory outside the main source root.
    """
    package_dir = _load_pyproject(root)["tool"]["setuptools"]["package-dir"]
    roots: list[tuple[Path, str]] = []
    for name, location in sorted(package_dir.items()):
        base = root / location
        if name:
            roots.append((base, name))
            continue
        for child in sorted(base.iterdir()):
            if child.is_dir() and (child / "__init__.py").is_file():
                roots.append((child, child.name))
    return roots


def discover_modules(roots: list[tuple[Path, str]]) -> dict[str, Path]:
    modules: dict[str, Path] = {}
    for base, prefix in roots:
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            parts = list(path.relative_to(base).parts)
            if parts[-1] == "__init__.py":
                parts = parts[:-1]
            else:
                parts[-1] = parts[-1][: -len(".py")]
            modules[".".join([prefix] + parts)] = path
    return modules


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _visit(node: ast.AST, scope: str, out: list[tuple[ast.AST, str]]) -> None:
    """Collect import statements, tagged with the scope that runs them.

    A class body runs when the module is imported, so it keeps the enclosing
    scope; a function body does not. ``if TYPE_CHECKING:`` never runs at all.
    """
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        out.append((node, scope))
        return
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        scope = DEFERRED
    elif isinstance(node, ast.If) and scope == MODULE and _is_type_checking(node.test):
        for sub in node.body:
            _visit(sub, TYPING, out)
        for sub in node.orelse:
            _visit(sub, scope, out)
        return
    for child in ast.iter_child_nodes(node):
        _visit(child, scope, out)


def _resolve(target: str, known: set[str]) -> str | None:
    """Longest known prefix of a dotted name (``a.b.C`` lands on ``a.b``)."""
    parts = target.split(".")
    while parts:
        candidate = ".".join(parts)
        if candidate in known:
            return candidate
        parts.pop()
    return None


def _absolute(node: ast.AST, module: str, is_package: bool) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    assert isinstance(node, ast.ImportFrom)
    level = node.level or 0
    if not level:
        base = node.module or ""
        return [base] + ["%s.%s" % (base, alias.name) for alias in node.names]
    owner = module.split(".")
    if not is_package:
        owner = owner[:-1]
    if level > 1:
        owner = owner[: len(owner) - (level - 1)]
    prefix = ".".join(owner + ([node.module] if node.module else []))
    if node.module:
        return [prefix] + ["%s.%s" % (prefix, alias.name) for alias in node.names]
    return ["%s.%s" % (prefix, alias.name) for alias in node.names]


def build_edges(modules: dict[str, Path]) -> dict[str, dict[str, set[str]]]:
    """module -> imported module -> set of scopes that import it."""
    known = set(modules)
    edges: dict[str, dict[str, set[str]]] = {name: {} for name in modules}
    for name, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found: list[tuple[ast.AST, str]] = []
        _visit(tree, MODULE, found)
        for node, scope in found:
            for target in _absolute(node, name, path.name == "__init__.py"):
                resolved = _resolve(target, known) if target else None
                if resolved is None or resolved == name:
                    continue
                edges[name].setdefault(resolved, set()).add(scope)
    return edges


def graph_for(edges: dict[str, dict[str, set[str]]], scopes: set[str]) -> dict[str, set[str]]:
    return {
        src: {dst for dst, seen in outs.items() if seen & scopes} for src, outs in edges.items()
    }


def strongly_connected(graph: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan, iterative, returning only the components that are cycles."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    stack: list[str] = []
    found: list[list[str]] = []
    counter = 0
    for root in graph:
        if root in index:
            continue
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack[root] = True
        work = [(root, iter(sorted(graph[root])))]
        while work:
            node, children = work[-1]
            descended = False
            for nxt in children:
                if nxt not in graph:
                    continue
                if nxt not in index:
                    index[nxt] = low[nxt] = counter
                    counter += 1
                    stack.append(nxt)
                    on_stack[nxt] = True
                    work.append((nxt, iter(sorted(graph[nxt]))))
                    descended = True
                    break
                if on_stack.get(nxt):
                    low[node] = min(low[node], index[nxt])
            if descended:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack[member] = False
                    component.append(member)
                    if member == node:
                        break
                if len(component) > 1:
                    found.append(sorted(component))
    return sorted(found, key=lambda c: (-len(c), c))


def owning_subpackage(module: str) -> str:
    """The area a module belongs to, at the granularity the baseline names.

    Backends are split one level deeper than the rest because they are
    separately owned and separately migrated.
    """
    parts = module.split(".")
    if parts[0] != "jittor":
        return parts[0]
    if len(parts) >= 3 and parts[1] == "backends":
        return ".".join(parts[:3])
    return ".".join(parts[:2]) if len(parts) > 1 else parts[0]


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


def check_coverage(report: dict) -> list[str]:
    """The graph is big enough that a "no violations" result means something."""
    problems = []
    for name, count in sorted(report["modules_per_root"].items()):
        if count == 0:
            problems.append(
                "package-dir root %r resolved to no modules; the layout moved "
                "under the scan or pyproject.toml is wrong" % name
            )
    if report["modules_checked"] < MIN_MODULES:
        problems.append(
            "only %d modules scanned, expected at least %d"
            % (report["modules_checked"], MIN_MODULES)
        )
    if report["import_edges"] < MIN_IMPORT_EDGES:
        problems.append(
            "only %d import edges, expected at least %d"
            % (report["import_edges"], MIN_IMPORT_EDGES)
        )
    if report["backend_modules"] < MIN_BACKEND_MODULES:
        problems.append(
            "only %d jittor.backends.* modules scanned, expected at least %d; "
            "the backends/ overlay is not being read"
            % (report["backend_modules"], MIN_BACKEND_MODULES)
        )
    return problems


def check_tools_below_framework(report: dict) -> list[str]:
    """``jittor_utils`` is below ``jittor``; importing up inverts the layers.

    This is the Python half of the three cycles named in the architecture
    audit (``jittor_utils`` <-> ``jittor.compiler``). It is closed today, and
    this is what keeps it closed.
    """
    problems = [
        "%s imports %s at import time" % (site, name)
        for site, name in sorted(report["tool_framework_imports"].items())
    ]
    deferred = report["tool_framework_deferred"]
    for site in sorted(set(deferred) - set(ALLOWED_DEFERRED_TOOL_IMPORTS)):
        problems.append(
            "%s imports %s inside a function; route the need through an "
            "injected service or add it to ALLOWED_DEFERRED_TOOL_IMPORTS "
            "with a reason" % (site, deferred[site])
        )
    for site in sorted(set(ALLOWED_DEFERRED_TOOL_IMPORTS) - set(deferred)):
        problems.append("listed exception %s is gone; drop it" % site)
    return problems


def check_cycle_surface(report: dict) -> list[str]:
    """No new import-time cycle, and no existing one growing."""
    problems = []
    new = sorted(set(report["cyclic_subpackages"]) - CYCLIC_SUBPACKAGES)
    if new:
        problems.append("these subpackages are newly on an import-time cycle: %s" % ", ".join(new))
    if report["cyclic_modules"] > MAX_CYCLIC_MODULES:
        problems.append(
            "%d modules sit on an import-time cycle, up from %d"
            % (report["cyclic_modules"], MAX_CYCLIC_MODULES)
        )
    if report["cycles"] > MAX_CYCLES:
        problems.append("%d import-time cycles, up from %d" % (report["cycles"], MAX_CYCLES))
    return problems


CONTRACTS = (
    ("coverage", check_coverage),
    ("tools-below-framework", check_tools_below_framework),
    ("cycle-surface", check_cycle_surface),
)


def build_report(root: Path = REPO_ROOT) -> dict:
    roots = scan_roots(root)
    modules = discover_modules(roots)
    edges = build_edges(modules)
    at_import = graph_for(edges, {MODULE})
    cycles = strongly_connected(at_import)
    cyclic_modules = [m for component in cycles for m in component]

    tool_imports: dict[str, str] = {}
    tool_deferred: dict[str, str] = {}
    for name, path in modules.items():
        if not name.split(".")[0] == "jittor_utils":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found: list[tuple[ast.AST, str]] = []
        _visit(tree, MODULE, found)
        for node, scope in found:
            for target in _absolute(node, name, path.name == "__init__.py"):
                if target != "jittor" and not target.startswith("jittor."):
                    continue
                site = "%s:%d" % (path.relative_to(root).as_posix(), node.lineno)
                if scope == MODULE:
                    tool_imports[site] = target
                elif scope == DEFERRED:
                    tool_deferred[site] = target
                break

    return {
        "modules_checked": len(modules),
        "modules_per_root": {
            prefix: sum(1 for m in modules if m == prefix or m.startswith(prefix + "."))
            for _base, prefix in roots
        },
        "import_edges": sum(len(v) for v in at_import.values()),
        "backend_modules": sum(1 for m in modules if m.startswith("jittor.backends.")),
        "cycles": len(cycles),
        "cycle_sizes": [len(c) for c in cycles],
        "cyclic_modules": len(cyclic_modules),
        "cyclic_subpackages": sorted({owning_subpackage(m) for m in cyclic_modules}),
        "largest_cycle_sample": cycles[0][:5] if cycles else [],
        "tool_framework_imports": tool_imports,
        "tool_framework_deferred": tool_deferred,
        "contracts": [name for name, _ in CONTRACTS],
    }


def run(root: Path = REPO_ROOT) -> tuple[dict, dict[str, list[str]]]:
    report = build_report(root)
    return report, {name: check(report) for name, check in CONTRACTS}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="print the raw report")
    args = parser.parse_args(argv)

    report, results = run()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    print(
        "scanned %d modules (%d in backends/) over %d package-dir roots, "
        "%d import-time edges"
        % (
            report["modules_checked"],
            report["backend_modules"],
            len(report["modules_per_root"]),
            report["import_edges"],
        )
    )
    print(
        "import-time cycles: %d %s, %d modules on a cycle, %d subpackages"
        % (
            report["cycles"],
            report["cycle_sizes"],
            report["cyclic_modules"],
            len(report["cyclic_subpackages"]),
        )
    )

    failures = 0
    for name, problems in results.items():
        print("%-24s %s" % (name, "FAIL (%d)" % len(problems) if problems else "ok"))
        for problem in problems:
            failures += 1
            print("    %s" % problem)
    if failures:
        print("\n%d violation(s) across %d contracts" % (failures, len(CONTRACTS)))
        return 1
    print("\n%d contracts, no violations" % len(CONTRACTS))
    return 0


if __name__ == "__main__":
    sys.exit(main())

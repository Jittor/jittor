"""Count nested def/class/lambda inside each Torch-compat installer.

The 7.03 task is "one module-level first-class object per Torch API", and the
measurable form of that is this count: an installer that still defines its APIs
inline cannot have module-level identity for them, and the objects cannot be
imported or unit-tested without running a full install first. Zero means the
installer only binds.

Run from the repo root:

    python agent/skills/torch-api-cohort-promotion/count_installer_closures.py
    python agent/skills/torch-api-cohort-promotion/count_installer_closures.py --only nn.py tensor.py
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import sys

#: Installer entry points are named ``install*`` or ``_install*`` by convention.
_INSTALLER_PREFIXES = ("install", "_install")

_NESTED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _counts(node):
    """Return (nested def/class, lambda) strictly inside ``node``."""
    defs = lambdas = 0
    for sub in ast.walk(node):
        if sub is node:
            continue
        if isinstance(sub, _NESTED):
            defs += 1
        elif isinstance(sub, ast.Lambda):
            lambdas += 1
    return defs, lambdas


def installers(path):
    """Yield ``(name, lineno, nested_defs, lambdas)`` for one source file."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith(_INSTALLER_PREFIXES):
            defs, lambdas = _counts(node)
            yield node.name, node.lineno, defs, lambdas


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default="python/jittor/compat",
        help="directory to scan (default: the compat tree)")
    parser.add_argument(
        "--only", nargs="*", default=None,
        help="restrict to these file names, e.g. nn.py tensor.py")
    parser.add_argument(
        "--min", type=int, default=0,
        help="only report installers with at least this many nested defs")
    args = parser.parse_args(argv)

    root = pathlib.Path(args.root)
    if not root.exists():
        parser.error("no such directory: %s (run from the repo root)" % root)

    rows = []
    for path in sorted(root.rglob("*.py")):
        if args.only and path.name not in args.only:
            continue
        rows.extend(
            (path, name, lineno, defs, lambdas)
            for name, lineno, defs, lambdas in installers(path))

    rows.sort(key=lambda row: (-row[3], str(row[0]), row[1]))
    total = 0
    for path, name, lineno, defs, lambdas in rows:
        if defs < args.min:
            continue
        total += defs
        mark = "CLEARED" if defs == 0 and lambdas == 0 else ""
        print("%-22s %-40s line %-6d nested=%-4d lambda=%-4d %s"
              % (path.name, name, lineno, defs, lambdas, mark))
    print("--- %d installers, %d nested def/class total" % (len(rows), total))
    return 0


if __name__ == "__main__":
    sys.exit(main())

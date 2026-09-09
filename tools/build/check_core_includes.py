#!/usr/bin/env python3
"""Resolve every ``#include "..."`` in the C++ core against the include roots.

Why this exists: the layout tasks (`4.15`, `1.05`, `2.23`, `3.24`) move files
inside the core, which invalidates the quoted include paths. Finding those the
usual way costs a full core rebuild per attempt -- 178 translation units, five
to nine minutes -- and the build stops at the first error, so each rebuild
reveals exactly one broken thing. Four rebuilds in a row is how `4.15`'s src
move was landed.

This does the same check statically in under a second, so a batch of moves can
be made and checked between steps without paying for a rebuild each time.

It is deliberately *not* a compiler: it does not preprocess, evaluate
``#if``, or follow angle-bracket includes (those are system or SDK headers).
It answers one question -- for every quoted include in the core, does the named
file exist under one of the roots the build actually passes with ``-I``? That
is the question a move breaks.

It cannot model every ``-I`` the build passes (each backend adds its own SDK
and per-library include directories), so on a tree that compiles it still
reports a couple of hundred includes it cannot place. That is why the useful
mode is **differential**: record the set on a known-good tree, then after a
move assert the set did not grow. A new entry is a real break; the standing
ones are this script's blind spots, not the build's.

Usage:
    check_core_includes.py --baseline FILE [repo]   record the current set
    check_core_includes.py --check FILE [repo]      fail only on new entries
    check_core_includes.py [repo]                   just print the set
"""

import os
import re
import sys

INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)

#: The roots the build passes with -I, in the order the compiler sees them.
#: Keep in step with compiler.py's cc_flags; a root missing here makes this
#: script report breakage that the build would not have, and a root here that
#: the build does not pass makes it miss real breakage.
ROOT_CANDIDATES = (
    "src",
    "backends/cuda",
    "backends/cuda/include",
)

SOURCE_SUFFIXES = (".cc", ".cu", ".cuh", ".h", ".hpp", ".cpp")

#: Trees whose includes this script checks. Generated headers under the build
#: cache are not here: they do not exist in a clean checkout.
SCAN_TREES = ("src", "backends")


def _roots(repo):
    found = [os.path.join(repo, r) for r in ROOT_CANDIDATES]
    return [r for r in found if os.path.isdir(r)]


def _sources(repo):
    for tree in SCAN_TREES:
        base = os.path.join(repo, tree)
        if not os.path.isdir(base):
            continue
        for dirpath, _, names in os.walk(base):
            if "__pycache__" in dirpath or "third_party" in dirpath:
                continue
            for name in names:
                if name.endswith(SOURCE_SUFFIXES):
                    yield os.path.join(dirpath, name)


def main(argv):
    mode = baseline_path = None
    args = list(argv[1:])
    if args and args[0] in ("--baseline", "--check"):
        mode, baseline_path = args[0], args[1]
        args = args[2:]
    repo = os.path.abspath(args[0] if args else ".")
    roots = _roots(repo)
    if not roots:
        print("no include roots exist under %s -- wrong repo root?" % repo)
        return 1

    files = list(_sources(repo))
    if len(files) < 100:
        print("only %d core sources found under %s; the scan trees are stale, "
              "which would make every check below pass for the wrong reason"
              % (len(files), repo))
        return 1

    unresolved = []
    checked = 0
    for path in files:
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError as exc:
            unresolved.append((path, "<unreadable>", str(exc)))
            continue
        here = os.path.dirname(path)
        for name in INCLUDE.findall(text):
            checked += 1
            # A quoted include resolves relative to the including file first,
            # then against each -I root, which is what the compiler does.
            if os.path.isfile(os.path.join(here, name)):
                continue
            if any(os.path.isfile(os.path.join(root, name)) for root in roots):
                continue
            unresolved.append((os.path.relpath(path, repo), name, ""))

    print("roots: %s" % ", ".join(os.path.relpath(r, repo) for r in roots))
    print("%d files, %d quoted includes" % (len(files), checked))

    current = sorted("%s -> %s" % (path, name) for path, name, _ in unresolved)
    if mode == "--baseline":
        open(baseline_path, "w").write("\n".join(current) + "\n")
        print("baseline written: %d unresolved (this script's blind spots)"
              % len(current))
        return 0
    if mode == "--check":
        previous = set(open(baseline_path).read().split("\n")) - {""}
        new = [line for line in current if line not in previous]
        gone = sorted(previous - set(current))
        print("baseline %d, now %d" % (len(previous), len(current)))
        if gone:
            print("%d no longer unresolved (fine, but says the tree moved):" % len(gone))
            for line in gone[:20]:
                print("  - %s" % line)
        if new:
            print("%d NEW unresolved -- a move broke these:" % len(new))
            for line in new[:60]:
                print("  + %s" % line)
            return 1
        print("no new unresolved includes")
        return 0
    if unresolved:
        print("%d unresolved:" % len(unresolved))
        for path, name, note in unresolved[:60]:
            print("  %-58s -> %s %s" % (path, name, note))
        if len(unresolved) > 60:
            print("  ... and %d more" % (len(unresolved) - 60))
        return 1
    print("all quoted includes resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

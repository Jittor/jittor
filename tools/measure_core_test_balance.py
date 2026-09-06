#!/usr/bin/env python3
"""Measure what the core test budget is actually spent on.

The audit entry this serves (``07-architecture.md`` §代码规模, "测试分布与风险倒挂")
compares two line counts: core C++ against ``tests/core``. Line counts are free to
read, which is why that comparison got written -- and ``gate-tier-budget`` §2.1
already established that lines are not seconds. This script exists because lines
are not *coverage* either, and the second mistake is the more expensive one: it
makes "write more test lines" look like an answer.

So it reports three separate things and never adds them up:

``sizes``
    Line counts per layer, so the audit's numbers can be refreshed instead of
    quoted. Vendored code (``src/third_party``) and the C++ unit tests
    (``src/tests``) are subtracted from "core" because neither is core the
    refactor has to keep working.

``subjects``
    Every ``tests/core`` file bucketed by the *subject it asserts on*, decided by
    which APIs it names -- not by its directory. This is the finding the ratio
    hides: a file can live in ``tests/core`` and asserts nothing about the core
    graph. The buckets are deliberately coarse and overlapping is reported rather
    than resolved, so a file counted in two buckets is visible as such.

``untested_headers``
    Core headers that no file under ``tests/`` names, by header or by any symbol
    it declares at namespace scope. This is a lower bound on what is untested and
    an over-estimate of what is tested: naming a symbol is not asserting on it.
    Read it as "these are certainly not covered", never as "the rest is".

Usage::

    python tools/measure_core_test_balance.py            # human-readable
    python tools/measure_core_test_balance.py --json      # for a structure test
"""

from __future__ import print_function

import argparse
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

CXX_SUFFIXES = (".cc", ".h", ".cu", ".cuh")

#: Trees excluded from "core": vendored third-party code and the C++ unit tests.
#: Both are inside ``src/`` and neither is the runtime under refactor.
CORE_EXCLUDED = ("src/third_party", "src/tests")

#: ``(bucket, regex)`` -- what a ``tests/core`` file is *about*.
#:
#: Decided by the APIs a file names, because that is checkable; a name-based
#: guess ("test_complex64_*" is about dtypes) silently misfiles anything renamed.
#: Order is irrelevant: a file joins every bucket it matches, and files matching
#: two or more are reported under ``multi`` so the overlap is not hidden.
SUBJECTS = (
    # the graph itself: liveness counters, the node registry, the dangling sweep
    ("graph_and_liveness", re.compile(
        r"\bliveness_info\b|\bgraph_check\b|\bdump_all_graphs\b|\bclean_graph\b"
        r"|\bcheck_graph\b|\blived_vars\b|\blived_ops\b|\bhold_vars\b"
        r"|\bnumber_of_lived")),
    # the executor: what runs, in what order, fused how
    ("executor", re.compile(
        r"\bexec_plan\b|\brun_sync\b|\bfused_op\b|\bfuse_ops\b|\bgopt_disable\b"
        r"|\bno_fuse\b|\bexecutor\b|\bprofiler\b|\bjit_key\b")),
    # autograd: the tape, the backward pass, leaf bookkeeping
    ("autograd", re.compile(
        r"\bjt\.grad\b|\bbackward\(|\bgrad_optional\b|\bstop_grad\b"
        r"|\bstop_fuse\b|\brequires_grad\b|\bFunction\b|\bgrad_hooker\b")),
    # dtype and numerics: promotion, complex, distributions, elementwise results
    ("dtype_and_numerics", re.compile(
        r"\bdtype\b|\bcomplex64\b|\bpromot|\bfloat16\b|\bbfloat16\b|\bfinfo\b"
        r"|\bnp\.testing\b|\ballclose\b|\bassert_array")),
    # the Python binding layer and its generator
    ("binding", re.compile(
        r"\bpyjt\b|\b_C\b|\bcore_api\b|\bNanoVector\b|\bNanoString\b"
        r"|\b__dict__\b|\bgetattr\(jt\b")),
    # process-level concerns: import, init, flags, teardown, signals
    ("process_and_build", re.compile(
        r"\bimport_side_effects\b|\bjt\.flags\b|\bcompile_extern\b|\bjt\.compiler\b"
        r"|\bbootstrap\b|\batexit\b|\bsignal\b|\bsubprocess\b|\brun_child")),
)


def _lines(path):
    try:
        with open(path, "rb") as handle:
            return handle.read().decode("utf8", "replace").count("\n")
    except OSError:
        return 0


def _tree(root, suffixes, excluded=()):
    """Files under ``root`` with ``suffixes``, skipping ``excluded`` subtrees."""
    found = []
    if not root.exists():
        return found
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        relative = path.relative_to(REPO_ROOT).as_posix()
        if any(("/" + item + "/") in ("/" + relative) or
               relative.startswith(item + "/") for item in excluded):
            continue
        found.append(path)
    return found


def measure_sizes():
    src = REPO_ROOT / "python/jittor/src"
    core = _tree(src, CXX_SUFFIXES, CORE_EXCLUDED)
    layers = {
        "core_cxx": core,
        "core_cxx_vendored": _tree(src / "third_party", CXX_SUFFIXES),
        "core_cxx_unit_tests": _tree(src / "tests", CXX_SUFFIXES),
        # 4.10 moved the accelerator kernels to a top-level ``backends/``; the
        # legacy per-library trees under ``python/jittor/extern`` still exist.
        "backends_cxx": _tree(REPO_ROOT / "backends", CXX_SUFFIXES),
        "extern_cxx": _tree(REPO_ROOT / "python/jittor/extern", CXX_SUFFIXES),
        "tests_core": _tree(REPO_ROOT / "tests/core", (".py",)),
        "tests_structure": _tree(REPO_ROOT / "tests/structure", (".py",)),
        "tests_compat": _tree(REPO_ROOT / "tests/compat", (".py",)),
        "tests_all": _tree(REPO_ROOT / "tests", (".py",)),
    }
    return {
        name: {"files": len(paths), "lines": sum(_lines(path) for path in paths)}
        for name, paths in layers.items()
    }


def measure_subjects():
    cases = re.compile(r"^\s*def (test\w*)", re.M)
    files = _tree(REPO_ROOT / "tests/core", (".py",))
    per_file = {}
    for path in files:
        text = path.read_text("utf8", "replace")
        buckets = tuple(
            name for name, pattern in SUBJECTS if pattern.search(text)
        )
        per_file[path.relative_to(REPO_ROOT).as_posix()] = {
            "lines": text.count("\n"),
            "cases": len(cases.findall(text)),
            "subjects": buckets,
        }
    totals = {}
    for name, _pattern in SUBJECTS:
        owned = [item for item in per_file.values() if name in item["subjects"]]
        totals[name] = {
            "files": len(owned),
            "lines": sum(item["lines"] for item in owned),
            "cases": sum(item["cases"] for item in owned),
        }
    unclassified = [
        path for path, item in per_file.items() if not item["subjects"]
    ]
    multi = [
        path for path, item in per_file.items() if len(item["subjects"]) > 1
    ]
    return {
        "per_file": per_file,
        "totals": totals,
        "unclassified": sorted(unclassified),
        "multi": sorted(multi),
    }


#: Namespace-scope declarations in a core header, as ``gen_jit_flags`` and
#: ``pyjt_compiler`` already scan them: a return type then a name then ``(``.
#: Deliberately loose -- a false symbol only makes a header look *more* covered,
#: which keeps ``untested_headers`` a lower bound.
_DECLARATION = re.compile(
    r"^(?:[A-Za-z_][\w:<>,*& ]*?)\b([a-z_]\w{3,})\s*\(", re.M)


def _strip_cxx_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def measure_headers():
    src = REPO_ROOT / "python/jittor/src"
    headers = _tree(src, (".h",), CORE_EXCLUDED)
    test_text = []
    for path in _tree(REPO_ROOT / "tests", (".py",)):
        test_text.append(path.read_text("utf8", "replace"))
    for path in _tree(src / "tests", (".cc", ".h")):
        test_text.append(path.read_text("utf8", "replace"))
    corpus = "\n".join(test_text)
    named, unnamed = [], []
    for path in headers:
        relative = path.relative_to(REPO_ROOT).as_posix()
        stem = path.stem
        body = _strip_cxx_comments(path.read_text("utf8", "replace"))
        symbols = set(_DECLARATION.findall(body))
        hit = stem in corpus or any(
            symbol in corpus for symbol in symbols if len(symbol) > 4
        )
        (named if hit else unnamed).append(relative)
    return {"named": sorted(named), "unnamed": sorted(unnamed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = {
        "sizes": measure_sizes(),
        "subjects": measure_subjects(),
        "headers": measure_headers(),
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    sizes = report["sizes"]
    print("== sizes ==")
    for name in sorted(sizes):
        item = sizes[name]
        print("  %-22s %4d files %7d lines" % (name, item["files"], item["lines"]))
    core = sizes["core_cxx"]["lines"]
    tests_core = sizes["tests_core"]["lines"]
    print("  tests/core : core_cxx = %.2f" % (tests_core / float(core)))

    print("== tests/core subjects (a file joins every bucket it matches) ==")
    for name, item in sorted(report["subjects"]["totals"].items(),
                             key=lambda kv: -kv[1]["lines"]):
        print("  %-22s %4d files %7d lines %5d cases"
              % (name, item["files"], item["lines"], item["cases"]))
    print("  unclassified: %d" % len(report["subjects"]["unclassified"]))
    for path in report["subjects"]["unclassified"]:
        print("      " + path)

    headers = report["headers"]
    total = len(headers["named"]) + len(headers["unnamed"])
    print("== core headers no test names (lower bound on untested) ==")
    print("  %d of %d" % (len(headers["unnamed"]), total))
    for path in headers["unnamed"]:
        print("      " + path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

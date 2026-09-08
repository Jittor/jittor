"""The import-direction gate, and proof that the gate itself can fail.

``tools/lint/check_import_layering.py`` is the checker; this is one of the two
places that runs it (the other is ``nox -s imports``). It is a plain pytest
module with no optional dependency, so it cannot degrade into a skip: a lint
rule that is only enforced when some tool happens to be installed is a rule
that reports "clean" on the machine that has not installed it.

Two things are asserted, and the second matters as much as the first:

1. the contracts hold on this tree, and
2. the contracts *reject* a tree that violates them.

(2) is here because the way a layout check usually breaks is by scanning
nothing. ``jittor.backends.<x>`` lives in the top-level ``backends/`` directory
and is spliced onto ``python/jittor/backends`` at runtime, so a scanner rooted
naively at ``python/`` reads 86 fewer modules and still reports a healthy
total. The counter-example tests below drive each contract with a report that
should fail, so a checker that has quietly stopped looking at anything is
itself a test failure.
"""

import copy
import functools
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "tools" / "lint" / "check_import_layering.py"


@functools.lru_cache(maxsize=None)
def _checker():
    spec = importlib.util.spec_from_file_location("jittor_check_import_layering", CHECKER)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(REPO_ROOT / "tools"))
    return module


@functools.lru_cache(maxsize=None)
def _cached_report():
    """Parsing 384 modules once is enough for the whole file."""
    module = _checker()
    return module, module.build_report(REPO_ROOT)


def _report():
    module, report = _cached_report()
    return module, copy.deepcopy(report)


# --------------------------------------------------------------------------
# The gate actually looked at something.
# --------------------------------------------------------------------------


def test_every_declared_package_root_resolves_to_modules():
    """A ``package-dir`` entry that maps to nothing must fail, not pass."""
    module, report = _report()
    empty = sorted(name for name, count in report["modules_per_root"].items() if not count)
    assert empty == [], (
        "these package-dir roots resolved to no modules, which would make "
        "every contract below pass over an empty graph: %s" % empty
    )
    assert len(report["modules_per_root"]) >= 4, report["modules_per_root"]


def test_the_backends_overlay_is_inside_the_scan():
    """The half-migrated backends live outside ``python/``; scan them anyway.

    ``pyproject.toml`` maps ``jittor.backends.<x>`` to the top-level
    ``backends/<x>``. Those modules import ``jittor`` at module scope and sit
    on the largest cycle in the tree, so a scan that misses them under-reports
    the cycle surface while looking entirely healthy.
    """
    module, report = _report()
    assert report["backend_modules"] >= module.MIN_BACKEND_MODULES, report["backend_modules"]
    per_root = report["modules_per_root"]
    for backend in ("jittor.backends.acl", "jittor.backends.cuda", "jittor.backends.rocm"):
        assert per_root.get(backend, 0) > 0, (backend, per_root)


def test_native_source_namespace_is_explicitly_classified_as_data():
    module, report = _report()
    assert "jittor.src" not in report["modules_per_root"]
    source = report["data_only_roots"]["jittor.src"]
    assert source["resource_files"] > 0 and source["python_files"] == 0
    assert source["reason"]
    source["python_files"] = 1
    assert any("unexpectedly contains Python" in error for error in module.check_coverage(report))
    source["python_files"] = 0
    source["resource_files"] = 0
    assert any("no declared resources" in error for error in module.check_coverage(report))


def test_the_graph_is_large_enough_for_a_clean_result_to_mean_something():
    module, report = _report()
    assert report["modules_checked"] >= module.MIN_MODULES, report["modules_checked"]
    assert report["import_edges"] >= module.MIN_IMPORT_EDGES, report["import_edges"]
    assert len(report["contracts"]) >= 3, report["contracts"]


# --------------------------------------------------------------------------
# The contracts hold.
# --------------------------------------------------------------------------


def test_all_import_contracts_hold():
    module = _checker()
    _report_, results = module.run(REPO_ROOT)
    failures = {name: problems for name, problems in results.items() if problems}
    assert failures == {}, "\n".join(
        "%s: %s" % (name, problem)
        for name, problems in sorted(failures.items())
        for problem in problems
    )


def test_the_build_tools_stay_below_the_framework():
    """Audited cycle 1 of 3: ``jittor_utils`` <-> ``jittor.compiler``.

    Closed as measured on this tree -- there is no framework import in
    ``jittor_utils`` at module scope *or* inside a function. This is the lock
    that keeps it closed.
    """
    _module, report = _report()
    assert report["tool_framework_imports"] == {}, report["tool_framework_imports"]
    assert report["tool_framework_deferred"] == {}, report["tool_framework_deferred"]


def test_the_cycle_surface_does_not_grow():
    module, report = _report()
    assert set(report["cyclic_subpackages"]) <= module.CYCLIC_SUBPACKAGES, sorted(
        set(report["cyclic_subpackages"]) - module.CYCLIC_SUBPACKAGES
    )
    assert report["cyclic_modules"] <= module.MAX_CYCLIC_MODULES, report["cyclic_modules"]
    assert report["cycles"] <= module.MAX_CYCLES, report["cycle_sizes"]


# --------------------------------------------------------------------------
# The contracts reject what they are supposed to reject.
# --------------------------------------------------------------------------


def _baseline_report():
    """A private copy of the real report, safe to corrupt on purpose."""
    _module, report = _report()
    return report


def test_a_tools_layer_import_of_the_framework_is_reported():
    """Reintroduce audited cycle 1 in the report and require a violation."""
    module = _checker()
    report = _baseline_report()
    report["tool_framework_imports"] = {"python/jittor_utils/misc.py:12": "jittor.compiler"}
    problems = module.check_tools_below_framework(report)
    assert problems, "reintroducing the jittor_utils -> jittor import was not reported"
    assert any("jittor_utils/misc.py:12" in p and "jittor.compiler" in p for p in problems), (
        problems
    )


def test_a_deferred_tools_layer_import_is_reported():
    module = _checker()
    report = _baseline_report()
    report["tool_framework_deferred"] = {"python/jittor_utils/lock.py:40": "jittor"}
    problems = module.check_tools_below_framework(report)
    assert any("jittor_utils/lock.py:40" in p for p in problems), problems


def test_a_new_subpackage_on_a_cycle_is_reported():
    module = _checker()
    report = _baseline_report()
    report["cyclic_subpackages"] = sorted(
        set(report["cyclic_subpackages"]) | {"jittor.serialization"}
    )
    problems = module.check_cycle_surface(report)
    assert any("jittor.serialization" in p for p in problems), problems


def test_a_growing_cycle_is_reported():
    module = _checker()
    report = _baseline_report()
    report["cyclic_modules"] = module.MAX_CYCLIC_MODULES + 1
    problems = module.check_cycle_surface(report)
    assert any("import-time cycle" in p for p in problems), problems


def test_an_extra_cycle_is_reported():
    module = _checker()
    report = _baseline_report()
    report["cycles"] = module.MAX_CYCLES + 1
    problems = module.check_cycle_surface(report)
    assert any("import-time cycles" in p for p in problems), problems


def test_a_scan_that_found_nothing_is_reported():
    """The empty-set failure: a root that stops resolving must be loud."""
    module = _checker()
    report = _baseline_report()
    report["modules_per_root"] = dict(report["modules_per_root"])
    report["modules_per_root"]["jittor.backends.cuda"] = 0
    problems = module.check_coverage(report)
    assert any("jittor.backends.cuda" in p for p in problems), problems


def test_losing_the_backends_overlay_is_reported():
    """Exactly the regression that made an earlier contract test vacuous."""
    module = _checker()
    report = _baseline_report()
    report["backend_modules"] = 1
    problems = module.check_coverage(report)
    assert any("backends/ overlay" in p for p in problems), problems


def test_a_shrunken_graph_is_reported():
    module = _checker()
    report = _baseline_report()
    report["modules_checked"] = 10
    report["import_edges"] = 5
    problems = module.check_coverage(report)
    assert len(problems) >= 2, problems

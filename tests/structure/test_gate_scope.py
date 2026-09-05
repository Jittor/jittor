"""The gate's scope is the tree minus stated exceptions, and it stays that way.

Before this, the gate was a hand-written list of paths in ``noxfile.py``.
Nothing tied it to the tree, so it drifted until 98 of 332 test files were
reachable from any workflow and 234 were reachable from none -- tests that were
written, reviewed, merged, and then never run again. The drift was invisible
because a shrinking gate looks exactly like a passing one.

These tests make the scope falsifiable: the reachable share is asserted, every
exclusion has to justify itself, and the runner used locally has to select the
same files as the gate.
"""

import ast
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = REPO_ROOT / "tests"

if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from _helpers import gate_scope  # noqa: E402


def _all_test_files():
    return {
        path.relative_to(REPO_ROOT).as_posix()
        for path in TEST_ROOT.rglob("test_*.py")
    }


def _gate_files():
    native = gate_scope.selected_files(REPO_ROOT, gate_scope.native_arguments())
    torch = gate_scope.selected_files(REPO_ROOT, gate_scope.torch_arguments())
    return native | torch


def test_the_cpu_gate_reaches_every_test_file_it_does_not_exclude():
    expected = _all_test_files() - set(gate_scope.excluded_paths())
    missing = sorted(expected - _gate_files())
    assert missing == [], (
        "these test files are in the tree but no CPU gate would run them:\n"
        + "\n".join(missing)
    )


def test_core_property_tests_are_owned_by_the_cpu_gate():
    """10.18: graph, liveness, and executor properties stay in both modes.

    The CPU gate is intentionally tree-based, so a future exclusion or mode
    split must not silently remove the tests that protect the core runtime.
    Keep this contract focused on representative property suites rather than
    freezing the entire ``tests/core`` directory.
    """
    required = {
        "tests/core/test_traversal_state_isolation.py",
        "tests/core/test_pyjt_binding_protocol.py",
        "tests/core/test_autograd_engine.py",
    }
    native = gate_scope.selected_files(REPO_ROOT, gate_scope.native_arguments())
    torch = gate_scope.selected_files(REPO_ROOT, gate_scope.torch_arguments())
    assert required <= native, "native CPU gate lost core property tests"
    assert required <= (native | torch), "CPU gate lost core property tests"


def test_the_two_sessions_do_not_run_the_same_file_twice():
    """Torch mode is a split, not an overlap: each file has exactly one owner."""
    native = gate_scope.selected_files(REPO_ROOT, gate_scope.native_arguments())
    torch = gate_scope.selected_files(REPO_ROOT, gate_scope.torch_arguments())
    assert sorted(native & torch) == []


def test_every_exclusion_states_a_reason_and_still_exists():
    problems = []
    for entry in gate_scope.EXCLUDED:
        path, reason = entry
        if not reason or not reason.strip():
            problems.append("%s is excluded with no reason" % path)
        if not (REPO_ROOT / path).exists():
            problems.append("%s is excluded but no longer exists" % path)
    assert problems == [], "\n".join(problems)


def test_the_gate_covers_almost_the_whole_tree():
    """A floor, so a future exclusion spree has to argue with a number.

    The audit measured 74 of 289 files reachable. The point of the rewrite is
    that the ratio cannot quietly fall back.
    """
    total = len(_all_test_files())
    reached = len(_gate_files())
    assert total >= 280, total
    assert reached >= total - 5, (
        "gate reaches %d of %d test files; excluding this many needs a much "
        "better reason than any recorded in gate_scope.EXCLUDED"
        % (reached, total)
    )


def test_the_noxfile_has_no_hand_written_test_whitelist_for_cpu():
    """The CPU gate must not grow a second, silently-drifting source of truth."""
    source = (REPO_ROOT / "noxfile.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="noxfile.py")
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            assert getattr(target, "id", None) != "CPU_TESTS", (
                "CPU_TESTS is back; the CPU gate selects by exclusion from "
                "tests/_helpers/gate_scope.py"
            )


def test_the_local_runner_and_the_gate_select_the_same_files():
    """``tools/run_test_suite.py`` is what a developer runs before pushing."""
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "jittor_run_test_suite", REPO_ROOT / "tools" / "run_test_suite.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(REPO_ROOT / "tools"))
    assert module._session_arguments("native") == list(gate_scope.native_arguments())
    assert module._session_arguments("torch") == list(gate_scope.torch_arguments())


def test_local_runner_matches_gate_execution_contract():
    """The CLI must fail closed and use nox's smoke distribution policy."""
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "jittor_run_test_suite_execution_contract",
            REPO_ROOT / "tools" / "run_test_suite.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(REPO_ROOT / "tools"))
    assert module._session_environment("native")[
        "JITTOR_TEST_REQUIRE_EXECUTION"] == "1"
    assert module._parallel_arguments(2, distribution="loadgroup") == [
        "-n", "2", "--dist", "loadgroup"]


def test_standalone_runner_uses_runtime_worker_policy_when_jobs_omitted(monkeypatch):
    """The local command and nox must start the same effective worker count."""
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "jittor_run_test_suite_worker_policy",
            REPO_ROOT / "tools" / "run_test_suite.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(REPO_ROOT / "tools"))
    monkeypatch.setenv("JITTOR_GATE_WORKERS", "4")
    from _helpers.tiers import effective_cpu_count, runtime_workers
    assert module._runtime_jobs(None) == runtime_workers(
        4, available=effective_cpu_count())
    assert module._runtime_jobs(0) == 0


def test_default_nox_sessions_include_the_cpu_numeric_gate():
    """A default green nox run must exercise numerical CPU behavior (10.02)."""
    source = (REPO_ROOT / "noxfile.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="noxfile.py")
    sessions = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(getattr(target, "id", None) == "sessions"
               or getattr(target, "attr", None) == "sessions"
               for target in node.targets):
            sessions = ast.literal_eval(node.value)
            break
    assert sessions is not None
    assert "cpu" in sessions


def test_full_nox_session_is_the_periodic_complete_gate():
    """The scheduled CI entry point must target the full CPU session (10.01)."""
    source = (REPO_ROOT / "noxfile.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="noxfile.py")
    full = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "full"
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "cpu"
        for node in ast.walk(full)
    )
    workflow = (REPO_ROOT / ".github" / "workflows" / "cpu.yml").read_text(
        encoding="utf-8"
    )
    assert "python -m nox -s full" in workflow


def test_skip_reason_buckets_are_stable_and_other_is_counted(monkeypatch):
    """10.05: accepted environment reasons are separated from unexplained ones."""
    import conftest as policy

    previous = policy._SKIP_REASON_BUCKETS.copy()
    try:
        policy._SKIP_REASON_BUCKETS.clear()
        policy._SKIP_REASON_BUCKETS.update({
            policy.classify_skip_reason_bucket("CUDA backend missing"): 2,
            policy.classify_skip_reason_bucket("mystery skip"): 1,
        })
        buckets = dict(policy._skip_reason_buckets())
        assert buckets["accelerator"] == 2
        assert buckets["other"] == 1
        assert policy.classify_skip_reason_bucket("torch download") == "torch"
        assert policy.classify_skip_reason_bucket("manual backend") == "backend"
        assert policy._other_skip_count() == 1
    finally:
        policy._SKIP_REASON_BUCKETS.clear()
        policy._SKIP_REASON_BUCKETS.update(previous)


def test_skip_reason_summary_and_threshold_are_wired():
    """10.05: CI summary prints buckets and the execution gate rejects other>0."""
    source = (REPO_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert "_report_skip_reason_buckets(terminalreporter)" in source
    assert "other skipped:" in source
    assert "_other_skip_count() > 0" in source

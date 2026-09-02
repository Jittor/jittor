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

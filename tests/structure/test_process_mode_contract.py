"""Which process mode a test session runs in is stated, not inferred.

Torch compatibility mode is process-global: it changes lazy execution,
reduction defaults and what a gradient means for everything in the interpreter.
It used to be selected by reading ``sys.argv`` -- if the selection mentioned a
Torch-mode path, the *whole process* switched. So naming one more directory on
the command line changed the semantics of every other directory named with it,
and the same test passed or failed depending on how it was invoked. ``-k``,
xdist workers and IDE runners each produced a different answer from the same
source, and the CPU gate's ``--collect-only tests`` never even checked that the
62 Torch-mode files still imported.

Now ``JITTOR_TORCH_SHIM`` decides and nothing else does. The command line is
read in exactly one place and for exactly one purpose: to say what is wrong
when a native session is pointed at Torch-mode paths, instead of importing them
natively and failing on a module-scope import somewhere unrelated.
"""

import ast
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = REPO_ROOT / "tests"

if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from _helpers.child_process import run_python_child  # noqa: E402

#: A native file, and a Torch-mode file that is cheap to collect.
_NATIVE_TARGET = "tests/core/test_flags.py"
_TORCH_TARGET = "tests/compat/vllm"


def _collect(targets, torch_mode=None):
    environment = {"JITTOR_TEST_DEVICES": "cpu", "nvcc_path": ""}
    if torch_mode is not None:
        environment["JITTOR_TORCH_SHIM"] = "1" if torch_mode else "0"
    else:
        environment["JITTOR_TORCH_SHIM"] = ""
    return run_python_child(
        ["-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"]
        + list(targets),
        cwd=REPO_ROOT, env=environment, merge_stderr=True, timeout=900)


def test_conftest_does_not_read_the_command_line_to_choose_a_mode():
    """The regression this file exists for, caught statically and for free."""
    source = (TEST_ROOT / "conftest.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="conftest.py")
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "argv":
            if isinstance(node.value, ast.Name) and node.value.id == "sys":
                offenders.append(node.lineno)
    assert offenders == [], (
        "conftest.py reads sys.argv at line(s) %s; the process mode comes from "
        "JITTOR_TORCH_SHIM" % offenders)


def test_naming_a_torch_path_alongside_a_native_one_does_not_change_its_meaning():
    """Adding a directory must not silently reinterpret the others.

    Before: this selection set ``JITTOR_TORCH_SHIM=1`` for the whole process, so
    ``tests/core/test_flags.py`` -- a native file nobody asked to change -- ran
    with the shim installed. After: the session refuses and says which variable
    to set, so the two meanings can never be produced by the same command.
    """
    alone = _collect([_NATIVE_TARGET])
    assert alone.returncode == 0, alone.stdout[-3000:]
    assert _NATIVE_TARGET in alone.stdout

    together = _collect([_NATIVE_TARGET, _TORCH_TARGET])
    assert together.returncode != 0, (
        "a native selection silently absorbed a Torch-mode path:\n"
        + together.stdout[-3000:])
    assert "JITTOR_TORCH_SHIM" in together.stdout, together.stdout[-3000:]


def test_the_same_selection_is_collectable_once_the_mode_is_stated():
    """The error is a missing decision, not a ban."""
    stated = _collect([_NATIVE_TARGET, _TORCH_TARGET], torch_mode=True)
    assert stated.returncode == 0, stated.stdout[-3000:]
    assert _TORCH_TARGET in stated.stdout, stated.stdout[-3000:]


def test_manual_probes_are_opt_in_by_variable_not_by_selection_shape():
    """``JITTOR_TEST_MANUAL``, not "did you name a whole directory".

    The old rule read the shape of the selection, and applied it *before* some
    of the markers were attached: ``test_notebooks.py`` got its ``manual``
    marker after the skip decision had already been made, so a whole-tree run
    executed it anyway -- 537 seconds, the slowest item in the suite.
    """
    source = (TEST_ROOT / "conftest.py").read_text(encoding="utf-8")
    assert "SELECTION_IS_BROAD" not in source
    assert "JITTOR_TEST_MANUAL" in source
    tree = ast.parse(source, filename="conftest.py")
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "pytest_collection_modifyitems")
    lines = ast.unparse(function).splitlines()
    marked = next(i for i, line in enumerate(lines)
                  if "pytest.mark.manual" in line)
    decided = next(i for i, line in enumerate(lines)
                   if "manual_enabled" in line and "not " in line)
    assert marked < decided, (
        "the manual marker is attached after the decision that reads it; that "
        "ordering is why test_notebooks.py was never actually deselected")


if __name__ == "__main__":
    print(os.environ.get("JITTOR_TORCH_SHIM"))

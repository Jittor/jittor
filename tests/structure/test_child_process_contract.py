"""A test-launched child process must import the tree under test.

``tests/conftest.py`` puts the checkout this session tests on its own
``sys.path`` and does not export ``PYTHONPATH``. A bare
``subprocess.run([sys.executable, ...])`` therefore hands the child whatever
``jittor`` the interpreter's environment resolves -- in a development checkout,
an editable install pointing at a *different* working tree.

The quiet failure is the expensive one: the child usually still works, the test
passes, and it proved nothing about the code under test. The loud version landed
while this branch was being built. ``[0.08]`` renamed a core symbol
(``set_lock_path`` to ``set_lock_fd``); ``test_tracer``'s child imported the main
tree's ``compiler.py`` -- the old name -- while loading the core this branch had
just built, and died with an ``AttributeError`` that pointed at neither tree.

So the rule below is mechanical rather than heuristic: nothing under ``tests/``
names the interpreter itself, and anything that launches it pins ``PYTHONPATH``
through ``_helpers.child_process``. An earlier version of this check only fired
when the child's *source* mentioned "jittor", which missed every launch that
passed the source in a variable -- that is, most of them.
"""

import ast
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = REPO_ROOT / "tests"

#: The one module allowed to name the interpreter and to launch it directly.
HELPER = TEST_ROOT / "_helpers" / "child_process.py"

_LAUNCHERS = {
    "run", "Popen", "call", "check_call", "check_output", "getoutput",
    "getstatusoutput", "system", "popen", "spawnv", "spawnvp", "execv", "execvp",
}
_LAUNCHER_MODULES = {"subprocess", "sp", "os"}

#: Names that resolve to the interpreter a child must run.
_INTERPRETER_NAMES = {"PYTHON", "python_executable"}


#: The suite runner lives outside tests/ but starts the same children, and it
#: got this wrong in the way that matters: its warm-up compiled the main tree
#: into the session's cache, so every pytest run that followed compiled the
#: branch a second time.
_ALSO_SCANNED = (REPO_ROOT / "tools" / "run_test_suite.py",)


def _python_files():
    return sorted(TEST_ROOT.rglob("*.py")) + [
        path for path in _ALSO_SCANNED if path.is_file()]


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _is_launcher(node):
    """Whether this call starts a process."""
    name = _dotted(node.func)
    leaf = name.rsplit(".", 1)[-1]
    if leaf not in _LAUNCHERS:
        return False
    if "." in name and name.split(".")[0] not in _LAUNCHER_MODULES:
        return False
    return True


def _names_the_interpreter(node):
    """Whether this subtree reaches the interpreter or an MPI launcher."""
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and _dotted(child) == "sys.executable":
            return True
        if isinstance(child, ast.Name) and child.id in _INTERPRETER_NAMES:
            return True
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            if "mpirun" in child.value:
                return True
    return False


def _pins_pythonpath(node):
    """Whether the call hands the child an environment from ``child_env``."""
    for keyword in node.keywords:
        if keyword.arg != "env":
            continue
        for child in ast.walk(keyword.value):
            if isinstance(child, ast.Name) and child.id == "child_env":
                return True
            if isinstance(child, ast.Attribute) and child.attr == "child_env":
                return True
    return False


def test_no_test_names_the_interpreter_directly():
    """``sys.executable`` is spelled ``child_process.PYTHON`` under ``tests/``.

    One name, one place, so the launch check below cannot be defeated by
    assigning the interpreter to a variable first.
    """
    offenders = []
    for path in _python_files():
        if path == HELPER:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and _dotted(node) == "sys.executable":
                offenders.append(
                    "%s:%d names sys.executable; import PYTHON from "
                    "_helpers.child_process instead"
                    % (path.relative_to(REPO_ROOT).as_posix(), node.lineno)
                )
    assert offenders == [], "\n".join(offenders)


def test_every_child_launch_pins_this_tree():
    """Launching the interpreter without a pinned ``PYTHONPATH`` is the bug.

    Either call a runner in ``_helpers.child_process`` (``run_python_child``,
    ``run_child_script``, ``run_mpi_python``, ``shell``) or, when the test needs
    the raw handle -- ``subprocess.Popen`` for a process that must stay alive --
    pass ``env=child_env(...)`` explicitly.
    """
    offenders = []
    for path in _python_files():
        if path == HELPER:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_launcher(node):
                continue
            if not _names_the_interpreter(node):
                continue
            if _pins_pythonpath(node):
                continue
            offenders.append(
                "%s:%d launches the interpreter without pinning PYTHONPATH; "
                "use _helpers.child_process"
                % (path.relative_to(REPO_ROOT).as_posix(), node.lineno)
            )
    assert offenders == [], "\n".join(offenders)


def _child_process_module():
    if str(TEST_ROOT) not in sys.path:
        sys.path.insert(0, str(TEST_ROOT))
    from _helpers import child_process

    return child_process


def test_the_helper_pins_this_tree_first():
    import os

    child_process = _child_process_module()
    env = child_process.child_env({"SOMETHING": "1"})
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(REPO_ROOT / "python")
    assert env["SOMETHING"] == "1"
    # An inherited PYTHONPATH is kept, but it never wins.
    env = child_process.child_env({"PYTHONPATH": "/somewhere/else"})
    parts = env["PYTHONPATH"].split(os.pathsep)
    assert parts[0] == str(REPO_ROOT / "python")
    assert "/somewhere/else" in parts


def test_the_helper_and_conftest_agree_on_the_source_tree():
    """One decision, one implementation.

    If ``conftest`` computed the parent's ``sys.path`` and the helper computed
    the child's ``PYTHONPATH`` separately, the two would drift and the drift
    would be silent -- the child would simply test another checkout.
    """
    import conftest

    child_process = _child_process_module()
    assert conftest.source_python_dir is child_process.source_python_dir


def test_the_default_child_timeout_is_generous_and_configurable():
    import os

    child_process = _child_process_module()
    # A cold child compiles the core. 180 s was tuned for an idle machine and
    # turned into a recurring false red whenever anything else was running.
    assert child_process.DEFAULT_TIMEOUT >= 600
    previous = os.environ.get("JITTOR_TEST_CHILD_TIMEOUT")
    os.environ["JITTOR_TEST_CHILD_TIMEOUT"] = "1234"
    try:
        assert child_process.default_timeout() == 1234
        assert child_process.default_timeout(7) == 7
    finally:
        if previous is None:
            os.environ.pop("JITTOR_TEST_CHILD_TIMEOUT", None)
        else:
            os.environ["JITTOR_TEST_CHILD_TIMEOUT"] = previous

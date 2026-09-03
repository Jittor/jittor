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
import os
from pathlib import Path
import sys
import time

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
    from _helpers.root_conftest import root_conftest_imports_from_the_helper

    # Asserted on the file, not on a module object. `import conftest` picks
    # whichever of this tree's two conftest modules pytest loaded first, so the
    # object form passed under `pytest tests/structure` and failed in the whole
    # Torch-mode session -- see tests/_helpers/root_conftest.py.
    assert root_conftest_imports_from_the_helper("source_python_dir")


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


def test_a_complete_environment_is_not_silently_merged():
    """Merging a full environment onto ``os.environ`` cannot remove anything.

    This is how a *deliberately native* probe ended up inheriting
    ``JITTOR_TORCH_SHIM=1``. The caller built ``dict(os.environ)``, popped the
    four Torch variables, and passed the result -- and the helper merged it back
    onto ``os.environ``, restoring every one of them. The child then refused to
    compose with "cannot install Jittor Torch compatibility over an existing
    Torch module graph", in a test about import cycles.

    Removals are the reason a caller passes a whole environment, so the helper
    now refuses the ambiguous call instead of quietly dropping them.
    """
    import os

    import pytest

    child_process = _child_process_module()
    complete = dict(os.environ)
    complete.pop("JITTOR_TORCH_SHIM", None)
    with pytest.raises(AssertionError) as raised:
        child_process.child_env(complete)
    assert "inherit=False" in str(raised.value)
    # Stated instead of merged: the removal survives.
    kept = child_process.child_env(complete, inherit=False)
    assert "JITTOR_TORCH_SHIM" not in kept


def test_without_torch_mode_clears_every_variable_that_installs_the_shim():
    """One option, not four pops repeated at each call site.

    ``tests/structure`` is itself a Torch-mode path, so a probe that has to
    start native cannot assume these are unset -- it has to clear them. Leaving
    that to each caller is what got lost when the call sites were collected into
    this helper.
    """
    import os

    child_process = _child_process_module()
    previous = {name: os.environ.get(name)
                for name in child_process.TORCH_MODE_VARIABLES}
    for name in child_process.TORCH_MODE_VARIABLES:
        os.environ[name] = "1"
    try:
        env = child_process.child_env(without_torch_mode=True)
        assert not [name for name in child_process.TORCH_MODE_VARIABLES
                    if name in env]
        inherited = child_process.child_env()
        assert all(name in inherited
                   for name in child_process.TORCH_MODE_VARIABLES)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


# --------------------------------------------------------------------------
# The timeout has to end the whole tree, not just the direct child (9.23).
# --------------------------------------------------------------------------

#: A child that hangs *and* leaves a grandchild holding the same stdout pipe.
#:
#: This is not a contrived shape: it is what every dataset test looks like when
#: the loader deadlocks. ``subprocess.run(timeout=N)`` SIGKILLs the direct child
#: and then drains the pipes -- and the pipe is still open, because the
#: grandchild inherited the write end and nothing killed *it*. The drain has no
#: deadline, so the helper never returns: a `timeout=300` was measured still
#: waiting ten minutes in.
_HANGS_WITH_A_GRANDCHILD = """
import subprocess, sys, time

# inherits this process' stdout, so it holds the write end of the pipe
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
with open(%r, "w") as handle:
    handle.write(str(child.pid))
sys.stdout.write("started\\n")
sys.stdout.flush()
time.sleep(300)
"""


def _still_running(pid):
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def test_a_timeout_ends_the_grandchildren_too(tmp_path):
    """A hung grandchild must not turn a timeout into a hung session."""
    from _helpers.child_process import run_child_script

    pid_file = tmp_path / "grandchild.pid"
    started = time.time()
    try:
        run_child_script(
            _HANGS_WITH_A_GRANDCHILD % str(pid_file),
            timeout=2,
            directory=str(tmp_path),
        )
    except AssertionError as failure:
        assert "did not finish within 2 s" in str(failure), failure
    else:
        raise AssertionError("the hung child was reported as finishing")
    elapsed = time.time() - started
    # The budget is 2 s plus one drain window; anything near 300 s means the
    # drain waited for the grandchild again.
    assert elapsed < 60, "the helper took %.1f s to give up on a 2 s child" % elapsed

    grandchild = int(pid_file.read_text())
    for _ in range(200):
        if not _still_running(grandchild):
            break
        time.sleep(0.05)
    assert not _still_running(grandchild), (
        "grandchild %d outlived the timeout: killing the direct child leaves "
        "the rest of the tree running (and holding the pipe)" % grandchild)

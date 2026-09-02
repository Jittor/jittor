"""Launch a Python child process that imports the tree under test.

Why this exists
---------------
``tests/conftest.py`` decides which checkout this session imports (see
:func:`source_python_dir`) and puts it on *this* process' ``sys.path``. It does
not export ``PYTHONPATH``, so a bare ``subprocess.run([sys.executable, ...])``
from a test imports whatever ``jittor`` the interpreter's environment resolves
-- in a development checkout that is the editable install, i.e. some *other*
working tree.

The failure that costs the most is not the loud one. A child that imports
another tree usually still works, so the test passes and proves nothing about
the code under test: on a branch that changed behaviour, the child measured the
tree it was not testing. The loud version showed up when a rename landed:
``[0.08]`` renamed core's ``set_lock_path`` to ``set_lock_fd``, the child
imported the *main* tree's ``compiler.py`` (old name) while loading the
*branch*-built core (new name), and died with an ``AttributeError`` that pointed
at neither tree.

So: every child process a test starts goes through this module, which pins
``PYTHONPATH`` to the checkout the session itself imports.

The second reason is timeouts. A cold child pays for a full core compile. A
timeout tuned for an idle machine turns into a recurring false red as soon as
anything else runs on the box, so the default here is generous and the message
says what actually happened.

What to call
------------
=========================== =====================================================
:func:`run_python_child`    ``python <args>`` -- the common case
                            (``crash_isolated=True`` when it may die by signal)
:func:`run_child_script`    write a source string to a file and run it
:func:`run_mpi_python`      ``mpirun -np N python <args>``
:func:`shell`               a shell command line that needs quoting or job control
:func:`child_env`           build the environment yourself, launch it yourself
=========================== =====================================================

``tests/structure/test_child_process_contract.py`` fails the gate on any launch
in ``tests/`` that names the interpreter without going through this module.
"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[2]

#: The interpreter a child must run. Tests name this rather than
#: ``sys.executable`` so the static contract check can see every launch.
PYTHON = sys.executable

#: Wall-clock budget for a child that cold-starts jittor.
#:
#: Each child imports jittor from scratch and may compile the core against a
#: cold cache. 180 s was comfortable on an idle machine and stopped being
#: comfortable once a dozen agents shared the box: the child is correct, gets
#: killed anyway, reads as a red gate, and costs someone a bisect. Wall clock is
#: not the property under test, so this budget exists only to turn a genuine
#: hang into a failure rather than a hung session.
#:
#: Kept under the gates' ``--timeout=900`` so an overrun surfaces as this
#: module's assertion -- which names the command -- instead of pytest killing
#: the whole test with no idea which child stalled.
DEFAULT_TIMEOUT = 600

#: Both names are honoured: the first is this helper's, the second predates it.
TIMEOUT_VARIABLES = ("JITTOR_TEST_CHILD_TIMEOUT", "JITTOR_TEST_SUBPROCESS_TIMEOUT")

#: Variables that ask a child to install the Torch compatibility layer.
#:
#: A probe that installs a stand-in ``torch`` module and then imports jittor
#: needs all four gone, or composition refuses with "cannot install Jittor Torch
#: compatibility over an existing Torch module graph". The parent inherits them
#: from the session -- ``tests/structure`` is itself a Torch-mode path -- so
#: "the environment I did not set them in" is not the same as "they are unset".
TORCH_MODE_VARIABLES = (
    "REAL_TORCH_SITE",
    "JITTOR_TORCH_SHIM",
    "JITTOR_TORCH_PROJECT_ROOT",
    "JITTOR_TORCH_RUNTIME_ROOT",
)


def source_python_dir():
    """The ``python/`` directory this test session imports jittor from.

    ``tests/conftest.py`` imports this function, so the parent's ``sys.path``
    and the ``PYTHONPATH`` of every child it starts can never disagree.

    ``JITTOR_SOURCE_ROOT`` names the checkout to import. Unset, it is the
    checkout this file belongs to. Set but empty, nothing is pinned and whatever
    is installed wins -- for a child that means an untouched ``PYTHONPATH``.
    """
    override = os.environ.get("JITTOR_SOURCE_ROOT")
    if override is None:
        root = REPO_ROOT
    elif override.strip():
        root = Path(override.strip()).expanduser().resolve()
    else:
        return None
    return str(root / "python")


def python_executable():
    """The interpreter children run; a function so call sites read as intent."""
    return PYTHON


def child_env(extra=None, inherit=True, without_torch_mode=False):
    """An environment whose ``PYTHONPATH`` starts at this session's checkout.

    ``extra`` is applied on top of the inherited environment, but the pinned
    directory is prepended afterwards and always comes first: an inherited or
    caller-supplied ``PYTHONPATH`` is kept behind it, never in front of it.
    Pass ``inherit=False`` when ``extra`` is already the complete environment --
    a caller that *removed* a variable needs that, since merging onto
    ``os.environ`` would put it straight back.
    """
    if inherit and extra and "PATH" in extra:
        # Merging a complete environment onto os.environ cannot *remove*
        # anything, so every `env.pop(...)` the caller did is silently undone.
        # That is not hypothetical: it turned a deliberately native probe into
        # one that inherited JITTOR_TORCH_SHIM=1 and failed on composition.
        raise AssertionError(
            "child_env() was handed what looks like a complete environment "
            "(it contains PATH) while inherit=True. Merging it onto os.environ "
            "cannot remove a variable, so any env.pop() the caller did is lost. "
            "Pass inherit=False for a complete environment, or pass only the "
            "variables to change."
        )
    env = dict(os.environ) if inherit else {}
    if extra:
        env.update({key: str(value) for key, value in extra.items()})
    if without_torch_mode:
        for name in TORCH_MODE_VARIABLES:
            env.pop(name, None)
    pinned = source_python_dir()
    if pinned is not None:
        existing = env.get("PYTHONPATH", "")
        parts = [pinned] + [
            part for part in existing.split(os.pathsep) if part and part != pinned
        ]
        env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def default_timeout(timeout=None):
    """The budget for one child. ``timeout=0`` means "no limit".

    A whole-suite runner has no natural bound -- capping it would only turn a
    long run into a failure -- so it opts out explicitly rather than by
    forgetting to pass anything.
    """
    if timeout is not None:
        return timeout or None
    for name in TIMEOUT_VARIABLES:
        configured = os.environ.get(name, "").strip()
        if configured.isdigit():
            return int(configured)
    return DEFAULT_TIMEOUT


def _timeout_failure(command, seconds):
    printable = command if isinstance(command, str) else " ".join(
        str(part) for part in command)
    return AssertionError(
        "child process did not finish within %d s: %s\n"
        "A cold child compiles the Jittor core, so this is a loaded machine or a "
        "cold cache far more often than a real hang. Raise "
        "JITTOR_TEST_CHILD_TIMEOUT if that is the case." % (seconds, printable)
    )


def _crash_isolated(command, env):
    """Put a shell between pytest and a child that is expected to crash.

    Jittor installs a *process-level* ``SIGCHLD`` handler (see
    ``src/utils/log.cc``): when a direct child dies from a signal rather than
    exiting, the handler quick-exits the parent. That makes the standard
    technique -- "run the case that segfaults in a child so it cannot take the
    session down" -- do exactly what it was meant to prevent: the child aborts,
    the handler fires inside pytest, and pytest vanishes mid-run with no output
    at all (``-q`` buffers it, so it is lost). It reads as "the runner broke",
    not "a test failed", and it has already cost two partitions an afternoon
    each (6.C31).

    ``sh`` between the two absorbs the signal death: pytest's direct child
    always exits normally, with ``128 + signo``, which is ``CLD_EXITED`` and
    leaves the handler alone. ``returncode`` is still 134 or 139, so the crash
    remains assertable.

    ``gdb_path`` is cleared for the same reason ``tools/run_test_suite.py``
    clears it: Jittor's crash handler forks gdb for a backtrace, and gdb
    ptrace-stops the child first -- if gdb then dies, the child stays stopped
    forever and the timeout is the only thing that ends it.
    """
    env = dict(env or {})
    env.setdefault("gdb_path", "")
    if os.name != "posix":
        return command, env
    return ["/bin/sh", "-c", '"$@"; exit $?', "sh"] + command, env


def _run(command, env, timeout, cwd, text, check, input, merge_stderr,
         shell=False, inherit=True, without_torch_mode=False):
    seconds = default_timeout(timeout)
    # Jittor's own logging is not ASCII (op keys are separated by U+00AB), so
    # decoding a child's output by the ambient locale fails outright under
    # LANG=C. Decode as UTF-8 and never let a stray byte become the failure.
    decoding = {"encoding": "utf-8", "errors": "replace"} if text else {}
    try:
        return subprocess.run(
            command,
            env=child_env(env, inherit=inherit,
                          without_torch_mode=without_torch_mode),
            cwd=None if cwd is None else str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
            timeout=seconds,
            check=check,
            input=input,
            shell=shell,
            **decoding
        )
    except subprocess.TimeoutExpired as expired:
        raise _timeout_failure(command, seconds) from expired


def run_python_child(args, *, env=None, timeout=None, cwd=None, text=True,
                     check=False, input=None, merge_stderr=False, inherit=True,
                     crash_isolated=False, without_torch_mode=False):
    """Run ``[PYTHON, *args]`` against this tree, with a clear timeout.

    ``merge_stderr`` folds stderr into stdout, which is what most callers want
    when they print the child's output on failure.

    ``crash_isolated=True`` for a child that is *expected* to die from a signal
    -- see :func:`_crash_isolated`. Opt-in on purpose: the extra shell changes
    what a timeout does. ``subprocess.run`` kills only its direct child on
    ``TimeoutExpired``, with an uncatchable SIGKILL, so behind a wrapper the
    grandchild is orphaned rather than killed. Only a crash test should pay
    that; everything else is safer without it.
    """
    command = [PYTHON] + [str(arg) for arg in args]
    if crash_isolated:
        command, env = _crash_isolated(command, env)
    return _run(command, env, timeout, cwd, text, check, input, merge_stderr,
                inherit=inherit, without_torch_mode=without_torch_mode)


def run_child_script(source, *, env=None, timeout=None, cwd=None, text=False,
                     check=False, merge_stderr=False, directory=None,
                     name="child", inherit=True, crash_isolated=False,
                     without_torch_mode=False):
    """Write ``source`` to a file and run it, so tracebacks name real lines.

    ``python -c`` reports ``<string>`` for every frame, which makes a failing
    child unreadable. Bytes by default, because the callers that predate this
    helper decode the streams themselves.
    """
    if directory is None:
        directory = tempfile.mkdtemp(prefix="jittor-child-")
    path = os.path.join(str(directory), "%s_%d.py" % (name, os.getpid()))
    with open(path, "w") as handle:
        handle.write(source)
    command = [PYTHON, path]
    if crash_isolated:
        command, env = _crash_isolated(command, env)
    return _run(command, env, timeout, cwd, text, check, None,
                merge_stderr, inherit=inherit,
                without_torch_mode=without_torch_mode)


def mpirun_path():
    """The launcher next to the ``mpicc`` Jittor was built against.

    Imported lazily: static structure tests use this module and must not pull
    the runtime in.
    """
    import jittor as jt

    return jt.compile_extern.mpicc_path.replace("mpicc", "mpirun")


def run_mpi_python(num_procs, args, *, env=None, timeout=None, cwd=None,
                   text=True, check=False, merge_stderr=False, launcher=None,
                   inherit=True, without_torch_mode=False):
    """Run ``mpirun -np N python <args>`` against this tree.

    ``mpirun`` starts the ranks itself, so the parent's ``sys.path`` reaches
    none of them: without the pinned ``PYTHONPATH`` every rank imports the
    installed jittor instead of the checkout under test.
    """
    command = [launcher or mpirun_path(), "-np", str(num_procs), PYTHON]
    command += [str(arg) for arg in args]
    return _run(command, env, timeout, cwd, text, check, None, merge_stderr,
                inherit=inherit, without_torch_mode=without_torch_mode)


def shell(command, *, env=None, timeout=None, cwd=None, text=True,
          check=False, merge_stderr=False, inherit=True,
          without_torch_mode=False):
    """Run a shell command line that needs its own quoting or job control.

    A few tests build a pipeline (``$(python -m jittor_utils.config ...)``) or
    start two children and ``wait`` on both. They still have to reach this tree,
    so they run here instead of through ``os.system``.
    """
    return _run(command, env, timeout, cwd, text, check, None, merge_stderr,
                shell=True, inherit=inherit,
                without_torch_mode=without_torch_mode)


def shell_status(command, *, env=None, timeout=None, cwd=None):
    """``os.system``-shaped: run a shell command line, return its exit status."""
    return shell(command, env=env, timeout=timeout, cwd=cwd,
                 merge_stderr=True).returncode

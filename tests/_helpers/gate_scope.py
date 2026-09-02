"""What a gate runs: the whole test tree, minus paths that state why not.

The gate used to be a hand-written list of paths in ``noxfile.py``. Nothing kept
it in step with the tree, so it drifted: of 329 test files, 97 were reachable
from any workflow and 232 were reachable from none. A test outside that list was
written, reviewed, merged -- and never run again by anything. Two consequences
were measured rather than guessed: the OpInfo backward battery instantiated zero
cases for a year, and 19 files under ``tests/backends/cpu``, ``tests/data`` and
``tests/nn`` had no gate at all.

So the default is inverted. A gate runs ``tests/``. A path that a gate must
*not* run is listed here with the reason it cannot, and
``tests/structure/test_gate_scope.py`` fails if a reason is missing, if an
excluded path has disappeared, or if the reachable share of the tree drops.

"Cannot run here" is a narrow claim. A test that needs an accelerator already
skips itself, and a skip is information: it shows up in the summary and 0.18
turns "this entry only ever skips" into a gate failure. Excluding it instead
would hide it. The list below is therefore for tests that *break* a shared
session rather than tests that merely have nothing to do in it.

Two processes, not one
----------------------
Torch compatibility mode is process-global: it changes lazy execution, reduction
defaults and gradient semantics for everything in the interpreter. The paths in
``process_modes.TORCH_MODE_PATHS`` own that mode, the rest of the tree asserts
native behaviour, and one ``pytest tests`` run cannot cover both. That is a
split, not an exclusion -- every file is still run, in the session that owns it.
"""

from _helpers.process_modes import TORCH_MODE_PATHS


#: The root every gate starts from.
TEST_ROOT = "tests"

#: ``(path, reason)`` -- a path no CPU gate runs, and why it cannot.
#:
#: Empty is the correct state. Anything added here is a test the gate stops
#: protecting, so the reason has to say what breaks, not that it is slow or
#: inconvenient: a slow test belongs in the full tier (0.15), and a test that
#: cannot run on this hardware belongs behind a skip, where it stays visible.
EXCLUDED = ()


#: Substrings of a skip reason that name something *this machine* lacks.
#:
#: A gate entry that only ever skips looks exactly like one that passes. That is
#: how 227 operators' backward formulas stayed unverified in three green gates
#: (0.01). But "this file executed nothing" is only a finding when the file
#: could have run: on a CPU box, ``tests/backends/cuda`` executing nothing is a
#: fact about the box.
#:
#: A rule, not a list, on purpose. The list version would be 73 paths on this
#: machine and a different 73 on the next one, and every device test added later
#: would have to remember to join it. The rule instead reads what the test
#: itself said when it skipped: if every case in a file skipped for a reason
#: that names missing hardware, a missing launcher or a missing independent
#: PyTorch, the environment explains it. If even one case skipped for a reason
#: that does not, the file is unexplained and
#: ``JITTOR_TEST_REQUIRE_EXECUTION=1`` fails the run.
#:
#: This is deliberately not satisfied by a bare "skip": a reason that does not
#: say what is missing cannot be checked by anyone later, which is the whole
#: problem being fixed.
#: Matched as substrings of the lowercased reason. Deliberately *names of
#: things* rather than phrases: "No CUDA found", "cuda is required" and "not
#: use cublas, skip" are three ways of saying the same fact about the machine,
#: and a phrase list would have to grow one entry per author.
ENVIRONMENT_SKIP_PATTERNS = (
    # accelerators and the libraries that only exist alongside them
    "cuda", "cudnn", "cublas", "cutt", "cusparse", "cufft", "curand",
    "gpu", "accelerator", "acl", "npu", "ascend", "cann", "rocm", "hip",
    "triton",
    # an independent PyTorch build, which only the oracle sessions have
    "torch",
    # multi-rank launchers
    "mpi", "nccl", "world size",
    # opt-in assets and probes
    "download", "dataset", "network", "manual probe",
)

#: ``(path, reason)`` -- a file exempt even though its skips do not explain it.
#: Empty is the correct state; the rule above should cover the honest cases.
EXECUTES_NOTHING = ()


def excluded_paths():
    return tuple(path for path, _reason in EXCLUDED)


def _ignores(paths):
    return tuple("--ignore=" + path for path in paths)


def native_arguments():
    """pytest arguments for the session that owns native semantics."""
    return (TEST_ROOT,) + _ignores(tuple(TORCH_MODE_PATHS) + excluded_paths())


def torch_arguments():
    """pytest arguments for the session that owns Torch compatibility mode."""
    excluded = excluded_paths()
    selected = tuple(path for path in TORCH_MODE_PATHS if path not in excluded)
    return selected + _ignores(excluded)


def selected_files(repo_root, arguments):
    """The test files a pytest invocation with ``arguments`` would collect.

    Static, so a structure test can measure gate reach without running anything.
    """
    from pathlib import Path

    root = Path(repo_root)
    ignored = tuple(
        argument[len("--ignore="):] for argument in arguments
        if argument.startswith("--ignore=")
    )
    selected = tuple(
        argument for argument in arguments if not argument.startswith("-")
    )
    found = set()
    for target in selected:
        path = root / target.split("::", 1)[0]
        candidates = sorted(path.rglob("test_*.py")) if path.is_dir() else [path]
        for candidate in candidates:
            if not candidate.is_file():
                continue
            relative = candidate.relative_to(root).as_posix()
            if any(relative == item or relative.startswith(item.rstrip("/") + "/")
                   for item in ignored):
                continue
            found.add(relative)
    return found

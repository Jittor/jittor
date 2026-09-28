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

from _helpers.process_modes import TORCH_MODE_PATHS, NATIVE_MODE_PATHS


#: The root every gate starts from.
TEST_ROOT = "tests"
TEST_ROOTS = (TEST_ROOT, "compat/tests", "adapters/tests")

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
    # accelerators and the libraries that only exist alongside them. `nvcc` is
    # here because a CUDA *compiler* is the thing whose absence a CPU-only build
    # states most directly (`tests/build/test_cuda_arch_flags.py` says exactly
    # "no nvcc", and a one-word reason like that matches none of the library
    # names around it).
    "cuda", "cudnn", "cublas", "cutt", "cusparse", "cufft", "curand",
    "nvcc",
    "gpu", "accelerator", "acl", "npu", "ascend", "cann", "rocm", "hip",
    "triton",
    # an independent PyTorch build, which only the oracle sessions have.
    # Withdrawn when JITTOR_REQUIRE_REAL_TORCH=1 -- see REAL_TORCH_PATTERNS.
    "torch",
    # multi-rank launchers
    "mpi", "nccl", "world size",
    # opt-in assets and probes
    "download", "dataset", "network", "manual probe",
    # Optional third-party libraries the shim is validated against. Absence is a
    # fact about the machine; `JITTOR_REQUIRE_OPTIONAL_DEPS=1` is what turns it
    # into a configuration error, the way REAL_TORCH_PATTERNS does for torch.
    # Most of these reasons happen to contain "torch" and were therefore
    # covered by accident; these did not, so a CPU-only session reported the
    # files as unexplained and the whole selection exited non-zero with every
    # test passing.
    "tensordict", "mmcv", "mmengine",
    # Facts about the *runner* rather than the machine's hardware: a case that
    # asserts directory permissions cannot hold when the suite runs as root,
    # because root bypasses them.
    "root ignores",
    # A backend library the build has *switched off* -- `use_mkl=0`, or any other
    # policy that says no. The wording is jittor's own and templated over the
    # library (`python/jittor/_runtime/backend_libraries.py`: "%s has an
    # enabled-policy and it currently says no"), so match the shape: it is as
    # much an environment fact as the library being absent. Measured: nine such
    # skips in `tests/backends/cpu/test_mkl_conv_op.py` + `test_onednn_contract.py`
    # alone, every one of them counted as `other`, and `other > 0` reds the run.
    "enabled-policy",
    # How this cache was *built*, which is as much a fact about the machine as
    # what is installed on it. `tests/core/test_graph_build_profile.py` skips
    # itself unless the core carries `-DJT_GRAPH_BUILD_PROFILE` (9 cases), and
    # the ops that ask for the vendored `cub` skip when the build has no cub
    # (4 cases across `tests/ops/test_argsort_op.py` and `test_arg_reduce_op.py`).
    # Both are the same shape as "not use cublas, skip" above.
    "jt_graph_build_profile", "cub",
    # The library is absent from *this cache* rather than from the machine:
    # `tests/build/test_download_safety.py` loads the real MKL and skips when the
    # cache has none (`use_mkl=0` again).
    "mkl",
    # The runner turns the crash handler's debugger off on purpose -- forking gdb
    # ptrace-stops the child, and a gdb that then dies leaves it stopped forever,
    # so `tests/_helpers/child_process.py` and `tools/run_test_suite.py` both
    # clear `gdb_path`. `tests/bindings/test_tracer.py::test_breakpoint` is the
    # case that needs it.
    "gdb is disabled",
    # Not a missing dependency but a documented non-reproduction:
    # `test_core_invariant_properties.py::test_the_leak_is_two_vars_per_occurrence`
    # pins the size of a leak that the module docstring records as driven by
    # holder teardown order, so a shape may balance instead of leaking. The
    # shapes and their counts are named in that file's `KNOWN_LEAKING_SHAPES`;
    # the case still fails when a *different* number appears.
    "nothing leaked in this environment",
    # A case that is deliberately not run unless asked for, where the reason says
    # how to ask: `tests/core/test_executor_python_threads.py` documents a
    # segfaulting thread race and gates itself behind `JT_TEST_THREAD_RACE=1`.
    # Its wording names no hardware, so it counted as `other` and the file -- a
    # single-case file, so it also executed nothing -- red the run for doing what
    # it was written to do.
    "jt_test_thread_race",
)

#: The subset of the above that stops being an explanation once a session
#: promises to *have* an independent PyTorch.
#:
#: This is the fail-open trap in the audit, stated as code. The comparison
#: against real PyTorch is the project's core claim, and its tests skip
#: themselves when ``REAL_TORCH_PYTHON`` is unset -- correctly, since comparing
#: Jittor's own shim against itself proves nothing. But a nightly gate whose
#: whole purpose is that comparison then reports success while doing nothing,
#: for exactly the reason it was built to prevent. So in a session that declares
#: ``JITTOR_REQUIRE_REAL_TORCH=1``, "no torch" is a *configuration error*, not a
#: fact about the machine, and every such skip fails the run.
REAL_TORCH_PATTERNS = ("torch",)


#: ``(path, reason)`` -- a file exempt even though its skips do not explain it.
#: Empty is the correct state; the rule above should cover the honest cases.
EXECUTES_NOTHING = ()


def excluded_paths():
    return tuple(path for path, _reason in EXCLUDED)


def _ignores(paths):
    return tuple("--ignore=" + runnable(path) for path in paths)


#: How a path under `compat/` has to be spelled for pytest to run it.
#:
#: `compat/` is its own distribution: it carries a `jittor.compat` package whose
#: `__init__.py` imports relatively out of `jittor`, and its own pytest ini.
#: Named by that path from the repository root, pytest imports it as a
#: top-level `compat` -- `ImportError: attempted relative import beyond
#: top-level package`, 140 errors in the torch session before a test runs.
#: Named through `python/jittor/compat`, the symlink a source checkout already
#: relies on, the same files run: pytest finds `compat/pyproject.toml` as the
#: inifile and the package is imported as `jittor.compat`, which is its name.
#: `docs/development/test-system.md` has told people this for a while; the gate
#: arguments themselves did not follow it.
_COMPAT_LINK = "python/jittor/"


def runnable(path):
    """``path`` spelled the way pytest can actually be pointed at it."""
    from pathlib import Path

    if not path.startswith("compat/"):
        return path
    root = Path(__file__).resolve().parents[2]
    linked = _COMPAT_LINK + path
    return linked if (root / linked).exists() else path


def canonical(path):
    """The inverse: the repository-relative path, whatever spelling came in."""
    return path[len(_COMPAT_LINK):] if path.startswith(_COMPAT_LINK + "compat/") else path


def native_arguments():
    """pytest arguments for the session that owns native semantics."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    pending = list(TORCH_MODE_PATHS)
    ignored = []
    while pending:
        path = pending.pop()
        if path in NATIVE_MODE_PATHS:
            continue
        if any(item.startswith(path.rstrip("/") + "/") for item in NATIVE_MODE_PATHS):
            pending.extend(child.relative_to(root).as_posix()
                           for child in sorted((root / path).iterdir())
                           if child.name != "__pycache__")
        else:
            ignored.append(path)
    return tuple(runnable(path) for path in TEST_ROOTS) \
        + _ignores(tuple(sorted(ignored)) + excluded_paths())


def torch_arguments():
    """pytest arguments for the session that owns Torch compatibility mode."""
    excluded = excluded_paths()
    selected = tuple(runnable(path) for path in TORCH_MODE_PATHS
                     if path not in excluded)
    return selected + _ignores(excluded + NATIVE_MODE_PATHS)


def selected_files(repo_root, arguments):
    """The test files a pytest invocation with ``arguments`` would collect.

    Static, so a structure test can measure gate reach without running anything.
    """
    from pathlib import Path

    root = Path(repo_root)
    ignored = tuple(
        canonical(argument[len("--ignore="):]) for argument in arguments
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
            # Resolved, so a file reached through `python/jittor/compat` is
            # reported under `compat/`: the arguments are spelled for pytest,
            # the answer is about the repository.
            relative = candidate.resolve().relative_to(root.resolve()).as_posix()
            if any(relative == item or relative.startswith(item.rstrip("/") + "/")
                   for item in ignored):
                continue
            found.add(relative)
    return found

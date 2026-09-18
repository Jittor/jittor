#!/usr/bin/env python3
"""Run the repository test suite at one of three tiers and report one result.

Jittor's Torch compatibility mode is process-global: it changes lazy execution,
reduction defaults and gradient semantics for everything in the interpreter.
Native tests and Torch-compatibility tests therefore cannot share a process, and
a single ``pytest tests`` run cannot cover both. This script runs each mode in
its own pytest session, with its own JIT cache, and prints a combined summary.

Usage::

    python tools/run_test_suite.py --tier core       # ~1 min, both modes
    python tools/run_test_suite.py --tier smoke      # what a pull request waits for
    python tools/run_test_suite.py                   # the whole tree
    python tools/run_test_suite.py --session native
    python tools/run_test_suite.py --tier core --backend cuda   # on the GPU
    python tools/run_test_suite.py -- -x -k conv     # extra pytest arguments

The tiers answer different questions. ``core`` is for between edits: one file
per fundamental, named in ``tests/_helpers/tiers.CORE_FILES``, serial, and
runnable without pytest-xdist. ``smoke`` is the pull-request gate: the whole
tree minus the files in ``SLOW_FILES``. ``full`` is everything. Only ``full``
is a statement about the tree; the other two are statements about time.

``--backend`` picks the device. The default ``cpu`` is what this script has
always run and what ``nox -s cpu`` gates; ``cuda`` mirrors ``nox -s cuda``. They
are different statements, not degrees of the same one: half precision, mixed
precision and ``.cuda()``/``.cpu()`` placement skip entirely on ``cpu``, so a
green CPU tier says nothing about them. Each backend keeps its own JIT cache.

Runtime state (JIT caches, temporary files) is written under
``$JITTOR_LAB_ROOT/_state/test-suite`` so it never lands in the checkout.
"""

from __future__ import print_function

import argparse
import os
from pathlib import Path
import re
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO_ROOT / "tests"))
from _helpers.child_process import PYTHON, run_python_child, source_python_dir  # noqa: E402
from _helpers.gate_scope import (  # noqa: E402
    native_arguments,
    torch_arguments,
)

SESSIONS = ("native", "torch")
#: Which device the sessions execute on. ``cpu`` is what `nox -s cpu` gates and
#: what this script has always run; ``cuda`` mirrors `nox -s cuda`, which is the
#: only place half-precision, mixed precision and `.cuda()`/`.cpu()` placement
#: are exercised at all -- on ``cpu`` those tests skip, so a green CPU tier says
#: nothing about them.
BACKENDS = ("cpu", "cuda")

_COUNT = re.compile(r"(\d+) (passed|failed|skipped|error|errors|xfailed|xpassed)")
_WARMUP_MARKERS = {"cpu": "JITTOR_TEST_SUITE_CPU_READY",
                   "cuda": "JITTOR_TEST_SUITE_CUDA_READY"}
_WARMUP_ATTEMPTS = 3

#: Backend libraries the warm-up builds up front.
#:
#: Compute dependencies only. The communicators (``mpi``, ``nccl``, ``hccl``)
#: are deliberately absent: their loaders initialise a communicator, which
#: means waiting for the other ranks, and there are none here -- a warm-up
#: that can block forever is worse than the stall it replaces.
_WARMUP_LIBRARIES = (
    "mkl", "cub", "cutt", "cudnn", "cublas", "curand", "cufft", "cusparse",
)

# Keep the standalone runner's worker split identical to nox.  The shared
# policy owns both the pool list and the budget calculation.
sys.path.insert(0, str(REPO_ROOT / "tests"))
try:
    from _helpers.tiers import THREAD_POOL_ENV_NAMES  # noqa: E402
finally:
    sys.path.remove(str(REPO_ROOT / "tests"))
_THREAD_ENV_NAMES = THREAD_POOL_ENV_NAMES

#: ``compiler.JIT_UTILS_UPDATED_EXIT_CODE``. A cold or stale cache rebuilds
#: ``jit_utils`` and the process cannot reload it, so it exits and asks to be
#: re-run -- which is what the warm-up loop is for.
#:
#: The retry used to work by accident: the rebuild exited *zero* and printed a
#: message, so "no marker, exit 0" caught it. 0.11 made that exit non-zero so CI
#: could see it, and the same change made this loop give up on the first
#: attempt. The condition has to name the code, not rely on the exit status
#: being wrong.
_JIT_UTILS_UPDATED_EXIT_CODE = 3


def _lab_root():
    configured = os.environ.get("JITTOR_LAB_ROOT")
    root = Path(configured) if configured else REPO_ROOT.parent / "jittor-lab"
    return root.expanduser().resolve()


def _resolve_nvcc():
    """The nvcc the CUDA backend will build with, or None."""
    configured = os.environ.get("nvcc_path") or os.environ.get("NVCC_PATH")
    if configured and Path(configured).exists():
        return configured
    found = shutil.which("nvcc")
    if found:
        return found
    default = Path("/usr/local/cuda/bin/nvcc")
    return str(default) if default.exists() else None


def _session_environment(session, serial_compile=False, backend="cpu"):
    # One JIT cache per (backend, session). A CUDA core and a CPU core are
    # different objects compiled from the same sources, and sharing a
    # JITTOR_HOME between them means every switch rebuilds the tree. The CPU
    # path keeps its historical directory so the caches already warm there stay
    # warm; only the CUDA backend gets a new one.
    name = session if backend == "cpu" else "%s-%s" % (session, backend)
    state = _lab_root() / "_state" / "test-suite" / name
    (state / "home").mkdir(parents=True, exist_ok=True)
    (state / "tmp").mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["JITTOR_HOME"] = str(state / "home")
    environment["TMPDIR"] = str(state / "tmp")
    if backend == "cuda":
        nvcc = _resolve_nvcc()
        if not nvcc:
            raise SystemExit(
                "--backend cuda needs nvcc: set nvcc_path or put nvcc on PATH")
        environment["nvcc_path"] = nvcc
        environment["JITTOR_TEST_DEVICES"] = "cuda"
        # Both names are read by tests/_helpers/pytest_policy.py, and both are
        # what make this a gate rather than a run: without them a build that
        # silently fell back to CPU reports the same green as a real one,
        # because every accelerator test simply skips.
        environment["JITTOR_TEST_REQUIRE_CUDA"] = "1"
        environment["JITTOR_TEST_ACCELERATOR_MIN_EXECUTED"] = "1"
    else:
        environment["nvcc_path"] = ""
        environment["JITTOR_TEST_DEVICES"] = "cpu"
        environment.pop("JITTOR_TEST_REQUIRE_CUDA", None)
        environment.pop("JITTOR_TEST_ACCELERATOR_MIN_EXECUTED", None)
    environment["REAL_TORCH_SITE"] = ""
    environment["JITTOR_TORCH_SHIM"] = "1" if session == "torch" else "0"
    # Keep the standalone runner fail-closed like both nox gate sessions.  A
    # skip without an explicit environment reason is otherwise reported but
    # still counted as a successful suite, making the CLI a weaker gate than
    # the command it is meant to reproduce.
    environment["JITTOR_TEST_REQUIRE_EXECUTION"] = "1"
    # Set, never inherited -- like every other name in this function. On by
    # default, like the gate: it used to be forced off here with no reason
    # recorded, which made this script measure something `nox -s cpu` does not
    # run, and the parallel op compiler is where a cold whole-tree run's time
    # goes. ``--serial-compile`` restores the old value for bisecting a compile
    # failure; tests/codegen/test_parallel_compile_attribution.py is why that
    # is a diagnostic convenience and not a correctness measure.
    environment["use_parallel_op_compiler"] = "0" if serial_compile else "16"
    # Jittor's segfault handler shells out to gdb for a backtrace. That is
    # useful interactively and ruinous in a suite: gdb ptrace-stops the process
    # first, and if gdb itself dies -- on this distribution it crashes into the
    # apport hook -- the process is left stopped forever, so one crashing test
    # hangs the whole session instead of failing it. A crash here should be a
    # reported failure; run the test on its own to get a backtrace.
    environment.setdefault("gdb_path", "")
    return environment


def _session_arguments(session, tier="full"):
    """The same selection `nox -s cpu` uses, from the same source.

    The core tier is the exception: it names its files outright
    (``tiers.CORE_FILES``) instead of taking the tree and removing things, so
    the selection is what is listed and nothing else.
    """
    if tier == "core":
        return list(_tiers().core_paths(session))
    return list(torch_arguments() if session == "torch" else native_arguments())


def _parse_counts(output):
    for line in reversed(output.splitlines()):
        matches = _COUNT.findall(line)
        if not matches:
            continue
        counts = {}
        for number, kind in matches:
            counts["error" if kind == "errors" else kind] = int(number)
        return counts
    return {}


def _warmup(environment):
    if environment.get("JITTOR_TORCH_SHIM") == "1":
        source = source_python_dir()
        expected = str(Path(source).parent / "compat") if source else ""
        # Read installation metadata without importing Jittor or Torch. A
        # different checkout's editable frontend must not satisfy this gate.
        check = r'''
import json, pathlib, sys
from urllib.parse import unquote, urlparse
try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata
expected = sys.argv[1]
source_python = sys.argv[2] if len(sys.argv) > 2 else ""
try:
    dist = metadata.distribution("jittor-torch")
except metadata.PackageNotFoundError:
    dist = None
valid = dist is not None
if valid and expected:
    record = json.loads(dist.read_text("direct_url.json") or "{}")
    source = pathlib.Path(unquote(urlparse(record.get("url", "")).path)).resolve()
    valid = record.get("dir_info", {}).get("editable", False) and source == pathlib.Path(expected).resolve()
# The other way to reach this checkout's frontend, and the property being
# checked is the same one: that `import torch` resolves to *this* tree. A
# source checkout puts `python/` on the path and reaches the compat package
# through `python/jittor/compat`, which is a symlink into the checkout. That
# is verifiable here without importing jittor or torch, which is what this
# check is careful not to do -- and without it the suite cannot be run at all
# from a plain checkout, only from an installed one.
if not valid and expected and source_python:
    linked = pathlib.Path(source_python) / "jittor" / "compat"
    valid = linked.is_dir() and linked.resolve() == pathlib.Path(expected).resolve()
if not valid:
    target = expected or "jittor-torch"
    print("Torch suite needs the matching frontend installation. Run:", file=sys.stderr)
    args = [sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation"]
    if expected:
        args.append("-e")
    args.append(target)
    import shlex
    print(" ".join(shlex.quote(arg) for arg in args), file=sys.stderr)
    sys.exit(1)
'''
        checked = run_python_child(
            ["-c", check, expected, source or ""], cwd=REPO_ROOT, env=environment,
            inherit=False, merge_stderr=True, timeout=30)
        if checked.returncode:
            return checked.returncode, checked.stdout
    backend = "cuda" if environment.get("JITTOR_TEST_DEVICES") == "cuda" else "cpu"
    marker = _WARMUP_MARKERS[backend]
    # The CUDA probe asserts the build has CUDA and then runs on it. Both
    # halves matter: a tree built without nvcc imports and computes perfectly
    # well on the host, so a session that only checked `import jittor` would
    # report a green CUDA gate for a CPU run.
    device_setup = ("assert jt.compiler.has_cuda; jt.flags.use_cuda = 1; "
                    if backend == "cuda" else "assert not jt.compiler.has_cuda; ")
    probe = (
        "import jittor as jt; "
        + device_setup +
        "assert not getattr(jt.compiler, 'has_acl', 0); "
        "jt.flags.use_parallel_op_compiler = 0; "
        # Build the lazily-registered backend libraries here, where they are
        # attributable, instead of leaving them to whichever test first
        # reaches one. oneDNN is the reason: it is compiled from source on a
        # cold cache, it is registered as a lazy loader, and nothing before
        # this line touches it -- so a cold `tests/nn` spent eight minutes
        # inside the first CPU convolution it collected, with no output, and
        # the test that paid for it was picked by collection order.
        # probe_library reports rather than raises, so a backend that is
        # merely absent does not turn the warm-up into a failure.
        "from jittor._runtime.backend_libraries import probe_library as _probe; "
        "[ _probe(_name, load=True) for _name in %r ]; "
        "x = (jt.array([1.0, 2.0]) * 2).sum(); x.sync(); "
        "assert float(x.item()) == 6.0; "
        "print(%r)" % (_WARMUP_LIBRARIES, marker)
    )
    outputs = []
    for _attempt in range(_WARMUP_ATTEMPTS):
        # Through the helper, so the warm-up compiles *this* checkout. A bare
        # child imported whatever the editable install points at, so the warm-up
        # filled the session's JITTOR_HOME with the main tree's core and the
        # pytest run that followed compiled everything a second time. The log
        # said so all along: the `src:` line it prints names the tree it
        # actually imported.
        completed = run_python_child(
            ["-c", probe], cwd=REPO_ROOT, env=environment, inherit=False,
            merge_stderr=True, timeout=0)
        outputs.append(completed.stdout)
        if completed.returncode == _JIT_UTILS_UPDATED_EXIT_CODE:
            continue
        if completed.returncode != 0:
            return completed.returncode, "\n".join(outputs)
        if marker in completed.stdout:
            return 0, "\n".join(outputs)
    outputs.append("warmup did not execute the %s probe after %d attempts"
                   % (backend, _WARMUP_ATTEMPTS))
    return 1, "\n".join(outputs)


def _has_pytest_timeout():
    try:
        import importlib.util
        return importlib.util.find_spec("pytest_timeout") is not None
    except (ImportError, ValueError):
        return False


def _tiers():
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    try:
        from _helpers import tiers
    finally:
        sys.path.remove(str(REPO_ROOT / "tests"))
    return tiers


def _tier_arguments(tier):
    """What the fast tier drops, from the same list the nox session reads.

    The core tier drops nothing: its files are the selection itself, so a
    marker filter on top of them could only take coverage away silently.
    """
    if tier != "smoke":
        return []
    return ["-m", "not slow"]


def _split_threads(environment, jobs):
    """Each worker gets its share of the cores; see tiers.worker_thread_budget."""
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    try:
        from _helpers.tiers import apply_worker_thread_budget
    finally:
        sys.path.remove(str(REPO_ROOT / "tests"))
    apply_worker_thread_budget(environment, jobs)


def _parallel_arguments(jobs, distribution="loadfile"):
    """xdist arguments using the same distribution policy as nox."""
    if not jobs or jobs <= 1:
        return []
    try:
        import xdist  # noqa: F401
    except ImportError:
        raise SystemExit(
            "--jobs %d needs pytest-xdist (requirements/dev-tools.txt); "
            "running serially instead would report a wall clock for a gate "
            "nobody runs" % jobs)
    return ["-n", str(jobs), "--dist", distribution]


def _runtime_jobs(requested):
    """Resolve the omitted ``--jobs`` value exactly like the nox gates.

    ``None`` means the caller wants the gate policy: use the configured worker
    count and cap it to the CPU quota.  Zero remains an explicit serial
    diagnostic mode, which is useful when bisecting a failure and must not be
    confused with the normal runner default.
    """
    if requested is not None:
        if isinstance(requested, bool) or not isinstance(requested, int):
            raise SystemExit("--jobs must be a non-negative integer")
        if requested < 0:
            raise SystemExit("--jobs must be a non-negative integer")
        return requested
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    try:
        from _helpers.tiers import effective_cpu_count, runtime_workers
    finally:
        sys.path.remove(str(REPO_ROOT / "tests"))
    raw = os.environ.get("JITTOR_GATE_WORKERS", "4")
    try:
        configured = int(raw)
    except ValueError:
        raise SystemExit("JITTOR_GATE_WORKERS must be a positive integer")
    return runtime_workers(configured, available=effective_cpu_count())


def _run(session, extra, quiet, tier="full", jobs=None, serial_compile=False,
         backend="cpu"):
    environment = _session_environment(session, serial_compile=serial_compile,
                                       backend=backend)
    _split_threads(environment, jobs)
    command = [PYTHON, "-m", "pytest"]
    command += _session_arguments(session, tier)
    command += ["-p", "no:cacheprovider"]
    # pytest-timeout is a declared dev tool, and the per-test bound is worth
    # having; but it is a safety net, and a checkout without it should still be
    # able to run its own tests rather than fail on the argument.
    if _has_pytest_timeout():
        command += ["--timeout=900"]
    command += _tier_arguments(tier)
    distribution = "loadgroup" if tier == "smoke" else "loadfile"
    command += _parallel_arguments(jobs, distribution=distribution)
    command += ["-q"] if quiet else []
    command += extra
    print("=== {} session ({}) ===".format(session, backend), flush=True)
    warmup_code, warmup_output = _warmup(environment)
    print(warmup_output, flush=True)
    if warmup_code != 0:
        return warmup_code, {}, warmup_output
    print(" ".join(command), flush=True)
    # timeout=0: a whole-suite run has no natural bound, and capping it would
    # turn a long run into a failure rather than reporting one.
    completed = run_python_child(
        command[1:], cwd=REPO_ROOT, env=environment, inherit=False,
        merge_stderr=True, timeout=0)
    print(completed.stdout, flush=True)
    output = warmup_output + "\n" + completed.stdout
    return completed.returncode, _parse_counts(completed.stdout), output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", choices=SESSIONS, action="append", default=None)
    parser.add_argument("--tier", choices=("core", "smoke", "full"), default="full",
                        help="core runs only tests/_helpers/tiers.CORE_FILES "
                             "(about a minute, both modes, serial by default); "
                             "smoke drops the files recorded in "
                             "tests/_helpers/tiers.SLOW_FILES; full is the tree")
    parser.add_argument("--jobs", type=int, default=None,
                        help="xdist workers per session (default: runtime gate "
                             "policy; use 0 for explicit serial mode)")
    parser.add_argument("--backend", choices=BACKENDS, default="cpu",
                        help="device the sessions execute on (default: cpu, "
                             "which is what `nox -s cpu` gates); cuda mirrors "
                             "`nox -s cuda` and is the only way to gate "
                             "half-precision, AMP and .cuda()/.cpu() placement, "
                             "which skip on cpu")
    parser.add_argument("--serial-compile", action="store_true",
                        help="use_parallel_op_compiler=0; for bisecting a "
                             "compile failure, not for timing a gate")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("extra", nargs="*", help="extra pytest arguments")
    options = parser.parse_args()

    sessions = options.session or list(SESSIONS)
    results = {}
    failures = []
    # The core tier is small enough that workers buy nothing -- and staying
    # serial keeps it runnable where pytest-xdist is not installed.
    jobs = 0 if (options.tier == "core" and options.jobs is None) \
        else _runtime_jobs(options.jobs)
    print("runtime workers: %d%s" % (
        jobs, " (explicit)" if options.jobs is not None else " (policy)"),
          flush=True)
    for session in sessions:
        code, counts, output = _run(
            session, options.extra, not options.verbose,
            tier=options.tier, jobs=jobs,
            serial_compile=options.serial_compile,
            backend=options.backend)
        results[session] = (code, counts)
        for line in output.splitlines():
            if line.startswith("FAILED ") or line.startswith("ERROR "):
                failures.append("[{}] {}".format(session, line))

    print("=" * 72)
    total = {}
    for session in sessions:
        code, counts = results[session]
        print("{:8s} exit={}  {}".format(
            session,
            code,
            "  ".join("%s=%d" % (kind, value) for kind, value in sorted(counts.items())),
        ))
        for kind, value in counts.items():
            total[kind] = total.get(kind, 0) + value
    print("backend   {}".format(options.backend))
    print("combined  {}".format(
        "  ".join("%s=%d" % (kind, value) for kind, value in sorted(total.items()))
    ))
    if failures:
        print("-" * 72)
        for failure in failures:
            print(failure)
    return 0 if all(code == 0 for code, _counts in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

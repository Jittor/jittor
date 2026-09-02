#!/usr/bin/env python3
"""Run the complete repository test suite and report one combined result.

Jittor's Torch compatibility mode is process-global: it changes lazy execution,
reduction defaults and gradient semantics for everything in the interpreter.
Native tests and Torch-compatibility tests therefore cannot share a process, and
a single ``pytest tests`` run cannot cover both. This script runs each mode in
its own pytest session, with its own JIT cache, and prints a combined summary.

Usage::

    python tools/run_test_suite.py                  # both sessions, CPU
    python tools/run_test_suite.py --session native
    python tools/run_test_suite.py -- -x -k conv    # extra pytest arguments

Runtime state (JIT caches, temporary files) is written under
``$JITTOR_LAB_ROOT/_state/test-suite`` so it never lands in the checkout.
"""

from __future__ import print_function

import argparse
import os
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO_ROOT / "tests"))
from _helpers.child_process import PYTHON, run_python_child  # noqa: E402
from _helpers.gate_scope import (  # noqa: E402
    native_arguments,
    torch_arguments,
)

SESSIONS = ("native", "torch")

_COUNT = re.compile(r"(\d+) (passed|failed|skipped|error|errors|xfailed|xpassed)")
_WARMUP_MARKER = "JITTOR_TEST_SUITE_CPU_READY"
_WARMUP_ATTEMPTS = 3

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


def _session_environment(session, serial_compile=False):
    state = _lab_root() / "_state" / "test-suite" / session
    (state / "home").mkdir(parents=True, exist_ok=True)
    (state / "tmp").mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["JITTOR_HOME"] = str(state / "home")
    environment["TMPDIR"] = str(state / "tmp")
    environment["nvcc_path"] = ""
    environment["JITTOR_TEST_DEVICES"] = "cpu"
    environment["REAL_TORCH_SITE"] = ""
    environment["JITTOR_TORCH_SHIM"] = "1" if session == "torch" else "0"
    # Set, never inherited -- like every other name in this function. On by
    # default, like the gate: it used to be forced off here with no reason
    # recorded, which made this script measure something `nox -s cpu` does not
    # run, and the parallel op compiler is where a cold whole-tree run's time
    # goes. ``--serial-compile`` restores the old value for bisecting a compile
    # failure; tests/compiler/test_parallel_compile_attribution.py is why that
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


def _session_arguments(session):
    """The same selection `nox -s cpu` uses, from the same source."""
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
    probe = (
        "import jittor as jt; "
        "assert not jt.compiler.has_cuda; "
        "assert not getattr(jt.compiler, 'has_acl', 0); "
        "jt.flags.use_parallel_op_compiler = 0; "
        "x = (jt.array([1.0, 2.0]) * 2).sum(); x.sync(); "
        "assert float(x.item()) == 6.0; "
        "print(%r)" % _WARMUP_MARKER
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
        if _WARMUP_MARKER in completed.stdout:
            return 0, "\n".join(outputs)
    outputs.append("warmup did not execute the CPU probe after %d attempts" % _WARMUP_ATTEMPTS)
    return 1, "\n".join(outputs)


def _tier_arguments(tier):
    """What the fast tier drops, from the same list the nox session reads."""
    if tier != "smoke":
        return []
    return ["-m", "not slow"]


def _parallel_arguments(jobs):
    """xdist, per file. ``noxfile._xdist`` explains why not per test."""
    if not jobs or jobs <= 1:
        return []
    try:
        import xdist  # noqa: F401
    except ImportError:
        raise SystemExit(
            "--jobs %d needs pytest-xdist (requirements/dev-tools.txt); "
            "running serially instead would report a wall clock for a gate "
            "nobody runs" % jobs)
    return ["-n", str(jobs), "--dist", "loadfile"]


def _run(session, extra, quiet, tier="full", jobs=0, serial_compile=False):
    environment = _session_environment(session, serial_compile=serial_compile)
    command = [PYTHON, "-m", "pytest"]
    command += _session_arguments(session)
    command += ["-p", "no:cacheprovider", "--timeout=900"]
    command += _tier_arguments(tier)
    command += _parallel_arguments(jobs)
    command += ["-q"] if quiet else []
    command += extra
    print("=== {} session ===".format(session), flush=True)
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
    parser.add_argument("--tier", choices=("smoke", "full"), default="full",
                        help="smoke drops the files recorded in "
                             "tests/_helpers/tiers.SLOW_FILES")
    parser.add_argument("--jobs", type=int, default=0,
                        help="xdist workers per session (default: one process)")
    parser.add_argument("--serial-compile", action="store_true",
                        help="use_parallel_op_compiler=0; for bisecting a "
                             "compile failure, not for timing a gate")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("extra", nargs="*", help="extra pytest arguments")
    options = parser.parse_args()

    sessions = options.session or list(SESSIONS)
    results = {}
    failures = []
    for session in sessions:
        code, counts, output = _run(
            session, options.extra, not options.verbose,
            tier=options.tier, jobs=options.jobs,
            serial_compile=options.serial_compile)
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

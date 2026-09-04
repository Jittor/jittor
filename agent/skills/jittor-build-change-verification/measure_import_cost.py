"""Measure and attribute the wall-clock cost of ``import jittor``.

Run it, do not read it, when you need to answer "where did those seconds go".
A profiler answers that question with 40% overhead on this import, which is
enough to reorder the phases; every number here is plain ``perf_counter``.

Three levels, because no single mechanism sees all of it:

* level 1 -- module bodies, from a child process started with
  ``-X importtime``. Cheap and exact, but its unit is "a module", so
  ``jittor.compiler`` shows up as one 1.4 s lump.
* level 2 -- the build fan-out, by patching the ``jittor_utils`` entry points
  that ``compiler.py`` calls. This has to be done from *outside*: the
  interesting work happens in ``compiler.py``'s module body, so by the time
  anything in ``jittor`` is importable it has already run. ``jittor_utils`` is
  a separate package and ``compiler.py`` reaches it by attribute lookup, so a
  patch installed before ``import jittor`` survives.
* level 3 -- the source generators, by calling them again after the import.
  They are idempotent (same inputs, same output files), so a second call
  prices the first.

Usage:

    EXPECT_JITTOR_SRC=<worktree>/python \\
    PYTHONPATH=<worktree>/python JITTOR_HOME=... TMPDIR=... \\
    python agent/skills/jittor-build-change-verification/measure_import_cost.py

``EXPECT_JITTOR_SRC`` is not optional: without it the numbers may well come
from the *installed* jittor rather than the tree you are changing, and there
is no way to tell afterwards. Add ``--json <path>`` to keep a machine-readable
copy for a before/after diff.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time


def level1_module_bodies(expect_src, env_extra=None):
    """Cumulative import time per module, from a child with -X importtime."""
    environment = dict(os.environ)
    # The child must import the same tree; -X importtime says nothing about
    # which jittor it timed.
    environment["PYTHONPATH"] = (expect_src + os.pathsep
                                 + environment.get("PYTHONPATH", ""))
    if env_extra:
        environment.update(env_extra)
    result = subprocess.run(
        [sys.executable, "-X", "importtime", "-c",
         "import jittor, os, sys;"
         "sys.stderr.write('SRCLINE ' + jittor.__file__ + '\\n')"],
        env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    text = result.stderr.decode("utf8", "replace")
    if result.returncode != 0:
        raise RuntimeError("importtime child failed:\n" + text)
    source = re.search(r"^SRCLINE (.*)$", text, re.M)
    if not source or not source.group(1).startswith(expect_src):
        raise RuntimeError("importtime child imported %r, expected a tree "
                           "under %r" % (source and source.group(1),
                                         expect_src))
    modules = {}
    for line in text.splitlines():
        row = re.match(r"import time:\s+(\d+)\s*\|\s*(\d+)\s*\|(\s*)(\S+)",
                       line)
        if row:
            modules[row.group(4)] = {"self": int(row.group(1)) / 1e6,
                                     "cumulative": int(row.group(2)) / 1e6,
                                     "depth": len(row.group(3)) - 1}
    return modules


def level2_and_3(expect_src):
    """Import jittor in this process with the build fan-out instrumented."""
    phases = {}
    calls = []

    def account(key, seconds, count=1):
        entry = phases.setdefault(key, {"s": 0.0, "n": 0})
        entry["s"] += seconds
        entry["n"] += count

    import jittor_utils as jit_utils

    inner_run_cmds = jit_utils.run_cmds
    inner_run_cmd = jit_utils.run_cmd

    def run_cmds(cmds, *args, **kw):
        # Pool creation is one-off and lands on whichever call is first, so
        # record it rather than smearing it over the per-command average.
        created_pool = jit_utils.pool_size == 0
        start = time.perf_counter()
        try:
            return inner_run_cmds(cmds, *args, **kw)
        finally:
            elapsed = time.perf_counter() - start
            calls.append({"msg": kw.get("msg", args[2] if len(args) > 2
                                        else "?"),
                          "commands": len(cmds), "s": round(elapsed, 4),
                          "created_pool": created_pool})
            account("run_cmds", elapsed, len(cmds))

    def run_cmd(*args, **kw):
        start = time.perf_counter()
        try:
            return inner_run_cmd(*args, **kw)
        finally:
            account("run_cmd", time.perf_counter() - start)

    jit_utils.run_cmds = run_cmds
    jit_utils.run_cmd = run_cmd

    start = time.perf_counter()
    import jittor
    total = time.perf_counter() - start

    if not jittor.__file__.startswith(expect_src):
        raise RuntimeError("imported %r, expected a tree under %r"
                           % (jittor.__file__, expect_src))

    import jittor.compiler as compiler
    import jittor.pyjt_compiler as pyjt_compiler

    for name, call in (
        ("gen_jit_flags", compiler.gen_jit_flags),
        ("gen_jit_tests", compiler.gen_jit_tests),
        ("gen_pyjt", lambda: pyjt_compiler.compile(compiler.cache_path,
                                                   compiler.jittor_path)),
    ):
        start = time.perf_counter()
        call()
        account(name, time.perf_counter() - start)

    return {"total": round(total, 4), "phases": phases, "run_cmds": calls,
            "src": os.path.dirname(jittor.__file__),
            "has_cuda": int(jittor.has_cuda),
            "cache_path": compiler.cache_path}


INTERESTING = (
    "jittor", "jittor.compiler", "jittor.compile_extern", "jittor.init_cupy",
    "jittor.compat.triton", "jittor_core", "jittor_mpi_core", "jittor_utils",
    "cupy", "numpy", "jittor.extern.acl.acl_compiler",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="write the raw measurement here")
    options = parser.parse_args()

    expect = os.environ.get("EXPECT_JITTOR_SRC")
    if not expect:
        sys.exit("set EXPECT_JITTOR_SRC=<worktree>/python -- without it this "
                 "may be measuring the installed jittor")
    expect = os.path.abspath(expect)

    modules = level1_module_bodies(expect)
    measured = level2_and_3(expect)
    report = {"level1_module_bodies": modules, "level2_3": measured}

    print("src:        %s" % measured["src"])
    print("cache_path: %s" % measured["cache_path"])
    print("has_cuda:   %s" % measured["has_cuda"])
    print("")
    print("level 1  module bodies (child, -X importtime)")
    for name in INTERESTING:
        row = modules.get(name)
        if row:
            print("  %8.3f s cumulative  %8.3f s self   %s"
                  % (row["cumulative"], row["self"], name))
    print("")
    print("level 2  build fan-out through the compile pool "
          "(this process, %.3f s total import)" % measured["total"])
    for call in measured["run_cmds"]:
        print("  %8.3f s  %4d commands%s  %s"
              % (call["s"], call["commands"],
                 " (+pool)" if call["created_pool"] else "        ",
                 call["msg"]))
    print("")
    print("level 3  source generators, re-invoked")
    for name in ("gen_jit_flags", "gen_jit_tests", "gen_pyjt", "run_cmds",
                 "run_cmd"):
        row = measured["phases"].get(name)
        if row:
            print("  %8.3f s  n=%-4d  %s" % (row["s"], row["n"], name))

    if options.json:
        temporary = options.json + ".tmp"
        with open(temporary, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        os.replace(temporary, options.json)
        print("\nwrote %s" % options.json)


if __name__ == "__main__":
    main()

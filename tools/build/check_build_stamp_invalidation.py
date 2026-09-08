"""Prove that each recorded build input still forces a rebuild.

The build stamps added for 9.01 (``51d0439f`` for the core, ``d23f9bba6`` for
the bundled op libraries and ``libcuda_extern``) turn a warm ``import jittor``
from "hash every translation unit's dependency closure" into "compare one JSON
file and skip". That is where the seconds went, and it is also the dangerous
kind of optimization: the failure mode is not a crash but a *silent* one, where
the stamp says "current", the build step is skipped, and the process keeps
running the previous ``.so`` against edited sources.

``tests/compiler/test_import_bootstrap_laziness.py`` covers the comparison
field by field, in-process, which is the check that notices a field quietly
dropping out of the record. This script covers the other axis: it perturbs the
real tree and the real cache and asserts that a real child process really
recompiles. Four perturbations, one per class of input:

* a ``src/`` source edit          -- the core source walk
* an ``extern/`` source edit      -- the same walk's second root
* a compile ingredient change     -- ``nvcc_flags``, via the generator digest
* a deleted product               -- the ``.so`` the stamp describes

Each one must produce a non-empty compile fan-out. A perturbation that gets
skipped silently is the bug this script exists to catch, so "no fan-out" is
reported as a failure rather than as a pleasant surprise.

Usage (the environment matters; see the worktree's handoff notes):

    JITTOR_HOME=... TMPDIR=... nvcc_path=... \
    PYTHONPATH=<worktree>/python \
    python tools/build/check_build_stamp_invalidation.py

Exits 0 when every perturbation triggered a rebuild, 1 otherwise.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

MARKER = "STAMP_PROBE_RESULT "

# Records every compile fan-out on the import path. ``run_cmds`` is the single
# funnel for them, so an empty list is a trustworthy "compiled nothing" rather
# than an absence of evidence.
PROBE = r"""
import json
import os
import time

import jittor_utils

fanouts = []
_inner = jittor_utils.run_cmds


def run_cmds(cmds, *args, **kw):
    fanouts.append([kw.get("msg", args[2] if len(args) > 2 else "?"),
                    len(cmds)])
    return _inner(cmds, *args, **kw)


jittor_utils.run_cmds = run_cmds

started = time.perf_counter()
import jittor
elapsed = time.perf_counter() - started

print("STAMP_PROBE_RESULT " + json.dumps({
    "fanouts": fanouts,
    "commands": sum(count for _, count in fanouts),
    "elapsed": round(elapsed, 3),
    "src": os.path.dirname(jittor.__file__),
    "has_cuda": int(jittor.has_cuda),
    "core_output": jittor.compiler.core_output_path,
    "cache_path": jittor.compiler.cache_path,
}))
"""


def probe(label):
    """Import jittor in a child and report what it compiled."""
    result = subprocess.run(
        [sys.executable, "-c", PROBE], cwd=REPO_ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    text = result.stdout.decode("utf8", "replace")
    if result.returncode != 0:
        raise SystemExit("[%s] child import failed:\n%s"
                         % (label, text[-4000:]))
    for line in text.splitlines():
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER):])
    raise SystemExit("[%s] child produced no result:\n%s"
                     % (label, text[-4000:]))


class Perturbation(object):
    """A reversible edit to the tree or the cache, plus its restore."""

    def __init__(self, name, apply, restore):
        self.name = name
        self.apply = apply
        self.restore = restore


def append_line(path):
    """Edit a source file the way a developer would: change it, then undo.

    The appended line has to keep the file compilable -- a syntax error would
    make the child fail for a reason that says nothing about the stamp -- so
    the comment marker follows the file's language.
    """
    absolute = os.path.join(REPO_ROOT, path)
    marker = b"#" if absolute.endswith(".py") else b"//"
    with open(absolute, "rb") as handle:
        original = handle.read()

    def apply():
        with open(absolute, "ab") as handle:
            handle.write(b"\n" + marker + b" jittor build-stamp probe\n")

    def restore():
        with open(absolute, "wb") as handle:
            handle.write(original)

    return Perturbation("edit " + path, apply, restore)


def edit_generator_flags():
    """Change a compile ingredient that does not move ``cache_path``.

    ``nvcc_flags`` is the ingredient the handoff notes name, but setting it
    from the environment re-partitions ``cache_path``, which produces a cold
    build in a *different* directory -- correct behaviour, and no evidence at
    all about the stamp. So perturb the same ingredient where the stamp reads
    it: ``compiler.py`` is recorded by content digest precisely because it
    generates the flags and the C++ it compiles.
    """
    return append_line(os.path.join("python", "jittor", "compiler.py"))


def delete_product(state):
    """Remove the ``.so`` a stamp describes, leaving the stamp behind."""
    target = state["core_output"]
    saved = tempfile.mkdtemp(prefix="stamp-probe-")
    kept = os.path.join(saved, os.path.basename(target))

    def apply():
        shutil.move(target, kept)

    def restore():
        # Restore only if the rebuild did not already write a new one; the
        # point is to leave the cache warm either way.
        if not os.path.exists(target):
            shutil.move(kept, target)
        shutil.rmtree(saved, ignore_errors=True)

    return Perturbation("delete " + os.path.basename(target), apply, restore)


def main():
    if not os.environ.get("PYTHONPATH", "").startswith(
            os.path.join(REPO_ROOT, "python")):
        sys.exit("set PYTHONPATH=%s so the child imports this checkout"
                 % os.path.join(REPO_ROOT, "python"))

    print("warming the cache, so the baseline means something")
    warm = probe("warm")
    if warm["src"] != os.path.join(REPO_ROOT, "python", "jittor"):
        sys.exit("child imported %r, not this checkout" % warm["src"])
    print("  baseline: %s in %.3f s, %d command(s) in %d fan-out(s)"
          % ("compiled nothing" if not warm["fanouts"] else "COMPILED",
             warm["elapsed"], warm["commands"], len(warm["fanouts"])))
    if warm["fanouts"]:
        # Not fatal: the first run after a rebase legitimately builds. Re-warm
        # and insist the second one is quiet, or every result below is noise.
        warm = probe("warm-again")
        print("  re-warmed: %d command(s)" % warm["commands"])
    if warm["fanouts"]:
        sys.exit("the cache will not go quiet, so no perturbation below can "
                 "be attributed: %r" % (warm["fanouts"],))
    print("  has_cuda=%d  cache_path=%s" % (warm["has_cuda"],
                                            warm["cache_path"]))

    perturbations = [
        append_line(os.path.join("python", "jittor", "src", "lock.cc")),
        append_line(os.path.join("python", "jittor", "extern", "mkl",
                                 "ops", "mkl_matmul_op.cc")),
        edit_generator_flags(),
        delete_product(warm),
    ]

    failures = []
    for perturbation in perturbations:
        print("\n%s" % perturbation.name)
        perturbation.apply()
        try:
            started = time.perf_counter()
            observed = probe(perturbation.name)
        finally:
            perturbation.restore()
        wall = time.perf_counter() - started
        if observed["commands"]:
            print("  REBUILT: %d command(s) in %.1f s -- %s"
                  % (observed["commands"], wall,
                     ", ".join("%s(%d)" % (msg, count)
                               for msg, count in observed["fanouts"])))
        else:
            print("  SILENTLY SKIPPED after %.1f s -- stamp claimed current"
                  % wall)
            failures.append(perturbation.name)
        # Leave the cache warm for the next perturbation, so each one is
        # measured against "nothing to do" rather than against the last edit.
        probe("re-warm after " + perturbation.name)

    print("")
    if failures:
        print("FAIL: %d perturbation(s) did not trigger a rebuild:" %
              len(failures))
        for name in failures:
            print("  %s" % name)
        return 1
    print("OK: all %d perturbations triggered a rebuild" % len(perturbations))
    return 0


if __name__ == "__main__":
    sys.exit(main())

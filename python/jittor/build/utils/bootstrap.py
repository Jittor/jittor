# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Build everything ``import jittor`` needs, on purpose rather than by accident.

    python -m jittor_utils.bootstrap            # build whatever is missing
    python -m jittor_utils.bootstrap --check    # build nothing; is it ready?

``import jittor`` compiles whatever the cache is missing. That is convenient
once and wrong every time after: it puts a forty-second to several-minute
build inside an import, where nothing expects one -- a web worker starting up,
a test collecting, a read-only container that will fail at the end of the
build for a reason unrelated to the build. ``JITTOR_NO_BUILD=1`` makes an
import refuse to compile and say so immediately; this is the counterpart that
is allowed to compile, so the two together turn "does importing jittor build
anything" into a question with an enforceable answer.

Two design points that are not obvious:

*Why a child process rather than ``import jittor`` here.* ``jittor_utils`` is
the build-tooling layer that ``jittor.compiler`` imports, so it must not
import the runtime back -- ``tests/structure/test_build_config_boundaries.py``
fails the gate on exactly that. Running the import in a child keeps the
dependency one-way, and pays for itself twice over: the child is where the
build's authorisation belongs, and a build that ends in ``jit_utils was
rebuilt ... rerun the same command`` can simply be rerun here instead of
becoming the caller's problem.

*Why not ``python -m jittor.bootstrap``.* That would run the ``jittor``
package body first, so the build it is supposed to authorise would already
have happened by the time the module got control.

Every build input is part of the cache key, so bootstrap with the same
``JITTOR_HOME`` and the same flags (``nvcc_path``, ``cc_flags``,
``nvcc_flags``) the real process will use -- a CUDA build and a CPU-only build
are two different caches, and bootstrapping one does nothing for the other.
"""

import argparse
import os
import subprocess
import sys
import time

#: Run in the child. Importing proves the core loads; it does not prove the
#: core computes, and a cache that dlopens but produces wrong answers is
#: exactly what a bootstrap step should catch before anything depends on it.
_PROBE = """
import os, jittor
value = (jittor.ones(3) * 2).sum().item()
print("JITTOR_SRC " + os.path.dirname(jittor.__file__))
print("CACHE_PATH " + jittor.compiler.cache_path)
print("HAS_CUDA %d" % int(bool(jittor.has_cuda)))
print("VALUE %r" % (value,))
"""

#: ``jit_utils_core`` is built before any code in ``jittor`` runs, so it is
#: outside the ``JITTOR_NO_BUILD`` gate: a fresh build configuration rebuilds
#: it and exits asking for a rerun. That is by design (see the 0.11 note in
#: ``compiler.py``), and doing what the message says is this module's job.
_RERUN = "rerun the same command"


def _field(output, name):
    for line in output.splitlines():
        if line.startswith(name + " "):
            return line[len(name) + 1:]
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m jittor_utils.bootstrap",
        description="Build the jittor core and its bundled op libraries.")
    parser.add_argument(
        "--check", action="store_true",
        help="build nothing; exit non-zero if an import would have to build")
    options = parser.parse_args(argv)

    environment = dict(os.environ)
    # Authorise the child explicitly. Inheriting JITTOR_NO_BUILD=1 would make
    # bootstrap refuse to do the one job it has -- and the environment that
    # sets that variable is precisely the one that needs bootstrapping.
    environment["JITTOR_NO_BUILD"] = "1" if options.check else "0"
    # Pin the child to the checkout this module belongs to. Without it, a
    # development tree run with PYTHONPATH would bootstrap the cache of
    # whichever jittor the bare interpreter resolves -- usually the editable
    # install, i.e. some other working tree -- and report success for a cache
    # the caller will never use.
    tree = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    existing = [part for part in environment.get("PYTHONPATH", "").split(
        os.pathsep) if part and part != tree]
    environment["PYTHONPATH"] = os.pathsep.join([tree] + existing)

    started = time.perf_counter()
    for _ in range(2):
        child = subprocess.run([sys.executable, "-c", _PROBE],
                               env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
        output = child.stdout.decode("utf-8", "replace")
        if _RERUN not in output:
            break
        print("jit_utils was rebuilt; rerunning", file=sys.stderr)
    elapsed = time.perf_counter() - started

    if child.returncode != 0:
        # Under --check this is the expected outcome for an unbuilt cache, and
        # the child's own message already names what is missing and what to
        # run. Pass it through rather than summarising it away.
        sys.stderr.write(output)
        return 1

    value = _field(output, "VALUE")
    if value != "6.0":
        sys.stderr.write(output)
        print("bootstrap produced a core that computes incorrectly: "
              "expected 6.0, got %s" % value, file=sys.stderr)
        return 1

    print("jittor:     %s" % _field(output, "JITTOR_SRC"))
    print("cache_path: %s" % _field(output, "CACHE_PATH"))
    print("has_cuda:   %s" % _field(output, "HAS_CUDA"))
    print("%s in %.1fs" % ("verified" if options.check else "bootstrapped",
                           elapsed))
    return 0


if __name__ == "__main__":
    sys.exit(main())

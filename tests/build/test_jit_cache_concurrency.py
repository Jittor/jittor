"""Several processes reading and writing one JIT cache at the same time.

``jit_compiler::compile()`` answers a warm cache without taking the build lock:
it compares the key the compile command would produce against the key recorded
in ``<product>.key`` and, when they agree, ``dlopen``s the product. Nothing
serialises that read against a build running in another process, so the whole
thing rests on one property of how a build is published:

    ``cache_compile()`` renames the finished product into place **first** and
    writes ``<product>.key`` **second**, and both are ``rename()`` within one
    directory, which is atomic.

So "I can see the matching key" implies "the directory entry next to it already
points at the complete product of that build". If either half of that stops
holding -- a product written in place instead of renamed, a key published before
the product it describes, a key left behind by a product that was removed --
then a reader can map half a shared library, and the failure will look like a
corrupt kernel or a missing symbol somewhere else entirely.

The tests here are the ones that fail when it stops holding:

``test_publish_order_never_exposes_a_key_without_its_product``
    watches a slow build from another thread and fails the moment a sample
    shows the final key next to anything but the final product.
``test_a_key_without_its_product_is_not_a_hit``
    removes the product and keeps the key: the probe must not call that a hit,
    because the fast path would go straight to ``dlopen`` on a missing file.
``test_probe_and_cache_compile_agree_on_every_transition``
    pins the probe to the decision ``cache_compile()`` itself makes, so the
    unlocked pre-check can never say "cached" where the locked path would build.
``test_concurrent_cold_compiles_agree``
    the end-to-end one: eight processes cold-compile overlapping kernel sets
    into one shared cache, over several rounds, and every one of them has to
    come back with the same numbers.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

import pytest

from _helpers.child_process import run_child_script


REPO_ROOT = Path(__file__).resolve().parents[2]
JITTOR = REPO_ROOT / "python" / "jittor"
SRC = REPO_ROOT / "src"


@pytest.fixture(scope="module")
def probe_harness(tmp_path_factory):
    """A binary that runs one ``cache_compile`` step, or just the probe.

    Linking the two functions on their own keeps these tests away from a built
    Jittor core: what is under test is the cache protocol in
    ``src/utils/cache_compile.cc``, not any op.
    """
    directory = tmp_path_factory.mktemp("cache-probe-harness")
    source = directory / "main.cc"
    source.write_text(
        """
#include <cstdio>
#include <string>
#include "utils/cache_compile.h"

// argv[1] is "compile" or "probe"; the rest is the command, cache path and
// jittor path cache_compile() takes.
int main(int argc, char** argv) {
    if (argc != 5) return 64;
    std::string mode = argv[1];
    if (mode == "probe") {
        auto probe = jittor::jit_compiler::cache_compile_probe(argv[2]);
        printf("%s\\n", probe.up_to_date ? "cached" : "stale");
        return 0;
    }
    bool ran = jittor::jit_compiler::cache_compile(argv[2], argv[3], argv[4]);
    printf("%s\\n", ran ? "built" : "cached");
    return 0;
}
""",
        encoding="utf-8",
    )
    executable = directory / "cache_probe_harness"
    subprocess.run(
        [
            os.environ.get("CXX", "g++"),
            "-std=c++14",
            "-I" + str(SRC),
            str(source),
            str(SRC / "utils" / "log.cc"),
            str(SRC / "utils" / "tracer.cc"),
            str(SRC / "utils" / "str_utils.cc"),
            str(SRC / "utils" / "cache_compile.cc"),
            "-lpthread",
            "-ldl",
            "-o",
            str(executable),
        ],
        check=True,
    )
    return executable


def _fake_compiler(path, body):
    """A stand-in compiler: writes a depfile, then the product ``body`` says."""
    path.write_text(
        """#!/usr/bin/env python3
import sys
import time

args = sys.argv[1:]
output = args[args.index("-o") + 1]
source = args[0]
if "-MF" in args:
    with open(args[args.index("-MF") + 1], "w") as handle:
        handle.write(output + ": " + source + "\\n")
%s
""" % body,
        encoding="utf-8",
    )
    path.chmod(0o755)


PRODUCT = b"a complete shared library" * 4096


def _run_harness(harness, mode, command, tmp_path):
    return subprocess.run(
        [str(harness), mode, command, str(tmp_path), str(JITTOR)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _workspace(tmp_path, name):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "obj_files").mkdir()
    source = directory / "kernel.cc"
    source.write_text("int kernel = 1;\n", encoding="utf-8")
    return directory, source, directory / "kernel.so"


def test_probe_and_cache_compile_agree_on_every_transition(
        probe_harness, tmp_path):
    """The unlocked probe must never claim a hit where the build would run.

    A probe that says "cached" too eagerly is the dangerous direction: the fast
    path skips the lock *and* the build on the strength of it. So each state is
    checked against what ``cache_compile()`` then does -- ``built`` or
    ``cached`` -- rather than against an expectation written out by hand.
    """
    directory, source, output = _workspace(tmp_path, "agree")
    compiler = tmp_path / "fake-compiler"
    _fake_compiler(compiler, "open(output, 'wb').write(%r)" % PRODUCT)
    command = "%s %s -o %s" % (compiler, source, output)

    def probe():
        result = _run_harness(probe_harness, "probe", command, directory)
        assert result.returncode == 0, result.stdout
        return result.stdout.strip()

    def build():
        result = _run_harness(probe_harness, "compile", command, directory)
        assert result.returncode == 0, result.stdout
        return result.stdout.strip()

    def both(note):
        """What the probe says, and what cache_compile() then does."""
        said = probe()
        did = build()
        assert (said == "cached") == (did == "cached"), \
            "%s: probe said %r, cache_compile did %r" % (note, said, did)
        return said

    assert both("nothing built yet") == "stale"
    assert both("the same command again") == "cached"

    # An input that changed has to be seen by both.
    source.write_text("int kernel = 2;\n", encoding="utf-8")
    assert both("the source changed") == "stale"
    assert both("and settled again") == "cached"

    # And so does a command that changed, even with the same inputs.
    other = "%s %s -DEXTRA -o %s" % (compiler, source, output)
    other_probe = _run_harness(probe_harness, "probe", other, directory)
    assert other_probe.stdout.strip() == "stale", other_probe.stdout


def test_a_key_without_its_product_is_not_a_hit(probe_harness, tmp_path):
    """A key whose product is gone must read as "build", not as "cached".

    ``clean_cache``, a full disk or a stray ``rm`` can leave the key behind.
    The fast path answers a hit by calling ``dlopen`` straight away, so a probe
    that trusted the key alone would hand it a path that is not there.
    """
    directory, source, output = _workspace(tmp_path, "orphan-key")
    compiler = tmp_path / "fake-compiler-2"
    _fake_compiler(compiler, "open(output, 'wb').write(%r)" % PRODUCT)
    command = "%s %s -o %s" % (compiler, source, output)

    assert _run_harness(probe_harness, "compile", command, directory
                        ).stdout.strip() == "built"
    assert _run_harness(probe_harness, "probe", command, directory
                        ).stdout.strip() == "cached"

    key = Path(str(output) + ".key")
    recorded = key.read_text()
    output.unlink()
    assert key.read_text() == recorded, "the key itself must be untouched"

    assert _run_harness(probe_harness, "probe", command, directory
                        ).stdout.strip() == "stale"


def test_publish_order_never_exposes_a_key_without_its_product(
        probe_harness, tmp_path):
    """Sample a rebuild from outside: the new key must never lead the product.

    This is the property the unlocked fast path is built on, so it is checked
    by watching rather than by reading the source. The stand-in compiler takes
    long enough that a poller gets thousands of samples across the window in
    which the product is replaced and the key rewritten.

    The interesting build is the *second* one: the first leaves an older
    product and an older key for it to replace. A reader that sees the old pair
    simply builds, so that state is harmless; the one that cannot be recovered
    from is the new key sitting next to a product that is still the old one, or
    worse, half of the new one. Each sample reads the key first and the product
    second, so a product read after a new key is guaranteed to be at least as
    new as the key -- the check has no window of its own.
    """
    directory, source, output = _workspace(tmp_path, "publish-order")
    compiler = tmp_path / "fake-compiler-slow"
    # The product is derived from the source, so the two builds are told apart
    # by content rather than by timing.
    _fake_compiler(
        compiler,
        "time.sleep(0.6)\n"
        "body = open(source, 'rb').read()\n"
        "open(output, 'wb').write(body * 4096)")
    command = "%s %s -o %s" % (compiler, source, output)

    assert _run_harness(probe_harness, "compile", command, directory
                        ).stdout.strip() == "built"
    key_path = Path(str(output) + ".key")
    first_key = key_path.read_text()
    first_product = output.read_bytes()

    source.write_text("int kernel = 3;\n", encoding="utf-8")
    second_product = source.read_bytes() * 4096

    violations = []
    samples = [0]
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            try:
                key = key_path.read_text()
                product = output.read_bytes()
            except OSError:
                continue
            samples[0] += 1
            if key not in (first_key, ""):
                # The key is no longer the one the first build published, so
                # the second build has committed: its product must be there
                # whole.
                if product != second_product:
                    violations.append(
                        ("key led product", key[:40], len(product)))
            elif product not in (first_product, second_product):
                violations.append(("partial product", len(product)))

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()
    try:
        result = _run_harness(probe_harness, "compile", command, directory)
    finally:
        stop.set()
        poller.join(timeout=10)
    assert result.stdout.strip() == "built", result.stdout
    assert output.read_bytes() == second_product
    assert key_path.read_text() != first_key
    assert samples[0] > 100, "the poller barely sampled the window: %d" % samples[0]
    assert not violations, violations[:8]
    assert not list(directory.glob("*.tmp.*"))


# --------------------------------------------------------------------------
# The end-to-end race: real processes, real kernels, one shared cache.
# --------------------------------------------------------------------------

#: Executed by every child. ``rotation`` shifts the order the kernels are
#: reached in, so the processes do not march through the cache in lockstep:
#: each one compiles kernels the others are already reading and reads kernels
#: the others are still compiling, which is the overlap under test.
#: Executed by every child. The cases it walks are handed to it in the
#: environment, already rotated, so each process reaches the kernels in a
#: different order: every one of them compiles kernels the others are already
#: reading and reads kernels the others are still compiling, which is the
#: overlap under test.
CHILD = r'''
import json
import os

import numpy as np
import jittor as jt

jt.flags.use_cuda = 0

# Passed in the environment rather than on the command line: run_child_script
# runs the file with no arguments.
cases = json.loads(os.environ["JITTOR_RACE_CASES"])

results = {}
for shape, dtype in cases:
    count = shape[0] * shape[1]
    base = np.arange(count, dtype=dtype).reshape(shape)
    x = jt.array(base)
    y = jt.maximum(x * 3 + 1, 2)
    value = y.sum() + y.transpose().sum() + (y * y).sum()
    results["%s%s" % (dtype, shape)] = float(value.numpy())
print("RESULT " + json.dumps(results, sort_keys=True))
'''

#: Small kernels, but each (shape, dtype) pair is its own JIT key, so there are
#: enough of them for eight processes to collide over.
CASES = [[list(shape), dtype]
         for dtype in ("float32", "float64", "int32", "int64")
         for shape in ((6, 7), (5, 5), (9, 3), (4, 8))]

WORKERS = 8
ROUNDS = 3


def _child_env(home, cache_name, rotation=0):
    rotated = CASES[rotation:] + CASES[:rotation]
    return {
        "JITTOR_HOME": str(home),
        "cache_name": cache_name,
        "JITTOR_RACE_CASES": json.dumps(rotated),
        # A compile that needs the lock must not wait out a stuck holder for
        # half an hour: inside this test that is the bug, not a slow machine.
        "JT_LOCK_TIMEOUT": "600",
        "use_parallel_op_compiler": "4",
    }


def _run_round(source_directory, home, cache_name, rotations):
    """Start every child at once and collect what each of them printed."""
    outcomes = [None] * len(rotations)

    def run(index, rotation):
        outcomes[index] = run_child_script(
            CHILD + "\n", env=_child_env(home, cache_name, rotation),
            directory=source_directory, name="race%d" % index,
            text=True, merge_stderr=True, timeout=1800)

    threads = [threading.Thread(target=run, args=(index, rotation))
               for index, rotation in enumerate(rotations)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return outcomes


def _results_of(outcome, label):
    assert outcome is not None, "%s never ran" % label
    assert outcome.returncode == 0, "%s failed:\n%s" % (label, outcome.stdout)
    lines = [line for line in outcome.stdout.splitlines()
             if line.startswith("RESULT ")]
    assert len(lines) == 1, "%s printed no result:\n%s" % (label, outcome.stdout)
    return json.loads(lines[0][len("RESULT "):])


def _jit_products(home):
    return sorted(Path(home).rglob("jit/*.so"))


@pytest.mark.slow
@pytest.mark.cpu
def test_concurrent_cold_compiles_agree(tmp_path):
    """Eight processes, one cache, overlapping kernels, repeated.

    Round one is the cold one: every kernel is compiled by whichever process
    reaches it first while the others are reading the same directory. Later
    rounds are warm, which is the case the fast path exists for -- every child
    decides "cached" without the lock and maps a product some other process
    published. A half-written product would show up as a child that died in
    ``dlopen``, a missing symbol, or a number that disagrees with the rest.
    """
    home = tmp_path / "home"
    home.mkdir()
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    cache_name = "racecache"

    # One serial child first. It pays for the core build, which is not what
    # this test is about, and its numbers are the reference the racing children
    # are compared against.
    reference_run = run_child_script(
        CHILD + "\n", env=_child_env(home, cache_name, 0), directory=scripts,
        name="reference", text=True, merge_stderr=True, timeout=1800)
    reference = _results_of(reference_run, "reference child")
    assert reference, "the child computed nothing"

    for round_index in range(ROUNDS):
        # A fresh JIT directory for the first round makes it a real cold race;
        # the core built above is kept, so the race is over op kernels.
        if round_index == 0:
            for products in {path.parent for path in _jit_products(home)}:
                shutil.rmtree(products)

        # Distinct rotations, so no two workers walk the kernels in the
        # same order and the overlap is genuinely staggered.
        rotations = [(worker * 3) % len(CASES) for worker in range(WORKERS)]
        outcomes = _run_round(scripts, home, cache_name, rotations)
        for worker, outcome in enumerate(outcomes):
            label = "round %d worker %d" % (round_index, worker)
            assert _results_of(outcome, label) == reference, \
                "%s disagreed:\n%s" % (label, outcome.stdout)

    products = _jit_products(home)
    assert products, "no kernels were compiled at all"
    for product in products:
        key = Path(str(product) + ".key")
        assert key.is_file() and key.stat().st_size, \
            "%s has no recorded key" % product
        with open(product, "rb") as handle:
            assert handle.read(4) == b"\x7fELF", \
                "%s is not a complete shared object" % product
    leftovers = [str(path) for path in Path(home).rglob("*.tmp.*")]
    assert not leftovers, "temporary build products were left behind: %s" % \
        leftovers[:8]

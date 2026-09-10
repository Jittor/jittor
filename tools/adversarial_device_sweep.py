#!/usr/bin/env python3
"""Run every OpInfo operator on adversarial inputs and compare CPU against CUDA.

The hand-written device-agreement checks covered seven operators and found
three disagreements, two of them Critical. This is the same question asked of
the whole database instead: for each operator, evaluate it on inputs built from
the values where implementations diverge -- NaN, both infinities, both signed
zeros, ties, and a subnormal -- and report where the two devices answer
differently.

Comparing the devices against *each other* rather than against NumPy is what
makes it cheap to run over hundreds of operators: it needs no per-operator
reference, and a disagreement is a defect on one side whichever side that is.
`tests/backends/parity` asks this question already, but from inputs drawn to be
ordinary, which is why the divergences it exists to catch survived in it.

Usage::

    PYTHONPATH=<repo>/python python tools/adversarial_device_sweep.py [--json out.json]
"""

import argparse
import json
import pathlib
import sys

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: The values implementations disagree about. Two vectors rather than one,
#: because they probe different mechanisms: the first is about what the format
#: cannot represent (non-finite, signed zero, subnormal), the second about what
#: the *arithmetic* loses (magnitudes far enough apart that addition drops one
#: of them, values at the edge of the integer range, exact ties). A defect
#: reachable by only one of them would look like an absence in the other.
VECTORS = {
    "nonfinite": np.array(
        [np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, -1.0, 1e-45, 3.0, 3.0],
        dtype="float32"),
    # float16 reaches its own limits at ordinary magnitudes: it overflows above
    # 65504 and goes subnormal below 6.1e-5, both of which a float32 kernel
    # would never notice. Values are chosen to sit either side of each edge, so
    # a kernel that computes in float32 and narrows at the end answers
    # differently from one that computes in float16 throughout.
    "half": np.array(
        [65504.0, 65520.0, 60000.0, 6.0e-5, 6.0e-8, 1.0, -1.0, 2048.0,
         2049.0, 2049.0],
        dtype="float16"),
    "magnitude": np.array(
        [1e30, 1e-30, 1.0, -1.0, 16777216.0, 16777217.0, 2147483647.0,
         -2147483648.0, 0.5, 0.5],
        dtype="float32"),
}

#: Selected at run time; kept as a module global so the evaluator stays simple.
ADVERSARIAL = VECTORS["nonfinite"]


def _load_ops():
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    from opinfo.database import op_db
    return op_db


def _evaluate(jt, operator, use_cuda, arity):
    # The vector carries its own dtype: the half vector is only meaningful as
    # float16, and building it as float32 would move every edge it probes.
    with jt.flag_scope(use_cuda=use_cuda):
        args = [jt.array(ADVERSARIAL) for _ in range(arity)]
        if arity == 2:
            args[1] = jt.array(np.roll(ADVERSARIAL, 3))
        out = operator(*args)
        return np.asarray(out.numpy(), dtype=np.float64)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json")
    parser.add_argument("--vector", choices=sorted(VECTORS), default="nonfinite",
                        help="which adversarial input to sweep with")
    args = parser.parse_args(argv)

    global ADVERSARIAL
    ADVERSARIAL = VECTORS[args.vector]
    print("vector: %s" % args.vector)

    op_db = _load_ops()
    import jittor as jt
    if not jt.compiler.has_cuda:
        print("no accelerator: this sweep compares two devices and needs both")
        return 2

    # Each operator is announced before it runs and the file is flushed, because
    # a crash here takes the process with it: the first run of this sweep ended
    # in `Segfault, exit` with no way to tell which of 231 operators did it. A
    # sweep that cannot name its own casualty is a sweep that has to be run
    # again from the start.
    progress = None
    if args.json:
        progress = pathlib.Path(args.json).with_suffix(".progress")

    def announce(name):
        if progress is None:
            return
        with progress.open("a", encoding="utf-8") as handle:
            handle.write(name + "\n")
            handle.flush()

    # Resume rather than restart. An operator that segfaults takes the process
    # with it, so a sweep that always begins at the front can never get past the
    # first crasher -- `take` stopped this one at 47 of 231. The progress file
    # is the record of what has been attempted; anything already in it is
    # skipped, so re-running walks forward through the crashes one at a time
    # instead of repeating the work before them. Per-operator subprocesses would
    # also work and cost a jittor import each; this costs one restart per crash.
    attempted = set()
    if progress is not None and progress.is_file():
        attempted = {line.strip() for line in
                     progress.read_text(encoding="utf-8").splitlines() if line.strip()}
        if attempted:
            print("resuming: %d operator(s) already attempted" % len(attempted))

    disagree, agree, unprobed = [], 0, 0
    for info in op_db:
        if info.name in attempted:
            continue
        announce(info.name)
        try:
            operator = info.op
        except Exception:
            unprobed += 1
            continue
        if operator is None:
            unprobed += 1
            continue
        for arity in (1, 2):
            try:
                cpu = _evaluate(jt, operator, 0, arity)
                gpu = _evaluate(jt, operator, 1, arity)
            except Exception:
                continue
            if cpu.shape != gpu.shape:
                disagree.append({"op": info.name, "arity": arity,
                                 "why": "shape %s vs %s" % (cpu.shape, gpu.shape)})
            elif not np.allclose(cpu, gpu, rtol=1e-5, atol=1e-6, equal_nan=True):
                bad = int((~np.isclose(cpu, gpu, rtol=1e-5, atol=1e-6,
                                       equal_nan=True)).sum())
                disagree.append({
                    "op": info.name, "arity": arity,
                    "why": "%d of %d elements differ" % (bad, cpu.size),
                    "cpu": cpu.tolist(), "cuda": gpu.tolist(),
                })
            else:
                agree += 1
            break
        else:
            unprobed += 1

    if progress is not None and attempted:
        print("note: %d operator(s) were skipped as already attempted; a name "
              "that appears in the progress file but in no result is one that "
              "ended the process." % len(attempted))
    print("operators compared=%d  disagreements=%d  unprobed=%d"
          % (agree + len(disagree), len(disagree), unprobed))
    for row in disagree:
        print("  %-26s %s" % (row["op"], row["why"]))
    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps({"disagree": disagree, "agree": agree,
                        "unprobed": unprobed}, indent=2, sort_keys=True),
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

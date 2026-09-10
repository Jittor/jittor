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

#: The values implementations disagree about, in one vector.
ADVERSARIAL = np.array(
    [np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, -1.0, 1e-45, 3.0, 3.0],
    dtype="float32")


def _load_ops():
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    from opinfo.database import op_db
    return op_db


def _evaluate(jt, operator, use_cuda, arity):
    with jt.flag_scope(use_cuda=use_cuda):
        args = [jt.array(ADVERSARIAL) for _ in range(arity)]
        if arity == 2:
            args[1] = jt.array(np.roll(ADVERSARIAL, 3))
        out = operator(*args)
        return np.asarray(out.numpy(), dtype=np.float64)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json")
    args = parser.parse_args(argv)

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

    disagree, agree, unprobed = [], 0, 0
    for info in op_db:
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

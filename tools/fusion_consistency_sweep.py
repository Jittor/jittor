#!/usr/bin/env python3
"""Fusing an expression must not change its answer.

Every other sweep here needs to know what the right answer is. This one does
not: it evaluates the same expression twice, once with the fusion pass enabled
and once with ``no_fuse`` set, and compares the two. Whichever is right, they
have to agree -- an optimisation that changes results is not an optimisation.

That makes it cheap to point at expressions no reference exists for, and it is
the one invariant that reaches the fusion pass at all. It is also not
hypothetical: a cancellation check written for the semantic probe returned
``1.0`` when other work shared its graph and ``0.0`` when run alone, on the same
build. That observation had no stable expectation to assert against and was
withdrawn; stated as "fused and unfused must match" it becomes assertable
without knowing which value is correct.

Expressions are built to give the fusion pass something to do -- chains of
elementwise operations, reductions over them, and mixtures of magnitudes and
special values where reassociation is visible rather than in the last bit.

Usage::

    PYTHONPATH=<repo>/python python tools/fusion_consistency_sweep.py [--device cuda]
"""

import argparse
import json
import pathlib

import numpy as np


def build_cases(jt):
    """``(name, callable)`` pairs; each takes the input Vars and returns a Var."""
    return [
        ("chain add-mul-sub", lambda a, b: ((a + b) * a - b)),
        ("cancellation", lambda a, b: ((a + b) - a)),
        ("reduction of a chain", lambda a, b: ((a * b + a) - b).sum()),
        ("mean of a product", lambda a, b: (a * b).mean()),
        ("max of a difference", lambda a, b: (a - b).max()),
        ("min of a sum", lambda a, b: (a + b).min()),
        ("nested arithmetic", lambda a, b: (((a * 2.0) + (b * 3.0)) / 2.0 - a)),
        ("divide then multiply", lambda a, b: (a / b) * b),
        ("subtract self via chain", lambda a, b: (a + b) - (b + a)),
        ("abs of a difference", lambda a, b: jt.abs(a - b).sum()),
        ("exp of a scaled sum", lambda a, b: jt.exp((a + b) * 1e-3).sum()),
        ("sqrt of squares", lambda a, b: jt.sqrt(a * a + b * b).sum()),
    ]


#: Inputs where reassociation shows up above the last bit: magnitudes far
#: enough apart that addition order matters, plus the special values that the
#: device sweep found every divergence in.
def inputs():
    ordinary = np.array([1e8, 1.0, -1e8, 1.0, 3.0, 3.0, 0.5, -0.5], dtype="float32")
    special = np.array([1e8, 1.0, 2.0, 4.0, 0.25, 8.0, 16.0, 0.125], dtype="float32")
    return ordinary, special


def evaluate(jt, fn, a_np, b_np, no_fuse, use_cuda):
    with jt.flag_scope(use_cuda=use_cuda, no_fuse=no_fuse):
        a, b = jt.array(a_np), jt.array(b_np)
        return np.asarray(fn(a, b).numpy(), dtype=np.float64)


def opinfo_cases(jt):
    """One case per OpInfo entry, wrapped so fusion has an expression to fuse.

    A bare `op(x)` is a single kernel with nothing to fuse into, so the two
    evaluations would be trivially identical and the sweep would report a clean
    bill of health it had not earned. Each operator is therefore placed inside a
    small chain -- `op(a + b) * 2 - b` -- which is what a real graph looks like
    and what gives the pass something to do.
    """
    import sys, pathlib as _pl
    sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "tests"))
    from opinfo.database import op_db
    cases = []
    for info in op_db:
        try:
            operator = info.op
        except Exception:
            continue
        if operator is None:
            continue
        for arity in (1, 2):
            if arity == 1:
                fn = (lambda o: lambda a, b: o(a + b) * 2.0 - b)(operator)
            else:
                fn = (lambda o: lambda a, b: o(a + b, b) * 2.0 - a)(operator)
            cases.append(("%s/%d" % (info.name, arity), fn))
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--json")
    parser.add_argument("--all-ops", action="store_true",
                        help="sweep every OpInfo entry instead of the written cases")
    parser.add_argument("--progress",
                        help="append each case name before running it, so a "
                             "crash names its own casualty")
    args = parser.parse_args(argv)

    import jittor as jt
    use_cuda = 1 if args.device == "cuda" else 0
    a_np, b_np = inputs()

    cases = opinfo_cases(jt) if args.all_ops else build_cases(jt)

    attempted = set()
    progress = pathlib.Path(args.progress) if args.progress else None
    if progress is not None and progress.is_file():
        attempted = {l.strip() for l in
                     progress.read_text(encoding="utf-8").splitlines() if l.strip()}

    rows, differ = [], 0
    for name, fn in cases:
        if name in attempted:
            continue
        if progress is not None:
            with progress.open("a", encoding="utf-8") as h:
                h.write(name + "\n"); h.flush()
        try:
            fused = evaluate(jt, fn, a_np, b_np, 0, use_cuda)
            plain = evaluate(jt, fn, a_np, b_np, 1, use_cuda)
        except Exception as exc:  # noqa: BLE001 - a failure is a result
            rows.append({"case": name, "status": "ERROR",
                         "detail": "%s: %s" % (type(exc).__name__, str(exc)[:100])})
            continue
        same = (fused.shape == plain.shape and
                np.allclose(fused, plain, rtol=0, atol=0, equal_nan=True))
        close = (fused.shape == plain.shape and
                 np.allclose(fused, plain, rtol=1e-6, atol=1e-6, equal_nan=True))
        if same:
            rows.append({"case": name, "status": "IDENTICAL"})
        elif close:
            # Worth separating: a last-bit difference is reassociation doing
            # what it is allowed to do, and a large one is a different answer.
            rows.append({"case": name, "status": "CLOSE",
                         "fused": fused.tolist(), "unfused": plain.tolist()})
        else:
            differ += 1
            rows.append({"case": name, "status": "DIFFERENT",
                         "fused": fused.tolist(), "unfused": plain.tolist()})

    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print("device=%s cases=%d %s" % (args.device, len(rows), counts))
    for row in rows:
        if row["status"] in ("DIFFERENT", "CLOSE", "ERROR"):
            print("  [%s] %-26s %s" % (
                row["status"], row["case"],
                row.get("detail") or "fused=%s unfused=%s"
                % (row.get("fused"), row.get("unfused"))))
    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps({"device": args.device, "counts": counts, "rows": rows},
                       indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

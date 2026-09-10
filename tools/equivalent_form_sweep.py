#!/usr/bin/env python3
"""Two spellings of the same quantity must produce the same value.

The sweeps that worked this session all share one shape: they assert that two
ways of computing something agree, without needing to know what the right answer
is. The one that failed -- comparing analytic gradients against a difference
quotient -- needed an external reference, and every finding it produced was the
reference being wrong rather than the code. So this one keeps to the shape that
worked.

Each case is a pair of expressions that are mathematically identical and take
different paths through the implementation: a fused operator against the
composition it stands for, a reduction over a contiguous tensor against the same
reduction over a transposed view, a library function against its textbook
definition. A disagreement means one of the two paths is wrong, and it is
visible without deciding which.

Tolerances are per-pair rather than global. Some identities are exact in
floating point (`x.T.T` is the same buffer) and are held to it; others
legitimately differ in the last bits because the two paths accumulate in a
different order, and holding those to equality would report arithmetic as a
defect.

Usage::

    PYTHONPATH=<repo>/python python tools/equivalent_form_sweep.py [--device cuda]
"""

import argparse
import json
import pathlib

import numpy as np


def cases(jt):
    """``(name, left, right, tolerance)``; both callables take one 2-D Var."""
    def softmax_manual(x):
        shifted = x - jt.max(x, 1, keepdims=True)
        e = jt.exp(shifted)
        return e / jt.sum(e, 1, keepdims=True)

    return [
        # Exact: the same elements in the same order.
        ("transpose twice is identity", lambda x: x, lambda x: jt.transpose(x, (1, 0)).transpose(1, 0), 0.0),
        ("reshape round trip", lambda x: x, lambda x: x.reshape(-1).reshape(x.shape), 0.0),
        ("negate twice", lambda x: x, lambda x: -(-x), 0.0),
        ("subtract as add of negative", lambda x: x - x, lambda x: x + (-x), 0.0),

        # Reassociation is allowed to move the last bits.
        ("sum of a transpose", lambda x: jt.sum(x), lambda x: jt.sum(jt.transpose(x, (1, 0))), 1e-5),
        ("mean as sum over count", lambda x: jt.mean(x),
         lambda x: jt.sum(x) / float(x.numel()), 1e-5),
        ("sum along both axes", lambda x: jt.sum(x),
         lambda x: jt.sum(jt.sum(x, 1), 0), 1e-5),
        ("square via multiply", lambda x: x * x, lambda x: x ** 2, 1e-6),
        ("double via add", lambda x: x * 2.0, lambda x: x + x, 0.0),

        # A fused operator against the composition it replaces.
        ("softmax against its definition", lambda x: jt.nn.softmax(x, dim=1),
         softmax_manual, 1e-5),
        ("matmul against broadcast-multiply-sum",
         lambda x: jt.matmul(x, jt.transpose(x, (1, 0))),
         lambda x: jt.sum(x.unsqueeze(1) * x.unsqueeze(0), 2), 1e-4),
        ("max minus min against ptp",
         lambda x: jt.max(x) - jt.min(x),
         lambda x: jt.max(x - jt.min(x)), 1e-5),

        # A reduction over a non-contiguous view must match the contiguous one.
        ("reduction over a slice", lambda x: jt.sum(x[:, 1:]),
         lambda x: jt.sum(x) - jt.sum(x[:, :1]), 1e-4),
    ]


def evaluate(jt, fn, data, use_cuda):
    with jt.flag_scope(use_cuda=use_cuda):
        return np.asarray(fn(jt.array(data)).numpy(), dtype=np.float64)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--json")
    parser.add_argument("--input", default="ordinary",
                        choices=("ordinary", "special"),
                        help="ordinary random values, or the special values "
                             "every device divergence this session was found in")
    args = parser.parse_args(argv)

    import jittor as jt
    use_cuda = 1 if args.device == "cuda" else 0
    if args.input == "ordinary":
        data = np.random.RandomState(0).randn(6, 8).astype("float32")
    else:
        # The two productive ideas combined: identities that need no reference,
        # evaluated on the values that produced every divergence found so far.
        # An identity that holds for ordinary numbers and breaks here is exactly
        # the defect neither idea finds alone.
        row = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, -1.0, 1e-30],
                       dtype="float32")
        data = np.stack([np.roll(row, k) for k in range(6)]).astype("float32")
    print("input: %s" % args.input)

    rows, bad = [], 0
    for name, left, right, tol in cases(jt):
        try:
            a = evaluate(jt, left, data, use_cuda)
            b = evaluate(jt, right, data, use_cuda)
        except Exception as exc:  # noqa: BLE001 - a failure is a result
            rows.append({"case": name, "status": "ERROR",
                         "detail": "%s: %s" % (type(exc).__name__, str(exc)[:90])})
            bad += 1
            continue
        if a.shape != b.shape:
            rows.append({"case": name, "status": "SHAPE",
                         "detail": "%s against %s" % (a.shape, b.shape)})
            bad += 1
            continue
        # Three comparisons, not one. `a - b` is NaN wherever either side is,
        # so a plain gap reports every NaN position as a disagreement even when
        # both sides agree that the answer is NaN -- which is what this sweep
        # fed in deliberately. The first version did exactly that and called
        # seven identities broken.
        nan_a, nan_b = np.isnan(a), np.isnan(b)
        inf_a, inf_b = np.isinf(a), np.isinf(b)
        detail = None
        if not np.array_equal(nan_a, nan_b):
            status, detail = "DIFFERENT", "NaN appears in different positions"
        elif not np.array_equal(inf_a, inf_b):
            status, detail = "DIFFERENT", "infinities appear in different positions"
        elif inf_a.any() and not np.array_equal(np.signbit(a[inf_a]),
                                                np.signbit(b[inf_a])):
            # Sign compared only where both are infinite. The first attempt
            # wrote `inf_mask * np.sign(x)`, and `np.sign(nan)` is `nan`, so
            # every NaN position poisoned the comparison and five identities
            # were reported broken. Handling special values correctly in the
            # *checker* turned out to be as easy to get wrong as handling them
            # in the code under test.
            status, detail = "DIFFERENT", "infinities differ in sign"
        else:
            finite = ~(nan_a | inf_a)
            if not finite.any():
                status, gap = "OK", 0.0
            else:
                gap = float(np.max(np.abs(a[finite] - b[finite])))
                scale = max(float(np.max(np.abs(a[finite]))), 1e-6)
                status = "OK" if gap <= max(tol * scale, tol) else "DIFFERENT"
        if status != "OK":
            bad += 1
        row = {"case": name, "status": status, "tolerance": tol}
        if detail:
            row["detail"] = detail
        else:
            row["gap"] = gap
        rows.append(row)

    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print("device=%s cases=%d %s" % (args.device, len(rows), counts))
    for row in rows:
        if row["status"] != "OK":
            print("  [%s] %-38s %s" % (
                row["status"], row["case"],
                row.get("detail") or "max gap %.3e against tolerance %.0e"
                % (row["gap"], row["tolerance"])))
    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps({"device": args.device, "counts": counts, "rows": rows},
                       indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

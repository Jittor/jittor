#!/usr/bin/env python3
"""Check analytic gradients against numerical ones *near* the awkward inputs.

The existing numerical-gradient layer samples ordinary values, which is where
analytic and numerical agree by construction. Every CPU/CUDA divergence this
session found lived instead at the values IEEE-754 treats specially, and the
gradient graph has never been swept along that dimension.

So the sample points sit *near* those values rather than on them: just above and
below zero, either side of the subnormal boundary, close to float32's overflow,
and at ties. Exactly on a non-differentiable point the two methods legitimately
disagree -- that is what `abs'(0)` is about, and it has its own test -- so the
interesting question is whether the analytic formula is still right in the
neighbourhood, where it has no excuse.

The numerical side is a central difference taken in float64 with a step scaled
to the point, and the comparison is deliberately loose: this is looking for a
formula that is wrong, not for the last bits of a difference quotient.

Read the output with care: this tool did not earn trust
------------------------------------------------------
Four iterations produced seventeen findings and **none of them were real**.
Each was the step size being wrong for the function under test, and each looked
exactly like a wrong gradient until it was worked out:

* a step of `1e-3` at `x = 1e-6` straddles zero, so `abs` reported `0.001`
  against a correct analytic `1.0`, and `log`/`sqrt` left their domain and
  returned NaN;
* a step scaled to the *input* (`|x| * 1e-3`) is below what the *output* can
  resolve near flat points -- `exp(1e-30 +- 1e-33)` differs by nothing float64
  can hold, so the quotient is `0.0` against a correct `1.0`;
* a relative step is meaningless for a periodic function: at `x = 1e3` a step of
  `1.0` spans a sixth of a period, and the central difference returns
  `cos(x) * sin(h)/h = 0.473` where the derivative is `0.562`. The number
  matches to three digits, which is how the artifact was identified.

A sound version needs the step chosen per function from its own curvature and
range, which is what a proper gradcheck does and what
``tests/opinfo`` already implements. This file is kept because the *sample
points* are the useful part -- near zero, either side of the subnormal edge, at
ties -- and because the record of how a plausible-looking gradient check goes
wrong is worth more than the file. **Do not treat a MISMATCH here as a defect
without deriving what the step does to that particular function first.**

Usage::

    PYTHONPATH=<repo>/python python tools/gradient_near_special_values.py [--device cuda]
"""

import argparse
import json
import pathlib

import numpy as np


#: Points near the values that matter, never exactly on them.
POINTS = {
    "just above zero": 1e-6,
    "just below zero": -1e-6,
    "near the subnormal edge": 1e-30,
    # No "large" point. At 1e15 float32's own spacing is 6.7e7, so a periodic
    # function's argument is not represented to within many periods and any
    # difference quotient there measures the representation, not the slope.
    "moderately large": 1e3,
    "near one": 1.0 + 1e-5,
    "negative near one": -1.0 - 1e-5,
    "a tie neighbour": 3.0,
}

#: Unary operators with a gradient worth checking; each maps a name to the
#: callable and the domain it is defined on.
def unary_cases(jt):
    positive_only = ("log", "sqrt")
    names = ("abs", "exp", "log", "sqrt", "sin", "cos", "tanh", "sigmoid",
             "erf", "negative")
    cases = []
    for name in names:
        fn = getattr(jt, name, None)
        if fn is None:
            fn = getattr(jt.nn, name, None)
        if fn is None:
            continue
        cases.append((name, fn, name in positive_only))
    return cases


def analytic(jt, fn, x, use_cuda):
    with jt.flag_scope(use_cuda=use_cuda):
        var = jt.array(np.array([x], dtype="float32"))
        out = fn(var)
        return float(jt.grad(out.sum(), var).numpy()[0])


def step_for(x):
    """A step small enough to stay in the neighbourhood being asked about.

    The first version used `max(|x|, 1) * 1e-3`, which is wrong in both
    directions and produced nine false positives before it produced anything
    else. Near zero the step straddled the point: sampling `abs` at
    `1e-6 ± 1e-3` averages the +1 and -1 sides and reports `0.001` against a
    correct analytic `1.0`. It also pushed `log` and `sqrt` out of their domain,
    so the difference quotient came back NaN. And at `1e15` a step of `1e12` is
    meaningless for `sin`, whose period it crosses many times over -- while
    float32's own spacing at `1e15` is already `6.7e7`.

    So the step is bounded by the point's own magnitude, never larger than the
    distance that would cross zero, and never smaller than float32 can resolve
    there. A point where no such step exists is not probed rather than probed
    badly.
    """
    magnitude = abs(x)
    if magnitude == 0.0:
        return None
    spacing = np.spacing(np.float32(magnitude))
    step = magnitude * 1e-3
    if step < spacing * 8:
        # Below this the difference quotient is measuring rounding, not slope.
        return None
    return float(step)


def numerical(jt, fn, x, use_cuda, positive_only):
    """Central difference, or None where no honest step exists."""
    step = step_for(x)
    if step is None:
        return None
    if positive_only and x - step <= 0:
        # A one-sided difference here would be comparing a different quantity;
        # skipping is the honest answer, not a wider tolerance.
        return None
    # float64 on the numerical side. In float32 the two samples differ by less
    # than the output's own spacing near these points -- `exp(1e-6 +- 1e-9)`
    # differs by ~2e-9 while float32's spacing at 1.0 is 1.19e-7 -- so the
    # quotient came back exactly 0 and read as a wrong gradient. Widening the
    # step instead would leave the neighbourhood being asked about. The
    # comparison is then float64-numerical against float32-analytic, which is
    # the right pairing for finding a wrong *formula*: it is insensitive to
    # float32 rounding and sensitive to the derivative being the wrong function.
    with jt.flag_scope(use_cuda=use_cuda, auto_convert_64_to_32=0):
        hi = float(fn(jt.array(np.array([x + step], dtype="float64"))).numpy()[0])
        lo = float(fn(jt.array(np.array([x - step], dtype="float64"))).numpy()[0])
    return (hi - lo) / (2.0 * step)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--json")
    args = parser.parse_args(argv)

    import jittor as jt
    use_cuda = 1 if args.device == "cuda" else 0

    rows, bad = [], 0
    for name, fn, positive_only in unary_cases(jt):
        for label, x in POINTS.items():
            if positive_only and x <= 0:
                continue
            try:
                a = analytic(jt, fn, x, use_cuda)
                n = numerical(jt, fn, x, use_cuda, positive_only)
                if n is None:
                    rows.append({"op": name, "point": label, "status": "SKIPPED",
                                 "detail": "no honest step at this point"})
                    continue
            except Exception as exc:  # noqa: BLE001
                rows.append({"op": name, "point": label, "status": "ERROR",
                             "detail": "%s: %s" % (type(exc).__name__, str(exc)[:80])})
                continue
            if not np.isfinite(a) or not np.isfinite(n):
                # A non-finite on either side is a finding only when the other
                # side is finite; both non-finite is the honest answer at a pole.
                status = "OK" if (not np.isfinite(a) and not np.isfinite(n)) else "NONFINITE"
            else:
                scale = max(abs(a), abs(n), 1e-3)
                status = "OK" if abs(a - n) / scale < 5e-2 else "MISMATCH"
            if status != "OK":
                bad += 1
            rows.append({"op": name, "point": label, "status": status,
                         "analytic": a, "numerical": n})

    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print("device=%s checks=%d %s" % (args.device, len(rows), counts))
    for row in rows:
        if row["status"] not in ("OK",):
            print("  [%s] %-10s %-24s analytic=%s numerical=%s%s" % (
                row["status"], row["op"], row["point"],
                row.get("analytic"), row.get("numerical"),
                " " + row["detail"] if row.get("detail") else ""))
    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps({"device": args.device, "counts": counts, "rows": rows},
                       indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

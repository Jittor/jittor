#!/usr/bin/env python3
"""Find dtypes an OpInfo entry never tests but the operator accepts anyway.

`bitwise_not` returned True for every bool input and had an OpInfo entry the
whole time. The entry declared ``dtypes=_INT`` -- ``integral_types()``, which
does not contain bool -- and its sample generator additionally forced integers
(``dt = _int_dtype(dtype)``), so no test could feed it a bool even by accident.
The operator accepted bool perfectly well. It just answered wrong.

That is the shape of the hole this probes: a declared dtype set narrower than
what the operator actually runs on. Every such pair is a capability the suite
exercises nothing of, while the op reads as covered.

Note also that ``OpInfo.dtypes`` defaults to ``floating_types()``. An entry that
simply omits the argument therefore tests floats only, silently.

The probe calls each operator directly rather than through
``sample_inputs_func``, because the sample generator is part of the narrowing:
it coerces the dtype it is handed. Arity is discovered by trying unary then
binary; an entry that answers neither is reported as unprobed rather than
counted as clean -- a denominator that quietly drops the hard cases would
flatter the result.

Usage::

    PYTHONPATH=<repo>/python python tools/opinfo_dtype_gaps.py [--json out.json]
"""

import argparse
import json
import os
import sys
import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_db():
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    from opinfo.database import op_db
    return op_db


def _candidate_dtypes():
    """bool plus the integral and floating widths, as dtype name strings."""
    return (
        "bool",
        "uint8", "int8", "int16", "int32", "int64",
        "float16", "float32", "float64",
    )


def _name(dtype):
    return getattr(dtype, "name", None) or str(dtype)


def _make(jt, dtype_name, size=4):
    import numpy as np
    if dtype_name == "bool":
        return jt.array(np.array([True, False, True, False][:size]))
    if dtype_name.startswith("u"):
        return jt.array(np.arange(1, size + 1, dtype=dtype_name))
    if dtype_name.startswith("int"):
        return jt.array(np.arange(1, size + 1, dtype=dtype_name))
    return jt.array(np.arange(1, size + 1, dtype=dtype_name))


def _runs(op, jt, dtype_name):
    """Does the operator produce a value for this dtype? Returns arity or None."""
    for arity in (1, 2):
        try:
            args = [_make(jt, dtype_name) for _ in range(arity)]
            out = op(*args)
        except Exception:
            continue
        try:
            if hasattr(out, "sync"):
                out.sync()
            elif hasattr(out, "numpy"):
                out.numpy()
        except Exception:
            continue
        return arity
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="write the full result here")
    args = parser.parse_args(argv)

    op_db = _load_db()
    import jittor as jt

    gaps = []
    unprobed = []
    clean = 0
    for info in op_db:
        declared = {_name(d) for d in getattr(info, "dtypes", ()) or ()}
        try:
            operator = info.op
        except Exception:
            unprobed.append({"op": info.name, "why": "no callable"})
            continue
        if operator is None:
            unprobed.append({"op": info.name, "why": "no callable"})
            continue
        probed_any = False
        for dtype_name in _candidate_dtypes():
            if dtype_name in declared:
                probed_any = True
                continue
            arity = _runs(operator, jt, dtype_name)
            if arity is not None:
                probed_any = True
                gaps.append({
                    "op": info.name,
                    "dtype": dtype_name,
                    "arity": arity,
                    "declared": sorted(declared),
                })
        if not probed_any:
            unprobed.append({"op": info.name, "why": "no arity answered"})
        else:
            clean += 1

    result = {
        "ops": len(op_db),
        "gaps": len(gaps),
        "unprobed": len(unprobed),
        "gap_detail": gaps,
        "unprobed_detail": unprobed,
    }
    print("ops=%d  untested-but-working (op,dtype) pairs=%d  unprobed=%d"
          % (result["ops"], result["gaps"], result["unprobed"]))
    by_op = {}
    for gap in gaps:
        by_op.setdefault(gap["op"], []).append(gap["dtype"])
    for op_name in sorted(by_op)[:25]:
        print("  %-28s %s" % (op_name, ",".join(sorted(by_op[op_name]))))
    if len(by_op) > 25:
        print("  ... and %d more ops" % (len(by_op) - 25))
    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print("detail written to", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

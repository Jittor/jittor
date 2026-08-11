# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Coverage / health report for the modern jittor test suite.

Prints what the op registry actually covers -- per op: which domains, dtypes,
whether it has an independent forward reference, and whether forward / backward /
double-backward are exercised -- plus the active device matrix. The audit's "no
silent caps" principle: gaps (no ref, forward-only, gradgrad-unsupported) are
reported explicitly rather than passing quietly.

Run::  PYTHONPATH=tests python -m opinfo.report
"""
import sys

from _helpers import common as cu
from .database import op_db, _loaded_modules


def build_rows():
    rows = []
    for op in sorted(op_db, key=lambda o: o.full_name):
        rows.append({
            "name": op.full_name,
            "dtypes": len(op.dtypes),
            "ref": op.ref is not None,
            "fwd": True,
            "bwd": op.supports_autograd,
            "gradgrad": op.supports_autograd and op.supports_gradgrad,
        })
    return rows


def main():
    devices = cu.get_all_device_types()
    rows = build_rows()
    n = len(rows)
    no_ref = [r["name"] for r in rows if not r["ref"]]
    fwd_only = [r["name"] for r in rows if not r["bwd"]]
    no_gg = [r["name"] for r in rows if r["bwd"] and not r["gradgrad"]]

    print("=" * 70)
    print("  Jittor torch-grade test coverage report")
    print("=" * 70)
    print(f"  devices active : {', '.join(devices)}"
          f"   (cuda={cu.HAS_CUDA}, acl/npu={cu.HAS_ACL})")
    print(f"  op_db modules  : {', '.join(sorted(_loaded_modules))}")
    print(f"  operators      : {n}")
    print(f"  with fwd ref   : {n - len(no_ref)}/{n}")
    print(f"  with backward  : {n - len(fwd_only)}/{n}  (gradcheck)")
    print(f"  with gradgrad  : {n - len(fwd_only) - len(no_gg)}/{n}  (gradgradcheck)")
    print("-" * 70)
    print(f"  {'op':28s} {'dtypes':>6} {'ref':>4} {'bwd':>4} {'gg':>4}")
    print("-" * 70)
    for r in rows:
        print(f"  {r['name']:28s} {r['dtypes']:>6} "
              f"{'Y' if r['ref'] else '-':>4} "
              f"{'Y' if r['bwd'] else '-':>4} "
              f"{'Y' if r['gradgrad'] else '-':>4}")
    print("-" * 70)
    if no_ref:
        print(f"  forward-only (no numpy ref, fwd not value-checked): {', '.join(no_ref)}")
    if fwd_only:
        print(f"  no backward (supports_autograd=False): {', '.join(fwd_only)}")
    if no_gg:
        print(f"  no 2nd-order (supports_gradgrad=False): {', '.join(no_gg)}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

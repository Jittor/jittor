#!/usr/bin/env python3
"""Saving, reloading and recomputing must give back the same numbers.

Two more pairs that need no reference value, in the shape that has been the only
productive one this session:

* **gradient mode.** A forward pass under ``no_grad`` must produce exactly the
  values the ordinary forward produces. ``no_grad`` is a statement about the
  backward pass; if it also moves the forward, then every evaluation loop is
  computing something slightly different from what training computed.
* **a process boundary.** Weights written to disk, read back in a *fresh
  interpreter*, and run on the same input must give bit-identical outputs.
  Everything else here runs in one process, where a defect in serialisation is
  masked by the live objects still being correct. Corruption that only shows up
  after a restart is the quietest kind there is: it appears in a later run, on
  a different machine, with nothing left to point at the save.

The second check runs the reload in a subprocess on purpose. Loading into the
same interpreter shares the allocator, the JIT cache and any global state that
the save might have depended on, so it can pass while a real reload fails.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np


#: The child is told which device to use and made to prove it landed there.
#:
#: The first version set nothing and relied on the default. `use_cuda` defaults
#: to **0** in a fresh process even on a machine with eight GPUs, so the
#: `--device cuda` run was comparing a CUDA parent against a CPU child and
#: calling the difference a serialisation defect. It reported OK for months
#: because under `-Ofast` the two devices happened to agree bit-for-bit on this
#: model; `-O3` moved the CPU result by 1.49e-08 and the luck ran out.
#:
#: The `location()` assertion is the part that matters. Passing the flag is
#: easy to get right and easy to have silently ignored -- a fallback, a missing
#: driver, a flag consumed too late -- and then the check goes back to
#: comparing two devices while looking like it compares two processes.
CHILD = r'''
import json, sys, numpy as np, jittor as jt
from jittor import nn
state_path, input_path, out_path, device = sys.argv[1:5]
jt.flags.use_cuda = 1 if device == "cuda" else 0
jt.set_global_seed(0)
model = nn.Sequential(nn.Linear(8, 16), nn.Relu(), nn.Linear(16, 4))
model.load(state_path)
x = jt.array(np.load(input_path))
x.sync()
where = x.location()
expected = "device" if device == "cuda" else "cpu"
assert where == expected, "child asked for %s, tensor is on %s" % (device, where)
y = model(x)
np.save(out_path, y.numpy())
'''


def build_model(jt):
    from jittor import nn
    jt.set_global_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.Relu(), nn.Linear(16, 4))


def check_no_grad_matches(jt, use_cuda, rows):
    with jt.flag_scope(use_cuda=use_cuda):
        model = build_model(jt)
        x = jt.array(np.random.RandomState(1).randn(4, 8).astype("float32"))
        ordinary = model(x).numpy().copy()
        with jt.flag_scope(no_grad=1):
            guarded = model(x).numpy().copy()
    identical = bool(np.array_equal(ordinary, guarded))
    rows.append({
        "case": "no_grad forward matches ordinary forward",
        "status": "OK" if identical else "DIFFERENT",
        "detail": "" if identical else "max gap %.3e" % float(
            np.max(np.abs(ordinary - guarded))),
    })


def check_reload_in_a_fresh_process(jt, use_cuda, rows, device):
    with tempfile.TemporaryDirectory(prefix="jittor-roundtrip-") as tmp:
        tmp = pathlib.Path(tmp)
        state, inp, out = tmp / "m.pkl", tmp / "x.npy", tmp / "y.npy"
        with jt.flag_scope(use_cuda=use_cuda):
            model = build_model(jt)
            data = np.random.RandomState(2).randn(4, 8).astype("float32")
            np.save(inp, data)
            model.save(str(state))
            here = model(jt.array(data)).numpy().copy()

        script = tmp / "child.py"
        script.write_text(CHILD, encoding="utf-8")
        env = dict(os.environ)
        env["JITTOR_TORCH_SHIM"] = "0"
        result = subprocess.run(
            [sys.executable, str(script), str(state), str(inp), str(out), device],
            capture_output=True, text=True, env=env, timeout=900)
        if result.returncode != 0 or not out.is_file():
            # The last 200 characters of stderr are whatever Jittor printed on
            # the way down, which is rarely the reason. Pull the exception line
            # out instead -- a failure that does not say why is the thing this
            # file exists to avoid producing.
            lines = [l.strip() for l in (result.stderr or "").splitlines() if l.strip()]
            reason = next((l for l in reversed(lines)
                           if "Error" in l or "error" in l), "")
            rows.append({"case": "reload in a fresh process", "status": "ERROR",
                         "detail": reason or (result.stderr or "")[-200:]})
            return
        there = np.load(out)

    if here.shape != there.shape:
        rows.append({"case": "reload in a fresh process", "status": "DIFFERENT",
                     "detail": "shape %s against %s" % (here.shape, there.shape)})
        return
    identical = bool(np.array_equal(here, there))
    rows.append({
        "case": "reload in a fresh process",
        "status": "OK" if identical else "DIFFERENT",
        # Bit-identical is the right bar: the same weights and the same input on
        # the same device should not be allowed to drift at all. A tolerance
        # here would hide exactly the corruption the check is for.
        "detail": "" if identical else "max gap %.3e" % float(
            np.max(np.abs(here - there))),
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--json")
    args = parser.parse_args(argv)

    import jittor as jt
    use_cuda = 1 if args.device == "cuda" else 0

    rows = []
    check_no_grad_matches(jt, use_cuda, rows)
    check_reload_in_a_fresh_process(jt, use_cuda, rows, args.device)

    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print("device=%s cases=%d %s" % (args.device, len(rows), counts))
    for row in rows:
        if row["status"] != "OK":
            print("  [%s] %-40s %s" % (row["status"], row["case"], row["detail"]))
    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps({"device": args.device, "counts": counts, "rows": rows},
                       indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

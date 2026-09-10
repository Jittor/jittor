#!/usr/bin/env python3
"""Find operations that change something they were only asked to read.

The coverage layers answer whether an entry point is *reached*. They cannot
answer what reaching it does to everything else, and that is where the last two
defects a person found by hand actually lived: ``a.numpy()`` relocated ``a``
itself to host memory, and the next device operation paid 215x to bring it back.
``numpy`` was never in a coverage gap -- it is called thousands of times. Nothing
asserted what it did to its receiver.

Contracts written by hand do not close that: they cover what someone thought to
check. This checks a *property* instead -- call an operation, then compare the
inputs against a snapshot of themselves -- so it reports the operations nobody
thought about, which is the only kind that is still hiding.

What a snapshot holds is deliberately not "residency". It is every observable
the Var exposes, content included, so an operation that quietly changes dtype,
shape, gradient state or values is the same finding by the same mechanism. The
probe was written without naming the two known defects; it finds them because
they are instances of the property, not because they are in a list.

Mutation is legitimate for some operations. Those are declared in
``INTENTIONALLY_MUTATING`` with the reason, following the rule the repository
already applies to ``gate_scope.EXCLUDED``: something exempted from a check has
to say why. An operation that mutates without being declared is the finding.

Usage::

    PYTHONPATH=<repo>/python python tools/side_effect_probe.py --device cuda
    # run under an isolated JITTOR_HOME; residency findings need a real device
"""

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "tests" / "structure" / "public_api_manifest.json"

#: Operations whose whole purpose is to change their receiver or an argument.
#: Each needs a reason, so the list cannot quietly absorb a real defect.
INTENTIONALLY_MUTATING = {
    "assign": "replaces the receiver's contents by definition",
    "update": "in-place state update",
    "swap": "exchanges two Vars' storage",
    "share_with": "makes the receiver alias another's allocation",
    "start_grad": "flips the receiver's gradient state",
    "stop_grad": "flips the receiver's gradient state",
    "stop_fuse": "flips a fusion flag on the receiver",
    "detach_inplace": "named for what it does",
    "sync": "materializes the receiver -- residency change is the point",
    "fetch_sync": "materializes the receiver",
    "migrate_to_cpu": "moves the receiver, by name",
    "migrate_to_gpu": "moves the receiver, by name",
    "to_device": "moves the receiver, by name",
    "cuda": "requests a device placement",
    "cpu": "requests a host copy",
    "npu": "requests a device placement",
    "to": "requests a placement or dtype",
}

#: Entries that compile source they are handed. Calling them with a guessed
#: argument builds an invalid kernel, and the failure arrives from the compiler
#: rather than from the call, which would end the probe instead of the case.
#: Skipped because there is nothing to learn here, not because they are exempt.
TAKES_SOURCE = ("code", "numpy_code", "reindex", "reindex_reduce", "reindex_var")


def _is_inplace_name(name):
    # Torch's trailing-underscore convention. `__init__`-style dunders are not
    # operations under test.
    return name.endswith("_") and not name.startswith("__")


def snapshot(var, jt):
    """Every observable of a Var, including its values.

    Content is hashed rather than kept so a large tensor costs a fixed amount,
    and it is read through a copy so taking the snapshot cannot itself be the
    thing that moves the tensor -- the failure mode this probe exists to find.
    """
    state = {}
    for attribute in ("dtype", "shape", "device_id", "requires_grad"):
        try:
            value = getattr(var, attribute, None)
            state[attribute] = str(value() if callable(value) else value)
        except Exception:
            state[attribute] = "<unreadable>"
    for method in ("location", "is_stop_grad", "is_stop_fuse"):
        try:
            fn = getattr(var, method, None)
            state[method] = str(fn()) if callable(fn) else "<absent>"
        except Exception:
            state[method] = "<unreadable>"
    # Content is fingerprinted by *reductions*, not by reading the buffer.
    # `var.data`/`numpy()` is the very path that relocates a device tensor, so a
    # snapshot taken that way would move the tensor before the operation under
    # test ran, and both snapshots would agree -- the probe would destroy the
    # evidence it exists to collect. A reduction produces a new scalar Var and
    # leaves its source where it is, so the fingerprint is free of side effects
    # of its own. Two moments catch a permutation that preserves the sum.
    try:
        state["sum"] = "%.6g" % float(var.sum().item())
        state["sq"] = "%.6g" % float((var * var).sum().item())
    except Exception:
        state["sum"] = state["sq"] = "<unreadable>"
    return state


def _make(jt, shape=(8, 8)):
    var = jt.array(np.arange(int(np.prod(shape)), dtype="float32").reshape(shape) + 1.0)
    var.sync()
    return var


def _entries():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for owner_key, names in data.items():
        for name in names:
            yield owner_key, name


def probe(jt, device, limit=None, checkpoint=None):
    """Probe each entry, writing results as they are produced.

    Jittor reports a failed kernel build with ``LOGf``, which terminates the
    process: it cannot be caught, so one bad entry would otherwise discard
    everything probed before it and leave no record of which entry it was.
    Each result is therefore flushed as it is made, and the entry about to be
    called is written first.
    """
    results = []
    seen = 0

    def flush(current=None):
        if checkpoint is None:
            return
        pathlib.Path(checkpoint).write_text(json.dumps(
            {"attempting": current, "results": results}, indent=2), encoding="utf-8")
    for owner_key, name in _entries():
        if name.startswith("__"):
            continue
        if name in TAKES_SOURCE:
            continue
        if name in INTENTIONALLY_MUTATING or _is_inplace_name(name):
            continue
        if limit is not None and seen >= limit:
            break

        # Only Var-receiver entries can be probed this way: a module-level
        # function's first argument is not reliably a Var, and guessing wrong
        # would produce errors rather than findings.
        if owner_key != "Var":
            continue
        seen += 1

        flush(current=name)
        try:
            receiver = _make(jt)
        except Exception as exc:
            results.append({"op": name, "status": "SETUP", "detail": str(exc)[:80]})
            continue

        before = snapshot(receiver, jt)
        try:
            member = getattr(receiver, name)
        except Exception:
            continue
        if not callable(member):
            continue

        called = False
        for args in ((), (_make(jt),), (0,)):
            try:
                out = member(*args)
                # Force the lazy graph now, inside the guard: a bad kernel
                # otherwise surfaces during the next snapshot and takes the
                # whole probe with it instead of failing this one case.
                if hasattr(out, "sync"):
                    out.sync()
                called = True
                break
            except Exception:
                continue
        if not called:
            results.append({"op": name, "status": "UNCALLED",
                            "detail": "no argument shape accepted"})
            continue

        try:
            after = snapshot(receiver, jt)
        except Exception as exc:
            results.append({"op": name, "status": "UNREADABLE",
                            "detail": str(exc).splitlines()[0][:80]})
            continue
        changed = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
        if changed:
            results.append({"op": name, "status": "MUTATES", "changed": changed})
        else:
            results.append({"op": name, "status": "OK"})
    flush(current=None)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--json", help="write the full result here")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    import jittor as jt
    with jt.flag_scope(use_cuda=1 if args.device == "cuda" else 0):
        results = probe(jt, args.device, args.limit,
                        checkpoint=(args.json + ".partial") if args.json else None)

    counts = {}
    for row in results:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print("device=%s probed=%d %s" % (args.device, len(results), counts))
    for row in results:
        if row["status"] == "MUTATES":
            fields = ", ".join(
                "%s %s->%s" % (k, v[0], v[1]) for k, v in sorted(row["changed"].items()))
            print("  [MUTATES]  %-22s %s" % (row["op"], fields))
    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps({"device": args.device, "counts": counts, "results": results},
                       indent=2, sort_keys=True), encoding="utf-8")
        print("detail written to", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())

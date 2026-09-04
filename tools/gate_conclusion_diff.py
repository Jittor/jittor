#!/usr/bin/env python3
"""Prove that a gate change did not lose a conclusion -- before believing it is faster.

A gate optimisation is judged on one criterion, and wall clock is not it:

    the two runs must reach the **same conclusion for the same nodeids**, item by
    item. Same *count* is not the same set, and "faster" is worthless if the
    faster run stopped answering for something.

That is not a hypothetical. ``0.16`` measured ``-n 4`` on the device parity
battery: 6% faster and **three of twenty-six conclusions gone**, with a green
exit status and a plausible summary line. The rule it left behind is the reason
this script exists -- *a validator that sometimes gives no answer is worse than
a slow one.*

Usage::

    # 1. record the baseline, then the candidate
    python tools/gate_conclusion_diff.py record --out /run/base.json -- \
        tests/backends/parity/test_device_parity.py -q
    python tools/gate_conclusion_diff.py record --out /run/cand.json \
        --env JITTOR_PARITY_REF_CACHE=1 -- \
        tests/backends/parity/test_device_parity.py -q

    # 2. the criterion: exits non-zero on any per-nodeid difference
    python tools/gate_conclusion_diff.py compare /run/base.json /run/cand.json

``record`` never fails on a red run: a suite whose conclusion is "these six
failed" is a legitimate baseline, and demanding green first would make the
criterion unusable exactly when it matters. Only ``compare`` decides.

Both records must select the same tests; the *configuration* is what differs
(``--env``, extra pytest arguments). Runtime state stays outside the checkout:
each run gets its own ``JITTOR_HOME``/``TMPDIR`` unless the caller pins them.
"""

from __future__ import print_function

import argparse
import json
import os
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(REPO_ROOT / "tests"))
from _helpers.child_process import PYTHON, run_python_child  # noqa: E402

PLUGIN_MODULE = "gate_conclusion_plugin"
OUT_VARIABLE = "GATE_CONCLUSION_OUT"


def _parse_env(pairs):
    overrides = {}
    for pair in pairs or ():
        if "=" not in pair:
            raise SystemExit("--env expects NAME=VALUE, got %r" % pair)
        name, value = pair.split("=", 1)
        overrides[name] = value
    return overrides


def _child_environment(overrides, record_path):
    environment = dict(os.environ)
    environment.update(overrides)
    environment[OUT_VARIABLE] = str(record_path)
    # The plugin lives in tools/, which is not importable by default. `-p` takes
    # a module name, so the directory has to be on the child's path; the
    # checkout's own `python/` is prepended by child_env afterwards and stays
    # first.
    existing = environment.get("PYTHONPATH", "")
    parts = [str(TOOLS_DIR)] + [p for p in existing.split(os.pathsep) if p]
    environment["PYTHONPATH"] = os.pathsep.join(parts)
    return environment


def record(options, pytest_arguments):
    if not pytest_arguments:
        raise SystemExit("record needs pytest arguments after `--`")
    out = Path(options.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_name(out.name + ".session")
    if raw.exists():
        raw.unlink()
    overrides = _parse_env(options.env)
    environment = _child_environment(overrides, raw)
    command = ["-m", "pytest", "-p", PLUGIN_MODULE, "-p", "no:cacheprovider"]
    command += list(pytest_arguments)
    print("=== recording %s ===" % (options.label or out.name), flush=True)
    print(" ".join([PYTHON] + command), flush=True)
    started = time.time()
    # timeout=0: this runs whole gate batteries, and a cap would turn a long
    # baseline into a failure instead of reporting one.
    completed = run_python_child(
        command, cwd=REPO_ROOT, env=environment, inherit=False,
        merge_stderr=True, timeout=0)
    wall = time.time() - started
    print(completed.stdout, flush=True)
    if not raw.exists():
        raise SystemExit(
            "the session wrote no record (%s). The plugin only writes at "
            "sessionfinish, so the process was killed, or collection failed "
            "before `-p %s` loaded. Exit status was %d."
            % (raw, PLUGIN_MODULE, completed.returncode))
    session = json.loads(raw.read_text(encoding="utf-8"))
    raw.unlink()
    payload = {
        "label": options.label or out.name,
        "pytest_arguments": list(pytest_arguments),
        "env_overrides": overrides,
        "exit_code": completed.returncode,
        "wall_seconds": round(wall, 2),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    payload.update(session)
    _write_atomic(out, payload)
    print("collected=%d concluded=%d wall=%.1fs exit=%d -> %s"
          % (len(payload["collected"]), len(payload["conclusions"]), wall,
             completed.returncode, out), flush=True)
    return 0


def _load(path):
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    for key in ("collected", "conclusions"):
        if key not in data:
            raise SystemExit("%s is not a gate conclusion record (no %r)"
                             % (path, key))
    return data


def _describe(record_data):
    return "%s (%s)" % (record_data.get("label", "?"),
                        record_data.get("recorded_at", "?"))


def compare(options):
    baseline = _load(options.baseline)
    candidate = _load(options.candidate)
    differences = []
    notes = []

    base_collected = set(baseline["collected"])
    cand_collected = set(candidate["collected"])
    for nodeid in sorted(base_collected - cand_collected):
        differences.append("NOT COLLECTED any more: %s" % nodeid)
    for nodeid in sorted(cand_collected - base_collected):
        differences.append("NEWLY COLLECTED: %s" % nodeid)

    # Collected but never concluded: the shape 0.16's lost conclusions had.
    unconcluded = set()
    for name, data in (("baseline", baseline), ("candidate", candidate)):
        missing = set(data["collected"]) - set(data["conclusions"])
        for nodeid in sorted(missing):
            unconcluded.add(nodeid)
            differences.append(
                "COLLECTED BUT NO CONCLUSION in %s: %s" % (name, nodeid))

    base_conclusions = baseline["conclusions"]
    cand_conclusions = candidate["conclusions"]
    for nodeid in sorted(set(base_conclusions) & set(cand_conclusions)):
        before = base_conclusions[nodeid]
        after = cand_conclusions[nodeid]
        if before["status"] != after["status"]:
            differences.append("STATUS %s -> %s: %s"
                               % (before["status"], after["status"], nodeid))
        elif (before["status"] == "skipped"
              and _normalize(before.get("reason")) != _normalize(after.get("reason"))):
            differences.append(
                "SKIP REASON changed: %s\n    before: %s\n    after:  %s"
                % (nodeid, before.get("reason"), after.get("reason")))
    for nodeid in sorted(set(base_conclusions) - set(cand_conclusions)):
        if nodeid not in cand_collected or nodeid in unconcluded:
            continue  # already reported, as not collected or as unconcluded
        differences.append("CONCLUSION LOST: %s (was %s)"
                           % (nodeid, base_conclusions[nodeid]["status"]))

    print("baseline  %s" % _describe(baseline))
    print("candidate %s" % _describe(candidate))
    print("selection %r vs %r" % (baseline.get("pytest_arguments"),
                                  candidate.get("pytest_arguments")))
    print("env       %r vs %r" % (baseline.get("env_overrides"),
                                  candidate.get("env_overrides")))
    print("-" * 72)
    print("collected   %d -> %d" % (len(base_collected), len(cand_collected)))
    print("concluded   %d -> %d" % (len(base_conclusions), len(cand_conclusions)))
    for status in sorted(set(_statuses(base_conclusions)) | set(_statuses(cand_conclusions))):
        print("  %-13s %d -> %d" % (status,
                                    _statuses(base_conclusions).get(status, 0),
                                    _statuses(cand_conclusions).get(status, 0)))
    base_wall = baseline.get("wall_seconds")
    cand_wall = candidate.get("wall_seconds")
    if base_wall and cand_wall:
        print("wall        %.1fs -> %.1fs (%.2fx)"
              % (base_wall, cand_wall, base_wall / cand_wall))
    print("-" * 72)
    if differences:
        print("DIFFERENT CONCLUSIONS (%d):" % len(differences))
        for line in differences:
            print("  " + line)
        print("\nThe candidate is not equivalent. Speed is not the criterion: "
              "a run that answers for fewer nodeids is a worse gate, however "
              "fast it is.")
        return 1
    print("IDENTICAL: every collected nodeid concluded the same way in both runs.")
    for line in notes:
        print("note: " + line)
    return 0


def _statuses(conclusions):
    counts = {}
    for data in conclusions.values():
        counts[data["status"]] = counts.get(data["status"], 0) + 1
    return counts


def _normalize(reason):
    """Compare skip reasons without the parts that move between machines."""
    if not reason:
        return ""
    return " ".join(str(reason).split())


def _write_atomic(path, payload):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".partial")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    recorder = subparsers.add_parser(
        "record", help="run a pytest selection and write its per-nodeid record")
    recorder.add_argument("--out", required=True, help="record file to write")
    recorder.add_argument("--label", default=None)
    recorder.add_argument("--env", action="append", default=[],
                          metavar="NAME=VALUE",
                          help="environment override for this run (repeatable)")
    recorder.add_argument("pytest_arguments", nargs=argparse.REMAINDER)

    comparer = subparsers.add_parser(
        "compare", help="exit non-zero unless both records concluded identically")
    comparer.add_argument("baseline")
    comparer.add_argument("candidate")

    options = parser.parse_args(argv)
    if options.command == "record":
        arguments = list(options.pytest_arguments)
        if arguments and arguments[0] == "--":
            arguments = arguments[1:]
        return record(options, arguments)
    return compare(options)


if __name__ == "__main__":
    sys.exit(main())

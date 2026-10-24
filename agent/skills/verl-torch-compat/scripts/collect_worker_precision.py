#!/usr/bin/env python3
"""Collect explicit startup precision readbacks, never execution/L5 proof.

Provide driver, raw raylet and worker logs explicitly; startup messages can
precede Ray's worker stdout redirection. Required PIDs must come from the
actual run, not a count or a default. No framework is imported or executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


SCHEMA = "verl-worker-startup-precision-v1"
MARKER = "VERL_CUDA_STRICT_FP32_READY"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
LINE = re.compile(r"^(?:\((?P<prefix>[^)]*)\)\s*)?" + MARKER + r"(?:\s+(?P<payload>.*))?$")
EXPECTED = {
    "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
    "cudnn_benchmark": False, "float32_matmul_precision": "highest",
}


def positive_pid(value):
    if not value.isascii() or not value.isdecimal() or int(value) < 1:
        raise argparse.ArgumentTypeError("PID must be a positive decimal integer")
    return int(value)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def validate_marker(value, framework):
    if not isinstance(value, dict) or set(value) != {"framework", "pid", "runtime_conditions", "scope"}:
        raise ValueError("precision marker schema must contain exactly framework, pid, runtime_conditions, scope")
    if value["framework"] != framework:
        raise ValueError("precision marker framework differs from requested framework")
    if type(value["pid"]) is not int or value["pid"] < 1:
        raise ValueError("precision marker PID must be a positive JSON integer")
    if value["scope"] != "public_precision_readback_only":
        raise ValueError("precision marker scope must be public_precision_readback_only")
    conditions = value["runtime_conditions"]
    if not isinstance(conditions, dict) or set(conditions) != set(EXPECTED):
        raise ValueError("precision marker runtime_conditions schema changed")
    if any(conditions[key] is not False for key in EXPECTED if key != "float32_matmul_precision"):
        raise ValueError("precision marker TF32/benchmark readbacks must be JSON boolean false")
    if conditions["float32_matmul_precision"] != "highest":
        raise ValueError("precision marker float32_matmul_precision must be highest")


def collect(logs, framework, required_pids):
    if framework not in ("torch", "jittor"):
        raise ValueError("framework must be torch or jittor")
    if any(type(pid) is not int or pid < 1 for pid in required_pids):
        raise ValueError("required PID must be a positive integer")
    log_manifest, workers = [], {}
    for index, path in enumerate(logs):
        path = Path(path).resolve()
        raw = path.read_bytes()
        log_manifest.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            match = LINE.fullmatch(ANSI.sub("", line).strip())
            if match is None:
                continue
            try:
                value = json.loads(match.group("payload") or "", object_pairs_hook=_unique_object,
                                   parse_constant=_reject_constant)
                validate_marker(value, framework)
                prefix_pid = re.search(r"\bpid=(\d+)\b", match.group("prefix") or "")
                if prefix_pid and int(prefix_pid.group(1)) != value["pid"]:
                    raise ValueError("Ray prefix PID differs from precision marker PID")
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid precision marker: {error}") from error
            pid = value["pid"]
            if pid in workers and workers[pid]["record"] != value:
                raise ValueError(f"{path}:{line_number}: conflicting precision readbacks for PID {pid}")
            worker = workers.setdefault(pid, {"record": value, "observations": []})
            worker["observations"].append({"log_index": index, "line": line_number})
    if not workers:
        raise ValueError("no precision markers in explicit input logs")
    missing = sorted(set(required_pids) - set(workers))
    if missing:
        raise ValueError("missing required PIDs: " + ", ".join(map(str, missing)))
    return {
        "schema": SCHEMA, "framework": framework, "status": "startup_precision_observed",
        "scope": "public_precision_readback_only", "logs": log_manifest,
        "required_pids": sorted(set(required_pids)), "observed_pids": sorted(workers),
        "required_pid_coverage": "complete" if required_pids else "not_requested",
        "workers": [dict(workers[pid]["record"], observations=workers[pid]["observations"])
                    for pid in sorted(workers)],
        "unproven": ["computation_thread_precision", "device_execution", "backend_fallback", "performance", "L5"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="append", type=Path, required=True,
                        help="Explicit log path; repeat for driver/raylet/worker logs")
    parser.add_argument("--framework", choices=("torch", "jittor"), required=True)
    parser.add_argument("--require-pid", action="append", type=positive_pid, default=[],
                        help="Actual run PID whose startup marker is required; repeat as needed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        # Never overwrite source logs or an older successful evidence artifact.
        if args.output.exists():
            raise ValueError("output already exists; provide a fresh evidence path")
        report = collect(args.log, args.framework, args.require_pid)
        with args.output.open("x", encoding="utf-8") as output:
            output.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (OSError, ValueError, UnicodeError) as error:
        print("precision collection failed: " + str(error), file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "observed_pids": report["observed_pids"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

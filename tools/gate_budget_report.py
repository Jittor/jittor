#!/usr/bin/env python3
"""Print the measured CPU smoke budget and its current bottleneck."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from _helpers.tiers import budget_report  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers", type=int, default=None,
        help="xdist workers actually started; omitted means runtime cgroup cap")
    parser.add_argument(
        "--configured-workers", type=int, default=None,
        help="workers requested by the gate before runtime cgroup capping")
    parser.add_argument("--json", action="store_true")
    options = parser.parse_args(argv)
    configured_workers = options.configured_workers
    if configured_workers is None:
        configured_text = os.environ.get("JITTOR_GATE_WORKERS")
        if configured_text is not None:
            try:
                configured_workers = int(configured_text)
            except ValueError:
                parser.error("JITTOR_GATE_WORKERS must be a positive integer")
    try:
        report = budget_report(options.workers, configured_workers)
    except ValueError as exc:
        parser.error(str(exc))
    if options.json:
        print(json.dumps(report, sort_keys=True, indent=2))
    else:
        print("smoke budget: %.0fs / %.0fs (headroom %.0fs, %d actual/%d "
              "configured workers; %d CPU quota; %d threads/worker)" % (
            report["predicted_seconds"], report["budget_seconds"],
            report["headroom_seconds"], report["workers"],
            report["configured_workers"], report["effective_cpus"],
            report["threads_per_worker"]))
        for name, item in report["sessions"].items():
            print("%s: %.0fs (%s; work %.0fs, floor %.0fs, startup %.0fs)" % (
                name, item["predicted_seconds"], item["bottleneck"],
                item["work_bound_seconds"], item["longest_file_seconds"],
                item["startup_seconds"]))
    return 0 if report["headroom_seconds"] >= 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

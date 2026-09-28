#!/usr/bin/env python3
"""One-process, one-frontend float32 ACL matmul latency benchmark."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from _paths import LAB_ROOT, REPO_ROOT


def command(args):
    result = subprocess.run(args, cwd=str(REPO_ROOT), text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {"returncode": result.returncode, "stdout": result.stdout,
            "stderr": result.stderr}


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "compat"), required=True)
    parser.add_argument("--m", type=positive, default=256)
    parser.add_argument("--k", type=positive, default=256)
    parser.add_argument("--n", type=positive, default=256)
    parser.add_argument("--slots", type=positive, default=3)
    parser.add_argument("--warmup", type=positive, default=10)
    parser.add_argument("--repeats", type=positive, default=50)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if REPO_ROOT == output or REPO_ROOT in output.parents:
        parser.error("output must be outside the checkout")
    if LAB_ROOT not in output.parents:
        parser.error("output must be under JITTOR_LAB_ROOT")
    if not os.environ.get("JITTOR_HOME"):
        parser.error("set a dedicated JITTOR_HOME for this mode and run")
    state = Path(os.environ["JITTOR_HOME"]).resolve()
    if LAB_ROOT not in state.parents or REPO_ROOT in state.parents:
        parser.error("JITTOR_HOME must be outside the checkout under JITTOR_LAB_ROOT")
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        parser.error("set ASCEND_RT_VISIBLE_DEVICES to an allocated device")
    # Explicit activation below is the sole owner of this process's mode.
    for name in ("JITTOR_TORCH_SHIM", "JITTOR_TORCH_INDEPENDENT"):
        os.environ.pop(name, None)
    os.environ["backend_fallback"] = "error"
    if os.environ.get("sync_run", "0") != "0" or os.environ.get("JT_SYNC", "0") != "0":
        parser.error("disable per-operator diagnostic synchronization for timing")

    import numpy as np
    import jittor as jt
    from jittor._runtime.fallback import forbid_backend_fallbacks

    if not getattr(jt.compiler, "has_acl", False):
        raise RuntimeError("real ACL backend is required")
    jt.flags.use_acl = 1
    jt.flags.use_cuda = 1
    jt.flags.sync_run = 0
    if args.mode == "compat":
        from jittor.compat.shim import activate
        activate(
            independent_namespace=True,
            runtime_root=str(state.parent / "torch-runtime"),
            auto_scan_extensions=False, build_extensions=False,
            local_home=False, configure_cuda=False,
        )
        import torch
        if torch is jt or torch.Tensor is jt.Var:
            raise RuntimeError("independent Torch frontend was not activated")
        frontend = torch
        make = lambda value: torch.tensor(value, dtype=torch.float32, device="npu")
    else:
        if "torch" in sys.modules:
            raise RuntimeError("native process unexpectedly imported torch")
        frontend = jt
        make = jt.array

    rng = np.random.RandomState(args.seed)
    host = [(rng.uniform(-0.5, 0.5, (args.m, args.k)).astype("float32"),
             rng.uniform(-0.5, 0.5, (args.k, args.n)).astype("float32"))
            for _ in range(args.slots)]
    # Float64 host accumulation supplies an independent reference, outside timing.
    references = [a.astype("float64") @ b.astype("float64") for a, b in host]
    rtol, atol = 2e-4, 2e-4
    before = jt.core.backend_fallback_count()
    errors = []
    samples = []
    with forbid_backend_fallbacks(), jt.no_grad():
        inputs = [(make(a), make(b)) for a, b in host]

        def step(index):
            a, b = inputs[index % args.slots]
            return frontend.matmul(a, b)

        def validate(index):
            result = step(index)
            jt.sync_all(True)
            a, b = inputs[index]
            if any(value.location() != "device" for value in (a, b, result)):
                raise RuntimeError("matmul inputs/output must reside on the NPU")
            actual = result.numpy()
            np.testing.assert_allclose(actual, references[index], rtol=rtol, atol=atol)
            errors.append(float(np.max(np.abs(actual - references[index]))))

        # Materialize every input slot and validate it before any measurements.
        for index in range(args.slots):
            validate(index)
        for index in range(args.warmup):
            result = step(index)
            jt.sync_all(True)
        for index in range(args.repeats):
            jt.sync_all(True)
            started = time.perf_counter()
            result = step(index)
            jt.sync_all(True)
            samples.append((time.perf_counter() - started) * 1000.0)
            if result.location() != "device":
                raise RuntimeError("timed output did not execute on device")
        for index in range(args.slots):
            validate(index)
    after = jt.core.backend_fallback_count()
    if after != before:
        raise RuntimeError("backend fallback attempts changed during benchmark")

    revision = command(["git", "rev-parse", "HEAD"])
    status = command(["git", "status", "--porcelain"])
    diff = command(["git", "diff", "HEAD", "--binary"])
    report = {
        "schema_version": 1, "status": "passed", "mode": args.mode,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workload": {key: value for key, value in vars(args).items() if key != "output"},
        "dtype": "float32", "phase": "forward_no_grad", "sync_mode": "per_call",
        "timing_scope": "Python matmul construction through complete device synchronization; no D2H",
        "cold_compile_included": False, "samples_ms": samples,
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
        "correctness": {"reference": "NumPy float64 matmul", "rtol": rtol, "atol": atol,
                        "max_abs_error": max(errors), "all_input_slots_checked_before_and_after": True},
        "device_evidence": {"has_acl": bool(jt.compiler.has_acl),
                            "sync_run": int(jt.flags.sync_run),
                            "use_acl": int(jt.flags.use_acl), "use_cuda": int(jt.flags.use_cuda),
                            "location": "device", "fallback_before": before, "fallback_after": after},
        "environment": {"python": sys.version, "executable": sys.executable,
                        "platform": platform.platform(), "numpy": np.__version__,
                        "jittor_file": jt.__file__,
                        # The independent frontend is a virtual module and
                        # intentionally need not expose a filesystem __file__.
                        "frontend_file": getattr(frontend, "__file__", None),
                        "frontend_distinct_from_native": frontend is not jt,
                        "variables": {key: os.environ.get(key) for key in (
                            "JITTOR_HOME", "cache_name", "ASCEND_RT_VISIBLE_DEVICES",
                            "ASCEND_HOME", "ASCEND_HOME_PATH", "CANN_SET_ENV", "backend_fallback",
                            "sync_run", "JT_SYNC", "OMP_NUM_THREADS")},
                        "npu_smi": command(["npu-smi", "info"])},
        "revision": revision, "git_status": status,
        "tracked_diff_sha256": hashlib.sha256(diff["stdout"].encode()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "publication_note": "dirty checkout results are provisional; preserve untracked source separately",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mode": args.mode, "median_ms": report["median_ms"],
                      "p95_ms": report["p95_ms"], "output": str(output)}))


if __name__ == "__main__":
    main()

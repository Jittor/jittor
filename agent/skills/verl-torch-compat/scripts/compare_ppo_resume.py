"""Validate fresh-process PPO checkpoint continuation within one runtime.

The baseline must execute the whole trajectory and save a model+critic state
after ``checkpoint_after`` updates.  The resume artifact must be produced by a
different process, load that runtime-local checkpoint, and execute exactly the
remaining updates.  Cross-runtime numerical parity remains the responsibility
of ``compare_training_workflow.py``; this comparator proves continuation is not
a same-process no-op.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load_manifest(path):
    return json.loads(path.with_suffix(".json").read_text())


def compare(baseline_path, resume_path, atol, rtol):
    baseline = load_manifest(baseline_path)
    resume = load_manifest(resume_path)
    require(baseline.get("phase") == "baseline", "baseline phase mismatch")
    require(resume.get("phase") == "resume", "resume phase mismatch")
    require(baseline.get("algorithm") == resume.get("algorithm") == "ppo",
            "only PPO continuation is accepted")
    require(baseline.get("critic") == resume.get("critic") == "learned",
            "PPO continuation requires the learned critic")
    for key in ("framework", "device", "steps", "checkpoint_after", "seed", "lr",
                "initial_weight_sha256", "checkpoint_sha256", "parameter_names",
                "critic_parameter_names", "versions", "model_config"):
        require(baseline.get(key) == resume.get(key), "resume contract differs: " + key)
    require(baseline["framework"] in ("torch", "jittor"), "invalid framework")
    if baseline["framework"] == "jittor":
        require(baseline.get("fallback_count") == 0 and resume.get("fallback_count") == 0,
                "candidate fallback count must be zero")
    checkpoint_after = baseline["checkpoint_after"]
    steps = baseline["steps"]
    require(baseline.get("executed_steps") == list(range(steps)),
            "baseline did not execute the whole trajectory")
    require(resume.get("start_step") == checkpoint_after, "resume start step mismatch")
    require(resume.get("executed_steps") == list(range(checkpoint_after, steps)),
            "resume did not execute exactly the continuation")
    require(baseline.get("process_id") != resume.get("process_id"),
            "resume must run in a fresh process")

    compared = 0
    maximum = 0.0
    with np.load(baseline_path, allow_pickle=False) as expected, np.load(
            resume_path, allow_pickle=False) as actual:
        expected_names = {
            name for name in expected.files
            if any(name.startswith("step_%03d/" % step)
                   for step in range(checkpoint_after, steps))
        }
        require(set(actual.files) == expected_names,
                "resume artifact is missing or adds continuation tensors")
        for name in sorted(expected_names):
            left, right = expected[name], actual[name]
            require(left.shape == right.shape and left.dtype == right.dtype,
                    "resume tensor metadata differs: " + name)
            require(np.isfinite(left).all() and np.isfinite(right).all(),
                    "resume tensor is nonfinite: " + name)
            error = float(np.max(np.abs(left.astype(np.float64) - right)))
            maximum = max(maximum, error)
            if left.dtype.kind in "biu":
                require(np.array_equal(left, right), "resume integer tensor differs: " + name)
            else:
                require(np.allclose(left, right, atol=atol, rtol=rtol),
                        "resume float tensor differs: " + name)
            compared += 1
    require(compared > 0, "no continuation tensors compared")
    return {
        "status": "pass",
        "framework": baseline["framework"],
        "device": baseline["device"],
        "checkpoint_after": checkpoint_after,
        "resumed_steps": list(range(checkpoint_after, steps)),
        "tensor_count": compared,
        "max_abs_error": maximum,
        "atol": atol,
        "rtol": rtol,
        "scope": "fresh-process runtime-local PPO actor+critic checkpoint continuation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.atol < 0 or args.rtol < 0:
        parser.error("tolerances must be nonnegative")
    result = compare(args.baseline, args.resume, args.atol, args.rtol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Required Accelerate-only gate, split into unit and real NCCL stages.

Use an environment containing Accelerate, pytest, and nox. The NCCL stage uses
the maintained repository NCCL session; expose two devices and configure the
host's NCCL workaround externally when necessary. CPU unit coverage alone is
not acceptance of distributed execution.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
UNIT_TESTS = (
    "compat/tests/torch/test_accelerate_single_process.py",
    "compat/tests/torch/test_accelerate_training_control.py",
    "compat/tests/torch/test_accelerate_scaler_protocol.py",
    "compat/tests/torch/test_accelerate_amp.py",
    "compat/tests/torch/test_torch_rng_state.py",
    "compat/tests/torch/test_torch_sampler_rng.py",
    "compat/tests/torch/test_checkpoint_state_dict.py",
)
DISTRIBUTED_MODULE = "compat/tests/torch/test_accelerate_distributed_integration.py"
DISTRIBUTED_TESTS = tuple(DISTRIBUTED_MODULE + "::" + name for name in (
    "test_accumulate_real_ddp_tail_and_gather_for_metrics",
    "test_real_ddp_no_sync_then_closing_backward",
    "test_fsdp2_full_model_optimizer_checkpoint_resume",
))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("unit", "nccl", "all"), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    extra = args.pytest_args
    if extra and extra[0] == "--":
        extra = extra[1:]
    env = dict(os.environ)
    env["JITTOR_TORCH_SHIM"] = "1"
    env["JITTOR_REQUIRE_ACCELERATE"] = "1"
    env["JITTOR_TEST_DEVICES"] = args.device
    env["JITTOR_CUDA"] = "1" if args.device == "cuda" else "0"
    env.pop("use_cuda", None)
    if args.device == "cuda":
        env["JITTOR_TEST_REQUIRE_CUDA"] = "1"
    else:
        env.pop("JITTOR_TEST_REQUIRE_CUDA", None)
    if args.stage in ("unit", "all"):
        # Import failure is a gate failure, not a collection-time optional skip.
        subprocess.run([sys.executable, "-c", "import accelerate, pytest"],
                       check=True, env=env, cwd=str(ROOT))
        subprocess.run([sys.executable, "-m", "pytest", "-q", *UNIT_TESTS, *extra],
                       check=True, env=env, cwd=str(ROOT))
    if args.stage in ("nccl", "all"):
        devices = [item for item in env.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
        if len(devices) != 2:
            parser.error("NCCL stage requires exactly two explicit CUDA_VISIBLE_DEVICES")
        env["JITTOR_NCCL_WORLD_SIZE"] = "2"
        env["JITTOR_TEST_DEVICES"] = "cuda"
        env["JITTOR_CUDA"] = "1"
        env["JITTOR_TEST_REQUIRE_CUDA"] = "1"
        env["JITTOR_ACCELERATE_DISTRIBUTED_REQUIRED"] = "1"
        # Each scenario gets a fresh rank world. Rebuilding a process group
        # across distinct backends within one pytest interpreter is not gated.
        for target in DISTRIBUTED_TESTS:
            subprocess.run([sys.executable, "-m", "nox", "-s", "nccl", "--",
                            target, *extra], check=True, env=env, cwd=str(ROOT))


if __name__ == "__main__":
    main()

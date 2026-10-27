"""Real two-rank BF16 Engine training; independent of the world-one FP32 gate."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from _helpers.child_process import PYTHON
from _ecosystem_harness import REAL_TORCH_PYTHON, _enabled, _runner_package_site
from _ecosystem_deepspeed_multirank import reserve_case, run_pair
from _ecosystem_deepspeed_comparison import _compare_engine_outputs


@pytest.mark.parametrize("zero_stage", (1, 2))
@pytest.mark.parametrize("perturbation", (0., .125))
def test_two_rank_bfloat16_engine_training_and_resume(zero_stage, perturbation):
    required = _enabled("JITTOR_REQUIRE_DEEPSPEED_MULTIRANK")
    if not required:
        pytest.skip("manual two-rank DeepSpeed gate: run nox -s ecosystem_deepspeed_multirank")
    assert REAL_TORCH_PYTHON, "DeepSpeed multirank requires an independent Torch interpreter"
    device = os.environ.get("JITTOR_TEST_DEVICES", "cpu")
    assert device in ("cpu", "cuda")
    devices = tuple(item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip())
    if device == "cuda":
        assert len(devices) == 2 and len(set(devices)) == 2
    else:
        assert not devices, "CPU Engine ranks require an empty CUDA allocation"
    root = os.environ.get("JITTOR_DEEPSPEED_MULTIRANK_ROOT", "").strip()
    assert root, "Set a new external JITTOR_DEEPSPEED_MULTIRANK_ROOT for this run"
    case = reserve_case(root, zero_stage, "AdamW", "bfloat16", 2, perturbation, device=device)
    cache = os.environ.get("JITTOR_DEEPSPEED_MULTIRANK_CACHE_ROOT") or str(Path(root) / "_caches")
    repository = Path(__file__).resolve().parents[3]
    sources = subprocess.check_output(
        ["git", "-C", str(repository), "ls-files", "--", "src", "python", "compat", "backends",
         "adapters", "noxfile.py"],
        text=True).splitlines()
    source_hashes = {name: hashlib.sha256((repository / name).read_bytes()).hexdigest()
                     for name in sources if (repository / name).is_file()}
    assert not subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True), \
        "Formal Engine measurements require a clean committed source tree"
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    (case / "source.json").write_text(json.dumps({"commit": commit, "sha256": source_hashes},
                                                indent=2, sort_keys=True) + "\n")
    reference_paths = run_pair(REAL_TORCH_PYTHON, "torch", case, cache, devices,
                               _runner_package_site(REAL_TORCH_PYTHON))
    actual_paths = run_pair(PYTHON, "jittor", case, cache, devices, _runner_package_site(PYTHON))
    errors = []
    for rank, (actual_path, reference_path) in enumerate(zip(actual_paths, reference_paths)):
        with np.load(actual_path) as actual, np.load(reference_path) as reference:
            errors.append(_compare_engine_outputs(
                actual, reference, device=device, world_size=2, rank=rank, dtype="bfloat16",
                optimizer="AdamW", zero_stage=zero_stage, perturbation=perturbation))
            metadata = json.loads(str(actual["metadata"]))
            assert metadata["jittor_commit"] == commit
            expected = ("deepspeed.runtime.bf16_optimizer.BF16_Optimizer" if zero_stage == 1 else
                        "deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer")
            assert metadata["optimizer_class"] == expected
            assert 0 < sum(metadata["optimizer_partition_sizes"]) < metadata["model_parameter_elements"]
    assert all(hashlib.sha256((repository / name).read_bytes()).hexdigest() == digest
               for name, digest in source_hashes.items()), "Source changed during DeepSpeed measurement"
    assert subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"],
                                   text=True).strip() == commit, "Commit changed during DeepSpeed measurement"
    assert not subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True), \
        "Source tree became dirty during Engine measurement"
    raw = [*actual_paths, *reference_paths]
    raw.extend(case.rglob("*.log"))
    raw.extend(case.rglob("*-launch.json"))
    raw.extend(case.rglob("launcher.json"))
    raw.extend((case / "protocol.json", case / "source.json"))
    result = {"status": "passed", "device": device, "world_size": 2, "dtype": "bfloat16",
              "optimizer": "AdamW", "zero_stage": zero_stage, "perturbation": perturbation,
              "commit": commit, "errors": errors,
              "raw_sha256": {str(path.relative_to(case)): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in raw}}
    (case / "comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("DeepSpeed two-rank BF16", zero_stage, perturbation, errors, case)

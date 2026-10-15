"""Admit measured FlashAttention CUDA training without granting CPU breadth."""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def artifact(role, path):
    path = Path(path).resolve()
    return {"role": role, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def platform_from_metadata(candidate, reference):
    actual, oracle = candidate["platform"], reference["platform"]
    if actual["cuda_visible_devices"] != oracle["cuda_visible_devices"]:
        raise ValueError("Oracle/candidate allocation mismatch")
    visible = actual["cuda_visible_devices"].split(",")
    rows = [[part.strip() for part in row] for row in csv.reader(actual["gpu_driver_inventory"])]
    allocated = [row for token in visible for row in rows if token.strip() in (row[0], row[2])]
    if len(allocated) != 1 or actual["actual_device_count"] != 1 or oracle["actual_device_count"] != 1:
        raise ValueError("FlashAttention gate requires one actually observed allocated CUDA device")
    if actual["actual_device"] != oracle["actual_device"] or actual["actual_device"] != allocated[0][1]:
        raise ValueError("Oracle/candidate physical accelerator mismatch")
    if tuple(candidate["actual_dispatch"]) != ("cuda", 0):
        raise ValueError("Candidate did not observe native CUDA dispatch")
    if not reference["binary_extension"].endswith((".so", ".pyd")):
        raise ValueError("Missing independent binary Torch extension")
    compiler = actual["compiler_environment"]
    toolkit = re.search(r"release ([0-9.]+)", compiler["nvcc"])
    if toolkit is None:
        raise ValueError("Unobserved candidate CUDA toolkit")
    return {"os": actual["os"], "cpu": actual["cpu"], "accelerator_model": actual["actual_device"],
            "card_count": 1, "card_indices": visible, "driver": allocated[0][3],
            "cuda": toolkit.group(1), "cann": "not-applicable", "nvcc": compiler["nvcc"],
            "gcc": compiler["cxx"], "nvidia_wheels": actual["nvidia_wheels"],
            "precision": candidate["precision"],
            "jittor_flags": {"use_cuda": 1, "use_acl": 0,
                             "graph_replay": "default" if candidate["native_flags"]["auto_graph_replay"] else "eager",
                             "tf32": candidate["precision"]["matmul"], "hf32": False},
            "oracle": {"is_shim": False, "torch_version": reference["torch"],
                       "torch_npu_version": "not-applicable", "binary_extension": reference["binary_extension"],
                       "cuda": oracle["torch_cuda"]},
            "native_aliases": candidate["native_flag_aliases"]}


def collect_case(directory, inventory, baseline, probe_hash, comparison, report, command, output):
    directory = Path(directory)
    metadata = {}
    results = []
    expected = {"base-comparison.json", "perturbed-comparison.json", "perturb-comparison.json",
                "perturbation.json", "training-reference-metadata.json", "training-candidate-metadata.json"}
    expected.update(runtime + "-" + label + suffix for runtime in ("torch", "jittor")
                    for label in ("base", "perturbed") for suffix in (".json", ".npz", ".log"))
    if {path.name for path in directory.iterdir() if path.is_file()} != expected:
        raise ValueError("Incomplete or unexpected FlashAttention gate artifacts: " + str(directory))
    checked = 0
    for label in ("base", "perturbed"):
        result = comparison.compare(directory / ("torch-" + label + ".npz"),
                                    directory / ("jittor-" + label + ".npz"))
        if result["status"] != "passed":
            raise ValueError("FlashAttention comparison no longer passes")
        original = read(directory / (label + "-comparison.json"))
        for runtime, field in (("torch", "reference_metadata"), ("jittor", "candidate_metadata")):
            meta = read(directory / (runtime + "-" + label + ".json"))
            if meta != original[field] or meta != result[field]:
                raise ValueError("Changed original runtime measurement")
            if meta["baseline"] != baseline or meta["script_sha256"] != probe_hash:
                raise ValueError("Unbound source baseline or runtime probe")
            if meta["upstream"] != inventory["upstream"]:
                raise ValueError("Upstream source identity mismatch")
            for source in inventory["sources"]:
                if meta["source_hashes"].get(source["path"]) != source["sha256"]:
                    raise ValueError("Unbound upstream source: " + source["path"])
            metadata[runtime, label] = meta
            with np.load(directory / (runtime + "-" + label + ".npz"), allow_pickle=False) as arrays:
                for name in arrays.files:
                    if not np.isfinite(arrays[name]).all():
                        raise ValueError("Nonfinite original training array")
                    if "grad" in name and not np.any(arrays[name]):
                        raise ValueError("Missing nonzero gradient: " + name)
        checked += 30
        results.extend(result["tensors"])
    for runtime in ("torch", "jittor"):
        base, changed = metadata[runtime, "base"], metadata[runtime, "perturbed"]
        if base["causal"] != changed["causal"] or base["head_dim"] != changed["head_dim"]:
            raise ValueError("Input perturbation changed the attention protocol")
        if any(base["input_hashes"][name] != changed["input_hashes"][name] for name in ("q", "k", "v", "target")):
            raise ValueError("Input perturbation changed fixed inputs")
        if base["input_hashes"]["noise"] == changed["input_hashes"]["noise"]:
            raise ValueError("Input perturbation did not change differentiable input")
        with np.load(directory / (runtime + "-base.npz")) as before, np.load(directory / (runtime + "-perturbed.npz")) as after:
            if not any(not np.array_equal(before[f"step{step}_output"], after[f"step{step}_output"]) for step in range(3)):
                raise ValueError("Perturbation did not affect actual attention outputs")
    candidate, reference = metadata["jittor", "base"], metadata["torch", "base"]
    measured = candidate["measurement"]
    platform = platform_from_metadata(candidate, reference)
    platform_path = output / (directory.name + "-platform.json")
    platform_path.write_text(json.dumps(platform, indent=2) + "\n")
    roles = {"reference": "torch-base.npz", "candidate": "jittor-base.npz", "comparison": "base-comparison.json",
             "base-comparison": "base-comparison.json", "perturb-comparison": "perturb-comparison.json",
             "training-reference-metadata": "training-reference-metadata.json",
             "training-candidate-metadata": "training-candidate-metadata.json"}
    bindings = [artifact(role, directory / name) for role, name in roles.items()]
    bindings.extend(artifact("raw/" + name, directory / name) for name in sorted(expected - set(roles.values())))
    bindings.append(artifact("platform", platform_path))
    limit = max(row["limit"] for row in results)
    return {"repo": "flash-attention", "case": measured["case"], "status": "passed", "level": 3, "tier": "core",
            "jittor_commit": baseline, "upstream": inventory["upstream"],
            "downstream_versions": {"candidate": candidate["flash_attn"], "oracle": reference["flash_attn"]},
            "registry_refs": [{"repo": "flash-attention", "axis": "attention_api", "member": candidate["api"]}],
            "device": "cuda", "observed_device": measured["device"], "platform": platform, "inputs": candidate["inputs"],
            "numerics": {"passed": True, "finite": True, "checked_tensors": checked,
                         "missing_tensors": [], "input_perturbation_verified": True,
                         "max_abs": max(row.get("max_abs", row["max_normalized_error"] * row["reference_scale"]) for row in results),
                         "max_scaled_relative": max(row["max_normalized_error"] for row in results), "atol": 0., "rtol": limit,
                         "parameter_gradients": {"requested": 18, "checked": 18},
                         "input_gradients": {"requested": 6, "checked": 6}, "per_tensor_criteria": results},
            "training": {"steps": measured["steps"], "trainable": len(measured["trainable_parameters"]),
                         "frozen": len(measured["frozen_parameters"]), "trainable_parameters": measured["trainable_parameters"],
                         "frozen_parameters": measured["frozen_parameters"], "optimizer": "SGD(lr=.01)",
                         "loss_trajectory_passed": True, "parameter_trajectory_passed": True},
            "fallback": {"policy": measured["fallback_policy"], "count": measured["fallback_count"]},
            "evidence": {"command": command, "report": report,
                         "gate_nodeid": "compat/tests/torch/test_ecosystem_flash_attention.py::test_real_flash_attention_training",
                         "artifacts": bindings}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "upstream", "evidence-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--report", default="docs/results/2026-10-15-flash-attention-training-oracles.md")
    args = parser.parse_args()
    root = args.root.resolve()
    evidence = args.evidence_dir.resolve()
    output = args.output_dir.resolve()
    if output == root or root in output.parents or output == evidence or evidence in output.parents:
        parser.error("scorecard products must stay outside source and immutable runtime evidence")
    tools = root / "agent/skills/torch-compat-repo-runbook/scripts"
    sys.path.insert(0, str(tools))
    import registry_inventory
    import audit_coverage
    inventory = registry_inventory.inventory("flash-attention", args.upstream.resolve())
    script = "agent/skills/flash-attention-torch-compat/scripts/flash_training_probe.py"
    probe_hash = hashlib.sha256(subprocess.check_output(["git", "-C", str(root), "show", args.baseline + ":" + script])).hexdigest()
    path = root / "agent/skills/flash-attention-torch-compat/scripts/compare_training.py"
    spec = importlib.util.spec_from_file_location("flash_training_comparison", path)
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    output.mkdir(parents=True, exist_ok=False)
    directories = sorted(path for path in evidence.iterdir() if path.is_dir())
    if len(directories) != 12:
        raise ValueError("Expected the complete twelve-case FlashAttention CUDA training gate")
    command = "JITTOR_TEST_DEVICES=cuda JITTOR_REQUIRE_FLASH_ATTENTION=1 JITTOR_REQUIRE_REAL_TORCH=1 <jittor-python> -m pytest -q compat/tests/torch/test_ecosystem_flash_attention.py"
    records = [collect_case(path, inventory, args.baseline, probe_hash, comparison, args.report, command, output) for path in directories]
    if len({row["case"] for row in records}) != 12:
        raise ValueError("Duplicate FlashAttention training cases")
    result = audit_coverage.scorecard({"flash-attention": inventory}, records, args.baseline, root=root)
    result["auditor_commit"] = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    result["scope"] = "Independent CUDA L3 executions. CPU upstream execution is unavailable, so shared accelerator breadth remains zero; no CPU prerequisite is waived."
    (output / "scorecard.json").write_text(json.dumps(result, indent=2) + "\n")
    print(len(result["accepted"]), "accepted;", len(result["rejected"]), "rejected; CPU prerequisite preserved")
    if result["rejected"]:
        raise ValueError(result["rejected"])


if __name__ == "__main__":
    main()

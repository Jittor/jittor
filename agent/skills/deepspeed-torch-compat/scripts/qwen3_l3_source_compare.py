"""Verify saved Qwen3 weights against their exact two-rank training source.

Run in the independent PyTorch interpreter. This replays stored evidence on CPU;
it does not rerun training, prove a new device, or replace L2/L4 comparisons.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

CHECKPOINT = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
PROBE = "2960393d80038115eb7edffa34a83e005e28f926a566a1723e67a5454aae8ebd"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_equal_finite(saved, original, label):
    """Reject non-finite evidence even when both sides contain identical Inf."""
    require(np.isfinite(saved).all() and np.isfinite(original).all(),
            label + ": non-finite saved or training values")
    require(np.array_equal(saved, original), label + ": saved tensor differs from training")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(source, roundtrip, stage):
    require(not hasattr(torch, "_torch_compat_install_context") and hasattr(torch, "_C"),
            "independent binary PyTorch oracle required")
    training_comparison = read(source / "comparison.json")
    require(training_comparison["status"] == "passed"
            and training_comparison["errors"] == []
            and training_comparison["evidence_gaps"] == [], "training comparison failed")
    manifests = {}
    source_reports = {}
    diagnostics = {}
    # Validate every manifest before opening large tensor files.
    for runtime in ("oracle", "shim"):
        for rank in (0, 1):
            label = runtime + "/rank%d" % rank
            directory = roundtrip / label
            training = source / label
            item = read(directory / "report.json")
            report = read(training / "report.json")
            diagnostic = read(directory / "roundtrip-diagnostic.json")
            for record in (item, report):
                require(record["status"] == "passed" and record["runtime"] == runtime
                        and record["rank"] == rank and record["world_size"] == 2
                        and record["zero_stage"] == stage and record["device"] == "npu",
                        label + ": manifest identity/device/stage mismatch")
                require(record["parameter_count"] == 310, label + ": incomplete parameters")
                require(record["fallback_delta"] == (0 if runtime == "shim" else None),
                        label + ": fallback evidence mismatch")
            require(report["steps"] == 3 and report["dtype"] == "float32"
                    and report["backend"] == "hccl", label + ": training scope mismatch")
            require(report["optimizer_config"]["eps"] == 1e-8
                    and report["checkpoint_sha256"] == CHECKPOINT,
                    label + ": optimizer/checkpoint mismatch")
            require(report["torch_is_shim"] is (runtime == "shim"),
                    label + ": runtime identity mismatch")
            if runtime == "oracle":
                require(report["torch_has_c_extension"] is True, label + ": binary oracle absent")
            require(item["source_training_report_sha256"] == sha(training / "report.json"),
                    label + ": source report SHA mismatch")
            require(item["probe_sha256"] == PROBE, label + ": unknown roundtrip probe")
            require(item["parameter_worst_abs"] == 0 and item["changed_parameter_names"] == [],
                    label + ": parameters changed during roundtrip")
            require(diagnostic["parameter_worst_abs"] == 0
                    and diagnostic["changed_parameter_names"] == [],
                    label + ": parameter diagnostic mismatch")
            for field in ("loss", "logits"):
                error = item[field + "_worst_abs"]
                tolerance = diagnostic[field + "_tolerance"]
                require(math.isfinite(error) and error >= 0 and math.isfinite(tolerance)
                        and tolerance >= 5e-5 and error <= tolerance
                        and diagnostic[field + "_worst_abs"] == error,
                        label + ": task roundtrip mismatch: " + field)
            require(len(report["parameters"]) == 310, label + ": missing parameter names")
            require(item["tokenizer_text"] == "计图兼容层测试"
                    and item["token_ids"] == [37643, 28029, 114288, 99371, 81705],
                    label + ": tokenizer mismatch")
            manifests[label] = item
            source_reports[label] = report
            diagnostics[label] = diagnostic
    base = source_reports["oracle/rank0"]
    for label, report in source_reports.items():
        require(report["parameters"] == base["parameters"]
                and report["versions"] == base["versions"],
                label + ": cross-runtime parameter/version mismatch")
    counts = {}
    for label, report in source_reports.items():
        directory = roundtrip / label
        keys = set()
        for file in sorted((directory / "saved-model").glob("*.safetensors")):
            with safe_open(file, framework="np") as weights:
                for name in weights.keys():
                    require(name not in keys and name in report["parameters"],
                            label + ": duplicate/unexpected saved parameter " + name)
                    keys.add(name)
                    saved = weights.get_tensor(name)
                    original = np.load(source / label / "step2/updated" / (name + ".npy"),
                                       mmap_mode="r", allow_pickle=False)
                    require(saved.shape == original.shape == tuple(report["parameters"][name])
                            and saved.dtype == original.dtype == np.dtype("float32"),
                            label + ": saved shape/dtype mismatch " + name)
                    a, b = saved.reshape(-1), original.reshape(-1)
                    for start in range(0, a.size, 1024 * 1024):
                        require_equal_finite(a[start:start + 1024 * 1024],
                                             b[start:start + 1024 * 1024],
                                             label + ": " + name)
        require(keys == set(report["parameters"]) and len(keys) == 310,
                label + ": incomplete saved weights")
        counts[label] = len(keys)
    return dict(status="passed", zero_stage=stage, manifests=4,
                parameter_tensors_per_manifest=310, saved_tensor_checks=counts,
                saved_vs_training_worst_abs=0, parameter_roundtrip_worst_abs=0,
                reports=manifests, diagnostics=diagnostics,
                source=str(source.resolve()), roundtrip=str(roundtrip.resolve()),
                source_comparison_sha256=sha(source / "comparison.json"),
                comparator_sha256=sha(Path(__file__)),
                oracle_version=str(torch.__version__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--roundtrip", type=Path, required=True)
    parser.add_argument("--zero-stage", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "fresh output required")
    result = verify(args.source, args.roundtrip, args.zero_stage)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("reports", "diagnostics")}, indent=2))


if __name__ == "__main__":
    main()

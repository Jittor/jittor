"""Compare Qwen3 real-checkpoint L3 evidence against genuine PyTorch."""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def load_report(root, runtime, rank):
    path = root / runtime / ("rank%d.json" % rank)
    report = json.loads(path.read_text())
    assert report["status"] == "l3-roundtrip-passed", (path, report.get("status"))
    assert report["runtime"] == runtime
    assert report["rank"] == rank
    assert report["world_size"] == 2
    assert report["backend"] == "hccl"
    assert report["zero_stage"] == 0
    assert report["parameter_count"] > 500_000_000
    assert report["roundtrip_loss_abs"] <= 1e-5
    assert report["roundtrip_logits_max_abs"] <= 1e-5
    if runtime == "shim":
        assert report["fallback_delta"] == 0
    return report


def compare_values(reference, candidate):
    errors = [abs(float(a) - float(b)) for a, b in zip(reference, candidate)]
    worst = max(errors, default=0.0)
    scale = max([abs(float(value)) for value in reference] + [1e-12])
    tolerance = 5e-5 + 5e-5 * scale
    assert worst <= tolerance, (worst, tolerance)
    return worst, worst / scale


def compare_saved_tensors(root, oracle, shim):
    oracle_files = {item["name"] for item in oracle["saved_weight_sha256"]}
    shim_files = {item["name"] for item in shim["saved_weight_sha256"]}
    assert oracle_files == shim_files
    count = 0
    for name in sorted(oracle_files):
        oracle_path = root / "oracle" / "saved-rank0" / name
        shim_path = root / "shim" / "saved-rank0" / name
        with safe_open(oracle_path, framework="pt", device="cpu") as left:
            with safe_open(shim_path, framework="pt", device="cpu") as right:
                assert left.keys() == right.keys()
                for key in left.keys():
                    a = left.get_tensor(key)
                    b = right.get_tensor(key)
                    assert a.shape == b.shape and a.dtype == b.dtype
                    assert torch.equal(a, b), key
                    count += 1
    oracle_config = json.loads(
        (root / "oracle" / "saved-rank0" / "config.json").read_text())
    shim_config = json.loads(
        (root / "shim" / "saved-rank0" / "config.json").read_text())
    assert oracle_config == shim_config
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    assert not hasattr(torch, "_torch_compat_install_context")
    reports = {
        runtime: [load_report(args.root, runtime, rank) for rank in (0, 1)]
        for runtime in ("oracle", "shim")
    }
    for runtime in ("oracle", "shim"):
        a, b = reports[runtime]
        assert a["source_checkpoint"] == b["source_checkpoint"]
        assert a["saved_weight_sha256"] == b["saved_weight_sha256"]
        assert a["saved_weight_bytes"] == b["saved_weight_bytes"]

    worst_loss = 0.0
    worst_logit_abs = 0.0
    worst_logit_rel_field = 0.0
    for rank in (0, 1):
        oracle = reports["oracle"][rank]
        shim = reports["shim"][rank]
        assert oracle["source_checkpoint"] == shim["source_checkpoint"]
        assert oracle["parameter_count"] == shim["parameter_count"]
        worst_loss = max(worst_loss, abs(oracle["loss"] - shim["loss"]))
        absolute, relative = compare_values(
            oracle["logits_last_first8"], shim["logits_last_first8"])
        worst_logit_abs = max(worst_logit_abs, absolute)
        worst_logit_rel_field = max(worst_logit_rel_field, relative)

    tensor_count = compare_saved_tensors(
        args.root, reports["oracle"][0], reports["shim"][0])
    result = {
        "status": "passed",
        "ranks": 2,
        "parameter_count": reports["shim"][0]["parameter_count"],
        "source_checkpoint": reports["shim"][0]["source_checkpoint"],
        "loss_max_abs": worst_loss,
        "logits_max_abs": worst_logit_abs,
        "logits_max_rel_field": worst_logit_rel_field,
        "roundtrip_tensor_count": tensor_count,
        "roundtrip_tensor_mismatches": 0,
        "shim_fallback_delta": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

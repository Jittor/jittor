import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    details = {}
    categories = {}
    for rank in (0, 1):
        oracle_report = json.loads((args.root / "oracle" / f"rank{rank}.json").read_text())
        shim_report = json.loads((args.root / "shim" / f"rank{rank}.json").read_text())
        assert oracle_report["runtime"] == "oracle"
        assert shim_report["runtime"] == "shim"
        for field in ("rank", "world_size", "backend", "deepspeed", "steps", "zero_stage", "contiguous_gradients", "parameter_count"):
            assert oracle_report[field] == shim_report[field], (rank, field)
        assert shim_report["fallback_delta"] == 0

        oracle = np.load(args.root / "oracle" / f"rank{rank}.npz", allow_pickle=False)
        shim = np.load(args.root / "shim" / f"rank{rank}.npz", allow_pickle=False)
        assert set(oracle.files) == set(shim.files)
        details[str(rank)] = {}
        for key in oracle.files:
            reference = np.asarray(oracle[key])
            actual = np.asarray(shim[key])
            assert reference.shape == actual.shape, (rank, key, reference.shape, actual.shape)
            assert reference.dtype == actual.dtype, (rank, key, reference.dtype, actual.dtype)
            assert np.isfinite(actual).all(), (rank, key)
            absolute = float(np.max(np.abs(actual - reference))) if reference.size else 0.0
            scale = float(np.max(np.abs(reference))) if reference.size else 0.0
            tolerance = 2e-5 + 2e-4 * scale
            assert absolute <= tolerance, (rank, key, absolute, tolerance, scale)
            kind = key.split("/", 1)[0]
            categories.setdefault(kind, []).append((absolute, scale, rank, key))
            details[str(rank)][key] = {
                "max_abs": absolute,
                "field_scaled_relative": absolute / max(scale, 1e-30),
                "tolerance": tolerance,
            }

    summary = {}
    for kind, values in categories.items():
        worst = max(values, key=lambda item: item[0])
        summary[kind] = {
            "worst_abs": worst[0],
            "reference_scale": worst[1],
            "rank": worst[2],
            "key": worst[3],
        }

    updated_sync = {}
    for runtime in ("oracle", "shim"):
        rank0 = np.load(args.root / runtime / "rank0.npz", allow_pickle=False)
        rank1 = np.load(args.root / runtime / "rank1.npz", allow_pickle=False)
        updated = [key for key in rank0.files if key.startswith("updated/")]
        assert updated
        updated_sync[runtime] = max(
            float(np.max(np.abs(rank0[key] - rank1[key]))) for key in updated
        )
        assert updated_sync[runtime] == 0.0, (runtime, updated_sync[runtime])

    zero_stages = {
        json.loads((args.root / runtime / "rank0.json").read_text())["zero_stage"]
        for runtime in ("oracle", "shim")
    }
    assert len(zero_stages) == 1, zero_stages
    zero_stage = zero_stages.pop()
    contiguous_modes = {
        json.loads((args.root / runtime / "rank0.json").read_text())["contiguous_gradients"]
        for runtime in ("oracle", "shim")
    }
    assert len(contiguous_modes) == 1, contiguous_modes
    report = {
        "status": "passed",
        "scope": "DeepSpeed 0.17.6 ZeRO Stage %d, two Ascend NPUs, three FP32 AdamW steps" % zero_stage,
        "oracle": "independent genuine PyTorch plus torch_npu processes",
        "contiguous_gradients": contiguous_modes.pop(),
        "summary": summary,
        "updated_parameter_rank_sync_max_abs": updated_sync,
        "details": details,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["summary"], sort_keys=True))
    print(json.dumps({"updated_parameter_rank_sync_max_abs": updated_sync}, sort_keys=True))


if __name__ == "__main__":
    main()

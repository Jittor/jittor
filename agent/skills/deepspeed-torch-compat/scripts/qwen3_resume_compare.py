"""Compare two-rank Qwen3 ZeRO-1/2 checkpoint/resume against independent PyTorch."""
import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--zero-stage", type=int, choices=(1,2,3), default=1)
args = parser.parse_args()
root = args.root
reports = {}
for runtime in ("oracle", "shim"):
    for rank in (0, 1):
        p = root / runtime / ("rank%d" % rank) / "report.json"
        report = json.loads(p.read_text())
        assert report["runtime"] == runtime and report["rank"] == rank
        assert report["status"] == "executed"
        assert report["model_restored_exact"]
        assert report["zero_stage"] == args.zero_stage
        if runtime == "shim":
            assert report["fallback_delta"] == 0
        reports[(runtime, rank)] = report

param_files = sorted((root / "oracle" / "rank0" / "first" / "updated").glob("*.npy"))
assert len(param_files) == 310, len(param_files)
fields = ["input_ids", "loss", "logits", "input_grad"]
fields += ["updated/" + p.stem for p in param_files]
assert len(fields) == 314
rows = []
scales = {}
for phase in ("first", "resumed"):
    for rank in (0, 1):
        aroot = root / "oracle" / ("rank%d" % rank) / phase
        broot = root / "shim" / ("rank%d" % rank) / phase
        for name in fields:
            a = np.load(aroot / (name + ".npy"), mmap_mode="r")
            b = np.load(broot / (name + ".npy"), mmap_mode="r")
            assert a.shape == b.shape and a.dtype == b.dtype, (phase, rank, name)
            assert np.isfinite(a).all() and np.isfinite(b).all()
            difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
            worst = float(np.max(difference)) if a.size else 0.0
            scale = float(np.max(np.abs(a))) if a.size else 0.0
            category = "updated" if name.startswith("updated/") else name
            key = (phase, rank, category)
            scales[key] = max(scales.get(key, 0.0), scale)
            rows.append(dict(phase=phase, rank=rank, name=name,
                             category=category, worst_abs=worst,
                             oracle_scale=scale, elements=int(a.size)))

errors = []
category_worst = {}
for row in rows:
    key = (row["phase"], row["rank"], row["category"])
    tolerance = (0.0 if row["category"] == "input_ids"
                 else 5e-5 + 5e-5 * scales[key])
    row["tolerance"] = tolerance
    row["bad"] = row["worst_abs"] > tolerance
    category_worst[str(key)] = max(
        category_worst.get(str(key), 0.0), row["worst_abs"])
    if row["bad"]:
        errors.append(dict(type="cross_runtime", **row))

replay_worst = {}
replay_nonexact_count = {}
rank_sync_worst = {}
for runtime in ("oracle", "shim"):
    for rank in (0, 1):
        p = root / runtime / ("rank%d" % rank)
        for name in fields:
            a = np.load(p / "first" / (name + ".npy"), mmap_mode="r")
            b = np.load(p / "resumed" / (name + ".npy"), mmap_mode="r")
            assert a.shape == b.shape and a.dtype == b.dtype
            worst = float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
            replay_worst[runtime] = max(replay_worst.get(runtime, 0.0), worst)
            if worst != 0.0:
                replay_nonexact_count[runtime] = replay_nonexact_count.get(runtime, 0) + 1
            # This is a numerical training contract, not a bitwise serialization
            # contract. The model weights at load were checked separately by SHA256.
            # Keep this stricter absolute floor from the fixed model tolerance.
            if worst > 5e-5:
                errors.append(dict(type="checkpoint_replay", runtime=runtime,
                                   rank=rank, name=name, worst_abs=worst,
                                   tolerance=5e-5))
    for phase in ("first", "resumed"):
        p0 = root / runtime / "rank0" / phase
        p1 = root / runtime / "rank1" / phase
        for name in fields[4:]:
            a = np.load(p0 / (name + ".npy"), mmap_mode="r")
            b = np.load(p1 / (name + ".npy"), mmap_mode="r")
            worst = float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
            rank_sync_worst[runtime] = max(rank_sync_worst.get(runtime, 0.0), worst)
            if worst != 0.0:
                errors.append(dict(type="rank_sync", runtime=runtime,
                                   phase=phase, name=name, worst_abs=worst))

result = dict(
    status="passed" if not errors else "failed",
    zero_stage=args.zero_stage, parameters=310, fields_per_phase_rank=len(fields),
    cross_runtime_fields=len(rows),
    category_worst_abs=category_worst,
    checkpoint_replay_worst_abs=replay_worst,
    checkpoint_replay_nonexact_fields=replay_nonexact_count,
    rank_sync_worst_abs=rank_sync_worst,
    errors=errors,
    reports={runtime + str(rank): reports[(runtime, rank)]
             for runtime in ("oracle", "shim") for rank in (0, 1)})
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(result, indent=2))
print(json.dumps({key: result[key] for key in
                  ("status", "parameters", "cross_runtime_fields",
                   "checkpoint_replay_worst_abs", "rank_sync_worst_abs")},
                 indent=2))
print("errors", len(errors))

"""Compare two-rank Qwen3 cache, greedy, and beam generation with real PyTorch."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--model-root", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
expected = (
    "input_ids", "prefill_last_logits", "decode_last_logits",
    "recomputed_last_logits", "greedy_cached", "greedy_uncached",
    "beam_cached",
)
floating = frozenset(expected[1:4])
reports = {}
arrays = {}
errors = []
probe = Path(__file__).with_name("qwen3_generation_probe_mask.py")
probe_hash = hashlib.sha256(probe.read_bytes()).hexdigest()
for runtime in ("oracle", "shim"):
    for rank in (0, 1):
        directory = args.root / runtime / ("rank%d" % rank)
        report = json.loads((directory / "report.json").read_text())
        assert report["status"] == "passed"
        assert report["runtime"] == runtime and report["rank"] == rank
        assert report["probe_sha256"] == probe_hash
        assert Path(report["model_dir"]).resolve() == (
            args.model_root / runtime / ("rank%d" % rank) / "saved-model").resolve()
        assert set(report["records"]) == set(expected)
        assert report["greedy_cached"] == report["greedy_uncached"]
        if runtime == "shim":
            assert report["fallback_delta"] == 0
        reports[runtime, rank] = report
        for name in expected:
            array = np.load(directory / (name + ".npy"), allow_pickle=False)
            record = report["records"][name]
            assert record["shape"] == list(array.shape)
            assert record["dtype"] == str(array.dtype)
            assert record["device"] == "npu"
            assert np.isfinite(array).all()
            arrays[runtime, rank, name] = array
        assert np.array_equal(
            arrays[runtime, rank, "greedy_cached"],
            arrays[runtime, rank, "greedy_uncached"])
        assert np.array_equal(
            arrays[runtime, rank, "greedy_cached"],
            np.asarray(report["greedy_cached"]))
        assert np.array_equal(
            arrays[runtime, rank, "beam_cached"],
            np.asarray(report["beam"]))
        assert report["cache_worst_abs"] <= report["cache_tolerance"]
metrics = []
for name in expected:
    scale = max(float(np.max(np.abs(arrays["oracle", rank, name])))
                for rank in (0, 1))
    tolerance = 5e-5 + 5e-5 * scale if name in floating else 0.0
    for rank in (0, 1):
        a, b = arrays["oracle", rank, name], arrays["shim", rank, name]
        assert a.shape == b.shape and a.dtype == b.dtype
        delta = float(np.max(np.abs(
            a.astype(np.float64) - b.astype(np.float64))))
        metrics.append(dict(field=name, rank=rank, worst_abs=delta,
                            tolerance=tolerance))
        if delta > tolerance:
            errors.append(dict(field=name, rank=rank, worst_abs=delta,
                               tolerance=tolerance))
    for runtime in ("oracle", "shim"):
        a, b = arrays[runtime, 0, name], arrays[runtime, 1, name]
        assert a.shape == b.shape and a.dtype == b.dtype
        if not np.array_equal(a, b):
            errors.append(dict(field=name, runtime=runtime,
                               reason="rank mismatch"))
result = dict(status="passed" if not errors else "failed",
              scope="same trained model, single-node two-NPU generation",
              fields=len(metrics), metrics=metrics, errors=errors,
              cache_worst_abs={
                  runtime + str(rank): reports[runtime, rank]["cache_worst_abs"]
                  for runtime in ("oracle", "shim") for rank in (0, 1)},
              greedy_tokens=reports["shim", 0]["greedy_cached"],
              beam_tokens=reports["shim", 0]["beam"])
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(result, indent=2))
print(json.dumps({key: result[key] for key in
                  ("status", "fields", "errors", "cache_worst_abs",
                   "greedy_tokens", "beam_tokens")}, indent=2))

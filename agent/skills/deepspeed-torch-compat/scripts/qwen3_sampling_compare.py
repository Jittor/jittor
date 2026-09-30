"""Audit two-rank Qwen3 default sampling against an independent torch oracle."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--generation-root", type=Path, required=True)
parser.add_argument("--model-root", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

reports = {}
errors = []
for runtime in ("oracle", "shim"):
    config_path = args.model_root / runtime / "rank0" / "saved-model" / "generation_config.json"
    config = json.loads(config_path.read_text())
    expected = {"do_sample": True, "top_k": 20, "top_p": 0.95, "temperature": 0.6}
    for name, value in expected.items():
        assert config[name] == value, (runtime, name, config[name], value)
    for rank in (0, 1):
        p = args.root / runtime / ("rank%d" % rank) / "report.json"
        report = json.loads(p.read_text())
        assert report["runtime"] == runtime and report["rank"] == rank
        assert report["status"] == "passed"
        if runtime == "shim":
            assert report["fallback_delta"] == 0
        reports[(runtime, rank)] = report

def load(runtime, rank, name):
    return np.load(args.root / runtime / ("rank%d" % rank) / (name + ".npy"))

def worst(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert np.isfinite(a).all() and np.isfinite(b).all()
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0

def tolerance(reference):
    return 5e-5 + 5e-5 * float(np.max(np.abs(reference))) if reference.size else 5e-5

def candidates(runtime, rank):
    indices = load(runtime, rank, "top20_indices")[0]
    logits = load(runtime, rank, "top20_values")[0].astype(np.float64) / 0.6
    weights = np.exp(logits - np.max(logits))
    probabilities = weights / weights.sum()
    ascending = np.argsort(probabilities)
    remove = np.cumsum(probabilities[ascending]) <= 0.05
    keep = np.ones(20, dtype=bool)
    keep[ascending[remove]] = False
    return indices[keep], probabilities[keep] / probabilities[keep].sum()

metrics = {}
for rank in (0, 1):
    for runtime in ("oracle", "shim"):
        values = load(runtime, rank, "full_sorted_values")
        indices = load(runtime, rank, "full_sorted_indices")
        source = np.load(
            args.generation_root / runtime / ("rank%d" % rank) /
            "prefill_last_logits.npy").reshape(values.shape)
        assert values.shape == indices.shape == source.shape
        assert np.all(np.diff(values, axis=-1) <= 0), (runtime, rank, "sort_order")
        assert np.array_equal(np.take_along_axis(source, indices, -1), values), (
            runtime, rank, "sort_indices")
        top_values = load(runtime, rank, "top20_values")
        top_indices = load(runtime, rank, "top20_indices")
        assert top_values.shape == top_indices.shape == (1, 20)
        assert np.array_equal(
            np.take_along_axis(source, top_indices, -1), top_values), (
            runtime, rank, "top20_indices")
        sampled = load(runtime, rank, "sample_default")
        repeated = load(runtime, rank, "sample_default_repeat")
        assert np.array_equal(sampled, repeated), (runtime, rank, "seed_replay")
        assert sampled.shape == (1, 8), (runtime, rank, sampled.shape)
        assert np.array_equal(sampled[:, :4], load(runtime, rank, "input_ids"))
    for name in ("full_sorted_values", "top20_values",
                 "first_step_top2_values", "first_step_top2_probabilities"):
        a, b = load("oracle", rank, name), load("shim", rank, name)
        delta, limit = worst(a, b), tolerance(a)
        metrics[name + "_rank" + str(rank)] = {
            "worst_abs": delta, "tolerance": limit}
        if delta > limit:
            errors.append({"rank": rank, "field": name,
                           "worst_abs": delta, "tolerance": limit})
    for name in ("top20_indices", "first_step_top2_indices", "input_ids"):
        if not np.array_equal(load("oracle", rank, name), load("shim", rank, name)):
            errors.append({"rank": rank, "field": name, "reason": "indices differ"})
    oracle_ids, oracle_prob = candidates("oracle", rank)
    shim_ids, shim_prob = candidates("shim", rank)
    if not np.array_equal(oracle_ids, shim_ids):
        errors.append({"rank": rank, "field": "top_p_candidates"})
    delta = float(np.max(np.abs(oracle_prob - shim_prob)))
    metrics["top_p_rank" + str(rank)] = {
        "kept": int(len(oracle_ids)), "candidate_ids": oracle_ids.tolist(),
        "probability_worst_abs": delta}
    if delta > 5e-5:
        errors.append({"rank": rank, "field": "top_p_probabilities",
                       "worst_abs": delta, "tolerance": 5e-5})
    metrics["full_sort_index_mismatches_rank" + str(rank)] = int(
        np.count_nonzero(load("oracle", rank, "full_sorted_indices") !=
                         load("shim", rank, "full_sorted_indices")))

for runtime in ("oracle", "shim"):
    for name in ("sample_default", "sample_default_repeat"):
        if not np.array_equal(load(runtime, 0, name), load(runtime, 1, name)):
            errors.append({"runtime": runtime, "field": name,
                           "reason": "ranks differ under same manual seed"})

token_mismatches = {}
for rank in (0, 1):
    a = load("oracle", rank, "sample_default")
    b = load("shim", rank, "sample_default")
    token_mismatches[str(rank)] = {
        "new_token_mismatches": int(np.count_nonzero(a[:, 4:] != b[:, 4:])),
        "oracle_tokens": a.tolist(),
        "shim_tokens": b.tolist(),
    }
l4_tokens_equal = all(row["new_token_mismatches"] == 0
                      for row in token_mismatches.values())
result = {
    "status": ("failed" if errors else
               "passed" if l4_tokens_equal else "partial"),
    "operator_semantics_status": "passed" if not errors else "failed",
    "fixed_seed_token_status": "passed" if l4_tokens_equal else "failed",
    "sampling_l4_subtask_status": (
        "passed" if not errors and l4_tokens_equal else "failed"),
    "scope": "Qwen3 default sampling only; not the full DeepSpeed L4 gate",
    "oracle_is_independent": True,
    "generation_config": expected,
    "generation_config_sha256": hashlib.sha256(
        (args.model_root / "oracle" / "rank0" / "saved-model" /
         "generation_config.json").read_bytes()).hexdigest(),
    "metrics": metrics,
    "token_mismatches": token_mismatches,
    "errors": errors,
    "probe_sha256": {runtime + str(rank): reports[(runtime, rank)]["probe_sha256"]
                     for runtime in ("oracle", "shim") for rank in (0, 1)},
}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(result, indent=2))
print(json.dumps({key: result[key] for key in
                  ("status", "operator_semantics_status",
                   "fixed_seed_token_status", "sampling_l4_subtask_status", "errors")},
                 indent=2))

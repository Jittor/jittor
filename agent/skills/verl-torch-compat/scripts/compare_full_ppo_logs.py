#!/usr/bin/env python3
"""Parse PPO logs; compare reported metrics only with paired run manifests.

Manifests constrain a comparison, but are self-reported metadata, not proof of
numerical parity, actual device execution, zero fallback, or an L4/L5 pass.
Schema 1 requires role/framework/backend/world_size/model/dtype/exit_code,
SHA-256 fields log_sha256/weights_sha256/input_sha256/verl_source_sha256,
downstream_versions (verl/transformers/tensordict/ray), completed_steps and
protocol (config_sha256/warmup_steps/timed_steps/throughput_scope/memory_metric).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Optional


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMBER = re.compile(
    r"^(?:np\.[A-Za-z0-9_]+\()?"
    r"([-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf))\)?$",
    re.IGNORECASE,
)
STAGE_TIMINGS = ("timing_s/gen", "timing_s/update_actor", "timing_s/update_critic")

REQUIRED = (
    "actor/loss",
    "actor/grad_norm",
    "critic/loss",
    "critic/grad_norm",
    "actor/perf/max_memory_allocated_gb",
    "critic/perf/max_memory_allocated_gb",
    "timing_s/step",
    "perf/total_num_tokens",
    "perf/throughput",
)


def _number(value: str) -> Optional[float]:
    match = NUMBER.match(value.strip())
    return float(match.group(1)) if match else None


def parse_log(path: Path) -> dict[str, object]:
    raw_log = path.read_bytes()
    return _parse_metrics(raw_log.decode(errors="replace"), str(path),
                          hashlib.sha256(raw_log).hexdigest())


def parse_text(text: str, source: str = "<RAM>") -> dict[str, object]:
    """Parse a RAM string; SHA binds its exact UTF-8 bytes, with no log-file IO."""
    if not isinstance(text, str) or not isinstance(source, str):
        raise TypeError("text and source must be strings")
    return _parse_metrics(text, source, hashlib.sha256(text.encode("utf-8")).hexdigest())


def _parse_metrics(text: str, source: str, log_sha256: str) -> dict[str, object]:
    text = ANSI.sub("", text).replace("\r", "\n")
    metric_lines = []
    for line in text.splitlines():
        match = re.search(r"\bstep:", line)
        if match:
            metric_lines.append(line[match.start():])
    if not metric_lines:
        raise ValueError(f"{source}: no completed step metrics")
    steps = []
    for line in metric_lines:
        fields: dict[str, float] = {}
        seen = set()
        for item in line.split(" - "):
            if ":" not in item:
                continue
            key, raw = item.split(":", 1)
            key = key.strip()
            if key in seen:
                raise ValueError(f"{source}: duplicate metric {key}")
            seen.add(key)
            value = _number(raw)
            if value is not None:
                fields[key] = value
            elif key in STAGE_TIMINGS:
                raise ValueError(f"{source}: invalid stage timing {key}")
        missing = [key for key in ("step",) + REQUIRED if key not in fields]
        if missing:
            raise ValueError(f"{source}: missing metrics: {', '.join(missing)}")
        nonfinite = [key for key, value in fields.items() if not math.isfinite(value)]
        if nonfinite:
            raise ValueError(f"{source}: non-finite metrics: {', '.join(nonfinite)}")
        step = fields.pop("step")
        if step != len(steps) + 1:
            raise ValueError(f"{source}: steps must be unique and consecutive from 1")
        for key in ("timing_s/step", "perf/throughput", "perf/total_num_tokens",
                    "actor/perf/max_memory_allocated_gb", "critic/perf/max_memory_allocated_gb"):
            if fields[key] <= 0:
                raise ValueError(f"{source}: {key} must be positive")
        tokens = fields["perf/total_num_tokens"]
        if tokens != int(tokens):
            raise ValueError(f"{source}: token count must be an integer")
        for key in STAGE_TIMINGS:
            if key in fields and fields[key] < 0:
                raise ValueError(f"{source}: stage timing {key} must be nonnegative")
        steps.append({"step": int(step), "metrics": {key: fields[key] for key in REQUIRED},
                      "stage_metrics": {key: fields.get(key) for key in STAGE_TIMINGS}})
    if "Training Progress: 100%" not in text:
        raise ValueError(f"{source}: training progress did not reach 100%")
    if "Traceback (most recent call last):" in text:
        raise ValueError(f"{source}: traceback present despite a metric line")
    return {"log": source, "log_sha256": log_sha256,
            "status": "log_smoke_only", "steps": steps, "metrics": steps[-1]["metrics"],
            "stage_metrics": steps[-1]["stage_metrics"]}


def _digest(value, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"manifest {name} must be a lowercase SHA-256")


def validate_manifest(manifest, parsed, role: str) -> None:
    """Validate metadata structure and bind it to every observed log step."""
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    if type(manifest.get("schema")) is not int or manifest["schema"] != 1:
        raise ValueError("manifest schema must be 1")
    if manifest.get("role") != role:
        raise ValueError(f"manifest role must be {role}")
    expected = "jittor" if role == "candidate" else "torch"
    if manifest.get("framework") != expected or manifest.get("backend") not in ("npu", "cuda"):
        raise ValueError(f"manifest requires {expected} on npu or cuda")
    if type(manifest.get("world_size")) is not int or manifest["world_size"] < 1:
        raise ValueError("manifest world_size must be a positive integer")
    if type(manifest.get("exit_code")) is not int or manifest["exit_code"] != 0:
        raise ValueError("manifest requires exit_code 0")
    for key in ("log_sha256", "weights_sha256", "input_sha256", "verl_source_sha256"):
        _digest(manifest.get(key), key)
    if manifest["log_sha256"] != parsed["log_sha256"]:
        raise ValueError("manifest log hash does not match parsed log")
    if not isinstance(manifest.get("model"), str) or not manifest["model"].strip():
        raise ValueError("manifest model must be a nonempty identifier")
    if manifest.get("dtype") not in ("fp32", "bf16", "fp16"):
        raise ValueError("manifest dtype must be fp32, bf16 or fp16")
    versions = manifest.get("downstream_versions")
    if not isinstance(versions, dict) or not all(
        isinstance(versions.get(name), str) and versions[name].strip()
        for name in ("verl", "transformers", "tensordict", "ray")
    ) or not all(isinstance(key, str) and isinstance(value, str) and value.strip()
                 for key, value in versions.items()):
        raise ValueError("manifest downstream_versions requires verl, transformers, tensordict and ray")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("manifest protocol must be an object")
    _digest(protocol.get("config_sha256"), "protocol.config_sha256")
    if protocol.get("throughput_scope") not in ("global", "per_rank"):
        raise ValueError("manifest protocol requires explicit throughput_scope")
    if protocol.get("memory_metric") != "allocated":
        raise ValueError("manifest protocol memory_metric must be allocated")
    warmup, timed = protocol.get("warmup_steps"), protocol.get("timed_steps")
    if not isinstance(warmup, list) or not isinstance(timed, list) or len(warmup) < 2 or len(timed) < 3:
        raise ValueError("manifest protocol requires at least 2 warmup and 3 timed steps")
    expected_steps = list(range(1, len(warmup) + len(timed) + 1))
    if (not all(type(step) is int for step in warmup + timed)
            or warmup + timed != expected_steps):
        raise ValueError("manifest warmup then timed steps must be consecutive and disjoint")
    observed = [row["step"] for row in parsed["steps"]]
    completed = manifest.get("completed_steps")
    if (not isinstance(completed, list) or not all(type(step) is int for step in completed)
            or observed != expected_steps or completed != observed):
        raise ValueError("manifest completed_steps must match all observed protocol steps")


def compare(candidate: dict[str, object], oracle: dict[str, object],
            candidate_manifest=None, oracle_manifest=None) -> dict[str, float]:
    if candidate_manifest is None or oracle_manifest is None:
        raise ValueError("performance comparison requires both candidate and oracle manifests")
    validate_manifest(candidate_manifest, candidate, "candidate")
    validate_manifest(oracle_manifest, oracle, "oracle")
    for key in ("backend", "world_size", "model", "weights_sha256", "dtype", "downstream_versions",
                "verl_source_sha256", "input_sha256", "protocol"):
        if candidate_manifest[key] != oracle_manifest[key]:
            raise ValueError(f"candidate and oracle manifest {key} differ")
    for left_step, right_step in zip(candidate["steps"], oracle["steps"]):
        if left_step["metrics"]["perf/total_num_tokens"] != right_step["metrics"]["perf/total_num_tokens"]:
            raise ValueError("candidate and oracle token counts differ")
    timed = set(candidate_manifest["protocol"]["timed_steps"])

    def aggregate(parsed):
        rows = [row["metrics"] for row in parsed["steps"] if row["step"] in timed]
        return {key: (max(row[key] for row in rows) if "max_memory" in key
                      else statistics.median(row[key] for row in rows)) for key in REQUIRED}

    left, right = aggregate(candidate), aggregate(oracle)
    ratios = {
        "step_time_ratio_candidate_over_oracle": left["timing_s/step"] / right["timing_s/step"],
        "throughput_ratio_candidate_over_oracle": left["perf/throughput"] / right["perf/throughput"],
        "actor_peak_memory_ratio_candidate_over_oracle": (
            left["actor/perf/max_memory_allocated_gb"] / right["actor/perf/max_memory_allocated_gb"]
        ),
        "critic_peak_memory_ratio_candidate_over_oracle": (
            left["critic/perf/max_memory_allocated_gb"] / right["critic/perf/max_memory_allocated_gb"]
        ),
    }
    if not all(math.isfinite(value) and value > 0 for value in ratios.values()):
        raise ValueError("derived performance ratios must be finite and positive")
    return ratios



def compare_stage_metrics(candidate, oracle, candidate_manifest, oracle_manifest):
    """Additional reported scopes only; missing stage timings stay unmeasured."""
    compare(candidate, oracle, candidate_manifest, oracle_manifest)
    timed = set(candidate_manifest["protocol"]["timed_steps"])
    left = [row for row in candidate["steps"] if row["step"] in timed]
    right = [row for row in oracle["steps"] if row["step"] in timed]
    stages = {}
    for key in STAGE_TIMINGS:
        missing_left = [row["step"] for row in left if row.get("stage_metrics", {}).get(key) is None]
        missing_right = [row["step"] for row in right if row.get("stage_metrics", {}).get(key) is None]
        a = None if missing_left else statistics.median(row["stage_metrics"][key] for row in left)
        b = None if missing_right else statistics.median(row["stage_metrics"][key] for row in right)
        if any(value is not None and not math.isfinite(value) for value in (a, b)):
            raise ValueError("derived stage median must be finite")
        ratio = a / b if a is not None and b is not None and a > 0 and b > 0 else None
        if ratio is not None and not math.isfinite(ratio):
            raise ValueError("derived stage ratio must be finite")
        stages[key] = dict(candidate_median_seconds=a, oracle_median_seconds=b,
                           ratio_candidate_over_oracle=ratio,
                           missing_candidate_steps=missing_left, missing_oracle_steps=missing_right)
    counts = {}
    for name, rows in (("candidate", left), ("oracle", right)):
        tokens = [int(row["metrics"]["perf/total_num_tokens"]) for row in rows]
        try:
            seconds = math.fsum(row["metrics"]["timing_s/step"] for row in rows)
        except OverflowError as exc:
            raise ValueError("reported timed-step sum overflow") from exc
        throughput = sum(tokens) / seconds
        if not math.isfinite(throughput) or throughput <= 0:
            raise ValueError("derived counted throughput must be finite and positive")
        counts[name] = dict(reported_tokens_by_step=tokens, reported_total_tokens=sum(tokens),
                            timed_step_sum_seconds=seconds,
                            batch_tokens_per_second_from_reported_counts=throughput,
                            per_gpu_tokens_per_second_from_reported_counts=throughput / candidate_manifest["world_size"])
    return dict(status="reported_metrics_comparison_only", stages=stages, token_counts=counts,
                memory_scope="separate actor/critic allocator peaks, not whole-PPO peak",
                unavailable_scopes=["complete_fit_wall", "startup_JIT_attribution",
                                    "whole_PPO_peak_memory", "response_only_token_count"],
                performance_acceptance_proven=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--oracle-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    candidate = parse_log(args.candidate)
    oracle = parse_log(args.oracle)
    if bool(args.candidate_manifest) != bool(args.oracle_manifest):
        parser.error("provide both manifests, or neither for log parsing smoke only")
    report = {
        "candidate": candidate, "oracle": oracle, "status": "log_smoke_only",
        "limitations": ["Manifest identity and provenance are self-reported, not independently attested.",
                        "Finite logs do not prove numerical parity, L4 performance acceptance, or L5 zero fallback."],
    }
    if args.candidate_manifest:
        candidate_manifest = json.loads(args.candidate_manifest.read_text())
        oracle_manifest = json.loads(args.oracle_manifest.read_text())
        report["comparison"] = compare(candidate, oracle, candidate_manifest, oracle_manifest)
        report["stage_comparison"] = compare_stage_metrics(
            candidate, oracle, candidate_manifest, oracle_manifest)
        report["status"] = "reported_metrics_comparison_only"
        report["manifests"] = {"candidate": candidate_manifest, "oracle": oracle_manifest}
        report["aggregation"] = {"latency_and_throughput": "median of timed steps",
                                 "memory": "maximum reported allocated peak across timed steps"}
        report["protocol"] = candidate_manifest["protocol"]
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()

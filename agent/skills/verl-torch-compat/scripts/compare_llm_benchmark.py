"""Report repeated real-model verl PPO benchmark ratios without hiding scope."""

import argparse
import json
import math
import statistics
from pathlib import Path


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def flatten_numbers(value):
    if isinstance(value, list):
        flattened = []
        for item in value:
            flattened.extend(flatten_numbers(item))
        return flattened
    require(isinstance(value, (int, float)) and math.isfinite(value),
            "invalid PPO log-prob trace value")
    return [float(value)]


def cosine_similarity(left, right):
    require(len(left) == len(right) > 0, "sampled tensor shape differs")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        require(left_norm == right_norm == 0 and left == right,
                "sampled tensor has a one-sided zero norm")
        return 1.0
    return dot / (left_norm * right_norm)


def load_many(paths, framework):
    reports = [json.loads(path.read_text()) for path in paths]
    require(reports, "at least one benchmark report is required")
    for report in reports:
        require(report.get("framework") == framework, "framework label mismatch")
        require(report.get("task") == "train" and report.get("objective") == "ppo",
                "benchmark is not a PPO train workload")
        times = report.get("step_times_s")
        require(isinstance(times, list) and len(times) >= 3,
                "each run needs at least three timed steps")
        require(all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0
                    for value in times), "invalid step time")
    return reports


def cuda_runtime_conditions(reports):
    """Require actual readbacks with the same strict CUDA precision policy."""
    anchor = reports[0].get("runtime_conditions")
    for report in reports:
        conditions = report.get("runtime_conditions")
        require(isinstance(conditions, dict), "missing CUDA runtime conditions")
        for name in ("cuda_matmul_allow_tf32", "cudnn_allow_tf32", "cudnn_benchmark"):
            require(conditions.get(name) is False,
                    "CUDA runtime conditions require explicit false: " + name)
        require(conditions.get("float32_matmul_precision") == "highest",
                "CUDA runtime conditions require float32_matmul_precision=highest")
        require(conditions == anchor, "CUDA runtime conditions differ across runs")
    return anchor


def cuda_performance_context(reports):
    """Validate recorded authorization and conservatively aggregate both sides."""
    shared = False
    eligible = True
    for report in reports:
        require(type(report.get("shared_cuda")) is bool and
                type(report.get("performance_eligible")) is bool,
                "CUDA performance context needs explicit boolean policy fields")
        authorization = report.get("runtime_authorization")
        require(isinstance(authorization, dict), "missing CUDA performance context authorization")
        for name in ("VERL_ALLOW_SHARED_CUDA", "VERL_L4_PERFORMANCE_ELIGIBLE"):
            require(authorization.get(name) in ("0", "1"),
                    "CUDA performance context has invalid authorization: " + name)
        expected_shared = authorization["VERL_ALLOW_SHARED_CUDA"] == "1"
        expected_eligible = (authorization["VERL_L4_PERFORMANCE_ELIGIBLE"] == "1"
                             and not expected_shared)
        require(report["shared_cuda"] is expected_shared and
                report["performance_eligible"] is expected_eligible,
                "CUDA performance context contradicts recorded authorization")
        shared |= expected_shared
        eligible &= expected_eligible
    return shared, eligible


def compare(oracle_paths, candidate_paths, target_ratio):
    oracle = load_many(oracle_paths, "torch")
    candidate = load_many(candidate_paths, "jittor")
    require(len(oracle) == len(candidate), "interleaved run counts differ")
    keys = ("task", "objective", "device", "world_size", "model", "parameter_count",
            "dtype", "parameter_dtypes", "initial_old_log_prob_dtype", "attention",
            "learning_rate", "adam_eps",
            "batch_size_per_rank", "seq_len", "warmup_steps",
            "timed_steps", "blocked_optional_packages", "cpu_thread_env")
    anchor = oracle[0]
    for report in oracle + candidate:
        for key in keys:
            require(report.get(key) == anchor.get(key), "benchmark contract differs: " + key)
    runtime_conditions = None
    shared_cuda, performance_eligible = False, True
    if anchor["device"] == "cuda":
        runtime_conditions = cuda_runtime_conditions(oracle + candidate)
        shared_cuda, performance_eligible = cuda_performance_context(oracle + candidate)
    for report in candidate:
        require(report.get("fallback_delta") == 0, "candidate fallback count is not zero")
        require(report.get("parameter_location") in ("cpu", "device"),
                "candidate parameter location is invalid")
        require(report.get("output_location") == report.get("parameter_location"),
                "candidate output left the selected backend")
        require(report.get("transformers_materialized_init") is True,
                "candidate did not use materialized Transformers loading")
        normalized_model = report["model"].lower().replace("_", ".")
        if "qwen3.5" in normalized_model:
            require(report.get("qwen35_functional_fallback") is True,
                    "candidate did not use the Qwen3.5 autograd-safe fallback")
    loss_errors = []
    log_prob_errors = []
    gradient_errors = []
    gradient_cosines = []
    update_cosines = []
    oracle_has_nonzero_update = False
    candidate_has_nonzero_update = False
    for expected, actual in zip(oracle, candidate):
        require(len(expected.get("losses", ())) == len(actual.get("losses", ())) >= 3,
                "timed loss counts differ")
        for left, right in zip(expected["losses"], actual["losses"]):
            require(math.isfinite(left) and math.isfinite(right), "nonfinite benchmark loss")
            loss_errors.append(abs(left - right))
        for key in ("initial_old_log_prob", "cold_log_prob", "timed_log_probs"):
            left = expected.get(key)
            right = actual.get(key)
            require(left is not None and right is not None, "missing PPO log-prob trace: " + key)
            left_flat = flatten_numbers(left)
            right_flat = flatten_numbers(right)
            require(len(left_flat) == len(right_flat) > 0,
                    "PPO log-prob trace shape differs: " + key)
            log_prob_errors.extend(abs(a - b) for a, b in zip(left_flat, right_flat))
        require(not actual.get("cold_missing_gradient_samples"),
                "candidate has missing sampled gradients")
        for field, cosines, minimum in (
            ("cold_gradient_samples", gradient_cosines, 0.999),
            ("cold_update_samples", update_cosines, 0.99),
        ):
            expected_samples = expected.get(field, {})
            actual_samples = actual.get(field, {})
            require(expected_samples and set(expected_samples) == set(actual_samples),
                    field + " keys differ")
            for name in expected_samples:
                left_flat = flatten_numbers(expected_samples[name])
                right_flat = flatten_numbers(actual_samples[name])
                if field == "cold_update_samples":
                    oracle_has_nonzero_update |= any(value != 0 for value in left_flat)
                    candidate_has_nonzero_update |= any(value != 0 for value in right_flat)
                cosine = cosine_similarity(left_flat, right_flat)
                require(cosine >= minimum, field + " cosine too small: " + name)
                cosines.append(cosine)
                if field == "cold_gradient_samples":
                    gradient_errors.extend(
                        abs(a - b) for a, b in zip(left_flat, right_flat)
                    )
    require(max(loss_errors) <= 1e-4, "benchmark losses are not numerically comparable")
    require(max(log_prob_errors) <= 1e-3,
            "benchmark selected log-probs are not numerically comparable")
    require(max(gradient_errors) <= 5e-4,
            "sampled PPO gradients are not numerically comparable")
    require(oracle_has_nonzero_update and candidate_has_nonzero_update,
            "sampled PPO updates are all zero")

    oracle_medians = [statistics.median(report["step_times_s"]) for report in oracle]
    candidate_medians = [statistics.median(report["step_times_s"]) for report in candidate]
    oracle_median = statistics.median(oracle_medians)
    candidate_median = statistics.median(candidate_medians)
    ratio = candidate_median / oracle_median
    oracle_memory = [report.get("peak_memory_bytes") for report in oracle]
    candidate_memory = [report.get("peak_memory_bytes") for report in candidate]
    memory_comparable = all(isinstance(value, int) and value > 0
                            for value in oracle_memory + candidate_memory)
    if shared_cuda:
        status = "observation_only_shared_cuda"
    elif not performance_eligible:
        status = "observation_only_performance_ineligible"
    else:
        status = "meets_target" if ratio <= target_ratio else "misses_target"
    result = {
        "status": status,
        "shared_cuda": shared_cuda,
        "performance_eligible": performance_eligible,
        "target_step_ratio": target_ratio,
        "candidate_over_oracle_step_ratio": ratio,
        "oracle_run_medians_s": oracle_medians,
        "candidate_run_medians_s": candidate_medians,
        "oracle_median_s": oracle_median,
        "candidate_median_s": candidate_median,
        "oracle_tokens_per_s": anchor["batch_size_per_rank"] * (anchor["seq_len"] - 1) / oracle_median,
        "candidate_tokens_per_s": anchor["batch_size_per_rank"] * (anchor["seq_len"] - 1) / candidate_median,
        "candidate_fallback_count": 0,
        "loss_max_abs_error": max(loss_errors),
        "selected_log_prob_max_abs_error": max(log_prob_errors),
        "sampled_gradient_max_abs_error": max(gradient_errors),
        "sampled_gradient_min_cosine": min(gradient_cosines),
        "sampled_update_min_cosine": min(update_cosines),
        "device": anchor["device"],
        "model": anchor["model"],
        "parameter_count": anchor["parameter_count"],
        "run_count_per_framework": len(oracle),
        "timed_steps_per_run": anchor["timed_steps"],
        "memory_scope": ("Linux process VmHWM including model load" if anchor["device"] == "cpu"
                         else "framework accelerator allocator peak"),
        "memory_comparable": memory_comparable,
        "oracle_peak_memory_bytes": oracle_memory,
        "candidate_peak_memory_bytes": candidate_memory,
        "scope": "real-model actor PPO update performance; no critic, rollout server or Ray trainer",
    }
    if runtime_conditions is not None:
        result["runtime_conditions"] = runtime_conditions
    if not performance_eligible:
        result["scope"] = ("real-model actor PPO update observations only; not performance acceptance; "
                           "no critic, rollout server or Ray trainer")
    if memory_comparable:
        result["candidate_over_oracle_peak_memory_ratio"] = (
            statistics.median(candidate_memory) / statistics.median(oracle_memory)
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", action="append", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--target-ratio", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.target_ratio) or args.target_ratio <= 0:
        parser.error("target-ratio must be finite and positive")
    result = compare(args.oracle, args.candidate, args.target_ratio)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

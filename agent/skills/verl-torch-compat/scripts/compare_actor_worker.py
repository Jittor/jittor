"""Fail-closed three-step actor-worker comparison; no Torch import or JIT."""
import argparse
import json
import math
from pathlib import Path
import re

import numpy as np

from worker_buffer_contract import buffer_shapes, canonical_buffer, validate_buffer_manifest, validate_buffer_arrays
from worker_loading_provenance import validate_loading

from compare_training_worker import validate_compatibility

from actor_worker_fixture import (
    ADVANTAGES, INPUT_IDS, MODEL_SPEC, SCOPE, SCRIPT_NAMES, STAGES,
    parameter_shapes, training_config,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_parameter(name):
    require(isinstance(name, str) and name and all(name.split(".")), "invalid parameter name")
    name = ".".join(part for part in name.split(".") if part != "_fsdp_wrapped_module")
    require(name in parameter_shapes(), "unexpected actor parameter: " + name)
    return name


def normalize_arrays(arrays):
    result = {}
    for name, value in arrays.items():
        if name.startswith("initial_buffer/"):
            name = "initial_buffer/" + canonical_buffer(name[len("initial_buffer/"):])
        else:
            matched = re.fullmatch(r"(initial/|step_[0-9]{3}/(?:param_grad|post_weight)/)(.+)", name)
            name = matched.group(1) + canonical_parameter(matched.group(2)) if matched else name
        require(name not in result, "normalized snapshot collision")
        result[name] = np.asarray(value)
    return result


def valid_hashes(mapping, required):
    require(isinstance(mapping, dict) and set(mapping) >= required, "incomplete SHA manifest")
    require(all(isinstance(name, str) and isinstance(value, str) and
                re.fullmatch(r"[0-9a-f]{64}", value) for name, value in mapping.items()), "invalid SHA256")


def validate_manifest(manifest, framework):
    validate_compatibility(manifest.get("compatibility"), framework)
    required = {
        "schema": 2, "artifact_type": "verl_actor_worker", "scope": SCOPE,
        "framework": framework, "status": "passed_requested_scope", "requested_stage": "train",
        "model_device": "npu", "batch_preprocessing_device": "cpu", "requested_backend": "hccl",
        "actual_backend": "hccl", "world_size": 1, "rank": 0,
        "gradient_scope": "all parameters post-clip/pre-SGD",
        "optimizer_class": "worker_gradient_observer.ObservedSGD",
        "worker_class": "verl.workers.engine_workers.TrainingWorker",
        "engine_class": "verl.workers.engine.fsdp.transformer_impl.FSDPEngineWithLMHead",
        "model_spec": MODEL_SPEC, "parameter_count": 3648,
    }
    for key, value in required.items():
        require(manifest.get(key) == value, "wrong or missing actor metadata: " + key)
    validate_buffer_manifest(manifest)
    validate_loading(manifest.get("loading_provenance"), framework)
    names = manifest.get("parameter_names")
    require(isinstance(names, list) and names, "empty parameter manifest")
    canonical = [canonical_parameter(name) for name in names]
    require(len(canonical) == len(set(canonical)), "parameter normalization collision")
    require(set(canonical) == set(parameter_shapes()), "incomplete actor parameter manifest")
    stages = manifest.get("stages", [])
    require([stage.get("name") for stage in stages] == STAGES and
            all(stage.get("status") == "passed" for stage in stages), "incomplete actor stages")
    versions = manifest.get("versions", {})
    require(set(versions) == {"verl", "transformers", "tensordict", "ray", "peft", "accelerate"}
            and all(isinstance(v, str) and v for v in versions.values()), "missing versions")
    require(versions["verl"] == "0.9.1", "unvalidated verl release")
    valid_hashes(manifest.get("checkpoint_sha256"), {"config.json"})
    require(any(name.endswith(".safetensors") for name in manifest["checkpoint_sha256"]), "missing weight SHA")
    valid_hashes(manifest.get("script_sha256"), SCRIPT_NAMES)
    attention = manifest.get("actual_attention")
    require(isinstance(attention, dict) and attention and all(value == "eager" for value in attention.values()),
            "actor did not use the controlled eager attention path")
    config = manifest.get("training_config", {})
    steps = config.get("steps")
    require(type(steps) is int and 3 <= steps <= 999, "acceptance requires at least three steps")
    require(config == training_config(steps), "wrong fixed actor training configuration")
    summaries = manifest.get("training_steps", [])
    require(len(summaries) == steps, "incomplete step summaries")
    for step, summary in enumerate(summaries):
        require(summary.get("step") == step and summary.get("captured_gradient_count") == 14,
                "incomplete actor gradient capture")
        for key in ("loss", "grad_norm", "largest_parameter_change"):
            require(type(summary.get(key)) in (int, float) and math.isfinite(summary[key]), "invalid step metric")
        require(summary["grad_norm"] > 0 and summary["largest_parameter_change"] > 0, "no real update")
    if framework == "jittor":
        require(type(manifest.get("fallback_count")) is int and manifest["fallback_count"] == 0,
                "candidate fallback count must be zero")
    else:
        require(manifest.get("oracle_fallback_status") == "not_instrumented"
                and manifest.get("oracle_fallback_count") is None, "oracle fallback remains unknown")
    return steps


def validate_arrays(manifest, arrays, steps):
    expected = {"bootstrap/all_reduce": (2,), "infer/log_probs": (2, 4),
                "fixture/old_log_probs": (2, 4), "fixture/advantages": (2, 4),
                "fixture/input_ids": (2, 8), "fixture/response_mask": (2, 4)}
    expected.update({"initial_buffer/" + name: shape for name, shape in buffer_shapes().items()})
    for name, shape in parameter_shapes().items():
        expected["initial/" + name] = shape
        for step in range(steps):
            for kind in ("param_grad", "post_weight"):
                expected["step_{:03d}/{}/{}".format(step, kind, name)] = shape
    for step in range(steps):
        prefix = "step_{:03d}/".format(step)
        expected.update({prefix + "loss": (), prefix + "grad_norm": (), prefix + "post_log_probs": (2, 4)})
    require(set(arrays) == set(expected), "missing or unexpected actor snapshot tensors")
    for name, shape in expected.items():
        array = arrays[name]
        dtype = np.dtype("int64") if name == "fixture/input_ids" else np.dtype("float32") if shape else np.dtype("float64")
        require(array.shape == shape and array.dtype == dtype and np.isfinite(array).all(),
                "invalid actor shape/dtype/numerics: " + name)
    require(np.array_equal(arrays["bootstrap/all_reduce"], [1.0, 3.0]), "wrong collective result")
    validate_buffer_arrays(manifest, arrays)
    require(np.array_equal(arrays["fixture/input_ids"], INPUT_IDS), "wrong input fixture")
    require(np.array_equal(arrays["fixture/advantages"], np.asarray(ADVANTAGES, dtype=np.float32)), "wrong advantages")
    require(np.array_equal(arrays["fixture/response_mask"], np.ones((2, 4), dtype=np.float32)), "wrong response mask")
    require(np.array_equal(arrays["fixture/old_log_probs"], arrays["infer/log_probs"]), "old policy anchor differs")
    for step in range(steps):
        prefix = "step_{:03d}/".format(step)
        summary = manifest["training_steps"][step]
        for key in ("loss", "grad_norm"):
            require(float(arrays[prefix + key]) == summary[key], "metric snapshot differs from summary")
        largest, squared_gradient_norm = 0.0, 0.0
        for name in parameter_shapes():
            previous = "initial/" + name if step == 0 else "step_{:03d}/post_weight/{}".format(step - 1, name)
            weight, gradient = arrays[prefix + "post_weight/" + name], arrays[prefix + "param_grad/" + name]
            largest = max(largest, float(np.max(np.abs(weight - arrays[previous]))))
            squared_gradient_norm += float(np.sum(gradient.astype(np.float64) ** 2))
            # Verify that the observed post-clip gradients explain real SGD
            # updates. This is independent of the cross-runtime tolerance.
            require(np.allclose(weight, arrays[previous] - np.float32(0.01) * gradient,
                                atol=2e-7, rtol=2e-6), "SGD update does not match captured gradient: " + name)
        require(largest > 0 and largest == summary["largest_parameter_change"], "parameter update summary mismatch")
        require(0 < squared_gradient_norm <= 1.0001, "all-zero or unclipped captured gradients")
        prior_probs = arrays["infer/log_probs"] if step == 0 else arrays["step_{:03d}/post_log_probs".format(step - 1)]
        require(not np.array_equal(arrays[prefix + "post_log_probs"], prior_probs), "updated weights did not change inference")


def compare(oracle_manifest, oracle_arrays, candidate_manifest, candidate_arrays, atol, rtol):
    require(all(math.isfinite(v) and v >= 0 for v in (atol, rtol)), "tolerances must be finite and nonnegative")
    steps = validate_manifest(oracle_manifest, "torch")
    require(validate_manifest(candidate_manifest, "jittor") == steps, "step count differs")
    for key in ("checkpoint_sha256", "script_sha256", "versions", "training_config", "model_spec", "rope_config"):
        require(oracle_manifest[key] == candidate_manifest[key], "protocol differs: " + key)
    oracle, candidate = normalize_arrays(oracle_arrays), normalize_arrays(candidate_arrays)
    validate_arrays(oracle_manifest, oracle, steps)
    validate_arrays(candidate_manifest, candidate, steps)
    failures, maximum = [], 0.0
    for name in sorted(oracle):
        expected, actual = oracle[name], candidate[name]
        error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
        maximum = max(maximum, error)
        exact = name.startswith(("initial/", "initial_buffer/")) or expected.dtype.kind in "biu"
        matched = np.array_equal(actual, expected) if exact else np.allclose(actual, expected, atol=atol, rtol=rtol)
        if not matched:
            failures.append({"tensor": name, "max_abs_error": error})
    return {"status": "fail" if failures else "pass", "scope": SCOPE, "steps": steps,
            "parameter_tensors": 14, "initial_buffer_tensors": 2, "tensor_count": len(oracle), "max_abs_error": maximum,
            "atol": atol, "rtol": rtol, "failures": failures,
            "candidate_fallback_count": 0, "oracle_fallback_count": None,
            "oracle_fallback_status": "not_instrumented",
            "oracle_caveat": "retain and review native NPU stderr for CPU-fallback warnings",
            "candidate_compatibility": candidate_manifest["compatibility"],
            "oracle_loading_provenance": oracle_manifest["loading_provenance"],
            "candidate_loading_provenance": candidate_manifest["loading_provenance"],
            "oracle_compatibility": oracle_manifest["compatibility"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--rtol", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        oracle_manifest = json.loads(args.oracle.read_text())
        candidate_manifest = json.loads(args.candidate.read_text())
        with np.load(args.oracle.with_suffix(".npz"), allow_pickle=False) as oracle, np.load(
                args.candidate.with_suffix(".npz"), allow_pickle=False) as candidate:
            result = compare(oracle_manifest, dict(oracle), candidate_manifest, dict(candidate), args.atol, args.rtol)
    except (ValueError, KeyError, TypeError, OSError) as error:
        result = {"status": "fail", "contract_error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

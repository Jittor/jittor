"""Fail-closed comparison of real single-rank tiny-Qwen3 critic worker artifacts.

This compares worker schema-3 numerical evidence, including nonpersistent
initial buffers, not a full trainer or speed.
The official NPU oracle may emit CPU-fallback warnings for jagged conversion;
its fallback instrumentation is explicitly unknown and must not be called zero.
"""

import argparse
import json
import math
from pathlib import Path
import re

import numpy as np

from worker_buffer_contract import buffer_shapes, canonical_buffer, validate_buffer_manifest, validate_buffer_arrays
from worker_loading_provenance import validate_loading


MODEL_SPEC = {
    "model_type": "qwen3", "vocab_size": 32, "hidden_size": 16,
    "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
    "num_key_value_heads": 2, "head_dim": 8, "num_labels": 1,
    "attention_bias": False, "tie_word_embeddings": False,
}
STAGES = ["runtime", "import", "distributed_init", "init", "reset", "infer", "train"]
VERSIONS = {"verl", "transformers", "tensordict", "ray", "peft", "accelerate"}
SCRIPT_NAMES = {"probe_training_worker.py", "worker_gradient_observer.py", "worker_buffer_contract.py", "worker_loading_provenance.py"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parameter_shapes():
    """Exact controlled architecture, including the token classifier bias."""
    shapes = {"model.embed_tokens.weight": (32, 16), "model.norm.weight": (16,),
              "score.weight": (1, 16), "score.bias": (1,)}
    prefix = "model.layers.0."
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        shapes[prefix + "self_attn." + projection + ".weight"] = (16, 16)
    for norm in ("q_norm", "k_norm"):
        shapes[prefix + "self_attn." + norm + ".weight"] = (8,)
    shapes[prefix + "mlp.gate_proj.weight"] = (32, 16)
    shapes[prefix + "mlp.up_proj.weight"] = (32, 16)
    shapes[prefix + "mlp.down_proj.weight"] = (16, 32)
    for norm in ("input_layernorm", "post_attention_layernorm"):
        shapes[prefix + norm + ".weight"] = (16,)
    return shapes


def normalize_parameter(name):
    require(isinstance(name, str) and name, "invalid parameter name")
    parts = name.split(".")
    require(all(parts), "empty parameter path component")
    normalized = ".".join(part for part in parts if part != "_fsdp_wrapped_module")
    require(normalized in parameter_shapes(), "unexpected Qwen3 parameter: " + name)
    return normalized


def normalized_names(names):
    require(isinstance(names, list) and names, "empty parameter manifest")
    result = [normalize_parameter(name) for name in names]
    require(len(result) == len(set(result)), "parameter name normalization collision")
    require(set(result) == set(parameter_shapes()), "incomplete controlled Qwen3 parameter manifest")
    return result


def normalize_arrays(arrays):
    normalized = {}
    for name, value in arrays.items():
        if name.startswith("initial_buffer/"):
            key = "initial_buffer/" + canonical_buffer(name[len("initial_buffer/"):])
        elif name.startswith("initial/"):
            key = "initial/" + normalize_parameter(name[len("initial/"):])
        else:
            matched = re.fullmatch(r"(step_[0-9]{3}/(?:param_grad|post_weight)/)(.+)", name)
            key = matched.group(1) + normalize_parameter(matched.group(2)) if matched else name
        require(key not in normalized, "snapshot name normalization collision: " + key)
        normalized[key] = np.asarray(value)
    return normalized


def valid_hashes(mapping, required):
    require(isinstance(mapping, dict) and set(mapping) >= required, "missing hash manifest")
    for name, digest in mapping.items():
        require(isinstance(name, str) and isinstance(digest, str)
                and re.fullmatch(r"[0-9a-f]{64}", digest) is not None, "invalid SHA256")


def validate_compatibility(metadata, framework):
    """Candidate provenance is required, not falsely equated with the oracle."""
    require(isinstance(metadata, dict), "missing compatibility provenance")
    require(metadata.get("harness_private_patches") is False, "unreported private harness patches")
    require(metadata.get("observation_scope") ==
            "verl/TensorDict nested integration adapters only; not exhaustive", "wrong adapter observation scope")
    if framework == "torch":
        require(metadata.get("framework_layer") == "native_torch"
                and metadata.get("status") == "not_applicable"
                and metadata.get("adapters") == {}
                and metadata.get("distribution_version") is None
                and metadata.get("distribution_version_status") == "not_applicable",
                "native Torch must not claim Jittor adapters")
        return
    require(metadata.get("framework_layer") == "jittor.compat.torch"
            and metadata.get("status") == "observed", "candidate compatibility layer not observed")
    version = metadata.get("distribution_version")
    status = metadata.get("distribution_version_status")
    require((status == "installed_metadata" and isinstance(version, str) and bool(version))
            or (status == "metadata_unavailable" and version is None), "invalid adapter distribution version provenance")
    adapters = metadata.get("adapters")
    require(isinstance(adapters, dict) and set(adapters) == {"verl", "tensordict"}, "incomplete observed adapter set")
    for name, targets, versions, scope in (
        ("verl", {"get_tensordict", "get", "chunk_tensordict", "concat_tensordict", "index_select_tensor_dict"},
         ["0.9.1"], "five pinned TensorDict helper type predicates; no mathematical replacement"),
        ("tensordict", {"_validate_value_generic", "_validate_value_devicefree"},
         ["0.10.0"], "0.10.0 accepted Nested leaf registry and two private validators"),
    ):
        item = adapters[name]
        require(isinstance(item, dict) and item.get("loaded") is True, "adapter not loaded: " + name)
        require(item.get("supported_versions") == versions and item.get("scope") == scope, "adapter version/scope differs: " + name)
        valid_hashes({"source": item.get("source_sha256")}, {"source"})
        active = item.get("active_targets")
        require(isinstance(active, dict) and set(active) == targets
                and all(value is True for value in active.values()), "adapter activation incomplete: " + name)
    require(adapters["tensordict"].get("nested_leaf_registered") is True, "Nested leaf registration not observed")
    require(adapters["verl"].get("upstream_source_sha256") ==
            "4dc4c321a51e256232751ddb5cb2a2be129076e7f02f36a685fcad6d0d98f5f1", "unvalidated verl adapter source target")


def validate_manifest(manifest, framework):
    require(manifest.get("schema") == 3, "worker schema 3 with initial buffers required")
    validate_buffer_manifest(manifest)
    validate_loading(manifest.get("loading_provenance"), framework)
    require(manifest.get("framework") == framework, "wrong runtime identity")
    validate_compatibility(manifest.get("compatibility"), framework)
    require(manifest.get("status") == "passed_requested_scope", "worker did not finish successfully")
    require(manifest.get("requested_stage") == "train", "full worker train stage required")
    require(manifest.get("model_device") == "npu", "real NPU worker required")
    require(manifest.get("batch_preprocessing_device") == "cpu", "batch preprocessing differs")
    require(manifest.get("actual_backend") == manifest.get("requested_backend") == "hccl", "actual HCCL required")
    require(manifest.get("world_size") == 1 and manifest.get("rank") == 0, "single-rank scope required")
    require(manifest.get("gradient_scope") == "all parameters post-clip/pre-SGD", "all post-clip gradients required")
    require(manifest.get("optimizer_class") == "worker_gradient_observer.ObservedSGD", "wrong optimizer observer")
    require(manifest.get("worker_class") == "verl.workers.engine_workers.TrainingWorker", "not a real TrainingWorker")
    require(manifest.get("engine_class") == "verl.workers.engine.fsdp.transformer_impl.FSDPEngineWithValueHead", "wrong value engine")
    require(manifest.get("model_spec") == MODEL_SPEC, "model outside controlled tiny Qwen3 scope")
    normalized_names(manifest.get("parameter_names"))
    require(manifest.get("parameter_count") == 3153, "incomplete parameter count")
    stages = manifest.get("stages", [])
    require([stage.get("name") for stage in stages] == STAGES
            and all(stage.get("status") == "passed" for stage in stages), "incomplete execution stages")
    versions = manifest.get("versions", {})
    require(set(versions) == VERSIONS and all(isinstance(v, str) and v for v in versions.values()), "missing dependency versions")
    require(versions["verl"] == "0.9.1", "unvalidated verl release")
    hashes = manifest.get("checkpoint_sha256")
    valid_hashes(hashes, {"config.json"})
    require(any(name.endswith(".safetensors") for name in hashes), "checkpoint has no weight SHA")
    valid_hashes(manifest.get("script_sha256"), SCRIPT_NAMES)
    attention = manifest.get("actual_attention")
    require(isinstance(attention, dict) and attention
            and all(v == "flash_attention_2" for v in attention.values()), "actual attention is unknown or differs")
    config = manifest.get("training_config", {})
    required_config = {
        "optimizer": "SGD", "lr": 0.01, "weight_decay": 0.0, "clip_grad": 1.0,
        "cliprange_value": 0.5, "dtype": "float32", "batch_size": 2,
        "sequence_length": 8, "response_length": 4, "micro_batch_size": 2,
        "engine_seed": 42, "mask_seed": None, "mask_rule": "all_valid",
        "input_ids": [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 5, 9, 10, 11, 12]],
        "returns": [[0.4, 0.3, 0.2, 0.1], [-0.1, 0.1, 0.3, 0.5]],
        "gradient_capture": True,
        "wrap_policy": {"disable": True},
    }
    require(set(config) == set(required_config) | {"steps"}, "incomplete training configuration")
    require(all(config[key] == value for key, value in required_config.items()), "training fixture differs")
    steps = config["steps"]
    require(type(steps) is int and 1 <= steps <= 999, "invalid step count")
    summaries = manifest.get("training_steps", [])
    require(len(summaries) == steps, "incomplete step summaries")
    for step, summary in enumerate(summaries):
        require(summary.get("step") == step and summary.get("captured_gradient_count") == 15, "incomplete step gradients")
        for key in ("loss", "grad_norm", "largest_parameter_change"):
            require(isinstance(summary.get(key), (int, float)) and math.isfinite(summary[key]), "invalid step metric")
        require(summary["grad_norm"] > 0 and summary["largest_parameter_change"] > 0, "no valid update")
    if framework == "jittor":
        require(type(manifest.get("fallback_count")) is int and manifest["fallback_count"] == 0, "candidate fallback not zero")
    else:
        require(manifest.get("oracle_fallback_status") == "not_instrumented"
                and manifest.get("oracle_fallback_count") is None, "oracle fallback must remain explicitly unknown")
    return steps


def validate_arrays(manifest, arrays, steps):
    expected = {"bootstrap/all_reduce": (2,), "infer/values": (2, 4), "post_train/values": (2, 4)}
    expected.update({"initial_buffer/" + name: shape for name, shape in buffer_shapes().items()})
    for name, shape in parameter_shapes().items():
        expected["initial/" + name] = shape
        for step in range(steps):
            for kind in ("param_grad", "post_weight"):
                expected["step_{:03d}/{}/{}".format(step, kind, name)] = shape
    for step in range(steps):
        expected["step_{:03d}/loss".format(step)] = ()
        expected["step_{:03d}/grad_norm".format(step)] = ()
    require(set(arrays) == set(expected), "missing or unexpected snapshot tensors")
    for name, shape in expected.items():
        value = arrays[name]
        require(value.shape == shape and value.dtype.kind == "f" and value.size > 0, "invalid shape/dtype: " + name)
        if shape:
            require(value.dtype == np.dtype("float32"), "non-FP32 worker payload: " + name)
        require(np.isfinite(value).all(), "nonfinite tensor: " + name)
    require(np.array_equal(arrays["bootstrap/all_reduce"], [1.0, 3.0]), "wrong collective result")
    validate_buffer_arrays(manifest, arrays)
    for step in range(steps):
        prefix = "step_{:03d}/".format(step)
        summary = manifest["training_steps"][step]
        for metric in ("loss", "grad_norm"):
            require(float(arrays[prefix + metric]) == summary[metric], "summary differs from snapshot")
        largest = 0.0
        for name in parameter_shapes():
            previous_key = "initial/" + name if step == 0 else "step_{:03d}/post_weight/{}".format(step - 1, name)
            weight = arrays[prefix + "post_weight/" + name]
            gradient = arrays[prefix + "param_grad/" + name]
            delta = np.abs(weight - arrays[previous_key])
            largest = max(largest, float(np.max(delta)))
            # Captured post-clip gradients must explain each real SGD update,
            # independently of the caller's cross-runtime comparison tolerance.
            require(np.allclose(weight, arrays[previous_key] - np.float32(0.01) * gradient,
                                atol=2e-7, rtol=2e-6),
                    "SGD update does not match captured gradient: " + prefix + name)
        require(largest > 0 and largest == summary["largest_parameter_change"], "parameter update summary mismatch")


def compare(oracle_manifest, oracle_arrays, candidate_manifest, candidate_arrays, atol, rtol):
    require(all(math.isfinite(t) and t >= 0 for t in (atol, rtol)), "tolerances must be finite and nonnegative")
    steps = validate_manifest(oracle_manifest, "torch")
    require(validate_manifest(candidate_manifest, "jittor") == steps, "step counts differ")
    for key in ("checkpoint_sha256", "script_sha256", "versions", "training_config", "model_spec", "rope_config"):
        require(oracle_manifest[key] == candidate_manifest[key], "protocol differs: " + key)
    oracle, candidate = normalize_arrays(oracle_arrays), normalize_arrays(candidate_arrays)
    validate_arrays(oracle_manifest, oracle, steps)
    validate_arrays(candidate_manifest, candidate, steps)
    failures = []
    maximum = 0.0
    for name in sorted(oracle):
        expected, actual = oracle[name], candidate[name]
        require(expected.dtype == actual.dtype, "dtype differs: " + name)
        error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
        maximum = max(maximum, error)
        # Shared checkpoint initialization is exact; tolerance cannot hide a
        # weight-loading mismatch before training begins.
        matched = np.array_equal(actual, expected) if name.startswith(("initial/", "initial_buffer/")) else np.allclose(actual, expected, atol=atol, rtol=rtol)
        if not matched:
            failures.append({"tensor": name, "max_abs_error": error})
    return {
        "status": "fail" if failures else "pass", "steps": steps,
        "parameter_tensors": 15, "initial_buffer_tensors": 2, "tensor_count": len(oracle),
        "atol": atol, "rtol": rtol, "max_abs_error": maximum, "failures": failures,
        "scope": "single-rank in-process critic worker; not full verl trainer or performance",
        "candidate_fallback_count": 0, "oracle_fallback_count": None,
        "oracle_fallback_status": "not_instrumented",
        "oracle_caveat": "native NPU jagged conversion may warn of CPU fallback; retain and review stderr",
        "candidate_compatibility": candidate_manifest["compatibility"],
        "oracle_compatibility": oracle_manifest["compatibility"],
        "oracle_loading_provenance": oracle_manifest["loading_provenance"],
        "candidate_loading_provenance": candidate_manifest["loading_provenance"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True, help="worker JSON report")
    parser.add_argument("--candidate", type=Path, required=True, help="worker JSON report")
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

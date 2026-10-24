"""Strict, host-only comparison of tiny real-worker forward diagnostics.

This is NOT worker/trainer acceptance or a performance comparison. The first
different hook output is an observation in hook-completion order, not proof of
the causal faulty operator. Native Torch NPU fallback remains uninstrumented.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

import actor_worker_fixture as actor
from worker_buffer_contract import buffer_shapes, canonical_buffer, validate_buffer_manifest, validate_buffer_arrays
from worker_loading_provenance import validate_loading
from compare_training_worker import (
    MODEL_SPEC, SCRIPT_NAMES, STAGES, VERSIONS, parameter_shapes, require,
    valid_hashes, validate_compatibility,
)

SCHEMA = "verl-worker-forward-trace-v2"
TRACE_SCRIPTS = {"trace_worker_forward.py", "compare_training_worker.py", "actor_worker_fixture.py"}


def controlled_events(role):
    """Complete one-layer Qwen3 hook inventory, including both RoPE outputs."""
    hidden = [2, 8, 16]
    entries = [("Embedding", "model.embed_tokens", [([], hidden)]),
               ("Qwen3RotaryEmbedding", "model.rotary_emb",
                [([["index", 0]], [2, 8, 8]), ([["index", 1]], [2, 8, 8])])]
    layer = "model.layers.0."
    entries.append(("Qwen3RMSNorm", layer + "input_layernorm", [([], hidden)]))
    for projection in ("q", "k", "v"):
        entries.append(("Linear", layer + "self_attn." + projection + "_proj", [([], hidden)]))
        if projection != "v":
            entries.append(("Qwen3RMSNorm", layer + "self_attn." + projection + "_norm", [([], [2, 8, 2, 8])]))
    entries.extend([
        ("Linear", layer + "self_attn.o_proj", [([], hidden)]),
        ("Qwen3Attention", layer + "self_attn", [([["index", 0]], hidden)]),
        ("Qwen3RMSNorm", layer + "post_attention_layernorm", [([], hidden)]),
        ("Linear", layer + "mlp.gate_proj", [([], [2, 8, 32])]),
        ("Linear", layer + "mlp.up_proj", [([], [2, 8, 32])]),
        ("Linear", layer + "mlp.down_proj", [([], hidden)]),
        ("Qwen3MLP", layer + "mlp", [([], hidden)]),
        ("Qwen3DecoderLayer", "model.layers.0", [([], hidden)]),
        ("Qwen3RMSNorm", "model.norm", [([], hidden)]),
        ("Qwen3Model", "model", [([["key", "last_hidden_state"]], hidden)]),
    ])
    output = [2, 8, 1 if role == "critic" else 32]
    entries.extend([
        ("Linear", "score" if role == "critic" else "lm_head", [([], output)]),
        ("Qwen3ForTokenClassification" if role == "critic" else "Qwen3ForCausalLM",
         "", [([["key", "logits"]], output)]),
    ])
    return entries


def normalize_parameter(name, shapes):
    require(isinstance(name, str) and name and all(name.split(".")), "invalid parameter path")
    name = ".".join(part for part in name.split(".") if part != "_fsdp_wrapped_module")
    require(name in shapes, "unexpected parameter: " + name)
    return name


def validate(manifest, arrays, framework):
    require(manifest.get("schema") == SCHEMA, "wrong trace schema")
    validate_buffer_manifest(manifest)
    require(manifest.get("status") == "diagnostic_completed", "trace did not complete")
    require(manifest.get("diagnostic_only") is True and manifest.get("hook_removed") is True,
            "diagnostic scope or cleanup missing")
    role = manifest.get("trace_role")
    require(role in ("critic", "actor"), "unknown worker role")
    require(manifest.get("framework") == framework, "wrong framework identity")
    validate_loading(manifest.get("loading_provenance"), framework)
    require(manifest.get("requested_stage") == "infer", "infer-only diagnostic required")
    require(manifest.get("model_device") == manifest.get("verl_device_name") == "npu", "NPU required")
    require(manifest.get("requested_backend") == manifest.get("actual_backend") == "hccl", "HCCL required")
    require(manifest.get("world_size") == 1 and manifest.get("rank") == 0, "single-rank scope required")
    require(manifest.get("batch_preprocessing_device") == "cpu", "preprocessing differs")
    require(manifest.get("worker_class") == "verl.workers.engine_workers.TrainingWorker", "not real worker")
    engine = "FSDPEngineWithValueHead" if role == "critic" else "FSDPEngineWithLMHead"
    require(manifest.get("engine_class") == "verl.workers.engine.fsdp.transformer_impl." + engine, "wrong engine")
    require(manifest.get("model_spec") == (MODEL_SPEC if role == "critic" else actor.MODEL_SPEC), "wrong model")
    stages = manifest.get("stages", [])
    require([s.get("name") for s in stages] == STAGES[:-1]
            and all(s.get("status") == "passed" for s in stages), "incomplete infer stages")
    versions = manifest.get("versions", {})
    require(set(versions) == VERSIONS and all(isinstance(v, str) and v for v in versions.values())
            and versions["verl"] == "0.9.1", "missing or unsupported versions")
    validate_compatibility(manifest.get("compatibility"), framework)
    if framework == "jittor":
        require(type(manifest.get("fallback_count")) is int and manifest["fallback_count"] == 0, "candidate fallback not zero")
    else:
        require(manifest.get("oracle_fallback_status") == "not_instrumented"
                and manifest.get("oracle_fallback_count") is None, "oracle fallback must remain unknown")
    valid_hashes(manifest.get("checkpoint_sha256"), {"config.json"})
    require(any(k.endswith(".safetensors") for k in manifest["checkpoint_sha256"]), "missing checkpoint weights")
    valid_hashes(manifest.get("script_sha256"), SCRIPT_NAMES if role == "critic" else actor.SCRIPT_NAMES)
    valid_hashes(manifest.get("trace_script_sha256"), TRACE_SCRIPTS)
    config = manifest.get("training_config", {})
    if role == "actor":
        require(config == actor.training_config(3), "actor fixture differs")
    else:
        expected = dict(optimizer="SGD", lr=0.01, weight_decay=0.0, clip_grad=1.0,
                        cliprange_value=0.5, dtype="float32", steps=3, batch_size=2,
                        sequence_length=8, response_length=4, micro_batch_size=2,
                        engine_seed=42, mask_seed=None, mask_rule="all_valid",
                        input_ids=actor.INPUT_IDS, returns=[[.4, .3, .2, .1], [-.1, .1, .3, .5]],
                        gradient_capture=False, wrap_policy={"disable": True})
        require(config == expected, "critic fixture differs")

    shapes = parameter_shapes() if role == "critic" else actor.parameter_shapes()
    names = manifest.get("parameter_names", [])
    normalized = [normalize_parameter(n, shapes) for n in names]
    require(len(normalized) == len(set(normalized)) and set(normalized) == set(shapes), "incomplete or colliding parameters")
    require(manifest.get("parameter_count") == sum(math.prod(s) for s in shapes.values()), "wrong parameter count")
    values = {}
    for key, value in arrays.items():
        if key.startswith("initial_buffer/"):
            key = "initial_buffer/" + canonical_buffer(key[len("initial_buffer/"):])
        elif key.startswith("initial/"):
            key = "initial/" + normalize_parameter(key[8:], shapes)
        require(key not in values, "array normalization collision")
        values[key] = np.asarray(value)
    expected_arrays = {"initial/" + n: s for n, s in shapes.items()}
    expected_arrays.update({"initial_buffer/" + n: s for n, s in buffer_shapes().items()})
    expected_arrays.update({"bootstrap/all_reduce": (2,), "infer/" + ("values" if role == "critic" else "log_probs"): (2, 4)})
    if role == "actor":
        expected_arrays.update({"fixture/" + n: (2, 8) if n == "input_ids" else (2, 4)
                                for n in ("input_ids", "response_mask", "advantages", "old_log_probs")})

    trace = manifest.get("forward_trace", {})
    for key, value in {"hook_api": "torch.nn.Module.register_forward_hook", "active_stage": "infer",
                       "order": "forward-hook completion", "output_mutation": False,
                       "host_snapshots_synchronize": True, "max_calls": 128, "max_leaf_elements": 4096}.items():
        require(trace.get(key) == value, "wrong trace contract: " + key)
    events = trace.get("events", [])
    inventory = controlled_events(role)
    require(len(events) == len(inventory), "incomplete controlled module inventory")
    # v1 candidate named_modules includes the known FSDP wrapper's root `module`.
    # Permit exactly that root iff the final role-model hook is at `module`, and
    # every event is then verified against the complete controlled Qwen3 tree.
    prefix = "module." if framework == "jittor" and events[-1].get("module_path") == "module" else ""
    calls = {}
    for index, (event, (cls, path, leaves)) in enumerate(zip(events, inventory)):
        raw_path = prefix + path if path else prefix[:-1]
        require(event.get("module_path") == raw_path, "unrecognized controlled module path")
        call = calls.get(cls, 0)
        calls[cls] = call + 1
        require(event.get("class_name") == cls and event.get("call_index") == index
                and event.get("class_call_index") == call and event.get("module_instance_index") == call,
                "class/call/instance alignment differs")
        require(len(event.get("leaves", [])) == len(leaves), "missing output leaf")
        for leaf_index, (leaf, (leaf_path, shape)) in enumerate(zip(event["leaves"], leaves)):
            key = "trace/%03d/%s/class_call_%03d/tensor_%03d" % (index, cls, call, leaf_index)
            require(leaf.get("path") == leaf_path and leaf.get("shape") == shape
                    and leaf.get("array_key") == key, "output leaf path/shape/key differs")
            require(leaf.get("device") == "npu:0", "trace output not on local NPU")
            require(leaf.get("numpy_dtype") == "float32" and leaf.get("tensor_dtype") == "torch.float32",
                    "non-FP32 trace output")
            require(leaf.get("requires_grad") is False, "infer output unexpectedly tracks gradients")
            expected_arrays[key] = tuple(shape)
    attention = manifest.get("actual_attention", {})
    implementation = "flash_attention_2" if role == "critic" else "eager"
    require(isinstance(attention, dict) and attention and all(v == implementation for v in attention.values()),
            "actual attention unknown or differs")
    require(set(values) == set(expected_arrays), "missing or unexpected snapshot arrays")
    for key, shape in expected_arrays.items():
        value = values[key]
        dtype = np.dtype("int64" if key == "fixture/input_ids" else "float32")
        require(value.shape == shape and value.dtype == dtype and np.isfinite(value).all(), "bad shape/dtype/nonfinite: " + key)
    require(np.array_equal(values["bootstrap/all_reduce"], [1, 3]), "wrong collective result")
    validate_buffer_arrays(manifest, values)
    if role == "actor":
        for key, expected in (("input_ids", actor.INPUT_IDS), ("advantages", actor.ADVANTAGES),
                              ("response_mask", np.ones((2, 4))), ("old_log_probs", values["infer/log_probs"])):
            require(np.array_equal(values["fixture/" + key], np.asarray(expected, dtype=values["fixture/" + key].dtype)), "actor fixture snapshot differs")
    return values, prefix


def compare(oracle_manifest, oracle_arrays, candidate_manifest, candidate_arrays, atol=1e-6, rtol=1e-6):
    require(all(isinstance(t, (int, float)) and math.isfinite(t) and t >= 0 for t in (atol, rtol)),
            "tolerances must be finite and nonnegative")
    oracle, _ = validate(oracle_manifest, oracle_arrays, "torch")
    candidate, prefix = validate(candidate_manifest, candidate_arrays, "jittor")
    for key in ("trace_role", "checkpoint_sha256", "script_sha256", "trace_script_sha256", "versions", "training_config", "model_spec", "rope_config"):
        require(oracle_manifest[key] == candidate_manifest[key], "protocol differs: " + key)
    for key in oracle:
        if key.startswith(("initial/", "initial_buffer/")):
            require(np.array_equal(oracle[key], candidate[key]), "initial parameters/buffers not exactly equal: " + key)
    comparisons = []
    for event in oracle_manifest["forward_trace"]["events"]:
        for leaf in event["leaves"]:
            key = leaf["array_key"]
            expected, actual = oracle[key].astype(np.float64), candidate[key].astype(np.float64)
            delta = np.abs(actual - expected)
            comparisons.append({"call_index": event["call_index"], "class_name": event["class_name"],
                                "class_call_index": event["class_call_index"], "module_path": event["module_path"],
                                "leaf_path": leaf["path"], "shape": leaf["shape"], "array_key": key,
                                "max_abs_error": float(delta.max()),
                                "different_elements": int(np.count_nonzero(delta > atol + rtol * np.abs(expected)))})
    first = next((item for item in comparisons if item["different_elements"]), None)
    endpoint = "infer/" + ("values" if oracle_manifest["trace_role"] == "critic" else "log_probs")
    endpoint_equal = bool(np.allclose(candidate[endpoint], oracle[endpoint], atol=atol, rtol=rtol))
    return {"status": "diagnostic_difference" if first or not endpoint_equal else "diagnostic_equal",
            "diagnostic_only": True, "role": oracle_manifest["trace_role"], "atol": atol, "rtol": rtol,
            "scope": "forward-hook completion order only; not causal proof, trainer acceptance or performance",
            "initial_parameters_exact": True, "initial_buffers_exact": True, "initial_buffer_tensors": 2,
            "parameter_tensors": len(oracle_manifest["parameter_names"]),
            "candidate_root_wrapper_prefix": prefix, "first_difference": first, "comparisons": comparisons,
            "max_abs_error": max(item["max_abs_error"] for item in comparisons),
            "infer_equal": endpoint_equal,
            "infer_max_abs_error": float(np.abs(candidate[endpoint].astype(np.float64) - oracle[endpoint]).max()),
            "candidate_fallback_count": 0, "oracle_fallback_count": None, "oracle_fallback_status": "not_instrumented",
            "oracle_loading_provenance": oracle_manifest["loading_provenance"],
            "candidate_loading_provenance": candidate_manifest["loading_provenance"],
            "oracle_caveat": "native NPU jagged conversion may warn of CPU fallback; retain and review stderr"}


def load(path):
    path = Path(path)
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
        require(len(arrays.files) == len(set(arrays.files)), "duplicate NPZ member")
        return json.loads(path.read_text()), {key: arrays[key] for key in arrays.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    args = parser.parse_args()
    require(args.output.resolve() not in {args.oracle.resolve(), args.candidate.resolve(),
            args.oracle.with_suffix(".npz").resolve(), args.candidate.with_suffix(".npz").resolve()}, "output would overwrite an input")
    try:
        result = compare(*load(args.oracle), *load(args.candidate), args.atol, args.rtol)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        result = {"status": "invalid_diagnostic", "diagnostic_only": True, "error": str(exc)}
    result.update(oracle=str(args.oracle), candidate=str(args.candidate))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "comparisons"}, allow_nan=False))
    return {"diagnostic_equal": 0, "diagnostic_difference": 1, "invalid_diagnostic": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())

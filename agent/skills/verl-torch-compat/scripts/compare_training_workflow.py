"""Compare every saved step of the synthetic verl training workflow probe."""

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


def validate_schema4(manifest):
    """Derive coverage from model config, not just matching self-reported lists."""
    if manifest.get("schema_version", 1) < 4:
        return
    assert manifest["schema_version"] == 4, "unknown workflow schema"
    spec = importlib.util.spec_from_file_location(
        "workflow_contract", Path(__file__).with_name("probe_training_workflow.py"))
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    config = manifest["model_config"]
    assert manifest["text_only"] is True
    assert manifest["input_gradients"] == ["input_scale", "embeddings"]
    learned = manifest["critic"] == "learned"
    assert manifest["critic_input_gradients"] == (["input_scale", "embeddings"] if learned else [])
    assert manifest["embedding_input_shape"] == [4, 8, config["hidden_size"]]
    expected = {"actor": contract.parameter_spec(config, "actor")}
    if learned:
        expected["critic"] = contract.parameter_spec(config, "critic")
    assert manifest["parameter_contract"] == expected, "wrong independent parameter contract"
    for role, names, count in (("actor", "parameter_names", "parameter_count"),
                                ("critic", "critic_parameter_names", "critic_parameter_count")):
        assert manifest[names] == sorted(expected.get(role, {}).get("shapes", {})), "missing parameter: " + role
        assert manifest[count] == expected.get(role, {}).get("numel", 0), "wrong parameter count"
    assert manifest["dtype"] == "float32" and manifest["attention"] == "eager"
    assert manifest["versions"]["transformers"] == "5.9.0", "unverified model architecture version"
    loading = manifest["loading_info"]
    roles = ["actor"] + (["critic"] if learned else [])
    if manifest["algorithm"] == "rlhf":
        roles += ["actor", "critic"]
    assert isinstance(loading, list)
    if manifest["model"] == "local-pretrained":
        assert [entry["role"] for entry in loading] == roles, "missing loading evidence"
    else:
        assert manifest["model"] == "tiny-qwen3" and config["model_type"] == "qwen3" and not loading
    for entry in loading:
        assert set(entry["info"]) == {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}
        contract.validate_loading_info(entry["info"], config, entry["role"])
    adapter = manifest["qwen35_adapter"]
    required = config["model_type"] == "qwen3_5_text"
    assert adapter["required"] is required
    assert adapter["active"] is (required and manifest["framework"] == "jittor"), "wrong adapter activation"
    expected_bindings = ["torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"] if required else []
    assert [entry["name"] for entry in adapter["bindings"]] == expected_bindings
    for entry in adapter["bindings"]:
        assert entry["module"] and entry["qualname"]
        assert len(entry["source_sha256"]) == 64 and all(c in "0123456789abcdef" for c in entry["source_sha256"])
    if required:
        assert adapter.get("instance_profile") == "hf_torch_gated_delta_instances_v1", "missing Qwen3.5 instance profile"
        instances = adapter.get("instances")
        instance_roles = {"actor": "model"}
        if learned:
            instance_roles["critic"] = "backbone"
        if manifest["algorithm"] == "rlhf":
            instance_roles.update(reference="model", reward="backbone")
        assert isinstance(instances, dict) and set(instances) == set(instance_roles), "wrong Qwen3.5 instance roles"
        module_bindings = {entry["name"]: entry for entry in adapter["bindings"]}
        for role, prefix in instance_roles.items():
            expected_paths = ["{}.layers.{}.linear_attn".format(prefix, index)
                              for index, kind in enumerate(config["layer_types"]) if kind == "linear_attention"]
            evidence = instances[role]
            assert evidence["expected_layers"] == len(expected_paths), "wrong Qwen3.5 instance count"
            assert [layer["path"] for layer in evidence["layers"]] == expected_paths, "incomplete Qwen3.5 instance layers"
            for layer in evidence["layers"]:
                assert layer["class_module"] == "transformers.models.qwen3_5.modeling_qwen3_5"
                assert layer["class_name"] == "Qwen3_5GatedDeltaNet"
                assert [(b["attribute"], b["target"]) for b in layer["bindings"]] == [
                    ("chunk_gated_delta_rule", "torch_chunk_gated_delta_rule"),
                    ("recurrent_gated_delta_rule", "torch_recurrent_gated_delta_rule")], "incomplete instance bindings"
                for binding in layer["bindings"]:
                    assert binding["identity_matches"] is True, "wrong actual instance binding identity"
                    assert binding["functional_adapter"] is adapter["active"], "wrong instance adapter marker"
                    assert all(binding[key] == module_bindings[binding["target"]][key]
                               for key in ("module", "qualname", "source_sha256")), "instance callable source differs"
    else:
        assert "instances" not in adapter and "instance_profile" not in adapter, "unexpected Qwen3.5 instance evidence"
    hashes = manifest["script_sha256"]
    assert set(hashes) == {"probe_training_workflow.py", "compare_training_workflow.py"}
    assert all(isinstance(h, str) and len(h) == 64 and all(c in "0123456789abcdef" for c in h)
               for h in hashes.values()), "invalid script digest"
    start = manifest["start_step"]
    assert manifest["steps"] > start >= 0
    assert manifest["executed_steps"] == list(range(start, manifest["steps"])), "incomplete step coverage"


def validate_schema4_array(manifest, name, value):
    if manifest.get("schema_version", 1) < 4:
        return
    parts = name.split("/", 2)
    if len(parts) < 3:
        return
    group, key = parts[1:]
    expected = None
    if group in ("param_grad", "post_weight", "critic_param_grad", "critic_post_weight"):
        role = "critic" if group.startswith("critic_") else "actor"
        expected = manifest["parameter_contract"][role]["shapes"][key]
    elif group in ("input", "input_grad", "critic_input", "critic_input_grad") and key == "embeddings":
        expected = manifest["embedding_input_shape"]
    if expected is not None:
        assert list(value.shape) == expected, "incomplete tensor shape: " + name
        assert value.dtype == np.float32, "non-FP32 tensor: " + name


def validate_runtime_conditions(oracle, candidate):
    """CUDA artifacts require explicit matched full-FP32 policy evidence.

    CPU/NPU artifacts predate this field and retain their existing contract.
    Matching missing or wrong policy on both sides is not CUDA acceptance.
    """
    if oracle["device"] != "cuda":
        return None
    for manifest in (oracle, candidate):
        conditions = manifest.get("runtime_conditions")
        assert isinstance(conditions, dict), "missing CUDA runtime conditions"
        assert all(conditions.get(name) is False for name in
                   ("cuda_matmul_allow_tf32", "cudnn_allow_tf32", "cudnn_benchmark")), \
            "incorrect or missing CUDA runtime conditions"
        assert conditions.get("float32_matmul_precision") == "highest", \
            "incorrect or missing CUDA runtime conditions"
    assert oracle["runtime_conditions"] == candidate["runtime_conditions"], \
        "fixture differs: CUDA runtime conditions"
    return oracle["runtime_conditions"]


def validate_coverage(manifest, names):
    """Reject missing gradients even when BOTH snapshots lost the same key."""
    if manifest.get("schema_version", 1) < 2:
        return  # Historical actor-only artifacts predate explicit name manifests.
    assert manifest["parameter_names"], "empty actor parameter manifest"
    learned = manifest["critic"] == "learned"
    assert bool(manifest["critic_parameter_names"]) == learned, "critic manifest mismatch"
    for step in manifest.get("executed_steps", range(manifest["steps"])):
        prefix = "step_{:03d}/".format(step)
        required = ["input_ids", "response", "rewards", "response_mask", "values",
                    "advantages", "returns", "old_log_prob", "log_prob", "logits",
                    "loss", "clip_fraction", "ppo_kl", "lower_clip_fraction",
                    "input_scale", "input_grad/input_scale", "post_update_log_prob",
                    "post_update_loss"]
        if manifest.get("schema_version", 1) >= 3:
            required.append("token_level_scores")
            if manifest["algorithm"] == "rlhf":
                required += ["ref_log_prob", "reward_model_scores"]
            elif manifest["algorithm"] == "rlvr":
                required.append("verifier_scores")
        if manifest.get("schema_version", 1) >= 4:
            required += ["input/embeddings", "input_grad/embeddings"]
            if learned:
                required += ["critic_input/embeddings", "critic_input_grad/embeddings"]
        required += [group + "/" + name for group in ("param_grad", "post_weight")
                     for name in manifest["parameter_names"]]
        if learned:
            required += ["value_predictions", "value_loss", "value_clip_fraction",
                         "post_update_values", "post_update_value_loss", "critic_input_grad/input_scale"]
            required += [group + "/" + name for group in ("critic_param_grad", "critic_post_weight")
                         for name in manifest["critic_parameter_names"]]
        missing = [prefix + name for name in required if prefix + name not in names]
        assert not missing, "incomplete training snapshot: " + repr(missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--rtol", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not all(math.isfinite(v) and v >= 0 for v in (args.atol, args.rtol)):
        parser.error("tolerances must be finite and nonnegative")
    oracle_manifest = json.loads(args.oracle.with_suffix(".json").read_text())
    candidate_manifest = json.loads(args.candidate.with_suffix(".json").read_text())
    assert oracle_manifest["framework"] == "torch"
    assert candidate_manifest["framework"] == "jittor"
    assert candidate_manifest["fallback_count"] == 0
    for key in ("device", "algorithm", "steps", "lr", "dtype", "attention", "model_config",
                "versions", "parameter_count", "reward_rule", "grpo_group_ids",
                "blocked_optional_packages"):
        assert oracle_manifest[key] == candidate_manifest[key], "fixture differs: " + key
    runtime_conditions = validate_runtime_conditions(oracle_manifest, candidate_manifest)
    for key, legacy_default in (("critic", "zero"), ("critic_parameter_count", 0)):
        assert oracle_manifest.get(key, legacy_default) == candidate_manifest.get(key, legacy_default), key
    assert oracle_manifest.get("schema_version", 1) == candidate_manifest.get("schema_version", 1)
    for key, default in (("phase", "full"), ("start_step", 0),
                         ("executed_steps", list(range(oracle_manifest["steps"]))),
                         ("checkpoint_after", None)):
        assert oracle_manifest.get(key, default) == candidate_manifest.get(key, default), key
    if oracle_manifest.get("schema_version", 1) >= 2:
        for key in ("parameter_names", "critic_parameter_names", "initial_weight_sha256"):
            assert oracle_manifest[key] == candidate_manifest[key], "fixture differs: " + key
        assert oracle_manifest["initial_weight_sha256"], "missing independent initial weights"
    if oracle_manifest.get("schema_version", 1) >= 3:
        for key in ("reward_source", "reference_policy", "kl_ratio",
                    "reference_parameter_count", "reward_parameter_count",
                    "reference_parameter_names", "reward_parameter_names"):
            assert oracle_manifest[key] == candidate_manifest[key], "fixture differs: " + key
        rlhf = oracle_manifest["algorithm"] == "rlhf"
        assert oracle_manifest["reference_policy"] is rlhf, "reference policy scope mismatch"
        assert bool(oracle_manifest["reference_parameter_names"]) is rlhf
        assert bool(oracle_manifest["reward_parameter_names"]) is rlhf
    for manifest in (oracle_manifest, candidate_manifest):
        validate_schema4(manifest)
    if oracle_manifest.get("schema_version", 1) >= 4:
        for key in ("parameter_contract", "loading_info", "script_sha256", "input_gradients",
                    "critic_input_gradients", "embedding_input_shape", "text_only", "model"):
            assert oracle_manifest[key] == candidate_manifest[key], "fixture differs: " + key
    groups = {}
    failures = []
    with np.load(args.oracle, allow_pickle=False) as oracle, np.load(
            args.candidate, allow_pickle=False) as candidate:
        assert set(oracle.files) == set(candidate.files), "snapshot keys differ"
        validate_coverage(oracle_manifest, set(oracle.files))
        validate_coverage(candidate_manifest, set(candidate.files))
        for name in sorted(oracle.files):
            expected, actual = oracle[name], candidate[name]
            assert expected.shape == actual.shape, "shape differs: " + name
            assert expected.dtype == actual.dtype, "dtype differs: " + name
            assert np.isfinite(expected).all() and np.isfinite(actual).all(), name
            validate_schema4_array(oracle_manifest, name, expected)
            validate_schema4_array(candidate_manifest, name, actual)
            integer = expected.dtype.kind in "biu"
            passed = (np.array_equal(actual, expected) if integer else
                      np.allclose(actual, expected, atol=args.atol, rtol=args.rtol))
            error = float(np.max(np.abs(actual.astype(np.float64) - expected)))
            group = name.split("/", 2)[1]
            stats = groups.setdefault(group, {"count": 0, "max_abs_error": 0.0})
            stats["count"] += 1
            stats["max_abs_error"] = max(stats["max_abs_error"], error)
            if not passed:
                failures.append({"tensor": name, "max_abs_error": error})
        tensor_count = len(oracle.files)
    result = {"status": "fail" if failures else "pass", "tensor_count": tensor_count,
              "algorithm": oracle_manifest["algorithm"], "steps": oracle_manifest["steps"],
              "oracle_device": oracle_manifest["device"],
              "candidate_device": candidate_manifest["device"],
              "atol": args.atol, "rtol": args.rtol,
              "groups": groups, "failures": failures,
              "scope": "synthetic single-process integration; not full verl trainer"}
    if runtime_conditions is not None:
        result["runtime_conditions"] = runtime_conditions
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

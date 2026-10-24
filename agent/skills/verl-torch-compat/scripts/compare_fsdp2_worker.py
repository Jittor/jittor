"""Fail-closed four-artifact comparison for the fixed two-rank worker protocol."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from compare_training_worker import require, valid_hashes, validate_compatibility
from fsdp2_worker_fixture import INPUTS, MODEL_SPEC, SCRIPT_NAMES, STAGES, TARGETS, parameter_shapes, protocol, scope
from worker_buffer_contract import buffer_shapes, rope_config, validate_buffer_arrays
from worker_loading_provenance import validate_loading
from worker_npu_provenance import validate_npu_policy


def snapshot_stages():
    return ["initial"] + ["step_%03d/%s" % (step, kind) for step in range(3)
                          for kind in ("pre_weight", "param_grad", "post_weight")]


def validate_manifest(m, framework):
    role = m.get("role")
    require(role in ("actor", "critic"), "invalid worker role")
    algorithm = m.get("algorithm", "fixed")
    require(algorithm in ("fixed", "ppo", "grpo", "rlhf", "rlvr"), "invalid workflow algorithm")
    require(not (role == "critic" and algorithm in ("grpo", "rlvr")), "invalid critic workflow")
    rank = m.get("rank")
    require(type(rank) is int and rank in (0, 1), "invalid rank")
    require(type(m.get("local_rank")) is int and m["local_rank"] >= 0, "invalid local rank")
    shapes = parameter_shapes(role)
    fields = {
        "schema": 1 if algorithm == "fixed" else 2,
        "artifact_type": "verl_fsdp2_worker", "scope": scope(algorithm),
        "framework": framework, "status": "passed_requested_scope", "requested_stage": "train",
        "world_size": 2, "requested_backend": "hccl", "actual_backend": "hccl",
        "model_device": "npu", "batch_preprocessing_device": "cpu", "async_weight_loading": False,
        "gradient_scope": "all parameters post-clip/pre-SGD",
        "worker_class": "verl.workers.engine_workers.TrainingWorker",
        "engine_class": "verl.workers.engine.fsdp.transformer_impl.FSDPEngineWith" +
                        ("LMHead" if role == "actor" else "ValueHead"),
        "optimizer_class": "worker_gradient_observer.ObservedSGD", "model_spec": MODEL_SPEC,
        "parameter_count": 3136 if role == "actor" else 3153,
        "parameter_names": sorted(shapes), "training_config": protocol(role, algorithm),
    }
    for key, expected in fields.items():
        require(m.get(key) == expected, "wrong or missing worker metadata: " + key)
    require([item.get("name") for item in m.get("stages", [])] == STAGES and
            all(item.get("status") == "passed" for item in m["stages"]), "incomplete worker stages")
    expected_alias_checks = ["initial", "reset_verified"] + ["step_%03d/%s" % (step, kind) for step in range(3)
                                          for kind in ("pre_weight", "observer", "param_grad", "post_weight")]
    require(m.get("alias_checks") == expected_alias_checks, "incomplete actual alias identity checks")
    placements = m.get("local_shard_observations", {})
    require(set(placements) == set(snapshot_stages()), "missing local shard observations")
    for stage, items in placements.items():
        require(set(items) == set(shapes), "incomplete local tensor observations")
        for name, item in items.items():
            count = item.get("local_numel")
            require(type(count) is int and 0 <= count <= int(np.prod(shapes[name])), "invalid local shard extent")
            require(item.get("residency") == ("verified_npu" if count else "empty_not_materialized"),
                    "false residency claim for local slice")
        require(sum(item["local_numel"] for item in items.values()) > 0, "rank owns no observed shard storage")
    attention = m.get("actual_attention", {})
    expected_attention = protocol(role)["expected_loaded_attention"]
    require(attention and all(value == expected_attention for value in attention.values()),
            "wrong actual attention for worker role")
    versions = m.get("versions", {})
    require(set(versions) == {"verl", "transformers", "tensordict", "ray", "peft", "accelerate", "nvtx"}
            and all(isinstance(v, str) and v for v in versions.values()), "missing dependency versions")
    require(versions["verl"] == "0.9.1" and versions["transformers"] == "5.9.0"
            and versions["tensordict"] == "0.10.0" and versions["nvtx"] == "0.2.16",
            "unvalidated pinned dependencies")
    marker_module = "verl.utils.profiler.nvtx_profile"
    require(m.get("profiler_markers") == {
        "implementation": marker_module,
        "functions": {name: marker_module for name in
                      ("mark_annotate", "mark_end_range", "mark_start_range", "marked_timer")},
        "scope": "NVTX markers only; MSTX/NPU profiling not validated",
    }, "missing or unsupported profiler marker implementation")
    valid_hashes(m.get("checkpoint_sha256"), {"config.json"})
    require(any(key.endswith(".safetensors") for key in m["checkpoint_sha256"]), "missing checkpoint weight hash")
    valid_hashes(m.get("script_sha256"), SCRIPT_NAMES)
    validate_loading(m.get("loading_provenance"), framework)
    validate_npu_policy(m.get("npu_initialization"), framework)
    validate_compatibility(m.get("compatibility"), framework)
    require(m.get("rope_config") == rope_config(m.get("rope_config", {})), "invalid RoPE configuration")
    require(m.get("buffer_count") == 2 and set(m.get("initial_buffers", {})) == set(buffer_shapes()),
            "missing RoPE buffer inventory")
    for name, item in m["initial_buffers"].items():
        require(item == {"shape": [4], "dtype": "torch.float32", "device": "npu:%d" % m["local_rank"],
                         "persistent": False, "array_key": "initial_buffer/" + name}, "invalid buffer metadata")
    require(len(m.get("training_steps", [])) == 3, "requires three upstream updates")
    for step, item in enumerate(m["training_steps"]):
        require(item.get("step") == step and item.get("captured_gradient_count") == len(shapes), "missing gradients")
        require(all(type(item.get(k)) in (float, int) and math.isfinite(item[k]) for k in ("loss", "grad_norm"))
                and item["grad_norm"] > 0, "invalid training metrics")
    if framework == "jittor":
        require(type(m.get("fallback_count")) is int and m["fallback_count"] == 0, "candidate fallback occurred")
    else:
        require(m.get("oracle_fallback_count") is None and m.get("oracle_fallback_status") == "not_instrumented",
                "oracle fallback must remain explicitly unknown")
    return role, rank, algorithm


def validate_arrays(m, arrays):
    role, rank, algorithm = m["role"], m["rank"], m.get("algorithm", "fixed")
    shapes = parameter_shapes(role)
    expected = {"bootstrap/all_reduce": (2,), "infer/output": (2, 4), "fixture/anchor": (2, 4),
                "fixture/input_ids": (2, 8), "fixture/targets": (2, 4)}
    expected.update({"initial_buffer/" + name: shape for name, shape in buffer_shapes().items()})
    if algorithm != "fixed":
        expected.update({"fixture/" + name: (2, 4) for name in
                         ("token_level_scores", "rewards", "advantages", "returns")})
        if algorithm == "rlhf":
            expected.update({"fixture/ref_log_prob": (2, 4), "fixture/reward_model_scores": (2, 1)})
        if algorithm == "rlvr":
            expected["fixture/verifier_scores"] = (2, 1)
    for stage in snapshot_stages():
        expected.update({stage + "/" + name: shape for name, shape in shapes.items()})
    for step in range(3):
        expected.update({"step_%03d/%s" % (step, kind): () for kind in ("loss", "grad_norm")})
        expected["step_%03d/output" % step] = (2, 4)
    require(set(arrays) == set(expected), "missing or unexpected full-tensor snapshots")
    for name, shape in expected.items():
        array = np.asarray(arrays[name])
        dtype = np.dtype("int64") if name == "fixture/input_ids" else np.dtype("float32") if shape else np.dtype("float64")
        require(array.shape == shape and array.dtype == dtype and np.isfinite(array).all(), "invalid snapshot: " + name)
    require(np.array_equal(arrays["bootstrap/all_reduce"], [3.0, 6.0]), "wrong real collective result")
    require(np.array_equal(arrays["fixture/input_ids"], INPUTS[rank]), "rank data fixture differs")
    if algorithm == "fixed":
        require(np.array_equal(arrays["fixture/targets"], np.asarray(TARGETS[role][rank], dtype=np.float32)),
                "wrong targets")
    else:
        target_key = "fixture/advantages" if role == "actor" else "fixture/returns"
        require(np.array_equal(arrays["fixture/targets"], arrays[target_key]), "worker target differs from algorithm output")
        require(np.any(arrays["fixture/rewards"] != 0), "workflow rewards are all zero")
        if algorithm == "rlhf":
            expected_rewards = arrays["fixture/token_level_scores"] - np.float32(0.1) * (
                -arrays["fixture/ref_log_prob"]
            )
            require(np.allclose(arrays["fixture/rewards"], expected_rewards, atol=1e-7, rtol=1e-7),
                    "RLHF KL reward contract differs")
        if algorithm == "rlvr":
            require(np.array_equal(arrays["fixture/token_level_scores"][:, -1:],
                                   arrays["fixture/verifier_scores"]), "RLVR verifier score was not applied")
    require(np.array_equal(arrays["fixture/anchor"], arrays["infer/output"]), "old-policy/value anchor differs")
    validate_buffer_arrays(m, arrays)
    for step in range(3):
        prefix = "step_%03d/" % step
        norm_sq, largest = 0.0, 0.0
        for name in shapes:
            previous_key = "initial/" + name if step == 0 else "step_%03d/post_weight/%s" % (step - 1, name)
            pre, grad, post = [arrays[prefix + kind + "/" + name] for kind in ("pre_weight", "param_grad", "post_weight")]
            require(np.array_equal(pre, arrays[previous_key]), "parameter changed outside upstream update")
            require(np.allclose(post, pre - np.float32(0.01) * grad, atol=2e-7, rtol=2e-6),
                    "SGD update is not explained by captured full gradient: " + name)
            norm_sq += float(np.sum(grad.astype(np.float64) ** 2))
            largest = max(largest, float(np.max(np.abs(post - pre))))
        require(largest > 0 and 0 < norm_sq <= 1.0001, "no update, zero gradient, or absent global clipping")
        for kind in ("loss", "grad_norm"):
            require(float(arrays[prefix + kind]) == m["training_steps"][step][kind], "metric summary mismatch")
        prior = arrays["infer/output"] if step == 0 else arrays["step_%03d/output" % (step - 1)]
        require(not np.array_equal(prior, arrays[prefix + "output"]), "weights changed but inference did not")


def compare(oracle_records, candidate_records, atol, rtol):
    require(all(math.isfinite(v) and v >= 0 for v in (atol, rtol)), "invalid tolerances")
    runtimes = {}
    role = algorithm = None
    for framework, records in (("torch", oracle_records), ("jittor", candidate_records)):
        require(len(records) == 2, "exactly two rank artifacts are required per runtime")
        ranks = {}
        for manifest, arrays in records:
            current_role, rank, current_algorithm = validate_manifest(manifest, framework)
            require(role is None or current_role == role, "mixed actor/critic artifacts")
            require(algorithm is None or current_algorithm == algorithm, "mixed workflow algorithms")
            role = current_role
            algorithm = current_algorithm
            require(rank not in ranks, "duplicate rank artifact")
            validate_arrays(manifest, arrays)
            ranks[rank] = (manifest, arrays)
        require(set(ranks) == {0, 1}, "incomplete rank set")
        left, right = ranks[0], ranks[1]
        for key in ("versions", "profiler_markers", "npu_initialization", "checkpoint_sha256", "script_sha256", "training_config", "loading_provenance", "compatibility"):
            require(left[0][key] == right[0][key], "rank protocol differs: " + key)
        for name in left[1]:
            if name.startswith(("initial/", "initial_buffer/")) or any("/" + kind + "/" in name for kind in
                    ("pre_weight", "param_grad", "post_weight")) or name.endswith("/grad_norm"):
                require(np.array_equal(left[1][name], right[1][name]), "ranks disagree on global snapshot: " + name)
        for stage in snapshot_stages():
            for name, shape in parameter_shapes(role).items():
                count = sum(ranks[r][0]["local_shard_observations"][stage][name]["local_numel"] for r in (0, 1))
                size = int(np.prod(shape))
                require(size <= count <= size + 1, "local shards are missing or replicated: " + name)
        runtimes[framework] = ranks
    failures, maximum = [], 0.0
    for rank in (0, 1):
        oracle, candidate = runtimes["torch"][rank], runtimes["jittor"][rank]
        for key in ("versions", "profiler_markers", "checkpoint_sha256", "script_sha256", "training_config", "model_spec", "rope_config"):
            require(oracle[0][key] == candidate[0][key], "runtime protocol differs: " + key)
        for name, expected in oracle[1].items():
            actual = candidate[1][name]
            error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
            maximum = max(maximum, error)
            exact = (name.startswith(("initial/", "initial_buffer/", "fixture/input_ids"))
                     or (algorithm == "fixed" and name == "fixture/targets"))
            if not (np.array_equal(actual, expected) if exact else np.allclose(actual, expected, atol=atol, rtol=rtol)):
                failures.append(dict(rank=rank, tensor=name, max_abs_error=error))
    return dict(status="fail" if failures else "pass", scope=scope(algorithm), role=role,
                algorithm=algorithm, world_size=2, steps=3,
                max_abs_error=maximum, atol=atol, rtol=rtol, failures=failures,
                tensors_per_rank=len(runtimes["torch"][0][1]), candidate_fallback_count=0,
                versions=runtimes["torch"][0][0]["versions"],
                profiler_markers=runtimes["torch"][0][0]["profiler_markers"],
                npu_initialization={framework: ranks[0][0]["npu_initialization"]
                                    for framework, ranks in runtimes.items()},
                oracle_fallback_count=None, oracle_fallback_status="not_instrumented")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, nargs=2, required=True)
    parser.add_argument("--candidate", type=Path, nargs=2, required=True)
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--rtol", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    def load(paths):
        records = []
        for path in paths:
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
                records.append((json.loads(path.read_text()), dict(arrays)))
        return records
    try:
        result = compare(load(args.oracle), load(args.candidate), args.atol, args.rtol)
    except (ValueError, KeyError, TypeError, OSError) as error:
        result = {"status": "fail", "contract_error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

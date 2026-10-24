"""Strict eight-artifact model+empty-SGD-state resume comparison; run on server.

Within-runtime baseline versus fresh-process continuation is EXACT. Explicit
finite tolerances apply only to independent Torch versus Jittor parity.
Neither this protocol nor its synthetic host fixtures establish full resume.
"""
import argparse
import json
import math
from pathlib import Path
import re

import numpy as np

from compare_training_worker import require, valid_hashes, validate_compatibility
from fsdp2_checkpoint_fixture import (SCRIPT_NAMES, SCOPE, alias_checks, checkpoint_files,
                                      protocol, snapshots, stages, steps)
from fsdp2_worker_fixture import INPUTS, MODEL_SPEC, TARGETS, parameter_shapes
from worker_buffer_contract import buffer_shapes, rope_config, validate_buffer_arrays
from worker_loading_provenance import validate_loading
from worker_npu_provenance import validate_npu_policy


def validate_manifest(m, framework, phase):
    role, rank = m.get("role"), m.get("rank")
    require(role in ("actor", "critic"), "invalid role")
    require(type(rank) is int and rank in (0, 1), "invalid rank")
    shapes = parameter_shapes(role)
    require(m.get("worker_output_devices") == {
        key: "cpu" for key in ["infer/output", "checkpoint_point/output"] +
        ["step_%03d/output" % step for step in steps(phase)]}, "wrong worker host-return observations")
    expected = dict(schema=1, artifact_type="verl_fsdp2_checkpoint", scope=SCOPE,
                    framework=framework, phase=phase, status="checkpoint_next_step_completed",
                    requested_stage="checkpoint_next_step", world_size=2,
                    requested_backend="hccl", actual_backend="hccl", local_rank=rank,
                    model_device="npu", batch_preprocessing_device="cpu", async_weight_loading=False,
                    gradient_scope="all parameters post-clip/pre-SGD",
                    worker_class="verl.workers.engine_workers.TrainingWorker",
                    engine_class="verl.workers.engine.fsdp.transformer_impl.FSDPEngineWith" +
                                 ("LMHead" if role == "actor" else "ValueHead"),
                    optimizer_class="worker_gradient_observer.ObservedSGD", model_spec=MODEL_SPEC,
                    checkpoint_manager="verl.utils.checkpoint.fsdp_checkpoint_manager.FSDPCheckpointManager",
                    observed_checkpoint_config=dict(save_contents=["model", "optimizer"], load_contents=["model", "optimizer"],
                                                    async_save=False, strict=True, save_lora_only=False),
                    parameter_names=sorted(shapes), parameter_count=sum(int(np.prod(s)) for s in shapes.values()),
                    training_config=protocol(role), alias_checks=alias_checks(phase),
                    checkpoint_buffer_residency="all verified NPU; reconstructed nonpersistent buffers")
    for key, value in expected.items():
        require(m.get(key) == value, "missing/wrong checkpoint metadata: " + key)
    require([item.get("name") for item in m.get("stages", [])] == stages(phase)
            and all(item.get("status") == "passed" for item in m["stages"]), "incomplete real checkpoint stages")
    process = m.get("process", {})
    require(set(process) == {"pid", "token"} and type(process["pid"]) is int and process["pid"] > 0
            and isinstance(process["token"], str) and re.fullmatch("[0-9a-f]{32}", process["token"]),
            "missing fresh process provenance")
    valid_hashes(m.get("script_sha256"), SCRIPT_NAMES)
    valid_hashes(m.get("checkpoint_sha256"), {"config.json"})
    require(any(name.endswith(".safetensors") for name in m["checkpoint_sha256"]), "missing original model hash")
    valid_hashes(m.get("training_checkpoint_sha256"), checkpoint_files())
    require(not any("extra_state" in name or name.startswith("/") or ".." in Path(name).parts
                    for name in m["training_checkpoint_sha256"]), "unexpected extra/unsafe checkpoint files")
    receipt = m.get("anchor_receipt", {})
    receipt_keys = {"framework", "role", "rank", "checkpoint_sha256", "script_sha256", "training_config"}
    require(set(receipt) == receipt_keys | {"baseline_process", "anchor_sha256"}, "invalid original anchor receipt")
    require(all(receipt[key] == m[key] for key in receipt_keys), "persisted anchor protocol mismatch")
    valid_hashes({"anchor": receipt["anchor_sha256"]}, {"anchor"})
    require(isinstance(receipt["baseline_process"], dict), "missing anchor source process")
    if phase == "baseline":
        require(receipt["baseline_process"] == process, "anchor did not originate in baseline process")
    else:
        require(receipt["baseline_process"].get("token") != process["token"], "resume reused baseline process")
    optimizer = m.get("checkpoint_optimizer", {})
    require(optimizer.get("state_empty") is True and optimizer.get("parameter_identity_verified") is True,
            "optimizer state/identity not verified")
    groups = optimizer.get("groups", [])
    require(groups and all(set(g) == {"lr", "momentum", "weight_decay", "parameter_count"}
                           and g["lr"] == 0.01 and g["momentum"] == g["weight_decay"] == 0
                           and type(g["parameter_count"]) is int and g["parameter_count"] > 0 for g in groups)
            and sum(g["parameter_count"] for g in groups) == len(shapes), "invalid restored SGD groups")
    if phase == "resume":
        require(m.get("resume_lr_sentinel") == dict(before_load=[0.02] * len(groups),
                after_load=[0.01] * len(groups), updates_before_load=0), "optimizer group sentinel not restored")
    else:
        require("resume_lr_sentinel" not in m, "baseline cannot claim resume sentinel")
    placements = m.get("local_shard_observations", {})
    require(set(placements) == set(snapshots(phase)), "incomplete full-shard observations")
    for observed in placements.values():
        require(set(observed) == set(shapes), "missing parameter shard observation")
        for name, item in observed.items():
            count = item.get("local_numel")
            require(type(count) is int and 0 <= count <= int(np.prod(shapes[name])), "invalid local shard count")
            require(item.get("residency") == ("verified_npu" if count else "empty_not_materialized"),
                    "wrong NPU shard residency")
        require(sum(item["local_numel"] for item in observed.values()) > 0, "rank has no NPU storage")
    versions = m.get("versions", {})
    require(set(versions) == {"verl", "transformers", "tensordict", "ray", "peft", "accelerate", "nvtx"}
            and all(isinstance(v, str) and v for v in versions.values()), "missing dependency versions")
    require(all(versions[k] == v for k, v in {"verl": "0.9.1", "transformers": "5.9.0",
                "tensordict": "0.10.0", "nvtx": "0.2.16"}.items()), "unsupported dependency pin")
    marker = "verl.utils.profiler.nvtx_profile"
    require(m.get("profiler_markers") == dict(implementation=marker,
            functions={name: marker for name in ("mark_annotate", "mark_end_range", "mark_start_range", "marked_timer")},
            scope="NVTX markers only; MSTX/NPU profiling not validated"), "wrong marker implementation")
    attention = "eager" if role == "actor" else "flash_attention_2"
    require(m.get("actual_attention") and all(v == attention for v in m["actual_attention"].values()), "wrong attention")
    validate_compatibility(m.get("compatibility"), framework)
    validate_loading(m.get("loading_provenance"), framework)
    validate_npu_policy(m.get("npu_initialization"), framework)
    require(m.get("rope_config") == rope_config(m.get("rope_config", {})), "wrong RoPE configuration")
    require(m.get("buffer_count") == 2 and set(m.get("initial_buffers", {})) == set(buffer_shapes()), "missing buffers")
    for name, item in m["initial_buffers"].items():
        require(item == dict(shape=[4], dtype="torch.float32", device="npu:%d" % rank,
                             persistent=False, array_key="initial_buffer/" + name), "invalid buffer inventory")
    updates = m.get("training_steps", [])
    require([item.get("step") for item in updates] == steps(phase), "wrong number/order of actual updates")
    for item in updates:
        require(item.get("captured_gradient_count") == len(shapes), "missing full gradients")
        require(all(type(item.get(k)) in (int, float) and math.isfinite(item[k]) for k in ("loss", "grad_norm"))
                and item["grad_norm"] > 0, "nonfinite/empty metrics")
    if framework == "jittor":
        require(type(m.get("fallback_count")) is int and m["fallback_count"] == 0, "candidate fallback")
    else:
        require(m.get("oracle_fallback_count") is None and m.get("oracle_fallback_status") == "not_instrumented",
                "oracle fallback is not instrumented")
    return role, rank


def validate_arrays(m, arrays):
    role, rank, phase = m["role"], m["rank"], m["phase"]
    shapes = parameter_shapes(role)
    expected = {"bootstrap/all_reduce": (2,), "infer/output": (2, 4), "fixture/anchor": (2, 4),
                "fixture/input_ids": (2, 8), "fixture/targets": (2, 4), "checkpoint_point/output": (2, 4)}
    for stage in ("initial_buffer", "checkpoint_point/buffer"):
        expected.update({stage + "/" + name: shape for name, shape in buffer_shapes().items()})
    for stage in snapshots(phase):
        expected.update({stage + "/" + name: shape for name, shape in shapes.items()})
    for step in steps(phase):
        expected.update({"step_%03d/%s" % (step, k): () for k in ("loss", "grad_norm")})
        expected["step_%03d/output" % step] = (2, 4)
    require(set(arrays) == set(expected), "missing or unexpected full snapshots")
    for name, shape in expected.items():
        array = np.asarray(arrays[name])
        dtype = np.dtype("int64") if name == "fixture/input_ids" else np.dtype("float32") if shape else np.dtype("float64")
        require(array.shape == shape and array.dtype == dtype and np.isfinite(array).all(), "invalid array " + name)
    require(np.array_equal(arrays["bootstrap/all_reduce"], [3, 6]), "wrong HCCL bootstrap")
    require(np.array_equal(arrays["fixture/input_ids"], INPUTS[rank]), "wrong fixed rank inputs")
    require(np.array_equal(arrays["fixture/targets"], np.asarray(TARGETS[role][rank], dtype=np.float32)), "wrong targets")
    require(np.array_equal(arrays["infer/output"], arrays["fixture/anchor"]), "persisted anchor/fresh initialization mismatch")
    validate_buffer_arrays(m, arrays)
    for name in buffer_shapes():
        require(np.array_equal(arrays["checkpoint_point/buffer/" + name], arrays["initial_buffer/" + name]),
                "nonpersistent buffer changed during restore")
    for item in m["training_steps"]:
        step = item["step"]
        prefix = "step_%03d/" % step
        norm, changed = 0.0, False
        for name in shapes:
            previous = ("checkpoint_point/weight/" if step == 2 else "initial/" if step == 0
                        else "step_%03d/post_weight/" % (step - 1)) + name
            pre, grad, post = [arrays[prefix + kind + "/" + name] for kind in ("pre_weight", "param_grad", "post_weight")]
            require(np.array_equal(pre, arrays[previous]), "weight changed outside actual update")
            require(np.allclose(post, pre - np.float32(0.01) * grad, atol=2e-7, rtol=2e-6), "invalid SGD update")
            norm += float(np.sum(grad.astype(np.float64) ** 2))
            changed = changed or not np.array_equal(pre, post)
        require(changed and 0 < norm <= 1.0001, "no update or absent clipping")
        require(all(float(arrays[prefix + k]) == item[k] for k in ("loss", "grad_norm")), "metric summary mismatch")
        prior = "checkpoint_point/output" if step == 2 else "infer/output" if step == 0 else "step_000/output"
        require(not np.array_equal(arrays[prior], arrays[prefix + "output"]), "update did not affect inference")
    if phase == "baseline":
        for name in shapes:
            require(np.array_equal(arrays["checkpoint_point/weight/" + name], arrays["step_001/post_weight/" + name]),
                    "checkpoint save changed weights")
        require(np.array_equal(arrays["checkpoint_point/output"], arrays["step_001/output"]), "save changed inference")


def compare(oracle_baseline, oracle_resume, candidate_baseline, candidate_resume, atol, rtol):
    require(all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in (atol, rtol)), "invalid tolerances")
    matrix, role = {}, None
    for framework, phase, records in (("torch", "baseline", oracle_baseline), ("torch", "resume", oracle_resume),
                                     ("jittor", "baseline", candidate_baseline), ("jittor", "resume", candidate_resume)):
        require(len(records) == 2, "exactly two ranks per runtime and phase required")
        ranks = {}
        for manifest, arrays in records:
            current_role, rank = validate_manifest(manifest, framework, phase)
            require(role is None or current_role == role, "mixed roles")
            role = current_role
            require(rank not in ranks, "duplicate rank")
            validate_arrays(manifest, arrays)
            ranks[rank] = (manifest, arrays)
        require(set(ranks) == {0, 1}, "missing rank")
        matrix[framework, phase] = ranks
        for key in ("versions", "profiler_markers", "npu_initialization", "checkpoint_sha256", "script_sha256",
                    "training_config", "training_checkpoint_sha256", "loading_provenance", "compatibility"):
            require(ranks[0][0][key] == ranks[1][0][key], "rank metadata differs: " + key)
        for name in ranks[0][1]:
            if name.startswith(("initial/", "initial_buffer/", "checkpoint_point/weight/", "checkpoint_point/buffer/")) or any(
                    "/" + kind + "/" in name for kind in ("pre_weight", "param_grad", "post_weight")) or name.endswith("/grad_norm"):
                require(np.array_equal(ranks[0][1][name], ranks[1][1][name]), "global snapshots differ across ranks: " + name)
        for stage in snapshots(phase):
            for name, shape in parameter_shapes(role).items():
                count = sum(ranks[r][0]["local_shard_observations"][stage][name]["local_numel"] for r in (0, 1))
                require(int(np.prod(shape)) <= count <= int(np.prod(shape)) + 1, "missing or replicated local shards")
    for framework in ("torch", "jittor"):
        for rank in (0, 1):
            before, resumed = matrix[framework, "baseline"][rank], matrix[framework, "resume"][rank]
            for key in ("versions", "profiler_markers", "npu_initialization", "checkpoint_sha256", "script_sha256",
                        "training_config", "training_checkpoint_sha256", "anchor_receipt", "checkpoint_optimizer",
                        "loading_provenance", "compatibility", "actual_attention", "rope_config"):
                require(before[0][key] == resumed[0][key], "resume provenance differs: " + key)
            require(before[0]["process"]["token"] != resumed[0]["process"]["token"], "not a fresh process")
            for name, actual in resumed[1].items():
                require(np.array_equal(actual, before[1][name]), "resume differs from uninterrupted run: " + name)
    require(len({m["process"]["token"] for ranks in matrix.values() for m, _ in ranks.values()}) == 8,
            "eight separate rank-process executions required")
    failures, maximum = [], 0.0
    for phase in ("baseline", "resume"):
        for rank in (0, 1):
            oracle, candidate = matrix["torch", phase][rank], matrix["jittor", phase][rank]
            for key in ("versions", "profiler_markers", "checkpoint_sha256", "script_sha256", "training_config", "rope_config"):
                require(oracle[0][key] == candidate[0][key], "oracle/candidate protocol differs: " + key)
            for name, expected in oracle[1].items():
                actual = candidate[1][name]
                error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
                maximum = max(maximum, error)
                exact = name.startswith(("initial/", "initial_buffer/", "checkpoint_point/buffer/", "fixture/input_ids", "fixture/targets"))
                if not (np.array_equal(actual, expected) if exact else np.allclose(actual, expected, atol=atol, rtol=rtol)):
                    failures.append(dict(phase=phase, rank=rank, tensor=name, max_abs_error=error))
    return dict(status="fail" if failures else "pass", scope=SCOPE, role=role, world_size=2,
                within_runtime_resume="exact", max_abs_error=maximum, atol=atol, rtol=rtol, failures=failures,
                candidate_fallback_count=0, oracle_fallback_count=None, oracle_fallback_status="not_instrumented",
                rng_restore=False, scheduler_restore=False, optimizer_state="empty SGD state only")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("oracle-baseline", "oracle-resume", "candidate-baseline", "candidate-resume"):
        parser.add_argument("--" + name, type=Path, nargs=2, required=True)
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--rtol", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    def read(paths):
        records = []
        for path in paths:
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
                records.append((json.loads(path.read_text()), dict(arrays)))
        return records
    try:
        result = compare(*(read(getattr(args, key)) for key in
                           ("oracle_baseline", "oracle_resume", "candidate_baseline", "candidate_resume")),
                         args.atol, args.rtol)
    except (ValueError, KeyError, TypeError, OSError) as error:
        result = dict(status="fail", contract_error=str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

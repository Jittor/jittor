"""Pure-host synthetic contracts: never represented as executed checkpoint evidence."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from compare_fsdp2_checkpoint import compare
from fsdp2_checkpoint_fixture import (SCRIPT_NAMES, SCOPE, alias_checks, checkpoint_files,
                                      protocol, snapshots, stages)
from test_compare_fsdp2_worker import fixture as worker_fixture
from worker_buffer_contract import buffer_shapes
from probe_fsdp2_checkpoint import CheckpointRecorder, training_files, observe_worker_output


def fixture(framework, rank, role, phase):
    manifest, arrays = worker_fixture(framework, rank, role)
    token = ("1" if framework == "torch" else "2") + str(rank) + "0" * 30
    original = dict(pid=100 + rank, token=token)
    manifest.update(artifact_type="verl_fsdp2_checkpoint", scope=SCOPE, phase=phase,
                    status="checkpoint_next_step_completed", requested_stage="checkpoint_next_step",
                    training_config=protocol(role), script_sha256={name: "c" * 64 for name in SCRIPT_NAMES},
                    process=original.copy() if phase == "baseline" else dict(pid=200 + rank, token=token[:-1] + "f"),
                    training_checkpoint_sha256={name: ("a" if framework == "torch" else "b") * 64
                                                for name in checkpoint_files()},
                    checkpoint_optimizer=dict(state_empty=True, parameter_identity_verified=True,
                        groups=[dict(lr=0.01, momentum=0, weight_decay=0, parameter_count=len(manifest["parameter_names"]))]),
                    checkpoint_buffer_residency="all verified NPU; reconstructed nonpersistent buffers",
                    checkpoint_manager="verl.utils.checkpoint.fsdp_checkpoint_manager.FSDPCheckpointManager",
                    observed_checkpoint_config=dict(save_contents=["model", "optimizer"], load_contents=["model", "optimizer"],
                                                    async_save=False, strict=True, save_lora_only=False),
                    stages=[dict(name=stage, status="passed") for stage in stages(phase)],
                    alias_checks=alias_checks(phase), actual_attention={"<root>": "eager" if role == "actor" else "flash_attention_2"})
    manifest["worker_output_devices"] = {key: "cpu" for key in
        ["infer/output", "checkpoint_point/output"] +
        ["step_%03d/output" % step for step in ([0, 1, 2] if phase == "baseline" else [2])]}
    manifest["anchor_receipt"] = {key: deepcopy(manifest[key]) for key in
        ("framework", "role", "rank", "checkpoint_sha256", "script_sha256", "training_config")}
    manifest["anchor_receipt"].update(baseline_process=original, anchor_sha256="e" * 64)
    for name in manifest["parameter_names"]:
        arrays["checkpoint_point/weight/" + name] = arrays["step_001/post_weight/" + name].copy()
    for name in buffer_shapes():
        arrays["checkpoint_point/buffer/" + name] = arrays["initial_buffer/" + name].copy()
    arrays["checkpoint_point/output"] = arrays["step_001/output"].copy()
    manifest["local_shard_observations"]["checkpoint_point/weight"] = deepcopy(manifest["local_shard_observations"]["initial"])
    if phase == "resume":
        arrays = {key: value for key, value in arrays.items() if not key.startswith(("step_000/", "step_001/"))}
        manifest["training_steps"] = manifest["training_steps"][2:]
        manifest["local_shard_observations"] = {key: value for key, value in manifest["local_shard_observations"].items()
                                               if key in snapshots(phase)}
        manifest["resume_lr_sentinel"] = dict(before_load=[0.02], after_load=[0.01], updates_before_load=0)
    return manifest, arrays


class TestFSDP2CheckpointArtifacts(unittest.TestCase):
    def test_worker_output_observes_cpu_return_without_claiming_device_compute(self):
        record = SimpleNamespace(report={}, tensor=mock.Mock())
        value = SimpleNamespace(device=SimpleNamespace(type="cpu"), shape=(2, 4))
        observe_worker_output(record, "infer/output", value)
        self.assertEqual(record.report["worker_output_devices"], {"infer/output": "cpu"})
        record.tensor.assert_called_once_with("infer/output", value)
        value.device.type = "npu"
        with self.assertRaisesRegex(AssertionError, "CPU"):
            observe_worker_output(record, "unexpected", value)
        self.assertNotIn("unexpected", record.report["worker_output_devices"])

    def test_missing_worker_output_observation_fails(self):
        records = self.records()
        records[3][0][0]["worker_output_devices"].pop("step_002/output")
        with self.assertRaisesRegex(ValueError, "host-return"):
            compare(*records, 1, 1)

    def records(self, role="actor"):
        return [[fixture(framework, rank, role, phase) for rank in (0, 1)]
                for framework, phase in (("torch", "baseline"), ("torch", "resume"),
                                         ("jittor", "baseline"), ("jittor", "resume"))]

    def test_both_roles_complete_eight_artifacts(self):
        for role in ("actor", "critic"):
            result = compare(*self.records(role), 1e-6, 1e-5)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["within_runtime_resume"], "exact")
            self.assertFalse(result["rng_restore"])
            self.assertFalse(result["scheduler_restore"])

    def test_missing_gradient_on_every_side_still_fails(self):
        records = self.records()
        for group in records:
            for _, arrays in group:
                arrays.pop(next(name for name in arrays if "step_002/param_grad/" in name))
        with self.assertRaisesRegex(ValueError, "snapshots"):
            compare(*records, 100, 100)

    def test_missing_or_wrong_contract_fails(self):
        changes = [lambda m: m.update(schema=0), lambda m: m.update(artifact_type="verl_fsdp2_worker"),
                   lambda m: m.update(status="passed_requested_scope"), lambda m: m.update(model_device="cpu"),
                   lambda m: m.update(fallback_count=1), lambda m: m.pop("loading_provenance"),
                   lambda m: m.pop("npu_initialization"), lambda m: m["versions"].pop("nvtx"),
                   lambda m: m["script_sha256"].pop("probe_fsdp2_checkpoint.py"),
                   lambda m: m["training_checkpoint_sha256"].pop(next(iter(checkpoint_files()))),
                   lambda m: m["training_checkpoint_sha256"].update(extra_state_world_size_2_rank_0="a" * 64),
                   lambda m: m["checkpoint_optimizer"].update(state_empty=False),
                   lambda m: m["checkpoint_optimizer"].update(parameter_identity_verified=False),
                   lambda m: m["checkpoint_optimizer"]["groups"][0].update(lr=0.02),
                   lambda m: m["checkpoint_optimizer"]["groups"][0].update(momentum=0.9),
                   lambda m: m["observed_checkpoint_config"]["save_contents"].append("extra"),
                   lambda m: m.pop("checkpoint_manager"),
                   lambda m: m["initial_buffers"].pop(next(iter(buffer_shapes()))),
                   lambda m: m["training_steps"].clear(),
                   lambda m: m["stages"].pop(), lambda m: m["alias_checks"].pop()]
        for change in changes:
            records = self.records()
            change(records[3][0][0])
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError)):
                compare(*records, 100, 100)

    def test_shared_wrong_actual_attention_rejected(self):
        for role, wrong in (("actor", "flash_attention_2"), ("critic", "eager")):
            records = self.records(role)
            for group in records:
                for m, _ in group:
                    m["actual_attention"] = {"<root>": wrong}
            with self.assertRaisesRegex(ValueError, "attention"):
                compare(*records, 100, 100)

    def test_same_process_resume_rejected(self):
        records = self.records()
        records[3][0][0]["process"] = deepcopy(records[2][0][0]["process"])
        with self.assertRaisesRegex(ValueError, "process"):
            compare(*records, 100, 100)

    def test_missing_rank_and_local_device_evidence_rejected(self):
        records = self.records()
        records[3] = [records[3][0], records[3][0]]
        with self.assertRaisesRegex(ValueError, "duplicate rank"):
            compare(*records, 1, 1)
        records = self.records()
        m = records[3][0][0]
        name = m["parameter_names"][0]
        m["local_shard_observations"]["checkpoint_point/weight"][name]["residency"] = "cpu"
        with self.assertRaisesRegex(ValueError, "residency"):
            compare(*records, 1, 1)

    def test_checkpoint_weights_and_update_equation_are_not_optional(self):
        records = self.records()
        name = records[3][0][0]["parameter_names"][0]
        records[3][0][1]["checkpoint_point/weight/" + name] += np.float32(0.1)
        with self.assertRaisesRegex(ValueError, "outside actual update"):
            compare(*records, 100, 100)
        records = self.records()
        records[3][0][1]["step_002/post_weight/" + name] += np.float32(0.1)
        with self.assertRaisesRegex(ValueError, "SGD update"):
            compare(*records, 100, 100)

    def test_checkpoint_or_anchor_receipt_mismatch_rejected(self):
        for key in ("training_checkpoint_sha256", "checkpoint_sha256", "script_sha256"):
            records = self.records()
            metadata = records[3][0][0][key]
            metadata[next(iter(metadata))] = "f" * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                compare(*records, 100, 100)
        records = self.records()
        records[3][0][0]["anchor_receipt"]["anchor_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "provenance"):
            compare(*records, 100, 100)

    def test_resume_lr_sentinel_must_restore_without_prior_update(self):
        for value in (None, dict(before_load=[0.01], after_load=[0.01], updates_before_load=0),
                      dict(before_load=[0.02], after_load=[0.02], updates_before_load=0),
                      dict(before_load=[0.02], after_load=[0.01], updates_before_load=1)):
            records = self.records()
            records[3][0][0]["resume_lr_sentinel"] = value
            with self.assertRaisesRegex(ValueError, "sentinel"):
                compare(*records, 100, 100)

    def test_nonfinite_arrays_and_tolerances_rejected(self):
        for value in (float("nan"), float("inf"), -1):
            with self.assertRaisesRegex(ValueError, "tolerances"):
                compare(*self.records(), value, 1)
        records = self.records()
        records[3][0][1]["step_002/output"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "invalid array"):
            compare(*records, 1, 1)

    def test_resume_must_match_exactly_even_with_permissive_cross_tolerance(self):
        records = self.records()
        records[3][0][1]["step_002/output"][0, 0] += np.float32(1e-6)
        with self.assertRaisesRegex(ValueError, "resume differs"):
            compare(*records, 100, 100)

    def test_parity_difference_is_reported_not_accepted(self):
        records = self.records()
        for phase in (2, 3):
            records[phase][0][1]["step_002/output"][0, 0] += np.float32(0.1)
        result = compare(*records, 1e-6, 1e-6)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["failures"][0]["tensor"], "step_002/output")

    def test_anchor_cannot_be_recomputed_after_load(self):
        records = self.records()
        records[3][0][1]["fixture/anchor"] = records[3][0][1]["checkpoint_point/output"].copy()
        with self.assertRaisesRegex(ValueError, "anchor"):
            compare(*records, 100, 100)

    def test_shared_corrupt_nonpersistent_buffers_fail_formula(self):
        records = self.records()
        for group in records:
            for _, arrays in group:
                for name in list(arrays):
                    if name.startswith(("initial_buffer/", "checkpoint_point/buffer/")):
                        arrays[name] = np.zeros_like(arrays[name])
        with self.assertRaises(ValueError):
            compare(*records, 100, 100)

    def test_real_api_surface_and_reused_reset_contract(self):
        path = Path(__file__).with_name("probe_fsdp2_checkpoint.py")
        tree = ast.parse(path.read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        public = {node.func.attr for node in calls if isinstance(node.func, ast.Attribute)
                  and isinstance(node.func.value, ast.Name) and node.func.value.id == "worker"}
        self.assertTrue({"save_checkpoint", "load_checkpoint", "infer_batch", "train_batch", "set_loss_fn"} <= public)
        self.assertFalse(any(isinstance(node.func, ast.Attribute) and node.func.attr in
                             {"backward", "step", "load_state_dict", "manual_seed"} for node in calls))
        save = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "save_checkpoint")
        self.assertEqual(ast.literal_eval(next(k.value for k in save.keywords if k.arg == "global_step")), 2)
        self.assertIn('args.backend, args.stop_after, args.steps = "hccl", "reset", 3', path.read_text())
        launcher = path.with_name("run_fsdp2_checkpoint.sh").read_text()
        self.assertIn("--phase", launcher)
        self.assertIn("--resume-dir", launcher)
        self.assertIn("signal_owned_group", launcher)
        self.assertIn("--kill-after=10s", launcher)

    def test_recorder_configures_before_worker_init_and_only_observes_results(self):
        # This is an import/config lifecycle double, not a simulated worker or tensor.
        config_module = ModuleType("verl.trainer.config")
        config_module.CheckpointConfig = SimpleNamespace
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict("sys.modules", {
                "verl": ModuleType("verl"), "verl.trainer": ModuleType("verl.trainer"),
                "verl.trainer.config": config_module}):
            args = SimpleNamespace(output=Path(temp) / "host.json", framework="torch", backend="hccl",
                                   stop_after="reset", steps=3, prepare_checkpoint=False, capture_gradients=True)
            recorder = CheckpointRecorder(args)
            imported = object()
            self.assertIs(recorder.stage("import", lambda: imported), imported)
            self.assertEqual(vars(args.checkpoint_config), dict(save_contents=["model", "optimizer"],
                             load_contents=["model", "optimizer"], async_save=False, strict=True, save_lora_only=False))
            worker, model = object(), object()
            self.assertEqual(recorder.stage("init", lambda: (worker, model)), (worker, model))
            self.assertIs(recorder.worker, worker)
            self.assertIs(recorder.model_config, model)

    def test_training_file_manifest_requires_actual_shards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(ValueError):
                training_files(root)
            for name in checkpoint_files():
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_bytes(b"host fixture only, not a checkpoint")
            (root / "fsdp_config.json").write_text(json.dumps(dict(FSDP_version=2, world_size=2)))
            self.assertEqual(set(training_files(root)), checkpoint_files())
            (root / "fsdp_config.json").write_text(json.dumps(dict(FSDP_version=1, world_size=2)))
            with self.assertRaisesRegex(ValueError, "FSDP2"):
                training_files(root)
            (root / "fsdp_config.json").write_text(json.dumps(dict(FSDP_version=2, world_size=2)))
            (root / "extra_state_world_size_2_rank_0.pt").write_bytes(b"forbidden")
            with self.assertRaises(ValueError):
                training_files(root)


if __name__ == "__main__":
    unittest.main()

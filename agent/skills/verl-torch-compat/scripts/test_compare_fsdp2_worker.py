"""Synthetic host contracts, NOT worker execution or NPU evidence."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from compare_fsdp2_worker import compare, snapshot_stages
from fsdp2_worker_fixture import INPUTS, MODEL_SPEC, SCRIPT_NAMES, SCOPE, STAGES, TARGETS, parameter_shapes, protocol, scope
from test_compare_training_worker import compatibility_fixture
from test_worker_buffer_contract import add_buffer_fixture, loading_fixture
from worker_npu_provenance import expected_policy
import worker_npu_provenance as npu_provenance


def fixture(framework, rank, role):
    shapes = parameter_shapes(role)
    manifest = {
        "schema": 1, "artifact_type": "verl_fsdp2_worker", "scope": SCOPE, "role": role,
        "framework": framework, "status": "passed_requested_scope", "requested_stage": "train",
        "world_size": 2, "rank": rank, "local_rank": rank,
        "requested_backend": "hccl", "actual_backend": "hccl", "model_device": "npu",
        "batch_preprocessing_device": "cpu", "async_weight_loading": False,
        "gradient_scope": "all parameters post-clip/pre-SGD",
        "worker_class": "verl.workers.engine_workers.TrainingWorker",
        "engine_class": "verl.workers.engine.fsdp.transformer_impl.FSDPEngineWith" +
                        ("LMHead" if role == "actor" else "ValueHead"),
        "optimizer_class": "worker_gradient_observer.ObservedSGD",
        "model_spec": deepcopy(MODEL_SPEC), "parameter_names": sorted(shapes),
        "parameter_count": sum(int(np.prod(shape)) for shape in shapes.values()),
        "training_config": protocol(role),
        "actual_attention": {"<root>": protocol(role)["expected_loaded_attention"]},
        "versions": {"verl": "0.9.1", "transformers": "5.9.0", "tensordict": "0.10.0",
                     "ray": "2.58.0", "peft": "0.17.0", "accelerate": "1.9.0", "nvtx": "0.2.16"},
        "profiler_markers": {
            "implementation": "verl.utils.profiler.nvtx_profile",
            "functions": {name: "verl.utils.profiler.nvtx_profile" for name in
                          ("mark_annotate", "mark_end_range", "mark_start_range", "marked_timer")},
            "scope": "NVTX markers only; MSTX/NPU profiling not validated"},
        "checkpoint_sha256": {"config.json": "a" * 64, "model.safetensors": "b" * 64},
        "script_sha256": {name: "c" * 64 for name in SCRIPT_NAMES},
        "compatibility": compatibility_fixture(framework), "loading_provenance": loading_fixture(framework),
        "npu_initialization": expected_policy(framework, "d" * 64),
        "stages": [{"name": stage, "status": "passed"} for stage in STAGES],
        "fallback_count": 0, "oracle_fallback_count": None, "oracle_fallback_status": "not_instrumented",
        "alias_checks": ["initial", "reset_verified"] + ["step_%03d/%s" % (step, kind) for step in range(3)
                                       for kind in ("pre_weight", "observer", "param_grad", "post_weight")],
        "training_steps": [], "local_shard_observations": {},
    }
    for stage in snapshot_stages():
        manifest["local_shard_observations"][stage] = {}
        for name, shape in shapes.items():
            size = int(np.prod(shape))
            local = (size + 1) // 2 if rank == 0 else size // 2
            manifest["local_shard_observations"][stage][name] = {
                "local_numel": local, "residency": "verified_npu" if local else "empty_not_materialized"}
    arrays = {"bootstrap/all_reduce": np.asarray([3, 6], dtype=np.float32),
              "fixture/input_ids": np.asarray(INPUTS[rank], dtype=np.int64),
              "fixture/targets": np.asarray(TARGETS[role][rank], dtype=np.float32),
              "infer/output": np.full((2, 4), rank + 0.25, dtype=np.float32),
              "fixture/anchor": np.full((2, 4), rank + 0.25, dtype=np.float32)}
    for name, shape in shapes.items():
        arrays["initial/" + name] = np.zeros(shape, dtype=np.float32)
    for step in range(3):
        prefix = "step_%03d/" % step
        for name, shape in shapes.items():
            previous = "initial/" + name if step == 0 else "step_%03d/post_weight/%s" % (step - 1, name)
            pre = arrays[previous].copy()
            grad = np.full(shape, 0.001, dtype=np.float32)
            arrays[prefix + "pre_weight/" + name] = pre
            arrays[prefix + "param_grad/" + name] = grad
            arrays[prefix + "post_weight/" + name] = pre - np.float32(0.01) * grad
        arrays[prefix + "output"] = np.full((2, 4), rank + 0.25 + (step + 1) * 0.01, dtype=np.float32)
        arrays[prefix + "loss"] = np.asarray(float(rank + step))
        arrays[prefix + "grad_norm"] = np.asarray(0.8)
        manifest["training_steps"].append(dict(step=step, loss=float(rank + step), grad_norm=0.8,
                                              captured_gradient_count=len(shapes)))
    add_buffer_fixture(manifest, arrays)
    for item in manifest["initial_buffers"].values():
        item["device"] = "npu:%s" % rank
    return manifest, arrays


def workflow_fixture(framework, rank, role, algorithm):
    manifest, arrays = fixture(framework, rank, role)
    manifest.update(schema=2, algorithm=algorithm, scope=scope(algorithm),
                    training_config=protocol(role, algorithm))
    token_scores = np.zeros((2, 4), dtype=np.float32)
    token_scores[:, -1] = [0.25, 0.5]
    rewards = token_scores.copy()
    arrays.update({"fixture/token_level_scores": token_scores,
                   "fixture/rewards": rewards,
                   "fixture/advantages": np.full((2, 4), 0.3, dtype=np.float32),
                   "fixture/returns": np.full((2, 4), 0.4, dtype=np.float32)})
    if algorithm == "rlhf":
        ref = np.asarray([[0.01, -0.02, 0.03, -0.04],
                          [-0.02, 0.01, -0.04, 0.03]], dtype=np.float32)
        arrays["fixture/ref_log_prob"] = ref
        arrays["fixture/reward_model_scores"] = token_scores[:, -1:].copy()
        arrays["fixture/rewards"] = token_scores + np.float32(0.1) * ref
    if algorithm == "rlvr":
        arrays["fixture/verifier_scores"] = token_scores[:, -1:].copy()
    arrays["fixture/targets"] = arrays["fixture/advantages" if role == "actor" else "fixture/returns"].copy()
    return manifest, arrays


class TestFSDP2WorkerComparison(unittest.TestCase):
    def records(self, role="actor"):
        return [[fixture(framework, rank, role) for rank in (0, 1)] for framework in ("torch", "jittor")]

    def test_both_roles_accept_complete_controlled_artifacts(self):
        for role, count in (("actor", 13), ("critic", 15)):
            with self.subTest(role=role):
                oracle, candidate = self.records(role)
                result = compare(oracle, candidate, 1e-6, 1e-5)
                self.assertEqual(result["status"], "pass")
                self.assertEqual(len(parameter_shapes(role)), count)
                self.assertEqual(result["versions"]["nvtx"], "0.2.16")
                self.assertEqual(result["profiler_markers"], oracle[0][0]["profiler_markers"])

    def test_all_four_distributed_workflow_contracts_are_explicit(self):
        for algorithm, role in (("ppo", "actor"), ("ppo", "critic"), ("grpo", "actor"),
                                ("rlhf", "actor"), ("rlhf", "critic"), ("rlvr", "actor")):
            with self.subTest(algorithm=algorithm, role=role):
                oracle = [workflow_fixture("torch", rank, role, algorithm) for rank in (0, 1)]
                candidate = [workflow_fixture("jittor", rank, role, algorithm) for rank in (0, 1)]
                result = compare(oracle, candidate, 1e-6, 1e-5)
                self.assertEqual(result["status"], "pass")
                self.assertEqual(result["algorithm"], algorithm)

    def test_workflow_target_must_be_actual_algorithm_output(self):
        oracle = [workflow_fixture("torch", rank, "actor", "rlvr") for rank in (0, 1)]
        candidate = [workflow_fixture("jittor", rank, "actor", "rlvr") for rank in (0, 1)]
        for _, arrays in oracle + candidate:
            arrays["fixture/targets"] += 1
        with self.assertRaisesRegex(ValueError, "algorithm output"):
            compare(oracle, candidate, 100, 100)

    def test_computed_workflow_targets_use_float_tolerance_but_fixed_targets_are_exact(self):
        oracle = [workflow_fixture("torch", rank, "actor", "ppo") for rank in (0, 1)]
        candidate = [workflow_fixture("jittor", rank, "actor", "ppo") for rank in (0, 1)]
        for _, arrays in candidate:
            arrays["fixture/advantages"] += np.float32(1e-7)
            arrays["fixture/targets"] += np.float32(1e-7)
        self.assertEqual(compare(oracle, candidate, 1e-6, 0)["status"], "pass")
        fixed_oracle, fixed_candidate = self.records()
        for _, arrays in fixed_candidate:
            arrays["fixture/targets"] += np.float32(1e-7)
        with self.assertRaisesRegex(ValueError, "wrong targets"):
            compare(fixed_oracle, fixed_candidate, 100, 100)

    def test_nvtx_missing_or_wrong_version_rejected_even_when_both_sides_match(self):
        for change in (lambda m: m["versions"].pop("nvtx"),
                       lambda m: m["versions"].update(nvtx="0.2.15")):
            oracle, candidate = self.records()
            for m, _ in oracle + candidate:
                change(m)
            with self.assertRaises(ValueError):
                compare(oracle, candidate, 1, 1)

    def test_role_specific_actual_attention_is_not_inferred_from_initial_request(self):
        for role, wrong in (("actor", "flash_attention_2"), ("critic", "eager")):
            oracle, candidate = self.records(role)
            for manifest, _ in oracle + candidate:
                manifest["actual_attention"] = {"<root>": wrong}
            with self.assertRaisesRegex(ValueError, "actual attention"):
                compare(oracle, candidate, 100, 100)

    def test_missing_or_falsified_npu_policy_fails_even_with_permissive_tolerance(self):
        changes = [lambda m: m.pop("npu_initialization"),
                   lambda m: m["npu_initialization"].update(requested_native_ops=False),
                   lambda m: m["npu_initialization"].update(adapter_source_sha256=None),
                   lambda m: m["npu_initialization"]["functions"]["Qwen3MLP.forward"].update(source_sha256="0" * 64),
                   lambda m: m["npu_initialization"].update(helper_implementation="fake.stub")]
        for change in changes:
            oracle, candidate = self.records()
            for m, _ in candidate:
                change(m)
            with self.assertRaisesRegex(ValueError, "NPU"):
                compare(oracle, candidate, 100, 100)

    def test_oracle_must_keep_its_own_npu_fused_policy(self):
        oracle, candidate = self.records()
        for m, _ in oracle:
            m["npu_initialization"] = deepcopy(candidate[0][0]["npu_initialization"])
        with self.assertRaisesRegex(ValueError, "NPU"):
            compare(oracle, candidate, 100, 100)

    def test_npu_observer_reads_all_bound_functions_without_invoking_them(self):
        def untouched():
            raise AssertionError("observer must not call model math")
        untouched.__module__ = npu_provenance.HF_MODULE
        module = SimpleNamespace(Qwen3RMSNorm=SimpleNamespace(forward=untouched),
                                 Qwen3MLP=SimpleNamespace(forward=untouched),
                                 apply_rotary_pos_emb=untouched)
        def helper():
            raise AssertionError("observer must not invoke initialization")
        helper.__module__ = npu_provenance.ADAPTER_MODULE
        helper.__qualname__ = "apply_native_npu_fsdp_policy"
        helper._jittor_verl_native_npu_policy = npu_provenance.POLICY
        helper_module = SimpleNamespace(apply_npu_fsdp_patches=helper)
        adapter = SimpleNamespace()
        modules = {npu_provenance.HF_MODULE: module, npu_provenance.HELPER_MODULE: helper_module,
                   npu_provenance.ADAPTER_MODULE: adapter}
        digest = npu_provenance.hashlib.sha256(b"read only source").hexdigest()
        expected_functions = {name: (untouched.__qualname__, digest)
                              for name in npu_provenance.FUNCTIONS["jittor"]}
        hashes = {id(module): npu_provenance.MODULE_HASHES[npu_provenance.HF_MODULE],
                  id(helper_module): npu_provenance.HELPER_HASH, id(adapter): "a" * 64}
        with mock.patch.dict(npu_provenance.sys.modules, modules), mock.patch.dict(
                npu_provenance.os.environ, {npu_provenance.ENVIRONMENT_KEY: "1"}), mock.patch.dict(
                npu_provenance.FUNCTIONS, jittor=expected_functions), mock.patch.object(
                npu_provenance, "_module_hash", side_effect=lambda value: hashes[id(value)]), mock.patch.object(
                npu_provenance.inspect, "getsource", return_value="read only source") as source:
            self.assertEqual(npu_provenance.observe_npu_policy("jittor"),
                             npu_provenance.expected_policy("jittor", "a" * 64))
            self.assertEqual(source.call_count, 3)
            untouched.__wrapped__ = lambda: None
            with self.assertRaisesRegex(ValueError, "wrapped"):
                npu_provenance.observe_npu_policy("jittor")
            del untouched.__wrapped__
            helper._jittor_verl_native_npu_policy = None
            with self.assertRaisesRegex(ValueError, "activation"):
                npu_provenance.observe_npu_policy("jittor")

    def test_missing_wrong_or_partial_marker_metadata_fails(self):
        mutations = [lambda m: m.pop("profiler_markers"),
                     lambda m: m["profiler_markers"].update(implementation="verl.utils.profiler.mstx_profile"),
                     lambda m: m["profiler_markers"]["functions"].pop("marked_timer"),
                     lambda m: m["profiler_markers"]["functions"].update(marked_timer="verl.utils.profiler.performance"),
                     lambda m: m["profiler_markers"].update(scope="NPU profiling supported")]
        for change in mutations:
            oracle, candidate = self.records()
            for m, _ in oracle + candidate:
                change(m)
            with self.assertRaisesRegex(ValueError, "profiler marker"):
                compare(oracle, candidate, 1, 1)

    def test_observer_reads_actual_bindings_and_rejects_unrelated_functions(self):
        path = Path(__file__).with_name("probe_fsdp2_worker.py")
        tree = ast.parse(path.read_text())
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "observe_profiler_markers")
        module_name = "verl.utils.profiler.nvtx_profile"
        names = ("mark_annotate", "mark_end_range", "mark_start_range", "marked_timer")
        functions = {}
        for name in names:
            def marker():
                self.fail("observation must not invoke markers")
            marker.__module__ = module_name
            functions[name] = marker
        owner = SimpleNamespace(__name__=module_name, **functions)
        selected = SimpleNamespace(**functions)
        modules = {module_name: owner, "verl.utils.profiler": selected}
        namespace = {"sys": SimpleNamespace(modules=modules)}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        observe = namespace["observe_profiler_markers"]
        self.assertEqual(observe(), fixture("torch", 0, "actor")[0]["profiler_markers"])
        selected.marked_timer = lambda: None
        selected.marked_timer.__module__ = module_name
        with self.assertRaisesRegex(RuntimeError, "binding"):
            observe()
        selected.marked_timer = functions["marked_timer"]
        functions["marked_timer"].__module__ = "verl.utils.profiler.mstx_profile"
        with self.assertRaisesRegex(RuntimeError, "binding"):
            observe()
        del modules[module_name]
        with self.assertRaisesRegex(RuntimeError, "not loaded"):
            observe()

    def test_missing_duplicate_or_mixed_ranks_fail(self):
        oracle, candidate = self.records()
        for bad in (candidate[:1], [candidate[0], candidate[0]],
                    [candidate[0], fixture("jittor", 1, "critic")]):
            with self.assertRaises(ValueError):
                compare(oracle, bad, 1, 1)

    def test_sgd_check_cannot_be_disabled_by_comparison_tolerance(self):
        oracle, candidate = self.records()
        key = "step_001/param_grad/model.norm.weight"
        for records in (oracle, candidate):
            for _, arrays in records:
                arrays[key] *= 5
        with self.assertRaisesRegex(ValueError, "SGD"):
            compare(oracle, candidate, 100, 100)

    def test_missing_gradient_is_rejected_even_when_missing_everywhere(self):
        oracle, candidate = self.records("critic")
        for records in (oracle, candidate):
            for _, arrays in records:
                del arrays["step_002/param_grad/score.bias"]
        with self.assertRaisesRegex(ValueError, "snapshots"):
            compare(oracle, candidate, 100, 100)

    def test_rank_disagreement_even_if_same_rank_oracle_matches(self):
        oracle, candidate = self.records()
        name = "model.norm.weight"
        for records in (oracle, candidate):
            arrays = records[1][1]
            arrays["initial/" + name] += 0.1
            for step in range(3):
                for kind in ("pre_weight", "post_weight"):
                    arrays["step_%03d/%s/%s" % (step, kind, name)] += 0.1
        with self.assertRaisesRegex(ValueError, "ranks disagree"):
            compare(oracle, candidate, 1, 1)

    def test_replicated_storage_cannot_claim_real_shards(self):
        oracle, candidate = self.records()
        for m, _ in candidate:
            m["local_shard_observations"]["initial"]["model.norm.weight"]["local_numel"] = 16
        with self.assertRaisesRegex(ValueError, "replicated"):
            compare(oracle, candidate, 1e-6, 1e-5)

    def test_empty_local_slice_never_claims_verified_residency(self):
        oracle, candidate = self.records("critic")
        candidate[1][0]["local_shard_observations"]["initial"]["score.bias"]["residency"] = "verified_npu"
        with self.assertRaisesRegex(ValueError, "false residency"):
            compare(oracle, candidate, 1, 1)

    def test_rope_formula_cannot_be_disabled_by_tolerance(self):
        oracle, candidate = self.records()
        for records in (oracle, candidate):
            for _, arrays in records:
                arrays["initial_buffer/model.rotary_emb.inv_freq"][:] = 0
        with self.assertRaisesRegex(ValueError, "RoPE formula"):
            compare(oracle, candidate, 100, 100)

    def test_source_hash_missing_stage_alias_checks_and_fallback_fail(self):
        mutations = [lambda m: m["script_sha256"].pop("probe_fsdp2_worker.py"),
                     lambda m: m["stages"].pop(), lambda m: m["alias_checks"].pop(),
                     lambda m: m.update(fallback_count=1)]
        for change in mutations:
            oracle, candidate = self.records()
            change(candidate[0][0])
            with self.assertRaises(ValueError):
                compare(oracle, candidate, 1, 1)

    def test_rank_inputs_must_really_differ(self):
        oracle, candidate = self.records()
        candidate[1][1]["fixture/input_ids"][:] = candidate[0][1]["fixture/input_ids"]
        with self.assertRaisesRegex(ValueError, "rank data fixture"):
            compare(oracle, candidate, 100, 100)

    def test_initial_weights_are_exact_not_tolerance_based(self):
        oracle, candidate = self.records()
        for _, arrays in candidate:
            for key in arrays:
                if key.startswith("initial/") or "/pre_weight/" in key or "/post_weight/" in key:
                    arrays[key] += 0.01
        result = compare(oracle, candidate, 100, 100)
        self.assertEqual(result["status"], "fail")

    def test_probe_uses_real_worker_loss_and_nonmutating_public_gathers(self):
        path = Path(__file__).with_name("probe_fsdp2_worker.py")
        tree = ast.parse(path.read_text())
        source = path.read_text()
        for expected in ("worker.train_batch(batch(anchor, target))", "partial(ppo_loss if args.role ==",
                         "global_batch_size=4", "value.full_tensor()", "with torch.no_grad()"):
            self.assertIn(expected, source)
        snapshot = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "snapshot")
        calls = [node.func.attr for node in ast.walk(snapshot)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        for mutation in ("assign", "update", "copy_", "to", "cuda", "npu", "backward"):
            self.assertNotIn(mutation, calls)
        self.assertIn("full_tensor", calls)
        self.assertIn("to_local", calls)

    def test_complete_modeling_modules_load_before_extra_site(self):
        path = Path(__file__).with_name("probe_fsdp2_worker.py")
        tree = ast.parse(path.read_text())
        imports = next(node for node in ast.walk(tree)
                       if isinstance(node, ast.FunctionDef) and node.name == "imports")
        preload = [node for node in ast.walk(imports) if isinstance(node, ast.ImportFrom)
                   and node.module == "transformers"]
        self.assertEqual(len(preload), 1)
        self.assertEqual({alias.name for alias in preload[0].names},
                         {"Qwen3ForCausalLM", "Qwen3ForTokenClassification"})
        append = [node for node in ast.walk(imports) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute) and node.func.attr == "append"]
        self.assertEqual(len(append), 1)
        self.assertLess(preload[0].lineno, append[0].lineno)


if __name__ == "__main__":
    unittest.main()

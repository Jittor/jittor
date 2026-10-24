"""Pure-host negative controls; synthetic records are NOT worker evidence."""
from copy import deepcopy
import unittest

import numpy as np

from actor_worker_fixture import (
    ADVANTAGES, INPUT_IDS, MODEL_SPEC, SCOPE, SCRIPT_NAMES, STAGES,
    parameter_shapes, training_config,
)
from compare_actor_worker import compare
from test_compare_training_worker import compatibility_fixture
from test_worker_buffer_contract import add_buffer_fixture, loading_fixture


def fixture(framework, wrapped=False):
    prefix = "_fsdp_wrapped_module." if wrapped else ""
    names = {name: prefix + name for name in parameter_shapes()}
    manifest = {
        "schema": 2, "artifact_type": "verl_actor_worker", "scope": SCOPE,
        "framework": framework, "status": "passed_requested_scope", "requested_stage": "train",
        "model_device": "npu", "batch_preprocessing_device": "cpu", "requested_backend": "hccl",
        "actual_backend": "hccl", "world_size": 1, "rank": 0,
        "gradient_scope": "all parameters post-clip/pre-SGD",
        "optimizer_class": "worker_gradient_observer.ObservedSGD",
        "worker_class": "verl.workers.engine_workers.TrainingWorker",
        "engine_class": "verl.workers.engine.fsdp.transformer_impl.FSDPEngineWithLMHead",
        "model_spec": deepcopy(MODEL_SPEC), "parameter_count": 3648,
        "parameter_names": list(names.values()), "stages": [{"name": name, "status": "passed"} for name in STAGES],
        "versions": {"verl": "0.9.1", "transformers": "5.9.0", "tensordict": "0.10.0",
                     "ray": "2.58.0", "peft": "0.17.0", "accelerate": "1.9.0"},
        "checkpoint_sha256": {"config.json": "a" * 64, "model.safetensors": "b" * 64},
        "script_sha256": {name: "c" * 64 for name in SCRIPT_NAMES},
        "actual_attention": {"<root>": "eager"}, "training_config": training_config(3),
        "oracle_fallback_status": "not_instrumented", "oracle_fallback_count": None,
        "fallback_count": 0, "training_steps": [],
        "compatibility": compatibility_fixture(framework),
        "loading_provenance": loading_fixture(framework),
    }
    arrays = {
        "bootstrap/all_reduce": np.array([1, 3], dtype=np.float32),
        "infer/log_probs": np.full((2, 4), -3, dtype=np.float32),
        "fixture/old_log_probs": np.full((2, 4), -3, dtype=np.float32),
        "fixture/advantages": np.asarray(ADVANTAGES, dtype=np.float32),
        "fixture/input_ids": np.asarray(INPUT_IDS, dtype=np.int64),
        "fixture/response_mask": np.ones((2, 4), dtype=np.float32),
    }
    for name, shape in parameter_shapes().items():
        arrays["initial/" + names[name]] = np.zeros(shape, dtype=np.float32)
    for step in range(3):
        current = "step_{:03d}/".format(step)
        largest = 0.0
        for name, shape in parameter_shapes().items():
            previous = "initial/" + names[name] if step == 0 else "step_{:03d}/post_weight/{}".format(step - 1, names[name])
            gradient = np.full(shape, 0.005, dtype=np.float32)
            values = arrays[previous] - np.float32(0.01) * gradient
            arrays[current + "param_grad/" + names[name]] = gradient
            arrays[current + "post_weight/" + names[name]] = values
            largest = max(largest, float(np.max(np.abs(values - arrays[previous]))))
        arrays[current + "post_log_probs"] = np.full((2, 4), -3 + (step + 1) * 0.01, dtype=np.float32)
        arrays[current + "loss"] = np.asarray(-0.2 - step, dtype=np.float64)
        arrays[current + "grad_norm"] = np.asarray(0.302, dtype=np.float64)
        manifest["training_steps"].append({
            "step": step, "loss": float(arrays[current + "loss"]), "grad_norm": 0.302,
            "largest_parameter_change": largest, "captured_gradient_count": 14,
        })
    add_buffer_fixture(manifest, arrays, prefix)
    return manifest, arrays


class TestActorWorkerComparison(unittest.TestCase):
    def setUp(self):
        self.oracle, self.oracle_arrays = fixture("torch", wrapped=True)
        self.candidate, self.candidate_arrays = fixture("jittor")

    def compare(self, atol=1e-6, rtol=1e-6):
        return compare(self.oracle, self.oracle_arrays, self.candidate, self.candidate_arrays, atol, rtol)

    def test_complete_three_step_fixture_and_wrapper_normalization(self):
        result = self.compare()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["tensor_count"], 115)
        self.assertEqual(result["parameter_tensors"], 14)
        self.assertIsNone(result["oracle_fallback_count"])

    def test_same_missing_gradient_or_parameter_on_both_sides_rejected(self):
        del self.oracle_arrays["step_002/param_grad/_fsdp_wrapped_module.lm_head.weight"]
        del self.candidate_arrays["step_002/param_grad/lm_head.weight"]
        with self.assertRaisesRegex(ValueError, "missing.*snapshot"):
            self.compare()
        self.oracle["parameter_names"].remove("_fsdp_wrapped_module.lm_head.weight")
        self.candidate["parameter_names"].remove("lm_head.weight")
        with self.assertRaisesRegex(ValueError, "incomplete.*manifest"):
            self.compare()

    def test_adapter_provenance_is_not_required_on_native_torch(self):
        result = self.compare()
        self.assertEqual(result["oracle_compatibility"]["adapters"], {})
        self.assertEqual(set(result["candidate_compatibility"]["adapters"]), {"verl", "tensordict"})
        self.candidate["compatibility"]["adapters"]["verl"]["active_targets"]["get"] = False
        with self.assertRaisesRegex(ValueError, "activation incomplete"):
            self.compare()

    def test_each_required_forward_group_is_mandatory(self):
        for key in ("fixture/advantages", "fixture/old_log_probs", "fixture/input_ids",
                    "fixture/response_mask", "infer/log_probs", "step_002/post_log_probs"):
            with self.subTest(key=key):
                saved = self.candidate_arrays.pop(key)
                with self.assertRaisesRegex(ValueError, "snapshot"):
                    self.compare()
                self.candidate_arrays[key] = saved

    def test_manifest_negative_controls(self):
        for key, value in (("framework", "torch"), ("model_device", "cpu"), ("actual_backend", "gloo"),
                           ("fallback_count", 1), ("fallback_count", False), ("world_size", 2),
                           ("status", "failed"), ("requested_stage", "infer"), ("parameter_count", 3647),
                           ("actual_attention", {"root": "flash_attention_2"}), ("stages", []),
                           ("script_sha256", {}), ("model_spec", {}), ("training_steps", []),
                           ("engine_class", "verl.workers.engine.fsdp.transformer_impl.FSDPEngineWithValueHead")):
            with self.subTest(key=key):
                old = self.candidate[key]
                self.candidate[key] = value
                with self.assertRaises(ValueError):
                    self.compare()
                self.candidate[key] = old

    def test_protocol_identity_is_required(self):
        for key, field, value in (("script_sha256", "probe_actor_worker.py", "d" * 64),
                                  ("checkpoint_sha256", "model.safetensors", "d" * 64),
                                  ("versions", "transformers", "5.8.0")):
            with self.subTest(key=key):
                old = self.candidate[key][field]
                self.candidate[key][field] = value
                with self.assertRaisesRegex(ValueError, "protocol"):
                    self.compare()
                self.candidate[key][field] = old

    def test_short_run_and_changed_fixture_rejected(self):
        for field, value in (("steps", 1), ("lr", 0.02), ("use_kl_loss", True)):
            with self.subTest(field=field):
                old = self.candidate["training_config"][field]
                self.candidate["training_config"][field] = value
                with self.assertRaises(ValueError):
                    self.compare()
                self.candidate["training_config"][field] = old

    def test_changed_fixture_array_and_anchor_rejected(self):
        for key in ("fixture/advantages", "fixture/old_log_probs", "fixture/input_ids", "fixture/response_mask"):
            with self.subTest(key=key):
                old = self.candidate_arrays[key]
                self.candidate_arrays[key] = old + 1
                with self.assertRaises(ValueError):
                    self.compare()
                self.candidate_arrays[key] = old

    def test_gradient_must_explain_sgd_update(self):
        self.candidate_arrays["step_001/param_grad/lm_head.weight"] *= 2
        with self.assertRaisesRegex(ValueError, "SGD update"):
            self.compare(atol=100, rtol=100)

    def test_initial_weights_exact_even_when_tolerance_is_large(self):
        # Translate this parameter's complete trajectory, preserving SGD and
        # update summaries, so only checkpoint identity can reject the change.
        for key in self.candidate_arrays:
            if key == "initial/lm_head.weight" or key.endswith("post_weight/lm_head.weight"):
                self.candidate_arrays[key] += np.float32(0.125)
        for step in range(3):
            largest = 0.0
            for name in parameter_shapes():
                previous = "initial/" + name if step == 0 else "step_{:03d}/post_weight/{}".format(step - 1, name)
                updated = self.candidate_arrays["step_{:03d}/post_weight/{}".format(step, name)]
                largest = max(largest, float(np.max(np.abs(updated - self.candidate_arrays[previous]))))
            self.candidate["training_steps"][step]["largest_parameter_change"] = largest
        self.assertEqual(self.compare(atol=100, rtol=100)["status"], "fail")

    def test_unchanged_inference_rejected(self):
        self.candidate_arrays["step_000/post_log_probs"] = self.candidate_arrays["infer/log_probs"].copy()
        with self.assertRaisesRegex(ValueError, "inference"):
            self.compare()

    def test_nonfinite_shape_dtype_and_tolerance_rejected(self):
        key = "step_000/param_grad/lm_head.weight"
        original = self.candidate_arrays[key]
        for bad in (np.full_like(original, np.nan), np.full_like(original, np.inf),
                    original[:0], original.astype(np.float64)):
            self.candidate_arrays[key] = bad
            with self.assertRaises(ValueError):
                self.compare()
        self.candidate_arrays[key] = original
        for bad in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                self.compare(atol=bad)

    def test_oracle_unknown_fallback_remains_explicit(self):
        self.oracle["oracle_fallback_count"] = 0
        with self.assertRaisesRegex(ValueError, "unknown"):
            self.compare()


if __name__ == "__main__":
    unittest.main()

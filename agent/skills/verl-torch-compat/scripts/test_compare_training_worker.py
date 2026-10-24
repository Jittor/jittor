"""Pure-host negative controls for the worker artifact comparator (no torch)."""

from copy import deepcopy
import unittest

import numpy as np

from compare_training_worker import MODEL_SPEC, SCRIPT_NAMES, STAGES, compare, parameter_shapes
from test_worker_buffer_contract import add_buffer_fixture, loading_fixture


def compatibility_fixture(framework):
    """Synthetic provenance for comparator tests, never execution evidence."""
    metadata = {
        "harness_private_patches": False,
        "framework_layer": "native_torch" if framework == "torch" else "jittor.compat.torch",
        "status": "not_applicable" if framework == "torch" else "observed",
        "observation_scope": "verl/TensorDict nested integration adapters only; not exhaustive",
        "distribution_version": None, "distribution_version_status": "not_applicable",
        "adapters": {},
    }
    if framework == "jittor":
        metadata.update(distribution_version="1.3.11.0", distribution_version_status="installed_metadata")
        metadata["adapters"] = {
            "verl": {
                "loaded": True, "source_sha256": "d" * 64, "supported_versions": ["0.9.1"],
                "scope": "five pinned TensorDict helper type predicates; no mathematical replacement",
                "upstream_source_sha256": "4dc4c321a51e256232751ddb5cb2a2be129076e7f02f36a685fcad6d0d98f5f1",
                "active_targets": {name: True for name in ("get_tensordict", "get", "chunk_tensordict", "concat_tensordict", "index_select_tensor_dict")},
            },
            "tensordict": {
                "loaded": True, "source_sha256": "e" * 64, "supported_versions": ["0.10.0"],
                "scope": "0.10.0 accepted Nested leaf registry and two private validators",
                "nested_leaf_registered": True,
                "active_targets": {name: True for name in ("_validate_value_generic", "_validate_value_devicefree")},
            },
        }
    return metadata


def fixture(framework, wrapped=False):
    """Synthetic comparator data only; deliberately not execution evidence."""
    shapes = parameter_shapes()
    prefix = "_fsdp_wrapped_module." if wrapped else ""
    names = {name: prefix + name for name in shapes}
    config = {
        "optimizer": "SGD", "lr": 0.01, "weight_decay": 0.0, "clip_grad": 1.0,
        "cliprange_value": 0.5, "dtype": "float32", "steps": 3, "batch_size": 2,
        "sequence_length": 8, "response_length": 4, "micro_batch_size": 2,
        "engine_seed": 42, "mask_seed": None, "mask_rule": "all_valid",
        "input_ids": [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 5, 9, 10, 11, 12]],
        "returns": [[0.4, 0.3, 0.2, 0.1], [-0.1, 0.1, 0.3, 0.5]],
        "gradient_capture": True,
        "wrap_policy": {"disable": True},
    }
    manifest = {
        "schema": 3, "framework": framework, "status": "passed_requested_scope",
        "requested_stage": "train", "model_device": "npu", "batch_preprocessing_device": "cpu",
        "requested_backend": "hccl", "actual_backend": "hccl", "world_size": 1, "rank": 0,
        "gradient_scope": "all parameters post-clip/pre-SGD",
        "optimizer_class": "worker_gradient_observer.ObservedSGD",
        "worker_class": "verl.workers.engine_workers.TrainingWorker",
        "engine_class": "verl.workers.engine.fsdp.transformer_impl.FSDPEngineWithValueHead",
        "model_spec": deepcopy(MODEL_SPEC), "parameter_names": list(names.values()),
        "parameter_count": 3153, "stages": [{"name": name, "status": "passed"} for name in STAGES],
        "versions": {"verl": "0.9.1", "transformers": "5.9.0", "tensordict": "0.10.0",
                     "ray": "2.58.0", "peft": "0.17.0", "accelerate": "1.9.0"},
        "checkpoint_sha256": {"config.json": "a" * 64, "model.safetensors": "b" * 64},
        "script_sha256": {name: "c" * 64 for name in SCRIPT_NAMES},
        "actual_attention": {"<root>": "flash_attention_2"}, "training_config": config,
        "oracle_fallback_status": "not_instrumented", "oracle_fallback_count": None,
        "fallback_count": 0, "training_steps": [],
        "compatibility": compatibility_fixture(framework),
        "loading_provenance": loading_fixture(framework),
    }
    arrays = {"bootstrap/all_reduce": np.array([1, 3], dtype=np.float32),
              "infer/values": np.zeros((2, 4), dtype=np.float32),
              "post_train/values": np.ones((2, 4), dtype=np.float32)}
    for name, shape in shapes.items():
        arrays["initial/" + names[name]] = np.zeros(shape, dtype=np.float32)
    for step in range(config["steps"]):
        step_prefix = "step_{:03d}/".format(step)
        largest = 0.0
        for name, shape in shapes.items():
            arrays[step_prefix + "param_grad/" + names[name]] = np.ones(shape, dtype=np.float32)
            previous_key = "initial/" + names[name] if step == 0 else "step_{:03d}/post_weight/{}".format(step - 1, names[name])
            previous = arrays[previous_key]
            values = previous - np.float32(config["lr"]) * arrays[step_prefix + "param_grad/" + names[name]]
            arrays[step_prefix + "post_weight/" + names[name]] = values
            largest = max(largest, float(np.max(np.abs(values - previous))))
        arrays[step_prefix + "loss"] = np.asarray(-0.3 - step, dtype=np.float64)
        arrays[step_prefix + "grad_norm"] = np.asarray(0.5 + step, dtype=np.float64)
        manifest["training_steps"].append({
            "step": step, "loss": float(arrays[step_prefix + "loss"]),
            "grad_norm": float(arrays[step_prefix + "grad_norm"]),
            "largest_parameter_change": largest, "captured_gradient_count": 15,
        })
    add_buffer_fixture(manifest, arrays, prefix)
    return manifest, arrays


class TestWorkerComparison(unittest.TestCase):
    def setUp(self):
        self.oracle, self.oracle_arrays = fixture("torch", wrapped=True)
        self.candidate, self.candidate_arrays = fixture("jittor")

    def compare(self, atol=1e-6, rtol=1e-6):
        return compare(self.oracle, self.oracle_arrays, self.candidate, self.candidate_arrays, atol, rtol)

    def test_complete_snapshots_and_controlled_wrapper_normalization(self):
        result = self.compare()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["parameter_tensors"], 15)
        self.assertEqual(result["steps"], 3)
        self.assertEqual(result["tensor_count"], 116)
        self.assertIsNone(result["oracle_fallback_count"])

    def test_same_missing_gradient_on_both_sides_rejected(self):
        del self.oracle_arrays["step_000/param_grad/_fsdp_wrapped_module.score.bias"]
        del self.candidate_arrays["step_000/param_grad/score.bias"]
        with self.assertRaisesRegex(ValueError, "missing.*snapshot"):
            self.compare()

    def test_adapter_provenance_is_side_specific_and_mandatory(self):
        result = self.compare()
        self.assertEqual(result["oracle_compatibility"]["adapters"], {})
        self.assertEqual(set(result["candidate_compatibility"]["adapters"]), {"verl", "tensordict"})
        self.oracle.pop("compatibility")
        self.candidate.pop("compatibility")
        with self.assertRaisesRegex(ValueError, "compatibility provenance"):
            self.compare()

    def test_same_missing_parameter_manifest_rejected(self):
        self.oracle["parameter_names"].remove("_fsdp_wrapped_module.score.bias")
        self.candidate["parameter_names"].remove("score.bias")
        with self.assertRaisesRegex(ValueError, "incomplete.*manifest"):
            self.compare()

    def test_missing_each_required_tensor_group_rejected(self):
        for name in ("initial/score.weight", "step_001/post_weight/score.bias", "infer/values",
                     "post_train/values", "step_001/loss", "step_001/grad_norm"):
            with self.subTest(name=name):
                saved = self.candidate_arrays.pop(name)
                with self.assertRaises(ValueError):
                    self.compare()
                self.candidate_arrays[name] = saved

    def test_corrupt_metadata_rejected(self):
        changes = [
            ("framework", "torch"), ("model_device", "cpu"), ("actual_backend", "gloo"),
            ("world_size", 2), ("fallback_count", 1), ("fallback_count", False),
            ("parameter_count", 3152), ("gradient_scope", "worker grad_norm only"),
            ("requested_stage", "infer"), ("status", "failed"), ("training_steps", []),
            ("stages", []), ("script_sha256", {}), ("checkpoint_sha256", {}),
            ("actual_attention", {"<root>": "eager"}), ("model_spec", {}),
            ("training_config", {}), ("versions", {}),
        ]
        for key, bad in changes:
            with self.subTest(key=key, value=bad):
                original = self.candidate[key]
                self.candidate[key] = bad
                with self.assertRaises(ValueError):
                    self.compare()
                self.candidate[key] = original

    def test_protocol_differences_rejected(self):
        for key, field, bad in (("versions", "transformers", "5.8.0"),
                                ("checkpoint_sha256", "model.safetensors", "d" * 64),
                                ("script_sha256", "probe_training_worker.py", "d" * 64)):
            with self.subTest(key=key):
                original = self.candidate[key][field]
                self.candidate[key][field] = bad
                with self.assertRaisesRegex(ValueError, "protocol differs"):
                    self.compare()
                self.candidate[key][field] = original

    def test_initial_weights_are_exact_even_with_large_tolerance(self):
        # Keep the update summary internally consistent so initial identity is
        # the part tested, rather than failing a later trajectory checksum.
        for name in self.candidate_arrays:
            if name == "initial/score.bias" or name.endswith("post_weight/score.bias"):
                self.candidate_arrays[name] = self.candidate_arrays[name] + np.float32(0.125)
        for step in range(self.candidate["training_config"]["steps"]):
            differences = []
            for name in parameter_shapes():
                previous = "initial/" + name if step == 0 else "step_{:03d}/post_weight/{}".format(step - 1, name)
                differences.append(float(np.max(np.abs(self.candidate_arrays["step_{:03d}/post_weight/{}".format(step, name)] - self.candidate_arrays[previous]))))
            self.candidate["training_steps"][step]["largest_parameter_change"] = max(differences)
        self.assertEqual(self.compare(atol=100, rtol=100)["status"], "fail")

    def test_gradient_numerical_error_fails(self):
        # This trajectory is internally valid SGD but differs from the oracle.
        self.candidate_arrays["step_002/param_grad/score.bias"] *= 7
        self.candidate_arrays["step_002/post_weight/score.bias"] = (
            self.candidate_arrays["step_001/post_weight/score.bias"]
            - np.float32(0.01) * self.candidate_arrays["step_002/param_grad/score.bias"])
        self.candidate["training_steps"][2]["largest_parameter_change"] = float(np.max(np.abs(
            self.candidate_arrays["step_002/post_weight/score.bias"]
            - self.candidate_arrays["step_001/post_weight/score.bias"])))
        self.assertEqual(self.compare()["status"], "fail")

    def test_every_step_and_parameter_requires_explanatory_gradient(self):
        # Identical corruption on both sides cannot be caught by a numerical
        # cross-runtime comparison, even with otherwise consistent summaries.
        for step in range(3):
            for name in parameter_shapes():
                for gradient in (0.0, 7.0):
                    with self.subTest(step=step, parameter=name, gradient=gradient):
                        self.setUp()
                        for arrays, prefix in ((self.oracle_arrays, "_fsdp_wrapped_module."),
                                               (self.candidate_arrays, "")):
                            arrays["step_{:03d}/param_grad/{}{}".format(step, prefix, name)].fill(gradient)
                        with self.assertRaisesRegex(ValueError, "SGD update does not match captured gradient"):
                            self.compare(atol=100, rtol=100)

    def test_every_step_and_parameter_rejects_forged_weight_updates(self):
        for step in range(3):
            for name in parameter_shapes():
                with self.subTest(step=step, parameter=name):
                    self.setUp()
                    for manifest, arrays, prefix in ((self.oracle, self.oracle_arrays, "_fsdp_wrapped_module."),
                                                    (self.candidate, self.candidate_arrays, "")):
                        arrays["step_{:03d}/post_weight/{}{}".format(step, prefix, name)] += np.float32(0.125)
                        # Recompute every checksum, so a fabricated update is
                        # rejected by SGD causality, not a stale max-delta field.
                        for index, summary in enumerate(manifest["training_steps"]):
                            deltas = []
                            for raw_name in manifest["parameter_names"]:
                                previous = "initial/" + raw_name if index == 0 else "step_{:03d}/post_weight/{}".format(index - 1, raw_name)
                                current = "step_{:03d}/post_weight/{}".format(index, raw_name)
                                deltas.append(float(np.max(np.abs(arrays[current] - arrays[previous]))))
                            summary["largest_parameter_change"] = max(deltas)
                    with self.assertRaisesRegex(ValueError, "SGD update does not match captured gradient"):
                        self.compare(atol=100, rtol=100)

    def test_each_runtime_is_independently_checked_for_valid_sgd(self):
        for framework in ("torch", "jittor"):
            for step in range(3):
                with self.subTest(framework=framework, step=step):
                    self.setUp()
                    arrays, prefix = ((self.oracle_arrays, "_fsdp_wrapped_module.")
                                      if framework == "torch" else (self.candidate_arrays, ""))
                    arrays["step_{:03d}/param_grad/{}score.bias".format(step, prefix)].fill(0)
                    with self.assertRaisesRegex(ValueError, "SGD update does not match captured gradient"):
                        self.compare(atol=100, rtol=100)

    def test_nonfinite_dtype_shape_and_tolerance_rejected(self):
        key = "step_001/param_grad/score.bias"
        original = self.candidate_arrays[key]
        for bad in (np.array([np.nan], dtype=np.float32), np.array([np.inf], dtype=np.float32),
                    np.zeros((0,), dtype=np.float32), np.ones((1, 1), dtype=np.float32),
                    np.ones(1, dtype=np.float64)):
            self.candidate_arrays[key] = bad
            with self.assertRaises(ValueError):
                self.compare()
        self.candidate_arrays[key] = original
        for bad in (float("nan"), float("inf"), -1.0):
            with self.assertRaises(ValueError):
                self.compare(atol=bad)
            with self.assertRaises(ValueError):
                self.compare(rtol=bad)

    def test_normalization_collision_rejected(self):
        self.candidate_arrays["initial/_fsdp_wrapped_module.score.bias"] = self.candidate_arrays["initial/score.bias"]
        with self.assertRaisesRegex(ValueError, "collision"):
            self.compare()

    def test_no_general_module_prefix_stripping(self):
        self.candidate["parameter_names"][0] = "module." + self.candidate["parameter_names"][0]
        with self.assertRaisesRegex(ValueError, "unexpected Qwen3"):
            self.compare()

    def test_oracle_zero_fallback_claim_rejected(self):
        self.oracle["oracle_fallback_count"] = 0
        with self.assertRaisesRegex(ValueError, "explicitly unknown"):
            self.compare()


if __name__ == "__main__":
    unittest.main()

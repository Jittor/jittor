"""Schema-4 host negative controls. Synthetic arrays are not device evidence."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = load("probe_training_workflow")
compare = load("compare_training_workflow")


class TestSchema4Artifacts(unittest.TestCase):
    def setUp(self):
        config = dict(model_type="qwen3", vocab_size=32, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                      head_dim=8, attention_bias=False, tie_word_embeddings=False)
        contracts = {role: probe.parameter_spec(config, role) for role in ("actor", "critic")}
        self.manifest = {
            "schema_version": 4, "framework": "torch", "device": "cpu", "fallback_count": None,
            "algorithm": "ppo", "steps": 1, "start_step": 0, "executed_steps": [0], "phase": "full",
            "lr": 0.01, "dtype": "float32", "attention": "eager", "model_config": config,
            "versions": {"transformers": "5.9.0", "verl": "0.9.1"}, "model": "tiny-qwen3",
            "parameter_contract": contracts, "parameter_names": sorted(contracts["actor"]["shapes"]),
            "parameter_count": contracts["actor"]["numel"], "critic": "learned",
            "critic_parameter_names": sorted(contracts["critic"]["shapes"]),
            "critic_parameter_count": contracts["critic"]["numel"], "text_only": True,
            "loading_info": [], "qwen35_adapter": {"required": False, "active": False, "bindings": []},
            "script_sha256": {name: "a" * 64 for name in
                              ("probe_training_workflow.py", "compare_training_workflow.py")},
            "input_gradients": ["input_scale", "embeddings"],
            "critic_input_gradients": ["input_scale", "embeddings"], "embedding_input_shape": [4, 8, 16],
            "reward_rule": "fixture", "reward_source": "synthetic_rule", "grpo_group_ids": [],
            "initial_weight_sha256": "b" * 64, "blocked_optional_packages": [],
            "reference_policy": False, "kl_ratio": None, "reference_parameter_count": 0,
            "reward_parameter_count": 0, "reference_parameter_names": [], "reward_parameter_names": [],
        }
        keys = ["input_ids", "response", "rewards", "response_mask", "values", "advantages", "returns",
                "old_log_prob", "log_prob", "logits", "loss", "clip_fraction", "ppo_kl", "lower_clip_fraction",
                "input_scale", "input_grad/input_scale", "post_update_log_prob", "post_update_loss",
                "token_level_scores", "value_predictions", "value_loss", "value_clip_fraction",
                "post_update_values", "post_update_value_loss", "critic_input_grad/input_scale"]
        self.arrays = {"step_000/" + key: np.array([0.25], dtype=np.float32) for key in keys}
        for role, groups in (("actor", ("param_grad", "post_weight")),
                              ("critic", ("critic_param_grad", "critic_post_weight"))):
            for name, shape in contracts[role]["shapes"].items():
                for group in groups:
                    self.arrays["step_000/" + group + "/" + name] = np.full(shape, 0.25, dtype=np.float32)
        for group in ("input", "input_grad", "critic_input", "critic_input_grad"):
            self.arrays["step_000/" + group + "/embeddings"] = np.full((4, 8, 16), 0.25, dtype=np.float32)

    def hybrid_manifest(self, framework="jittor"):
        manifest = copy.deepcopy(self.manifest)
        config = dict(model_type="qwen3_5_text", vocab_size=248320, hidden_size=1024,
                      intermediate_size=3584, num_hidden_layers=24, num_attention_heads=8,
                      num_key_value_heads=2, head_dim=256, tie_word_embeddings=True,
                      layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 6,
                      linear_num_key_heads=16, linear_num_value_heads=16,
                      linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4)
        manifest.update(model_config=config, model="local-pretrained", framework=framework,
                        embedding_input_shape=[4, 8, 1024])
        manifest["parameter_contract"] = {role: probe.parameter_spec(config, role) for role in ("actor", "critic")}
        for role, prefix in (("actor", ""), ("critic", "critic_")):
            spec = manifest["parameter_contract"][role]
            manifest[prefix + "parameter_names"] = sorted(spec["shapes"])
            manifest[prefix + "parameter_count"] = spec["numel"]
        empty = dict(missing_keys=[], unexpected_keys=[], mismatched_keys=[], error_msgs=[])
        manifest["loading_info"] = [{"role": role, "info": copy.deepcopy(empty)} for role in ("actor", "critic")]
        manifest["qwen35_adapter"] = {"required": True, "active": framework == "jittor", "bindings": [
            {"name": name, "module": "fixture", "qualname": name, "source_sha256": "a" * 64}
            for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")]}
        return manifest

    def test_qwen35_missing_instance_evidence_is_rejected(self):
        for framework in ("torch", "jittor"):
            with self.subTest(framework=framework), self.assertRaisesRegex(AssertionError, "instance"):
                compare.validate_schema4(self.hybrid_manifest(framework))

    def instance_manifest(self, framework="jittor"):
        manifest = self.hybrid_manifest(framework)
        adapter = manifest["qwen35_adapter"]
        adapter["instance_profile"] = "hf_torch_gated_delta_instances_v1"
        adapter["instances"] = {}
        for role, prefix in (("actor", "model"), ("critic", "backbone")):
            layers = []
            for index, kind in enumerate(manifest["model_config"]["layer_types"]):
                if kind != "linear_attention":
                    continue
                bindings = []
                for source in adapter["bindings"]:
                    binding = {key: source[key] for key in ("module", "qualname", "source_sha256")}
                    binding.update(attribute=source["name"].removeprefix("torch_"), target=source["name"],
                                   identity_matches=True, functional_adapter=framework == "jittor")
                    bindings.append(binding)
                layers.append({"path": "{}.layers.{}.linear_attn".format(prefix, index),
                    "class_module": "transformers.models.qwen3_5.modeling_qwen3_5",
                    "class_name": "Qwen3_5GatedDeltaNet", "bindings": bindings})
            adapter["instances"][role] = {"expected_layers": 18, "layers": layers}
        return manifest

    def test_qwen35_all_actual_instances_on_both_runtimes_pass(self):
        for framework in ("torch", "jittor"):
            with self.subTest(framework=framework):
                compare.validate_schema4(self.instance_manifest(framework))

    def test_qwen35_missing_layer_and_wrong_instance_provenance_fail(self):
        for role in ("actor", "critic"):
            for change in (
                lambda item: item.update(expected_layers=17),
                lambda item: item["layers"].pop(),
                lambda item: item["layers"][0].update(path="wrong.path"),
                lambda item: item["layers"][0].update(class_name="FusedDeltaNet"),
                lambda item: item["layers"][0]["bindings"].pop(),
                lambda item: item["layers"][0]["bindings"][0].update(identity_matches=False),
                lambda item: item["layers"][0]["bindings"][1].update(identity_matches=1),
                lambda item: item["layers"][0]["bindings"][0].update(functional_adapter=False),
                lambda item: item["layers"][0]["bindings"][1].update(source_sha256="b" * 64),
                lambda item: item["layers"][0]["bindings"][0].update(module="fla.ops"),
            ):
                manifest = self.instance_manifest()
                change(manifest["qwen35_adapter"]["instances"][role])
                with self.subTest(role=role, change=change), self.assertRaises(AssertionError):
                    compare.validate_schema4(manifest)
        manifest = self.instance_manifest()
        del manifest["qwen35_adapter"]["instances"]["critic"]
        with self.assertRaisesRegex(AssertionError, "roles"):
            compare.validate_schema4(manifest)

    def test_tiny_does_not_invent_qwen35_instance_evidence(self):
        compare.validate_schema4(self.manifest)
        self.manifest["qwen35_adapter"]["instances"] = {}
        with self.assertRaisesRegex(AssertionError, "unexpected.*instance"):
            compare.validate_schema4(self.manifest)

    def test_complete_schema4_contract_and_arrays_pass(self):
        compare.validate_schema4(self.manifest)
        compare.validate_coverage(self.manifest, set(self.arrays))
        for name, value in self.arrays.items():
            compare.validate_schema4_array(self.manifest, name, value)

    def test_missing_any_embedding_or_parameter_gradient_rejected(self):
        for name in list(self.arrays):
            if "param_grad/" not in name and not name.endswith("/embeddings"):
                continue
            with self.subTest(name=name), self.assertRaisesRegex(AssertionError, "incomplete training snapshot"):
                compare.validate_coverage(self.manifest, set(self.arrays) - {name})

    def test_both_manifests_omitting_same_parameter_cannot_pass(self):
        for role, field in (("actor", "parameter_names"), ("critic", "critic_parameter_names")):
            manifest = copy.deepcopy(self.manifest)
            removed = manifest[field].pop()
            del manifest["parameter_contract"][role]["shapes"][removed]
            with self.subTest(role=role), self.assertRaisesRegex(AssertionError, "independent parameter"):
                compare.validate_schema4(manifest)

    def test_sampled_or_wrong_dtype_gradient_rejected(self):
        name = "step_000/param_grad/model.embed_tokens.weight"
        for value in (np.ones(5, dtype=np.float32), np.ones((32, 16), dtype=np.float16)):
            with self.subTest(shape=value.shape, dtype=value.dtype), self.assertRaises(AssertionError):
                compare.validate_schema4_array(self.manifest, name, value)
        with self.assertRaises(AssertionError):
            compare.validate_schema4_array(self.manifest, "step_000/input_grad/embeddings",
                                          np.ones(1, dtype=np.float32))

    def test_missing_new_metadata_is_not_legacy_schema4(self):
        for key in ("parameter_contract", "input_gradients", "critic_input_gradients", "embedding_input_shape",
                    "loading_info", "qwen35_adapter", "script_sha256", "text_only"):
            manifest = copy.deepcopy(self.manifest)
            del manifest[key]
            with self.subTest(key=key), self.assertRaises((AssertionError, KeyError)):
                compare.validate_schema4(manifest)

    def test_false_dtype_alias_step_or_version_declaration_fails(self):
        for change in (lambda m: m.update(executed_steps=[]), lambda m: m.update(dtype="bfloat16"),
                       lambda m: m["versions"].update(transformers="5.8.0"),
                       lambda m: m["parameter_contract"]["actor"].update(aliases=[["a", "b"]]),
                       lambda m: m.update(input_gradients=["input_scale"]),
                       lambda m: m["qwen35_adapter"].update(active=True)):
            manifest = copy.deepcopy(self.manifest)
            change(manifest)
            with self.assertRaises(AssertionError):
                compare.validate_schema4(manifest)

    def test_loading_info_required_and_bad_text_load_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["model"] = "local-pretrained"
        with self.assertRaisesRegex(AssertionError, "loading evidence"):
            compare.validate_schema4(manifest)
        empty = dict(missing_keys=[], unexpected_keys=[], mismatched_keys=[], error_msgs=[])
        manifest["loading_info"] = [{"role": role, "info": copy.deepcopy(empty)} for role in ("actor", "critic")]
        compare.validate_schema4(manifest)
        manifest["loading_info"][1]["info"]["missing_keys"] = ["norm.weight"]
        with self.assertRaisesRegex(AssertionError, "missing pretrained text"):
            compare.validate_schema4(manifest)

    def compare_files(self, mutate=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for framework, stem in (("torch", "oracle"), ("jittor", "candidate")):
                manifest = copy.deepcopy(self.manifest)
                manifest.update(framework=framework, fallback_count=0 if framework == "jittor" else None)
                arrays = dict(self.arrays)
                if mutate is not None:
                    mutate(manifest, arrays)
                np.savez(root / (stem + ".npz"), **arrays)
                (root / (stem + ".json")).write_text(json.dumps(manifest))
            return subprocess.run([sys.executable, str(Path(compare.__file__)),
                "--oracle", str(root / "oracle.npz"), "--candidate", str(root / "candidate.npz"),
                "--atol", "1e-5", "--rtol", "1e-5", "--output", str(root / "result.json")],
                capture_output=True, text=True, timeout=30)

    def test_full_cli_and_symmetric_missing_gradient_negative(self):
        result = self.compare_files()
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.compare_files(lambda m, a: a.pop("step_000/critic_input_grad/embeddings"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("incomplete training snapshot", result.stderr)

    def test_legacy_schema3_does_not_claim_new_coverage(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["schema_version"] = 3
        manifest.pop("parameter_contract")
        compare.validate_schema4(manifest)
        compare.validate_coverage(manifest, {key for key in self.arrays if not key.endswith("/embeddings")})


if __name__ == "__main__":
    unittest.main()

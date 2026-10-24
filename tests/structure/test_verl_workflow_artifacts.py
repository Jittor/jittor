"""Negative controls for the standalone verl trajectory artifact comparator."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from _helpers.child_process import run_python_child


SCRIPT = (Path(__file__).resolve().parents[2] /
          "agent/skills/verl-torch-compat/scripts/compare_training_workflow.py")


class TestVerlWorkflowArtifacts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = {
            "schema_version": 2, "framework": "torch", "device": "cpu",
            "fallback_count": None, "algorithm": "ppo", "steps": 1, "lr": 0.01,
            "dtype": "float32", "attention": "eager", "model_config": {},
            "versions": {"verl": "0.9.1"}, "parameter_count": 1,
            "reward_rule": "fixture", "grpo_group_ids": [], "critic": "learned",
            "critic_parameter_count": 1, "parameter_names": ["weight"],
            "critic_parameter_names": ["value_head.weight"],
            "initial_weight_sha256": "independent-initial-weights",
            "blocked_optional_packages": [],
        }
        keys = ["input_ids", "response", "rewards", "response_mask", "values",
                "advantages", "returns", "old_log_prob", "log_prob", "logits",
                "clip_fraction", "ppo_kl", "lower_clip_fraction", "input_scale",
                "post_update_log_prob", "value_predictions", "value_clip_fraction",
                "post_update_values", "loss", "post_update_loss", "input_grad/input_scale",
                "param_grad/weight", "post_weight/weight", "value_loss",
                "post_update_value_loss", "critic_input_grad/input_scale",
                "critic_param_grad/value_head.weight", "critic_post_weight/value_head.weight"]
        self.arrays = {"step_000/" + name: np.array([0.25], dtype=np.float32) for name in keys}
        self.candidate = copy.deepcopy(self.manifest)
        self.candidate.update(framework="jittor", fallback_count=0)

    def compare(self, candidate_arrays=None, atol="0.00001"):
        for name, manifest, arrays in (
            ("oracle", self.manifest, self.arrays),
            ("candidate", self.candidate, self.arrays if candidate_arrays is None else candidate_arrays),
        ):
            np.savez(self.root / (name + ".npz"), **arrays)
            (self.root / (name + ".json")).write_text(json.dumps(manifest))
        return run_python_child(
            [str(SCRIPT), "--oracle", str(self.root / "oracle.npz"),
             "--candidate", str(self.root / "candidate.npz"), "--atol", atol,
             "--rtol", "0.00001", "--output", str(self.root / "result.json")],
        )

    def test_complete_matching_trajectory_passes(self):
        result = self.compare()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((self.root / "result.json").read_text())["status"], "pass")

    def test_missing_critic_gradient_on_both_sides_fails(self):
        del self.arrays["step_000/critic_param_grad/value_head.weight"]
        result = self.compare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("incomplete training snapshot", result.stderr)

    def test_critic_gradient_mismatch_fails(self):
        candidate_arrays = copy.deepcopy(self.arrays)
        candidate_arrays["step_000/critic_param_grad/value_head.weight"] += 0.1
        self.assertNotEqual(self.compare(candidate_arrays).returncode, 0)
        self.assertEqual(json.loads((self.root / "result.json").read_text())["status"], "fail")

    def test_missing_forward_state_on_both_sides_fails(self):
        for key in ("values", "advantages", "response", "value_predictions"):
            with self.subTest(key=key):
                value = self.arrays.pop("step_000/" + key)
                result = self.compare()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("incomplete training snapshot", result.stderr)
                self.arrays["step_000/" + key] = value

    def test_manifest_mismatches_fail(self):
        for key, value in (("device", "npu"), ("critic", "zero"),
                           ("fallback_count", 1), ("initial_weight_sha256", "other")):
            with self.subTest(key=key):
                original = self.candidate[key]
                self.candidate[key] = value
                self.assertNotEqual(self.compare().returncode, 0)
                self.candidate[key] = original

    def test_nonfinite_tensor_fails(self):
        candidate_arrays = copy.deepcopy(self.arrays)
        candidate_arrays["step_000/value_loss"][:] = np.nan
        self.assertNotEqual(self.compare(candidate_arrays).returncode, 0)

    def test_nonfinite_tolerance_fails(self):
        for value in ("nan", "inf", "-1"):
            with self.subTest(value=value):
                self.assertNotEqual(self.compare(atol=value).returncode, 0)

    def test_schema3_rlhf_requires_reference_reward_and_score_artifacts(self):
        fields = {
            "schema_version": 3, "algorithm": "rlhf",
            "reward_source": "independent_transformer_reward_model_plus_reference_kl",
            "reward_rule": "fixture rlhf", "reference_policy": True, "kl_ratio": 0.1,
            "reference_parameter_count": 1, "reward_parameter_count": 1,
            "reference_parameter_names": ["weight"], "reward_parameter_names": ["weight"],
        }
        self.manifest.update(fields)
        self.candidate.update(fields)
        for key in ("token_level_scores", "ref_log_prob", "reward_model_scores"):
            self.arrays["step_000/" + key] = np.array([0.25], dtype=np.float32)
        self.assertEqual(self.compare().returncode, 0)
        for key in ("ref_log_prob", "reward_model_scores"):
            with self.subTest(key=key):
                value = self.arrays.pop("step_000/" + key)
                result = self.compare()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("incomplete training snapshot", result.stderr)
                self.arrays["step_000/" + key] = value

    def test_schema3_rlvr_requires_verifier_and_matching_reward_contract(self):
        fields = {
            "schema_version": 3, "algorithm": "rlvr",
            "critic": "zero", "critic_parameter_count": 0, "critic_parameter_names": [],
            "reward_source": "deterministic_inverse_token_distance_verifier",
            "reward_rule": "fixture rlvr", "reference_policy": False, "kl_ratio": None,
            "reference_parameter_count": 0, "reward_parameter_count": 0,
            "reference_parameter_names": [], "reward_parameter_names": [],
        }
        self.manifest.update(fields)
        self.candidate.update(fields)
        for key in ("token_level_scores", "verifier_scores"):
            self.arrays["step_000/" + key] = np.array([0.25], dtype=np.float32)
        self.assertEqual(self.compare().returncode, 0)
        self.candidate["reward_source"] = "wrong"
        self.assertNotEqual(self.compare().returncode, 0)


if __name__ == "__main__":
    unittest.main()

import copy
import json
from pathlib import Path
import tempfile
import unittest

from compare_llm_benchmark import compare


class TestCompareLLMBenchmark(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.base = {
            "framework": "torch", "task": "train", "objective": "ppo",
            "device": "cpu", "world_size": 1, "model": "/models/qwen",
            "parameter_count": 494032768, "dtype": "float32", "attention": "eager",
            "learning_rate": 1e-5, "adam_eps": 1e-8,
            "parameter_dtypes": ["torch.float32"],
            "initial_old_log_prob_dtype": "torch.float32",
            "transformers_materialized_init": False,
            "qwen35_functional_fallback": False,
            "batch_size_per_rank": 1, "seq_len": 32, "warmup_steps": 2,
            "timed_steps": 3, "blocked_optional_packages": ["torch_npu"],
            "cpu_thread_env": {"OMP_NUM_THREADS": "16", "MKL_NUM_THREADS": "16"},
            "step_times_s": [10.0, 11.0, 12.0], "losses": [-1.0, -1.1, -1.2],
            "initial_old_log_prob": [[-2.0, -3.0]],
            "cold_log_prob": [[-2.0, -3.0]],
            "timed_log_probs": [[[-2.1, -3.1]], [[-2.2, -3.2]], [[-2.3, -3.3]]],
            "cold_gradient_samples": {"model.norm.weight": [1.0, -2.0]},
            "cold_update_samples": {"model.norm.weight": [0.1, -0.2]},
            "cold_missing_gradient_samples": [],
            "peak_memory_bytes": 1000, "parameter_location": "cpu",
            "output_location": "cpu",
        }
        self.candidate = copy.deepcopy(self.base)
        self.candidate.update(
            framework="jittor", fallback_delta=0,
            transformers_materialized_init=True,
            step_times_s=[18.0, 20.0, 22.0],
            losses=[-1.000001, -1.100001, -1.200001],
            peak_memory_bytes=900,
        )

    def write(self):
        oracle = self.root / "oracle.json"
        candidate = self.root / "candidate.json"
        oracle.write_text(json.dumps(self.base))
        candidate.write_text(json.dumps(self.candidate))
        return oracle, candidate

    def test_reports_target_and_memory_ratios(self):
        oracle, candidate = self.write()
        result = compare([oracle], [candidate], 2.0)
        self.assertEqual(result["status"], "meets_target")
        self.assertAlmostEqual(result["candidate_over_oracle_step_ratio"], 20 / 11)
        self.assertAlmostEqual(result["candidate_over_oracle_peak_memory_ratio"], 0.9)

    def test_missed_target_is_reported_not_hidden(self):
        oracle, candidate = self.write()
        result = compare([oracle], [candidate], 1.5)
        self.assertEqual(result["status"], "misses_target")

    def test_qwen35_requires_autograd_safe_fallback(self):
        self.base["model"] = self.candidate["model"] = "/models/qwen3_5-0.8b"
        oracle, candidate = self.write()
        with self.assertRaisesRegex(AssertionError, "autograd-safe fallback"):
            compare([oracle], [candidate], 2.0)
        self.candidate["qwen35_functional_fallback"] = True
        oracle, candidate = self.write()
        self.assertEqual(compare([oracle], [candidate], 2.0)["status"], "meets_target")

    def test_fallback_contract_or_loss_mismatch_is_rejected(self):
        for key, value in (("fallback_delta", 1), ("losses", [0.0, 0.0, 0.0]),
                           ("cold_log_prob", [[0.0, 0.0]])):
            with self.subTest(key=key):
                old = self.candidate[key]
                self.candidate[key] = value
                oracle, candidate = self.write()
                with self.assertRaises(AssertionError):
                    compare([oracle], [candidate], 2.0)
                self.candidate[key] = old

    def test_matching_zero_updates_are_valid_but_one_sided_zero_is_rejected(self):
        self.base["cold_update_samples"].update(zero_sample=[0.0, 0.0])
        self.candidate["cold_update_samples"].update(zero_sample=[0.0, 0.0])
        oracle, candidate = self.write()
        self.assertEqual(compare([oracle], [candidate], 2.0)["status"], "meets_target")

        self.candidate["cold_update_samples"]["zero_sample"] = [0.0, 1e-7]
        oracle, candidate = self.write()
        with self.assertRaisesRegex(AssertionError, "one-sided zero norm"):
            compare([oracle], [candidate], 2.0)

    def test_all_zero_sampled_updates_are_rejected(self):
        self.base["cold_update_samples"]["model.norm.weight"] = [0.0, 0.0]
        self.candidate["cold_update_samples"]["model.norm.weight"] = [0.0, 0.0]
        oracle, candidate = self.write()
        with self.assertRaisesRegex(AssertionError, "updates are all zero"):
            compare([oracle], [candidate], 2.0)

    def cuda_fixture(self):
        for report in (self.base, self.candidate):
            report.update(
                device="cuda", blocked_optional_packages=[], cpu_thread_env=None,
                shared_cuda=False, performance_eligible=True,
                runtime_authorization={"VERL_ALLOW_SHARED_CUDA": "0",
                                       "VERL_L4_PERFORMANCE_ELIGIBLE": "1"},
                runtime_conditions={"cuda_matmul_allow_tf32": False,
                                    "cudnn_allow_tf32": False,
                                    "cudnn_benchmark": False,
                                    "float32_matmul_precision": "highest"},
            )
        self.base.update(parameter_location="cuda:0", output_location="cuda:0")
        self.candidate.update(parameter_location="device", output_location="device")

    def mark_shared(self, report):
        report.update(shared_cuda=True, performance_eligible=False)
        report["runtime_authorization"]["VERL_ALLOW_SHARED_CUDA"] = "1"
        report["runtime_authorization"]["VERL_L4_PERFORMANCE_ELIGIBLE"] = "0"

    def test_cuda_requires_explicit_precision_on_either_or_both_sides(self):
        self.cuda_fixture()
        for left, right in ((True, False), (False, True), (False, False)):
            with self.subTest(left=left, right=right):
                self.cuda_fixture()
                if not left:
                    self.base.pop("runtime_conditions")
                if not right:
                    self.candidate.pop("runtime_conditions")
                oracle, candidate = self.write()
                with self.assertRaisesRegex(AssertionError, "CUDA runtime conditions"):
                    compare([oracle], [candidate], 2.0)

    def test_wrong_cuda_precision_even_if_matching_is_rejected(self):
        for key, value in (("cuda_matmul_allow_tf32", True), ("cudnn_allow_tf32", True),
                           ("cudnn_benchmark", True), ("float32_matmul_precision", "high"),
                           ("cuda_matmul_allow_tf32", 0)):
            for both in (False, True):
                with self.subTest(key=key, value=value, both=both):
                    self.cuda_fixture()
                    self.candidate["runtime_conditions"][key] = value
                    if both:
                        self.base["runtime_conditions"][key] = value
                    oracle, candidate = self.write()
                    with self.assertRaisesRegex(AssertionError, "CUDA runtime conditions"):
                        compare([oracle], [candidate], 2.0)

    def test_cuda_matching_precision_is_recorded_in_comparison(self):
        self.cuda_fixture()
        oracle, candidate = self.write()
        result = compare([oracle], [candidate], 2.0)
        self.assertEqual(result["runtime_conditions"], self.base["runtime_conditions"])
        self.assertIs(result["performance_eligible"], True)

    def test_shared_cuda_on_either_or_both_sides_cannot_meet_target(self):
        for left, right in ((True, False), (False, True), (True, True)):
            for target in (0.5, 100.0):
                with self.subTest(left=left, right=right, target=target):
                    self.cuda_fixture()
                    if left:
                        self.mark_shared(self.base)
                    if right:
                        self.mark_shared(self.candidate)
                    oracle, candidate = self.write()
                    result = compare([oracle], [candidate], target)
                    self.assertEqual(result["status"], "observation_only_shared_cuda")
                    self.assertIs(result["performance_eligible"], False)
                    self.assertIs(result["shared_cuda"], True)

    def test_cuda_missing_or_contradictory_authorization_is_rejected(self):
        for key, value in (("shared_cuda", None), ("performance_eligible", None),
                           ("runtime_authorization", None), ("shared_cuda", 0)):
            with self.subTest(key=key, value=value):
                self.cuda_fixture()
                if value is None:
                    self.candidate.pop(key)
                else:
                    self.candidate[key] = value
                oracle, candidate = self.write()
                with self.assertRaisesRegex(AssertionError, "CUDA performance context"):
                    compare([oracle], [candidate], 2.0)
        self.cuda_fixture()
        self.mark_shared(self.candidate)
        self.candidate["performance_eligible"] = True
        oracle, candidate = self.write()
        with self.assertRaisesRegex(AssertionError, "CUDA performance context"):
            compare([oracle], [candidate], 2.0)

    def test_explicit_ineligible_cuda_is_observation_not_meets_target(self):
        self.cuda_fixture()
        self.candidate["performance_eligible"] = False
        self.candidate["runtime_authorization"]["VERL_L4_PERFORMANCE_ELIGIBLE"] = "0"
        oracle, candidate = self.write()
        result = compare([oracle], [candidate], 100.0)
        self.assertEqual(result["status"], "observation_only_performance_ineligible")
        self.assertIs(result["performance_eligible"], False)

    def test_cuda_contract_checks_later_repeated_runs_on_both_sides(self):
        for side in ("oracle", "candidate"):
            for fault in ("precision", "authorization", "shared"):
                with self.subTest(side=side, fault=fault):
                    self.cuda_fixture()
                    oracle, candidate = self.write()
                    later = copy.deepcopy(self.base if side == "oracle" else self.candidate)
                    if fault == "precision":
                        later["runtime_conditions"].pop("cudnn_benchmark")
                    elif fault == "authorization":
                        later["runtime_authorization"]["VERL_ALLOW_SHARED_CUDA"] = "true"
                    else:
                        self.mark_shared(later)
                    later_path = self.root / "later.json"
                    later_path.write_text(json.dumps(later))
                    left = [oracle, later_path if side == "oracle" else oracle]
                    right = [candidate, later_path if side == "candidate" else candidate]
                    if fault == "shared":
                        result = compare(left, right, 100.0)
                        self.assertEqual(result["status"], "observation_only_shared_cuda")
                        self.assertIs(result["performance_eligible"], False)
                    else:
                        with self.assertRaisesRegex(AssertionError, "CUDA"):
                            compare(left, right, 100.0)

    def test_cuda_authorization_must_match_boolean_metadata(self):
        for name in ("VERL_ALLOW_SHARED_CUDA", "VERL_L4_PERFORMANCE_ELIGIBLE"):
            for value in (None, "", "true", "2", 0, "1" if name == "VERL_ALLOW_SHARED_CUDA" else "0"):
                with self.subTest(name=name, value=value):
                    self.cuda_fixture()
                    if value is None:
                        self.candidate["runtime_authorization"].pop(name)
                    else:
                        self.candidate["runtime_authorization"][name] = value
                    oracle, candidate = self.write()
                    with self.assertRaisesRegex(AssertionError, "CUDA performance context"):
                        compare([oracle], [candidate], 100.0)

    def test_legacy_npu_does_not_require_cuda_metadata(self):
        self.base.update(device="npu", parameter_location="npu:0", output_location="npu:0")
        self.candidate.update(device="npu", parameter_location="device", output_location="device")
        oracle, candidate = self.write()
        self.assertEqual(compare([oracle], [candidate], 2.0)["status"], "meets_target")


if __name__ == "__main__":
    unittest.main()

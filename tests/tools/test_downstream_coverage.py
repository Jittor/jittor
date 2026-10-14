"""Downstream breadth/scorecard negative contracts; no tensor runtime required."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "agent/skills/torch-compat-repo-runbook/scripts"
sys.path.insert(0, str(ROOT / "tests"))
from _helpers.child_process import PYTHON, run_python_child  # noqa: E402

sys.path.insert(0, str(SCRIPTS))
import audit_coverage as audit  # noqa: E402
import registry_inventory as registry  # noqa: E402


class CoverageAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.lab = Path(self.temp.name)
        self.root = self.lab / "checkout"
        (self.root / "docs/results").mkdir(parents=True)
        (self.root / "docs/results/evidence.md").write_text("measured evidence", encoding="utf-8")
        (self.root / "tests").mkdir()
        (self.root / "tests/test_case.py").write_text("def test_real():\n    pass\n", encoding="utf-8")
        self.baseline = "a" * 40
        self.snapshot = {"repo": "transformers", "upstream": {"commit": "b" * 40, "remote": "https://example.invalid/upstream.git", "version": "1.0"},
                         "axes": [{"axis": "model_family", "count": 2, "members": ["llama", "bert"]}]}
        self.sources = {"transformers": self.snapshot}
        raw = self.lab / "raw.json"
        raw.write_text('{"executed": true}', encoding="utf-8")
        self.row = {
            "repo": "transformers", "case": "llama_tiny", "status": "passed", "level": 2, "tier": "broad",
            "jittor_commit": self.baseline, "upstream": self.snapshot["upstream"],
            "downstream_versions": {"candidate": "1.0", "oracle": "1.0"},
            "registry_refs": [{"repo": "transformers", "axis": "model_family", "member": "llama"}],
            "device": "cpu", "observed_device": "cpu",
            "platform": {"os": "Linux", "cpu": "test CPU", "accelerator_model": "not-applicable",
                         "card_count": 0, "card_indices": [], "driver": "not-applicable", "cuda": "not-applicable",
                         "cann": "not-applicable", "nvcc": "not-applicable", "gcc": "test compiler", "nvidia_wheels": {},
                         "jittor_flags": {"use_cuda": 0, "use_acl": 0, "graph_replay": "eager", "tf32": False, "hf32": False},
                         "oracle": {"is_shim": False, "torch_version": "2.11", "torch_npu_version": "not-applicable", "binary_extension": "torch/_C.so"}},
            "inputs": {"seed": 1, "hf_hub_offline": True, "shape_kind": "tiny"},
            "numerics": {"passed": True, "finite": True, "checked_tensors": 4, "missing_tensors": [],
                         "input_perturbation_verified": True, "max_abs": 1e-6, "max_scaled_relative": 1e-5, "atol": 1e-4, "rtol": 1e-3,
                         "parameter_gradients": {"requested": 2, "checked": 2}, "input_gradients": {"requested": 1, "checked": 1}},
            "evidence": {"command": "python -m pytest tests/test_case.py::test_real", "report": "docs/results/evidence.md",
                         "gate_nodeid": "tests/test_case.py::test_real", "artifacts": [
                             {"role": role, "path": str(raw), "sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}
                             for role in ("reference", "candidate", "comparison", "platform")]},
        }

    def validate(self, row):
        return audit.validate_record(row, self.sources, self.baseline, self.root)

    def test_valid_l2_counts_one_of_entire_registry(self):
        result = audit.scorecard(self.sources, [self.row], self.baseline, self.root)
        cpu = next(row for row in result["coverage"] if row["device"] == "cpu")
        self.assertEqual((cpu["numerator"], cpu["denominator"]), (1, 2))

    def test_l1_import_is_not_breadth(self):
        self.row["level"] = 1
        result = audit.scorecard(self.sources, [self.row], self.baseline, self.root)
        self.assertTrue(all(row["numerator"] == 0 for row in result["coverage"]))

    def test_stale_sha_small_denominator_and_unknown_members_fail(self):
        for field, value in (("jittor_commit", "c" * 40), ("upstream", dict(self.snapshot["upstream"], commit="d" * 40)),
                             ("registry_refs", [{"repo": "transformers", "axis": "model_family", "member": "invented"}])):
            with self.subTest(field=field):
                row = copy.deepcopy(self.row)
                row[field] = value
                with self.assertRaises(ValueError):
                    self.validate(row)

    def test_skip_xfail_and_unrun_are_not_passed(self):
        for status in ("skipped", "xfail", "registered", "ran", "failed"):
            self.row["status"] = status
            with self.assertRaisesRegex(ValueError, "Only passed"):
                self.validate(self.row)

    def test_missing_platform_or_shim_oracle_fail(self):
        for alter in (lambda row: row.pop("platform"),
                      lambda row: row["platform"]["oracle"].update(is_shim=True),
                      lambda row: row["platform"]["jittor_flags"].update(use_cuda=1)):
            row = copy.deepcopy(self.row)
            alter(row)
            with self.assertRaises(ValueError):
                self.validate(row)

    def test_missing_gradient_perturbation_or_nan_fails(self):
        for alter in (lambda row: row["numerics"]["parameter_gradients"].update(checked=1),
                      lambda row: row["numerics"].update(input_perturbation_verified=False),
                      lambda row: row["numerics"].update(max_abs=float("nan"))):
            row = copy.deepcopy(self.row)
            alter(row)
            with self.assertRaises(ValueError):
                self.validate(row)

    def test_artifact_tampering_and_invented_gate_fail(self):
        self.row["evidence"]["artifacts"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.validate(self.row)
        self.row["evidence"]["gate_nodeid"] = "tests/test_case.py::test_invented"
        with self.assertRaisesRegex(ValueError, "Unknown gate"):
            self.validate(self.row)

    def test_single_step_does_not_establish_training(self):
        self.row["level"] = 3
        with self.assertRaisesRegex(ValueError, "multi-step"):
            self.validate(self.row)

    def accelerator_row(self):
        row = copy.deepcopy(self.row)
        row.update(device="cuda", observed_device="cuda", fallback={"count": 0, "policy": "error"})
        row["platform"].update(card_count=1, card_indices=["allocated"], accelerator_model="test GPU")
        row["platform"]["jittor_flags"]["use_cuda"] = 1
        return row

    def test_accelerator_needs_matching_cpu_and_zero_fallback(self):
        row = self.accelerator_row()
        result = audit.scorecard(self.sources, [row], self.baseline, self.root)
        self.assertTrue(all(entry["numerator"] == 0 for entry in result["coverage"]))
        result = audit.scorecard(self.sources, [self.row, row], self.baseline, self.root)
        self.assertEqual(next(entry for entry in result["coverage"] if entry["device"] == "cuda")["numerator"], 1)
        row["fallback"]["count"] = 1
        with self.assertRaisesRegex(ValueError, "zero-fallback"):
            self.validate(row)

    def test_l4_requires_real_workload_agreement_regime_and_process_memory(self):
        row = self.accelerator_row()
        row["level"] = 5
        row["inputs"]["shape_kind"] = "real"
        row["training"] = {"steps": 3, "loss_trajectory_passed": True, "parameter_trajectory_passed": True}
        row["speed"] = {"model": "llama", "batch": 2, "sequence": 128, "dtype": "float32", "precision": "strict",
                        "phase": "train", "jittor_seconds": 0.1, "oracle_seconds": 0.1,
                        "agreement": 1e-6, "agreement_tolerance": 1e-4,
                        "warmup": 3, "iterations": 10, "graph_regime": "eager", "no_concurrent_work": True,
                        "statistic": "median", "ratio_limit": 1.2,
                        "memory": {"source": "nvml_process", "jittor_peak_bytes": 1024, "oracle_peak_bytes": 1024}}
        self.assertEqual(self.validate(row), 5)
        for alter in (lambda value: value["inputs"].update(shape_kind="tiny"),
                      lambda value: value["speed"].pop("agreement"),
                      lambda value: value["speed"].update(graph_regime="default"),
                      lambda value: value["speed"].update(no_concurrent_work=False),
                      lambda value: value["speed"].update(ratio=2),
                      lambda value: value["speed"].update(ratio_limit=0.5),
                      lambda value: value["speed"]["memory"].update(source="runtime_allocator")):
            changed = copy.deepcopy(row)
            alter(changed)
            with self.assertRaises(ValueError):
                self.validate(changed)

    def test_installed_dependency_versions_must_match_upstream(self):
        self.row["downstream_versions"]["oracle"] = "other"
        with self.assertRaisesRegex(ValueError, "versions"):
            self.validate(self.row)

    def test_real_audit_is_host_only_and_does_not_promote_registration(self):
        result = run_python_child(
            ["-S", str(SCRIPTS / "audit_coverage.py"), "audit", "--json"],
            check=True, timeout=30)
        value = json.loads(result.stdout)
        self.assertTrue({"verl", "ms-swift", "transformers"}.issubset({row["repo"] for row in value["repos"]}))
        self.assertTrue(all(row["verified_level"] is None for row in value["repos"]))
        for row in value["repos"]:
            self.assertTrue({case["case"] for case in row["gate_cases"]}.issubset(row["registered_cases"]))


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_git("init", "-q")
        self.run_git("config", "user.name", "Test")
        self.run_git("config", "user.email", "test@example.invalid")
        self.run_git("remote", "add", "origin", "https://example.invalid/upstream.git")

    def run_git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.root)] + list(args), text=True).strip()

    def commit(self, path, content):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self.run_git("add", path)
        self.run_git("commit", "-qm", "test registry")

    def test_registry_is_rederived_after_upstream_change(self):
        path = "src/transformers/models/auto/modeling_auto.py"
        self.commit(path, 'raise RuntimeError("must not execute")\nMODEL_MAPPING_NAMES = {"llama": "LlamaModel"}\n')
        first = registry.inventory("transformers", self.root)
        self.commit(path, 'MODEL_MAPPING_NAMES = {"llama": "LlamaModel", "bert": "BertModel"}\n')
        second = registry.inventory("transformers", self.root)
        self.assertEqual((first["axes"][0]["count"], second["axes"][0]["count"]), (1, 2))
        self.assertNotEqual(first["upstream"]["commit"], second["upstream"]["commit"])

    def test_dirty_and_empty_registry_fail_closed(self):
        path = "src/transformers/models/auto/modeling_auto.py"
        self.commit(path, 'MODEL_MAPPING_NAMES = {}\n')
        with self.assertRaisesRegex(ValueError, "Empty"):
            registry.inventory("transformers", self.root)
        (self.root / path).write_text('MODEL_MAPPING_NAMES = {"llama":"Model"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dirty"):
            registry.inventory("transformers", self.root)

    def test_dynamic_registry_does_not_silently_shrink_denominator(self):
        self.commit("src/transformers/models/auto/modeling_auto.py", 'MODEL_MAPPING_NAMES = get_mapping()\n')
        with self.assertRaisesRegex(ValueError, "Unsupported registry"):
            registry.inventory("transformers", self.root)

    def test_peft_enum_is_enumerated_without_imports(self):
        self.commit("src/peft/utils/peft_types.py", 'import unavailable_torch\nclass PeftType:\n    LORA="LORA"\n    DORA="DORA"\nclass TaskType:\n    CAUSAL_LM="CAUSAL_LM"\n')
        result = registry.inventory("peft", self.root)
        self.assertEqual(result["axes"][0]["members"], ["DORA", "LORA"])


class VerifyRunnerTests(unittest.TestCase):
    def test_pair_device_precision_versions_and_threads_must_match(self):
        spec = importlib.util.spec_from_file_location("verify_repo_metadata", SCRIPTS / "verify_repo.py")
        verify = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verify)
        reference = {"case": "same", "device": "cpu", "seconds": 0.1, "tf32": {"matmul": False},
                     "dependencies": {"downstream": {"version": "1"}},
                     "runtime_conditions": {"affinity": [0], "runtime_threads": 2, "thread_env": {}, "precision": {}}}
        candidate = dict(reference, fallback_count=0, fallback_policy="error")
        self.assertEqual(verify._pair_problems(reference, candidate, "cpu"), [])
        for field, changed in (("device", "cuda"), ("tf32", {"matmul": True}),
                               ("dependencies", {"downstream": {"version": "2"}}),
                               ("runtime_conditions", {}), ("seconds", float("nan")), ("fallback_count", 1)):
            with self.subTest(field=field):
                row = copy.deepcopy(candidate)
                row[field] = changed
                self.assertTrue(verify._pair_problems(reference, row, "cpu"))

    def test_nonzero_exit_after_success_marker_is_failure(self):
        spec = importlib.util.spec_from_file_location("verify_repo_host", SCRIPTS / "verify_repo.py")
        verify = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verify)
        with tempfile.TemporaryDirectory() as directory:
            runner = Path(directory) / "runner.py"
            runner.write_text('print(\'ECOSYSTEM_RESULT {"seconds":1}\')\nraise SystemExit(3)\n', encoding="utf-8")
            verify.RUNNER = runner
            with self.assertRaisesRegex(SystemExit, "exit 3"):
                verify._run_case(PYTHON, "case", Path(directory) / "out.npz", {}, "cpu", 1, 0)


if __name__ == "__main__":
    unittest.main()

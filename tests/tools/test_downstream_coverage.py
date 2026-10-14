"""Downstream breadth/scorecard negative contracts; no tensor runtime required."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import types
from unittest import mock


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
        self.serial = 0
        self.bind_measurements(self.row)

    def test_deepspeed_engine_gate_is_separate_from_ordinary_model_cases(self):
        row = next(item for item in audit.audit(ROOT)["repos"] if item["repo"] == "deepspeed")
        self.assertEqual(row["registered_cases"], [])
        self.assertEqual(row["gate_cases"], [])
        self.assertEqual(row["dedicated_gates"], [{
            "protocol": "deepspeed-engine", "session": "ecosystem_deepspeed",
            "tests": ["compat/tests/torch/test_ecosystem_deepspeed.py::test_real_deepspeed_engine_training_and_resume"],
        }])
        self.assertIsNone(row["verified_level"])

    def test_dedicated_gate_rejects_missing_test_or_session(self):
        gate = self.root / "compat/tests/torch/test_ecosystem_deepspeed.py"
        gate.parent.mkdir(parents=True)
        test_source = "def test_real_deepspeed_engine_training_and_resume(): pass\n"
        session_source = "def ecosystem_deepspeed(session): pass\n"
        gate.write_text(test_source)
        nox = self.root / "noxfile.py"
        nox.write_text(session_source)
        self.assertEqual(len(audit.dedicated_gates(self.root)["deepspeed"]), 1)
        for target, source in ((gate, test_source), (nox, session_source)):
            with self.subTest(path=target.name):
                target.write_text("def unrelated(): pass\n")
                with self.assertRaisesRegex(ValueError, "dedicated gate"):
                    audit.dedicated_gates(self.root)
                target.write_text(source)

    def artifact_json(self, row, role, value):
        self.serial += 1
        path = self.lab / ("artifact-%d.json" % self.serial)
        path.write_text(json.dumps(value), encoding="utf-8")
        artifact = {"role": role, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        row["evidence"]["artifacts"] = [item for item in row["evidence"]["artifacts"] if item["role"] != role] + [artifact]

    def bind_measurements(self, row):
        enabled = row["platform"]["jittor_flags"]["tf32"]
        precision = {"matmul": enabled, "cudnn": enabled, "cudnn_benchmark": False,
                     "matmul_precision": "high" if enabled else "highest"}
        row["platform"]["precision"] = precision
        self.artifact_json(row, "platform", row["platform"])
        reference = {"case": row["case"], "device": row["device"], "runtime": "torch",
                     "tf32": precision, "runtime_conditions": {"precision": precision, "runtime_threads": 2},
                     "jittor_commit": row["jittor_commit"], "steps": 3,
                     "trainable_parameters": ["weight", "bias"], "frozen_parameters": []}
        candidate = dict(reference, runtime="jittor", fallback_policy="error", fallback_count=0,
                         backend={"use_cuda": bool(row["platform"]["jittor_flags"]["use_cuda"]),
                                  "use_acl": bool(row["platform"]["jittor_flags"]["use_acl"])})
        self.artifact_json(row, "base-comparison", {"baseline": row["jittor_commit"], "cases": [
            {"case": row["case"], "status": "passed", "oracle_result": reference, "jittor_result": candidate}]})
        self.artifact_json(row, "perturb-comparison", {"case": row["case"], "status": "passed",
                           "input_perturbation_verified": True, "reference_result": reference, "candidate_result": candidate})
        self.artifact_json(row, "training-reference-metadata", reference)
        self.artifact_json(row, "training-candidate-metadata", candidate)

    def document(self, row, role):
        artifact = next(item for item in row["evidence"]["artifacts"] if item["role"] == role)
        return json.loads(Path(artifact["path"]).read_text())

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
        self.bind_measurements(row)
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
        row["training"] = {"steps": 3, "loss_trajectory_passed": True, "parameter_trajectory_passed": True,
                           "trainable": 2, "frozen": 0, "trainable_parameters": ["weight", "bias"], "frozen_parameters": []}
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

    def test_actual_hashed_precision_overrides_a_stale_platform_label(self):
        row = self.accelerator_row()
        actual = {"matmul": True, "cudnn": True, "cudnn_benchmark": False, "matmul_precision": "high"}
        base = self.document(row, "base-comparison")
        for key in ("oracle_result", "jittor_result"):
            base["cases"][0][key]["tf32"] = actual
            base["cases"][0][key]["runtime_conditions"]["precision"] = actual
        self.artifact_json(row, "base-comparison", base)
        with self.assertRaisesRegex(ValueError, "Platform/measurement precision"):
            self.validate(row)
        # Updating only the report/platform label must not bless the old raw run.
        row = self.accelerator_row()
        row["platform"]["jittor_flags"]["tf32"] = True
        row["platform"]["precision"] = actual
        self.artifact_json(row, "platform", row["platform"])
        with self.assertRaisesRegex(ValueError, "Platform/measurement precision"):
            self.validate(row)

    def test_precision_is_checked_for_both_runtimes_and_every_phase(self):
        for role, key in (("base-comparison", "oracle_result"), ("base-comparison", "jittor_result"),
                          ("perturb-comparison", "reference_result"), ("perturb-comparison", "candidate_result"),
                          ("training-reference-metadata", None), ("training-candidate-metadata", None)):
            with self.subTest(role=role, runtime=key):
                row = self.accelerator_row()
                row["level"] = 3
                row["training"] = {"steps": 3, "loss_trajectory_passed": True, "parameter_trajectory_passed": True,
                                   "trainable": 2, "frozen": 0, "trainable_parameters": ["weight", "bias"], "frozen_parameters": []}
                document = self.document(row, role)
                value = document["cases"][0][key] if role == "base-comparison" else document[key] if key else document
                value["tf32"]["matmul_precision"] = "medium"
                self.artifact_json(row, role, document)
                with self.assertRaisesRegex(ValueError, "precision"):
                    self.validate(row)

    def test_missing_ambiguous_or_stale_measurement_json_is_rejected(self):
        for changed in ("missing", "duplicate", "platform", "case", "baseline", "device", "backend", "runtime"):
            with self.subTest(changed=changed):
                row = self.accelerator_row()
                if changed == "missing":
                    row["evidence"]["artifacts"] = [item for item in row["evidence"]["artifacts"] if item["role"] != "base-comparison"]
                elif changed == "duplicate":
                    row["evidence"]["artifacts"].append(dict(row["evidence"]["artifacts"][-1]))
                elif changed == "platform":
                    self.artifact_json(row, "platform", {"different": True})
                else:
                    base = self.document(row, "base-comparison")
                    if changed == "baseline":
                        base["baseline"] = "c" * 40
                    else:
                        candidate = base["cases"][0]["jittor_result"]
                        if changed == "case":
                            candidate["case"] = "other"
                        if changed == "device":
                            candidate["device"] = "cpu"
                        if changed == "backend":
                            candidate["backend"]["use_cuda"] = False
                        if changed == "runtime":
                            candidate["runtime_conditions"]["runtime_threads"] = 99
                    self.artifact_json(row, "base-comparison", base)
                with self.assertRaises(ValueError):
                    self.validate(row)

    def test_training_scope_must_match_the_actual_parameter_names(self):
        row = self.accelerator_row()
        row["level"] = 3
        row["training"] = {"steps": 3, "loss_trajectory_passed": True, "parameter_trajectory_passed": True,
                           "trainable": 2, "frozen": 0, "trainable_parameters": ["base.weight", "base.bias"],
                           "frozen_parameters": []}
        with self.assertRaisesRegex(ValueError, "parameter scope"):
            self.validate(row)

    def test_historical_cli_requires_an_explicit_full_commit(self):
        result = run_python_child(["-S", str(SCRIPTS / "audit_coverage.py"), "scorecard",
                                   "--baseline", "abcd", "--upstream", "transformers=missing",
                                   "--output", str(self.lab / "output.json")])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Historical baseline requires a full SHA", result.stderr)
        self.assertFalse((self.lab / "output.json").exists())

    def test_coverage_preserves_method_shape_and_trainable_scope(self):
        self.snapshot["axes"].append({"axis": "method", "count": 2, "members": ["LORA", "DENSE"]})
        row = self.accelerator_row()
        row["registry_refs"].append({"repo": "transformers", "axis": "method", "member": "LORA"})
        row["level"] = 3
        row["training"] = {"steps": 3, "loss_trajectory_passed": True, "parameter_trajectory_passed": True,
                           "trainable": 2, "frozen": 0, "trainable_parameters": ["weight", "bias"], "frozen_parameters": []}
        cpu = copy.deepcopy(row)
        cpu.update(device="cpu", observed_device="cpu")
        cpu["platform"].update(card_count=0, card_indices=[], accelerator_model="not-applicable")
        cpu["platform"]["jittor_flags"]["use_cuda"] = 0
        self.bind_measurements(cpu)
        value = audit.scorecard(self.sources, [cpu, row], self.baseline, self.root)
        self.assertEqual(value["schema_version"], 2)
        self.assertFalse(value["rejected"])
        family = next(item for item in value["coverage"] if item["axis"] == "model_family" and item["device"] == "cuda")
        scenario = family["verified_scenarios"][0]
        self.assertEqual(scenario["registry_refs"], row["registry_refs"])
        self.assertEqual((scenario["repo"], scenario["case"], scenario["level"], scenario["shape_kind"]),
                         (row["repo"], row["case"], 3, "tiny"))
        self.assertEqual(scenario["trainable_scope"]["trainable_parameters"], ["weight", "bias"])

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
    def test_artifact_move_keeps_explicit_cache_and_isolates_devices(self):
        sys.path.insert(0, str(SCRIPTS))
        import verify_repo as verify

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = {"JITTOR_ECOSYSTEM_CACHE_ROOT": str(root / "cache")}
            cpu = verify._runtime_environment(env, "jittor", "cpu", root / "run-one")
            again = verify._runtime_environment(env, "jittor", "cpu", root / "run-two")
            cuda = verify._runtime_environment(env, "jittor", "cuda", root / "run-two")
            self.assertEqual(cpu["JITTOR_HOME"], again["JITTOR_HOME"])
            self.assertEqual(cpu["JITTOR_TORCH_CACHE_ROOT"], again["JITTOR_TORCH_CACHE_ROOT"])
            self.assertNotEqual(cpu["JITTOR_HOME"], cuda["JITTOR_HOME"])
            self.assertNotEqual(cpu["JITTOR_TORCH_CACHE_ROOT"], cuda["JITTOR_TORCH_CACHE_ROOT"])
            oracle = verify._runtime_environment(cpu, "torch", "cpu", root / "reference")
            self.assertNotIn("JITTOR_HOME", oracle)
            self.assertNotIn("JITTOR_ECOSYSTEM_CACHE_ROOT", oracle)
            default = verify._runtime_environment({}, "jittor", "cpu", root / "ordinary")
            self.assertEqual(default["JITTOR_HOME"], str(root / "ordinary/jittor-cache"))

    def test_package_visibility_includes_explicit_site_and_interpreter_paths(self):
        sys.path.insert(0, str(ROOT / "compat/tests/torch"))
        import _ecosystem_harness as harness

        with tempfile.TemporaryDirectory() as directory:
            explicit = Path(directory) / "explicit"
            installed = Path(directory) / "installed"
            explicit.mkdir()
            installed.mkdir()
            (explicit / "explicit_probe_dependency.py").write_text("raise RuntimeError('must not import')\n")
            (installed / "installed_probe_dependency.py").write_text("raise RuntimeError('must not import')\n")
            with mock.patch.object(harness, "PACKAGE_SITE", str(explicit)), \
                    mock.patch.object(sys, "path", [str(installed)] + sys.path):
                self.assertTrue(harness._distributions_available(
                    ("explicit_probe_dependency", "installed_probe_dependency")))
                self.assertFalse(harness._distributions_available(("missing_probe_dependency",)))
            self.assertNotIn("explicit_probe_dependency", sys.modules)
            self.assertNotIn("installed_probe_dependency", sys.modules)

    def test_only_the_missing_dependency_own_required_flag_can_fail(self):
        sys.path.insert(0, str(ROOT / "compat/tests/torch"))
        import _ecosystem_harness as harness

        with mock.patch.object(harness, "_distributions_available",
                               side_effect=lambda names: "peft" not in names), \
                mock.patch.dict(os.environ, {"JITTOR_REQUIRE_TRANSFORMERS": "1",
                                             "JITTOR_REQUIRE_PEFT": "0"}):
            comparison = harness.EcosystemComparison()
            with self.assertRaisesRegex(unittest.SkipTest, "missing peft"):
                comparison._compare("peft_lora_llama")
            with mock.patch.dict(os.environ, {"JITTOR_REQUIRE_PEFT": "1"}):
                with self.assertRaisesRegex(AssertionError, "required ecosystem dependencies missing: peft"):
                    comparison._compare("peft_lora_llama")

    def test_runner_rejects_nonzero_exit_after_success_marker(self):
        sys.path.insert(0, str(ROOT / "compat/tests/torch"))
        import _ecosystem_harness as harness

        completed = subprocess.CompletedProcess([], 7, 'ECOSYSTEM_RESULT {"case":"transformers_bert","device":"cpu"}\n')
        for runtime, python in (("torch", "/independent/oracle/python"),
                                ("jittor", harness.PYTHON)):
            with self.subTest(runtime=runtime), \
                    mock.patch.object(harness, "_runner_package_site", return_value=""), \
                    mock.patch.object(harness, "run_python_child", return_value=completed), \
                    mock.patch.object(harness.subprocess, "run", return_value=completed):
                with self.assertRaisesRegex(AssertionError, "exit.*7"):
                    harness._run(python, runtime, "transformers_bert", Path("unused.npz"))

    def test_runner_rejects_success_marker_for_another_or_missing_case(self):
        sys.path.insert(0, str(ROOT / "compat/tests/torch"))
        import _ecosystem_harness as harness

        for reported_case in ("another_case", None):
            payload = {"case": reported_case, "device": "cpu"}
            completed = subprocess.CompletedProcess([], 0, "ECOSYSTEM_RESULT " + json.dumps(payload))
            for runtime, python in (("torch", "/independent/oracle/python"),
                                    ("jittor", harness.PYTHON)):
                with self.subTest(runtime=runtime, reported_case=reported_case), \
                        mock.patch.object(harness, "_runner_package_site", return_value=""), \
                        mock.patch.object(harness, "run_python_child", return_value=completed), \
                        mock.patch.object(harness.subprocess, "run", return_value=completed):
                    with self.assertRaisesRegex(AssertionError, "case mismatch"):
                        harness._run(python, runtime, "transformers_bert", Path("unused.npz"))

    def test_comparison_rejects_asymmetric_broadcast_or_nonfinite_arrays(self):
        sys.path.insert(0, str(ROOT / "compat/tests/torch"))
        import numpy as np
        import _ecosystem_harness as harness

        scenarios = [
            ("candidate-extra-gradient", "jittor", "grad::extra", np.ones(1), "tensor keys"),
            ("oracle-extra-gradient", "torch", "grad::extra", np.ones(1), "tensor keys"),
            ("broadcast-output", "jittor", "__output__", np.ones((2, 1)), "shape"),
            ("broadcast-gradient", "jittor", "grad::weight", np.ones((2, 1)), "shape"),
            ("oracle-nan-gradient", "torch", "grad::weight", np.full(2, np.nan), "finite"),
            ("candidate-nan-gradient", "jittor", "grad::weight", np.full(2, np.nan), "finite"),
            ("oracle-inf-gradient", "torch", "grad::weight", np.full(2, np.inf), "finite"),
            ("candidate-inf-gradient", "jittor", "grad::weight", np.full(2, np.inf), "finite"),
            ("oracle-nan-output", "torch", "__output__", np.full(2, np.nan), "finite"),
            ("candidate-inf-output", "jittor", "__output__", np.full(2, np.inf), "finite"),
        ]
        for label, changed_runtime, changed_key, changed_value, message in scenarios:
            with self.subTest(scenario=label):
                def run(_python, runtime, _case, output, **_kwargs):
                    arrays = {"__output__": np.ones(2), "grad::weight": np.ones(2)}
                    if runtime == changed_runtime:
                        arrays[changed_key] = changed_value
                    np.savez(output, **arrays)
                    if runtime == "torch":
                        np.savez(output.with_name("torch.weights.npz"), weight=np.ones(1))
                    return {"device": "cpu", "seconds": 1.0,
                            "tf32": {}, "runtime_conditions": {}, "dependencies": {},
                            "fallback_policy": "error", "fallback_count": 0}, ""

                with mock.patch.object(harness, "_run", side_effect=run), \
                        mock.patch.object(harness, "_distributions_available", return_value=True), \
                        mock.patch.object(harness, "_runner_package_site", return_value=""), \
                        mock.patch.object(harness, "SPEED_RATIO", ""):
                    with self.assertRaisesRegex(AssertionError, message):
                        harness.EcosystemComparison()._compare("transformers_bert")

    def test_required_gradient_cannot_disappear_from_both_runtimes(self):
        sys.path.insert(0, str(ROOT / "compat/tests/torch"))
        import numpy as np
        import _ecosystem_harness as harness

        required = "grad::encoder.pooler.dense.weight"
        for absent in ({"torch"}, {"jittor"}, {"torch", "jittor"}, set()):
            with self.subTest(absent=absent):
                def run(_python, runtime, _case, output, **_kwargs):
                    arrays = {"__output__": np.ones(1), "grad::other": np.ones(1)}
                    if runtime not in absent:
                        arrays[required] = np.ones(1)
                    np.savez(output, **arrays)
                    if runtime == "torch":
                        np.savez(output.with_name("torch.weights.npz"), weight=np.ones(1))
                    return {"device": "cpu", "seconds": 1.0,
                            "tf32": {}, "runtime_conditions": {}, "dependencies": {},
                            "fallback_policy": "error", "fallback_count": 0}, ""

                with mock.patch.object(harness, "_run", side_effect=run), \
                        mock.patch.object(harness, "_distributions_available", return_value=True), \
                        mock.patch.object(harness, "_runner_package_site", return_value=""), \
                        mock.patch.object(harness, "SPEED_RATIO", ""):
                    comparison = harness.EcosystemComparison()
                    if absent:
                        with self.assertRaisesRegex(AssertionError, "omitted required gradients"):
                            comparison._compare("transformers_bert", required_gradients=(required,))
                    else:
                        comparison._compare("transformers_bert", required_gradients=(required,))

    def test_metric_and_tensordict_cases_list_without_runtime_dependencies(self):
        for repo, case in (("torchmetrics", "torchmetrics_regression"),
                           ("tensordict", "tensordict_index_and_stack")):
            with self.subTest(repo=repo):
                result = run_python_child(
                    ["-S", str(SCRIPTS / "verify_repo.py"), "--repo", repo, "--list-only"],
                    without_torch_mode=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.split(), ["CASE", case, "requires", repo])

    def test_verl_selected_gate_cannot_skip_missing_dependency_sites(self):
        fake = types.ModuleType("_ecosystem_harness")
        fake.REAL_TORCH_PYTHON = "independent-oracle"
        fake.EcosystemComparison = unittest.TestCase
        fake._torch_shim_is_active = lambda: True
        fake._distributions_available = lambda requirements: False
        sys.path.insert(0, str(ROOT / "compat/tests/torch"))
        path = ROOT / "compat/tests/torch/test_ecosystem_verl.py"
        spec = importlib.util.spec_from_file_location("verl_gate_contract", path)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"_ecosystem_harness": fake}), \
                mock.patch.dict(os.environ, {"JITTOR_TEST_DEVICES": "cpu"}):
            spec.loader.exec_module(module)
        comparison = module.EcosystemVERL()
        with self.assertRaisesRegex(AssertionError, "requires available dependencies"):
            comparison.setUp()
        module._distributions_available = lambda requirements: True
        comparison.setUp()

    def test_verl_cases_and_separate_gate_are_discoverable_without_frameworks(self):
        result = run_python_child(
            ["-S", str(SCRIPTS / "verify_repo.py"), "--repo", "verl", "--list-only"],
            without_torch_mode=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual({line.split()[1] for line in result.stdout.splitlines()},
                         {"verl_ppo_gae_vanilla", "verl_distillation_k3"})
        for line in result.stdout.splitlines():
            self.assertIn("numpy", line)
            self.assertIn("transformers", line)
        rows = audit.audit(ROOT)["repos"]
        row = next(item for item in rows if item["repo"] == "verl")
        self.assertEqual({item["case"] for item in row["gate_cases"]},
                         {"verl_ppo_gae_vanilla", "verl_distillation_k3"})
        self.assertIsNone(row["verified_level"])

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

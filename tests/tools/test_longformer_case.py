"""Host-only admission of the real 32K Longformer workload."""
from pathlib import Path
import importlib.util
import os
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from _helpers.child_process import run_python_child  # noqa: E402


class LongformerCaseTests(unittest.TestCase):
    def test_builder_and_gradient_contract_load_without_tensor_dependencies(self):
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "import _ecosystem_longformer_cases as case; "
            "assert case.SEQUENCE_LENGTH == 32768; "
            "assert case.ATTENTION_WINDOW == 256; "
            "assert case.HIDDEN_SIZE == 32; "
            "assert len(set(case.LONGFORMER_PARAMETER_NAMES)) == 29; "
            "assert len(set(case.REQUIRED_GRADIENTS)) == 30; "
            "assert 'grad::model.pooler.dense.weight' in case.REQUIRED_GRADIENTS; "
            "assert 'grad::model.encoder.layer.0.attention.self.query_global.weight' in case.REQUIRED_GRADIENTS; "
            "assert 'ingrad::embedding_noise' in case.REQUIRED_GRADIENTS; "
            "assert 'torch' not in sys.modules and 'transformers' not in sys.modules"
        )
        result = run_python_child(["-S", "-c", code, str(ROOT / "compat/tests/torch")],
                                  without_torch_mode=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_registry_listing_and_audit_include_the_executable_node(self):
        script = ROOT / "agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py"
        result = run_python_child(["-S", str(script), "--repo", "transformers", "--list-only"],
                                  without_torch_mode=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CASE transformers_longformer_32768 requires transformers", result.stdout)
        sys.path.insert(0, str(script.parent))
        import audit_coverage
        row = next(row for row in audit_coverage.audit(ROOT)["repos"] if row["repo"] == "transformers")
        selected = next(case for case in row["gate_cases"] if case["case"] == "transformers_longformer_32768")
        self.assertEqual(selected["tests"], [
            "compat/tests/torch/test_ecosystem_longformer.py::test_transformers_longformer_32768"
        ])

    def test_selected_gate_fails_missing_prerequisites_instead_of_skipping(self):
        sys.path.insert(0, str(ROOT / "compat/tests/torch"))
        fake = types.ModuleType("_ecosystem_harness")
        fake.REAL_TORCH_PYTHON = ""
        fake.SPEED_RATIO = ""
        fake.EcosystemComparison = unittest.TestCase
        fake._enabled = lambda name: True
        fake._torch_shim_is_active = lambda: True
        fake._distributions_available = lambda names: True
        path = ROOT / "compat/tests/torch/test_ecosystem_longformer.py"
        spec = importlib.util.spec_from_file_location("longformer_admission_probe", path)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"_ecosystem_harness": fake}):
            spec.loader.exec_module(module)
        self.assertFalse(getattr(module.EcosystemLongformer, "__unittest_skip__", False))
        gate = module.EcosystemLongformer()
        with self.assertRaisesRegex(AssertionError, "independent real Torch"):
            gate.setUp()
        module.REAL_TORCH_PYTHON = "/declared-oracle"
        module._distributions_available = lambda names: False
        with self.assertRaisesRegex(AssertionError, "requires available Transformers"):
            gate.setUp()
        module._distributions_available = lambda names: True
        with mock.patch.dict(os.environ, {"JITTOR_TEST_DEVICES": "cpu,cuda"}):
            with self.assertRaisesRegex(AssertionError, "exactly one"):
                gate.setUp()
        module.SPEED_RATIO = "1.07"
        with self.assertRaisesRegex(AssertionError, "empty speed ratio"):
            gate.setUp()

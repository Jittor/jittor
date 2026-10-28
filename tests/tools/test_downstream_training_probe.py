"""Host-only negative contracts for the opt-in ecosystem training probe."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from _helpers.child_process import run_python_child  # noqa: E402

PROBE = ROOT / "agent/skills/torch-compat-repo-runbook/scripts/training_probe.py"


class TrainingProbeArguments(unittest.TestCase):
    def setUp(self):
        self.external = tempfile.TemporaryDirectory()
        self.addCleanup(self.external.cleanup)
        self.args = ["--root", str(ROOT), "--case", "transformers_gpt2", "--runtime", "torch",
                     "--device", "cpu", "--output", str(Path(self.external.name) / "probe.npz")]

    def test_help_has_no_tensor_or_downstream_dependencies(self):
        result = run_python_child(["-S", str(PROBE), "--help"], without_torch_mode=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--steps", result.stdout)

    def test_single_step_and_bad_learning_rates_fail_before_imports(self):
        for extra in (["--steps", "1"], ["--steps", "0"], ["--learning-rate", "nan"],
                      ["--learning-rate", "inf"], ["--learning-rate", "0"], ["--learning-rate", "-1"]):
            with self.subTest(arguments=extra):
                result = run_python_child(["-S", str(PROBE)] + self.args + extra,
                                          without_torch_mode=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertNotIn("No module named", result.stderr)

    def test_checkout_output_is_rejected_before_imports(self):
        result = run_python_child(["-S", str(PROBE)] + self.args + ["--output", str(ROOT / "probe.npz")],
                                  without_torch_mode=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("outside the checkout", result.stderr)
        self.assertFalse((ROOT / "probe.npz").exists())

    def test_unknown_case_does_not_import_numpy_or_torch(self):
        result = run_python_child(["-S", str(PROBE)] + self.args + ["--case", "invented_case"],
                                  without_torch_mode=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unknown registered case", result.stderr)
        self.assertNotIn("No module named", result.stderr)

    def test_missing_runner_root_is_rejected(self):
        result = run_python_child(["-S", str(PROBE)] + self.args + ["--root", str(Path(self.external.name) / "missing-checkout")],
                                  without_torch_mode=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("shared ecosystem runner", result.stderr)


if __name__ == "__main__":
    unittest.main()

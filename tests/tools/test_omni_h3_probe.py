"""H3 diagnostics preserve failure and never grant partial-gradient training."""

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

from _helpers.child_process import run_python_child


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "agent/skills/vllm-omni-torch-compat/scripts/probe_h3.py"


class H3ProbeContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("omni_h3_probe", SCRIPT)
        cls.probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.probe)

    def complete(self):
        return [
            dict(
                seed=seed,
                expected_parameters=["body.weight", "head.weight"],
                expected_inputs=["x"],
                missing_parameters=[],
                missing_inputs=[],
            )
            for seed in (7, 19)
        ]

    def test_full_two_input_gradient_ledger_is_accepted(self):
        self.probe.require_complete_gradients(self.complete())

    def test_partial_head_gradients_cannot_accept_the_full_model(self):
        for field, name in (("missing_parameters", "body.weight"), ("missing_inputs", "x")):
            for seed in (0, 1):
                with self.subTest(field=field, seed=seed):
                    records = self.complete()
                    records[seed][field].append(name)
                    with self.assertRaisesRegex(RuntimeError, "incomplete"):
                        self.probe.require_complete_gradients(records)

    def test_one_or_repeated_input_cannot_replace_both_fixed_inputs(self):
        rows = self.complete()
        for records in ([], rows[:1], [rows[0], copy.deepcopy(rows[0])]):
            with self.subTest(records=records):
                with self.assertRaisesRegex(RuntimeError, "both fixed-input"):
                    self.probe.require_complete_gradients(records)

    def test_missing_or_empty_expected_names_cannot_hide_an_unmeasured_gradient(self):
        for field in (
            "expected_parameters",
            "expected_inputs",
            "missing_parameters",
            "missing_inputs",
        ):
            records = self.complete()
            del records[0][field]
            with self.subTest(field=field):
                with self.assertRaisesRegex(RuntimeError, "metadata"):
                    self.probe.require_complete_gradients(records)
        for field in ("expected_parameters", "expected_inputs"):
            records = self.complete()
            records[0][field] = []
            with self.subTest(field=field):
                with self.assertRaisesRegex(RuntimeError, "expected parameter and input"):
                    self.probe.require_complete_gradients(records)

    def cli(self, *extra):
        return run_python_child(
            ["-S", str(SCRIPT), *extra],
            cwd=ROOT,
            without_torch_mode=True,
        )

    def arguments(self, output):
        return [
            "--runtime",
            "jittor",
            "--device",
            "cpu",
            "--source-root",
            str(ROOT),
            "--vllm-checkout",
            "/nonexistent/vllm",
            "--omni-checkout",
            "/nonexistent/omni",
            "--output",
            str(output),
        ]

    def test_help_needs_no_numpy_tensor_or_downstream_packages(self):
        result = self.cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--torch-math", result.stdout)
        self.assertIn("grad-audit", result.stdout)

    def test_candidate_cannot_relabel_itself_as_genuine_math_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.cli(*self.arguments(Path(directory) / "raw.npz"), "--torch-math")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("only the genuine-Torch calibration", result.stderr)

    def test_rejects_raw_artifacts_in_the_source_tree_before_imports(self):
        result = self.cli(*self.arguments(ROOT / "raw.npz"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside the source tree", result.stderr)

    def test_refuses_to_overwrite_existing_evidence_before_imports(self):
        for suffix in (".npz", ".json", ".weights.npz"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "raw.npz"
                saved = output.with_suffix(suffix)
                saved.write_bytes(b"sealed")
                result = self.cli(*self.arguments(output))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("preserving existing evidence", result.stderr)
                self.assertEqual(saved.read_bytes(), b"sealed")


if __name__ == "__main__":
    unittest.main()

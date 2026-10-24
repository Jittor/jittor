import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from compare_ppo_resume import compare


class TestComparePPOResume(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.baseline_path = self.root / "baseline.npz"
        self.resume_path = self.root / "resume.npz"
        self.manifest = {
            "phase": "baseline", "algorithm": "ppo", "critic": "learned",
            "framework": "jittor", "device": "cpu", "steps": 3,
            "checkpoint_after": 2, "start_step": 0, "executed_steps": [0, 1, 2],
            "process_id": 10, "seed": 7, "lr": 0.01,
            "initial_weight_sha256": "initial", "checkpoint_sha256": "checkpoint",
            "parameter_names": ["actor"], "critic_parameter_names": ["critic"],
            "versions": {"verl": "0.9.1"}, "model_config": {"hidden_size": 2},
            "fallback_count": 0,
        }
        self.baseline_arrays = {
            "step_000/loss": np.array(3.0, dtype=np.float32),
            "step_001/loss": np.array(2.0, dtype=np.float32),
            "step_002/loss": np.array(1.0, dtype=np.float32),
            "step_002/input_ids": np.array([1, 2], dtype=np.int64),
        }
        self.resume = copy.deepcopy(self.manifest)
        self.resume.update(
            phase="resume", start_step=2, executed_steps=[2], process_id=11
        )
        self.resume_arrays = {
            key: value.copy() for key, value in self.baseline_arrays.items()
            if key.startswith("step_002/")
        }

    def write(self):
        np.savez(self.baseline_path, **self.baseline_arrays)
        np.savez(self.resume_path, **self.resume_arrays)
        self.baseline_path.with_suffix(".json").write_text(json.dumps(self.manifest))
        self.resume_path.with_suffix(".json").write_text(json.dumps(self.resume))

    def test_exact_fresh_process_continuation_passes(self):
        self.write()
        result = compare(self.baseline_path, self.resume_path, 0.0, 0.0)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["tensor_count"], 2)

    def test_same_process_is_rejected(self):
        self.resume["process_id"] = self.manifest["process_id"]
        self.write()
        with self.assertRaisesRegex(AssertionError, "fresh process"):
            compare(self.baseline_path, self.resume_path, 0.0, 0.0)

    def test_missing_or_changed_continuation_is_rejected(self):
        del self.resume_arrays["step_002/input_ids"]
        self.write()
        with self.assertRaisesRegex(AssertionError, "missing or adds"):
            compare(self.baseline_path, self.resume_path, 0.0, 0.0)
        self.resume_arrays["step_002/input_ids"] = np.array([1, 2], dtype=np.int64)
        self.resume_arrays["step_002/loss"] += 0.1
        self.write()
        with self.assertRaisesRegex(AssertionError, "float tensor differs"):
            compare(self.baseline_path, self.resume_path, 0.0, 0.0)

    def test_fallback_and_contract_changes_are_rejected(self):
        for key, value in (("fallback_count", 1), ("checkpoint_sha256", "other"),
                           ("executed_steps", [])):
            with self.subTest(key=key):
                old = self.resume[key]
                self.resume[key] = value
                self.write()
                with self.assertRaises(AssertionError):
                    compare(self.baseline_path, self.resume_path, 0.0, 0.0)
                self.resume[key] = old


if __name__ == "__main__":
    unittest.main()

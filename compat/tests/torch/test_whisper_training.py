"""Original Whisper task-loss and three-step optimizer parity on CPU."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from _helpers.child_process import PYTHON, run_python_child
from _ecosystem_harness import (
    REAL_TORCH_PYTHON, _enabled, _require_case_dependencies,
    _runner_package_site, _torch_shim_is_active,
)

PROBE = Path(__file__).resolve().parents[3] / "agent/skills/whisper-torch-compat/acceptance_probe.py"


class OpenAIWhisperTrainingCPU(unittest.TestCase):
    def test_task_loss_three_sgd_steps(self):
        for ready, reason in (
            (bool(REAL_TORCH_PYTHON), "REAL_TORCH_PYTHON is not configured"),
            (_torch_shim_is_active(), "this interpreter does not run torch as Jittor"),
        ):
            if not ready:
                if _enabled("JITTOR_REQUIRE_WHISPER"):
                    self.fail(reason)
                self.skipTest(reason)
        _require_case_dependencies(self, "openai_whisper", ("whisper",))
        with tempfile.TemporaryDirectory(prefix="whisper-training-") as directory:
            root = Path(directory)
            for runtime, python in (("torch", REAL_TORCH_PYTHON), ("jittor", PYTHON)):
                output = root / (runtime + ".json")
                args = [str(PROBE), "--runtime", runtime, "--stage", "training",
                        "--fixture", str(root / "fixture.npz"), "--output", str(output)]
                env = os.environ.copy()
                env.update(JT_BACKEND="cpu", use_cuda="0", use_acl="0", JT_USE_CUDA="0",
                           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
                package_site = _runner_package_site(python)
                if package_site:
                    env["JITTOR_ECOSYSTEM_PACKAGE_SITE"] = package_site
                if runtime == "torch":
                    env["PYTHONPATH"] = ""
                    for key in ("JITTOR_TORCH_SHIM", "JITTOR_SOURCE_ROOT", "JITTOR_HOME"):
                        env.pop(key, None)
                    result = subprocess.run([python] + args, env=env, capture_output=True,
                                            text=True, encoding="utf-8", errors="replace", timeout=1800)
                else:
                    env["JITTOR_TORCH_SHIM"] = "1"
                    result = run_python_child(args, env=env, inherit=False, merge_stderr=True, timeout=1800)
                self.assertEqual(result.returncode, 0, (result.stdout + (getattr(result, "stderr", None) or ""))[-6000:])
                self.assertEqual(json.loads(output.read_text())["status"], "passed")
            spec = importlib.util.spec_from_file_location("whisper_acceptance_probe", PROBE)
            probe = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(probe)
            probe.compare_training(root / "torch.json", root / "jittor.json")

"""Numerical parity with PyTorch for the downstream libraries Jittor targets.

The existing downstream tests assert that a model *runs* under
``import torch`` -> Jittor and that gradients appear.  That is not the same as
agreeing with PyTorch.  This module runs the same case twice -- once in a real
PyTorch interpreter, once in this Jittor one -- from identical weights, and
compares the forward output, every parameter gradient and every input gradient.

Configuration
-------------

``REAL_TORCH_PYTHON``
    Interpreter whose ``import torch`` is an independent binary PyTorch.  The
    tests skip when it is unset, because a comparison against Jittor's own
    ``torch`` shim would be self-referential and would prove nothing.

``JITTOR_ECOSYSTEM_SPEED_RATIO``
    Optional upper bound on ``jittor_seconds / torch_seconds``.  The wall-clock
    numbers are always reported; they are only asserted when this is set, since
    a shared machine makes an unconditional timing gate flaky.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import _ecosystem_cases


RUNNER = Path(__file__).resolve().parent / "_ecosystem_runner.py"

REAL_TORCH_PYTHON = os.environ.get("REAL_TORCH_PYTHON", "").strip()

SPEED_RATIO = os.environ.get("JITTOR_ECOSYSTEM_SPEED_RATIO", "").strip()


def _torch_shim_is_active():
    """Whether this interpreter's ``torch`` is Jittor rather than PyTorch."""
    try:
        import torch
    except Exception:
        return False
    if hasattr(torch, "_torch_compat_install_context"):
        return True
    origin = str(getattr(torch, "__file__", ""))
    return "jittor" in origin or not hasattr(torch, "_C")


def _distributions_available(names):
    import importlib.util

    for name in names:
        try:
            if importlib.util.find_spec(name) is None:
                return False
        except (ImportError, ValueError):
            return False
    return True


def _run(python, runtime, case, output, weights=None, device="cpu"):
    command = [
        python, str(RUNNER), case, str(output),
        "--runtime", runtime, "--device", device,
    ]
    if weights is not None:
        command += ["--weights", str(weights)]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=1800,
    )
    marker = "ECOSYSTEM_RESULT "
    for line in completed.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker):]), completed.stdout
    raise AssertionError(
        "runner failed for {} under {}:\n{}".format(case, python, completed.stdout[-4000:])
    )


def _divergence(actual, expected, floor):
    """Largest deviation, measured against a scale that cannot collapse to zero.

    Some gradients are mathematically zero -- an attention key bias, for
    instance, cancels inside the softmax -- so both runtimes return float noise
    around 1e-8.  Dividing by that tensor's own maximum turns the noise into a
    huge ratio and reports a defect that is not there.  ``floor`` carries the
    scale of the whole comparison, so a tensor is only judged against a
    magnitude that is meaningful for this model.
    """
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    scale = max(float(np.abs(expected).max()), floor, 1e-6)
    return float(np.abs(actual - expected).max() / scale)


def _comparison_floor(reference, keys):
    """A small fraction of the largest reference magnitude in the comparison."""
    magnitudes = [float(np.abs(reference[key]).max()) for key in keys]
    return 1e-3 * max(magnitudes + [0.0])


class EcosystemComparison(unittest.TestCase):
    """The two-interpreter comparison itself, without any case selection.

    Kept separate from the classes below so another module can reuse the
    harness for a different set of cases without inheriting these test methods.
    """

    forward_tolerance = 2e-3
    backward_tolerance = 1e-2

    device = "cpu"

    def _compare(self, case):
        _builder, requirements = _ecosystem_cases.CASES[case]
        if not _distributions_available(requirements):
            self.skipTest("missing {}".format(", ".join(requirements)))

        with tempfile.TemporaryDirectory(prefix="jittor-ecosystem-") as directory:
            root = Path(directory)
            torch_output = root / "torch.npz"
            jittor_output = root / "jittor.npz"

            torch_report, torch_log = _run(
                REAL_TORCH_PYTHON, "torch", case, torch_output, device=self.device
            )
            weights = root / "torch.weights.npz"
            self.assertTrue(weights.exists(), torch_log[-2000:])
            jittor_report, _jittor_log = _run(
                sys.executable,
                "jittor",
                case,
                jittor_output,
                weights=weights,
                device=self.device,
            )

            reference = np.load(torch_output)
            candidate = np.load(jittor_output)

            missing = sorted(set(reference.files) - set(candidate.files))
            self.assertEqual(missing, [], "{}: Jittor produced no {}".format(case, missing))

            forward_error = _divergence(
                candidate["__output__"],
                reference["__output__"],
                _comparison_floor(reference, ["__output__"]),
            )
            self.assertLess(
                forward_error,
                self.forward_tolerance,
                "{} forward diverged: {:.3e}".format(case, forward_error),
            )

            worst_name, worst_error = None, 0.0
            gradients = [key for key in reference.files if key != "__output__"]
            self.assertTrue(gradients, "{} produced no gradients to compare".format(case))
            gradient_floor = _comparison_floor(reference, gradients)
            for key in gradients:
                error = _divergence(candidate[key], reference[key], gradient_floor)
                if error > worst_error:
                    worst_name, worst_error = key, error
            self.assertLess(
                worst_error,
                self.backward_tolerance,
                "{} gradient {} diverged: {:.3e}".format(case, worst_name, worst_error),
            )

            ratio = jittor_report["seconds"] / max(torch_report["seconds"], 1e-9)
            print(
                "[speed/{}] {}: torch {:.4f}s jittor {:.4f}s ratio {:.2f}x "
                "({} gradients compared)".format(
                    self.device,
                    case,
                    torch_report["seconds"],
                    jittor_report["seconds"],
                    ratio,
                    len(gradients),
                )
            )
            if SPEED_RATIO:
                self.assertLessEqual(
                    ratio,
                    float(SPEED_RATIO),
                    "{} is {:.2f}x slower than PyTorch".format(case, ratio),
                )


@unittest.skipUnless(REAL_TORCH_PYTHON, "REAL_TORCH_PYTHON is not configured")
@unittest.skipUnless(_torch_shim_is_active(), "this interpreter does not run torch as Jittor")
class EcosystemParity(EcosystemComparison):
    """Compare a downstream library's numbers between PyTorch and Jittor."""

    def test_transformers_gpt2(self):
        self._compare("transformers_gpt2")

    def test_transformers_llama(self):
        self._compare("transformers_llama")

    def test_transformers_bert(self):
        self._compare("transformers_bert")

    def test_transformers_vit(self):
        self._compare("transformers_vit")

    def test_diffusers_unet2d(self):
        self._compare("diffusers_unet2d")

    def test_transformers_t5(self):
        self._compare("transformers_t5")

    def test_transformers_whisper(self):
        self._compare("transformers_whisper")

    def test_diffusers_dit(self):
        self._compare("diffusers_dit")

    def test_peft_lora_llama(self):
        self._compare("peft_lora_llama")

    def test_mmcv_conv_module(self):
        self._compare("mmcv_conv_module")

    def test_mmengine_base_module(self):
        self._compare("mmengine_base_module")

    def test_ms_swift_lora_llama(self):
        self._compare("ms_swift_lora_llama")


def _cuda_is_available():
    """Whether this Jittor build can actually execute on a GPU."""
    try:
        import jittor as jt
    except Exception:
        return False
    return bool(jt.has_cuda)


@unittest.skipUnless(REAL_TORCH_PYTHON, "REAL_TORCH_PYTHON is not configured")
@unittest.skipUnless(_torch_shim_is_active(), "this interpreter does not run torch as Jittor")
@unittest.skipUnless(_cuda_is_available(), "CUDA is unavailable")
class EcosystemParityCUDA(EcosystemParity):
    """The same comparison with both runtimes executing on the GPU."""

    device = "cuda"
    # Accelerator kernels pick different accumulation orders than the CPU
    # reference implementations these libraries were written against.
    forward_tolerance = 5e-3
    backward_tolerance = 2e-2


if __name__ == "__main__":
    unittest.main()

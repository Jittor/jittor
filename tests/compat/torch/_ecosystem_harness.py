"""Two-interpreter comparison harness for downstream-library cases.

Real PyTorch and the Jittor shim both claim the ``torch`` namespace, so the
same case runs in two interpreters: ``REAL_TORCH_PYTHON`` produces the weights
and reference values, this interpreter recomputes from the same weights.
``EcosystemComparison`` owns the comparison; the test modules
(``test_ecosystem_parity``, ``test_ecosystem_speed``) own case selection and
the gates.

Configuration
-------------

``REAL_TORCH_PYTHON``
    Interpreter whose ``import torch`` is an independent binary PyTorch.  The
    tests skip when it is unset, because a comparison against Jittor's own
    ``torch`` shim would be self-referential and would prove nothing.

``JITTOR_ECOSYSTEM_PACKAGE_SITE``
    Optional site-packages directory for downstream libraries. When omitted,
    the harness derives it from this interpreter's installed Transformers.
    Both runtimes claim their independent ``torch`` namespace first, then use
    this same directory for the libraries under comparison.

``JITTOR_ECOSYSTEM_SPEED_RATIO``
    Optional upper bound on ``jittor_seconds / torch_seconds``.  The wall-clock
    numbers are always reported; they are only asserted when this is set, since
    a shared machine makes an unconditional timing gate flaky.

``JITTOR_ECOSYSTEM_TF32``
    CUDA precision policy for both runtimes. It defaults to enabled and controls
    matmul and cuDNN convolution together; the reports must agree on the state.

``JITTOR_ECOSYSTEM_CUDNN_BENCHMARK``
    Optional CUDA convolution autotuning switch. It defaults to disabled and is
    applied to both runtimes for controlled algorithm-selection experiments.
"""

import importlib.util
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


def _package_site():
    configured = os.environ.get("JITTOR_ECOSYSTEM_PACKAGE_SITE", "").strip()
    if configured:
        site = Path(configured).expanduser().resolve()
        if not site.is_dir():
            raise RuntimeError(
                "JITTOR_ECOSYSTEM_PACKAGE_SITE is not a directory: {}".format(site)
            )
        return str(site)
    spec = importlib.util.find_spec("transformers")
    origin = getattr(spec, "origin", None)
    if not origin:
        return ""
    return str(Path(origin).resolve().parents[1])


PACKAGE_SITE = _package_site()


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


def _cuda_is_available():
    """Whether this Jittor build can actually execute on a GPU."""
    try:
        import jittor as jt
    except Exception:
        return False
    return bool(jt.has_cuda)


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
    environment = os.environ.copy()
    if PACKAGE_SITE:
        environment["JITTOR_ECOSYSTEM_PACKAGE_SITE"] = PACKAGE_SITE
    completed = subprocess.run(
        command,
        env=environment,
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

    Kept separate from the classes in the test modules so each of them can
    reuse the harness for a different set of cases without inheriting the
    others' test methods.
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

            # Both runtimes report where they actually ran. Jittor enables CUDA
            # by default when a GPU is present, so a CPU run that forgets to
            # turn it off silently compares an accelerator against a CPU.
            for label, report in (("torch", torch_report), ("jittor", jittor_report)):
                self.assertEqual(
                    report.get("device"),
                    self.device,
                    "{}: {} ran on {}, not {}".format(
                        case, label, report.get("device"), self.device
                    ),
                )
                if PACKAGE_SITE:
                    self.assertEqual(
                        report.get("package_site"),
                        PACKAGE_SITE,
                        "{}: {} used a different downstream package site".format(
                            case, label
                        ),
                    )
            self.assertEqual(
                torch_report.get("dependencies"),
                jittor_report.get("dependencies"),
                "{} used different downstream dependency versions or origins".format(case),
            )
            self.assertEqual(
                torch_report.get("tf32"),
                jittor_report.get("tf32"),
                "{} used different CUDA TF32 policies".format(case),
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

"""Compare the actual downstream-owned MMEngine training trajectory."""

from contextlib import ExitStack
from pathlib import Path
import tempfile

import numpy as np

from _helpers.child_process import PYTHON
from _ecosystem_harness import REAL_TORCH_PYTHON, _comparison_floor, _divergence, _run_command
from _ecosystem_openmmlab_cases import PARAMETERS_BY_CASE, BUFFER_NAMES_BY_CASE

ROOT = Path(__file__).resolve().parents[3]
PROBE = ROOT / "agent/skills/torch-compat-repo-runbook/scripts/openmmlab_training_probe.py"


def _training_run(python, runtime, case, output, device, weights=None):
    command = [
        python,
        str(PROBE),
        "--root",
        str(ROOT),
        "--case",
        case,
        "--runtime",
        runtime,
        "--device",
        device,
        "--output",
        str(output),
        "--steps",
        "3",
        "--learning-rate",
        "0.01",
    ]
    if weights is not None:
        command += ["--weights", str(weights)]
    return _run_command(python, case, command)


def compare_training(test, case):
    with tempfile.TemporaryDirectory(
        prefix="jittor-openmmlab-training-"
    ) as directory, ExitStack() as stack:
        root = Path(directory)
        reference_report, _ = _training_run(
            REAL_TORCH_PYTHON, "torch", case, root / "torch.npz", test.device
        )
        candidate_report, _ = _training_run(
            PYTHON,
            "jittor",
            case,
            root / "jittor.npz",
            test.device,
            weights=root / "torch.weights.npz",
        )
        assert_training_reports(test, case, reference_report, candidate_report)
        reference = stack.enter_context(np.load(root / "torch.npz"))
        candidate = stack.enter_context(np.load(root / "jittor.npz"))
        assert_training_arrays(test, case, reference, candidate)


def assert_training_reports(test, case, reference, candidate):
    for label, report in (("oracle", reference), ("candidate", candidate)):
        test.assertEqual(report.get("case"), case, label)
        test.assertEqual(report.get("device"), test.device, label)
        test.assertEqual(report.get("steps"), 3, label)
        test.assertEqual(report.get("learning_rate"), 0.01, label)
        test.assertEqual(report.get("seed"), 0, label)
        test.assertEqual(report.get("mode"), "train", label)
        test.assertEqual(report.get("runtime"), "torch" if label == "oracle" else "jittor", label)
        test.assertEqual(
            report.get("trainable_parameters"), sorted(PARAMETERS_BY_CASE[case]), label
        )
        test.assertEqual(report.get("frozen_parameters"), [], label)
        test.assertEqual(report.get("input_gradient_names"), ["sample"], label)
        test.assertEqual(report.get("target_requires_grad"), False, label)
        test.assertEqual(report.get("buffer_names"), sorted(BUFFER_NAMES_BY_CASE[case]), label)
        protocol = report.get("training_protocol", {})
        test.assertEqual(protocol.get("owner"), "mmengine.model.BaseModel.train_step", label)
        test.assertEqual(protocol.get("wrapper"), "mmengine.optim.OptimWrapper", label)
        test.assertEqual(
            protocol.get("wrapper_built_by"), "mmengine.registry.OPTIM_WRAPPERS.build", label
        )
        test.assertEqual(protocol.get("optimizer"), "torch.optim.SGD", label)
        test.assertEqual(protocol.get("momentum"), 0.9, label)
        test.assertEqual(protocol.get("clip_grad"), {"max_norm": 1.0}, label)
        test.assertEqual(protocol.get("gradient_stage"), "after_clip_before_sgd_step", label)
        test.assertEqual(protocol.get("optimizer_updates"), 3, label)
        test.assertEqual(protocol.get("loss_forward_calls"), 3, label)
        test.assertEqual(
            protocol.get("log_keys"), ["loss", "loss_regression", "loss_regularization"], label
        )
    for key in (
        "dependencies",
        "tf32",
        "runtime_conditions",
        "weights_sha256",
        "buffer_names",
        "jittor_commit",
        "probe_sha256",
        "training_protocol",
    ):
        test.assertEqual(reference.get(key), candidate.get(key), key)
    test.assertEqual(candidate.get("fallback_count"), 0)
    test.assertEqual(candidate.get("fallback_policy"), "error")


def assert_training_arrays(test, case, reference, candidate):
    names = PARAMETERS_BY_CASE[case]
    required = set()
    for step in range(3):
        prefix = "step/%d/" % step
        required.update(prefix + "buffer::" + name for name in BUFFER_NAMES_BY_CASE[case])
        required.update(
            prefix + name
            for name in (
                "output",
                "loss",
                "loss_regression",
                "loss_regularization",
                "input::sample",
                "target",
                "ingrad::sample",
            )
        )
        required.update(
            prefix + kind + name
            for name in names
            for kind in ("grad::", "parameter::", "momentum::")
        )
    for label, arrays in (("oracle", reference), ("candidate", candidate)):
        test.assertEqual(
            sorted(required - set(arrays.files)), [], label + " missing training tensors"
        )
    test.assertEqual(set(reference.files), set(candidate.files), "trajectory tensor keys differ")
    for key in reference.files:
        test.assertEqual(reference[key].shape, candidate[key].shape, key + " shape differs")
        test.assertEqual(reference[key].dtype, candidate[key].dtype, key + " dtype differs")
        if key.endswith("buffer::block1.bn.num_batches_tracked"):
            test.assertEqual(reference[key].shape, (), key + " must be a scalar counter")
            test.assertEqual(reference[key].dtype, np.dtype("int64"), key + " must be int64")
        test.assertTrue(np.isfinite(reference[key]).all(), key + " oracle must be finite")
        test.assertTrue(np.isfinite(candidate[key]).all(), key + " candidate must be finite")
    for step in range(3):
        prefix = "step/%d/" % step
        keys = [key for key in reference.files if key.startswith(prefix)]
        gradients = [key for key in keys if "grad::" in key]
        gradient_floor = _comparison_floor(reference, gradients)
        for key in keys:
            if key.endswith("input::sample") or key.endswith("target"):
                np.testing.assert_array_equal(candidate[key], reference[key], err_msg=key)
                continue
            if not reference[key].size:
                continue  # Equal exact empty shapes have already been checked.
            is_gradient = key in gradients
            floor = gradient_floor if is_gradient else _comparison_floor(reference, [key])
            error = _divergence(candidate[key], reference[key], floor)
            tolerance = test.backward_tolerance if is_gradient else test.forward_tolerance
            test.assertLess(error, tolerance, "{} {} diverged: {:.3e}".format(case, key, error))
    print(
        "[train-step/{}] {}: real upstream train_step updates=3, {} tensors compared".format(
            test.device, case, len(reference.files)
        )
    )

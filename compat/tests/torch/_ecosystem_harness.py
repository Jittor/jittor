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
    Optional site-packages directory for downstream libraries in the Jittor
    interpreter. When omitted, the harness derives it from this interpreter's
    installed Transformers. Both runtimes claim their independent ``torch``
    namespace before loading these libraries.

``JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE``
    Optional independent site-packages directory for the real-PyTorch
    interpreter. This is required when the two interpreters have different
    CPython ABIs and the dependency tree includes ABI-specific extensions.

``JITTOR_ECOSYSTEM_PACKAGE_SITE_CROSS_ABI``
    Explicitly allow that directory in both interpreters when their CPython
    minor versions differ. Use this only for a dependency tree whose compiled
    modules use a compatible stable ABI; otherwise each interpreter must use
    its own packages.

``JITTOR_ECOSYSTEM_SPEED_RATIO``
    Optional upper bound on ``jittor_seconds / torch_seconds``.  The wall-clock
    numbers are always reported; they are only asserted when this is set, since
    a shared machine makes an unconditional timing gate flaky.

``JITTOR_ECOSYSTEM_TF32``
    CUDA precision policy for both runtimes. It defaults to enabled and controls
    matmul and cuDNN convolution together; the reports must agree on the state,
    including the matmul tier and not only the boolean.

``JITTOR_ECOSYSTEM_CUDNN_BENCHMARK``
    Optional CUDA convolution autotuning switch. It defaults to disabled and is
    applied to both runtimes for controlled algorithm-selection experiments.
"""

from _helpers import capability as _test_capability

import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from _helpers.child_process import PYTHON, run_python_child

import _ecosystem_cases


RUNNER = Path(__file__).resolve().parent / "_ecosystem_runner.py"

REAL_TORCH_PYTHON = os.environ.get("REAL_TORCH_PYTHON", "").strip()

SPEED_RATIO = os.environ.get("JITTOR_ECOSYSTEM_SPEED_RATIO", "").strip()

#: Timed repeats per case. The runner reports the fastest of them, which is the
#: right statistic -- interference can only make a sample slower -- but three
#: samples do not pin it down on a shared machine: the same PyTorch CPU case
#: has come back 25% apart between two whole-suite runs, which is far wider
#: than the ratios being judged. More samples cost only wall clock.
REPEATS = os.environ.get("JITTOR_ECOSYSTEM_REPEATS", "").strip()


def _enabled(name):
    value = os.environ.get(name, "").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def _configured_package_site(name):
    configured = os.environ.get(name, "").strip()
    if configured:
        site = Path(configured).expanduser().resolve()
        if not site.is_dir():
            raise RuntimeError(
                "{} is not a directory: {}".format(name, site)
            )
        return str(site)
    return ""


def _package_site():
    configured = _configured_package_site("JITTOR_ECOSYSTEM_PACKAGE_SITE")
    if configured:
        return configured
    spec = importlib.util.find_spec("transformers")
    origin = getattr(spec, "origin", None)
    if not origin:
        return ""
    return str(Path(origin).resolve().parents[1])


PACKAGE_SITE = _package_site()
REFERENCE_PACKAGE_SITE = _configured_package_site(
    "JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE"
)


def _reference_shares_this_abi():
    """Whether ``REAL_TORCH_PYTHON`` can import this interpreter's packages.

    Both sides are made to import the downstream libraries from one site
    directory, so that a parity failure is a Jittor difference and not a version
    difference -- the comparison asserts the two runs report identical
    dependency versions and origins. A site built for one CPython version cannot
    be imported by another, though: its extension modules carry an ABI tag, so
    the reference interpreter fails on the first compiled import (``regex`` is
    the one transformers reaches first) with an error that says nothing about
    the real problem. When the versions differ, each side imports its own copy
    instead and only the dependency *versions* are required to agree.
    """
    if not REAL_TORCH_PYTHON or not PACKAGE_SITE:
        return True
    try:
        completed = subprocess.run(
            [REAL_TORCH_PYTHON, "-c",
             "import sys; print('%d.%d' % sys.version_info[:2])"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if completed.returncode != 0:
        return False
    theirs = completed.stdout.decode("utf-8", "replace").strip()
    return theirs == "%d.%d" % sys.version_info[:2]


REFERENCE_ABI_MATCHES = _reference_shares_this_abi()
REFERENCE_SHARES_PACKAGE_SITE = (
    not REFERENCE_PACKAGE_SITE
    and bool(PACKAGE_SITE)
    and (
        REFERENCE_ABI_MATCHES
        or _enabled("JITTOR_ECOSYSTEM_PACKAGE_SITE_CROSS_ABI")
    )
)


def _runner_package_site(python):
    if python == PYTHON:
        return PACKAGE_SITE
    if REFERENCE_PACKAGE_SITE:
        return REFERENCE_PACKAGE_SITE
    if REFERENCE_SHARES_PACKAGE_SITE:
        return PACKAGE_SITE
    return ""


def _versions(report):
    """Just the version of each downstream dependency, without its origin."""
    return {
        name: entry.get("version")
        for name, entry in (report.get("dependencies") or {}).items()
    }


def _torch_shim_is_active():
    """Whether this interpreter's ``torch`` is Jittor rather than PyTorch."""
    try:
        # The deployed torch facade imports Jittor internally. Importing that
        # facade first leaves a partially initialized ``torch`` in sys.modules,
        # which the fail-closed shim installer must reject. Match the runner's
        # Jittor-first activation order when shim mode was explicitly requested.
        if _enabled("JITTOR_TORCH_SHIM"):
            import jittor  # noqa: F401
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
    return bool(_test_capability.check_accelerator('cuda', backend=jt).enabled and not _test_capability.check_accelerator('acl', backend=jt).enabled)


def _npu_is_available():
    """Whether this Jittor build can actually execute through ACL."""
    try:
        import jittor as jt
    except Exception:
        return False
    return bool(_test_capability.check_accelerator('acl', backend=jt).enabled)


def _distributions_available(names):
    for name in names:
        try:
            if PACKAGE_SITE:
                spec = importlib.machinery.PathFinder.find_spec(
                    name, [PACKAGE_SITE]
                )
            else:
                spec = importlib.util.find_spec(name)
            if spec is None:
                return False
        except (ImportError, ValueError):
            return False
    return True


def _run(python, runtime, case, output, weights=None, device="cpu", repeats=None):
    command = [
        python, str(RUNNER), case, str(output),
        "--runtime", runtime, "--device", device,
    ]
    if repeats:
        command += ["--repeats", str(repeats)]
    if weights is not None:
        command += ["--weights", str(weights)]
    environment = os.environ.copy()
    package_site = _runner_package_site(python)
    if package_site:
        environment["JITTOR_ECOSYSTEM_PACKAGE_SITE"] = package_site
    else:
        environment.pop("JITTOR_ECOSYSTEM_PACKAGE_SITE", None)
    if python == PYTHON:
        # The Jittor side must run *this* checkout, so it is pinned.
        completed = run_python_child(
            command[1:], env=environment, inherit=False, merge_stderr=True,
            timeout=1800)
    else:
        # The oracle is a different interpreter with its own real PyTorch
        # installation. Pinning this checkout onto it is exactly what the
        # comparison must not do, so it is launched unpinned and on purpose.
        # Test invocations commonly inherit Jittor's source and shim
        # variables; clear them so the oracle cannot import the deployed
        # facade and accidentally compare Jittor with itself.
        environment["PYTHONPATH"] = ""
        for name in (
            "JITTOR_SOURCE_ROOT", "JITTOR_HOME", "JITTOR_TORCH_CACHE_ROOT",
            "JITTOR_TORCH_SHIM", "JITTOR_TORCH_KEEP_HOME", "JT_BACKEND",
            "JT_USE_CUDA", "JT_BUILD_NVCC_PATH", "use_cuda", "nvcc_path",
        ):
            environment.pop(name, None)
        completed = subprocess.run(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            # The runner prints Jittor's own logging, which is not ASCII.
            encoding="utf-8",
            errors="replace",
            timeout=1800,
        )
    return _result_from_stdout(completed.stdout, case, python), completed.stdout


def _result_from_stdout(stdout, case, python):
    """The runner's JSON payload, wherever on its line the marker landed.

    Searched with `find`, not `startswith`: anything the child prints without a
    trailing newline pushes the marker into the middle of a line. On a 384-core
    machine numexpr prints exactly such a warning -- it caps its pool at 64
    threads and says so -- and a run that had produced its measurement was
    reported as "runner failed". A false red on a measurement gate is worse
    than a late true red: it teaches people to distrust the gate that also
    checks the numbers.
    """
    marker = "ECOSYSTEM_RESULT "
    for line in stdout.splitlines():
        position = line.find(marker)
        if position != -1:
            return json.loads(line[position + len(marker):])
    raise AssertionError(
        "runner failed for {} under {}:\n{}".format(case, python, stdout[-4000:])
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
    if actual.shape != expected.shape:
        raise AssertionError(
            "comparison shape mismatch: {} != {}".format(actual.shape, expected.shape)
        )
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError("comparison contains non-finite values")
    scale = max(float(np.abs(expected).max()), floor, 1e-6)
    return float(np.abs(actual - expected).max() / scale)


def _comparison_floor(reference, keys):
    """A small fraction of the largest reference magnitude in the comparison."""
    magnitudes = [float(np.abs(reference[key]).max()) for key in keys]
    return 1e-3 * max(magnitudes + [0.0])


def _validate_npu_evidence(reference, candidate, trainable, frozen, input_grad_na):
    """Validate one synchronized phase; shared by single/multistep workloads."""
    if not isinstance(reference, dict) or not isinstance(candidate, dict):
        raise AssertionError("missing NPU inventory")
    inventories = []
    for label, report in (("oracle", reference), ("candidate", candidate)):
        identity = report.get("backend_identity")
        tensors = report.get("tensors")
        if not isinstance(identity, dict) or not isinstance(tensors, dict) or not tensors:
            raise AssertionError(label + ": missing backend identity/tensor inventory")
        if type(identity.get("device_count")) is not int or identity["device_count"] != 1:
            raise AssertionError(label + ": expected one visible NPU")
        if not isinstance(identity.get("module"), str) or not identity["module"]:
            raise AssertionError(label + ": missing runtime module provenance")
        if label == "oracle":
            if identity.get("runtime") != "torch_npu" or not identity.get("torch_module"):
                raise AssertionError("oracle is not native torch_npu")
        elif (identity.get("runtime") != "jittor" or identity.get("build_backend") != "acl"
              or "acl" not in identity.get("registered_backends", [])):
            raise AssertionError("candidate is not an ACL provider")
        input_dtypes = report.get("input_dtypes")
        if not isinstance(input_dtypes, dict) or not input_dtypes:
            raise AssertionError(label + ": missing input dtype metadata")
        if not set(input_grad_na) <= set(input_dtypes):
            raise AssertionError(label + ": unknown non-differentiable input")
        if set(trainable) & set(frozen):
            raise AssertionError("trainable and frozen parameter sets overlap")
        expected = {"primary_output"}
        expected.update("parameter::" + name for name in list(trainable) + list(frozen))
        expected.update("grad::" + name for name in trainable)
        expected.update("input::" + name for name in input_dtypes)
        expected.update("ingrad::" + name for name in set(input_dtypes) - set(input_grad_na))
        # Buffer completeness is owned by runner.named_buffers enumeration;
        # compare its exact names across runtimes below rather than guessing.
        expected.update(name for name in tensors if name.startswith("buffer::"))
        if set(tensors) != expected:
            raise AssertionError(label + ": incomplete or extra tensor inventory")
        for name, tensor in tensors.items():
            if not isinstance(tensor, dict):
                raise AssertionError(label + ": malformed tensor " + name)
            shape, dtype = tensor.get("shape"), tensor.get("dtype")
            if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
                raise AssertionError(label + ": invalid tensor shape " + name)
            if not isinstance(dtype, str) or not dtype:
                raise AssertionError(label + ": missing tensor dtype " + name)
            if type(tensor.get("device_id")) is not int or tensor["device_id"] != 0:
                raise AssertionError(label + ": wrong device index " + name)
            if label == "oracle":
                if tensor.get("device_type") != "npu":
                    raise AssertionError("oracle CPU/non-NPU tensor: " + name)
            else:
                if type(tensor.get("placement_backend")) is not int or tensor["placement_backend"] not in (-1, 2):
                    raise AssertionError("candidate non-ACL placement: " + name)
                location = tensor.get("location")
                if location != "device":
                    if not (location == "none" and 0 in shape and
                            tensor.get("residency_exception") == "zero-sized tensor has no allocation"):
                        raise AssertionError("candidate non-device residency: " + name)
                elif "residency_exception" in tensor:
                    raise AssertionError("unexpected residency exception: " + name)
            if name.startswith("input::") and input_dtypes[name[7:]] != dtype:
                raise AssertionError(label + ": inconsistent input dtype " + name)
        if report.get("primary_dtype") != tensors["primary_output"]["dtype"]:
            raise AssertionError(label + ": inconsistent primary dtype")
        inventories.append(tensors)
    if reference["input_dtypes"] != candidate["input_dtypes"]:
        raise AssertionError("NPU input dtypes differ")
    if set(inventories[0]) != set(inventories[1]):
        raise AssertionError("NPU tensor inventory keys differ")
    for name in inventories[0]:
        for field in ("shape", "dtype"):
            if inventories[0][name][field] != inventories[1][name][field]:
                raise AssertionError("NPU tensor {} differs: {}".format(field, name))


class EcosystemComparison(unittest.TestCase):
    """The two-interpreter comparison itself, without any case selection.

    Kept separate from the classes in the test modules so each of them can
    reuse the harness for a different set of cases without inheriting the
    others' test methods.
    """

    forward_tolerance = 2e-3
    backward_tolerance = 1e-2

    device = "cpu"
    repeats = REPEATS

    def _compare_adamw3(self, reference, candidate, torch_report, jittor_report):
        names = torch_report["trainable_parameters"]
        self.assertEqual(len(names), 8)
        self.assertEqual(names, jittor_report["trainable_parameters"])
        for report in (torch_report, jittor_report):
            self.assertEqual(report.get("protocol"), "adamw3")
            self.assertEqual(report.get("steps"), 3)
            observations = report.get("step_observations", [])
            self.assertEqual(len(observations), 3)
            for step, observation in enumerate(observations):
                self.assertEqual(observation["step"], step)
                self.assertEqual(observation["trainable_parameters"], names)
                self.assertEqual(observation["device"], "npu")
                self.assertIn("fallback_count", observation)
                if report is torch_report:
                    self.assertIsNone(observation["fallback_count"])
                else:
                    self.assertIs(type(observation["fallback_count"]), int)
                    self.assertEqual(observation["fallback_count"], 0)
        self.assertEqual(torch_report["optimizer"], jittor_report["optimizer"])
        for step in range(3):
            native = torch_report["step_observations"][step]
            shim = jittor_report["step_observations"][step]
            self.assertEqual(native["loss_dtype"], "float32")
            self.assertEqual(shim["loss_dtype"], native["loss_dtype"])
            for phase in ("backward_npu_evidence", "update_npu_evidence"):
                a, b = native[phase], shim[phase]
                self.assertEqual(a["primary_dtype"], "float32")
                self.assertEqual(b["primary_dtype"], a["primary_dtype"])
                self.assertEqual(a["input_dtypes"], {"input_ids": "int64"})
                self.assertEqual(b["input_dtypes"], a["input_dtypes"])
                _validate_npu_evidence(
                    a, b, names, torch_report["frozen_parameters"],
                    torch_report["input_grad_not_applicable"])
        expected = {"initial::" + name for name in names}
        for step in range(3):
            prefix = "step::{}::".format(step)
            expected.add(prefix + "loss")
            for name in names:
                expected.update((prefix + "grad::" + name, prefix + "param::" + name,
                                 prefix + "delta::" + name))
        self.assertEqual(set(reference.files), expected)
        self.assertEqual(set(candidate.files), expected)
        for report, snapshot in ((torch_report, reference), (jittor_report, candidate)):
            self.assertEqual(set(report["trajectory_dtypes"]), expected)
            for key in expected:
                dtype = "float64" if "::delta::" in key else "float32"
                self.assertEqual(report["trajectory_dtypes"][key], dtype, key)
                self.assertEqual(str(snapshot[key].dtype), dtype, key)
        for name in names:
            self.assertTrue(np.array_equal(reference["initial::" + name],
                                           candidate["initial::" + name]), name)
        for step in range(3):
            prefix = "step::{}::".format(step)
            key = prefix + "loss"
            self.assertLess(_divergence(candidate[key], reference[key], 1e-6),
                            self.forward_tolerance, key)
            for kind, tolerance in (("grad::", self.backward_tolerance),
                                    ("param::", self.forward_tolerance)):
                keys = [prefix + kind + name for name in names]
                floor = _comparison_floor(reference, keys)
                for key in keys:
                    if kind == "grad::":
                        # Gradient scale is independent of parameters and updates.
                        scale = max(float(np.abs(reference[key]).max()), floor, 1e-12)
                        error = _divergence(candidate[key] / scale, reference[key] / scale, 1.0)
                    else:
                        error = _divergence(candidate[key], reference[key], floor)
                    self.assertLess(error, tolerance, key)
            delta_keys = [prefix + "delta::" + name for name in names]
            # Scale only against oracle updates, never initial parameter values.
            # No _divergence 1e-6 absolute floor: normalize first so tiny real
            # updates cannot disappear under the parameter comparison tolerance.
            delta_floor = max(_comparison_floor(reference, delta_keys), 1e-12)
            previous_prefix = "initial::" if step == 0 else "step::{}::param::".format(step - 1)
            for name, key in zip(names, delta_keys):
                for snapshot in (reference, candidate):
                    actual_delta = (snapshot[prefix + "param::" + name].astype("float64")
                                    - snapshot[previous_prefix + name].astype("float64"))
                    self.assertTrue(np.array_equal(snapshot[key], actual_delta),
                                    "inconsistent saved update: " + key)
                scale = max(float(np.abs(reference[key]).max()), delta_floor)
                error = _divergence(candidate[key] / scale, reference[key] / scale, 1.0)
                self.assertLess(error, self.backward_tolerance, "update delta " + key)
        print("[training/npu] ms_swift_lora_llama_adamw3: three loss/gradient/update steps matched")

    def _validate_lora_performance_evidence(self, oracle, shim):
        self.assertIsNone(oracle["fallback_count"])
        self.assertEqual(len(oracle["step_fallback_counts"]), oracle["timed_steps"])
        self.assertTrue(all(value is None for value in oracle["step_fallback_counts"]))
        self.assertIs(type(shim["fallback_count"]), int)
        self.assertEqual(shim["fallback_count"], 0)
        self.assertEqual(shim["fallback_policy"], "error")
        self.assertEqual(len(shim["step_fallback_counts"]), shim["timed_steps"])
        self.assertTrue(all(type(value) is int and value == 0
                            for value in shim["step_fallback_counts"]))
        for field in ("trainable_parameters", "frozen_parameters", "input_grad_not_applicable"):
            self.assertEqual(oracle[field], shim[field])
        for phase in ("warmup_npu_evidence", "npu_evidence"):
            _validate_npu_evidence(oracle[phase], shim[phase],
                oracle["trainable_parameters"], oracle["frozen_parameters"],
                oracle["input_grad_not_applicable"])

        self.assertEqual(len(oracle["loss_npu_evidence"]), 3 + oracle["timed_steps"])
        self.assertEqual(len(shim["loss_npu_evidence"]), len(oracle["loss_npu_evidence"]))
        for native_loss, candidate_loss in zip(oracle["loss_npu_evidence"], shim["loss_npu_evidence"]):
            inventories = []
            for report, loss in ((oracle, native_loss), (shim, candidate_loss)):
                self.assertEqual(loss["shape"], [])
                self.assertEqual(loss["dtype"], "float32")
                evidence = dict(report["npu_evidence"])
                evidence["tensors"] = dict(evidence["tensors"], primary_output=loss)
                evidence["primary_dtype"] = loss["dtype"]
                inventories.append(evidence)
            _validate_npu_evidence(*inventories, oracle["trainable_parameters"],
                oracle["frozen_parameters"], oracle["input_grad_not_applicable"])

    def _validate_lora_performance_artifacts(self, snapshot, report):
        names = report["trainable_parameters"]
        self.assertEqual(len(names), 88)
        expected = {"losses"} | {"final::" + name for name in names} | {"grad::" + name for name in names}
        self.assertEqual(set(snapshot.files), expected)
        self.assertEqual(set(report["artifact_dtypes"]), expected)
        for key in expected:
            self.assertEqual(report["artifact_dtypes"][key], "float32", key)
            self.assertEqual(str(snapshot[key].dtype), "float32", key)
            if key == "losses":
                shape = (3 + report["timed_steps"],)
            else:
                name = key.split("::", 1)[1]
                shape = tuple(report["npu_evidence"]["tensors"]["parameter::" + name]["shape"])
            self.assertEqual(tuple(snapshot[key].shape), shape, key)

    def _compare_lora_performance(self, reference, candidate, oracle, shim, oracle_log):
        self._validate_lora_performance_evidence(oracle, shim)
        self._validate_lora_performance_artifacts(reference, oracle)
        self._validate_lora_performance_artifacts(candidate, shim)
        # Known native CPU-fallback diagnostics are fatal; absence is not a
        # substitute for a universal native dispatch counter.
        lowered = oracle_log.lower()
        self.assertNotIn("fallback to run on the cpu", lowered)
        self.assertNotIn("fall back to cpu", lowered)
        for field in ("protocol", "warmup_steps", "timed_steps", "attention", "tuner",
                      "optimizer", "parameter_count", "trainable_parameter_count"):
            self.assertEqual(oracle[field], shim[field], field)
        # Config metadata can contain runtime-specific dtype/version objects;
        # compare every semantic dimension explicitly.
        for field in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                      "num_key_value_heads", "vocab_size", "max_position_embeddings",
                      "attention_dropout", "tie_word_embeddings", "use_cache"):
            self.assertEqual(oracle["config"][field], shim["config"][field], field)
        self.assertEqual(oracle["warmup_steps"], 3)
        for report in (oracle, shim):
            self.assertGreaterEqual(report["timed_steps"], 10)
            self.assertEqual(report["precision"]["dtype"], "float32")
            self.assertFalse(report["precision"]["allow_hf32"])
            self.assertEqual(report["precision"]["cube_math_type"], 0)
            stats = report["statistics"]
            self.assertEqual(len(stats["durations_seconds"]), report["timed_steps"])
            self.assertTrue(all(np.isfinite(x) and x > 0 for x in stats["durations_seconds"]))
            self.assertEqual(report["memory_end"]["units"], "bytes")
            self.assertGreater(report["memory_end"]["allocated"], 0)
            for phase in ("warmup_npu_evidence", "npu_evidence"):
                self.assertEqual(report[phase]["primary_dtype"], "float32")
                self.assertEqual(report[phase]["input_dtypes"], {"input_ids": "int64"})
        self.assertIsNone(shim["memory_end"]["peak_allocated"])
        self.assertIsNone(shim["memory_end"]["peak_reserved"])
        self.assertEqual(reference["losses"].shape, (3 + oracle["timed_steps"],))
        for key in reference.files:
            self.assertLess(_divergence(candidate[key], reference[key], 1e-6), 2e-2, key)
        ratio = shim["statistics"]["median_seconds"] / oracle["statistics"]["median_seconds"]
        print("[performance/npu] Swift LoRA 1.1B FP32: median ratio={:.3f}".format(ratio))
        if SPEED_RATIO:
            self.assertLessEqual(ratio, float(SPEED_RATIO))

    def _compare(self, case):
        _builder, requirements = _ecosystem_cases.CASES[case]
        if not _distributions_available(requirements):
            self.skipTest("missing {}".format(", ".join(requirements)))

        with tempfile.TemporaryDirectory(prefix="jittor-ecosystem-") as directory:
            root = Path(directory)
            torch_output = root / "torch.npz"
            jittor_output = root / "jittor.npz"

            torch_report, torch_log = _run(
                REAL_TORCH_PYTHON, "torch", case, torch_output,
                device=self.device, repeats=self.repeats,
            )
            weights = root / "torch.weights.npz"
            self.assertTrue(weights.exists(), torch_log[-2000:])
            jittor_report, _jittor_log = _run(
                PYTHON,
                "jittor",
                case,
                jittor_output,
                weights=weights,
                device=self.device,
                repeats=self.repeats,
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
                expected_site = _runner_package_site(
                    PYTHON if label == "jittor" else REAL_TORCH_PYTHON
                )
                if expected_site:
                    self.assertEqual(
                        report.get("package_site"),
                        expected_site,
                        "{}: {} used a different downstream package site".format(
                            case, label
                        ),
                    )
            if REFERENCE_SHARES_PACKAGE_SITE:
                self.assertEqual(
                    torch_report.get("dependencies"),
                    jittor_report.get("dependencies"),
                    "{} used different downstream dependency versions or origins"
                    .format(case),
                )
            else:
                # The two interpreters are different CPython versions, so they
                # cannot share one site directory -- its extension modules carry
                # an ABI tag. Each imports its own copy, and what has to match is
                # the version: the origins are expected to differ.
                self.assertEqual(
                    _versions(torch_report), _versions(jittor_report),
                    "{} used different downstream dependency versions".format(case),
                )
            self.assertEqual(
                torch_report.get("tf32"),
                jittor_report.get("tf32"),
                "{} used different CUDA TF32 policies".format(case),
            )
            self.assertEqual(
                torch_report.get("runtime_conditions"),
                jittor_report.get("runtime_conditions"),
                "{} timed the runtimes with different thread counts, affinity, "
                "or precision policy".format(case),
            )
            for field in ("trainable_parameters", "frozen_parameters",
                          "input_grad_not_applicable", "output_structure"):
                self.assertIn(field, torch_report)
                self.assertIn(field, jittor_report)
                self.assertEqual(torch_report[field], jittor_report[field],
                                 "{}: different {}".format(case, field))
            self.assertEqual(jittor_report.get("fallback_policy"), "error")
            self.assertEqual(jittor_report.get("fallback_count"), 0)
            if self.device == "npu":
                backend = jittor_report.get("backend") or {}
                self.assertTrue(backend.get("has_acl"), "ACL was not detected")
                self.assertTrue(backend.get("use_acl"), "ACL dispatch was not enabled")
                self.assertTrue(backend.get("use_cuda"), "device dispatch was not enabled")
                _validate_npu_evidence(
                    torch_report.get("npu_evidence"), jittor_report.get("npu_evidence"),
                    torch_report["trainable_parameters"], torch_report["frozen_parameters"],
                    torch_report["input_grad_not_applicable"])

            reference = np.load(torch_output)
            candidate = np.load(jittor_output)

            missing = sorted(set(reference.files) - set(candidate.files))
            self.assertEqual(missing, [], "{}: Jittor produced no {}".format(case, missing))
            extra = sorted(set(candidate.files) - set(reference.files))
            self.assertEqual(extra, [], "{}: Jittor produced extra {}".format(case, extra))

            if case == "ms_swift_lora_llama_adamw3":
                return self._compare_adamw3(reference, candidate, torch_report, jittor_report)

            if case == "large_ms_swift_lora_llama_1b_train":
                return self._compare_lora_performance(reference, candidate, torch_report,
                                                      jittor_report, torch_log)

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

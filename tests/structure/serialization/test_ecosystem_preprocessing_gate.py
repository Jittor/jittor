"""Fail-closed runner exits and original Whisper preprocessing gate contracts."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
ECOSYSTEM = ROOT / "compat/tests/torch"


def _runner_with_completion(completed):
    tree = ast.parse((ECOSYSTEM / "_ecosystem_harness.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_run")
    namespace = {
        "json": json, "os": SimpleNamespace(environ={}),
        "RUNNER": "runner.py", "PYTHON": "candidate-python",
        "_runner_package_site": lambda python: "",
        "subprocess": SimpleNamespace(run=lambda *args, **kwargs: completed,
                                      PIPE=-1, STDOUT=-2),
        "run_python_child": lambda *args, **kwargs: completed,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "harness.py", "exec"), namespace)
    return namespace["_run"]


@pytest.mark.parametrize("python,runtime", [("oracle-python", "torch"), ("candidate-python", "jittor")])
def test_success_marker_cannot_hide_a_failed_runner_exit(python, runtime):
    completed = SimpleNamespace(
        returncode=1,
        stdout='ECOSYSTEM_RESULT {"device": "cpu"}\nprocess shutdown failed\n',
    )
    run = _runner_with_completion(completed)
    with pytest.raises(AssertionError, match="exited with code 1"):
        run(python, runtime, "openai_whisper", "unused.npz")


@pytest.mark.parametrize("python,runtime", [("oracle-python", "torch"), ("candidate-python", "jittor")])
def test_zero_exit_with_result_is_accepted(python, runtime):
    completed = SimpleNamespace(returncode=0, stdout='ECOSYSTEM_RESULT {"device": "cpu"}\n')
    run = _runner_with_completion(completed)
    report, output = run(python, runtime, "openai_whisper", "unused.npz")
    assert report == {"device": "cpu"}
    assert output == completed.stdout


def _strict_comparison():
    import numpy as np
    tree = ast.parse((ECOSYSTEM / "_ecosystem_harness.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_assert_strict_results")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "harness.py", "exec"), namespace)
    return namespace["_assert_strict_results"]


def _preprocessing_result():
    import numpy as np
    arrays = {"__output__": np.ones((80, 100), dtype="float32"),
              "ingrad::audio": np.ones((16000,), dtype="float32")}
    report = {"state_manifest": {}, "state_fingerprints": {},
              "required_parameter_gradients": [], "required_input_gradients": ["audio"]}
    return arrays, report


def test_preprocessing_can_be_parameterless_without_relaxing_model_coverage():
    import unittest
    arrays, report = _preprocessing_result()
    compare = _strict_comparison()
    with pytest.raises(AssertionError, match="no trainable parameters"):
        compare(unittest.TestCase(), arrays, arrays, report, report)
    compare(unittest.TestCase(), arrays, arrays, report, report, allow_parameterless=True)


@pytest.mark.parametrize("failure", ["missing", "nonfinite", "dtype", "shape"])
def test_parameterless_preprocessing_still_requires_valid_complete_gradients(failure):
    import numpy as np
    import unittest
    arrays, report = _preprocessing_result()
    candidate = {key: value.copy() for key, value in arrays.items()}
    if failure == "missing":
        del candidate["ingrad::audio"]
    elif failure == "nonfinite":
        candidate["ingrad::audio"][0] = np.nan
    elif failure == "dtype":
        candidate["ingrad::audio"] = candidate["ingrad::audio"].astype("float64")
    else:
        candidate["ingrad::audio"] = candidate["ingrad::audio"].reshape((1, 16000))
    with pytest.raises(AssertionError):
        _strict_comparison()(unittest.TestCase(), arrays, candidate, report, report,
                             allow_parameterless=True)


def test_preprocessing_case_calls_original_whisper_and_keeps_one_second_input(monkeypatch):
    import sys
    from types import ModuleType
    upstream = ModuleType("whisper")
    calls = []
    expected = object()
    def log_mel(audio, n_mels):
        calls.append((audio, n_mels))
        return expected
    upstream.log_mel_spectrogram = log_mel
    monkeypatch.setitem(sys.modules, "whisper", upstream)
    tree = ast.parse((ECOSYSTEM / "_ecosystem_cases.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_openai_whisper_log_mel")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "cases.py", "exec"), namespace)
    torch = SimpleNamespace(nn=SimpleNamespace(Module=object))
    model, spec = namespace["_openai_whisper_log_mel"](torch)
    waveform = object()
    assert model.forward(waveform) is expected
    assert calls == [(waveform, 80)]
    assert spec == {"audio": ("float32", (16000,), None)}


def _thread_configurer(value):
    tree = ast.parse((ECOSYSTEM / "_ecosystem_runner.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_configure_cpu_threads")
    namespace = {"os": SimpleNamespace(environ={} if value is None else {"OMP_NUM_THREADS": value})}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "runner.py", "exec"), namespace)
    return namespace["_configure_cpu_threads"]


@pytest.mark.parametrize("runtime,device,budget,expected", [
    ("torch", "cpu", "8", [8]), ("torch", "cpu", " 2 ", [2]),
    ("jittor", "cpu", "8", []), ("torch", "cuda", "8", []),
    ("torch", "cpu", "", []), ("torch", "cpu", None, []),
])
def test_cpu_oracle_applies_declared_threads_without_using_shim_setter(runtime, device, budget, expected):
    calls = []
    torch = SimpleNamespace(set_num_threads=calls.append)
    _thread_configurer(budget)(torch, runtime, device)
    assert calls == expected


@pytest.mark.parametrize("budget", ["0", "-1", "eight", "2.5"])
def test_invalid_cpu_thread_budget_is_rejected(budget):
    calls = []
    with pytest.raises(ValueError, match="positive integer"):
        _thread_configurer(budget)(SimpleNamespace(set_num_threads=calls.append), "torch", "cpu")
    assert calls == []

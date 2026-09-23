"""Whisper's CPU gate must reject source/version drift before parity runs."""

import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
COMMIT = "86098128c0b4f24f0e2aa2994de830614b474227"


def _nox_contract(**overrides):
    tree = ast.parse((ROOT / "noxfile.py").read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", "").startswith(("WHISPER_CPU_", "_WHISPER_CPU_"))
            for target in node.targets
        ):
            nodes.append(node)
        if isinstance(node, ast.FunctionDef) and node.name in (
            "_validate_whisper_cpu_oracle", "whisper_cpu",
        ):
            node.decorator_list = []
            nodes.append(node)
    namespace = {"Path": Path, "json": json, "os": SimpleNamespace(cpu_count=lambda: 8), "PYTEST": "pytest==7.4.4",
                 "PYTEST_TIMEOUT": "pytest-timeout==2.3.1", "SCIPY": "scipy==1.13.1",
                 "SETUPTOOLS": "setuptools==83.0.0", "WHEEL": "wheel==0.45.1"}
    namespace.update(overrides)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "noxfile.py", "exec"), namespace)
    return SimpleNamespace(**namespace)


def _baseline(oracle):
    return {
        "python": [3, 11, 16], "torch": "2.4.1+cpu", "numpy": "1.26.4",
        "whisper": "20250625", "torch_is_shim": False, "torch_has_binary": True,
        "torch_cuda": None, "prefix": str(oracle),
        "package_site": str(oracle / "lib/python3.11/site-packages"),
        "source": {"url": "https://github.com/openai/whisper.git",
                   "vcs_info": {"vcs": "git", "commit_id": COMMIT}},
    }


def test_official_exact_cpu_oracle_is_accepted(tmp_path):
    oracle = tmp_path / "oracle"
    _nox_contract()._validate_whisper_cpu_oracle(_baseline(oracle), oracle)


@pytest.mark.parametrize("difference", [
    "python", "torch", "numpy", "whisper", "shim", "binary", "cuda",
    "prefix", "site", "source", "commit", "vcs",
])
def test_source_version_or_oracle_drift_is_rejected_before_testing(tmp_path, difference):
    oracle = tmp_path / "oracle"
    report = _baseline(oracle)
    if difference in ("python", "torch", "numpy", "whisper"):
        report[difference] = "a different version"
    elif difference == "shim":
        report["torch_is_shim"] = True
    elif difference == "binary":
        report["torch_has_binary"] = False
    elif difference == "cuda":
        report["torch_cuda"] = "12.1"
    elif difference == "prefix":
        report["prefix"] = str(tmp_path / "shared-env")
    elif difference == "site":
        report["package_site"] = str(tmp_path / "other-whisper")
    elif difference == "source":
        report["source"]["url"] = "https://example.invalid/whisper.git"
    elif difference == "commit":
        # The published release can have the same version as the tested main
        # checkout. That version match is insufficient evidence of equal code.
        report["source"]["vcs_info"]["commit_id"] = "0" * 40
    else:
        report["source"]["vcs_info"]["vcs"] = "a different vcs"
    with pytest.raises((RuntimeError, ValueError)):
        _nox_contract()._validate_whisper_cpu_oracle(report, oracle)


def test_session_isolated_oracle_and_required_cpu_selection(tmp_path):
    calls, installations, runs = [], [], []
    oracle = tmp_path / "oracle"
    baseline = _baseline(oracle)

    class Session:
        def install(self, *arguments):
            installations.append(arguments)

        def run(self, *arguments, **options):
            calls.append((arguments, deepcopy(options)))
            if "-c" in arguments:
                return json.dumps(baseline)
            return ""

        def log(self, message):
            pass

    contract = _nox_contract(
        _session_env=lambda session, name: (tmp_path, {
            "PYTHONPATH": str(ROOT / "python"), "JITTOR_HOME": "shared-jit-cache"}),
        _run_pytest_once=lambda session, args, env, **kwargs: runs.append((args, env, kwargs)),
    )
    contract.whisper_cpu(Session())
    assert len(runs) == 1
    arguments, env, _ = runs[0]
    assert "compat/tests/torch/test_ecosystem_parity.py::OpenAIWhisperParity::test_openai_whisper" in arguments
    assert "compat/tests/torch/test_torch_sparse_metadata.py" in arguments
    assert "compat/tests/torch/test_whisper_training.py" in arguments
    assert "--confcutdir=compat/tests" in arguments
    assert "compat/tests/torch/test_ecosystem_parity.py::OpenAIWhisperParity::test_openai_whisper_log_mel" in arguments
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert env[name] == "8"
    for name in ("JITTOR_REQUIRE_REAL_TORCH", "JITTOR_REQUIRE_WHISPER", "JITTOR_TEST_REQUIRE_EXECUTION"):
        assert env[name] == "1"
    assert env["JITTOR_TEST_DEVICES"] == "cpu"
    assert env["JT_USE_CUDA"] == "0"
    assert env["JT_BUILD_NVCC_PATH"] == ""
    assert env["JT_BACKEND_FALLBACK"] == "error"
    assert env["JITTOR_ECOSYSTEM_SPEED_RATIO"] == ""
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["JITTOR_HOME"] == str(tmp_path / "jittor-home")
    assert env["REAL_TORCH_PYTHON"] == str(oracle / "bin/python")
    assert env["JITTOR_ECOSYSTEM_PACKAGE_SITE"] == baseline["package_site"]
    assert env["JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE"] == baseline["package_site"]
    source_installs = [args for args, _ in calls if contract.WHISPER_CPU_SOURCE in args]
    assert len(source_installs) == 1
    assert contract.WHISPER_CPU_SOURCE.endswith("@" + COMMIT)
    # Installing Whisper/PyTorch into the shim would invalidate the comparison.
    assert all("torch" not in requirement for group in installations for requirement in group)
    for args, options in calls:
        if args[0] == env["REAL_TORCH_PYTHON"]:
            assert options["env"]["PYTHONPATH"] == ""
            assert options["env"]["JITTOR_TORCH_SHIM"] == "0"
    constraints = (tmp_path / "whisper-constraints.txt").read_text(encoding="utf-8")
    assert "torch==2.4.1+cpu\n" in constraints
    assert "numpy==1.26.4\n" in constraints
    assert json.loads((tmp_path / "whisper-baseline.json").read_text()) == baseline


def test_workflow_has_an_independent_pinned_whisper_job():
    document = yaml.safe_load((ROOT / ".github/workflows/ecosystem.yml").read_text(encoding="utf-8"))
    existing = document["jobs"]["ecosystem"]
    assert any("torch==2.7.1" in step.get("run", "") for step in existing["steps"])
    whisper = document["jobs"]["whisper-cpu"]
    assert whisper["needs"] == "baseline"
    setup = [step for step in whisper["steps"] if step.get("uses", "").startswith("actions/setup-python@")]
    assert setup[0]["with"]["python-version"] == "3.11.16"
    command = next(step["run"] for step in whisper["steps"]
                   if "nox -s whisper_cpu" in step.get("run", ""))
    assert 'JITTOR_LAB_ROOT="${RUNNER_TEMP}/jittor-lab-whisper"' in command
    assert not whisper.get("continue-on-error", False)
    assert not any(step.get("continue-on-error", False) for step in whisper["steps"])

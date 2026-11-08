"""Required CUDA configuration must not become an optional skip."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def gate():
    root = Path(__file__).resolve().parents[2]
    path = root / "compat/tests/torch/test_ecosystem_flash_attention.py"
    spec = importlib.util.spec_from_file_location("flash_training_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_required_cpu_cannot_report_skip(gate, monkeypatch):
    monkeypatch.setenv("JITTOR_TEST_DEVICES", "cpu")
    monkeypatch.setenv("JITTOR_REQUIRE_FLASH_ATTENTION", "1")
    with pytest.raises(pytest.fail.Exception, match="actual CUDA"):
        gate._configured()


def test_missing_required_cuda_configuration_fails(gate, monkeypatch):
    monkeypatch.setenv("JITTOR_TEST_DEVICES", "cuda")
    monkeypatch.setenv("JITTOR_REQUIRE_FLASH_ATTENTION", "1")
    monkeypatch.delenv("REAL_TORCH_PYTHON", raising=False)
    with pytest.raises(pytest.fail.Exception, match="REAL_TORCH_PYTHON"):
        gate._configured()


def test_optional_cpu_reports_cuda_only_boundary(gate, monkeypatch):
    monkeypatch.setenv("JITTOR_TEST_DEVICES", "cpu")
    monkeypatch.delenv("JITTOR_REQUIRE_FLASH_ATTENTION", raising=False)
    with pytest.raises(pytest.skip.Exception, match="no CPU execution"):
        gate._configured()

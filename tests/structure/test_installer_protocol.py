"""Installer callback contracts can be checked without importing a runtime."""
import importlib.util
from pathlib import Path

import pytest


def _contracts():
    source = Path(__file__).resolve().parents[2] / "compat/torch/contracts.py"
    spec = importlib.util.spec_from_file_location("installer_protocol_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_installer_protocol_checks_signatures_without_executing_callbacks():
    api = _contracts()
    calls = []
    class Installer:
        def __call__(self, context):
            calls.append(context)
    api.validate_installer(Installer(), "callable_object")
    api.validate_installer(lambda context, optional=None: calls.append(context), "function")
    assert calls == []
    for value in (object(), lambda: None, lambda context, missing: None):
        with pytest.raises(TypeError, match="installer 'invalid'"):
            api.validate_installer(value, "invalid")

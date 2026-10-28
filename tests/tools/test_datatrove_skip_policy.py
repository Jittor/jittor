"""DataTrove absence is optional until the gate explicitly requires it."""

from pathlib import Path
import runpy
import sys

import pytest

from _helpers import pytest_policy


@pytest.mark.parametrize("setting", ("1", "true", "yes", "on"))
def test_required_datatrove_withdraws_environment_skip(monkeypatch, setting):
    reason = {"datatrove is not installed"}
    monkeypatch.delenv("JITTOR_REQUIRE_DATATROVE", raising=False)
    assert pytest_policy._environment_explains(reason)
    monkeypatch.setenv("JITTOR_REQUIRE_DATATROVE", setting)
    assert not pytest_policy._environment_explains(reason)
    assert pytest_policy._environment_explains({"nvcc is not installed"})


@pytest.mark.parametrize("setting", ("1", "true", "yes", "on"))
def test_required_datatrove_cannot_silently_skip(monkeypatch, setting):
    monkeypatch.setenv("JITTOR_REQUIRE_DATATROVE", setting)
    source = Path(__file__).resolve().parents[2] / "compat/tests/torch/test_datatrove_compat.py"
    with monkeypatch.context() as context:
        context.setitem(sys.modules, "datatrove", None)
        with pytest.raises(ModuleNotFoundError, match="datatrove is not installed"):
            runpy.run_path(str(source))


def test_missing_optional_datatrove_has_an_explicit_skip(monkeypatch):
    monkeypatch.delenv("JITTOR_REQUIRE_DATATROVE", raising=False)
    source = Path(__file__).resolve().parents[2] / "compat/tests/torch/test_datatrove_compat.py"
    with monkeypatch.context() as context:
        context.setitem(sys.modules, "datatrove", None)
        namespace = runpy.run_path(str(source))
    test = namespace["TestDataTroveCompat"]
    assert test.__unittest_skip__
    assert test.__unittest_skip_why__ == "datatrove is not installed"

"""The standalone budget report must use the same worker policy as nox."""

import importlib.util
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_reporter():
    path = ROOT / "tools" / "gate_budget_report.py"
    spec = importlib.util.spec_from_file_location("gate_budget_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_uses_nox_worker_override(monkeypatch):
    module = _load_reporter()
    monkeypatch.setenv("JITTOR_GATE_WORKERS", "2")
    output = StringIO()
    with redirect_stdout(output):
        # The checked-in measurements are intentionally over budget at two
        # workers; the non-zero result is still a valid report and must not
        # hide the worker-policy value.
        assert module.main(["--json"]) == 2
    report = json.loads(output.getvalue())
    assert report["configured_workers"] == 2
    assert report["workers"] == module.budget_report(configured_workers=2)["workers"]


def test_report_rejects_invalid_worker_override(monkeypatch):
    module = _load_reporter()
    monkeypatch.setenv("JITTOR_GATE_WORKERS", "many")
    with pytest.raises(SystemExit) as exc:
        module.main(["--json"])
    assert exc.value.code == 2

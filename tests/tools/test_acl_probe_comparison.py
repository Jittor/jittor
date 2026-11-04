"""Host-only checks for the manual ACL probe's result validation."""
import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "tools/probes/acl/xcheck/compare.py"
@pytest.fixture
def comparison():
    spec = importlib.util.spec_from_file_location("acl_probe_comparison", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result():
    return {"loss": 2.0, "total_grad_l2": 3.0, "grads": {"weight": 3.0}}


def test_comparison_accepts_roundoff(comparison):
    candidate = _result()
    candidate["loss"] += 1e-6
    assert comparison.compare(_result(), candidate)["loss"] > 0


@pytest.mark.parametrize("fault", ["missing", "extra", "nonfinite", "drift", "empty"])
def test_comparison_rejects_invalid_evidence(comparison, fault):
    reference, candidate = _result(), _result()
    if fault == "missing":
        candidate["grads"].clear()
    elif fault == "extra":
        candidate["grads"]["unexpected"] = 0.0
    elif fault == "nonfinite":
        candidate["loss"] = float("nan")
    elif fault == "drift":
        candidate["grads"]["weight"] = 5.0
    else:
        reference["grads"].clear()
        candidate["grads"].clear()
    with pytest.raises(ValueError):
        comparison.compare(reference, candidate)

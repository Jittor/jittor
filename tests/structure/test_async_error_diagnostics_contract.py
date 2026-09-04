from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs/testing/async-error-diagnostics.md"


def test_async_error_contract_states_ring_and_cuda_requirements():
    text = GUIDE.read_text(encoding="utf-8")
    for token in (
        "TraceData",
        "per-thread ring",
        "operator id/name",
        "Python file and line",
        "stream identity",
        "allocation-free",
        "not-found",
        "CUDA probe",
        "CUDA_VISIBLE_DEVICES=0",
        "test_async_error_location.py",
    ):
        assert token in text

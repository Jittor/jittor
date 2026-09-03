"""The speed harness must make its comparison conditions auditable."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "tests" / "compat" / "torch"


def _source(name):
    return (ROOT / name).read_text(encoding="utf-8")


def test_runner_reports_threads_affinity_and_precision():
    source = _source("_ecosystem_runner.py")
    assert '"runtime_threads"' in source
    assert '"affinity"' in source
    assert '"thread_env"' in source
    assert '"precision": tf32' in source


def test_harness_requires_both_runtime_conditions_to_match():
    source = _source("_ecosystem_harness.py")
    assert source.count('get("runtime_conditions")') == 2
    assert "different thread counts, affinity" in source


def test_speed_measurements_default_to_at_least_ten_repeats():
    source = _source("test_ecosystem_speed.py")
    assert 'JITTOR_ECOSYSTEM_REPEATS", "10"' in source
    assert "if value < 10:" in source

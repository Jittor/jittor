"""Run the aggregate mmdetection compatibility checks in an isolated process."""

from pathlib import Path

from _helpers.child_process import run_python_child


CHECK_SCRIPT = Path(__file__).with_name("_mmdet_ops_checks.py")


def test_mmdet_ops():
    result = run_python_child(
        [CHECK_SCRIPT],
        cwd=Path(__file__).resolve().parents[2],
        merge_stderr=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout

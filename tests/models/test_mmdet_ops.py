"""Run the aggregate mmdetection compatibility checks in an isolated process."""

from pathlib import Path
import subprocess
import sys


CHECK_SCRIPT = Path(__file__).with_name("_mmdet_ops_checks.py")


def test_mmdet_ops():
    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT)],
        cwd=str(Path(__file__).resolve().parents[2]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout

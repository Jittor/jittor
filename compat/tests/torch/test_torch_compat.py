"""Run the legacy aggregate compatibility checks outside pytest's process."""

from pathlib import Path

import pytest

from _helpers.child_process import run_python_child


CHECK_SCRIPT = Path(__file__).with_name("_torch_compat_checks.py")


@pytest.mark.timeout(1800)
def test_torch_compat():
    result = run_python_child(
        [CHECK_SCRIPT],
        cwd=Path(__file__).resolve().parents[3],
        merge_stderr=True,
        timeout=1500,
    )
    assert result.returncode == 0, result.stdout

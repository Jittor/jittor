"""Run the legacy aggregate compatibility checks outside pytest's process."""

from pathlib import Path
import subprocess
import sys


CHECK_SCRIPT = Path(__file__).with_name("_torch_compat_checks.py")


def test_torch_compat():
    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT)],
        cwd=str(Path(__file__).resolve().parents[3]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout

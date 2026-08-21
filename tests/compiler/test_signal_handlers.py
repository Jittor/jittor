"""Process-level signal handlers must respect embedded host ownership."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import unittest


REPO_PYTHON = str(Path(__file__).resolve().parents[2] / "python")

_JUPYTER_CHILD_PROBE = r"""
import os
import signal
import subprocess
import sys

import jittor

child = subprocess.run([
    sys.executable,
    "-c",
    "import os, signal; os.kill(os.getpid(), signal.SIGKILL)",
])
print("CHILD_RETURN", child.returncode)
print("HOST_STILL_ALIVE")
"""


@unittest.skipIf(os.name == "nt", "uses POSIX SIGCHLD semantics")
class TestSignalHandlerOwnership(unittest.TestCase):
    def test_jupyter_host_keeps_sigchld_ownership(self):
        environment = dict(os.environ)
        for name in (
            "DISABLE_MULTIPROCESSING",
            "JT_NO_SIGNAL_HANDLER",
            "OMPI_COMM_WORLD_SIZE",
            "PMI_SIZE",
        ):
            environment.pop(name, None)
        environment.update({
            "PYTHONPATH": REPO_PYTHON + os.pathsep + environment.get("PYTHONPATH", ""),
            "JPY_PARENT_PID": str(os.getpid()),
            "CUDA_VISIBLE_DEVICES": "",
            "nvcc_path": "",
            "use_cuda": "0",
            "JITTOR_TORCH_SHIM": "0",
        })
        completed = subprocess.run(
            (sys.executable, "-c", _JUPYTER_CHILD_PROBE),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=120,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("CHILD_RETURN -{}".format(signal.SIGKILL), completed.stdout)
        self.assertIn("HOST_STILL_ALIVE", completed.stdout)


if __name__ == "__main__":
    unittest.main()

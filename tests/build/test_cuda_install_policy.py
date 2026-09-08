"""Importing a library must not restart the program, or fetch 2 GB unasked.

``install_cuda.check_cuda_env`` used to read ``/proc/self/cmdline`` and
``os.execl`` the interpreter so the loader would pick up a corrected
``LD_LIBRARY_PATH``. A library cannot do that to the program it was imported
into: a script started through its shebang has ``argv[0] == the script``, so
``argv[1:]`` drops the script and the "restart" runs ``python <first
argument>``; everything the process did before the import is gone; and inside
an MPI rank or a multiprocessing worker one rank re-exec'ing itself takes the
job with it. The whole thing was wrapped in ``except: pass``.

The same import also downloaded a ~2 GB CUDA toolkit by itself whenever the
machine had a driver but no nvcc on PATH, showing one line of output and no
way to decline.
"""

import os
from pathlib import Path
import unittest

from jittor_utils import install_cuda, manifest


SOURCE = Path(install_cuda.__file__)


class TestNoSelfRestart(unittest.TestCase):

    def test_the_module_never_execs_the_interpreter(self):
        text = SOURCE.read_text(encoding="utf8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            self.assertNotIn("os.exec", stripped,
                             "a library must not restart its host process")

    def test_checking_the_environment_returns(self):
        """It used to return only because the exec had failed."""
        self.assertIsNone(install_cuda.check_cuda_env())


class TestDownloadIsAskedFor(unittest.TestCase):

    def setUp(self):
        self.asset = manifest.jtcuda_asset([12, 2])
        self.saved_flag = install_cuda._install_cuda_requested
        self.saved_env = os.environ.pop("JTCUDA_AUTO_INSTALL", None)

    def tearDown(self):
        install_cuda._install_cuda_requested = self.saved_flag
        os.environ.pop("JTCUDA_AUTO_INSTALL", None)
        if self.saved_env is not None:
            os.environ["JTCUDA_AUTO_INSTALL"] = self.saved_env

    def test_an_import_does_not_download_by_itself(self):
        install_cuda._install_cuda_requested = False
        self.assertFalse(install_cuda._download_is_allowed(self.asset))

    def test_running_the_module_is_the_request(self):
        install_cuda._install_cuda_requested = True
        self.assertTrue(install_cuda._download_is_allowed(self.asset))

    def test_unattended_machines_can_opt_in(self):
        install_cuda._install_cuda_requested = False
        os.environ["JTCUDA_AUTO_INSTALL"] = "1"
        self.assertTrue(install_cuda._download_is_allowed(self.asset))

    def test_the_module_still_installs_when_run_as_a_command(self):
        text = SOURCE.read_text(encoding="utf8")
        main = text[text.index('if __name__ == "__main__":'):]
        self.assertIn("_install_cuda_requested = True", main)


if __name__ == "__main__":
    unittest.main()

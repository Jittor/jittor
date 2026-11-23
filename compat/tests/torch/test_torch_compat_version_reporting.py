"""Who owns what ``torch.__version__`` reports (task 7.14).

``import torch`` means ``torch`` IS the jittor module, so
``torch.__version__ = x`` sets ``jittor.__version__`` for every user in the
process.  The vLLM adapter used to do exactly that, from a package whose whole
staging contract is "public Jittor APIs only" -- reading a public API is public
use, writing through one is mutating the framework.

The decision now lives in the compatibility layer, which knows both numbers, as
an explicit and reversible call.  tests/structure/test_vllm_compat_structure.py
holds the other half: no staged adapter may assign to a torch/jittor attribute.
"""
import unittest

from packaging.version import Version

import jittor as jt
import torch


class TestTorchApiVersionReporting(unittest.TestCase):
    #: ``__jittor_version__`` is the name the *torch namespace* carries for
    #: jittor's version; on the jittor module itself it is ``__version__``.
    def setUp(self):
        self._saved = torch.__version__
        # xdist shares a worker between files, and another test may have opted
        # into reporting the Torch API level without restoring it. This class
        # pins the *default*, so state it instead of inheriting whatever the
        # process happens to hold; the toggle is documented as reversible.
        torch.compat_report_torch_api_version(False)

    def tearDown(self):
        torch.__version__ = self._saved

    def test_by_default_version_is_jittors_own(self):
        self.assertEqual(torch.__version__, jt.__version__)
        self.assertNotEqual(torch.__version__, torch.__torch_version__)

    def test_the_torch_api_level_is_separately_available(self):
        self.assertEqual(torch.version.__version__, torch.__torch_version__)

    def test_opting_in_reports_the_api_level(self):
        got = torch.compat_report_torch_api_version(True)
        self.assertEqual(got, torch.__torch_version__)
        self.assertEqual(torch.__version__, torch.__torch_version__)

    def test_it_is_reversible(self):
        torch.compat_report_torch_api_version(True)
        got = torch.compat_report_torch_api_version(False)
        self.assertEqual(got, jt.__version__)
        self.assertEqual(torch.__version__, jt.__version__)

    def test_it_is_idempotent(self):
        first = torch.compat_report_torch_api_version(True)
        second = torch.compat_report_torch_api_version(True)
        self.assertEqual(first, second)

    def test_release_tuple_comparisons_preserve_reported_text(self):
        release = Version(jt.__version__).release
        major = release[0]
        version = torch.__version__
        self.assertIsInstance(version, str)
        self.assertEqual(str(version), jt.__version__)
        self.assertGreaterEqual(version, (major - 1,))
        self.assertLessEqual(version, (major + 1,))
        self.assertGreater(version, (major - 1,))
        self.assertLess(version, (major + 1,))
        torch.compat_report_torch_api_version(True)
        self.assertGreaterEqual(torch.__version__, Version(torch.__torch_version__).release)
        self.assertEqual(torch.compat_report_torch_api_version(False), jt.__version__)

    def test_xpu_build_flag_follows_registered_backend(self):
        self.assertIs(torch._C._has_xpu, "xpu" in jt.core.registered_backends())


if __name__ == "__main__":
    unittest.main()

"""Who owns what ``torch.__version__`` reports (task 7.14).

``import jittor as torch`` means ``torch`` IS the jittor module, so
``torch.__version__ = x`` sets ``jittor.__version__`` for every user in the
process.  The vLLM adapter used to do exactly that, from a package whose whole
staging contract is "public Jittor APIs only" -- reading a public API is public
use, writing through one is mutating the framework.

The decision now lives in the compatibility layer, which knows both numbers, as
an explicit and reversible call.  tests/structure/test_vllm_compat_structure.py
holds the other half: no staged adapter may assign to a torch/jittor attribute.
"""
import unittest

import jittor as jt
import jittor as torch


class TestTorchApiVersionReporting(unittest.TestCase):
    def setUp(self):
        self._saved = torch.__version__

    def tearDown(self):
        torch.__version__ = self._saved

    def test_by_default_version_is_jittors_own(self):
        self.assertEqual(torch.__version__, jt.__jittor_version__)
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
        self.assertEqual(got, jt.__jittor_version__)
        self.assertEqual(torch.__version__, jt.__jittor_version__)

    def test_it_is_idempotent(self):
        first = torch.compat_report_torch_api_version(True)
        second = torch.compat_report_torch_api_version(True)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

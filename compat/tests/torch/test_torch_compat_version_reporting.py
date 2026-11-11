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
from types import ModuleType, SimpleNamespace

import pytest

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


@pytest.mark.parametrize("rocm,nvcc,resources,cuda,hip", [
    (False, None, {}, None, None),
    (False, (12, 4, 0), {}, "12.4", None),
    (True, (12, 4, 0), {"hip_version": "6.3.12345"}, None, "6.3.12345"),
])
def test_toolkit_identity_follows_the_selected_backend(monkeypatch, rocm, nvcc,
                                                      resources, cuda, hip):
    from jittor.compat.torch.installers.cuda import bindings
    from jittor.compat.torch.context import ModuleRegistry

    compiler = SimpleNamespace(nvcc_version=nvcc, build_config=SimpleNamespace(
        has_rocm=rocm, resources=resources))
    monkeypatch.setattr(bindings, "jt", SimpleNamespace(compiler=compiler, __version__="2.0.2"))
    target = ModuleType("toolkit_version_target")
    target.__version__ = "2.0.2"
    registry = ModuleRegistry(target, modules={})
    bindings._install_version(target, registry)
    assert target.version.cuda == cuda
    assert target.version.hip == hip
    assert registry.module_map["torch.version"] is target.version
    assert target.__version__ == "2.0.2"


def test_missing_hip_build_metadata_is_not_fabricated_or_swallowed(monkeypatch):
    from jittor.compat.torch.installers.cuda import bindings
    from jittor.compat.torch.context import ModuleRegistry

    compiler = SimpleNamespace(build_config=SimpleNamespace(has_rocm=True, resources={}))
    monkeypatch.setattr(bindings, "jt", SimpleNamespace(compiler=compiler))
    target = ModuleType("missing_hip_version")
    with pytest.raises(RuntimeError, match="missing the HIP SDK version"):
        bindings._install_version(target, ModuleRegistry(target, modules={}))


if __name__ == "__main__":
    unittest.main()

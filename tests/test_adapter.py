from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[1] / "src"))

from jittor_gs import _READONLY_FUNCTIONS, register_patches
from jittor_gs.runtime import _patch_lpips_module


class TestGaussianSplattingAdapter(unittest.TestCase):
    def test_lpips_criterion_is_reused(self):
        constructed = []

        class Criterion:
            def __init__(self, net_type, version):
                constructed.append((net_type, version))

            def to(self, device):
                self.device = device
                return self

            def __call__(self, x, y):
                return (self.device, x.value, y.value)

        module = SimpleNamespace(lpips=lambda *args, **kwargs: None, LPIPS=Criterion)
        self.assertTrue(_patch_lpips_module(module))
        x = SimpleNamespace(device="cuda:0", value=1)
        y = SimpleNamespace(device="cuda:0", value=2)
        self.assertEqual(module.lpips(x, y, net_type="vgg"), ("cuda:0", 1, 2))
        self.assertEqual(module.lpips(x, y, net_type="vgg"), ("cuda:0", 1, 2))
        self.assertEqual(constructed, [("vgg", "0.1")])
        self.assertFalse(_patch_lpips_module(module))

    def test_entry_point_registers_runtime_and_boundary_policies(self):
        registrations = {}
        captured = {}

        def register(path, callback):
            registrations.setdefault(path, []).append(callback)

        def register_readonly_extension_borrow(**kwargs):
            captured.update(kwargs)
            for path in kwargs["registry"]:
                kwargs["register_patch"](path, lambda module: True)

        jittor = ModuleType("jittor")
        jittor.__path__ = []
        torch_shim = ModuleType("jittor.torch_shim")
        torch_shim.__path__ = []
        readonly = ModuleType("jittor.torch_shim.readonly_extensions")
        readonly.register_readonly_extension_borrow = (
            register_readonly_extension_borrow
        )
        with mock.patch.dict(
            sys.modules,
            {
                "jittor": jittor,
                "jittor.torch_shim": torch_shim,
                "jittor.torch_shim.readonly_extensions": readonly,
            },
            clear=False,
        ):
            register_patches(register)
        self.assertIn("scene.gaussian_model", registrations)
        self.assertIn("lpipsPyTorch", registrations)
        for path in _READONLY_FUNCTIONS:
            self.assertIn(path, registrations)
        self.assertEqual(captured["registry"], _READONLY_FUNCTIONS)
        self.assertIs(captured["register_patch"], register)

        module = ModuleType("lpipsPyTorch")
        module.lpips = lambda *args: None
        module.LPIPS = object
        self.assertTrue(any(callback(module) for callback in registrations["lpipsPyTorch"]))

    def test_package_import_does_not_import_jittor(self):
        package_root = Path(__file__).resolve().parents[1]
        code = (
            "import sys; import jittor_gs; "
            "assert 'jittor' not in sys.modules"
        )
        import subprocess

        env = os.environ.copy()
        env["PYTHONPATH"] = os.fspath(package_root / "src")
        subprocess.run([sys.executable, "-c", code], check=True, env=env)


if __name__ == "__main__":
    unittest.main()

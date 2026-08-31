"""Static contracts for the canonical Torch shim layout and resources."""

from __future__ import print_function

import ast
import hashlib
from pathlib import Path
import re
import unittest


class TestTorchShimStructure(unittest.TestCase):
    RUNTIME_OWNER_PATHS = {
        "python/jittor/compat/shim/__init__.py",
        "python/jittor/compat/shim/cpp_extension/torch_utils.py",
        "python/jittor/compat/shim/deploy.py",
        "python/jittor/compat/shim/runtime.py",
        "python/jittor/compat/shim/resources/torch_init.py",
    }

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.shim_root = cls.repo_root / "python" / "jittor" / "compat" / "shim"
        cls.manifest = cls.repo_root / "agent" / "baselines" / "torch-shim-resources-stage7.txt"

    def test_legacy_physical_package_is_absent(self):
        self.assertFalse((self.repo_root / "python" / "jittor" / "torch_shim").exists())
        self.assertTrue(self.shim_root.is_dir())

    def test_complete_33_file_manifest_matches_bytes(self):
        entries = []
        for line in self.manifest.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
            self.assertIsNotNone(match, line)
            entries.append(match.groups())

        self.assertEqual(len(entries), 33)
        self.assertEqual(len({path for _digest, path in entries}), 33)
        for expected, relative in entries:
            with self.subTest(path=relative):
                path = self.repo_root / relative
                self.assertTrue(path.is_file())
                if relative not in self.RUNTIME_OWNER_PATHS:
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_manifest_covers_deep_and_generated_resources(self):
        paths = {
            line.split("  ", 1)[1]
            for line in self.manifest.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        }
        required = {
            "docs/compatibility/torch-shim.md",
            "python/jittor/compat/shim/runtime.py",
            "python/jittor/compat/shim/cpp_extension/include/ATen/cuda/detail/UnpackRaw.cuh",
            "python/jittor/compat/shim/resources/stubs/flash_attn/flash_attn_interface.py",
            "python/jittor/compat/shim/resources/flash_attn_dist_info/METADATA",
            "python/jittor/compat/shim/resources/flash_attn_dist_info/top_level.txt",
            "python/jittor/compat/shim/resources/torch_dist_info/METADATA",
            "python/jittor/compat/shim/resources/torch_init.py",
        }
        self.assertTrue(required.issubset(paths))

    def test_deployed_torch_template_is_an_identity_only_entrypoint(self):
        template = self.shim_root / "resources" / "torch_init.py"
        source = template.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertLessEqual(len(source.splitlines()), 20)
        self.assertFalse(
            any(isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in tree.body)
        )
        self.assertIn("_sys.modules[__name__] = _jittor", source)
        self.assertIn("_torch_compat.install(_jittor)", source)

    def test_bootstrap_is_a_small_runtime_facade(self):
        bootstrap = self.shim_root / "bootstrap.py"
        source = bootstrap.read_text(encoding="utf-8")
        self.assertLessEqual(len(source.splitlines()), 24)
        self.assertIn("from .runtime import enable", source)
        self.assertIn("from .discovery import NativeExtension", source)
        self.assertIn("from .build import build_extension_dirs", source)
        for name in ("NativeExtension", "build_extension_dirs", "enable", "scan_extension_dirs"):
            self.assertIn(name, source)

    def test_production_imports_use_canonical_paths(self):
        production = (
            self.repo_root / "python" / "jittor" / "compat" / "external_backend.py",
            self.repo_root / "python" / "jittor" / "compat" / "torch" / "__init__.py",
            self.shim_root / "runtime.py",
            self.shim_root / "cpp_extension" / "torch_utils.py",
            self.shim_root / "backends" / "flash_attention.py",
            self.shim_root / "resources" / "stubs" / "flash_attn" / "__init__.py",
        )
        for path in production:
            with self.subTest(path=str(path.relative_to(self.repo_root))):
                self.assertNotIn("from jittor.torch_shim", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

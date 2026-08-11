"""Static boundaries for Stage 7 compatibility composition."""

from __future__ import print_function

import ast
from pathlib import Path
import unittest


class TestRuntimeCompositionStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[2]
        cls.jittor = cls.repo / "python" / "jittor"
        cls.compat = cls.jittor / "compat"

    def test_root_contains_only_preflight_and_post_core_composition(self):
        path = self.jittor / "__init__.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        definitions = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        forbidden = {
            "_TorchShimFlagsProxy",
            "_apply_external_runtime_patches",
            "_install_torch_shim_runtime",
        }
        self.assertFalse(definitions & forbidden)
        self.assertFalse(any(name.startswith("_jt_torch_") for name in definitions))
        self.assertNotIn("jittor.torch_shim", source)
        self.assertIn("prepare_import_environment as _prepare_compat_import", source)
        self.assertIn("compose as _compose_compat_runtime", source)
        self.assertIn("JITTOR_TORCH_STRICT_BOOTSTRAP", source)
        self.assertIn("strict=_compat_is_truthy(", source)
        self.assertLess(
            source.index("_prepare_compat_import("),
            source.index("from jittor_utils import lock"),
        )

    def test_preflight_and_lazy_shim_are_stdlib_only(self):
        allowed = {
            "__future__", "dataclasses", "glob", "hashlib", "importlib",
            "os", "pathlib", "sys",
        }
        for relative in ("shim/preflight.py", "shim/__init__.py"):
            path = self.compat / relative
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name.split(".", 1)[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.add((node.module or "").split(".", 1)[0])
            with self.subTest(path=relative):
                self.assertTrue(imports.issubset(allowed), imports - allowed)

    def test_shim_runtime_is_orchestration_only(self):
        runtime = self.compat / "shim" / "runtime.py"
        tree = ast.parse(runtime.read_text(encoding="utf-8"), filename=str(runtime))
        definitions = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        ]
        self.assertEqual(definitions, ["enable"])
        composition = (self.compat / "runtime.py").read_text(encoding="utf-8")
        self.assertNotIn("apply_external_runtime_patches", composition)
        for name in ("preflight.py", "discovery.py", "build.py", "control.py"):
            self.assertTrue((runtime.parent / name).is_file())

    def test_alias_ownership_is_central(self):
        aliases = (self.compat / "_aliases.py").read_text(encoding="utf-8")
        for name in (
            "jittor.torch_compat",
            "jittor.torch_fsdp2_compat",
            "jittor.torch_shim",
            "jittor.triton_shim",
            "jittor.depthwise_conv",
        ):
            self.assertIn(name, aliases)
        for path in (
            self.compat / "shim" / "__init__.py",
            self.compat / "triton" / "__init__.py",
            self.compat / "fsdp2" / "__init__.py",
            self.jittor / "nn" / "modules" / "depthwise.py",
        ):
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=str(path.relative_to(self.repo))):
                self.assertNotIn("MetaPathFinder", source)
                self.assertNotIn("_LegacyAliasFinder", source)

    def test_new_modules_parse_as_python_37(self):
        paths = [
            self.compat / "_aliases.py",
            self.compat / "integrations.py",
            self.compat / "runtime.py",
            self.compat / "torch" / "context.py",
        ]
        paths.extend((self.compat / "torch" / "installers").glob("*.py"))
        paths.extend((self.compat / "shim").glob("*.py"))
        for path in paths:
            with self.subTest(path=str(path.relative_to(self.repo))):
                ast.parse(
                    path.read_text(encoding="utf-8"),
                    filename=str(path),
                    feature_version=(3, 7),
                )


if __name__ == "__main__":
    unittest.main()

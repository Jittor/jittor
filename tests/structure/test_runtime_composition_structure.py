"""Static boundaries for Stage 7 compatibility composition."""

from __future__ import print_function

import ast
import pickle
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
        self.assertFalse(definitions)
        self.assertLess(len(source.splitlines()), 220)
        self.assertNotIn("jittor.torch_shim", source)
        self.assertIn("prepare_import_environment as _prepare_compat_import", source)
        self.assertIn("from ._runtime.core_api import *", source)
        self.assertIn("compose as _compose_compat_runtime", source)
        self.assertIn("JITTOR_TORCH_STRICT_BOOTSTRAP", source)
        self.assertIn("strict=_compat_is_truthy(", source)
        self.assertLess(
            source.index("_prepare_compat_import("),
            source.index("from jittor_utils import lock"),
        )
        self.assertLess(
            source.index("from ._runtime.core_api import *"),
            source.index("from . import nn"),
        )
        self.assertLess(
            source.index("from . import nn"),
            source.index("_compose_compat_runtime("),
        )

    def test_core_api_identity_and_legacy_pickle_paths_are_stable(self):
        import jittor
        from jittor._runtime import core_api

        for name in (
            "Module", "Function", "flag_scope", "array", "make_module",
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(jittor, name), getattr(core_api, name))
        self.assertIs(jittor.flags, core_api.flags)
        for name in ("Module", "Function"):
            implementation = getattr(jittor, name)
            current = pickle.dumps(implementation, protocol=0)
            legacy = current.replace(
                b"cjittor._runtime.core_api\n", b"cjittor\n", 1,
            )
            self.assertIs(pickle.loads(legacy), implementation)

    def test_compat_composition_keeps_native_core_implementations_available(self):
        import jittor
        from jittor._runtime import core_api

        self.assertIsNot(jittor.grad, core_api.grad)
        self.assertEqual(
            jittor.grad.__module__, "jittor.compat.torch.installers.tensor",
        )
        self.assertIsNot(jittor.save, core_api.save)
        self.assertIsNot(jittor.load, core_api.load)
        required = [
            report
            for report in jittor._compat_composition_report.torch_reports
            if report.required
        ]
        self.assertTrue(required)
        self.assertTrue(all(report.status == "complete" for report in required))

    def test_moved_scope_state_stays_synchronized_with_the_root(self):
        import jittor
        from jittor._runtime import core_api

        self.assertIsNone(core_api.single_log_capture)
        self.assertIsNone(jittor.single_log_capture)
        with jittor.log_capture_scope(log_v=0):
            self.assertIs(core_api.single_log_capture, True)
            self.assertIs(jittor.single_log_capture, True)
        self.assertIsNone(core_api.single_log_capture)
        self.assertIsNone(jittor.single_log_capture)

        class FakeMPI(object):
            def __init__(self):
                self.state = True

            def get_state(self):
                return self.state

            def set_state(self, state):
                self.state = state

            def world_rank(self):
                return 0

        old_mpi = core_api.mpi
        old_core_in_mpi = core_api.in_mpi
        old_root_in_mpi = jittor.in_mpi
        old_compile_in_mpi = core_api.compile_extern.in_mpi
        fake_mpi = FakeMPI()
        try:
            core_api.mpi = fake_mpi
            core_api.in_mpi = True
            jittor.in_mpi = True
            core_api.compile_extern.in_mpi = True
            scope_type = getattr(core_api, "__single_process_scope")
            with scope_type(rank=0) as selected:
                self.assertTrue(selected)
                self.assertFalse(core_api.in_mpi)
                self.assertFalse(jittor.in_mpi)
                self.assertFalse(core_api.compile_extern.in_mpi)
            self.assertTrue(core_api.in_mpi)
            self.assertTrue(jittor.in_mpi)
            self.assertTrue(core_api.compile_extern.in_mpi)
            self.assertTrue(fake_mpi.state)
        finally:
            core_api.mpi = old_mpi
            core_api.in_mpi = old_core_in_mpi
            jittor.in_mpi = old_root_in_mpi
            core_api.compile_extern.in_mpi = old_compile_in_mpi

    def test_core_api_is_the_only_large_python_api_implementation(self):
        path = self.jittor / "_runtime" / "core_api.py"
        self.assertTrue(path.is_file())
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        definitions = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        self.assertIn("Module", definitions)
        self.assertIn("Function", definitions)
        self.assertIn("array", definitions)
        self.assertIn("flag_scope", definitions)
        self.assertNotIn("compose", definitions)

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
            self.jittor / "_runtime" / "__init__.py",
            self.jittor / "_runtime" / "core_api.py",
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

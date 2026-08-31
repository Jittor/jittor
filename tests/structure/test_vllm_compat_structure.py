"""The vLLM compatibility package is staged here, so pin what lets it leave.

It is meant to become a plugin in its own repository. That move is only a
directory move plus an entry-point declaration for as long as the package
depends on nothing a plugin could not reach: public Jittor APIs and the module
patcher. These rules fail the moment that stops being true.
"""

import ast
import unittest
from pathlib import Path

import jittor
from jittor.compat import vllm as vllm_compat

_PACKAGE = Path(vllm_compat.__file__).resolve().parent

# What an out-of-tree plugin could import just as well as this package can.
_ALLOWED_JITTOR_IMPORTS = {
    "jittor",
    "jittor.compat.module_patcher",
    "jittor.compat.permissive",
}


def _sources():
    return sorted(_PACKAGE.glob("*.py"))


def _imported_names(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                # A relative import stays inside this package, which travels
                # with it; only the name it binds matters.
                for alias in node.names:
                    yield "." * node.level + module + "." + alias.name, node
            else:
                for alias in node.names:
                    yield module + "." + alias.name, node


class TestVllmCompatIsRelocatable(unittest.TestCase):
    def test_architecture_records_the_staged_plugin_boundary(self):
        repo = _PACKAGE.parents[3]
        source = (
            repo / "docs" / "architecture" / "source-architecture.md"
        ).read_text(encoding="utf-8")
        self.assertIn("`jittor.compat.vllm` is a staged exception", source)
        self.assertIn("versioned, installable vLLM plugin", source)

    def test_the_package_is_present_and_small(self):
        names = {path.name for path in _sources()}
        self.assertEqual(
            names,
            {"__init__.py", "custom_ops.py", "flash_attn.py", "layers.py"})
        for path in _sources():
            with self.subTest(module=path.name):
                self.assertLessEqual(
                    len(path.read_text(encoding="utf-8").splitlines()), 300)

    def test_it_imports_jittor_only_through_its_public_entry_points(self):
        for path in _sources():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for name, node in _imported_names(tree):
                if not name.startswith("jittor"):
                    continue
                root = name if name in _ALLOWED_JITTOR_IMPORTS else name.rsplit(".", 1)[0]
                with self.subTest(module=path.name, imported=name):
                    self.assertIn(
                        root, _ALLOWED_JITTOR_IMPORTS,
                        "%s:%d imports %s; a plugin outside this repository "
                        "could not" % (path.name, node.lineno, name))

    def test_it_touches_no_private_jittor_attribute(self):
        for path in _sources():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Attribute):
                    continue
                if not node.attr.startswith("_") or node.attr.startswith("__"):
                    continue
                value = node.value
                root = value
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name) and root.id in ("jt", "jittor"):
                    self.fail("%s:%d reaches Jittor's private %s"
                              % (path.name, node.lineno, node.attr))

    def test_every_jittor_primitive_it_calls_is_actually_public(self):
        used = set()
        for path in _sources():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Attribute)
                        and isinstance(node.value.value, ast.Name)
                        and node.value.value.id == "jt"
                        and node.value.attr == "nn"):
                    used.add(node.attr)
        self.assertTrue(used, "expected the package to call jt.nn primitives")
        for name in sorted(used):
            with self.subTest(primitive=name):
                self.assertIn(name, dir(jittor.nn))
                self.assertIn(name, getattr(jittor.nn, "__all__", [name]) or [name])


class TestVllmCompatArmsItselfWithoutRunning(unittest.TestCase):
    def test_arming_puts_exactly_one_finder_on_the_import_path(self):
        import sys

        vllm_compat.register()
        vllm_compat.register()
        finders = [finder for finder in sys.meta_path
                   if type(finder).__name__ == "_ArmOnFirstImport"]
        self.assertEqual(len(finders), 1)

    def test_the_finder_ignores_everything_that_is_not_vllm(self):
        vllm_compat.register()
        import sys

        finder = next(f for f in sys.meta_path
                      if type(f).__name__ == "_ArmOnFirstImport")
        for name in ("numpy", "vllmish", "torch", "jittor.nn"):
            with self.subTest(module=name):
                self.assertIsNone(finder.find_spec(name))

    def test_the_operator_table_names_what_it_registers(self):
        from jittor.compat.vllm import custom_ops

        declared = {name for name, _ in custom_ops._OPERATORS}
        self.assertEqual(declared, set(custom_ops._IMPLEMENTATIONS))

    def test_every_layer_patch_is_registered_against_a_vllm_module(self):
        from jittor.compat.vllm import flash_attn, layers

        for source in (layers.PATCHES, flash_attn.PATCHES):
            self.assertTrue(source)
            for path, patch in source.items():
                with self.subTest(module=path):
                    self.assertTrue(path.startswith("vllm."))
                    self.assertTrue(callable(patch))

    def test_arming_registers_every_layer_patch(self):
        from jittor.compat.module_patcher import registered_module_patches
        from jittor.compat.vllm import flash_attn, layers

        vllm_compat.register()
        registry = registered_module_patches()
        for source in (layers.PATCHES, flash_attn.PATCHES):
            for path, patch in source.items():
                with self.subTest(module=path):
                    self.assertIn(patch, registry.get(path, ()))


if __name__ == "__main__":
    unittest.main()

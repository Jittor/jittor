"""The optional KV-cache import must not masquerade as supported computation."""
import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "compat/shim/resources/stubs/flash_attn"


class TestFlashCacheImportContract(unittest.TestCase):
    def load(self, native):
        tree = ast.parse((SOURCE / "__init__.py").read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "flash_attn_with_kvcache")
        namespace = {"_native_function": native}
        module = ast.Module(body=[function], type_ignores=[])
        exec(compile(module, str(SOURCE / "__init__.py"), "exec"), namespace)
        return namespace[function.name]

    def test_absent_native_cache_support_is_explicit(self):
        function = self.load(lambda name: None)
        with self.assertRaisesRegex(NotImplementedError, "KV-cache"):
            function(object(), object(), object())

    def test_native_receives_every_argument_unchanged(self):
        calls = []
        result = object()
        def native(*args, **kwargs):
            calls.append((args, kwargs))
            return result
        names = []
        function = self.load(lambda name: names.append(name) or native)
        query, keys, values = object(), object(), object()
        self.assertIs(function(query, keys, values, cache_seqlens=3, causal=True), result)
        self.assertEqual(names, ["flash_attn_with_kvcache"])
        self.assertEqual(calls, [((query, keys, values), {"cache_seqlens": 3, "causal": True})])

    def test_both_public_import_paths_export_the_entry(self):
        tree = ast.parse((SOURCE / "__init__.py").read_text())
        exports = next(ast.literal_eval(node.value) for node in tree.body
                       if isinstance(node, ast.Assign) and any(
                           isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets))
        self.assertIn("flash_attn_with_kvcache", exports)
        imports = ast.parse((SOURCE / "flash_attn_interface.py").read_text())
        self.assertTrue(any(isinstance(node, ast.ImportFrom) and
                            any(alias.name == "flash_attn_with_kvcache" for alias in node.names)
                            for node in imports.body))


if __name__ == "__main__":
    unittest.main()

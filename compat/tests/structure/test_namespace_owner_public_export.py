"""The optional adapters consume one public read-only namespace query."""
import ast
from pathlib import Path
import unittest


class TestNamespaceOwnerExport(unittest.TestCase):
    def test_public_root_reexports_existing_owner_without_a_second_implementation(self):
        root = Path(__file__).resolve().parents[2]
        tree = ast.parse((root / "torch/__init__.py").read_text())
        imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)
                   and node.module == "publication" and node.level == 1]
        self.assertEqual([(alias.name, alias.asname) for node in imports for alias in node.names],
                         [("namespace_owner", None)])
        self.assertFalse(any(isinstance(node, ast.FunctionDef) and node.name == "namespace_owner"
                             for node in tree.body))
        exports = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "__all__"
                               for target in node.targets))
        self.assertIn("namespace_owner", ast.literal_eval(exports))
        original = ast.parse((root / "torch/publication.py").read_text())
        node = next(node for node in original.body if isinstance(node, ast.FunctionDef)
                    and node.name == "namespace_owner")
        class TorchNamespace:
            def __init__(self, owner):
                self.owner = owner
        scope = {"TorchNamespace": TorchNamespace}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "namespace_owner", "exec"), scope)
        owner = object()
        self.assertIs(scope["namespace_owner"](TorchNamespace(owner)), owner)
        self.assertIsNone(scope["namespace_owner"](object()))


if __name__ == "__main__":
    unittest.main()

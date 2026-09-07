"""Linear algebra has concrete, acyclic domain owners and a re-export facade."""
import ast
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[2] / "python/jittor/linalg"


def test_linalg_implementations_are_owned_by_domain_modules():
    assert not PACKAGE.with_suffix(".py").exists()
    definitions = {}
    for path in PACKAGE.glob("*.py"):
        source = path.read_text()
        assert len(source.splitlines()) < 1500, path
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                assert node.name not in definitions, node.name
                definitions[node.name] = path.stem
    assert definitions["solve"] == "solving"
    assert definitions["svd"] == "decompositions"
    assert definitions["complex_svd"] == "complex"
    assert definitions["matrix_norm"] == "norms"
    assert definitions["einsum"] == "contractions"
    assert definitions["_matmul"] == "_helpers"


def test_linalg_facade_is_only_explicit_exports():
    tree = ast.parse((PACKAGE / "__init__.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Expr):
            assert isinstance(node.value, ast.Constant)
        elif isinstance(node, ast.Assign):
            assert [target.id for target in node.targets] in (["__all__"], ["_COMPLEX_EXPORTS"])
        elif isinstance(node, ast.FunctionDef):
            assert node.name in ("__getattr__", "__dir__")
        else:
            assert isinstance(node, ast.ImportFrom)
            assert node.level == 1
            assert all(alias.name != "*" for alias in node.names)

    exports = next(node for node in tree.body if isinstance(node, ast.Assign)
                   and node.targets[0].id == "_COMPLEX_EXPORTS")
    assert ast.literal_eval(exports.value) == (
        "complex_inv", "complex_eig", "complex_eigh", "complex_qr",
        "complex_svd", "complex_pinv",
    )
    assert not any(isinstance(node, ast.ImportFrom) and node.module == "complex"
                   for node in tree.body)


def test_linalg_domain_imports_are_acyclic():
    edges = {}
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text())
        edges[path.stem] = {
            node.module for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level == 1
        }

    def visit(module, ancestors):
        assert module not in ancestors, (module, ancestors)
        for dependency in edges[module]:
            assert dependency in edges, dependency
            visit(dependency, ancestors | {module})

    for module in edges:
        visit(module, set())

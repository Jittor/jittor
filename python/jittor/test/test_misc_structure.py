"""Architecture and compatibility contracts for the ``jittor.misc`` facade."""

import ast
import inspect
import pickle
from pathlib import Path
import types
import unittest
from unittest import mock

import jittor as jt
import jittor.misc as misc
from jittor._misc import runtime
from jittor._misc import shape_composition
from jittor._misc import shape_transforms


_PUBLIC_NAMES = {
    "CTCLoss", "Finfo", "Iterable", "Sequence", "all", "all_equal", "any",
    "arange", "arctan2", "atan2", "atleast_1d", "atleast_2d", "atleast_3d",
    "auto_parallel", "bernoulli", "bfloat16_finfo", "block_diag",
    "cartesian_prod", "chunk", "contiguous", "cpu", "cross", "ctc_loss",
    "cub_cumsum", "cuda", "cummax", "cummin", "cumprod", "cumsum",
    "deg2rad", "diag", "diagonal", "expand", "expm1", "finfo", "flip",
    "from_torch", "gather", "get_max_memory_treemap", "histc", "hypot",
    "iinfo", "index_add", "index_add_", "index_fill", "index_fill_",
    "index_select", "isfinite", "isin", "isinf", "isnan", "isneginf",
    "isposinf", "jt", "knn", "kthvalue", "linspace", "log2", "make_grid",
    "math", "median", "meshgrid", "multinomial", "ne", "new", "nms",
    "nonzero", "normalize", "np", "numpy_cumprod", "numpy_cumsum", "peek",
    "peek_s", "print_tree", "python_pass_wrapper", "rad2deg", "randperm",
    "repeat", "repeat_interleave", "roll", "rsqrt", "safe_log", "save_image",
    "scatter", "scatter_", "scatter_add", "scatter_add_", "scatter_reduce",
    "searchsorted", "set_global_seed", "sort", "split", "stack", "t", "time",
    "to", "tolist", "topk", "tril", "triu", "unbind", "unique",
    "unique_consecutive", "view_as",
}

_MOVED = {
    "repeat": shape_transforms.repeat,
    "chunk": shape_transforms.chunk,
    "expand": shape_transforms.expand,
    "atleast_1d": shape_composition.atleast_1d,
    "atleast_2d": shape_composition.atleast_2d,
    "atleast_3d": shape_composition.atleast_3d,
    "cartesian_prod": shape_composition.cartesian_prod,
    "block_diag": shape_composition.block_diag,
}


class _ModuleImportVisitor(ast.NodeVisitor):
    def __init__(self):
        self.violations = []

    def visit_FunctionDef(self, node):
        return

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name == "jittor" or alias.name.startswith("jittor."):
                self.violations.append((node.lineno, alias.name))

    def visit_ImportFrom(self, node):
        if node.level >= 2:
            self.violations.append((node.lineno, "." * node.level + (node.module or "")))
        elif node.level == 0 and node.module and (
            node.module == "jittor" or node.module.startswith("jittor.")
        ):
            self.violations.append((node.lineno, node.module))


class TestMiscStructure(unittest.TestCase):
    def test_public_surface_and_private_ownership(self):
        self.assertFalse(hasattr(misc, "__all__"))
        self.assertEqual(
            {name for name in vars(misc) if not name.startswith("_")},
            _PUBLIC_NAMES,
        )
        self.assertEqual(
            [name for name in sorted(_PUBLIC_NAMES) if not hasattr(jt, name)],
            [],
        )
        self.assertIsInstance(jt._misc, types.ModuleType)
        self.assertEqual(jt._misc.__name__, "jittor._misc")

        facade_tree = ast.parse(Path(misc.__file__).read_text(encoding="utf-8"))
        facade_definitions = {
            node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)
        }
        self.assertTrue(_MOVED.keys().isdisjoint(facade_definitions))
        self.assertIn("repeat_interleave", facade_definitions)

        private_definitions = {}
        for module in (shape_transforms, shape_composition):
            tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
            private_definitions[module.__name__] = {
                node.name for node in tree.body if isinstance(node, ast.FunctionDef)
            }
        self.assertEqual(
            private_definitions[shape_transforms.__name__],
            {"repeat", "chunk", "expand"},
        )
        self.assertEqual(
            private_definitions[shape_composition.__name__],
            {
                "atleast_1d", "atleast_2d", "atleast_3d",
                "cartesian_prod", "block_diag",
            },
        )

        implementations = tuple(_MOVED.values())
        self.assertEqual(len(implementations), 8)
        self.assertEqual(len({id(value) for value in implementations}), 8)
        for name, implementation in _MOVED.items():
            with self.subTest(name=name):
                self.assertIs(getattr(misc, name), implementation)
                self.assertIs(getattr(jt, name), implementation)

    def test_signatures_reflection_and_pickle(self):
        signatures = {
            "repeat": "(x, *shape)",
            "chunk": "(x, chunks, dim=0)",
            "expand": "(x, *shape)",
            "atleast_1d": "(*tensors)",
            "atleast_2d": "(*tensors)",
            "atleast_3d": "(*tensors)",
            "cartesian_prod": "(*tensors)",
            "block_diag": "(*tensors)",
        }
        for name, signature in signatures.items():
            function = _MOVED[name]
            with self.subTest(name=name):
                self.assertEqual(str(inspect.signature(function)), signature)
                self.assertEqual(function.__module__, "jittor.misc")
                self.assertEqual(function.__qualname__, name)
                self.assertIs(pickle.loads(pickle.dumps(function)), function)

    def test_var_bindings_and_root_scan_remain_stable(self):
        for name in ("repeat", "chunk", "expand"):
            with self.subTest(binding=name):
                self.assertIs(getattr(jt.Var, name), _MOVED[name])
        for name in (
            "atleast_1d", "atleast_2d", "atleast_3d",
            "cartesian_prod", "block_diag",
        ):
            with self.subTest(no_var_binding=name):
                self.assertFalse(hasattr(jt.Var, name))
        self.assertIs(jt.Var.repeat_interleave, misc.repeat_interleave)
        self.assertIs(jt.repeat_interleave, misc.repeat_interleave)
        for name in ("repeat_", "repeat_interleave_", "chunk_", "expand_"):
            with self.subTest(inplace=name):
                self.assertTrue(callable(getattr(jt.Var, name)))
        for name in ("repeat", "chunk", "expand"):
            wrapper = getattr(jt.Var, name + "_")
            captured = tuple(
                cell.cell_contents for cell in (wrapper.__closure__ or ())
            )
            with self.subTest(inplace_closure=name):
                self.assertIn(_MOVED[name], captured)

    def test_cartesian_product_dispatches_through_public_meshgrid(self):
        calls = []
        marker = object()

        class FakeGrid:
            def __init__(self, name):
                self.name = name

            def reshape(self, *shape):
                calls.append((self.name, shape))
                return f"column-{self.name}"

        def fake_meshgrid(tensors):
            calls.append(("meshgrid", tuple(tensors)))
            return (FakeGrid("a"), FakeGrid("b"))

        def fake_concat(columns, dim):
            calls.append(("concat", tuple(columns), dim))
            return marker

        a = jt.array([1, 2])
        b = jt.array([3, 4])
        with mock.patch.object(misc, "meshgrid", fake_meshgrid), \
                mock.patch.object(jt, "concat", fake_concat):
            self.assertIs(misc.cartesian_prod(a, b), marker)
        self.assertEqual(calls, [
            ("meshgrid", (a, b)),
            ("a", (-1, 1)),
            ("b", (-1, 1)),
            ("concat", ("column-a", "column-b"), 1),
        ])

    def test_repeat_resolves_public_sequence_and_numpy_dependencies(self):
        sequence_lookups = []
        numpy_lookups = []

        class FakeSequenceMeta(type):
            def __instancecheck__(cls, value):
                sequence_lookups.append(value)
                return isinstance(value, tuple)

        class FakeSequence(metaclass=FakeSequenceMeta):
            pass

        class FakeArray:
            def __init__(self, value):
                self.value = list(value)

            def __mul__(self, other):
                return FakeArray(a * b for a, b in zip(self.value, other.value))

            def tolist(self):
                return self.value

        class FakeNumpy:
            @staticmethod
            def array(value):
                numpy_lookups.append(value)
                return FakeArray(value)

        value = jt.array([1, 2])
        with mock.patch.object(misc, "Sequence", FakeSequence), \
                mock.patch.object(misc, "np", FakeNumpy):
            result = misc.repeat(value, (2,))
            self.assertEqual(result.shape, [4])
        self.assertEqual(sequence_lookups, [(2,)])
        self.assertEqual(len(numpy_lookups), 2)

    def test_runtime_import_direction_and_file_budgets(self):
        self.assertIs(runtime.jt._module, jt)
        facade_path = Path(misc.__file__).resolve()
        facade_tree = ast.parse(facade_path.read_text(encoding="utf-8"))
        statements = list(facade_tree.body)
        bind_index = next(
            index for index, node in enumerate(statements)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_bind_misc_runtime"
        )
        transform_index = next(
            index for index, node in enumerate(statements)
            if isinstance(node, ast.ImportFrom)
            and node.module == "_misc.shape_transforms"
        )
        composition_index = next(
            index for index, node in enumerate(statements)
            if isinstance(node, ast.ImportFrom)
            and node.module == "_misc.shape_composition"
        )
        self.assertLess(bind_index, transform_index)
        self.assertLess(bind_index, composition_index)
        self.assertLessEqual(len(facade_path.read_text(encoding="utf-8").splitlines()), 2850)

        for module in (runtime, shape_transforms, shape_composition):
            path = Path(module.__file__).resolve()
            source = path.read_text(encoding="utf-8")
            self.assertLessEqual(len(source.splitlines()), 200)
            tree = ast.parse(source, filename=str(path))
            visitor = _ModuleImportVisitor()
            visitor.visit(tree)
            self.assertEqual(visitor.violations, [])
            if module is not runtime:
                imports = [
                    node for node in tree.body
                    if isinstance(node, ast.ImportFrom) and node.level > 0
                ]
                self.assertEqual(len(imports), 1)
                self.assertTrue(all(
                    node.level == 1 and node.module == "runtime" for node in imports
                ))

    def test_setup_declares_private_package(self):
        setup_path = Path(misc.__file__).resolve().parents[2] / "setup.py"
        if not setup_path.is_file():
            self.skipTest("setup.py is only available in a source checkout")
        tree = ast.parse(setup_path.read_text(encoding="utf-8"), filename=str(setup_path))
        package_lists = [
            keyword.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (getattr(node.func, "id", None) or getattr(node.func, "attr", None)) == "setup"
            for keyword in node.keywords
            if keyword.arg == "packages" and isinstance(keyword.value, ast.List)
        ]
        self.assertEqual(len(package_lists), 1)
        packages = {
            item.value for item in package_lists[0].elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
        self.assertIn("jittor._misc", packages)


if __name__ == "__main__":
    unittest.main()

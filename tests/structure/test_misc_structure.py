"""Architecture and compatibility contracts for the ``jittor.misc`` facade."""

import ast
import inspect
import pickle
from pathlib import Path
import unittest
from unittest import mock

import jittor as jt
import jittor.misc as misc
from jittor.misc import shape_composition
from jittor.misc import shape_transforms
from jittor.misc import tensor_ops
from jittor.misc import concatenation
from jittor.misc import indexing


_PUBLIC_NAMES = {
    "CTCLoss", "Finfo", "Iterable", "Sequence", "all", "all_equal", "any",
    "arange", "arctan2", "atan2", "atleast_1d", "atleast_2d", "atleast_3d",
    "amax", "amin",
    "auto_parallel", "bernoulli", "bfloat16_finfo", "block_diag",
    "cartesian_prod", "cat", "chunk", "concat", "concatenation", "contiguous",
    "cpu", "count_nonzero", "cross", "ctc_loss",
    "cub_cumsum", "cuda", "cummax", "cummin", "cumprod", "cumsum",
    "deg2rad", "diag", "diagonal", "expand", "expm1", "finfo", "flip",
    "from_torch", "gather", "get_max_memory_treemap", "histc", "hypot",
    "iinfo", "index_add", "index_add_", "index_fill", "index_fill_", "indexing",
    "index_select", "isfinite", "isin", "isinf", "isnan", "isneginf",
    "isposinf", "jt", "knn", "kthvalue", "linspace", "log2", "make_grid",
    "math", "median", "meshgrid", "multinomial", "ne", "new", "nms",
    "nonzero", "normalize", "np", "numpy_cumprod", "numpy_cumsum", "peek",
    "peek_s", "print_tree", "python_pass_wrapper", "rad2deg", "randperm",
    "repeat", "repeat_interleave", "roll", "rsqrt", "safe_log", "save_image",
    "scatter", "scatter_", "scatter_add", "scatter_add_", "scatter_reduce",
    "searchsorted", "set_global_seed", "sort", "split", "stack", "t", "time",
    "tensor_ops", "to", "tolist", "topk", "tril", "triu", "unbind", "unique",
    "unique_consecutive", "view_as",
    "reductions", "shape_composition", "shape_transforms",
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
        self.assertFalse(hasattr(jt, "_misc"))

        facade_tree = ast.parse(Path(misc.__file__).read_text(encoding="utf-8"))
        self.assertFalse(any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for node in facade_tree.body
        ))

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
                expected_module = (
                    "jittor.misc.shape_transforms"
                    if name in {"repeat", "chunk", "expand"}
                    else "jittor.misc.shape_composition"
                )
                self.assertEqual(function.__module__, expected_module)
                self.assertEqual(function.__qualname__, name)
                self.assertIs(pickle.loads(pickle.dumps(function)), function)

                current_pickle = pickle.dumps(function, protocol=0)
                legacy_pickle = current_pickle.replace(
                    ("c" + expected_module + "\n").encode(),
                    b"cjittor.misc\n",
                    1,
                )
                self.assertIs(pickle.loads(legacy_pickle), function)

    def test_tensor_operations_use_real_paths_and_legacy_pickle_aliases(self):
        source = Path(tensor_ops.__file__).read_text(encoding="utf-8")
        for private_name in ("_cummax_min", "_CumMax", "_CumMin"):
            self.assertNotIn("jt.misc." + private_name, source)
        for name in ("repeat_interleave", "cumsum", "scatter_reduce", "CTCLoss"):
            implementation = getattr(tensor_ops, name)
            with self.subTest(name=name):
                self.assertIs(getattr(misc, name), implementation)
                if name != "cumsum":
                    self.assertIs(getattr(jt, name), implementation)
                self.assertEqual(implementation.__module__, tensor_ops.__name__)
                self.assertIs(
                    pickle.loads(pickle.dumps(implementation)), implementation,
                )

                current_pickle = pickle.dumps(implementation, protocol=0)
                legacy_pickle = current_pickle.replace(
                    ("c" + tensor_ops.__name__ + "\n").encode(),
                    b"cjittor.misc\n",
                    1,
                )
                self.assertIs(pickle.loads(legacy_pickle), implementation)

    def test_tensor_operation_dependencies_remain_dynamic(self):
        # cumsum resolves its kernel through the module at call time, so a
        # patch here takes effect. The vehicle used to be `numpy_cumsum`, which
        # cumsum called directly on CPU; cumsum now has one implementation for
        # both devices and reaches it through `_scan_2d`, so that is what the
        # late binding has to carry. `wraps` because the result is reshaped by
        # the caller and a bare marker would not survive it.
        marker = object()
        value = jt.array([1.0, 2.0])
        with mock.patch.object(misc, "_scan_2d",
                               wraps=misc._scan_2d) as patched:
            with jt.flag_scope(use_cuda=0):
                misc.cumsum(value, 0).sync()
        patched.assert_called_once()

        loss = misc.CTCLoss(blank=3, reduction="sum", zero_infinity=True)
        with mock.patch.object(misc, "ctc_loss", return_value=marker) as patched:
            self.assertIs(loss("log", "targets", "inputs", "targets_len"), marker)
        patched.assert_called_once_with(
            "log", "targets", "inputs", "targets_len", 3, "sum", True,
        )

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

    def test_package_import_direction(self):
        facade_path = Path(misc.__file__).resolve()
        facade_tree = ast.parse(facade_path.read_text(encoding="utf-8"))
        # # A line count is not an architecture contract: it goes red when someone
        # adds a necessary comment and stays green when someone adds a wrong
        # line. The structural assertions around it are the actual rule.
        for module in (concatenation, indexing, tensor_ops, shape_transforms,
                       shape_composition):
            path = Path(module.__file__).resolve()
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("preserve_facade_origins", source)
            self.assertNotIn("_JittorRuntimeProxy", source)
            tree = ast.parse(source, filename=str(path))
            imports_jittor = [
                node for node in tree.body
                if isinstance(node, ast.Import)
                and any(alias.name == "jittor" for alias in node.names)
            ]
            self.assertEqual(len(imports_jittor), 1)

    def test_package_discovery_includes_private_package(self):
        repo_root = Path(misc.__file__).resolve().parents[3]
        if not (repo_root / "pyproject.toml").is_file():
            self.skipTest("packaging metadata is only available in a source checkout")
        from setuptools import find_packages

        packages = find_packages(where=str(repo_root / "python"))
        self.assertIn("jittor.misc", packages)
        self.assertNotIn("jittor._misc", packages)


if __name__ == "__main__":
    unittest.main()

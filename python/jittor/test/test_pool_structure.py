"""Architecture and compatibility contracts for the ``jittor.pool`` facade."""

import ast
import inspect
import pickle
import unittest
from pathlib import Path
from unittest import mock

import jittor as jt
from jittor import nn
from jittor import pool as pool_facade
from jittor._pool import adaptive
from jittor._pool import core_2d
from jittor._pool import core_3d
from jittor._pool import layers
from jittor._pool import pooling_1d
from jittor._pool import runtime
from jittor._pool import unpool


_IMPLEMENTATION_MODULES = (
    core_2d, core_3d, pooling_1d, adaptive, layers, unpool,
)


def _summarize(value):
    if isinstance(value, jt.Module):
        return (
            type(value).__name__,
            tuple((key, _summarize(item)) for key, item in value.__dict__.items()),
        )
    return value


class TestPoolStructure(unittest.TestCase):
    def test_public_surface_and_private_ownership(self):
        expected_public = {
            "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d",
            "AdaptiveMaxPool2d", "AdaptiveMaxPool3d", "AvgPool1d",
            "AvgPool2d", "AvgPool3d", "MaxPool1d", "MaxPool2d",
            "MaxPool3d", "MaxUnpool2d", "MaxUnpool3d", "Module", "Pool",
            "Pool3d", "avg_pool2d", "init", "jt", "math", "max_pool2d",
            "max_pool3d", "np", "pool", "pool2d", "pool3d",
            "pool_use_code_op",
        }
        self.assertFalse(hasattr(pool_facade, "__all__"))
        self.assertEqual(
            {name for name in vars(pool_facade) if not name.startswith("_")},
            expected_public,
        )
        self.assertIs(jt.pool, pool_facade)
        self.assertIs(pool_facade.pool2d, pool_facade.pool)
        self.assertEqual(pool_facade.pool2d.__name__, "pool")
        self.assertEqual(pool_facade.pool2d.__qualname__, "pool")
        self.assertIs(
            pickle.loads(pickle.dumps(pool_facade.pool2d)),
            pool_facade.pool,
        )

        implementations = tuple(
            symbol
            for module in _IMPLEMENTATION_MODULES
            for symbol in module._FACADE_SYMBOLS
        )
        self.assertEqual(len(implementations), 22)
        self.assertEqual(len({id(symbol) for symbol in implementations}), 22)
        for symbol in implementations:
            with self.subTest(symbol=symbol.__name__):
                self.assertIs(getattr(pool_facade, symbol.__name__), symbol)

    def test_function_signatures_reflection_and_pickle(self):
        signatures = {
            "_triple": "(x)",
            "pool": "(x, kernel_size, op, padding=0, stride=None)",
            "pool3d": "(x, kernel_size, op, padding=0, stride=None)",
            "avg_pool2d": (
                "(x, kernel_size, stride=None, padding=0, ceil_mode=False, "
                "count_include_pad=True)"
            ),
            "_no_dilation": "(dilation)",
            "max_pool2d": (
                "(x=None, kernel_size=None, stride=None, padding=0, "
                "dilation=None, return_indices=None, ceil_mode=False, "
                "input=None)"
            ),
            "max_pool3d": (
                "(x, kernel_size, stride=None, padding=0, dilation=None, "
                "return_indices=None, ceil_mode=False)"
            ),
        }
        for name, signature in signatures.items():
            function = getattr(pool_facade, name)
            with self.subTest(function=name):
                self.assertEqual(str(inspect.signature(function)), signature)
                self.assertEqual(function.__module__, "jittor.pool")
                self.assertEqual(function.__qualname__, name)
                self.assertIs(pickle.loads(pickle.dumps(function)), function)

    def test_class_signatures_reflection_and_pickle(self):
        signatures = {
            "Pool": (
                "(kernel_size, stride=None, padding=0, dilation=None, "
                "return_indices=None, ceil_mode=False, count_include_pad=True, "
                "op='maximum')"
            ),
            "Pool3d": (
                "(kernel_size, stride=None, padding=0, dilation=None, "
                "return_indices=None, ceil_mode=False, count_include_pad=True, "
                "op='maximum')"
            ),
            "AdaptiveAvgPool2d": "(output_size)",
            "AdaptiveAvgPool1d": "(output_size)",
            "MaxPool1d": (
                "(kernel_size, stride=None, padding=0, dilation=1, "
                "return_indices=None, ceil_mode=False)"
            ),
            "AvgPool1d": (
                "(kernel_size, stride=None, padding=0, ceil_mode=False, "
                "count_include_pad=True)"
            ),
            "AdaptiveMaxPool2d": "(output_size, return_indices=False)",
            "AdaptiveAvgPool3d": "(output_size)",
            "AdaptiveMaxPool3d": "(output_size, return_indices=False)",
            "AvgPool2d": (
                "(kernel_size, stride=None, padding=0, ceil_mode=False, "
                "count_include_pad=True)"
            ),
            "AvgPool3d": (
                "(kernel_size, stride=None, padding=0, ceil_mode=False, "
                "count_include_pad=True)"
            ),
            "MaxPool2d": (
                "(kernel_size, stride=None, padding=0, dilation=None, "
                "return_indices=None, ceil_mode=False)"
            ),
            "MaxPool3d": (
                "(kernel_size, stride=None, padding=0, dilation=None, "
                "return_indices=None, ceil_mode=False)"
            ),
            "MaxUnpool2d": "(kernel_size, stride=None)",
            "MaxUnpool3d": "(kernel_size, stride=None)",
        }
        for name, signature in signatures.items():
            cls = getattr(pool_facade, name)
            with self.subTest(cls=name):
                self.assertEqual(str(inspect.signature(cls)), signature)
                self.assertIs(cls.__mro__[1], jt.Module)
                self.assertEqual(cls.__module__, "jittor.pool")
                self.assertEqual(cls.__qualname__, name)
                self.assertEqual(cls.__init__.__module__, "jittor.pool")
                self.assertEqual(cls.execute.__module__, "jittor.pool")
                self.assertIs(pickle.loads(pickle.dumps(cls)), cls)

    def test_default_instance_fields_state_and_pickle(self):
        cases = (
            (
                pool_facade.Pool(2),
                (
                    ("return_indices", None), ("kernel_size", (2, 2)),
                    ("op", "maximum"), ("stride", (2, 2)),
                    ("padding", (0, 0)), ("ceil_mode", False),
                    ("count_include_pad", False),
                ),
            ),
            (
                pool_facade.Pool3d(2),
                (
                    ("return_indices", None), ("kernel_size", (2, 2, 2)),
                    ("op", "maximum"), ("stride", (2, 2, 2)),
                    ("padding", (0, 0, 0)), ("ceil_mode", False),
                    ("count_include_pad", False),
                ),
            ),
            (pool_facade.AdaptiveAvgPool2d(2), (("output_size", 2),)),
            (pool_facade.AdaptiveAvgPool1d(2), (("output_size", 2),)),
            (
                pool_facade.MaxPool1d(2),
                (
                    ("kernel_size", 2), ("stride", 2), ("padding", 0),
                    ("ceil_mode", False), ("return_indices", None),
                ),
            ),
            (
                pool_facade.AvgPool1d(2),
                (
                    ("kernel_size", 2), ("stride", 2), ("padding", 0),
                    ("ceil_mode", False), ("count_include_pad", True),
                ),
            ),
            (
                pool_facade.AdaptiveMaxPool2d(2),
                (("output_size", 2), ("return_indices", False)),
            ),
            (
                pool_facade.AdaptiveAvgPool3d(2),
                (("output_size", (2, 2, 2)),),
            ),
            (
                pool_facade.AdaptiveMaxPool3d(2),
                (("output_size", (2, 2, 2)), ("return_indices", False)),
            ),
            (
                pool_facade.AvgPool2d(2),
                (("layer", (
                    "Pool",
                    (
                        ("return_indices", None), ("kernel_size", (2, 2)),
                        ("op", "mean"), ("stride", (2, 2)),
                        ("padding", (0, 0)), ("ceil_mode", False),
                        ("count_include_pad", False),
                    ),
                )),),
            ),
            (
                pool_facade.AvgPool3d(2),
                (("layer", (
                    "Pool3d",
                    (
                        ("return_indices", None),
                        ("kernel_size", (2, 2, 2)), ("op", "mean"),
                        ("stride", (2, 2, 2)), ("padding", (0, 0, 0)),
                        ("ceil_mode", False), ("count_include_pad", False),
                    ),
                )),),
            ),
            (
                pool_facade.MaxPool2d(2),
                (("_layer", (
                    "Pool",
                    (
                        ("return_indices", None), ("kernel_size", (2, 2)),
                        ("op", "maximum"), ("stride", (2, 2)),
                        ("padding", (0, 0)), ("ceil_mode", False),
                        ("count_include_pad", False),
                    ),
                )),),
            ),
            (
                pool_facade.MaxPool3d(2),
                (("_layer", (
                    "Pool3d",
                    (
                        ("return_indices", None),
                        ("kernel_size", (2, 2, 2)), ("op", "maximum"),
                        ("stride", (2, 2, 2)), ("padding", (0, 0, 0)),
                        ("ceil_mode", False), ("count_include_pad", False),
                    ),
                )),),
            ),
            (
                pool_facade.MaxUnpool2d(2),
                (("kernel_size", (2, 2)), ("stride", (2, 2))),
            ),
            (
                pool_facade.MaxUnpool3d(2),
                (("kernel_size", (2, 2, 2)), ("stride", (2, 2, 2))),
            ),
        )
        for instance, expected in cases:
            restored = pickle.loads(pickle.dumps(instance))
            with self.subTest(instance=type(instance).__name__):
                self.assertEqual(_summarize(instance)[1], expected)
                self.assertIs(type(restored), type(instance))
                self.assertEqual(_summarize(restored)[1], expected)
                self.assertEqual(tuple(instance.state_dict()), ())
                self.assertEqual(tuple(restored.state_dict()), ())

    def _assert_factory_dispatch(
        self, attribute, invoke, expected_args, expected_kwargs,
    ):
        marker = object()
        calls = []

        class FakeFactory:
            def __init__(self, *args, **kwargs):
                calls.append((args, kwargs))

            def __call__(self, value):
                calls.append(value)
                return marker

        with mock.patch.object(pool_facade, attribute, FakeFactory):
            self.assertIs(invoke(), marker)
        self.assertEqual(calls, [(expected_args, expected_kwargs), "input"])

    def test_functionals_dispatch_through_public_facade(self):
        self._assert_factory_dispatch(
            "Pool",
            lambda: pool_facade.pool("input", 2, "maximum", 1, 3),
            (2, 3, 1), {"op": "maximum"},
        )
        self._assert_factory_dispatch(
            "Pool3d",
            lambda: pool_facade.pool3d("input", 2, "minimum", 1, 3),
            (2, 3, 1), {"op": "minimum"},
        )
        self._assert_factory_dispatch(
            "AvgPool2d",
            lambda: pool_facade.avg_pool2d("input", 2, 3, 1, True, False),
            (2, 3, 1, True, False), {},
        )
        self._assert_factory_dispatch(
            "MaxPool2d",
            lambda: pool_facade.max_pool2d(kernel_size=2, input="input"),
            (2, None, 0, None, None, False), {},
        )
        self._assert_factory_dispatch(
            "MaxPool3d",
            lambda: pool_facade.max_pool3d("input", 2, 3, 1, 1, True, True),
            (2, 3, 1, 1, True, True), {},
        )

    def test_wrapper_constructors_dispatch_through_public_facade(self):
        calls = []

        class FakeCore:
            def __init__(self, *args, **kwargs):
                calls.append((args, kwargs))

        with mock.patch.object(pool_facade, "Pool", FakeCore):
            average = pool_facade.AvgPool2d(2, 3, 1, True, False)
        self.assertIsInstance(average.layer, FakeCore)
        self.assertEqual(calls.pop(), ((), {
            "kernel_size": 2, "stride": 3, "padding": 1,
            "ceil_mode": True, "count_include_pad": False, "op": "mean",
        }))

        dilation_calls = []
        with (
            mock.patch.object(pool_facade, "Pool", FakeCore),
            mock.patch.object(
                pool_facade, "_no_dilation",
                side_effect=lambda value: dilation_calls.append(value) or True,
            ),
        ):
            maximum = pool_facade.MaxPool2d(2, 3, 1, (1, 1), True, True)
        self.assertIsInstance(maximum._layer, FakeCore)
        self.assertEqual(dilation_calls, [(1, 1)])
        self.assertEqual(calls.pop(), ((), {
            "kernel_size": 2, "stride": 3, "padding": 1,
            "dilation": None, "return_indices": True, "ceil_mode": True,
            "op": "maximum",
        }))

        with mock.patch.object(pool_facade, "Pool3d", FakeCore):
            average3d = pool_facade.AvgPool3d(2, 3, 1, True, False)
        self.assertIsInstance(average3d.layer, FakeCore)
        self.assertEqual(calls.pop(), ((), {
            "kernel_size": 2, "stride": 3, "padding": 1,
            "ceil_mode": True, "count_include_pad": False, "op": "mean",
        }))

        with (
            mock.patch.object(pool_facade, "Pool3d", FakeCore),
            mock.patch.object(pool_facade, "_no_dilation", return_value=True),
        ):
            maximum3d = pool_facade.MaxPool3d(2, 3, 1, 1, True, True)
        self.assertIsInstance(maximum3d._layer, FakeCore)
        self.assertEqual(calls.pop(), ((), {
            "kernel_size": 2, "stride": 3, "padding": 1,
            "dilation": None, "return_indices": True, "ceil_mode": True,
            "op": "maximum",
        }))
        self.assertEqual(calls, [])

    def test_adaptive_and_triple_dependencies_are_dynamic(self):
        marker = object()
        calls = []

        class FakeMax:
            def __init__(self, *args, **kwargs):
                calls.append((args, kwargs))

            def __call__(self, value):
                calls.append(value)
                return marker

        class FakeTensor:
            shape = (1, 1, 4, 4)

        with mock.patch.object(pool_facade, "MaxPool2d", FakeMax):
            result = pool_facade.AdaptiveMaxPool2d(2, return_indices=True)(
                FakeTensor(),
            )
        self.assertIs(result, marker)
        self.assertEqual(calls, [
            ((), {"kernel_size": (2, 2), "stride": (2, 2),
                  "return_indices": True}),
            mock.ANY,
        ])

        calls.clear()

        class FakeTensor3d:
            shape = (1, 1, 4, 4, 4)

        with mock.patch.object(pool_facade, "MaxPool3d", FakeMax):
            result = pool_facade.AdaptiveMaxPool3d(2, return_indices=True)(
                FakeTensor3d(),
            )
        self.assertIs(result, marker)
        self.assertEqual(calls, [
            ((), {"kernel_size": (2, 2, 2), "stride": (2, 2, 2),
                  "return_indices": True}),
            mock.ANY,
        ])

        triple_calls = []
        original_triple = pool_facade._triple

        def traced_triple(value):
            triple_calls.append(value)
            return original_triple(value)

        with mock.patch.object(pool_facade, "_triple", traced_triple):
            pool_facade.Pool3d(2)
            pool_facade.AdaptiveAvgPool3d(3)
            pool_facade.AdaptiveMaxPool3d(4)
            pool_facade.MaxUnpool3d(5, 6)
        self.assertEqual(triple_calls, [2, 2, 0, 3, 4, 5, 6])

    def test_core_reads_public_backend_flag_at_execution_time(self):
        marker = object()

        class FakeTensor:
            dtype = "float32"

            def __init__(self, shape):
                self.shape = shape
                self.reindex_calls = []

            def reindex(self, *args, **kwargs):
                self.reindex_calls.append((args, kwargs))
                return self

            def reduce(self, *args, **kwargs):
                return marker

        two_dimensional = pool_facade.Pool(2)
        three_dimensional = pool_facade.Pool3d(2)
        with mock.patch.object(pool_facade, "pool_use_code_op", False):
            x2 = FakeTensor((1, 1, 4, 4))
            x3 = FakeTensor((1, 1, 4, 4, 4))
            self.assertIs(two_dimensional.execute(x2), marker)
            self.assertIs(three_dimensional.execute(x3), marker)
        self.assertEqual(len(x2.reindex_calls), 1)
        self.assertEqual(len(x3.reindex_calls), 1)

    def test_adaptive_execution_fields_and_pickle(self):
        cases = (
            (
                pool_facade.AdaptiveAvgPool2d(2), jt.ones((1, 1, 4, 4)),
                (
                    ("output_size", 2), ("sh", 2), ("sw", 2),
                    ("ksh", 2), ("ksw", 2),
                ),
            ),
            (
                pool_facade.AdaptiveMaxPool2d(2), jt.ones((1, 1, 4, 4)),
                (
                    ("output_size", 2), ("return_indices", False),
                    ("sh", 2), ("sw", 2), ("ksh", 2), ("ksw", 2),
                ),
            ),
            (
                pool_facade.AdaptiveAvgPool3d(2), jt.ones((1, 1, 4, 4, 4)),
                (
                    ("output_size", (2, 2, 2)), ("sd", 2), ("sh", 2),
                    ("sw", 2), ("ksd", 2), ("ksh", 2), ("ksw", 2),
                ),
            ),
            (
                pool_facade.AdaptiveMaxPool3d(2), jt.ones((1, 1, 4, 4, 4)),
                (
                    ("output_size", (2, 2, 2)), ("return_indices", False),
                    ("sd", 2), ("sh", 2), ("sw", 2), ("ksd", 2),
                    ("ksh", 2), ("ksw", 2),
                ),
            ),
        )
        for instance, value, expected in cases:
            instance(value).numpy()
            restored = pickle.loads(pickle.dumps(instance))
            with self.subTest(instance=type(instance).__name__):
                self.assertEqual(tuple(instance.__dict__.items()), expected)
                self.assertEqual(tuple(restored.__dict__.items()), expected)
                self.assertEqual(tuple(instance.state_dict()), ())
                self.assertEqual(tuple(restored.state_dict()), ())

    def test_nn_and_functional_aliases_remain_stable(self):
        shared = (
            "AdaptiveAvgPool1d", "AdaptiveAvgPool3d", "AdaptiveMaxPool2d",
            "AdaptiveMaxPool3d", "AvgPool1d", "AvgPool3d", "MaxPool1d",
            "MaxPool2d", "MaxPool3d", "MaxUnpool2d", "MaxUnpool3d",
            "Pool", "Pool3d", "max_pool2d", "max_pool3d", "pool",
            "pool2d", "pool3d",
        )
        for name in shared:
            with self.subTest(shared=name):
                public = getattr(nn, name)
                source = getattr(pool_facade, name)
                if name == "Pool" and public is not source:
                    self.assertTrue(jt.compiler.has_acl)
                else:
                    self.assertIs(public, source)

        self.assertIsNot(nn.AvgPool2d, pool_facade.AvgPool2d)
        self.assertIsNot(nn.AdaptiveAvgPool2d, pool_facade.AdaptiveAvgPool2d)
        self.assertIsNot(nn.avg_pool2d, pool_facade.avg_pool2d)
        module_classes = (
            "Pool", "Pool3d", "AdaptiveAvgPool2d", "AdaptiveAvgPool1d",
            "MaxPool1d", "AvgPool1d", "AdaptiveMaxPool2d",
            "AdaptiveAvgPool3d", "AdaptiveMaxPool3d", "AvgPool2d",
            "AvgPool3d", "MaxPool2d", "MaxPool3d", "MaxUnpool2d",
            "MaxUnpool3d",
        )
        for name in module_classes:
            with self.subTest(nn_modules=name):
                self.assertIs(getattr(nn.modules, name), getattr(nn, name))
        for name in ("pool", "pool2d", "pool3d", "max_pool2d", "max_pool3d"):
            self.assertIs(getattr(nn.functional, name), getattr(pool_facade, name))
        self.assertIs(nn.functional.avg_pool2d, nn.avg_pool2d)
        self.assertIs(nn.functional.adaptive_avg_pool2d, nn.adaptive_avg_pool2d)

    def test_private_import_direction_and_file_budgets(self):
        self.assertIs(runtime.jt._module, jt)
        facade_path = Path(pool_facade.__file__).resolve()
        facade_tree = ast.parse(facade_path.read_text(encoding="utf-8"))
        self.assertFalse(any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for node in facade_tree.body
        ))
        self.assertLessEqual(
            len(facade_path.read_text(encoding="utf-8").splitlines()), 100,
        )

        for module in _IMPLEMENTATION_MODULES:
            path = Path(module.__file__).resolve()
            tree = ast.parse(path.read_text(encoding="utf-8"))
            with self.subTest(module=module.__name__):
                self.assertLessEqual(
                    len(path.read_text(encoding="utf-8").splitlines()), 250,
                )
                imports = [
                    node for node in tree.body
                    if isinstance(node, (ast.Import, ast.ImportFrom))
                ]
                self.assertTrue(imports)
                self.assertTrue(all(
                    isinstance(node, ast.ImportFrom)
                    and node.level == 1
                    and node.module == "runtime"
                    for node in imports
                ))

        setup_path = facade_path.parents[2] / "setup.py"
        if not setup_path.exists():
            self.skipTest("source checkout metadata is unavailable")
        setup_tree = ast.parse(setup_path.read_text(encoding="utf-8"))
        package_lists = [
            keyword.value
            for node in ast.walk(setup_tree)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
            if keyword.arg == "packages" and isinstance(keyword.value, ast.List)
        ]
        self.assertEqual(len(package_lists), 1)
        packages = {
            item.value for item in package_lists[0].elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
        self.assertIn("jittor._pool", packages)


if __name__ == "__main__":
    unittest.main()

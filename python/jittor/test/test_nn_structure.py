"""Structural contracts for the modular :mod:`jittor.nn` implementation."""

from abc import abstractmethod as abc_abstractmethod
import ast
import importlib
import pickle
from pathlib import Path
import types as python_types
import unittest

import jittor
import jittor.nn as nn
from jittor._nn import activations
from jittor._nn import layer_norm_cuda
from jittor._nn import losses
from jittor._nn import normalization
from jittor._nn import recurrent_base
from jittor._nn import recurrent_cells
from jittor._nn import recurrent_layers
from jittor._nn import softmax
from jittor._nn import vector


_IMPLEMENTATION_MODULES = (
    activations, layer_norm_cuda, losses, normalization, recurrent_base,
    recurrent_cells, recurrent_layers, softmax, vector,
)
_ACL_PATCHED_FUNCTIONS = {"relu", "leaky_relu", "softmax"}


def _is_acl_wrapper(value):
    return getattr(value, "__module__", "").startswith("jittor.extern.acl")


class _ModuleImportVisitor(ast.NodeVisitor):
    """Inspect imports executed at module load while ignoring function bodies."""

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
            target = "." * node.level + (node.module or "")
            self.violations.append((node.lineno, target))
        elif node.level == 0 and node.module and (
            node.module == "jittor" or node.module.startswith("jittor.")
        ):
            self.violations.append((node.lineno, node.module))


def _moved_symbols():
    for module in _IMPLEMENTATION_MODULES:
        yield from module._FACADE_SYMBOLS


class TestNNStructure(unittest.TestCase):
    def test_private_implementation_package_is_not_shadowed(self):
        self.assertIsInstance(jittor._nn, python_types.ModuleType)
        self.assertEqual(jittor._nn.__name__, "jittor._nn")

    def test_public_nn_module_identity_remains_stable(self):
        self.assertIs(jittor.nn, nn)
        self.assertIs(importlib.import_module("jittor.nn"), nn)
        self.assertIs(nn.Module, jittor.Module)

    def test_facade_reexports_private_implementations(self):
        for implementation in _moved_symbols():
            name = implementation.__name__
            public = getattr(nn, name)
            with self.subTest(name=name):
                if name in _ACL_PATCHED_FUNCTIONS and public is not implementation:
                    self.assertTrue(_is_acl_wrapper(public))
                else:
                    self.assertIs(public, implementation)

    def test_moved_symbols_keep_public_reflection_and_pickle_contracts(self):
        for implementation in _moved_symbols():
            with self.subTest(name=implementation.__name__):
                self.assertEqual(implementation.__module__, "jittor.nn")
                public = getattr(nn, implementation.__name__)
                if (
                    implementation.__name__ in _ACL_PATCHED_FUNCTIONS
                    and public is not implementation
                ):
                    self.assertTrue(_is_acl_wrapper(public))
                else:
                    self.assertIs(pickle.loads(pickle.dumps(implementation)), public)

    def test_log_softmax_dispatches_through_public_softmax(self):
        original = nn.softmax
        calls = []
        marker = object()

        def replacement(*args, **kwargs):
            calls.append((args, kwargs))
            return marker

        nn.softmax = replacement
        try:
            self.assertIs(nn.log_softmax("input", dim=3), marker)
        finally:
            nn.softmax = original
        self.assertEqual(calls, [(('input',), {'dim': 3, 'log': True})])

    def test_normalization_helpers_dispatch_through_public_facade(self):
        class Marker:
            ndim = 3
            shape = (1, 2, 3)

            def reshape(self, *args):
                return self

            def __mul__(self, other):
                return self

            __rmul__ = __mul__

            def __add__(self, other):
                return self

            __radd__ = __add__

        marker = Marker()
        original_normalize = nn._ln_normalize
        normalize_calls = []

        def replacement_normalize(x, dims, eps):
            normalize_calls.append((x, dims, eps))
            return marker

        nn._ln_normalize = replacement_normalize
        try:
            self.assertIs(
                nn.instance_norm(marker, weight=None, bias=None),
                marker,
            )
            self.assertIs(nn.group_norm(marker, 1), marker)
        finally:
            nn._ln_normalize = original_normalize
        self.assertEqual([call[1] for call in normalize_calls], [[2], [2, 3]])

        original_factory = nn._ln_function_cls

        class FakeFunction:
            @staticmethod
            def apply(value):
                return value

        nn._ln_function_cls = lambda dims, eps: FakeFunction
        try:
            self.assertIs(nn._ln_normalize(marker, [2], 1e-5), marker)
        finally:
            nn._ln_function_cls = original_factory

    def test_normalization_public_contracts_remain_stable(self):
        for name in ("batch_norm", "instance_norm", "layer_norm", "group_norm"):
            with self.subTest(function=name):
                self.assertIs(getattr(nn.functional, name), getattr(nn, name))

        for cls in (nn.BatchNorm, nn.InstanceNorm, nn.LayerNorm, nn.GroupNorm):
            with self.subTest(cls=cls.__name__):
                self.assertEqual(cls.__module__, "jittor.nn")
                self.assertEqual(cls.__init__.__module__, "jittor.nn")
                if cls is nn.LayerNorm and _is_acl_wrapper(cls.execute):
                    pass
                else:
                    self.assertEqual(cls.execute.__module__, "jittor.nn")
                self.assertIs(pickle.loads(pickle.dumps(cls)), cls)

        cached_cls = nn._ln_function_cls((-1,), 1e-5)
        self.assertEqual(nn._ln_function_cls.__wrapped__.__module__, "jittor.nn")
        self.assertEqual(cached_cls.__module__, "jittor.nn")
        self.assertEqual(cached_cls.execute.__module__, "jittor.nn")
        self.assertEqual(cached_cls.grad.__module__, "jittor.nn")

    def test_recurrent_public_contracts_remain_stable(self):
        classes = (
            nn.LSTMCell, nn.RNNCell, nn.GRUCell, nn.RNNBase,
            nn.RNN, nn.LSTM, nn.GRU,
        )
        for cls in classes:
            with self.subTest(cls=cls.__name__):
                self.assertEqual(cls.__module__, "jittor.nn")
                self.assertEqual(cls.__qualname__, cls.__name__)
                self.assertIs(pickle.loads(pickle.dumps(cls)), cls)
                for member in vars(cls).values():
                    if isinstance(member, (staticmethod, classmethod)):
                        member = member.__func__
                    if callable(member) and hasattr(member, "__module__"):
                        self.assertEqual(member.__module__, "jittor.nn")

        for cls in (nn.RNN, nn.LSTM, nn.GRU):
            self.assertIs(cls.__mro__[1], nn.RNNBase)
        for cls in classes:
            self.assertIs(getattr(nn.modules, cls.__name__), cls)

        for cls in (nn.LSTMCell, nn.RNNCell, nn.GRUCell, nn.RNN, nn.LSTM, nn.GRU):
            instance = cls(2, 3)
            restored = pickle.loads(pickle.dumps(instance))
            with self.subTest(instance=cls.__name__):
                self.assertIs(type(restored), cls)
                self.assertEqual(
                    tuple(restored.state_dict().keys()),
                    tuple(instance.state_dict().keys()),
                )

    def test_recurrent_implementations_dispatch_through_public_facade(self):
        class Marker:
            def __add__(self, other):
                return self

            __radd__ = __add__

            def tanh(self):
                return self

        marker = Marker()
        holder = python_types.SimpleNamespace(
            hidden_size=1,
            bias=False,
            nonlinearity="tanh",
            weight_ih=marker,
            weight_hh=marker,
        )
        original_matmul = nn.matmul_transpose
        calls = []

        def replacement_matmul(left, right):
            calls.append((left, right))
            return marker

        nn.matmul_transpose = replacement_matmul
        try:
            self.assertIs(nn.RNNCell.execute(holder, marker, marker), marker)
        finally:
            nn.matmul_transpose = original_matmul
        self.assertEqual(len(calls), 2)

        source = Path(recurrent_base.__file__).read_text(encoding="utf-8")
        self.assertIn("jt.nn.init.uniform", source)
        self.assertIn("jt.nn.dropout", source)

    def test_tensor_method_bindings_remain_on_public_functions(self):
        for name in ("prelu", "hardswish", "hardsigmoid", "rrelu"):
            with self.subTest(name=name):
                self.assertIs(getattr(jittor.Var, name), getattr(nn, name))

    def test_key_reexports_and_aliases_remain_stable(self):
        from jittor import depthwise_conv, misc, optim, pool

        for name in ("SGD", "Adam", "AdamW", "RMSprop"):
            with self.subTest(name=name):
                self.assertIs(getattr(nn, name), getattr(optim, name))
        self.assertIs(nn.CTCLoss, misc.CTCLoss)
        self.assertIs(nn.DepthwiseConv, depthwise_conv.DepthwiseConv)
        self.assertIs(nn.abstractmethod, abc_abstractmethod)
        if nn.Pool is not pool.Pool:
            self.assertTrue(_is_acl_wrapper(nn.Pool))
        else:
            self.assertIs(nn.Pool, pool.Pool)
        self.assertIs(nn.MaxPool2d, pool.MaxPool2d)
        self.assertIs(nn.BatchNorm1d, nn.BatchNorm)
        self.assertIs(nn.BatchNorm2d, nn.BatchNorm)
        self.assertIs(nn.BatchNorm3d, nn.BatchNorm)
        self.assertIs(nn.InstanceNorm1d, nn.InstanceNorm)
        self.assertIs(nn.InstanceNorm2d, nn.InstanceNorm)
        self.assertIs(nn.InstanceNorm3d, nn.InstanceNorm)
        self.assertIs(nn.LayerNorm1d, nn.LayerNorm)
        self.assertIs(nn.LayerNorm2d, nn.LayerNorm)
        self.assertIs(nn.LayerNorm3d, nn.LayerNorm)
        if nn.Conv2d is not nn.Conv:
            self.assertTrue(_is_acl_wrapper(nn.Conv2d))
            self.assertTrue(_is_acl_wrapper(nn.Conv))
        else:
            self.assertIs(nn.Conv2d, nn.Conv)
        self.assertIs(nn.ConvTranspose2d, nn.ConvTranspose)
        if nn.conv is not nn.conv2d:
            self.assertTrue(_is_acl_wrapper(nn.conv2d))
        else:
            self.assertIs(nn.conv, nn.conv2d)
        if nn.ReLU is not nn.Relu:
            self.assertTrue(_is_acl_wrapper(nn.ReLU))
        else:
            self.assertIs(nn.ReLU, nn.Relu)

    def test_private_modules_do_not_import_root_at_module_scope(self):
        package_root = Path(activations.__file__).resolve().parent
        for path in package_root.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            visitor = _ModuleImportVisitor()
            visitor.visit(tree)
            with self.subTest(path=path.name):
                self.assertFalse(
                    visitor.violations,
                    f"{path.name} adds root import(s) at module load: "
                    f"{visitor.violations}",
                )

    def test_runtime_is_bound_before_private_modules_are_imported(self):
        facade_path = Path(nn.__file__).resolve()
        tree = ast.parse(facade_path.read_text(encoding="utf-8"), filename=str(facade_path))
        bind_index = next(
            index for index, node in enumerate(tree.body)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_bind_nn_runtime"
        )
        implementation_imports = [
            index for index, node in enumerate(tree.body)
            if isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module is not None
            and node.module.startswith("_nn.")
            and node.module != "_nn.runtime"
        ]
        self.assertEqual(len(implementation_imports), len(_IMPLEMENTATION_MODULES))
        self.assertTrue(all(bind_index < index for index in implementation_imports))

    def test_facade_contains_no_moved_definitions(self):
        facade_path = Path(nn.__file__).resolve()
        tree = ast.parse(facade_path.read_text(encoding="utf-8"), filename=str(facade_path))
        facade_definitions = {
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        moved_names = {symbol.__name__ for symbol in _moved_symbols()}
        self.assertFalse(facade_definitions & moved_names)

    def test_source_files_stay_within_architecture_budgets(self):
        facade_path = Path(nn.__file__).resolve()
        self.assertLessEqual(len(facade_path.read_text(encoding="utf-8").splitlines()), 3800)
        for module in _IMPLEMENTATION_MODULES:
            path = Path(module.__file__).resolve()
            with self.subTest(path=path.name):
                self.assertLessEqual(
                    len(path.read_text(encoding="utf-8").splitlines()),
                    350,
                )

    def test_setup_declares_private_implementation_package(self):
        package_root = Path(activations.__file__).resolve().parent
        setup_path = package_root.parents[2] / "setup.py"
        if not setup_path.is_file():
            self.skipTest("setup.py is only available in a source checkout")

        tree = ast.parse(setup_path.read_text(encoding="utf-8"), filename=str(setup_path))
        packages = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function_name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if function_name != "setup":
                continue
            for keyword in node.keywords:
                if keyword.arg == "packages":
                    packages = ast.literal_eval(keyword.value)
                    break
        self.assertIsNotNone(packages, "setup.py must declare packages explicitly")
        self.assertIn("jittor._nn", packages)


if __name__ == "__main__":
    unittest.main(verbosity=2)

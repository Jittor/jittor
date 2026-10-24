"""Execute real Module.to/device parsing without Jittor, JIT or hardware.

Only module traversal/conversion is observed: real residency and buffer values
require the separate accelerator regression, not this host routing contract.
"""
import ast
import functools
from pathlib import Path
import types
import typing
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[3]


class Dtype:
    _registry = {}

    def __init__(self, name):
        self.name = name


class TestModuleKeywordDevice(unittest.TestCase):
    def load(self, acl=True):
        self.calls = []
        self.model = object()
        self.jt = types.SimpleNamespace(Var=type("Tensor", (), {}),
            compiler=types.SimpleNamespace(has_acl=acl),
            flags=types.SimpleNamespace(use_cuda=0))
        namespace = dict(jt=self.jt, typing=typing, dtype=Dtype,
                         kind_of=lambda value: type(value).__name__,
                         EXPECTED=(AttributeError,), _functools=functools)
        tree = ast.parse((ROOT / "compat/torch/types.py").read_text())
        names = {"device", "_is_index", "_check_index", "_check_type", "_accelerator_type",
                 "_device_is_cuda"}
        tree.body = [node for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
                     or isinstance(node, ast.Assign) and any(
                         isinstance(target, ast.Name) and target.id == "_DEVICE_TYPES"
                         for target in node.targets)]
        exec(compile(tree, "canonical_device_contract", "exec"), namespace)
        def replace(module, convert):
            self.assertIs(module, self.model)
            self.calls.append(convert.args)
            return module
        namespace.update(_module_replace_vars=replace,
                         _module_to_conversion=lambda *args: None)
        path = ROOT / "compat/torch/installers/nn/module_methods.py"
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("_module_to", "_module_to_empty")]
        exec(compile(tree, str(path), "exec"), namespace)
        self.to = namespace["_module_to"]
        self.to_empty = namespace["_module_to_empty"]
        self.device = namespace["device"]

    def assert_target(self, kind, ordinal, dtype=None):
        self.assertEqual(len(self.calls), 1)
        actual_dtype, target, copy = self.calls[0]
        self.assertIsInstance(target, self.device)
        self.assertEqual((target.type, target.index), (kind, ordinal))
        self.assertEqual(actual_dtype, dtype)
        self.assertFalse(copy)

    def test_keyword_integer_uses_canonical_accelerator_not_cpu_noop(self):
        for acl, kind in ((True, "npu"), (False, "cuda")):
            for ordinal in (0, 1, np.int32(0), np.int64(1)):
                with self.subTest(acl=acl, ordinal=ordinal, type=type(ordinal)):
                    self.load(acl)
                    self.assertIs(self.to(self.model, device=ordinal, non_blocking=True), self.model)
                    self.assert_target(kind, int(ordinal))
                    self.assertEqual(self.jt.flags.use_cuda, 1)

    def test_positional_integer_uses_same_current_accelerator_kind(self):
        for acl, kind in ((True, "npu"), (False, "cuda")):
            self.load(acl)
            self.to(self.model, np.int64(1))
            self.assert_target(kind, 1)

    def test_negative_keyword_ordinals_reject_before_conversion(self):
        for ordinal in (-1, np.int32(-1), np.int64(-2)):
            self.load()
            with self.subTest(ordinal=ordinal), self.assertRaisesRegex(RuntimeError, "negative"):
                self.to(self.model, device=ordinal)
            self.assertEqual(self.calls, [])
            self.assertEqual(self.jt.flags.use_cuda, 0)

    def test_boolean_keyword_device_is_not_non_blocking_or_ordinal(self):
        for value in (False, True):
            self.load()
            with self.subTest(value=value), self.assertRaises(TypeError):
                self.to(self.model, device=value)
            self.assertEqual(self.calls, [])
            self.assertEqual(self.jt.flags.use_cuda, 0)

    def test_to_empty_delegates_integer_device_to_same_normalization(self):
        for ordinal in (0, np.int64(1)):
            self.load()
            self.assertIs(self.to_empty(self.model, device=ordinal), self.model)
            self.assert_target("npu", int(ordinal))
        for ordinal in (-1, True):
            self.load()
            with self.assertRaises((RuntimeError, TypeError)):
                self.to_empty(self.model, device=ordinal)
            self.assertEqual(self.calls, [])

    def test_keyword_dtype_retained_with_integer_device(self):
        self.load()
        self.to(self.model, device=1, dtype=Dtype("float32"), non_blocking=False)
        self.assert_target("npu", 1, "float32")

    def test_existing_explicit_devices_are_unchanged(self):
        for spelling in ("cpu", "npu:1", "cuda:1"):
            for as_object in (False, True):
                self.load()
                target = self.device(spelling) if as_object else spelling
                self.to(self.model, device=target)
                self.assertIs(self.calls[0][1], target)
                self.assertEqual(self.jt.flags.use_cuda, 0 if spelling == "cpu" else 1)

    def test_dtype_only_and_non_blocking_do_not_invent_device(self):
        for dtype in ("float32", "bool"):
            self.load()
            self.to(self.model, dtype=Dtype(dtype), non_blocking=True)
            self.assertEqual(self.calls, [(dtype, None, False)])
            self.assertEqual(self.jt.flags.use_cuda, 0)
        self.load()
        self.assertIs(self.to(self.model, non_blocking=True), self.model)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()

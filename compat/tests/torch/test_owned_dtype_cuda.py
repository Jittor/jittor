"""Genuine public dtype identity and actual CUDA allocation/autograd contracts."""
from contextlib import nullcontext
import copy
import pickle
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pytest
import torch

pytestmark = pytest.mark.cuda


class TestOwnedDtypeCUDA(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        self.candidate = hasattr(torch, "_torch_compat_install_context")
        scope = nullcontext()
        if self.candidate:
            from jittor._runtime.fallback import forbid_backend_fallbacks
            scope = forbid_backend_fallbacks()
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)

    def _resident(self, value):
        self.assertEqual(value.device, torch.device("cuda:0"))
        if self.candidate:
            import jittor as jt
            value.sync()
            self.assertEqual(jt.core.dispatch_context([value]), ("cuda", 0))
            self.assertEqual(jt.core.backend_fallback_count(), 0)
        else:
            torch.cuda.synchronize()

    def test_real_public_type_constructor_and_immutability(self):
        self.assertEqual(torch.dtype.__module__, "torch")
        self.assertEqual(torch.dtype.__qualname__, "dtype")
        self.assertIs(type(torch.dtype), type)
        self.assertEqual(torch.dtype.__bases__, (object,))
        self.assertEqual(torch.dtype.__mro__, (torch.dtype, object))
        self.assertIs(type(torch.float32), torch.dtype)
        self.assertIs(pickle.loads(pickle.dumps(torch.dtype)), torch.dtype)
        for args, kwargs in (((), {}), (("float32",), {}),
                             ((), {"name": "float32"})):
            with self.assertRaises(TypeError):
                torch.dtype(*args, **kwargs)
        with self.assertRaises(TypeError):
            type("DerivedDtype", (torch.dtype,), {})
        with self.assertRaises(AttributeError):
            torch.float32.name = "float64"
        with self.assertRaises(AttributeError):
            object.__setattr__(torch.float32, "name", "float64")
        self._resident(torch.ones(2, dtype=torch.float32, device="cuda:0"))

    def test_canonical_aliases_copy_and_all_pickle_protocols(self):
        for alias, canonical in (("float", "float32"), ("double", "float64"),
                                 ("half", "float16"), ("long", "int64"),
                                 ("int", "int32"), ("short", "int16"),
                                 ("cfloat", "complex64"), ("cdouble", "complex128")):
            self.assertIs(getattr(torch, alias), getattr(torch, canonical))
        for name in ("float16", "bfloat16", "float32", "float64", "int8",
                     "int16", "int32", "int64", "uint8", "bool", "complex64",
                     "complex128", "qint8", "quint8", "qint32"):
            value = getattr(torch, name)
            self.assertIs(type(value), torch.dtype)
            self.assertEqual(str(value), "torch." + name)
            self.assertNotEqual(value, name)
            self.assertNotEqual(value, "torch." + name)
            self.assertEqual({value: name}[value], name)
            self.assertIs(copy.copy(value), value)
            self.assertIs(copy.deepcopy(value), value)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                self.assertIs(pickle.loads(pickle.dumps(value, protocol)), value)
        self._resident(torch.zeros(2, dtype=torch.int64, device="cuda:0"))

    def test_cuda_dtype_parsing_full_input_and_parameter_gradients(self):
        value = torch.tensor([1., 2., 3.], dtype=torch.float32,
                             device="cuda:0", requires_grad=True)
        converted = value.to(torch.float64)
        self._resident(converted)
        self.assertIs(converted.dtype, torch.float64)
        (converted * converted).sum().backward()
        self.assertIsNotNone(value.grad)
        self._resident(value.grad)
        torch.testing.assert_close(value.grad, torch.tensor(
            [2., 4., 6.], device="cuda:0"), atol=1e-5, rtol=1e-5)
        model = torch.nn.Linear(2, 1, bias=True, device="cuda:0", dtype=torch.float32)
        model = model.to(device="cuda:0", dtype=torch.float64)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[2., 3.]], device="cuda:0", dtype=torch.float64))
            model.bias.copy_(torch.tensor([1.], device="cuda:0", dtype=torch.float64))
        x = torch.tensor([[4., 5.]], device="cuda:0", dtype=torch.float64,
                         requires_grad=True)
        output = model(x)
        self._resident(output)
        output.sum().backward()
        for parameter in model.parameters():
            self.assertIs(parameter.dtype, torch.float64)
            self.assertIsNotNone(parameter.grad)
            self._resident(parameter.grad)
        self.assertIsNotNone(x.grad)
        self._resident(x.grad)
        for observed, expected in ((x.grad, [[2., 3.]]),
                                   (model.weight.grad, [[4., 5.]]),
                                   (model.bias.grad, [1.])):
            torch.testing.assert_close(observed, torch.tensor(
                expected, device="cuda:0", dtype=torch.float64), atol=1e-5, rtol=1e-5)

    def test_native_admission_legacy_restore_and_installation_isolation(self):
        # Public allocation is common to both interpreters. Native converter
        # and legacy Jittor pickle checks intentionally exercise only Candidate.
        tensor = torch.tensor([1., 2.], device="cuda:0", dtype=torch.float32)
        self._resident(tensor)
        if self.candidate:
            import jittor as jt
            import numpy as np
            from jittor.compat.torch import types as dtype_owner
            self.assertEqual(jt.core._checked_dtype_name(torch.float32), "float32")
            native = jt.array(np.array([1., 2.], dtype="float32"), dtype=torch.float32)
            native.sync()
            self.assertEqual(jt.core.dispatch_context([native]), ("cuda", 0))
            cast = jt.core.ops.unary(native, torch.float64)
            cast.sync()
            self.assertEqual(jt.core.dispatch_context([cast]), ("cuda", 0))
            self.assertEqual(jt.core._checked_dtype_name(cast.dtype), "float64")
            self.assertEqual(jt.core.backend_fallback_count(), 0)
            self.assertIs(dtype_owner._restore_dtype("float32", True), torch.float32)
            with self.assertRaises(ValueError):
                dtype_owner._restore_dtype("float32", False)
            with self.assertRaises(ValueError):
                dtype_owner._restore_dtype("unregistered_dtype", False)
            with self.assertRaises(ValueError):
                dtype_owner.dtype("unregistered_dtype")
            self.assertNotIn("unregistered_dtype", dtype_owner.dtype._registry)

            class OldDtype(str):
                pass
            private_type = dtype_owner.dtype
            OldDtype.__name__ = OldDtype.__qualname__ = "dtype"
            OldDtype.__module__ = private_type.__module__
            legacy = OldDtype("float32")
            legacy.name, legacy._is_fp = "float32", True
            with patch.object(sys.modules[private_type.__module__], "dtype", OldDtype):
                payload = pickle.dumps(legacy)
            self.assertIs(pickle.loads(payload), torch.float32)

            other = SimpleNamespace()
            first = dtype_owner._make_dtypes(other)
            self.assertIsNot(other.dtype, torch.dtype)
            self.assertIsNot(other.float32, torch.float32)
            self.assertEqual(other.dtype.__bases__, (object,))
            second = dtype_owner._make_dtypes(other)
            self.assertEqual(set(second), set(first))
            self.assertTrue(all(second[name] is first[name] for name in first))
            for alias, canonical in (("float", "float32"), ("double", "float64"),
                                     ("half", "float16"), ("long", "int64"),
                                     ("int", "int32"), ("short", "int16"),
                                     ("cfloat", "complex64"), ("cdouble", "complex128")):
                self.assertIs(second[alias], second[canonical])
            self.assertIs(type(torch.float32), torch.dtype)
            self.assertTrue(dtype_owner._is_dtype(other.float32))
            forged = object.__new__(other.dtype)
            with self.assertRaises(TypeError):
                dtype_owner._is_dtype(forged)
            with self.assertRaises(TypeError):
                jt.core._checked_dtype_name(forged)

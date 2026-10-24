"""Native module scratch follows operands, while Torch factories default to CPU."""

from contextlib import ExitStack
import os
from pathlib import Path
import re
import unittest

import numpy as np
import jittor as jt
import torch


class TestNativeModuleCallRoutingSource(unittest.TestCase):
    """Host-only routing guard; device behavior is tested separately below.

    This class can be AST-extracted without importing Jittor when its core is
    frozen for a hardware run. It intentionally does not claim device evidence.
    """
    def test_nested_search_delegates_before_frontend_changes(self):
        root = next(path for path in Path(__file__).resolve().parents
                    if (path / "src/bindings/pyjt/py_module_call.cc").is_file())
        source = (root / "src/bindings/pyjt/py_module_call.cc").read_text()
        finder = source.split("PyObject* first_var(", 1)[1].split(
            "// Whether `module`", 1)[0]
        self.assertIn("bool& needs_deep_search", finder)
        self.assertIn("needs_deep_search = false;", finder)
        # Both positional and keyword loops stop at a container before any
        # later direct tensor. The direct Var check stays first.
        branches = re.findall(
            r"if \(PyObject_TypeCheck\(value, &PyjtVarHolder.ht_type\)\) return value;"
            r"\s*(?://[^\n]*\n\s*)*if \(PyDict_Check\(value\) \|\| PyList_Check\(value\) \|\| PyTuple_Check\(value\)\)"
            r"\s*\{\s*needs_deep_search = true;\s*return nullptr;\s*\}", finder)
        self.assertEqual(len(branches), 2)
        entry = source.split("PyObject* module_call_native(", 1)[1]
        delegated = entry.index("if (needs_deep_search) return call_with(slow, module, args, kwargs);")
        self.assertLess(entry.index("first_var(args, kwargs, needs_deep_search)"), delegated)
        self.assertLess(delegated, entry.index("set_tensor_frontend_type("))
        self.assertLess(delegated, entry.index("set_tensor_placement_context("))


class TestModuleScratchDevice(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(jt.runtime.scope(use_cuda=int(jt.flags.use_cuda)))
        previous_device = torch.get_default_device()
        self.addCleanup(torch.set_default_device, previous_device)
        torch.set_default_device("cpu")
        devices = os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(",")
        self.accelerator = "npu" if "npu" in devices else "cuda" if "cuda" in devices else None
        if self.accelerator == "npu":
            self.assertTrue(jt.compiler.has_acl, "NPU requested but unavailable")
            self.stack.enter_context(jt.runtime.scope(use_cuda=1, use_acl=1, backend_fallback="error"))
        elif self.accelerator == "cuda":
            self.assertTrue(jt.has_cuda, "CUDA requested but unavailable")
            self.stack.enter_context(jt.runtime.scope(use_cuda=1, backend_fallback="error"))
        else:
            self.stack.enter_context(jt.runtime.scope(use_cuda=0, backend_fallback="error"))
        from jittor._runtime.fallback import forbid_backend_fallbacks
        self.stack.enter_context(forbid_backend_fallbacks())

    def test_nested_keyword_native_scratch_and_public_factory(self):
        class Scratch(torch.nn.Module):
            def forward(self, *, batch):
                x = batch["x"][0]
                return jt.empty(x.shape, x.dtype), torch.tensor([2.0])

        for device in ("cpu", self.accelerator):
            if device is None:
                continue
            with self.subTest(device=device):
                value = torch.ones((2, 3), device=device)
                scratch, public_factory = Scratch()(batch={"x": [value]})
                scratch.sync()
                public_factory.sync()
                self.assertEqual(str(scratch.device).split(":")[0], device)
                self.assertEqual(scratch.location(), "cpu" if device == "cpu" else "device")
                self.assertEqual(str(public_factory.device), "cpu")
                self.assertEqual(public_factory.location(), "cpu")

    def test_embedding_forward_backward_after_module_transfer(self):
        for device in ("cpu", self.accelerator):
            if device is None:
                continue
            with self.subTest(device=device):
                model = torch.nn.Embedding(4, 3).to(device)
                weights = np.arange(12, dtype=np.float32).reshape(4, 3)
                with torch.no_grad():
                    model.weight.copy_(torch.tensor(weights, device=device))
                indices = torch.tensor([[1, 2, 1]], dtype=torch.long, device=device)
                output = model(indices)
                output.sum().backward()
                jt.sync_all(device != "cpu")
                self.assertEqual(output.location(), "cpu" if device == "cpu" else "device")
                self.assertEqual(model.weight.grad.location(), "cpu" if device == "cpu" else "device")
                np.testing.assert_array_equal(output.detach().cpu().numpy(), weights[[[1, 2, 1]]])
                np.testing.assert_array_equal(model.weight.grad.cpu().numpy(),
                                              [[0, 0, 0], [2, 2, 2], [1, 1, 1], [0, 0, 0]])

    def test_native_and_python_calls_preserve_nested_operand_order(self):
        from jittor.compat.torch import nn_frontend
        self.assertIsNotNone(nn_frontend._NATIVE_CALL, "native call must be registered")

        class Scratch(torch.nn.Module):
            def forward(self, *args, **kwargs):
                return jt.empty((2, 3), "float32"), torch.tensor([2.0])

        class Opaque:
            def __init__(self, tensor):
                self.tensor = tensor

            def __iter__(self):
                raise AssertionError("arbitrary objects must not be traversed")

        for first_device in ("cpu", self.accelerator):
            if first_device is None:
                continue
            later_device = self.accelerator if first_device == "cpu" and self.accelerator else "cpu"
            first = torch.ones(2, device=first_device)
            later = torch.ones(2, device=later_device)
            cycle = []
            cycle.extend([cycle, {"x": (first,)}])
            deep = first
            for _ in range(1500):
                deep = [deep]
            cases = (
                ("nested_before_direct", ({"x": [first]}, later), {}),
                ("direct_before_nested", (first, {"x": [later]}), {}),
                ("kwargs_insertion", (), {"batch": {"x": [first]}, "later": later}),
                ("args_before_kwargs", ([first],), {"later": later}),
                ("empty_before_direct", ({"empty": []}, first), {"later": later}),
                ("cycle", (cycle, later), {}),
                ("deep", (deep, later), {}),
                ("opaque", (Opaque(later), first), {}),
            )
            for name, args, kwargs in cases:
                for route in ("native", "python"):
                    with self.subTest(device=first_device, case=name, route=route):
                        module = Scratch()
                        before = jt.core._current_tensor_placement()
                        fallback_before = jt.core.backend_fallback_count()
                        if route == "native":
                            scratch, public = module(*args, **kwargs)
                        else:
                            scratch, public = nn_frontend.python_module_call(module, *args, **kwargs)
                        scratch.sync()
                        public.sync()
                        self.assertEqual(str(scratch.device).split(":")[0], first_device)
                        self.assertEqual(scratch.location(), "cpu" if first_device == "cpu" else "device")
                        self.assertEqual(str(public.device), "cpu")
                        self.assertEqual(public.location(), "cpu")
                        self.assertEqual(jt.core._current_tensor_placement(), before)
                        self.assertEqual(jt.core.backend_fallback_count(), fallback_before)

    def test_nested_exception_restores_native_and_python_call_scopes(self):
        from jittor.compat.torch import nn_frontend
        self.assertIsNotNone(nn_frontend._NATIVE_CALL)

        class Broken(torch.nn.Module):
            def forward(self, batch, later):
                raise ValueError("intentional nested module failure")

        for device in ("cpu", self.accelerator):
            if device is None:
                continue
            first = torch.ones(2, device=device)
            later = torch.ones(2, device=self.accelerator or "cpu")
            cycle = []
            cycle.extend([cycle, first])
            for route in ("native", "python"):
                with self.subTest(device=device, route=route):
                    before = jt.core._current_tensor_placement()
                    fallback_before = jt.core.backend_fallback_count()
                    module = Broken()
                    with self.assertRaisesRegex(ValueError, "intentional nested module failure"):
                        if route == "native":
                            module({"x": cycle}, later)
                        else:
                            nn_frontend.python_module_call(module, {"x": cycle}, later)
                    self.assertEqual(jt.core._current_tensor_placement(), before)
                    public = torch.ones(2)
                    public.sync()
                    self.assertEqual(str(public.device), "cpu")
                    self.assertEqual(public.location(), "cpu")
                    self.assertEqual(jt.core.backend_fallback_count(), fallback_before)

    def test_native_gather_scratch_follows_operand(self):
        device = self.accelerator or "cpu"
        values = torch.tensor([[0.0, 1.0, 2.0]], device=device, requires_grad=True)
        index = torch.tensor([[2, 0, 2]], device=device, dtype=torch.long)
        result = torch.gather(values, 1, index)
        result.sum().backward()
        jt.sync_all(device != "cpu")
        self.assertEqual(result.location(), "cpu" if device == "cpu" else "device")
        self.assertEqual(values.grad.location(), "cpu" if device == "cpu" else "device")
        np.testing.assert_array_equal(result.detach().cpu().numpy(), [[2, 0, 2]])
        np.testing.assert_array_equal(values.grad.cpu().numpy(), [[1, 0, 2]])

    def test_exception_restores_ambient_placement(self):
        class Broken(torch.nn.Module):
            def forward(self, x):
                raise ValueError("intentional module failure")

        value = torch.ones(2, device=self.accelerator or "cpu")
        before = jt.core._current_tensor_placement()
        with self.assertRaisesRegex(ValueError, "intentional module failure"):
            Broken()(value)
        self.assertEqual(jt.core._current_tensor_placement(), before)
        self.assertEqual(str(torch.ones(2).device), "cpu")


if __name__ == "__main__":
    unittest.main()

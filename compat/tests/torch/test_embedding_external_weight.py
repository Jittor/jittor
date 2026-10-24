"""Public Embedding supplied-weight ownership; v213 repr-free CPU/CUDA proposal.

Successor of v212: explicit boolean identity/type assertions prevent unittest
from dumping Tensor repr values when a constructor assertion fails. Math,
inputs, device checks and fixed full-array tolerances are unchanged.

Run a single backend/node per interpreter and independent UNIT caches with
JITTOR_TEST_DEVICES=cpu|cuda. The same file is runnable in independent official
Torch (no Jittor) and the normal candidate bootstrap. Fixed arrays stay in RAM;
all values and all gradients are compared, without hash-only shortcuts. Not
yet device-validated; not an embedding-retention or complete PPO acceptance.
Proposed permanent owner: compat/tests/torch/test_embedding_external_weight.py.
"""
from contextlib import ExitStack, contextmanager
import os
import sys
import unittest

import numpy as np


WEIGHT = np.arange(12, dtype=np.float32).reshape(4, 3) / 8 - .5
TOKENS = np.array([[0, 2, 2], [3, 0, 1]], dtype=np.int64)
COTANGENT = (np.arange(18, dtype=np.float32).reshape(2, 3, 3) - 8) / 8


class TestEmbeddingExternalWeight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.jt = sys.modules.get("jittor")
        if cls.jt is None:
            assert hasattr(torch, "_C")
            assert not any(name.startswith("jittor.") for name in sys.modules)
        else:
            from jittor.compat.torch.publication import namespace_owner
            assert namespace_owner(torch) is cls.jt and torch.Tensor is not cls.jt.Var

    @contextmanager
    def _scope(self, device):
        selected = os.environ.get("JITTOR_TEST_DEVICES", "cpu")
        if selected not in ("cpu", "cuda"):
            raise ValueError("select exactly one CPU or CUDA node per process")
        if selected != device:
            self.skipTest("node not selected by JITTOR_TEST_DEVICES")
        with ExitStack() as stack:
            if device == "cuda":
                self.assertTrue(self.torch.cuda.is_available(), "requested real CUDA unavailable")
                self.assertEqual(self.torch.cuda.current_device(), 0)
            if self.jt is not None:
                if device == "cuda":
                    self.assertTrue(self.jt.compiler.has_cuda and self.jt.compiler.is_cuda)
                    self.assertFalse(any(getattr(self.jt.compiler, name, False)
                                         for name in ("has_acl", "has_rocm", "has_corex")))
                from jittor._runtime.fallback import forbid_backend_fallbacks
                stack.enter_context(self.jt.runtime.scope(
                    use_cuda=int(device == "cuda"), device_id=0 if device == "cuda" else -1,
                    backend_fallback="error"))
                stack.enter_context(forbid_backend_fallbacks())
                self.assertEqual(self.jt.core.backend_fallback_count(), 0)
            yield
            if self.jt is not None:
                self.assertEqual(self.jt.core.backend_fallback_count(), 0)

    def _resident(self, tensor, device):
        self.assertEqual(tensor.device.type, device)
        if device == "cuda":
            self.assertEqual(tensor.device.index, 0)
        if self.jt is not None:
            native = getattr(tensor, "_var", tensor)
            native.sync()
            index = -1 if device == "cpu" else 0
            self.assertEqual(self.jt.core.dispatch_context([native]), (device, index))
            self.assertEqual(native.placement_backend, 0 if device == "cpu" else 1)
            self.assertEqual(native.location(), "cpu" if device == "cpu" else "device")
            self.assertEqual(self.jt.core.backend_fallback_count(), 0)
        elif device == "cuda":
            self.torch.cuda.synchronize(tensor.device)

    @staticmethod
    def _flags(value):
        return (value.requires_grad, value.is_leaf, value.grad_fn is None,
                tuple(value.shape), value.dtype, str(value.device))

    def _array(self, value, device):
        before = self._flags(value)
        self._resident(value, device)
        with self.torch.no_grad():
            clone = value.clone()
            self.assertTrue(clone is not value, "snapshot_clone_holder")
            self.assertFalse(clone.requires_grad)
            self._resident(clone, device)
            host = clone.cpu()
            self._resident(host, "cpu")
            array = np.array(host.numpy(), copy=True)
        self.assertEqual(self._flags(value), before)
        self._resident(value, device)
        self.assertTrue(bool(np.isfinite(array).all()), "snapshot_finite")
        return array

    def _equal(self, value, expected, device):
        actual = self._array(value, device)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        self.assertTrue(bool(np.all(error <= 1e-5 + 1e-5 * np.abs(expected))), "full_array_fixed_1e5")

    def _case(self, device, source_grad, freeze, parameter_source):
        torch = self.torch
        source = torch.tensor(WEIGHT, dtype=torch.float32, device=device, requires_grad=source_grad)
        if parameter_source:
            source = torch.nn.Parameter(source, requires_grad=source_grad)
        if source_grad:
            (source * torch.tensor(np.ones_like(WEIGHT), device=device)).sum().backward()
            self.assertTrue(source.grad is not None, "source_gradient_exists")
        source_flags = self._flags(source)
        original_grad = source.grad
        tokens = torch.tensor(TOKENS, dtype=torch.int64, device=device)
        # No test workaround sets requires_grad=True on supplied _weight.
        # The constructor owns the new Parameter's default trainability.
        module = torch.nn.Embedding(4, 3, _weight=source, _freeze=freeze)
        self.assertTrue(isinstance(module.weight, torch.nn.Parameter), "embedding_weight_must_be_parameter")
        self.assertTrue(module.weight is not source, "embedding_must_own_distinct_holder")
        self.assertEqual(module.weight.requires_grad, not freeze)
        self.assertTrue(module.weight.is_leaf)
        self.assertTrue(module.weight.grad_fn is None, "embedding_leaf_grad_fn")
        self.assertEqual(self._flags(source), source_flags)
        self.assertTrue(source.grad is original_grad, "source_grad_identity_unchanged")
        self.assertEqual(module.weight.data_ptr(), source.data_ptr())
        self.assertEqual(list(dict(module.named_parameters())), ["weight"])
        self.assertTrue(dict(module.named_parameters())["weight"] is module.weight, "registered_parameter_identity")
        self._equal(source, WEIGHT, device)
        self._equal(module.weight, WEIGHT, device)
        output = module(tokens)
        self.assertEqual(output.requires_grad, not freeze)
        self._equal(output, WEIGHT[TOKENS], device)
        if not freeze:
            cotangent = torch.tensor(COTANGENT, dtype=torch.float32, device=device)
            (output * cotangent).sum().backward()
            self.assertTrue(module.weight.grad is not None, "embedding_gradient_exists")
            expected_gradient = np.zeros_like(WEIGHT)
            np.add.at(expected_gradient, TOKENS.reshape(-1), COTANGENT.reshape(-1, 3))
            self._equal(module.weight.grad, expected_gradient, device)
            optimizer = torch.optim.SGD(module.parameters(), lr=.125)
            optimizer.step()
            expected_updated = WEIGHT - np.float32(.125) * expected_gradient
            self._equal(module.weight, expected_updated, device)
            self._equal(source, expected_updated, device)
            self.assertEqual(module.weight.data_ptr(), source.data_ptr())
        else:
            self.assertTrue(module.weight.grad is None, "frozen_weight_gradient_none")
            self._equal(source, WEIGHT, device)
        self.assertEqual(self._flags(source), source_flags)
        self.assertTrue(source.grad is original_grad, "source_grad_identity_unchanged")
        if original_grad is not None:
            self._equal(original_grad, np.ones_like(WEIGHT), device)
        self.assertFalse(tokens.requires_grad)
        self.assertTrue(tokens.grad is None, "integer_token_gradient_none")
        self._resident(tokens, device)

    def _run(self, device):
        with self._scope(device):
            for source_grad in (False, True):
                for freeze in (False, True):
                    for parameter_source in (False, True):
                        with self.subTest(source_grad=source_grad, freeze=freeze, parameter_source=parameter_source):
                            self._case(device, source_grad, freeze, parameter_source)

    def test_cpu(self):
        self._run("cpu")

    def test_cuda(self):
        self._run("cuda")


if __name__ == "__main__":
    unittest.main()

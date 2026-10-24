"""Full reductions must stay scalar when composed into row-wise log-probabilities.

Select JITTOR_LOGSUMEXP_RUNTIME=oracle in an independent binary-Torch process;
candidate requires JITTOR_TORCH_SHIM=1. NumPy float64 formulas independently
check every forward value and every FP32 input gradient. These tests cover
finite inputs, not the separate empty-input or NaN/Inf contracts.
"""
from contextlib import ExitStack, contextmanager
import importlib.util
import os
import unittest

import numpy as np


class TestLogsumexpScalar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runtime = os.environ.get("JITTOR_LOGSUMEXP_RUNTIME",
                                 "candidate" if os.environ.get("JITTOR_TORCH_SHIM") == "1" else "oracle")
        cls.jt = None
        if runtime == "candidate":
            import jittor as jt
            import torch
            from jittor.compat.torch.publication import namespace_owner
            from jittor.compat.shim.runtime import activation_status

            assert namespace_owner(torch) is jt and activation_status(jt).active
            assert torch.Tensor is not jt.Var
            cls.jt = jt
        elif runtime == "oracle":
            import torch

            assert hasattr(torch, "_C") and not hasattr(torch, "_torch_compat_install_context")
        else:
            raise ValueError("JITTOR_LOGSUMEXP_RUNTIME must be candidate or oracle")
        cls.torch = torch

    @contextmanager
    def _device(self, device):
        with ExitStack() as stack:
            jt, torch = self.jt, self.torch
            if jt is not None:
                if device == "npu" and not jt.compiler.has_acl:
                    self.skipTest("ACL backend is not installed")
                stack.enter_context(jt.runtime.scope(use_cuda=int(device == "npu"), use_acl=int(device == "npu")))
                if device == "npu":
                    self.assertGreater(jt.core.get_device_count(), 0)
                    index = jt.core.current_device()
                    self.assertGreaterEqual(index, 0)
                from jittor._runtime.fallback import forbid_backend_fallbacks

                before = jt.core.backend_fallback_count()
                stack.enter_context(forbid_backend_fallbacks())
            elif device == "npu":
                if importlib.util.find_spec("torch_npu") is None:
                    self.skipTest("native torch_npu is not installed")
                import torch_npu  # noqa: F401

                self.assertTrue(torch.npu.is_available())
                index = torch.npu.current_device()
            target = "npu:%d" % index if device == "npu" else "cpu"
            yield target
            if jt is not None:
                jt.sync_all(True)
                self.assertEqual(jt.core.backend_fallback_count(), before)

    def _placement(self, value, target):
        if self.jt is not None:
            native = getattr(value, "_var", value)
            native.sync()
            self.assertEqual(native.location(), "cpu" if target == "cpu" else "device")
            if target != "cpu":
                self.assertEqual(native.device_id, int(target.split(":")[1]))
        self.assertEqual(str(value.device), target)

    def _check(self, value, expected, target):
        self._placement(value, target)
        self.assertEqual(tuple(value.shape), expected.shape)
        self.assertEqual(value.dtype, self.torch.float32)
        np.testing.assert_allclose(value.detach().cpu().numpy(), expected, atol=2e-6, rtol=2e-5)

    def _reductions(self, device):
        torch = self.torch
        cases = (((5,), 0), ((5,), -1), ((5,), (0,)),
                 ((2, 3), (0, 1)), ((2, 3), (-1, -2)),
                 ((2, 1, 3), (0, 1, 2)), ((2, 1, 3), (-1, -2, -3)),
                 ((2, 1, 3), (0, 2)), ((2, 3), -1))
        with self._device(device) as target:
            for shape, dim in cases:
                for keepdim in (False, True):
                    for method in (False, True):
                        with self.subTest(shape=shape, dim=dim, keepdim=keepdim, method=method):
                            raw = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) * 0.17 - 1.3)
                            reference = raw.astype(np.float64)
                            maximum = reference.max(axis=dim, keepdims=True)
                            exponential = np.exp(reference - maximum)
                            total = exponential.sum(axis=dim, keepdims=True)
                            expected_keep = maximum + np.log(total)
                            expected = expected_keep if keepdim else np.squeeze(expected_keep, axis=dim)
                            seed = ((np.arange(expected.size).reshape(expected.shape) + 1) / 3.0).astype(np.float32)
                            seed_keep = seed.reshape(expected_keep.shape)
                            expected_gradient = exponential / total * seed_keep
                            source = torch.tensor(raw, dtype=torch.float32, device=target, requires_grad=True)
                            output = (source.logsumexp(dim, keepdim=keepdim) if method else
                                      torch.logsumexp(source, dim=dim, keepdim=keepdim))
                            self.assertEqual(tuple(output.shape), expected.shape)
                            self.assertEqual(output.ndim, expected.ndim)
                            cotangent = torch.tensor(seed, device=target)
                            gradient, = torch.autograd.grad((output * cotangent).sum(), (source,))
                            self._placement(source, target)
                            self._placement(cotangent, target)
                            self._check(output, expected, target)
                            self._check(gradient, expected_gradient, target)

    def _row_logprobs(self, device):
        torch = self.torch
        raw = np.array([[1., -2., 0.5, 3., -1.], [-1., 2., 3., -0.5, 1.],
                        [0.1, 0.2, -0.3, 1.5, -1.7], [2., 0.4, -1., 3., 1.]], dtype=np.float32)
        labels = np.array([3, 0, 2, 4], dtype=np.int64)
        seed = np.array([1., -2., 0.5, 1.5], dtype=np.float32)
        reference = raw.astype(np.float64)
        maximum = reference.max(axis=-1, keepdims=True)
        exponential = np.exp(reference - maximum)
        total = exponential.sum(axis=-1, keepdims=True)
        expected_normalizers = (maximum + np.log(total)).reshape(4)
        expected = reference[np.arange(4), labels] - expected_normalizers
        expected_gradient = -exponential / total
        expected_gradient[np.arange(4), labels] += 1
        expected_gradient *= seed[:, None]
        with self._device(device) as target:
            source = torch.tensor(raw, device=target, requires_grad=True)
            indices = torch.tensor(labels, dtype=torch.int64, device=target)
            selected = torch.gather(source, -1, indices.unsqueeze(-1)).squeeze(-1)
            rows = [torch.logsumexp(row, dim=-1) for row in source]
            for row in rows:
                self.assertEqual(tuple(row.shape), ())
                self.assertEqual(row.ndim, 0)
            normalizers = torch.stack(rows)
            output = selected - normalizers
            self.assertEqual(tuple(normalizers.shape), (4,))
            self.assertEqual(tuple(output.shape), (4,), "row log-probabilities must not broadcast to NxN")
            cotangent = torch.tensor(seed, device=target)
            gradient, = torch.autograd.grad((output * cotangent).sum(), (source,))
            for value in (source, indices, selected, cotangent):
                self._placement(value, target)
            self._check(normalizers, expected_normalizers, target)
            self._check(output, expected, target)
            self._check(gradient, expected_gradient, target)

    def test_cpu_full_and_partial_reduction_rank_values_and_gradients(self):
        self._reductions("cpu")

    def test_cpu_stacked_row_logprobs_and_all_input_gradients(self):
        self._row_logprobs("cpu")

    def test_npu_full_and_partial_reduction_rank_values_and_gradients(self):
        self._reductions("npu")

    def test_npu_stacked_row_logprobs_and_all_input_gradients(self):
        self._row_logprobs("npu")


if __name__ == "__main__":
    unittest.main()

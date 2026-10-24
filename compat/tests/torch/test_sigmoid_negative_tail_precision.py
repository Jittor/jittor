"""Public CPU/CUDA Sigmoid forward and first-order negative-tail regressions.

One selected CPU/CUDA backend per process, independent native/candidate public
Torch APIs. F32/F64 only; native source formula intentionally allows exp(-x)
overflow, unlike an alternative stable logistic algorithm. Arrays stay RAM-only.
No ACL/ROCm/F16/BF16/complex or full-PPO claim.
"""
from contextlib import ExitStack, contextmanager
import os
import sys
import unittest

import numpy as np


def fixture(dtype):
    tail = [-1000, -700, -350, -100] if dtype == "float64" else [-100]
    x = np.asarray(tail + [-80, -40, -31, -30, -29.5, -1, -.0, .0, 1, 20, 40, 1000], dtype=dtype)
    k = np.arange(x.size, dtype=np.int64)
    dy = (((k % 9) - 4) / 4).astype(dtype)
    dy[dy == 0] = .375
    return x, dy


def positive_fixture(dtype):
    x = np.array([-40, -16, -12, -8, -1, -.0, 0, 1, 8, 10, 12, 14, 16, 17, 20, 30, 40], dtype=dtype)
    x = np.broadcast_to(x, (4, x.size)).copy()
    dy = np.empty_like(x)
    dy[0] = 1
    dy[1] = 1000
    dy[2] = np.where(np.arange(x.shape[1]) % 2, -1000, 1000)
    dy[3] = np.asarray([.5, -1.25, 2, -.75, 1.5, -.5, .875, -2, .25, -1.5, 2.25, -.375, .625, -1, .75, -2.5, 1.25], dtype=dtype)
    return x, dy

def native_source_reference(x, dy):
    # Simulate the declared real scalar precision, not stable FP64 logistic.
    # Overflow is part of this specific ATen formula and is not hidden failure.
    one = x.dtype.type(1)
    with np.errstate(over="ignore", under="ignore"):
        y = one / (one + np.exp(-x))
    work = y.astype(np.float64)
    grad = dy.astype(np.float64) * (1 - work) * work
    return y, grad


def mismatch_count(actual, reference, *, atol=1e-5, rtol=1e-5):
    reference = np.asarray(reference, dtype=np.float64)
    if actual.shape != reference.shape:
        raise ValueError("shape differs; complete arrays required")
    failed = (~np.isfinite(actual) | ~np.isfinite(reference)
              | (np.abs(actual.astype(np.float64) - reference) > atol + rtol * np.abs(reference)))
    return int(np.count_nonzero(failed))


class TestSigmoidNegativeTailPrecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch, cls.jt = torch, None
        runtime = os.environ.get("JITTOR_SIGMOID_RUNTIME", "candidate" if os.environ.get("JITTOR_TORCH_SHIM") == "1" else "oracle")
        if runtime not in ("oracle", "candidate"):
            raise ValueError("JITTOR_SIGMOID_RUNTIME=oracle|candidate required")
        if runtime == "oracle":
            assert hasattr(torch, "_C") and not hasattr(torch, "_torch_compat_install_context")
            assert not any(name == "jittor" or name.startswith("jittor.") for name in sys.modules)
        else:
            import jittor as jt
            from jittor.compat.torch import namespace_owner
            from jittor.compat.shim.runtime import activation_status
            assert namespace_owner(torch) is jt and activation_status(jt).active
            assert torch.Tensor is not jt.Var
            cls.jt = jt

    @contextmanager
    def _scope(self, device):
        selected = os.environ.get("JITTOR_TEST_DEVICES", "cpu")
        if selected not in ("cpu", "cuda"):
            raise ValueError("exact single JITTOR_TEST_DEVICES=cpu|cuda required")
        if device != selected:
            self.skipTest("node excluded by explicit single-backend process selection")
        with ExitStack() as stack:
            if device == "cuda":
                self.assertTrue(self.torch.cuda.is_available(), "requested real CUDA unavailable")
                self.torch.cuda.set_device(0)
                self.assertEqual(self.torch.cuda.current_device(), 0)
                if self.jt is not None:
                    self.assertTrue(self.jt.compiler.has_cuda and self.jt.compiler.is_cuda)
                    self.assertFalse(any(getattr(self.jt.compiler, name, False) for name in ("has_acl", "has_rocm", "has_corex")))
            self.torch.backends.cuda.matmul.allow_tf32 = False
            self.torch.backends.cudnn.allow_tf32 = False
            self.torch.backends.cudnn.benchmark = False
            self.torch.set_float32_matmul_precision("highest")
            self.assertIs(self.torch.backends.cuda.matmul.allow_tf32, False)
            self.assertIs(self.torch.backends.cudnn.allow_tf32, False)
            self.assertIs(self.torch.backends.cudnn.benchmark, False)
            self.assertEqual(self.torch.get_float32_matmul_precision(), "highest")
            if self.jt is not None:
                from jittor._runtime.fallback import forbid_backend_fallbacks
                stack.enter_context(self.jt.runtime.scope(use_cuda=int(device == "cuda"),
                    backend_fallback="error", cuda_kernel_math="strict"))
                stack.enter_context(forbid_backend_fallbacks())
                self.assertEqual(self.jt.flags.cuda_kernel_math, "strict")
                self.assertEqual(self.jt.core.backend_fallback_count(), 0)
            yield
            if self.jt is not None:
                self.assertEqual(self.jt.core.backend_fallback_count(), 0)

    def _resident(self, tensor, device):
        self.assertEqual(tensor.device.type, device)
        ordinal = 0 if device == "cuda" else -1
        self.assertEqual(tensor.device.index, 0 if device == "cuda" else None)
        if self.jt is None:
            if device == "cuda":
                self.torch.cuda.synchronize(tensor.device)
            return
        native = getattr(tensor, "_var", tensor)
        native.sync()
        self.assertEqual(self.jt.core.dispatch_context([native]), (device, ordinal))
        self.assertEqual(int(native.device_id), ordinal)
        self.assertEqual(int(native.placement_backend), int(device == "cuda"))
        self.assertEqual(native.location(), "device" if device == "cuda" else "cpu")

    def _snapshot(self, tensor, device):
        self._resident(tensor, device)
        before = (tensor.requires_grad, tensor.is_leaf, tensor.dtype, tuple(tensor.shape))
        with self.torch.no_grad():
            clone = tensor.clone()
            self.assertIsNot(clone, tensor)
            self.assertIs(clone.requires_grad, False)
            self._resident(clone, device)
            host = clone.cpu()
            self._resident(host, "cpu")
            array = np.array(host.numpy(), copy=True)
        self._resident(tensor, device)
        self.assertEqual(before, (tensor.requires_grad, tensor.is_leaf, tensor.dtype, tuple(tensor.shape)))
        return array

    def _check(self, actual, reference, *, atol=1e-5, rtol=1e-5):
        self.assertEqual(mismatch_count(actual, reference, atol=atol, rtol=rtol), 0,
                         "complete-array numerical mismatch; raw arrays remain RAM-only")

    def _case(self, device, dtype, entry):
        torch = self.torch
        x_np, dy_np = fixture(dtype)
        ref_y, ref_dx = native_source_reference(x_np, dy_np)
        target = torch.device("cuda", 0) if device == "cuda" else torch.device("cpu")
        x = torch.tensor(x_np.copy(), dtype=getattr(torch, dtype), device=target, requires_grad=True)
        dy = torch.tensor(dy_np.copy(), dtype=getattr(torch, dtype), device=target, requires_grad=False)
        self.assertTrue(x.is_leaf and x.requires_grad and x.grad is None)
        self.assertIs(dy.requires_grad, False)
        self.assertTrue(self._snapshot(x, device).tobytes() == x_np.tobytes(), "input full bytes differ")
        self.assertTrue(self._snapshot(dy, device).tobytes() == dy_np.tobytes(), "cotangent full bytes differ")
        y = torch.sigmoid(x) if entry == "function" else x.sigmoid()
        self._resident(y, device)
        dx, = torch.autograd.grad(y, x, grad_outputs=dy, create_graph=False, retain_graph=False, allow_unused=False)
        self.assertIsNotNone(dx)
        self.assertIs(dx.requires_grad, False)
        self.assertEqual(dx.dtype, x.dtype)
        self.assertEqual(tuple(dx.shape), tuple(x.shape))
        actual_y, actual_dx = self._snapshot(y, device), self._snapshot(dx, device)
        self._check(actual_y, ref_y)
        self._check(actual_dx, ref_dx)
        # Normal (not subnormal) negative tails cannot be hidden by 1e-5 ATOL.
        tail = (x_np <= -31) & (x_np >= (-700 if dtype == "float64" else -80))
        tolerance = 2e-14 if dtype == "float64" else 2e-6
        self._check(actual_y[tail], ref_y[tail], atol=0, rtol=tolerance)
        self._check(actual_dx[tail], ref_dx[tail], atol=0, rtol=tolerance)
        zero = x_np == (-1000 if dtype == "float64" else -100)
        if bool(zero.any()):
            self.assertEqual(int(np.count_nonzero(actual_y[zero])), 0, "native formula overflow must give zero")
            self.assertEqual(int(np.count_nonzero(actual_dx[zero])), 0, "output-based derivative must give zero")
        self.assertTrue(self._snapshot(x, device).tobytes() == x_np.tobytes(), "math changed input bytes")
        self.assertTrue(self._snapshot(dy, device).tobytes() == dy_np.tobytes(), "math changed cotangent bytes")
        self.assertTrue(x.requires_grad and x.is_leaf and x.grad is None)
        self.assertIs(dy.requires_grad, False)

    def _run(self, device):
        with self._scope(device):
            for dtype in ("float32", "float64"):
                for entry in ("function", "method"):
                    with self.subTest(device=device, dtype=dtype, entry=entry):
                        self._case(device, dtype, entry)

    def test_cpu(self):
        self._run("cpu")

    def test_cuda(self):
        self._run("cuda")

    def _positive_case(self, device, dtype, entry):
        # Identical predeclared x/dY to private155; no threshold-driven rescale.
        torch = self.torch
        x_np, dy_np = positive_fixture(dtype)
        target = torch.device("cuda", 0) if device == "cuda" else torch.device("cpu")
        x = torch.tensor(x_np.copy(), dtype=getattr(torch, dtype), device=target, requires_grad=True)
        dy = torch.tensor(dy_np.copy(), dtype=getattr(torch, dtype), device=target, requires_grad=False)
        self.assertTrue(x.is_leaf and x.requires_grad and x.grad is None)
        self.assertIs(dy.requires_grad, False)
        self.assertTrue(self._snapshot(x, device).tobytes() == x_np.tobytes(), "input full bytes differ")
        self.assertTrue(self._snapshot(dy, device).tobytes() == dy_np.tobytes(), "cotangent full bytes differ")
        y = torch.sigmoid(x) if entry == "function" else x.sigmoid()
        self._resident(y, device)
        # Materialize actual rounded Y before backward, as in private155.
        actual_y = self._snapshot(y, device)
        self.assertEqual(actual_y.dtype.name, dtype)
        dx, = torch.autograd.grad(y, x, grad_outputs=dy, create_graph=False, retain_graph=False, allow_unused=False)
        self.assertIsNotNone(dx)
        self.assertIs(dx.requires_grad, False)
        self.assertEqual(dx.dtype, x.dtype)
        self.assertEqual(tuple(dx.shape), tuple(x.shape))
        self._resident(dx, device)
        actual_dx = self._snapshot(dx, device)
        # ATen real F32/F64 order; use actual Y rather than assume exp parity.
        one = actual_y.dtype.type(1)
        ref_dx = (dy_np * (one - actual_y)) * actual_y
        with np.errstate(over="ignore"):
            ref_y = one / (one + np.exp(-x_np))
        positive = x_np >= 8
        nonzero = ref_dx[positive] != 0
        self.assertTrue(bool((np.abs(ref_dx[positive][nonzero]) >= np.finfo(ref_dx.dtype).tiny).all()),
                        "subnormal reference must not be silently excluded")
        tolerance = 2e-14 if dtype == "float64" else 2e-6
        # Evaluate all complete arrays and exact-zero positive entries before
        # one numerical assertion; a regular failure cannot skip the tail.
        failures = {
            "forward_regular": mismatch_count(actual_y, ref_y),
            "vjp_regular": mismatch_count(actual_dx, ref_dx),
            "positive_vjp": mismatch_count(actual_dx[positive], ref_dx[positive], atol=0, rtol=tolerance),
            "positive_exact_zero": int(np.count_nonzero(actual_dx[positive][~nonzero])),
        }
        self.assertTrue(self._snapshot(x, device).tobytes() == x_np.tobytes(), "math changed input bytes")
        self.assertTrue(self._snapshot(dy, device).tobytes() == dy_np.tobytes(), "math changed cotangent bytes")
        self.assertTrue(x.requires_grad and x.is_leaf and x.grad is None)
        self.assertIs(dy.requires_grad, False)
        self.assertEqual(sum(failures.values()), 0,
                         "complete positive-saturation mismatches (counts only): " + str(failures))

    def _run_positive(self, device):
        with self._scope(device):
            for dtype in ("float32", "float64"):
                for entry in ("function", "method"):
                    with self.subTest(device=device, dtype=dtype, entry=entry):
                        self._positive_case(device, dtype, entry)

    def test_cpu_positive(self):
        self._run_positive("cpu")

    def test_cuda_positive(self):
        self._run_positive("cuda")


if __name__ == "__main__":
    unittest.main()

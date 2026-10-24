"""Public CPU/CUDA native-operation placement and factory contracts.

One explicitly selected backend node per process; independent official Torch
can run the same public checks. Arrays remain in RAM; failures omit values.
This adds no CORE whitelist entry and makes no PPO/L1-L5 claim.
"""
from contextlib import ExitStack, contextmanager, nullcontext
import os
import sys
import unittest

import numpy as np


class TestNativeOperationPlacement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch, cls.jt = torch, None
        runtime = os.environ.get("JITTOR_PLACEMENT_RUNTIME",
            "candidate" if os.environ.get("JITTOR_TORCH_SHIM") == "1" else "oracle")
        if runtime == "oracle":
            assert hasattr(torch, "_C") and not hasattr(torch, "_torch_compat_install_context")
            assert not any(n == "jittor" or n.startswith("jittor.") for n in sys.modules)
        elif runtime == "candidate":
            import jittor as jt
            from jittor.compat.torch import namespace_owner
            from jittor.compat.shim.runtime import activation_status
            assert namespace_owner(torch) is jt and activation_status(jt).active
            assert torch.Tensor is not jt.Var
            cls.jt = jt
        else:
            raise ValueError("JITTOR_PLACEMENT_RUNTIME=oracle|candidate required")

    @contextmanager
    def _scope(self, device, mode):
        selected = os.environ.get("JITTOR_TEST_DEVICES")
        if selected not in ("cpu", "cuda"):
            raise ValueError("explicit single JITTOR_TEST_DEVICES required")
        if selected != device:
            self.skipTest("node excluded by explicit device selection")
        with ExitStack() as stack:
            if device == "cuda":
                self.assertTrue(self.torch.cuda.is_available(), "actual CUDA required")
                self.torch.cuda.set_device(0)
                self.assertEqual(self.torch.cuda.current_device(), 0)
            if self.jt is not None:
                from jittor._runtime.fallback import forbid_backend_fallbacks
                if device == "cuda":
                    self.assertTrue(self.jt.compiler.has_cuda and self.jt.compiler.is_cuda)
                    self.assertFalse(any(getattr(self.jt.compiler, n, False) for n in ("has_acl", "has_rocm", "has_corex")))
                stack.enter_context(self.jt.runtime.scope(use_cuda=int(device == "cuda"),
                    device_id=0 if device == "cuda" else -1,
                    cuda_kernel_math=mode, backend_fallback="error", amp_reg=0))
                stack.enter_context(forbid_backend_fallbacks())
                self.assertEqual(self.jt.core.backend_fallback_count(), 0)
            yield
            if self.jt is not None:
                self.assertEqual(self.jt.core.backend_fallback_count(), 0)

    def _placement(self, value, device, *, sync=False):
        self.assertTrue(isinstance(value, self.torch.Tensor), "public Tensor required")
        if sync and self.jt is not None: value.sync()
        elif sync and device == "cuda": self.torch.cuda.synchronize(value.device)
        self.assertEqual(value.device.type, device)
        self.assertEqual(value.device.index, 0 if device == "cuda" else None)
        if self.jt is not None:
            self.assertEqual(value.placement_backend, int(device == "cuda"))
            self.assertEqual(int(value.device_id), 0 if device == "cuda" else -1)
            self.assertEqual(value.location(), "device" if device == "cuda" else "cpu")
            self.assertEqual(self.jt.core.dispatch_context([value]), (device, 0 if device == "cuda" else -1))

    def _read(self, value, device):
        self._placement(value, device, sync=True)
        before = (value.requires_grad, value.is_leaf, tuple(value.shape), value.dtype)
        with self.torch.no_grad():
            copy = value.clone()
            self.assertTrue(copy is not value and copy.requires_grad is False,
                            "independent stopped snapshot required")
            self._placement(copy, device, sync=True)
            host = copy.cpu()
            self._placement(host, "cpu", sync=True)
            array = np.array(host.numpy(), copy=True)
        self._placement(value, device)
        self.assertEqual(before, (value.requires_grad, value.is_leaf, tuple(value.shape), value.dtype))
        return array

    def _same(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(bool(np.isfinite(actual).all() and np.isfinite(expected).all()),
                        "finite fixture required")
        bad = np.abs(actual.astype(np.float64) - expected.astype(np.float64)) > 1e-5 + 1e-5 * np.abs(expected)
        self.assertEqual(int(np.count_nonzero(bad)), 0, "full-array comparison failed; values omitted")

    def _case(self, device, kind, requested, stopped):
        torch = self.torch
        raw = np.array([.25, 1., 4.], dtype=np.float32)
        # Do not materialize or snapshot x before the original operation.
        x = torch.tensor(raw.copy(), dtype=torch.float32, device=device, requires_grad=requested)
        flags = (x.requires_grad, x.is_leaf, tuple(x.shape), x.dtype)
        with torch.device("cpu"):
            with torch.no_grad() if stopped else nullcontext():
                if kind == "function":
                    y, expected = torch.rsqrt(x), 1. / np.sqrt(raw)
                elif kind == "method":
                    y, expected = x.rsqrt(), 1. / np.sqrt(raw)
                elif kind == "kwargs":
                    y, expected = torch.fft.fftshift(input=x, dim=0), np.fft.fftshift(raw)
                elif kind == "nested":
                    y = torch.meshgrid([x, x], indexing="ij")[0]
                    expected = np.repeat(raw[:, None], len(raw), axis=1)
                else: raise ValueError("undeclared fixture")
        self._placement(y, device, sync=True)
        self._placement(x, device)  # No x.sync/readback may mask its original state.
        if kind != "nested":
            self.assertIs(y.requires_grad, bool(requested and not stopped))
        # meshgrid yields views: Torch view/no_grad flags are not the same
        # contract as a new arithmetic result; do not redefine them here.
        self._same(self._read(y, device), expected)
        self._placement(x, device)  # Check again after the independent y readback.
        if requested and not stopped:
            cot = np.arange(1, int(np.prod(expected.shape)) + 1, dtype=np.float32).reshape(expected.shape)
            dy = torch.tensor(cot.copy(), dtype=torch.float32, device=device)
            dx, = torch.autograd.grad(y, (x,), grad_outputs=dy)
            if kind in ("function", "method"): reference = -.5 * cot * raw ** -1.5
            elif kind == "kwargs": reference = np.fft.ifftshift(cot)
            else: reference = cot.sum(axis=1)
            self._same(self._read(dx, device), reference)
        self.assertEqual(flags, (x.requires_grad, x.is_leaf, tuple(x.shape), x.dtype))
        self.assertTrue(x.grad is None, "functional VJP must not accumulate into x.grad")
        self.assertTrue(self._read(x, device).tobytes() == raw.tobytes(), "original input bytes changed")

    def _factories(self, device):
        torch = self.torch
        self.assertEqual(torch.get_default_device().type, "cpu", "fixture requires initial default CPU")
        for name in ("fftfreq", "rfftfreq"):
            reference = getattr(np.fft, name)(8, d=.5).astype(np.float32)
            with torch.device(device):
                ambient = getattr(torch.fft, name)(8, d=.5, dtype=torch.float32)
                explicit = getattr(torch.fft, name)(8, d=.5, dtype=torch.float32, device="cpu")
            default = getattr(torch.fft, name)(8, d=.5, dtype=torch.float32)
            with torch.device("cpu"):
                target = getattr(torch.fft, name)(8, d=.5, dtype=torch.float32, device=device)
            for value, declared in ((ambient, device), (explicit, "cpu"), (default, "cpu"), (target, device)):
                self._same(self._read(value, declared), reference)

    def _run(self, device):
        for mode in (("default", "strict") if device == "cuda" else ("default",)):
            with self._scope(device, mode):
                for kind in ("function", "method", "kwargs", "nested"):
                    for requested in (False, True):
                        for stopped in (False, True):
                            with self.subTest(device=device, mode=mode, kind=kind,
                                              requested=requested, no_grad=stopped):
                                self._case(device, kind, requested, stopped)
                self._factories(device)

    def test_cpu(self): self._run("cpu")
    def test_cuda(self): self._run("cuda")


if __name__ == "__main__": unittest.main()

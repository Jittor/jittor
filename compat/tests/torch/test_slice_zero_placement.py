"""Real CPU/NPU basic-slice regression with ordinary factories left on CPU.

Run CPU and NPU methods in separate serial processes. NumPy defines the full
forward result and nonconstant input-gradient scatter; it never computes a
candidate result. Native location and fallback counters are checked before
host snapshots. The optional two-device test needs two genuinely free NPUs.
Both Torch AND native Jittor factories must remain CPU: enabling global
use_cuda/use_acl in the NPU test masks the original native-zero allocation bug.
The plain CPU/NPU/two-device cases are boundary regressions, not historical
red evidence. The explicit frontend-scope cases reproduce the FSDP caller's
canonical tensor_frontend context without monkeypatching any implementation.
"""

from contextlib import ExitStack
import os
import unittest

import numpy as np


class TestSliceZeroPlacement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jittor as jt
        import torch
        from jittor.compat.shim.runtime import activation_status
        from jittor.compat.torch.publication import namespace_owner

        assert namespace_owner(torch) is jt and activation_status(jt).active
        cls.jt, cls.torch = jt, torch

    def resident(self, value, device):
        native = getattr(value, "_var", value)
        native.sync()
        self.assertEqual(str(value.device), device)
        self.assertEqual(native.location(), "cpu" if device == "cpu" else "device")
        if device != "cpu":
            self.assertEqual(native.device_id, int(device.split(":")[1]))

    def run_device(self, device, frontend_scope=False):
        jt, torch = self.jt, self.torch
        from jittor._runtime.fallback import forbid_backend_fallbacks

        with ExitStack() as stack:
            # Reproduce the caller's CPU ambient allocation policy explicitly.
            # This does not disable tensors explicitly placed on NPU: their
            # input placement must select the backend and temporary storage.
            stack.enter_context(jt.runtime.scope(use_cuda=0))
            if device != "cpu":
                if not jt.compiler.has_acl:
                    self.skipTest("ACL backend is not installed")
                # Query the registered backend without changing native factory
                # policy. Only explicit tensor device arguments select NPU.
                self.assertGreater(jt.core.backend_device_count("acl"), int(device.split(":")[1]))
            if frontend_scope:
                from jittor.compat.torch.frontend import tensor_frontend
                stack.enter_context(tensor_frontend(torch.Tensor))
            stack.enter_context(forbid_backend_fallbacks())
            before = jt.core.backend_fallback_count()
            # A global default-device change would hide the original bug.
            self.resident(torch.zeros(1), "cpu")
            native_default = jt.zeros(1)
            native_default.sync()
            self.assertEqual(native_default.location(), "cpu", "native default factory must remain CPU")
            for shape in ((8,), (2, 3, 8)):
                source = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 4
                for offset, selection in enumerate((slice(0, 3), slice(2, 6), slice(5, 8), slice(-6, -2), slice(0, 3))):
                    with self.subTest(device=device, shape=shape, selection=selection):
                        x = torch.tensor(source, device=device, requires_grad=True)
                        y = x[..., selection]
                        expected = source[..., selection]
                        weights = (np.arange(expected.size, dtype=np.float32).reshape(expected.shape) + offset + 1) / 4
                        cotangent = torch.tensor(weights, device=device)
                        loss = (y * cotangent).sum()
                        loss.backward()
                        self.assertIsNotNone(x.grad)
                        for tensor in (x, y, loss, x.grad):
                            self.resident(tensor, device)
                        reference = np.zeros(shape, dtype=np.float32)
                        reference[..., selection] = weights
                        np.testing.assert_array_equal(y.detach().cpu().numpy(), expected)
                        np.testing.assert_array_equal(x.grad.detach().cpu().numpy(), reference)
                        # Check input after materializing the slice/backward.
                        np.testing.assert_array_equal(x.detach().cpu().numpy(), source)
                        self.resident(torch.zeros(1), "cpu")
                        native_default = jt.zeros(1)
                        native_default.sync()
                        self.assertEqual(native_default.location(), "cpu", "slice leaked native device placement")
            jt.sync_all(True)
            self.assertEqual(jt.core.backend_fallback_count() - before, 0)

    def test_cpu(self):
        self.run_device("cpu")

    def test_npu(self):
        self.run_device("npu:%d" % int(os.environ.get("LOCAL_RANK", "0")))

    def test_cpu_frontend_scope(self):
        self.run_device("cpu", frontend_scope=True)

    def test_npu_frontend_scope(self):
        self.run_device("npu:%d" % int(os.environ.get("LOCAL_RANK", "0")), frontend_scope=True)

    def test_npu_two_devices(self):
        self.run_two_devices()

    def test_npu_frontend_scope_two_devices(self):
        self.run_two_devices(frontend_scope=True)

    def run_two_devices(self, frontend_scope=False):
        jt = self.jt
        if not jt.compiler.has_acl:
            self.skipTest("ACL backend is not installed")
        count = jt.core.backend_device_count("acl")
        if count < 2:
            self.skipTest("two visible NPUs required for cache isolation")
        # Reusing the same zero shape/dtype after the device changes must not
        # retrieve another device's cache entry. Do not set the factory default.
        for device in ("npu:0", "npu:1", "npu:0"):
            self.run_device(device, frontend_scope=frontend_scope)


if __name__ == "__main__":
    unittest.main()

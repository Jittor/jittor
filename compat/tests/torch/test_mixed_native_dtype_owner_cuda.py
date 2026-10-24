"""Real CUDA mixed-carrier dtype-reader regression, not whole-model PPO.

Candidate cases construct an actual legacy native Var and actual independent
Torch Tensor/Parameter. Native Torch may run the same fixed arithmetic/VJP
reference cases using ordinary tensors; it cannot claim a Jittor bridge.
Host NumPy is used only to observe/compare complete CUDA results, never as a
framework compute fallback. Assertions do not print tensor values.
"""
from contextlib import nullcontext
import unittest

import numpy as np
import pytest
import torch


pytestmark = pytest.mark.cuda


class TestMixedNativeDtypeOwnerCUDA(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("real CUDA hardware required; a skip is not acceptance")
        torch.cuda.set_device(0)
        self.jt = None
        scope = nullcontext()
        if hasattr(torch, "_torch_compat_install_context"):
            import jittor as jt
            from jittor._runtime.fallback import forbid_backend_fallbacks
            self.jt = jt
            scope = forbid_backend_fallbacks()
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)
        self.x = np.asarray([[1., -2., 3.], [4., .5, -1.5]], dtype=np.float32)
        self.w = np.asarray([2., -.5, 4.], dtype=np.float32)
        self.dy = np.asarray([[.25, -2., 1.], [-1., .5, 3.]], dtype=np.float32)
        self.y = np.asarray([[2., 1., 12.], [8., -.25, -6.]], dtype=np.float32)
        self.dx = np.asarray([[.5, 1., 4.], [-2., -.25, 12.]], dtype=np.float32)
        self.dw = np.asarray([-3.75, 4.25, -1.5], dtype=np.float32)

    def resident(self, value):
        if self.jt is None:
            self.assertEqual(value.device, torch.device("cuda:0"))
            self.assertEqual(value.dtype, torch.float32)
            torch.cuda.synchronize()
            return
        from jittor._core.dtypes import var_dtype_name
        self.assertTrue(self.jt.core._is_tensor_carrier(value))
        value.sync()
        self.assertEqual(var_dtype_name(value), "float32")
        self.assertEqual(self.jt.core.dispatch_context([value]), ("cuda", 0))
        self.assertEqual((value.location(), value.device_id, value.placement_backend),
                         ("device", 0, 1))
        self.assertEqual(self.jt.core.backend_fallback_count(), 0)

    def compare_complete(self, value, expected):
        self.resident(value)
        with torch.no_grad():
            if self.jt is None:
                observed = value.detach().cpu().numpy().copy()
            else:
                # Copy with the genuine actual receiver descriptor. Reading
                # this host observation must not relocate the original CUDA
                # operand/result or put a CPU op in the training graph.
                from jittor._core.carrier_protocol import native_api
                descriptor = native_api(torch.Tensor, value)["to_device"]
                self.assertTrue(isinstance(value, descriptor.__objclass__),
                                "observation descriptor does not own its actual receiver")
                expected_owner = (self.jt.core.Var if isinstance(value, self.jt.core.Var)
                                  else torch.Tensor._frontend_tensor_base)
                self.assertIs(descriptor.__objclass__, expected_owner)
                observed_copy = descriptor(value, -1)
                observed = np.array(observed_copy.numpy(), copy=True)
        self.assertTrue(observed.dtype == expected.dtype, "complete dtype differs")
        self.assertTrue(observed.shape == expected.shape, "complete shape differs")
        self.assertTrue(bool(np.isfinite(observed).all()), "nonfinite CUDA result")
        self.assertTrue(bool(np.array_equal(observed, expected)), "complete CUDA values differ")
        self.resident(value)

    def native_operand(self, values, like):
        if self.jt is None:
            return torch.tensor(values, dtype=torch.float32, device="cuda:0", requires_grad=True)
        jt = self.jt
        from jittor._core.var import placement_scope_like
        from jittor.compat.torch.frontend import tensor_frontend
        # Match the actual CUDA mixed-carrier reproduction. Placement alone
        # does not control the jt.array output carrier in an active frontend.
        # Native tensor_frontend can be a no-op: explicit class construction
        # and the mandatory exact-type check establish the native operand.
        # A native frontend scope may be a no-op. Request the reference's
        # genuine explicit construction placement instead of requiring an
        # implicitly placed CUDA Var to report explicit backend metadata.
        with tensor_frontend(jt.Var, like=like), placement_scope_like(like, override=True):
            value = jt.Var(np.asarray(values, dtype=np.float32)).cuda(0)
            value = value.start_grad()
            value.sync()
        self.assertIs(type(value), jt.core.Var, "mixed case did not construct a real native Var")
        self.assertNotIsInstance(value, torch.Tensor)
        self.resident(value)
        return value

    def operands(self, parameter):
        # Cover the actual RMSNorm-style Parameter[channels] broadcast too.
        public_values, native_values = (self.w, self.x) if parameter else (self.x, self.w)
        public = torch.tensor(public_values, dtype=torch.float32,
                              device="cuda:0", requires_grad=True)
        if parameter:
            public = torch.nn.Parameter(public, requires_grad=True)
            self.assertIs(type(public), torch.nn.Parameter)
        else:
            self.assertIs(type(public), torch.Tensor)
        native = self.native_operand(native_values, like=public)
        if self.jt is not None:
            self.assertFalse(isinstance(public, self.jt.core.Var),
                             "public carrier must remain a real sibling, not a fake Var base")
        self.resident(public)
        self.resident(native)
        return public, native

    def exercise(self, parameter, native_first):
        for no_grad in (False, True):
            with self.subTest(no_grad=no_grad):
                public, native = self.operands(parameter)
                left, right = (native, public) if native_first else (public, native)
                context = torch.no_grad() if no_grad else nullcontext()
                with context:
                    output = left * right
                self.compare_complete(output, self.y)
                self.assertTrue(bool(public.requires_grad))
                self.assertTrue(bool(native.requires_grad))
                self.assertEqual(bool(output.requires_grad), not no_grad)
                if not no_grad:
                    dy = torch.tensor(self.dy, dtype=torch.float32, device="cuda:0")
                    gradients = torch.autograd.grad(output, (public, native), grad_outputs=dy,
                                                    allow_unused=False, create_graph=False)
                    self.assertEqual(len(gradients), 2)
                    self.assertTrue(all(gradient is not None for gradient in gradients))
                    expected = (self.dw, self.dx) if parameter else (self.dx, self.dw)
                    for gradient, wanted in zip(gradients, expected):
                        self.compare_complete(gradient, wanted)
                self.compare_complete(public, self.w if parameter else self.x)
                self.compare_complete(native, self.x if parameter else self.w)

    def test_public_tensor_times_native_var(self):
        self.exercise(parameter=False, native_first=False)

    def test_native_var_times_public_tensor(self):
        self.exercise(parameter=False, native_first=True)

    def test_parameter_times_native_var(self):
        self.exercise(parameter=True, native_first=False)

    def test_native_var_times_parameter(self):
        self.exercise(parameter=True, native_first=True)

    def test_dtype_reader_uses_each_actual_descriptor_owner(self):
        public, native = self.operands(parameter=True)
        if self.jt is None:
            self.assertEqual(public.dtype, torch.float32)
            self.assertEqual(native.dtype, torch.float32)
        else:
            from jittor.compat.torch.context import get_install_context
            from jittor.compat.torch.installers.tensor.method_api import _own_dtype_name
            api = get_install_context(self.jt).state["tensor_native_api"]
            self.assertEqual(_own_dtype_name(public, api), "float32")
            self.assertEqual(_own_dtype_name(native, api), "float32")
        self.resident(public)
        self.resident(native)

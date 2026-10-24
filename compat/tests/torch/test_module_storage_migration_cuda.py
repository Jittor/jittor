"""Public Module.to storage round-trip regression on real CUDA, not PPO.

CPU is only an offloaded storage destination. All model arithmetic, complete
value comparisons, VJPs and optimizer updates execute on explicit CUDA:0.
The tests exercise production compatibility code, with no private prototypes.
Test-local diagnostics must not increase; startup counts are preserved and
reported separately, not cleared or certified as clean by this test.
"""
from contextlib import ExitStack
import ctypes
import unittest

import pytest
import torch


pytestmark = pytest.mark.cuda


class TestModuleStorageMigrationCUDA(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("real CUDA hardware required; a skip is not acceptance")
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        self.jt = None
        self.diagnostic_counts = None
        if hasattr(torch, "_torch_compat_install_context"):
            import jittor as jt
            from jittor.compat.diagnostics import counts
            from jittor._runtime.fallback import forbid_backend_fallbacks

            self.jt = jt
            self.diagnostic_counts = counts
            contexts.enter_context(forbid_backend_fallbacks())
            contexts.enter_context(jt.runtime.scope(
                use_cuda=1, device_id=0, backend_fallback="error"))
            self.assertEqual(jt.core.backend_fallback_count(), 0)
        torch.cuda.set_device(0)
        self.assertEqual(torch.cuda.current_device(), 0)
        self.driver = ctypes.CDLL("libcuda.so.1")
        self.driver.cuInit.argtypes = [ctypes.c_uint]
        self.driver.cuInit.restype = ctypes.c_int
        self.driver.cuPointerGetAttribute.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.driver.cuPointerGetAttribute.restype = ctypes.c_int
        self.assertEqual(self.driver.cuInit(0), 0)
        self.initial_counts = self.diagnostic_counts() if self.diagnostic_counts else {}
        startup = sum(self.initial_counts.values())
        self.runtime_evidence = dict(
            scope="real_CUDA_test_body_only_NOT_process_startup_or_PPO_acceptance",
            startup_swallowed_total=startup,
            startup_all_swallowed_zero=startup == 0,
            startup_nonzero_counts_preserved=startup > 0,
            startup_scope="before_model_construction_after_device_and_shim_setup",
            startup_label_or_exception_records_not_exported=True,
            CPU_storage_only=True)
        self.addCleanup(self.assert_no_new_diagnostics)

    def assert_no_new_diagnostics(self):
        torch.cuda.synchronize(0)
        if self.jt is not None:
            self.assertEqual(self.jt.core.backend_fallback_count(), 0)
            current = self.diagnostic_counts()
            self.runtime_evidence["new_swallowed_count"] = (
                sum(current.values()) - sum(self.initial_counts.values()))
            self.runtime_evidence["test_scope_diagnostic_counts_unchanged"] = (
                current == self.initial_counts)
            # A boolean assertion does not print private labels or records.
            self.assertTrue(current == self.initial_counts,
                            "test body introduced swallowed diagnostics")
            self.runtime_evidence["fallback_count"] = 0

    def resident(self, value, kind):
        self.assertEqual(value.device.type, kind)
        self.assertEqual(value.dtype, torch.float32)
        if kind == "cpu":
            self.assertIsNone(value.device.index)
            if self.jt is not None:
                value.sync()
                self.assertEqual((value.location(), int(value.device_id),
                                  int(value.placement_backend)), ("cpu", -1, 0))
            return
        self.assertEqual(value.device.index, 0)
        torch.cuda.synchronize(0)
        if self.jt is not None:
            value.sync()
            self.assertTrue(self.jt.core._is_tensor_carrier(value))
            self.assertEqual(tuple(self.jt.core.dispatch_context([value])), ("cuda", 0))
            self.assertEqual((value.location(), int(value.device_id),
                              int(value.placement_backend)), ("device", 0, 1))
            pointer = int(value.device_raw_ptr)
            self.assertEqual(self.jt.core.backend_fallback_count(), 0)
        else:
            pointer = int(value.data_ptr())
        self.assertGreater(pointer, 0)
        ordinal, memory_type = ctypes.c_int(-1), ctypes.c_int(-1)
        self.assertEqual(self.driver.cuPointerGetAttribute(
            ctypes.byref(ordinal), 9, ctypes.c_void_p(pointer)), 0)
        self.assertEqual(self.driver.cuPointerGetAttribute(
            ctypes.byref(memory_type), 2, ctypes.c_void_p(pointer)), 0)
        self.assertEqual((ordinal.value, memory_type.value), (0, 2))

    def equal_fill(self, value, fill):
        self.resident(value, "cuda")
        with torch.no_grad():
            expected = torch.full_like(value, fill, device="cuda:0")
            self.resident(expected, "cuda")
            self.assertTrue(bool(torch.equal(value, expected)),
                            "complete CUDA element comparison differs")
        self.resident(value, "cuda")

    def policy(self, value):
        policy = (bool(value.requires_grad), bool(value.is_leaf),
                  value.grad_fn is None, int(value._version))
        if self.jt is not None:
            policy += (bool(value.is_stop_grad()),)
        return policy

    def model(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.full(
                    (4, 4), .5, device="cuda:0", dtype=torch.float32))
                self.tied = self.weight
                self.frozen = torch.nn.Parameter(torch.full(
                    (4, 4), .125, device="cuda:0", dtype=torch.float32))
                self.frozen.requires_grad_(False)
                self.permanent = torch.nn.Parameter(torch.full(
                    (4, 4), 0., device="cuda:0", dtype=torch.float32),
                    requires_grad=False)
                self.register_buffer("marker", torch.full(
                    (4, 4), .25, device="cuda:0", dtype=torch.float32))

            def forward(self, x):
                return x * self.weight + self.frozen + self.marker

        return Model()

    def complete_vjp(self, model):
        x = torch.full((4, 4), 2., device="cuda:0", dtype=torch.float32,
                       requires_grad=True)
        output = model(x)
        self.equal_fill(output, 1.375)
        cotangent = torch.ones_like(output, device="cuda:0")
        self.resident(cotangent, "cuda")
        dw, dx = torch.autograd.grad(
            output, (model.weight, x), grad_outputs=cotangent,
            allow_unused=False, create_graph=False)
        self.equal_fill(dw, 2.)
        self.equal_fill(dx, .5)
        for gradient in (dw, dx):
            self.assertFalse(gradient.requires_grad)
            self.assertTrue(gradient.is_leaf)
            self.assertIsNone(gradient.grad_fn)

    def exercise_round_trip(self, published_gradient):
        model = self.model()
        weight, frozen, permanent = model.weight, model.frozen, model.permanent
        optimizer = torch.optim.SGD([weight], lr=.125)
        optimizer_identity = id(optimizer)

        def identities():
            self.assertIs(model.weight, weight)
            self.assertIs(model.tied, weight)
            self.assertIs(model.frozen, frozen)
            self.assertIs(model.permanent, permanent)
            self.assertEqual(id(optimizer), optimizer_identity)
            self.assertIs(optimizer.param_groups[0]["params"][0], weight)

        def policies():
            return tuple(self.policy(value) for value in
                         (weight, frozen, permanent, model.marker))

        identities()
        self.assertTrue(weight.requires_grad)
        for value in (frozen, permanent, model.marker):
            self.assertFalse(value.requires_grad)
        initial_policy = policies()
        self.complete_vjp(model)
        self.assertIsNone(weight.grad)
        prior_gradient = None
        prior_gradient_policy = None
        if published_gradient:
            x = torch.full((4, 4), 2., device="cuda:0", dtype=torch.float32,
                           requires_grad=True)
            model(x).sum().backward()
            prior_gradient = weight.grad
            self.assertIsNotNone(prior_gradient)
            self.equal_fill(prior_gradient, 2.)
            prior_gradient_policy = self.policy(prior_gradient)
            del x
        for destination, kind in (("cpu", "cpu"), ("cuda:0", "cuda")):
            with self.subTest(destination=kind, published_gradient=published_gradient):
                self.assertIs(model.to(device=destination), model)
                torch.cuda.synchronize(0)
                identities()
                for value in (weight, frozen, permanent, model.marker):
                    self.resident(value, kind)
                self.assertTrue(policies() == initial_policy,
                                "storage move changed leaf/grad/frozen/version policy")
                if published_gradient:
                    self.assertIs(weight.grad, prior_gradient)
                    self.resident(weight.grad, kind)
                    self.assertTrue(self.policy(weight.grad) == prior_gradient_policy,
                                    "storage move changed published gradient policy")
                else:
                    self.assertIsNone(weight.grad)
                for value in (frozen, permanent, model.marker):
                    self.assertIsNone(value.grad)
        self.equal_fill(weight, .5)
        self.equal_fill(frozen, .125)
        self.equal_fill(permanent, 0.)
        self.equal_fill(model.marker, .25)
        if published_gradient:
            self.equal_fill(weight.grad, 2.)
        self.complete_vjp(model)
        # A reversible freeze remains reversible, unlike merely reporting False.
        frozen.requires_grad_(True)
        self.assertTrue(frozen.requires_grad)
        frozen.requires_grad_(False)
        self.assertTrue(policies() == initial_policy,
                        "reversible freeze or permanent frozen policy changed")
        optimizer.zero_grad(set_to_none=True)
        self.assertIsNone(weight.grad)
        x = torch.full((4, 4), 2., device="cuda:0", dtype=torch.float32,
                       requires_grad=True)
        output = model(x)
        self.equal_fill(output, 1.375)
        output.sum().backward()
        self.equal_fill(weight.grad, 2.)
        self.equal_fill(x.grad, .5)
        self.assertIsNone(frozen.grad)
        self.assertIsNone(permanent.grad)
        optimizer.step()
        torch.cuda.synchronize(0)
        identities()
        self.equal_fill(weight, .25)
        self.equal_fill(frozen, .125)
        self.equal_fill(permanent, 0.)
        self.equal_fill(model.marker, .25)
        self.equal_fill(model(torch.full_like(x, 2., device="cuda:0")), .875)

    def test_storage_round_trip_without_published_gradient(self):
        self.exercise_round_trip(published_gradient=False)

    def test_storage_round_trip_with_published_gradient(self):
        self.exercise_round_trip(published_gradient=True)

    def test_grad_absence_and_real_managed_gradient(self):
        unmanaged = torch.full((2, 2), .5, device="cuda:0", dtype=torch.float32,
                               requires_grad=True)
        managed = torch.nn.Parameter(torch.full(
            (2, 2), .5, device="cuda:0", dtype=torch.float32))
        self.assertIsNone(unmanaged.grad)
        self.assertIsNone(managed.grad)
        optimizer = torch.optim.SGD([managed], lr=.125)
        self.assertIsNone(unmanaged.grad)
        self.assertIsNone(managed.grad)
        gradient = torch.full_like(managed, 3., device="cuda:0")
        self.resident(gradient, "cuda")
        managed.grad = gradient
        self.assertIs(managed.grad, gradient)
        self.equal_fill(managed.grad, 3.)
        self.assertIsNone(unmanaged.grad)
        optimizer.step()
        self.equal_fill(managed, .125)
        optimizer.zero_grad(set_to_none=True)
        self.assertIsNone(managed.grad)
        self.assertIsNone(unmanaged.grad)


if __name__ == "__main__":
    unittest.main()

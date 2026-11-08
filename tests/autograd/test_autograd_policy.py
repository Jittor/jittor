# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Autograd semantics are selected by a policy, not a core Torch flag."""

from _helpers import capability as _test_capability

import unittest

import jittor as jt


class TestAutogradPolicy(unittest.TestCase):
    def setUp(self):
        self._original = jt.autograd.get_policy()

    def tearDown(self):
        jt.autograd.set_policy(self._original)

    @staticmethod
    def _stopped_output():
        source = jt.array([1.0]).stop_grad()
        return source + 1

    @staticmethod
    def _assigned_parameter():
        target = jt.array([2.0])
        target.start_grad()
        target.assign(jt.array([3.0]).stop_grad())
        return target

    def test_native_policy_keeps_native_semantics(self):
        with jt.autograd.policy_scope(jt.autograd.NATIVE):
            self.assertFalse(self._stopped_output().is_stop_grad())
            self.assertFalse(self._assigned_parameter().requires_grad)

    def test_explicit_requires_grad_policy_matches_legacy_mode(self):
        with jt.autograd.policy_scope(jt.autograd.EXPLICIT_REQUIRES_GRAD):
            self.assertTrue(self._stopped_output().is_stop_grad())
            self.assertTrue(self._assigned_parameter().requires_grad)

    @unittest.skipUnless(_test_capability.check_accelerator('cuda', backend=jt).enabled, "CUDA is not available")
    def test_explicit_requires_grad_policy_on_cuda(self):
        with jt.flag_scope(use_cuda=1):
            with jt.autograd.policy_scope(jt.autograd.EXPLICIT_REQUIRES_GRAD):
                output = self._stopped_output()
                target = self._assigned_parameter()
                self.assertTrue(output.is_stop_grad())
                self.assertTrue(target.requires_grad)
                self.assertEqual(output.numpy().tolist(), [2.0])
                self.assertEqual(target.numpy().tolist(), [3.0])

    def test_policy_scope_restores_after_exception(self):
        jt.autograd.set_policy(jt.autograd.NATIVE)
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with jt.autograd.policy_scope(jt.autograd.EXPLICIT_REQUIRES_GRAD):
                self.assertIs(
                    jt.autograd.get_policy(),
                    jt.autograd.EXPLICIT_REQUIRES_GRAD,
                )
                raise RuntimeError("sentinel")
        self.assertIs(jt.autograd.get_policy(), jt.autograd.NATIVE)

    def test_policy_presets_are_immutable(self):
        with self.assertRaisesRegex(AttributeError, "immutable"):
            jt.autograd.NATIVE.stop_outputs_when_inputs_stopped = True


    def test_torch_edge_dtype_policy_is_opt_in_and_native_default_is_unchanged(self):
        class RawVJP(jt.Function):
            def execute(self, value, coefficient):
                self.coefficient = coefficient
                return value.cast("float32") * coefficient

            def grad(self, upstream):
                return upstream * self.coefficient, None

        setter = jt.core._set_torch_grad_edge_dtypes
        original = setter(False)
        try:
            with jt.flag_scope(auto_mixed_precision_level=0, amp_reg=0):
                def compute(dtype, half_ulp):
                    value = jt.array([.25]).cast(dtype).start_grad()
                    positive = jt.array([1. + half_ulp]).cast("float32").stop_grad()
                    negative = jt.array([-1.]).cast("float32").stop_grad()
                    loss = (RawVJP()(value, positive) + RawVJP()(value, negative)).sum()
                    return jt.grad(loss, [value])[0]
                for dtype, half_ulp in (("bfloat16", 2. ** -8),
                                        ("float16", 2. ** -11)):
                    baseline = compute(dtype, half_ulp)
                    self.assertEqual(baseline.dtype, "float32")
                    self.assertEqual(float(baseline.numpy()[0]), half_ulp)
                    self.assertFalse(setter(True))
                    try:
                        normalized = compute(dtype, half_ulp)
                        self.assertEqual(normalized.dtype, dtype)
                        self.assertEqual(float(normalized.float32().numpy()[0]), 0.)
                    finally:
                        self.assertTrue(setter(False))
                    restored = compute(dtype, half_ulp)
                    self.assertEqual(restored.dtype, "float32")
                    self.assertEqual(float(restored.numpy()[0]), half_ulp)
        finally:
            setter(original)


if __name__ == "__main__":
    unittest.main()

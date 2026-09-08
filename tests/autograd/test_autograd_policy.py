# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Autograd semantics are selected by a policy, not a core Torch flag."""

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

    @unittest.skipUnless(jt.has_cuda, "CUDA is not available")
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


if __name__ == "__main__":
    unittest.main()

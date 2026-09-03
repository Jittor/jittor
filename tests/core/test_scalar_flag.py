# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``_is_scalar`` and explicit-requires-grad must be independent Var flags.

They shared one bit, so ``requires_grad_(True)`` made a parameter look like a
Python scalar to type inference: its own dtype stopped taking part in
promotion, and the mixed-precision override was skipped on every operator
that touched it.
"""
import unittest

import numpy as np

import jittor as jt


class TestScalarFlag(unittest.TestCase):
    def setUp(self):
        self._amp = jt.flags.amp_reg

    def tearDown(self):
        jt.flags.amp_reg = self._amp

    def _pair(self, dtype):
        data = np.ones(4, dtype)
        return jt.array(data), jt.array(data)

    def test_requires_grad_does_not_disable_amp(self):
        # amp_prefer32 promotes a half computation to float32. A parameter must
        # not opt itself out of that.
        with jt.flag_scope(amp_reg=1):
            plain, other = self._pair("float16")
            self.assertEqual(str((plain * other).dtype), "float32")
            param, other = self._pair("float16")
            param.start_grad()
            self.assertEqual(str((param * other).dtype), "float32")

    def test_requires_grad_does_not_disable_amp_prefer16(self):
        with jt.flag_scope(amp_reg=2):
            plain, other = self._pair("float32")
            self.assertEqual(str((plain * other).dtype), "float16")
            param, other = self._pair("float32")
            param.start_grad()
            self.assertEqual(str((param * other).dtype), "float16")

    def test_requires_grad_keeps_dtype_promotion(self):
        # A parameter's own width must still take part in promotion. Before the
        # fix the wider operand was ignored and the result followed the narrow
        # one, exactly as it does for a real Python scalar.
        with jt.flag_scope(amp_reg=0):
            wide = jt.array(np.ones(4, "float32"))
            narrow = jt.array(np.ones(4, "float16"))
            self.assertEqual(str((wide * narrow).dtype), "float32")
            wide = jt.array(np.ones(4, "float32"))
            wide.start_grad()
            self.assertEqual(str((wide * narrow).dtype), "float32")

    def test_python_scalar_still_follows_the_other_operand(self):
        # The behaviour the flag exists for must be unchanged.
        with jt.flag_scope(amp_reg=0):
            narrow = jt.array(np.ones(4, "float16"))
            self.assertEqual(str((narrow * 2.0).dtype), "float16")

    def test_gradients_still_flow_after_the_flag_move(self):
        x = jt.array(np.ones(4, "float32"))
        x.start_grad()
        grad = jt.grad((x * 3).sum(), x)
        np.testing.assert_allclose(grad.numpy(), np.full(4, 3.0))


if __name__ == "__main__":
    unittest.main()

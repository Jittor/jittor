# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Auto-mixed-precision must cast a convolution's operands, not just its output.

``amp_prefer16`` is the register the torch shim's ``torch.autocast(float16)``
sets.  It biases an operator's *output* dtype to float16, but used to leave the
operands at their original dtype -- a "float32 in, float16 out" request that no
cuDNN algorithm satisfies, so ``cudnn_conv3d`` died on ``best_algo_idx == -1``
(and ``cudnn_conv`` would have too).  Torch's autocast casts the operands to the
compute dtype *before* building the operator; these tests pin that the
convolution ops do the same, in both directions, and that a genuinely mixed
operand pair is promoted rather than rejected.
"""

from _helpers import capability as _test_capability

import unittest

import numpy as np

import jittor as jt


@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No CUDA found")
@_test_capability.library_required('cudnn', backend=jt)
class TestCudnnConvAmpDtype(unittest.TestCase):
    def setUp(self):
        from contextlib import ExitStack as _TestPolicyStack
        stack = _TestPolicyStack()
        self.addCleanup(stack.close)
        stack.enter_context(jt.runtime.scope(use_cuda=1))

    def tearDown(self):
        jt.sync_all()

    def test_conv3d_amp_prefer16_casts_float32_operands(self):
        """The reported failure: float32 conv3d under torch.autocast(float16)."""
        rng = np.random.RandomState(7)
        x_np = rng.randn(1, 3, 4, 4, 4).astype("float32")
        w_np = rng.randn(4, 3, 2, 2, 2).astype("float32")
        with jt.flag_scope(amp_reg=jt.amp_flags.prefer16):
            y = jt.nn.conv3d(jt.array(x_np), jt.array(w_np))
            y.sync()
        self.assertEqual(y.dtype, "float16")
        # Same numbers as casting both operands explicitly.
        want = jt.nn.conv3d(jt.array(x_np).float16(), jt.array(w_np).float16())
        np.testing.assert_allclose(y.float32().numpy(), want.float32().numpy(),
                                   atol=1e-3, rtol=1e-3)

    def test_conv2d_amp_prefer16_casts_float32_operands(self):
        rng = np.random.RandomState(8)
        x_np = rng.randn(1, 3, 8, 8).astype("float32")
        w_np = rng.randn(4, 3, 3, 3).astype("float32")
        with jt.flag_scope(amp_reg=jt.amp_flags.prefer16):
            y = jt.nn.conv2d(jt.array(x_np), jt.array(w_np))
            y.sync()
        self.assertEqual(y.dtype, "float16")
        want = jt.nn.conv2d(jt.array(x_np).float16(), jt.array(w_np).float16())
        np.testing.assert_allclose(y.float32().numpy(), want.float32().numpy(),
                                   atol=1e-3, rtol=1e-3)

    def test_conv3d_backward_under_amp_prefer16(self):
        """The backward ops receive the same mixed request; they must cast too.

        Without the cast the weight gradient op gets float32 ``dy`` against a
        float16 filter and dies the same way the forward did.  The gradients
        that come back are float32 again: they cross the operands' casts on the
        way out, which is what keeps a float32 master weight float32 under
        autocast.
        """
        rng = np.random.RandomState(9)
        x_np = rng.randn(1, 2, 4, 4, 4).astype("float32")
        w_np = rng.randn(3, 2, 2, 2, 2).astype("float32")
        with jt.flag_scope(amp_reg=jt.amp_flags.prefer16):
            x = jt.array(x_np)
            w = jt.array(w_np)
            y = jt.nn.conv3d(x, w)
            gx, gw = jt.grad(y.sum(), [x, w])
            jt.sync([y, gx, gw])
        self.assertEqual(y.dtype, "float16")
        self.assertEqual(gx.dtype, x.dtype)
        self.assertEqual(gw.dtype, w.dtype)

    def test_mixed_operands_promote_without_amp(self):
        """No amp: a float16 input against a float32 filter promotes to float32."""
        rng = np.random.RandomState(10)
        x_np = rng.randn(1, 3, 4, 4, 4).astype("float16")
        w_np = rng.randn(4, 3, 2, 2, 2).astype("float32")
        y = jt.nn.conv3d(jt.array(x_np), jt.array(w_np))
        y.sync()
        self.assertEqual(y.dtype, "float32")
        want = jt.nn.conv3d(jt.array(x_np).float32(), jt.array(w_np))
        np.testing.assert_allclose(y.numpy(), want.numpy(), atol=2e-2, rtol=2e-2)

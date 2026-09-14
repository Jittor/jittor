# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The CUDA cuBLAS matmul must honour the amp register.

The cuBLAS ops used to take their output dtype from the operands and ignore
``jt.flags.amp_reg`` entirely, while the generic path the CPU and mixed-dtype
cases use does read it.  A float32 matmul under ``amp_prefer16`` therefore
stayed float32 on CUDA and became float16 on the CPU -- the same model, two
precisions, depending on the backend.  The ops now take their output dtype from
the same inference every other op uses and cast the operands to it.
"""

from _helpers import capability as _test_capability

import unittest

import numpy as np

import jittor as jt


@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No CUDA found")
@_test_capability.library_required('cublas', backend=jt)
class TestCublasMatmulAmpDtype(unittest.TestCase):
    def setUp(self):
        from contextlib import ExitStack as _TestPolicyStack
        stack = _TestPolicyStack()
        self.addCleanup(stack.close)
        stack.enter_context(jt.runtime.scope(use_cuda=1))

    def tearDown(self):
        jt.sync_all()

    def test_float32_matmul_under_amp_prefer16_is_float16(self):
        a = jt.random((8, 16))
        b = jt.random((16, 4))
        with jt.flag_scope(amp_reg=jt.amp_flags.prefer16):
            c = jt.nn.matmul(a, b)
            c.sync()
        self.assertEqual(c.dtype, "float16")
        want = jt.nn.matmul(a.float16(), b.float16())
        np.testing.assert_allclose(c.float32().numpy(), want.float32().numpy(),
                                   atol=1e-2, rtol=1e-2)

    def test_matmul_without_amp_stays_float32(self):
        a = jt.random((8, 16))
        b = jt.random((16, 4))
        c = jt.nn.matmul(a, b)
        c.sync()
        self.assertEqual(c.dtype, "float32")
        np.testing.assert_allclose(c.numpy(), (a.numpy() @ b.numpy()),
                                   atol=1e-5, rtol=1e-5)

    def test_batched_matmul_under_amp_prefer16_is_float16(self):
        a = jt.random((3, 8, 16))
        b = jt.random((3, 16, 4))
        with jt.flag_scope(amp_reg=jt.amp_flags.prefer16):
            c = jt.nn.bmm(a, b)
            c.sync()
        self.assertEqual(c.dtype, "float16")

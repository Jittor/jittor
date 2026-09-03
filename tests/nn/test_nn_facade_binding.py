# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Where a private ``nn`` helper has to be patched to intercept it.

Task 5.22. ``jittor.nn`` re-exports ~35 underscore-private names, and the
modules that define them used to call *back through the facade* to reach them.
Those calls are module-local now, which moves the seam: patching
``jt.nn._ln_normalize`` no longer intercepts ``layer_norm``.

That matters because a test in the compat suite patches exactly this name to
prove the CUDA no-grad LayerNorm never falls back to the composite path. If the
seam moves and nobody notices, that test keeps passing while measuring nothing
-- the worst way for a change like this to land. So pin both halves: patching
the defining module DOES intercept, and patching the facade does NOT.

Run::  python -m pytest tests/nn/test_nn_facade_binding.py
"""

import unittest

import numpy as np

import jittor as jt
from jittor import nn
from jittor.nn.functional import normalization as _normalization


class _Sentinel(Exception):
    pass


class TestPrivateHelpersAreBoundLocally(unittest.TestCase):
    def setUp(self):
        self.original = _normalization._ln_normalize
        self.x = jt.array(np.random.RandomState(0).randn(2, 8, 16)
                          .astype("float32"))
        self.layer = nn.LayerNorm(16)

    def tearDown(self):
        _normalization._ln_normalize = self.original

    def test_patching_the_defining_module_intercepts(self):
        def intercept(*args, **kwargs):
            raise _Sentinel

        _normalization._ln_normalize = intercept
        # use_cuda=0 explicitly: on CUDA, layer_norm takes a fused kernel and
        # never reaches the composite helper, so this would assert nothing and
        # pass or fail depending on which device the suite happened to select.
        with jt.flag_scope(use_cuda=0):
            with self.assertRaises(_Sentinel):
                self.layer(self.x)

    def test_patching_the_facade_does_not_intercept(self):
        # Documented on purpose: someone re-adding a jt.nn-level patch here
        # would get a green test that proves nothing.
        saved = nn._ln_normalize

        def intercept(*args, **kwargs):
            raise _Sentinel

        nn._ln_normalize = intercept
        try:
            with jt.flag_scope(use_cuda=0):
                out = self.layer(self.x)
            self.assertEqual(tuple(out.shape), (2, 8, 16))
        finally:
            nn._ln_normalize = saved

    def test_the_facade_still_exports_the_name(self):
        # Removing the export is a separate decision; ACL and the compat layer
        # reach several of these through jt.nn, and tests/ops/test_fft_op.py
        # calls nn._fft2 directly.
        for name in ("_ln_normalize", "_get_softmax_dim", "_fft2",
                     "_CUDNN_3D_HALF_DTYPES"):
            self.assertTrue(hasattr(nn, name), name)


class TestBackendHooksStayLateBound(unittest.TestCase):
    """The cross-module hooks ACL replaces must still resolve through jt.nn."""

    def test_normalization_reaches_the_cuda_kernels_through_the_facade(self):
        source = _normalization.__file__
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        for hook in ("_batch_norm_cuda", "_group_norm_cuda",
                     "_layer_norm_cuda"):
            self.assertIn("jt.nn.%s" % hook, text,
                          "%s is replaced by ACL at runtime; binding it "
                          "locally would pin the default forever" % hook)


if __name__ == "__main__":
    unittest.main()

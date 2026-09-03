# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Where a private ``nn`` helper has to be patched to intercept it.

Task 5.22. Private helpers live in their implementation modules, not on the
``jittor.nn`` facade. Patches therefore target the defining module.

That matters because a test in the compat suite patches exactly this name to
prove the CUDA no-grad LayerNorm never falls back to the composite path. If the
seam moves and nobody notices, that test keeps passing while measuring nothing
-- the worst way for a change like this to land. So pin both halves: patching
the defining module intercepts, and the facade does not expose the helper.

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

    def test_the_facade_does_not_export_private_helpers(self):
        for name in ("_ln_normalize", "_get_softmax_dim", "_fft2",
                     "_CUDNN_3D_HALF_DTYPES"):
            self.assertFalse(hasattr(nn, name), name)


class TestBackendHooksStayLateBound(unittest.TestCase):
    """Optional backend hooks resolve through the internal hook module."""

    def test_normalization_reaches_the_internal_backend_hooks(self):
        source = _normalization.__file__
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("_backend_hooks.batch_norm_cuda", text)
        self.assertIn("_backend_hooks.group_norm_cuda", text)
        self.assertNotIn("jt.nn._", text)


if __name__ == "__main__":
    unittest.main()

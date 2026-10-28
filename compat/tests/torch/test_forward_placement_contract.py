# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The two forward-placement owners must select the same tensor.

`compat/torch/nn_frontend.py::_placement_reference` is the Python path;
`src/bindings/pyjt/py_module_call.cc::first_placed_module_var` is the native
fast path. Both answer "when the input carries no explicit placement, which
module Var should the forward follow?" -- first explicitly placed parameter,
in module ``__dict__`` order. They are deliberately two implementations (the
fast path may not call Python), so this pins that they agree.

The probe places its parameter on CPU, leaves the input unplaced, and runs
with CUDA requested in the ambient scope: only a parameter-following forward
stays on CPU. A second case places the input explicitly on CUDA and checks
that the input then wins.
"""

import unittest

import numpy as np
import torch

import jittor as jt
from jittor.compat.torch import nn_frontend


class _Probe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4).cpu())

    def forward(self, value):
        return value + self.weight


def _device(value):
    return str(value.device)


class TestForwardPlacementContract(unittest.TestCase):
    def setUp(self):
        if nn_frontend._NATIVE_CALL is None:
            self.skipTest("native module call is not installed")
        self.module = _Probe()
        self.owner = type(self.module)._nn_frontend_owner

    def _both_paths(self, value):
        native = nn_frontend.module_call(self.module, value)
        saved = nn_frontend._NATIVE_CALL
        nn_frontend._NATIVE_CALL = None
        try:
            slow = nn_frontend.python_module_call(self.module, value)
        finally:
            nn_frontend._NATIVE_CALL = saved
        return native, slow

    def _check(self, use_cuda, make_value, placed_input):
        with jt.flag_scope(use_cuda=use_cuda):
            value = make_value()
            reference = nn_frontend._placement_reference(
                self.owner, self.module, (value,), {})
            # The rule returns the holder stored in ``__dict__``; ``module.weight``
            # builds a fresh frontend wrapper each access.
            expected = (value if placed_input else vars(self.module)["weight"])
            self.assertIs(reference, expected)
            native, slow = self._both_paths(value)
            np.testing.assert_allclose(native.numpy(), slow.numpy())
            self.assertEqual(_device(native), _device(slow))
            self.assertEqual(_device(native), _device(reference))

    def test_unplaced_input_follows_the_module_parameter(self):
        # A native Var has no explicit placement; a torch factory at CPU does.
        self._check(0, lambda: jt.ones(4), placed_input=False)

    def test_placed_input_wins_over_the_module_parameter(self):
        self._check(0, lambda: torch.ones(4), placed_input=True)

    @unittest.skipUnless(jt.compiler.has_cuda, "CUDA is not available")
    def test_cuda_ambient_scope_still_follows_a_cpu_parameter(self):
        self._check(1, lambda: jt.ones(4), placed_input=False)


if __name__ == "__main__":
    unittest.main()

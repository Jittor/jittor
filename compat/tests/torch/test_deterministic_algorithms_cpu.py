# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Torch's deterministic-policy query is answered without an accelerator knob.

The CPU backend has no runtime deterministic option, but torch's API answers
on every device and libraries such as torchmetrics ask unconditionally. The
query therefore reports disabled, and disabling an already-absent policy is a
no-op; enabling it still fails loudly because nothing can honour it.
"""

import unittest

import jittor as jt
import torch


class TestDeterministicAlgorithmsCpu(unittest.TestCase):
    def test_query_reports_disabled_on_cpu(self):
        with jt.flag_scope(use_cuda=0):
            self.assertFalse(torch.are_deterministic_algorithms_enabled())
            torch.use_deterministic_algorithms(False)
            self.assertFalse(torch.are_deterministic_algorithms_enabled())


if __name__ == "__main__":
    unittest.main()

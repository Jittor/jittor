# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""A checkpoint entry of the wrong shape is an error, not a log line.

``load_parameters`` used to ``LOG.e`` the mismatch, skip the entry and return
normally, leaving that parameter at its random initialisation. A training run
then started from a half-loaded model, and the only trace was one line in a log
nobody was reading. Torch raises; so does this now, naming every mismatch.
"""

import unittest

import numpy as np

import jittor as jt


class TestLoadStateDictShapeMismatch(unittest.TestCase):

    def test_a_mismatch_raises_and_names_every_entry(self):
        net = jt.nn.Linear(3, 4)
        state = {"weight": np.zeros((5, 3), "float32"),
                 "bias": np.zeros((6,), "float32")}
        with self.assertRaises(RuntimeError) as caught:
            net.load_state_dict(state)
        text = str(caught.exception)
        self.assertIn("weight", text)
        self.assertIn("[5, 3]", text)
        self.assertIn("[4, 3]", text)
        self.assertIn("bias", text)

    def test_a_matching_checkpoint_still_loads(self):
        net = jt.nn.Linear(3, 4)
        state = {"weight": np.ones((4, 3), "float32"),
                 "bias": np.full((4,), 2, "float32")}
        net.load_state_dict(state)
        np.testing.assert_array_equal(net.weight.numpy(), state["weight"])
        np.testing.assert_array_equal(net.bias.numpy(), state["bias"])

    def test_an_unknown_key_is_still_only_a_warning(self):
        net = jt.nn.Linear(3, 4)
        net.load_state_dict({"not_a_parameter": np.zeros((1,), "float32")})


if __name__ == "__main__":
    unittest.main()

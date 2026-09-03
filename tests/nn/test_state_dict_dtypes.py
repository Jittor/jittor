# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``state_dict(to="torch")`` hands over the dtypes it was given.

It built every entry with ``torch.Tensor(v.numpy())``. ``torch.Tensor`` is
``torch.FloatTensor``, so that is a cast to float32 applied to the whole
checkpoint -- and a state dict is not all floats. ``num_batches_tracked`` is
int, attention masks are bool, quantisation zero-points are int. On the torch
side ``load_state_dict`` then either refuses the entry or keeps the float copy.

``state_dict(to="numpy")`` never had the problem and is the control: the two
conversions have to describe the same tensors.
"""

import unittest

import numpy as np

import jittor as jt


def _as_numpy(t):
    """The returned object is a torch tensor or the shim's stand-in."""
    return t.numpy() if hasattr(t, "numpy") else np.asarray(t)


class _Model(jt.Module):
    def __init__(self):
        self.weight = jt.array(np.zeros((2, 3), dtype="float32"))
        self.register_buffer("steps", jt.array(np.array([7], dtype="int32")))
        self.register_buffer(
            "mask", jt.array(np.array([True, False]), dtype="bool"))
        self.register_buffer(
            "wide", jt.array(np.array([1, 2], dtype="int64"), dtype="int64"))

    def execute(self, x):
        return x


class TestStateDictDtypes(unittest.TestCase):

    def setUp(self):
        self.model = _Model()
        self.expected = {
            "weight": "float32", "steps": "int32",
            "mask": "bool", "wide": "int64",
        }

    def test_numpy_conversion_keeps_every_dtype(self):
        got = self.model.state_dict(to="numpy")
        self.assertEqual({k: str(v.dtype) for k, v in got.items()},
                         self.expected)

    def test_torch_conversion_keeps_every_dtype(self):
        """The one that used to force float32 on all four entries."""
        got = self.model.state_dict(to="torch")
        # `torch` here is whatever the process has installed under that name --
        # the shim in a Jittor dev environment, real torch elsewhere. Either way
        # the dtype it reports has to be the dtype it was handed, so the
        # assertion does not depend on which one it is.
        self.assertEqual({k: str(v.dtype).replace("torch.", "")
                          for k, v in got.items()}, self.expected)

    def test_values_survive_the_conversion(self):
        got = self.model.state_dict(to="torch")
        np.testing.assert_array_equal(_as_numpy(got["steps"]), [7])
        np.testing.assert_array_equal(_as_numpy(got["mask"]), [True, False])
        np.testing.assert_array_equal(_as_numpy(got["wide"]), [1, 2])

    def test_a_value_that_float32_cannot_hold(self):
        """int64 is not just a label here.

        2**53 + 1 is the smallest odd integer float32 (or float64) cannot name,
        so a checkpoint that went through the float cast came back with a
        different number, not merely a different dtype.
        """
        big = 2 ** 53 + 1
        model = _Model()
        model.register_buffer(
            "big", jt.array(np.array([big], dtype="int64"), dtype="int64"))
        got = model.state_dict(to="torch")
        self.assertEqual(int(_as_numpy(got["big"])[0]), big)


if __name__ == "__main__":
    unittest.main()

# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Native optimizer state_dict carries and restores per-parameter buffers.

This case used to be an empty expected failure (KI-TEST-001).  The native
optimizers keep momentum/accumulator Vars in ``param_groups``; the state dict
must round-trip those Vars, not just the hyperparameters, or a resumed run
restarts every buffer at zero.
"""

import unittest

import numpy as np

import jittor as jt
from jittor import nn


def _model():
    model = nn.Linear(2, 2)
    initial = model.state_dict()
    return model, initial


def _state_buffers(optimizer):
    exclude = {"params", "grads"}
    found = {}
    for index, group in enumerate(optimizer.param_groups):
        for key, value in group.items():
            if key in exclude:
                continue
            if isinstance(value, jt.Var):
                found[(index, key)] = value
            elif isinstance(value, (list, tuple)):
                for item_index, item in enumerate(value):
                    if isinstance(item, jt.Var):
                        found[(index, key, item_index)] = item
    return found


class TestOptStateDict(unittest.TestCase):
    def test_opt_state_dict(self):
        x = jt.array(np.arange(8, dtype="float32").reshape(4, 2) / 8)
        configurations = (
            (jt.nn.SGD, {"lr": .1, "momentum": .9}),
            (jt.nn.RMSprop, {"lr": .1}),
            (jt.nn.Adam, {"lr": .1}),
            (jt.nn.AdamW, {"lr": .1, "weight_decay": .01}),
        )
        for factory, kwargs in configurations:
            with self.subTest(optimizer=factory.__name__):
                model, initial = _model()
                optimizer = factory(model.parameters(), **kwargs)
                for _ in range(3):
                    optimizer.step(model(x).sum())

                saved = optimizer.state_dict()
                before = {key: value.numpy().copy()
                          for key, value in _state_buffers(optimizer).items()}
                self.assertTrue(before, "optimizer produced no state buffers")

                resumed_model, _ = _model()
                resumed_model.load_state_dict(initial)
                resumed = factory(resumed_model.parameters(), **kwargs)
                resumed.load_state_dict(saved)
                after = _state_buffers(resumed)
                self.assertEqual(set(after), set(before))
                for key, value in before.items():
                    np.testing.assert_array_equal(after[key].numpy(), value)

                # A resumed step must match the uninterrupted one, which only
                # holds if the buffers (not merely the hyperparameters) were
                # restored.
                optimizer.step(model(x).sum())
                resumed.step(resumed_model(x).sum())
                np.testing.assert_allclose(
                    resumed_model.weight.numpy(), model.weight.numpy(),
                    rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(
                    resumed_model.bias.numpy(), model.bias.numpy(),
                    rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Registering a gradient hook must not change the tensor's residency.

KI-AUTOGRAD-003: an older ``register_hook`` ended with
``v.swap(hooker(v)[0])``, so the caller's Var carried the state of the hooker's
not-yet-materialised output node and ``location()`` read ``"none"`` for a tensor
that was already materialised on the host or the device.  The hook now attaches
to the receiver without touching its graph or residency; these tests pin every
observable the probe compares (location, device, dtype/shape, value, gradient
state) across registration, and that the hook still rewrites the gradient.

Run::  python -m pytest -q tests/autograd/test_register_hook_residency.py
"""

import unittest

import numpy as np

import jittor as jt
from _helpers import capability as _test_capability


def _fingerprint(value):
    # Reductions leave the receiver where it is; ``numpy()`` would move it and
    # destroy the very evidence this test collects.
    return (float(value.sum().item()), float((value * value).sum().item()))


def _observables(value):
    return {
        "location": value.location(),
        "device": value.device,
        "device_id": value.device_id,
        "dtype": str(value.dtype),
        "shape": tuple(value.shape),
        "requires_grad": bool(value.requires_grad),
        "stop_grad": bool(value.is_stop_grad()),
        "value": _fingerprint(value),
    }


class TestRegisterHookResidency(unittest.TestCase):
    def _assert_hook_is_observably_inert(self, use_cuda):
        with jt.flag_scope(use_cuda=use_cuda):
            value = jt.array(np.arange(8, dtype="float32").reshape(2, 4))
            value.start_grad()
            value.sync()
            before = _observables(value)
            self.assertNotEqual(before["location"], "none",
                                "the fixture must start materialised")
            seen = []

            def hook(gradient):
                seen.append(tuple(gradient.shape))
                return gradient * 2.0

            handle = value.register_hook(hook)
            after = _observables(value)
            self.assertEqual(after, before,
                             "register_hook changed an observable of its receiver")

            # Registering must not have consumed or replaced the graph: the hook
            # runs once and its replacement reaches the caller's gradient.
            gradient = jt.grad((value * 3.0).sum(), value)
            if isinstance(gradient, list):
                gradient = gradient[0]
            np.testing.assert_allclose(gradient.numpy(), np.full((2, 4), 6.0))
            self.assertEqual(seen, [(2, 4)])
            self.assertEqual(_observables(value), before)

            handle.remove()
            self.assertEqual(_observables(value), before)

    def test_cpu(self):
        self._assert_hook_is_observably_inert(0)

    @unittest.skipUnless(
        _test_capability.check_accelerator("cuda", backend=jt).enabled,
        "CUDA is not available",
    )
    def test_cuda(self):
        self._assert_hook_is_observably_inert(1)


if __name__ == "__main__":
    unittest.main()

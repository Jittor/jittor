# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The fused CUDA SGD must answer exactly what the portable update answers.

One kernel launch for the whole parameter list instead of two elementwise ops
and a holder rebind each. That is only worth having if every option
combination agrees with the update it replaces, and if everything it cannot
serve falls back instead of failing.
"""

from _helpers import capability as _test_capability
import itertools
import unittest

import numpy as np

import jittor as jt
from jittor import nn


def _model(seed=0):
    jt.set_global_seed(seed)
    return nn.Sequential(nn.Linear(6, 5), nn.Relu(), nn.Linear(5, 4))


@unittest.skipIf(not _test_capability.machine_has_accelerator("cuda"), "no CUDA device")
class TestFusedSgdCuda(unittest.TestCase):

    def setUp(self):
        self._use_cuda = jt.flags.use_cuda
        jt.flags.use_cuda = 1
        self.x = jt.array(np.linspace(-1, 1, 18, dtype="float32").reshape(3, 6))

    def tearDown(self):
        jt.flags.use_cuda = self._use_cuda

    def _trajectory(self, fused, **opts):
        model = _model()
        opt = nn.SGD(model.parameters(), lr=0.05, fused=fused, **opts)
        out = []
        for _ in range(4):
            opt.step((model(self.x) ** 2).mean())
            out.append([p.numpy().copy() for p in model.parameters()])
        return out

    def test_every_option_combination_matches_the_portable_update(self):
        for momentum, weight_decay, dampening, nesterov in itertools.product(
                (0.0, 0.9), (0.0, 0.01), (0.0, 0.1), (False, True)):
            if nesterov and (momentum == 0 or dampening != 0):
                continue          # torch forbids it and so does jittor
            opts = dict(momentum=momentum, weight_decay=weight_decay,
                        dampening=dampening, nesterov=nesterov)
            with self.subTest(**opts):
                want = self._trajectory(False, **opts)
                got = self._trajectory(True, **opts)
                for step, (a, b) in enumerate(zip(got, want)):
                    for u, v in zip(a, b):
                        np.testing.assert_allclose(
                            u, v, rtol=1e-5, atol=1e-6,
                            err_msg=f"step {step} with {opts}")

    def test_a_dtype_it_cannot_serve_falls_back(self):
        model = nn.Linear(6, 5)
        model.weight.update(model.weight.float16())
        model.bias.update(model.bias.float16())
        opt = nn.SGD(model.parameters(), lr=0.05, fused=True)
        before = model.weight.numpy().copy()
        opt.step((model(self.x.float16()) ** 2).mean())
        self.assertFalse(np.array_equal(model.weight.numpy(), before))

    def test_it_is_off_when_asked(self):
        want = self._trajectory(False)
        got = self._trajectory(False)
        for a, b in zip(got, want):
            for u, v in zip(a, b):
                np.testing.assert_array_equal(u, v)

    def test_a_parameter_list_longer_than_one_chunk(self):
        # The argument struct bounds a launch, so a long list is split; the
        # split must not change the answer.
        from jittor.backends.cuda.kernels.optim import fused_sgd_cuda
        chunk = fused_sgd_cuda._CHUNK
        model = nn.Sequential(*[nn.Linear(4, 4) for _ in range(chunk // 2 + 4)])
        self.assertGreater(len(list(model.parameters())), chunk)
        x = jt.array(np.ones((2, 4), dtype="float32"))
        opt_f = nn.SGD(model.parameters(), lr=0.01, fused=True)
        start = [p.numpy().copy() for p in model.parameters()]
        loss = (model(x) ** 2).mean()
        grads = [g.numpy().copy() for g in jt.grad(loss, list(model.parameters()))]
        opt_f.step(loss)
        for p, s, g in zip(model.parameters(), start, grads):
            np.testing.assert_allclose(p.numpy(), s - 0.01 * g, rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

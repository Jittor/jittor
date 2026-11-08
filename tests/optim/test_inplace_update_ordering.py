# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""A fused optimizer update writes the parameters in place; what read them
before the update reads the values from before it.

Execution is lazy: a forward output nobody has read yet is computed whenever
something asks for it. `loss = out * mask` makes the gradients independent of
`out` itself, so nothing orders the forward before the update -- and an
in-place update that ran first handed `out` the new weights. The bias reached
the forward through a broadcast view that had already run, so it was not even
a direct reader of the parameter (`order_after_readers` in src/core/op.cc).

Run in a child: whether the forward has already run when the update is built
depends on the process' execution history, and a fresh one is the case that
went wrong.
"""

from _helpers import capability as _test_capability
from _helpers.child_process import run_child_script
import unittest

import jittor as jt

_PROBE = """
import numpy as np
import jittor as jt
from jittor import nn
jt.flags.use_cuda = 1
makers = {
    "sgd": lambda params: nn.SGD(params, lr=100.0),
    "sgd_momentum": lambda params: nn.SGD(params, lr=100.0, momentum=0.9),
    "adamw": lambda params: jt.optim.AdamW(params, lr=1.0, fused=True),
}
for name, make in makers.items():
    jt.set_global_seed(0)
    layer = nn.Linear(8, 4)
    weight, bias = layer.weight.numpy().copy(), layer.bias.numpy().copy()
    x = jt.array(np.random.RandomState(0).randn(3, 8).astype("float32"))
    out = layer(x)
    mask = jt.random(out.shape)
    loss = out * mask
    make(layer.parameters()).step(loss)
    jt.sync_all(True)
    forward = np.abs(out.numpy() - (x.numpy() @ weight.T + bias)).max()
    moved = np.abs(layer.weight.numpy() - weight).max()
    print("ORDER", name, forward, moved)
"""


@unittest.skipIf(not _test_capability.check_accelerator("cuda", backend=jt).enabled,
                 "no usable CUDA in this build")
class TestInplaceUpdateOrdering(unittest.TestCase):
    def test_a_pending_forward_reads_the_weights_from_before_the_update(self):
        result = run_child_script(_PROBE, text=True, name="inplace_order")
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [line.split() for line in result.stdout.splitlines()
                if line.startswith("ORDER ")]
        self.assertEqual([row[1] for row in rows], ["sgd", "sgd_momentum", "adamw"])
        for _, name, forward, moved in rows:
            with self.subTest(optimizer=name):
                self.assertLess(float(forward), 1e-4)
                self.assertGreater(float(moved), 1e-3)   # the update did happen


if __name__ == "__main__":
    unittest.main()

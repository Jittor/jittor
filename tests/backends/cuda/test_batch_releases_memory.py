# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""A batch frees each intermediate after its last use, not at the batch's end.

`Executor::run_sync` holds every var of a batch for the batch's duration, so
that another thread cannot destroy a node the plan still points at (ceae1910).
Held to the end, though, nothing the batch computed could be freed until every
kernel of it had run -- and a backward is one batch -- so a backward's peak
was the sum of all its intermediates rather than the most alive at once. It
grew with depth: an 8-layer Qwen3 training step peaked at 13.07 GB against
PyTorch's 9.17 GB, and the 28-layer model did not fit a 24 GB card that
PyTorch trains it on in 19 GB. The hold now ends after each var's last use.

The measure is the backward's peak *above* the forward's, in activation-sized
units, on a residual stack whose forward keeps every layer's activations for
the backward. Freed as it goes, the backward needs a few activations more than
the forward whatever the depth; held, the excess grows with every layer.
"""

from _helpers import capability as _test_capability

import unittest

import jittor as jt


def _has_cuda():
    return bool(_test_capability.check_accelerator("cuda", backend=jt).enabled)


def _peaks(layers, rows=4096, width=1024):
    """(forward peak, backward peak, one activation) in bytes."""
    ws = [jt.random((width, width)) * 0.01 for _ in range(layers)]
    x = jt.random((rows, width))
    jt.sync_all(True)
    jt.core.reset_device_memory_peak(0)
    h = x
    for w in ws:
        h = h + jt.nn.relu(jt.matmul(h, w))
    loss = h.sqr().mean()
    loss.sync()
    jt.sync_all(True)
    forward = jt.core.device_memory_peak(0)
    jt.core.reset_device_memory_peak(0)
    grads = jt.grad(loss, ws, retain_graph=False)
    jt.sync(grads)
    jt.sync_all(True)
    backward = jt.core.device_memory_peak(0)
    return forward, backward, rows * width * 4


@unittest.skipIf(not _has_cuda(), "the pools count device memory only")
class TestBatchReleasesMemory(unittest.TestCase):
    def test_the_backward_excess_does_not_grow_with_depth(self):
        with jt.flag_scope(use_cuda=1):
            forward, backward, activation = _peaks(16)
        excess = (backward - forward) / activation
        # 4.0 with the release; 23.8 when every var was held to the end.
        self.assertLessEqual(
            excess, 6.0,
            "a 16-layer backward peaked %.1f activations above its forward"
            % excess)

    def test_the_peak_counter_sees_inside_a_batch(self):
        # The counter is what the check above stands on: it has to record a
        # peak reached, and released again, inside one batch.
        with jt.flag_scope(use_cuda=1):
            jt.sync_all(True)
            jt.core.reset_device_memory_peak(0)
            before = jt.core.device_memory_peak(0)
            big = jt.random((8192, 8192))          # 256 MiB, gone by the end
            small = (big * 2).sum()
            small.sync()
            del big
            jt.sync_all(True)
            self.assertGreaterEqual(jt.core.device_memory_peak(0) - before,
                                    8192 * 8192 * 4)


if __name__ == "__main__":
    unittest.main()

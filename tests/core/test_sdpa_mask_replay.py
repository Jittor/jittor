# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Default-SDPA replay: the causal-mask verdict must not block a capture.

``causal_mask_rewrite`` reads a boolean attention mask back to the host to
decide whether the mask can be replaced by ``is_causal=True``.  That read is
a value-dependent branch, so a capture refuses the trace and the default-SDPA
path rebuilds its Python graph on every call.  While tracing, the rewrite now
leaves the mask on the device.  These tests pin three things: the rewrite
still runs outside a capture, inference and whole training steps now replay
with default SDPA, and a replay answers correctly for *different mask values*
of the same shape (the captured graph treats the mask as an input).
"""

from _helpers.runtime_policy import preserve_policy as _test_preserve_policy
from _helpers import capability as _test_capability
import unittest

import numpy as np

import jittor as jt
from jittor import nn
from jittor._runtime import step_capture
from jittor._runtime.graph_replay import graph_replay
from jittor.nn.functional.attention import causal_mask_rewrite


def _has_cuda():
    return _test_capability.check_accelerator("cuda", backend=jt).enabled


class _Attention(nn.Module):
    def execute(self, query, key, value, mask):
        return nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, is_causal=False)


def _inputs(seed=0, batch=2, heads=2, length=16, dim=8):
    rs = np.random.RandomState(seed)
    shape = (batch, heads, length, dim)
    query = jt.array(rs.randn(*shape).astype("float32")).cuda()
    key = jt.array(rs.randn(*shape).astype("float32")).cuda()
    value = jt.array(rs.randn(*shape).astype("float32")).cuda()
    mask = jt.array(np.tril(np.ones((1, 1, length, length))).astype("bool")).cuda()
    jt.sync([query, key, value, mask])
    return query, key, value, mask


def _random_mask(seed, length=16):
    rs = np.random.RandomState(seed)
    mask = jt.array((rs.rand(length, length) > 0.3).astype("bool")).reshape((1, 1, length, length)).cuda()
    jt.sync([mask])
    return mask


def _make_step(weight, lr=0.01):
    def step(query, key, value, mask):
        out = nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, is_causal=False)
        loss = (out.reshape((-1, out.shape[-1])) @ weight).sum()
        grad, = jt.grad(loss, [weight])
        # Not in place: the replay writes the new Var back into the holder.
        weight.update(weight - lr * grad)
        return loss
    return step


@unittest.skipUnless(_has_cuda(), "the rewrite only runs on the cuda/cpu SDPA paths")
@_test_preserve_policy(jt, "keep_graph", "auto_graph_replay")
class TestSdpaMaskReplay(unittest.TestCase):

    def test_rewrite_still_applies_outside_capture(self):
        query, key, _, mask = _inputs()
        self.assertTrue(causal_mask_rewrite(query, key, mask, False))

    def test_rewrite_is_skipped_while_tracing(self):
        query, key, _, mask = _inputs()
        from jittor._runtime import graph_replay as _graph_replay
        _graph_replay._TRACING[0] += 1
        try:
            self.assertFalse(causal_mask_rewrite(query, key, mask, False))
        finally:
            _graph_replay._TRACING[0] -= 1
        self.assertTrue(causal_mask_rewrite(query, key, mask, False))

    def test_inference_replay_engages_with_default_sdpa(self):
        capture_inputs = _inputs(seed=0)
        model = _Attention()
        with jt.no_grad():
            eager_capture = model(*capture_inputs).numpy()
        replay = graph_replay(model, *capture_inputs)
        with jt.no_grad():
            replay(*capture_inputs)  # capture
            replayed = replay(*capture_inputs)  # replay
        self.assertIsNone(replay.refused)
        self.assertGreaterEqual(replay.stats["replayed"], 1)
        np.testing.assert_allclose(replayed.numpy(), eager_capture,
                                   atol=1e-4, rtol=1e-4)

    def test_replay_reads_new_mask_values_not_the_traced_verdict(self):
        capture_inputs = _inputs(seed=0)
        model = _Attention()
        replay = graph_replay(model, *capture_inputs)
        with jt.no_grad():
            replay(*capture_inputs)  # capture on a causal mask
            other = _inputs(seed=1)
            other_mask = _random_mask(seed=7)
            other_inputs = (other[0], other[1], other[2], other_mask)
            eager = model(*other_inputs).numpy()
            replayed = replay(*other_inputs).numpy()
        self.assertIsNone(replay.refused)
        np.testing.assert_allclose(replayed, eager, atol=1e-4, rtol=1e-4)

    def test_step_capture_replays_a_default_sdpa_training_step(self):
        query, key, value, mask = _inputs()
        rs = np.random.RandomState(3)
        captured_weight = jt.array((0.1 * rs.randn(8, 8)).astype("float32")).cuda()
        eager_weight = captured_weight.clone()
        jt.sync([captured_weight, eager_weight])

        captured = step_capture.StepCapture(_make_step(captured_weight))
        eager = _make_step(eager_weight)
        captured_losses, eager_losses = [], []
        for _ in range(4):
            captured_losses.append(float(captured(query, key, value, mask).numpy()))
            eager_losses.append(float(eager(query, key, value, mask).numpy()))
        self.assertIsNone(captured.refused)
        self.assertGreaterEqual(captured.stats["replayed"], 1)
        np.testing.assert_allclose(captured_weight.numpy(), eager_weight.numpy(),
                                   atol=1e-4, rtol=1e-3)
        np.testing.assert_allclose(captured_losses, eager_losses, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()

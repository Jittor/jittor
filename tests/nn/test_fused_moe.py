# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Mixture-of-experts routing and expert compute, against a written reference.

One token and many tokens take different code paths -- gather the chosen experts
versus visit each active expert once -- so both are checked, and the one-token
case is also checked against the many-token path applied to the same row.
"""
import unittest

import numpy as np

import jittor as jt


def _reference(x, router_logits, w13, w2, top_k, renormalize, scoring):
    """The definition, in float64, one token at a time."""
    logits = router_logits.astype("float64")
    if scoring == "sigmoid":
        scores = 1.0 / (1.0 + np.exp(-logits))
    else:
        shifted = logits - logits.max(-1, keepdims=True)
        scores = np.exp(shifted)
        scores /= scores.sum(-1, keepdims=True)
    order = np.argsort(-scores, axis=-1, kind="stable")[:, :top_k]
    weights = np.take_along_axis(scores, order, axis=-1)
    if renormalize:
        weights = weights / (weights.sum(-1, keepdims=True) + 1e-20)
    intermediate = w13.shape[1] // 2
    out = np.zeros_like(x, dtype="float64")
    for token in range(x.shape[0]):
        for slot in range(top_k):
            expert = int(order[token, slot])
            hidden = x[token].astype("float64") @ w13[expert].astype("float64").T
            gate, up = hidden[:intermediate], hidden[intermediate:]
            activated = (gate / (1.0 + np.exp(-gate))) * up
            out[token] += weights[token, slot] * (
                activated @ w2[expert].astype("float64").T)
    return out


class TestFusedMoE(unittest.TestCase):
    def _check(self, tokens, experts=6, top_k=2, hidden=16, intermediate=12,
               renormalize=True, scoring="softmax", seed=5):
        rng = np.random.RandomState(seed)
        x = rng.randn(tokens, hidden).astype("float32") * 0.2
        logits = rng.randn(tokens, experts).astype("float32")
        w13 = rng.randn(experts, 2 * intermediate, hidden).astype("float32") * 0.1
        w2 = rng.randn(experts, hidden, intermediate).astype("float32") * 0.1
        expected = _reference(x, logits, w13, w2, top_k, renormalize, scoring)
        with jt.no_grad():
            got = jt.nn.fused_moe(
                jt.array(x), jt.array(logits), jt.array(w13), jt.array(w2),
                top_k, renormalize=renormalize, scoring=scoring)
        got = np.asarray(got.float32().numpy())
        self.assertFalse(np.isnan(got).any(), "fused_moe produced NaN")
        np.testing.assert_allclose(got, expected, atol=2e-4, rtol=0)
        return got

    def test_single_token_gathers_its_experts(self):
        self._check(1)

    def test_many_tokens_dispatch_per_expert(self):
        self._check(9)

    def test_sigmoid_scoring(self):
        self._check(5, scoring="sigmoid")

    def test_without_renormalisation(self):
        self._check(5, renormalize=False)

    def test_top_k_may_exceed_two(self):
        self._check(7, experts=8, top_k=4)

    def test_one_token_agrees_with_the_batched_path(self):
        # The two paths compute the same thing; a row taken alone must match the
        # same row inside a batch.
        rng = np.random.RandomState(17)
        x = rng.randn(4, 16).astype("float32") * 0.2
        logits = rng.randn(4, 6).astype("float32")
        w13 = rng.randn(6, 24, 16).astype("float32") * 0.1
        w2 = rng.randn(6, 16, 12).astype("float32") * 0.1
        with jt.no_grad():
            batched = jt.nn.fused_moe(
                jt.array(x), jt.array(logits), jt.array(w13), jt.array(w2), 2)
            alone = jt.nn.fused_moe(
                jt.array(x[2:3]), jt.array(logits[2:3]), jt.array(w13),
                jt.array(w2), 2)
        np.testing.assert_allclose(
            np.asarray(alone.float32().numpy())[0],
            np.asarray(batched.float32().numpy())[2], atol=2e-5, rtol=0)


if __name__ == "__main__":
    unittest.main()

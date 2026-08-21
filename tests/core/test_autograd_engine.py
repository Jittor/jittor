# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Core autodiff *engine* correctness — not per-op backward, but the graph machinery.

``test_ops``/``gradcheck`` verify each operator's derivative. This module verifies the
properties of the autodiff ENGINE itself, on identical inputs across CPU and the
accelerator, against hand-derived analytic gradients:

  * gradient ACCUMULATION when an input feeds multiple paths (diamond graphs),
  * BROADCAST-BACK (a grad must sum over the axes an input was broadcast along —
    a classic silent-wrong spot for reduce/binary backward),
  * stop_grad / detach cutting one path of a diamond,
  * grad routing through pure view ops (reshape/transpose/getitem),
  * higher-order grad through a FUSED expression (the grad must itself be
    differentiable — the property create_graph relies on).

These are the failure modes that produce wrong *training* gradients while every
single-op test passes. References are analytic (closed-form), never jittor-vs-jittor.

Run::  python -m pytest tests/core/test_autograd_engine.py
"""
import unittest

import numpy as np
import jittor as jt

from _helpers.common import (
    JittorTestCase, get_all_device_types, use_cuda_for,
)


class TestAutogradEngine(JittorTestCase):

    def _devices(self, body):
        for d in get_all_device_types():
            with self.subTest(device=d):
                with jt.flag_scope(use_cuda=use_cuda_for(d)):
                    body(d)

    def _grad(self, loss, xs):
        g = jt.grad(loss, xs)
        return g if isinstance(g, (list, tuple)) else [g]

    # -- diamond: one input, two paths -> gradients accumulate --------------------
    def test_diamond_accumulates(self):
        x0 = np.random.RandomState(0).randn(5).astype("float32")

        def body(dev):
            x = jt.array(x0)
            z = x * x                          # shared subgraph
            loss = (z * 2 + z).sum()           # = 3 x^2  -> d/dx = 6x
            self.assertEqual(self._grad(loss, [x])[0], 6 * x0, atol=1e-4, rtol=1e-4,
                             msg=f"diamond accumulate [{dev}]")
        self._devices(body)

    def test_shared_input_two_ops(self):
        rng = np.random.RandomState(1)
        a0 = rng.randn(4).astype("float32"); b0 = rng.randn(4).astype("float32")
        c0 = rng.randn(4).astype("float32")

        def body(dev):
            a = jt.array(a0); b = jt.array(b0); c = jt.array(c0)
            loss = (a * b + (a + c)).sum()     # d/da = b + 1
            self.assertEqual(self._grad(loss, [a])[0], b0 + 1, atol=1e-5,
                             msg=f"shared-input grad [{dev}]")
        self._devices(body)

    # -- broadcast-back: grad must SUM over broadcast/reduced axes -----------------
    def test_reduce_then_broadcast_grad(self):
        x0 = np.random.RandomState(2).randn(3, 4).astype("float32")
        w0 = np.random.RandomState(3).randn(4).astype("float32")

        def body(dev):
            x = jt.array(x0)
            loss = (x.sum(0) * jt.array(w0)).sum()      # d/dx[i,j] = w[j] for all i
            ref = np.broadcast_to(w0, (3, 4))
            self.assertEqual(self._grad(loss, [x])[0], ref, atol=1e-5,
                             msg=f"reduce-broadcast grad [{dev}]")
        self._devices(body)

    def test_broadcast_binary_grad(self):
        a0 = np.random.RandomState(4).randn(3, 4).astype("float32")
        b0 = np.random.RandomState(5).randn(4).astype("float32")

        def body(dev):
            a = jt.array(a0); b = jt.array(b0)
            loss = (a + b).sum()                # d/db_j = sum_i 1 = 3
            self.assertEqual(self._grad(loss, [b])[0], np.full(4, 3.0, "float32"),
                             atol=1e-5, msg=f"broadcast-binary grad [{dev}]")
        self._devices(body)

    # -- stop_grad cuts one path of the diamond -----------------------------------
    def test_stop_grad_midgraph(self):
        x0 = np.random.RandomState(6).randn(5).astype("float32")

        def body(dev):
            x = jt.array(x0)
            y = x * 2
            loss = (y.stop_grad() * x).sum()    # only the bare x carries grad -> 2x
            self.assertEqual(self._grad(loss, [x])[0], 2 * x0, atol=1e-5,
                             msg=f"stop_grad midgraph [{dev}]")
        self._devices(body)

    # -- grad routes correctly through pure view ops ------------------------------
    def test_grad_through_views(self):
        x0 = np.random.RandomState(7).randn(2, 6).astype("float32")
        w0 = np.random.RandomState(8).randn(3, 4).astype("float32")

        def body(dev):
            x = jt.array(x0)
            y = x.reshape(4, 3).transpose(0, 1)     # (2,6) -> (4,3) -> (3,4)
            loss = (y * jt.array(w0)).sum()
            # d loss/dy = w0 (3,4); undo transpose(0,1) -> (4,3); undo reshape -> (2,6)
            ref = np.transpose(w0, (1, 0)).reshape(2, 6)
            self.assertEqual(self._grad(loss, [x])[0], ref, atol=1e-5,
                             msg=f"grad through views [{dev}]")
        self._devices(body)

    # -- higher-order grad through a FUSED expression -----------------------------
    def test_second_order_fused(self):
        x0 = np.random.RandomState(10).randn(4).astype("float32")

        def body(dev):
            x = jt.array(x0)
            loss = (jt.exp(x) + x * x * x).sum()   # f'' = exp(x) + 6x
            g1 = self._grad(loss, [x])[0]
            g2 = self._grad(g1.sum(), [x])[0]
            self.assertEqual(g2, np.exp(x0) + 6 * x0, atol=1e-3, rtol=1e-3,
                             msg=f"2nd-order fused [{dev}]")
        self._devices(body)

    # -- matmul chain: grad through A@B@C -----------------------------------------
    def test_matmul_chain_grad(self):
        rng = np.random.RandomState(11)
        A0 = rng.randn(2, 3).astype("float32")
        B0 = rng.randn(3, 4).astype("float32")
        C0 = rng.randn(4, 2).astype("float32")

        def body(dev):
            A = jt.array(A0); B = jt.array(B0); C = jt.array(C0)
            loss = (A @ B @ C).sum()
            gA = self._grad(loss, [A])[0]
            ref = np.ones((2, 2)) @ C0.T @ B0.T    # d/dA sum(ABC) = ones @ (BC)^T
            self.assertEqual(gA, ref, atol=1e-4, rtol=1e-4,
                             msg=f"matmul chain grad [{dev}]")
        self._devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)

# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Normalization-layer parity AND backward numerical stability.

This module closes the single most important hole the audit found: the legacy
``test_torch_compat_norm`` is FORWARD-ONLY, yet every normalization bug this project
fixed was a *backward* one -- the float32 catastrophic-cancellation in the
small-variance gradient (commits ``d4c7927a`` / ``98dfaf04`` / ``48024e98``) and the
BatchNorm ``running_var`` Bessel correction (``4a5063ff``).

Why a dedicated module and not just ``test_ops.py`` gradcheck: gradcheck runs in
float64, where the cancellation does not occur, so it cannot see this bug. The
regression is specifically that jittor's *float32* analytical gradient must match a
*float64* numerical reference at SMALL VARIANCE. If the stable jt.Function backward
is reverted to the naive composite, the float32 gradient drifts 1-10% and these
tests fail loudly. (The 1st-order *formula* is still covered generically in
``test_ops.py``; this adds the precision dimension torch verifies in test_nn.)

Run::  python -m pytest tests/nn/test_norm.py
"""
import unittest

import numpy as np
import jittor as jt
from jittor import nn

from _helpers.common import (
    JittorTestCase, net_scaled_max_err, get_all_device_types, use_cuda_for,
)
from _helpers.gradcheck import numerical_vjp

F = nn.functional


class _NormBase(JittorTestCase):
    # the small-variance gradient must match the float64 reference this tightly;
    # the naive composite backward drifts well past this at scale 1e-3.
    STABILITY_TOL = 1e-3

    def _check_backward_stable(self, fwd, x_np, label):
        """jittor float32 d/dx vs a float64 finite-difference reference."""
        x32 = jt.array(x_np.astype("float32"), dtype="float32")
        out = fwd(x32)
        rng = np.random.RandomState(7)
        cot = rng.randn(*tuple(out.shape)).astype("float32")
        g32 = jt.grad((out * jt.array(cot)).sum(), [x32])[0].numpy()
        ref = numerical_vjp(fwd, [jt.array(x_np.astype("float64"), dtype="float64")],
                            [cot], eps=1e-6)[0]
        err = net_scaled_max_err(g32, ref)
        self.assertLess(
            err, self.STABILITY_TOL,
            f"{label}: float32 backward drifts from float64 reference "
            f"(net-scaled err {err:.2e} >= {self.STABILITY_TOL:.0e}) -- the stable "
            f"norm backward may have regressed to the cancelling composite form")

    def _for_devices(self, body):
        for d in get_all_device_types():
            with self.subTest(device=d):
                with jt.flag_scope(use_cuda=use_cuda_for(d)):
                    body(d)


class TestLayerNorm(_NormBase):
    def _ref(self, x, w, b, eps=1e-5):
        mean = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * w + b

    def test_forward(self):
        C = 6
        x = np.random.RandomState(0).randn(4, C).astype("float32")
        w = np.random.RandomState(1).randn(C).astype("float32")
        b = np.random.RandomState(2).randn(C).astype("float32")

        def body(dev):
            out = F.layer_norm(jt.array(x), (C,), jt.array(w), jt.array(b), 1e-5)
            self.assertEqual(out, self._ref(x, w, b), atol=1e-4, rtol=1e-4,
                             msg=f"layer_norm fwd [{dev}]")
        self._for_devices(body)

    def test_backward_small_variance(self):
        # the marquee regression: tiny variance -> the cancelling composite backward
        # is 1-10% wrong in float32; the stable jt.Function backward is not.
        C = 8
        x = (np.random.RandomState(3).randn(5, C) * 1e-3).astype("float32")
        w = np.ones(C, "float32")
        b = np.zeros(C, "float32")
        self._check_backward_stable(
            lambda v: F.layer_norm(v, (C,), jt.array(w.astype(str(v.dtype))),
                                   jt.array(b.astype(str(v.dtype))), 1e-5),
            x, "LayerNorm grad @ var~1e-6")

    @unittest.skipUnless(jt.has_cuda, "CUDA LayerNorm fast path needs CUDA")
    def test_cuda_fast_path_forward_and_all_gradients(self):
        rng = np.random.RandomState(20260826)
        shape = (3, 5, 1024)
        weight_np = rng.randn(shape[-1]).astype("float32")
        bias_np = rng.randn(shape[-1]).astype("float32")
        cot_np = rng.randn(*shape).astype("float32")

        for low_variance in (False, True):
            x_np = rng.randn(*shape).astype("float32")
            if low_variance:
                x_np = 1.0 + x_np * 1e-3

            def run(use_cuda):
                with jt.flag_scope(use_cuda=use_cuda):
                    x = jt.array(x_np)
                    weight = jt.array(weight_np)
                    bias = jt.array(bias_np)
                    if use_cuda:
                        self.assertIsNotNone(jt.nn._layer_norm_cuda(
                            x, (shape[-1],), weight, bias, 1e-5
                        ))
                    output = F.layer_norm(
                        x, (shape[-1],), weight, bias, 1e-5
                    )
                    grads = jt.grad(
                        (output * jt.array(cot_np)).sum(),
                        [x, weight, bias],
                    )
                    return jt.fetch_sync([output] + grads)

            expected = run(0)
            actual = run(1)
            for name, got, ref in zip(
                    ("output", "grad_x", "grad_weight", "grad_bias"),
                    actual, expected):
                atol = 2e-3 if low_variance else 4e-4
                rtol = 5e-4 if low_variance else 4e-4
                np.testing.assert_allclose(
                    got, ref, atol=atol, rtol=rtol,
                    err_msg="CUDA LayerNorm %s low_variance=%s"
                    % (name, low_variance),
                )


class TestRMSNorm(_NormBase):
    @unittest.skipUnless(jt.has_cuda, "CUDA RMSNorm fast path needs CUDA")
    def test_cuda_training_forward_and_all_gradients(self):
        rng = np.random.RandomState(20260827)
        shape = (3, 5, 1024)
        x_np = rng.randn(*shape).astype("float32")
        gamma_np = rng.randn(shape[-1]).astype("float32")
        cot_np = rng.randn(*shape).astype("float32")
        epsilon = 1e-6

        def run(use_cuda):
            with jt.flag_scope(use_cuda=use_cuda):
                x = jt.array(x_np)
                gamma = jt.array(gamma_np)
                if use_cuda:
                    output = jt.nn._rms_norm_training_cuda(
                        x, gamma, epsilon
                    )
                    self.assertIsNotNone(output)
                else:
                    variance = (x * x).mean(-1, keepdims=True)
                    output = x * jt.rsqrt(variance + epsilon) * gamma
                grads = jt.grad(
                    (output * jt.array(cot_np)).sum(), [x, gamma]
                )
                return jt.fetch_sync([output] + grads)

        expected = run(0)
        actual = run(1)
        for name, got, ref in zip(
                ("output", "grad_x", "grad_gamma"), actual, expected):
            np.testing.assert_allclose(
                got, ref, atol=4e-4, rtol=4e-4,
                err_msg="CUDA RMSNorm %s" % name,
            )


class TestGroupNorm(_NormBase):
    @unittest.skipUnless(jt.has_cuda, "CUDA GroupNorm fast path needs CUDA")
    def test_cuda_fast_path_forward_and_all_gradients(self):
        rng = np.random.RandomState(20260823)
        shape = (2, 32, 16, 16)
        groups = 8
        x_np = rng.randn(*shape).astype("float32")
        weight_np = rng.randn(shape[1]).astype("float32")
        bias_np = rng.randn(shape[1]).astype("float32")
        cot_np = rng.randn(*shape).astype("float32")

        def run(use_cuda):
            with jt.flag_scope(use_cuda=use_cuda):
                x = jt.array(x_np)
                weight = jt.array(weight_np)
                bias = jt.array(bias_np)
                if use_cuda:
                    self.assertIsNotNone(
                        jt.nn._group_norm_cuda(x, groups, weight, bias, 1e-5)
                    )
                output = F.group_norm(x, groups, weight, bias, 1e-5)
                grads = jt.grad((output * jt.array(cot_np)).sum(), [x, weight, bias])
                return jt.fetch_sync([output] + grads)

        reference = run(0)
        actual = run(1)
        for label, got, expected in zip(
            ("output", "input grad", "weight grad", "bias grad"),
            actual,
            reference,
        ):
            np.testing.assert_allclose(
                got,
                expected,
                rtol=2e-3,
                atol=2e-3,
                err_msg="CUDA GroupNorm {}".format(label),
            )

    def test_backward_small_variance(self):
        N, C, H, W, G = 2, 8, 4, 4, 4
        x = (np.random.RandomState(4).randn(N, C, H, W) * 1e-3).astype("float32")
        w = np.ones(C, "float32")
        b = np.zeros(C, "float32")
        self._check_backward_stable(
            lambda v: F.group_norm(v, G, jt.array(w.astype(str(v.dtype))),
                                   jt.array(b.astype(str(v.dtype))), 1e-5),
            x, "GroupNorm grad @ var~1e-6")


class TestInstanceNorm(_NormBase):
    def test_backward_small_variance(self):
        N, C, L = 2, 6, 8
        x = (np.random.RandomState(5).randn(N, C, L) * 1e-3).astype("float32")
        w = np.ones(C, "float32")
        b = np.zeros(C, "float32")
        self._check_backward_stable(
            lambda v: F.instance_norm(v, None, None, jt.array(w.astype(str(v.dtype))),
                                      jt.array(b.astype(str(v.dtype))), 0.1, 1e-5),
            x, "InstanceNorm grad @ var~1e-6")


class TestBatchNorm(_NormBase):
    @unittest.skipUnless(jt.has_cuda, "CUDA BatchNorm eval fast path needs CUDA")
    def test_cuda_eval_fast_path_forward_and_all_gradients(self):
        rng = np.random.RandomState(20260829)
        shape = (2, 16, 8, 8)
        x_np = rng.randn(*shape).astype("float32")
        weight_np = rng.randn(shape[1]).astype("float32")
        bias_np = rng.randn(shape[1]).astype("float32")
        mean_np = rng.randn(shape[1]).astype("float32")
        variance_np = (np.abs(rng.randn(shape[1])) + 0.5).astype("float32")
        cot_np = rng.randn(*shape).astype("float32")

        def run(use_cuda):
            with jt.flag_scope(use_cuda=use_cuda):
                x = jt.array(x_np)
                weight = jt.array(weight_np)
                bias = jt.array(bias_np)
                mean = jt.array(mean_np).stop_grad()
                variance = jt.array(variance_np).stop_grad()
                if use_cuda:
                    output = jt.nn._batch_norm_eval_cuda(
                        x, weight, bias, mean, variance, 1e-5
                    )
                    self.assertIsNotNone(output)
                else:
                    scale = weight / jt.sqrt(variance + 1e-5)
                    shift = bias - mean * scale
                    output = (
                        x * scale.reshape((1, -1, 1, 1))
                        + shift.reshape((1, -1, 1, 1))
                    )
                grads = jt.grad(
                    (output * jt.array(cot_np)).sum(), [x, weight, bias]
                )
                return jt.fetch_sync([output] + grads)

        expected = run(0)
        actual = run(1)
        for name, got, ref in zip(
                ("output", "grad_x", "grad_weight", "grad_bias"),
                actual, expected):
            np.testing.assert_allclose(
                got, ref, atol=2e-3, rtol=2e-3,
                err_msg="CUDA eval BatchNorm %s" % name,
            )

    @unittest.skipUnless(jt.has_cuda, "CUDA BatchNorm fast path needs CUDA")
    def test_cuda_fast_path_forward_and_all_gradients(self):
        rng = np.random.RandomState(20260828)
        shape = (2, 32, 8, 8)
        x_np = rng.randn(*shape).astype("float32")
        weight_np = rng.randn(shape[1]).astype("float32")
        bias_np = rng.randn(shape[1]).astype("float32")
        cot_np = rng.randn(*shape).astype("float32")

        def run(use_cuda):
            with jt.flag_scope(use_cuda=use_cuda):
                x = jt.array(x_np)
                weight = jt.array(weight_np)
                bias = jt.array(bias_np)
                if use_cuda:
                    output = jt.nn._batch_norm_cuda(
                        x, weight, bias, 1e-5
                    )
                    self.assertIsNotNone(output)
                else:
                    mean = x.mean((0, 2, 3), keepdims=True)
                    variance = ((x - mean) * (x - mean)).mean(
                        (0, 2, 3), keepdims=True
                    )
                    output = (
                        (x - mean) * jt.rsqrt(variance + 1e-5)
                        * weight.reshape((1, -1, 1, 1))
                        + bias.reshape((1, -1, 1, 1))
                    )
                grads = jt.grad(
                    (output * jt.array(cot_np)).sum(), [x, weight, bias]
                )
                return jt.fetch_sync([output] + grads)

        expected = run(0)
        actual = run(1)
        for name, got, ref in zip(
                ("output", "grad_x", "grad_weight", "grad_bias"),
                actual, expected):
            np.testing.assert_allclose(
                got, ref, atol=2e-3, rtol=2e-3,
                err_msg="CUDA BatchNorm %s" % name,
            )

    def test_backward_small_variance_train(self):
        # BatchNorm train-mode backward was the worst (~10% at small variance).
        N, C = 16, 6
        x = (np.random.RandomState(6).randn(N, C) * 1e-3).astype("float32")
        w = np.ones(C, "float32")
        b = np.zeros(C, "float32")

        # jittor's F.batch_norm is eval-only (asserts not training); train-mode
        # BatchNorm -- and its stable jt.Function backward -- lives in the module.
        def fwd(v):
            dt = str(v.dtype)
            bn = nn.BatchNorm(C, momentum=0.1, eps=1e-5)
            bn.train()
            bn.weight = jt.array(w.astype(dt), dtype=dt)
            bn.bias = jt.array(b.astype(dt), dtype=dt)
            bn.running_mean = jt.array(np.zeros(C, dt), dtype=dt)
            bn.running_var = jt.array(np.ones(C, dt), dtype=dt)
            return bn(v)
        self._check_backward_stable(fwd, x, "BatchNorm(train) grad @ var~1e-6")

    def test_running_var_bessel(self):
        # torch updates running_var with the UNBIASED (Bessel, n/(n-1)) batch
        # variance though it normalizes with the biased one (commit 4a5063ff).
        N, C = 8, 4
        x = np.random.RandomState(11).randn(N, C).astype("float32")

        def body(dev):
            bn = nn.BatchNorm(C, momentum=0.1, eps=1e-5)
            bn.train()
            bn(jt.array(x))
            biased = x.var(0)                      # ddof=0
            unbiased = x.var(0, ddof=1)            # ddof=1 (Bessel)
            got = bn.running_var.numpy()
            tracked = int(bn.num_batches_tracked.item())
            # running_var = (1-m)*1 + m*var_used ; torch uses the UNBIASED var
            expect_unbiased = 0.9 * 1.0 + 0.1 * unbiased
            expect_biased = 0.9 * 1.0 + 0.1 * biased
            err_u = net_scaled_max_err(got, expect_unbiased)
            err_b = net_scaled_max_err(got, expect_biased)
            self.assertLess(err_u, 1e-4,
                            f"[{dev}] running_var should use the Bessel-corrected "
                            f"(unbiased) batch variance; got err vs unbiased {err_u:.2e}, "
                            f"vs biased {err_b:.2e}")
            self.assertEqual(tracked, 1, f"[{dev}] num_batches_tracked")
        self._for_devices(body)


class TestChannelBias(_NormBase):
    @unittest.skipUnless(jt.has_cuda, "CUDA channel bias fast path needs CUDA")
    def test_cuda_forward_and_bias_gradient(self):
        rng = np.random.RandomState(20260830)
        shape = (2, 24, 8, 8)
        x_np = rng.randn(*shape).astype("float32")
        bias_np = rng.randn(shape[1]).astype("float32")
        cot_np = rng.randn(*shape).astype("float32")

        def run(use_cuda):
            with jt.flag_scope(use_cuda=use_cuda):
                x = jt.array(x_np)
                bias = jt.array(bias_np)
                if use_cuda:
                    output = jt.nn._channel_bias_add_cuda(x, bias)
                    self.assertIsNotNone(output)
                else:
                    output = x + bias.reshape((1, -1, 1, 1))
                grads = jt.grad(
                    (output * jt.array(cot_np)).sum(), [x, bias]
                )
                return jt.fetch_sync([output] + grads)

        expected = run(0)
        actual = run(1)
        for name, got, ref in zip(
                ("output", "grad_x", "grad_bias"), actual, expected):
            np.testing.assert_allclose(
                got, ref, atol=2e-3, rtol=2e-3,
                err_msg="CUDA channel bias %s" % name,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

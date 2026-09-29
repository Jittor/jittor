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

from _helpers import capability as _test_capability
import unittest

import numpy as np
import jittor as jt
from jittor import nn
from jittor.backends.cuda.kernels.nn.batch_norm_training_cuda import (
    _batch_norm_cuda,
    _batch_norm_eval_cuda,
)
from jittor.backends.cuda.kernels.nn.channel_bias_cuda import _channel_bias_add_cuda
from jittor.backends.cuda.kernels.nn.group_norm_cuda import (
    _group_norm_cuda,
    _group_norm_cuda_cls,
)
from jittor.backends.cuda.kernels.nn.layer_norm_training_cuda import _layer_norm_cuda
from jittor.backends.cuda.kernels.nn.rms_norm_training_cuda import _rms_norm_training_cuda

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

    @unittest.skipUnless(_test_capability.check_accelerator('cuda', backend=jt).enabled, "CUDA LayerNorm fast path needs CUDA")
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
                        self.assertIsNotNone(_layer_norm_cuda(
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
    @unittest.skipUnless(_test_capability.check_accelerator('cuda', backend=jt).enabled, "CUDA RMSNorm fast path needs CUDA")
    def test_cuda_training_forward_and_all_gradients(self):
        self._check_cuda_training((3, 5, 1024))

    @unittest.skipUnless(_test_capability.check_accelerator('cuda', backend=jt).enabled, "CUDA RMSNorm fast path needs CUDA")
    def test_cuda_gamma_gradient_over_many_row_segments(self):
        # 4096 rows cut into segments, a width that is not a multiple of the
        # 32-channel blocks the gamma gradient is summed in.
        self._check_cuda_training((8, 512, 100))

    def _check_cuda_training(self, shape):
        rng = np.random.RandomState(20260827)
        x_np = rng.randn(*shape).astype("float32")
        gamma_np = rng.randn(shape[-1]).astype("float32")
        cot_np = rng.randn(*shape).astype("float32")
        epsilon = 1e-6

        def run(use_cuda):
            with jt.flag_scope(use_cuda=use_cuda):
                x = jt.array(x_np)
                gamma = jt.array(gamma_np)
                if use_cuda:
                    output = _rms_norm_training_cuda(
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
    @unittest.skipUnless(_test_capability.check_accelerator('cuda', backend=jt).enabled, "CUDA GroupNorm fast path needs CUDA")
    def test_cuda_half_precision_and_many_segments(self):
        # Half precision is how diffusion UNets run their GroupNorms; it used
        # to fall through to the generic path. The shapes cut each group into
        # several segments and take the float4 and the scalar kernels.
        cases = (((2, 64, 32, 32), 32, "float32"), ((3, 40, 7, 9), 8, "float32"),
                 ((2, 64, 32, 32), 32, "float16"), ((2, 96, 16, 16), 32, "bfloat16"))
        for shape, groups, dtype in cases:
            with self.subTest(shape=shape, dtype=dtype):
                rng = np.random.RandomState(sum(shape))
                x_np = (rng.randn(*shape) * 2 + 3).astype("float32")
                weight_np = rng.randn(shape[1]).astype("float32")
                bias_np = rng.randn(shape[1]).astype("float32")
                cot_np = rng.randn(*shape).astype("float32")
                with jt.flag_scope(use_cuda=1):
                    x, weight, bias, cot = (jt.array(a).cast(dtype)
                                            for a in (x_np, weight_np, bias_np, cot_np))
                    output = _group_norm_cuda(x, groups, weight, bias, 1e-5)
                    self.assertIsNotNone(output)
                    grads = jt.grad((output.float32() * cot.float32()).sum(), [x, weight, bias])
                    got = [a.float32().numpy() for a in [output] + list(grads)]
                # float64 reference from the values the kernel actually saw
                xr, wr, br, cr = (jt.array(a).cast(dtype).float32().numpy().astype(np.float64)
                                  for a in (x_np, weight_np, bias_np, cot_np))
                n = shape[0]
                xg = xr.reshape(n, groups, -1)
                mean = xg.mean(-1, keepdims=True)
                rstd = 1 / np.sqrt(xg.var(-1, keepdims=True) + 1e-5)
                xhat = ((xg - mean) * rstd).reshape(shape)
                y = xhat * wr[None, :, None, None] + br[None, :, None, None]
                g = (cr * wr[None, :, None, None]).reshape(n, groups, -1)
                xh = xhat.reshape(n, groups, -1)
                gx = (rstd * (g - g.mean(-1, keepdims=True)
                              - xh * (g * xh).mean(-1, keepdims=True))).reshape(shape)
                gw = (cr * xhat).sum((0, 2, 3))
                gb = cr.sum((0, 2, 3))
                tol = 2e-3 if dtype == "float32" else 3e-2
                for name, value, ref in zip(("y", "grad_x", "grad_weight", "grad_bias"),
                                            got, (y, gx, gw, gb)):
                    np.testing.assert_allclose(value, ref, rtol=tol,
                                               atol=tol * np.abs(ref).max(), err_msg=name)

    @unittest.skipUnless(_test_capability.check_accelerator('cuda', backend=jt).enabled, "CUDA GroupNorm fast path needs CUDA")
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
                        _group_norm_cuda(x, groups, weight, bias, 1e-5)
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

    @unittest.skipUnless(_test_capability.check_accelerator('cuda', backend=jt).enabled, "CUDA GroupNorm fast path needs CUDA")
    def test_cuda_forward_hands_statistics_not_a_full_size_intermediate(self):
        """The forward must not stash a whole normalized feature map (8.20).

        Carrying ``xhat`` to the backward costs an extra full-size write in the
        forward and an extra full-size allocation for the whole forward-backward
        interval; ``mean`` and ``rstd`` are two floats per group, and the
        backward recomputes ``xhat`` from ``x`` at the same memory traffic.
        LayerNorm and BatchNorm here, and torch's ``native_group_norm``, all
        already hand over the statistics.

        Counted with ``use_stat_allocator=2``, which sits *above* the caching
        allocator and therefore sees every var the operators ask for rather
        than only the requests that missed the device cache. The two
        ``jt.code`` operators are driven through the ``Function`` directly: a
        loss like ``(y * cotangent).sum()`` would allocate full-size
        intermediates of its own and bury the quantity under test.
        """
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            shape, groups = (2, 32, 16, 16), 8
            full_size = 4 * int(np.prod(shape))
            rng = np.random.RandomState(20260906)

            with jt.flag_scope(use_cuda=1):
                x = jt.array(rng.randn(*shape).astype("float32"))
                weight = jt.array(rng.randn(shape[1]).astype("float32"))
                bias = jt.array(rng.randn(shape[1]).astype("float32"))
                cotangent = jt.array(rng.randn(*shape).astype("float32"))
                cls = _group_norm_cuda_cls(shape, groups, 1e-5)

                def forward_and_backward():
                    function = cls()
                    output = function.execute(x, weight, bias)
                    grads = function.grad(cotangent)
                    jt.sync([output] + list(grads), device_sync=True)

                forward_and_backward()          # compile outside the measurement
                jt.sync_all(True)
                _test_policy_stack.enter_context(jt.runtime.scope(use_stat_allocator=2))
                try:
                    forward_and_backward()
                    allocated = int(jt.introspection.counters.allocator.allocated_bytes)
                finally:
                    _test_policy_stack.enter_context(jt.runtime.scope(use_stat_allocator=0))

            # y and grad_x are unavoidable, so anything below 1.5 copies means the
            # counter did not see the operators at all and this assertion would be
            # passing for the wrong reason.
            self.assertGreater(
                allocated, 1.5 * full_size,
                "the allocator counter saw %d B, less than the output and the "
            "input gradient together -- the measurement, not the kernel, is "
            "what changed" % allocated)
            self.assertLess(
                allocated, 2.5 * full_size,
                "GroupNorm forward+backward asked for %.2f full-size copies of "
            "%s; two (y and grad_x) is all it needs. A third means the "
            "forward is materializing xhat for the backward again"
                % (allocated / float(full_size), shape))

    @unittest.skipUnless(_test_capability.check_accelerator('cuda', backend=jt).enabled, "CUDA GroupNorm fast path needs CUDA")
    def test_cuda_fast_path_at_the_num_groups_boundaries(self):
        """num_groups of 1 and of C, and the shape the fast path must refuse.

        num_groups == C makes group_size == H*W, which is below one warp for
        the small spatial sizes here, and num_groups == 1 makes a single group
        spanning every channel: both exercise the channel index arithmetic the
        backward uses to recompute xhat.
        """
        shape = (3, 6, 5, 7)
        rng = np.random.RandomState(20260907)
        x_np = rng.randn(*shape).astype("float32")
        weight_np = rng.randn(shape[1]).astype("float32")
        bias_np = rng.randn(shape[1]).astype("float32")
        cot_np = rng.randn(*shape).astype("float32")

        for groups in (1, shape[1]):
            with self.subTest(num_groups=groups):
                def run(use_cuda):
                    with jt.flag_scope(use_cuda=use_cuda):
                        x = jt.array(x_np)
                        weight = jt.array(weight_np)
                        bias = jt.array(bias_np)
                        if use_cuda:
                            self.assertIsNotNone(
                                _group_norm_cuda(x, groups, weight, bias, 1e-5),
                                "the fast path refused num_groups=%d" % groups)
                        output = F.group_norm(x, groups, weight, bias, 1e-5)
                        grads = jt.grad((output * jt.array(cot_np)).sum(),
                                        [x, weight, bias])
                        return jt.fetch_sync([output] + grads)

                reference = run(0)
                actual = run(1)
                for name, got, expected in zip(
                        ("output", "grad_x", "grad_weight", "grad_bias"),
                        actual, reference):
                    np.testing.assert_allclose(
                        got, expected, rtol=2e-3, atol=2e-3,
                        err_msg="CUDA GroupNorm %s at num_groups=%d"
                                % (name, groups))

        with jt.flag_scope(use_cuda=1):
            # C % num_groups != 0: the kernel indexes with a compile-time
            # channels_per_group, so it must decline rather than read past a row
            self.assertIsNone(_group_norm_cuda(
                jt.array(x_np), 4, jt.array(weight_np), jt.array(bias_np), 1e-5))


class TestGroupNormActivation(unittest.TestCase):
    """`silu(group_norm(x))` runs the activation inside the group norm's pass.

    The group norm's unexecuted output offers the activation; forward and
    backward then run as the normalization's own kernels, which recompute the
    pre-activation value instead of storing it.
    """

    @unittest.skipUnless(_test_capability.check_accelerator('cuda', backend=jt).enabled,
                         "CUDA GroupNorm fast path needs CUDA")
    def test_silu_matches_and_takes_no_kernel_of_its_own(self):
        rng = np.random.RandomState(7)
        for shape in ((2, 8, 4, 4), (3, 6, 5, 7)):    # float4 path, scalar path
            x_np = rng.randn(*shape).astype("float32")
            w_np = rng.randn(shape[1]).astype("float32")
            b_np = rng.randn(shape[1]).astype("float32")
            cot_np = rng.randn(*shape).astype("float32")

            def run(use_cuda, fused=True):
                with jt.flag_scope(use_cuda=use_cuda):
                    x, w, b = jt.array(x_np), jt.array(w_np), jt.array(b_np)
                    y = F.group_norm(x, 2, w, b, 1e-5)
                    out = F.silu(y if fused else y + 0.0)
                    grads = jt.grad((out * jt.array(cot_np)).sum(), [x, w, b])
                    return [out] + grads

            with self.subTest(shape=shape):
                for got, expected in zip(jt.fetch_sync(run(1)), jt.fetch_sync(run(0))):
                    np.testing.assert_allclose(got, expected, rtol=2e-3, atol=2e-3)

                def kernels(fused):
                    jt.sync(run(1, fused))
                    jt.sync_all(True)
                    with jt.flag_scope(use_cuda=1), jt.profile() as p:
                        jt.sync(run(1, fused))
                        jt.sync_all(True)
                    return len(p.result.kernel_records)
                self.assertLess(kernels(True), kernels(False))


class TestActivationAfterInPlaceResidual(unittest.TestCase):
    """``out = norm(x); out += r; act(out)`` -- the torchvision bottleneck.

    The in-place add rebinds the norm's output object to the sum, so the norm's
    offer to apply the activation in its own pass no longer describes it. Taking
    the offer anyway applied the activation to the normalization and dropped
    the residual.
    """

    @unittest.skipUnless(_test_capability.check_accelerator('cuda', backend=jt).enabled,
                         "the fused normalizations are CUDA kernels")
    def test_the_residual_is_kept(self):
        rng = np.random.RandomState(13)
        x_np = rng.randn(4, 8, 6, 6).astype("float32")
        r_np = 3 * rng.randn(4, 8, 6, 6).astype("float32")

        def batch_norm(x):
            bn = nn.BatchNorm2d(8)
            bn.train()
            return bn(x)

        cases = (
            ("batch_norm+relu", batch_norm, nn.relu),
            ("group_norm+silu", lambda x: F.group_norm(x, 2, jt.ones(8), jt.zeros(8), 1e-5),
             F.silu),
        )
        for name, norm, act in cases:
            def run(use_cuda, in_place):
                with jt.flag_scope(use_cuda=use_cuda):
                    out = norm(jt.array(x_np))
                    if in_place:
                        out += jt.array(r_np)
                    else:
                        out = out + jt.array(r_np)
                    return act(out)
            with self.subTest(name):
                expected = run(0, False).numpy()
                np.testing.assert_allclose(run(1, True).numpy(), expected,
                                           rtol=1e-4, atol=1e-4)


class TestBatchNormActivation(unittest.TestCase):
    """`relu(batch_norm(x))` in training runs the activation in the norm's pass.

    The statistics and the output are separate operators: the fused output
    reuses the call's statistics, which the running buffers read, so they are
    computed once and the buffers move once.
    """

    @unittest.skipUnless(_test_capability.check_accelerator('cuda', backend=jt).enabled,
                         "CUDA batch norm fast path needs CUDA")
    def test_relu_matches_and_takes_no_kernel_of_its_own(self):
        rng = np.random.RandomState(11)
        for shape in ((4, 8, 6, 6), (3, 5, 7, 7)):    # float4 path, scalar path
            x_np = rng.randn(*shape).astype("float32")
            cot_np = rng.randn(*shape).astype("float32")

            def run(use_cuda, fused=True):
                with jt.flag_scope(use_cuda=use_cuda):
                    bn = nn.BatchNorm2d(shape[1])
                    bn.train()
                    x = jt.array(x_np)
                    y = bn(x)
                    out = nn.relu(y if fused else y + 0.0)
                    grads = jt.grad((out * jt.array(cot_np)).sum(), [x, bn.weight, bn.bias])
                    return [out] + grads + [bn.running_mean, bn.running_var]

            with self.subTest(shape=shape):
                for got, expected in zip(jt.fetch_sync(run(1)), jt.fetch_sync(run(0))):
                    np.testing.assert_allclose(got, expected, rtol=2e-3, atol=2e-3)
                for got, expected in zip(jt.fetch_sync(run(1)), jt.fetch_sync(run(1, False))):
                    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

    @unittest.skipUnless(_test_capability.check_accelerator('cuda', backend=jt).enabled,
                         "CUDA batch norm fast path needs CUDA")
    def test_the_relu_forward_is_not_a_kernel(self):
        x = jt.array(np.random.RandomState(12).randn(4, 8, 6, 6).astype("float32"))

        def kernels(fused):
            with jt.flag_scope(use_cuda=1):
                bn = nn.BatchNorm2d(8)
                bn.train()
                jt.sync([nn.relu(bn(x) if fused else bn(x) + 0.0)])
                jt.sync_all(True)
                with jt.profile() as p:
                    jt.sync([nn.relu(bn(x) if fused else bn(x) + 0.0)])
                    jt.sync_all(True)
            return len(p.result.kernel_records)
        self.assertLess(kernels(True), kernels(False))


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
    @unittest.skipUnless(_test_capability.check_accelerator('cuda', backend=jt).enabled, "CUDA BatchNorm fast path needs CUDA")
    def test_cuda_statistics_over_many_segments_and_odd_planes(self):
        # The reduction is cut into segments per channel and combined; the
        # elementwise part takes float4 only where the plane allows it. A
        # large mean next to a small spread checks the variance's accuracy.
        from jittor.backends.cuda.kernels.nn.batch_norm_training_cuda import (
            _batch_norm_cuda_statistics,
        )
        for shape in ((16, 8, 28, 28), (32, 12, 7, 7), (4, 3, 9, 13)):
            rng = np.random.RandomState(sum(shape))
            x_np = (rng.randn(*shape) * 0.5 + 40.0).astype("float32")
            weight_np = rng.randn(shape[1]).astype("float32")
            bias_np = rng.randn(shape[1]).astype("float32")
            cot_np = rng.randn(*shape).astype("float32")
            with jt.flag_scope(use_cuda=1):
                x, weight, bias = (jt.array(a) for a in (x_np, weight_np, bias_np))
                y, mean, var = _batch_norm_cuda_statistics(x, weight, bias, 1e-5)
                grads = jt.grad((y * jt.array(cot_np)).sum(), [x, weight, bias])
                got = jt.fetch_sync([y, mean, var] + grads)
            x64 = x_np.astype(np.float64)
            mean_ref = x64.mean((0, 2, 3))
            var_ref = x64.var((0, 2, 3))
            rstd = 1 / np.sqrt(var_ref + 1e-5)
            xhat = (x64 - mean_ref[None, :, None, None]) * rstd[None, :, None, None]
            y_ref = xhat * weight_np[None, :, None, None] + bias_np[None, :, None, None]
            g = cot_np.astype(np.float64)
            gw = (g * xhat).sum((0, 2, 3))
            gb = g.sum((0, 2, 3))
            n = shape[0] * shape[2] * shape[3]
            gx = (weight_np * rstd)[None, :, None, None] * (
                g - (gb / n)[None, :, None, None] - xhat * (gw / n)[None, :, None, None])
            for name, value, ref in zip(("y", "mean", "var", "grad_x", "grad_weight", "grad_bias"),
                                        got, (y_ref, mean_ref, var_ref, gx, gw, gb)):
                np.testing.assert_allclose(value, ref, rtol=2e-3, atol=2e-3 * np.abs(ref).max(),
                                           err_msg="%s %s" % (shape, name))

    @unittest.skipUnless(_test_capability.check_accelerator('cuda', backend=jt).enabled, "CUDA BatchNorm eval fast path needs CUDA")
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
                    output = _batch_norm_eval_cuda(
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

    @unittest.skipUnless(_test_capability.check_accelerator('cuda', backend=jt).enabled, "CUDA BatchNorm fast path needs CUDA")
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
                    output = _batch_norm_cuda(
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
    @unittest.skipUnless(_test_capability.check_accelerator('cuda', backend=jt).enabled, "CUDA channel bias fast path needs CUDA")
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
                    output = _channel_bias_add_cuda(x, bias)
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


class TestConvBiasFuses(unittest.TestCase):
    """A training convolution adds its bias as an ordinary broadcast add.

    The add then fuses with what follows it -- a UNet's time-embedding add,
    a residual add -- and its gradient is an ordinary reduction, where a
    dedicated kernel pair wrote the biased output out and read it back:
    0.66 ms of a DDPM UNet training step.
    """

    @unittest.skipUnless(_test_capability.check_accelerator('cuda', backend=jt).enabled,
                         "cuDNN convolution needs CUDA")
    def test_the_bias_add_takes_no_kernel_of_its_own(self):
        rng = np.random.RandomState(7)
        x_np = rng.randn(2, 8, 10, 10).astype("float32")
        w_np = rng.randn(16, 8, 3, 3).astype("float32")
        b_np = rng.randn(16).astype("float32")
        e_np = rng.randn(2, 16, 1, 1).astype("float32")

        def run(use_cuda):
            with jt.flag_scope(use_cuda=use_cuda):
                x, w, b, e = (jt.array(t) for t in (x_np, w_np, b_np, e_np))
                y = jt.nn.conv2d(x, w, b, padding=1) + e
                grads = jt.grad((y * y).sum(), [x, w, b])
                return [y] + grads

        for got, expected in zip(jt.fetch_sync(run(1)), jt.fetch_sync(run(0))):
            np.testing.assert_allclose(got, expected, rtol=2e-3, atol=2e-3)
        with jt.flag_scope(use_cuda=1), jt.profile() as p:
            jt.sync(run(1))
            jt.sync_all(True)
        names = [k["name"] for k in p.result.kernel_records]
        self.assertFalse([n for n in names if "channel_bias" in n], names)


class TestNormalizeIsOneImplementation(unittest.TestCase):
    """``jt.normalize`` and ``jt.nn.normalize`` were two same-named functions
    with DIFFERENT semantics. They are now one, on torch's rule.

    Reference values checked against real PyTorch 2.12.1
    (``torch.nn.functional.normalize``, defaults ``p=2, dim=1, eps=1e-12``,
    formula ``v / max(||v||_p, eps)``) in a subprocess; the two-stage form is
    "the reference below == torch" first, then "jittor == reference".

    The three ways the ``jt.normalize`` spelling used to differ:

    * ``eps`` clamped the SUM OF SQUARES, not the norm, so its effective floor
      was ``sqrt(eps)``; with the old default ``1e-30`` that floor was
      ``1e-15`` and ``normalize([1e-20, 0])`` came out 1000x larger than torch.
    * ``p=1`` had no eps protection at all, so **a zero vector gave NaN**.
    * ``p=inf`` tripped an ``assert``.
    """

    X = [[3.0, 4.0], [0.0, 0.0], [1e-20, 0.0]]

    # stage 1: exactly what torch 2.12.1 returns for self.X
    TORCH = {
        2: [[0.6, 0.8], [0.0, 0.0], [1e-08, 0.0]],
        1: [[3.0 / 7.0, 4.0 / 7.0], [0.0, 0.0], [1e-08, 0.0]],
        float("inf"): [[0.75, 1.0], [0.0, 0.0], [1e-08, 0.0]],
        3: [[0.666971743106842, 0.8892956972122192], [0.0, 0.0], [1e-08, 0.0]],
    }

    def _x(self):
        return jt.array(np.array(self.X, dtype="float32"))

    def test_both_spellings_match_torch(self):
        for p, expected in self.TORCH.items():
            for name, fn in (("jt.normalize", jt.normalize),
                             ("jt.nn.normalize", jt.nn.normalize)):
                with self.subTest(p=p, fn=name):
                    got = fn(self._x(), p=p, dim=1).numpy()
                    np.testing.assert_allclose(got, expected, rtol=1e-5,
                                               atol=1e-12)

    def test_the_two_spellings_agree_with_each_other(self):
        for p in self.TORCH:
            with self.subTest(p=p):
                np.testing.assert_allclose(
                    jt.normalize(self._x(), p=p, dim=1).numpy(),
                    jt.nn.normalize(self._x(), p=p, dim=1).numpy(),
                    rtol=0, atol=0)

    def test_zero_vector_is_not_nan(self):
        # jt.normalize(x, p=1) used to divide by an unclamped sum of absolute
        # values, so an all-zero row produced NaN and poisoned everything after
        for p in (1, 2, float("inf")):
            with self.subTest(p=p):
                got = jt.normalize(self._x(), p=p, dim=1).numpy()
                assert not np.isnan(got).any(), \
                    "p=%s: a zero row must normalize to zeros, not NaN" % p
                np.testing.assert_allclose(got[1], [0.0, 0.0], atol=0)

    def test_var_method_spelling_agrees(self):
        x = self._x()
        np.testing.assert_allclose(
            x.normalize().numpy(), jt.nn.normalize(x).numpy(), rtol=0, atol=0)

    def test_default_eps_is_torchs(self):
        import inspect
        for fn in (jt.normalize, jt.nn.normalize):
            with self.subTest(fn=fn.__name__):
                self.assertEqual(
                    inspect.signature(fn).parameters["eps"].default, 1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestBatchNormEvalCoefficients(unittest.TestCase):
    """Inference batch norm keeps its per-channel scale and shift between calls."""

    def _check(self):
        from jittor.nn.functional.normalization import _EVAL_COEFFICIENTS, batch_norm
        rng = np.random.RandomState(0)
        x = jt.array(rng.randn(4, 3, 5, 5).astype("float32"))
        mean = jt.array(rng.randn(3).astype("float32")).stop_grad()
        var = jt.array(rng.rand(3).astype("float32") + 0.5).stop_grad()
        weight = jt.array(rng.randn(3).astype("float32")).stop_grad()
        bias = jt.array(rng.randn(3).astype("float32")).stop_grad()

        def want():
            m, v, w, b = (t.numpy().reshape(1, 3, 1, 1) for t in (mean, var, weight, bias))
            return (x.numpy() - m) / np.sqrt(v + 1e-5) * w + b

        with jt.no_grad():
            first = batch_norm(x, mean, var, weight, bias, training=False).numpy()
            kept = getattr(var, _EVAL_COEFFICIENTS, None)
            np.testing.assert_allclose(first, want(), rtol=1e-5, atol=1e-5)
            second = batch_norm(x, mean, var, weight, bias, training=False).numpy()
            np.testing.assert_allclose(second, first, rtol=0, atol=0)
            # A new statistic is a new Var: the kept coefficients are not reused.
            var.update(var * 2)
            third = batch_norm(x, mean, var, weight, bias, training=False).numpy()
            np.testing.assert_allclose(third, want(), rtol=1e-5, atol=1e-5)
        # A graph that differentiates through the parameters keeps nothing.
        grad_weight = jt.array(np.ones(3, "float32"))
        other = jt.array(np.ones(3, "float32")).stop_grad()
        batch_norm(x, mean, other, grad_weight, bias, training=False)
        self.assertIsNone(getattr(other, _EVAL_COEFFICIENTS, None))
        return kept

    def test_cpu(self):
        with jt.flag_scope(use_cuda=0):
            self.assertIsNotNone(self._check())

    @unittest.skipIf(not _test_capability.check_accelerator("cuda", backend=jt).enabled,
                     "no usable CUDA in this build")
    def test_cuda(self):
        with jt.flag_scope(use_cuda=1):
            self._check()

# ***************************************************************
# Copyright (c) 2026 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""CUDA training and vectorized inference parity for ``jt.nn.fused_add_rms_norm``."""

import unittest

import numpy as np

import jittor as jt
from _helpers import capability as _test_capability
from jittor._runtime.dispatch import select_kernel


def _reference_add_rms_norm(x, residual, weight, eps, dy, dt, weight_offset=0.0):
    """Exact float64 reference for forward and gradients."""
    xf = x.astype(np.float64)
    rf = residual.astype(np.float64)
    wf = weight.astype(np.float64) + float(weight_offset)
    total = xf + rf
    rstd = 1.0 / np.sqrt((total * total).mean(axis=-1, keepdims=True) + eps)
    xhat = total * rstd
    normed = xhat * wf
    grad_total = np.zeros_like(total)
    grad_w = np.zeros_like(wf)
    if dy is not None:
        dyf = dy.astype(np.float64)
        g = dyf * wf
        mean_gx = (g * xhat).mean(axis=-1, keepdims=True)
        grad_total = grad_total + rstd * (g - xhat * mean_gx)
        axes = tuple(range(total.ndim - 1))
        grad_w = (dyf * xhat).sum(axis=axes)
    if dt is not None:
        grad_total = grad_total + dt.astype(np.float64)
    return normed, total, grad_total, grad_total, grad_w


@unittest.skipIf(
    not _test_capability.check_accelerator("cuda", backend=jt).enabled,
    "no usable CUDA in this build",
)
class TestFusedAddRMSNormTraining(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261125)

    def _tol(self, dtype):
        if dtype == "float32":
            return 2e-4, 2e-4
        if dtype == "float16":
            return 2e-2, 2e-2
        return 5e-2, 5e-2

    def _run_case(self, shape, dtype, mode="both", eps=1e-6, weight_offset=0.0,
                  strided=False):
        hidden = shape[-1]
        x_np = (self.rng.standard_normal(shape) * 0.5).astype(np.float32)
        r_np = (self.rng.standard_normal(shape) * 0.5).astype(np.float32)
        w_np = (self.rng.standard_normal((hidden,)) * 0.5 + 1.0).astype(np.float32)
        dy_np = (self.rng.standard_normal(shape) * 0.5).astype(np.float32) if mode in ("both", "normed") else None
        dt_np = (self.rng.standard_normal(shape) * 0.5).astype(np.float32) if mode in ("both", "total") else None

        with jt.flag_scope(use_cuda=1, backend_fallback="error", auto_graph_replay=0):
            x = jt.array(x_np).cast(dtype)
            r = jt.array(r_np).cast(dtype)
            w = jt.array(w_np).cast(dtype)
            # Round inputs to target dtype for exact reference comparison.
            x_q = x.float32().numpy()
            r_q = r.float32().numpy()
            w_q = w.float32().numpy()
            x = jt.array(x_q).cast(dtype).start_grad()
            r = jt.array(r_q).cast(dtype).start_grad()
            w = jt.array(w_q).cast(dtype).start_grad()

            x_in = x.transpose(0, 1).transpose(0, 1) if (strided and len(shape) >= 2) else x
            r_in = r.transpose(0, 1).transpose(0, 1) if (strided and len(shape) >= 2) else r
            y, t = jt.nn.fused_add_rms_norm(
                x_in, r_in, w, eps=eps, weight_offset=weight_offset
            )
            dy_q = dt_q = None
            terms = []
            if dy_np is not None:
                dy = jt.array(dy_np).cast(dtype).stop_grad()
                dy_q = dy.float32().numpy()
                if strided and len(shape) >= 2:
                    terms.append((y.transpose(0, 1) * dy.transpose(0, 1)).sum())
                else:
                    terms.append((y * dy).sum())
            if dt_np is not None:
                dt = jt.array(dt_np).cast(dtype).stop_grad()
                dt_q = dt.float32().numpy()
                if strided and len(shape) >= 2:
                    terms.append((t.transpose(0, 1) * dt.transpose(0, 1)).sum())
                else:
                    terms.append((t * dt).sum())
            loss = terms[0] if len(terms) == 1 else terms[0] + terms[1]
            gx, gr, gw = jt.grad(loss, [x, r, w])
            jt.sync([y, t, gx, gr, gw])
            for tensor in (y, t, gx, gr, gw):
                self.assertGreaterEqual(tensor.device_id, 0)
            got = [v.float32().numpy() for v in (y, t, gx, gr, gw)]

        ref = _reference_add_rms_norm(
            x_q, r_q, w_q, eps, dy_q, dt_q, weight_offset=weight_offset
        )
        rtol, atol = self._tol(dtype)
        for name, actual, expected in zip(("normed", "total", "dx", "dr", "dw"), got, ref):
            scale = max(1.0, float(np.abs(expected).max()))
            np.testing.assert_allclose(
                actual, expected, rtol=rtol, atol=atol * scale,
                err_msg=f"mismatch on {name} for shape={shape} dtype={dtype} mode={mode}",
            )

    def test_dispatch_selection_and_declines(self):
        from jittor.backends.cuda.kernels.nn.rms_norm_training_cuda import (
            _add_rms_norm_training_cuda,
        )
        with jt.flag_scope(use_cuda=0):
            x = jt.ones((4, 64)).start_grad()
            r = jt.ones((4, 64)).start_grad()
            w = jt.ones((64,)).start_grad()
            self.assertIsNone(select_kernel("nn.add_rms_norm.training", x, r, w, 1e-6))

        with jt.flag_scope(use_cuda=1):
            x = jt.ones((4, 64)).start_grad()
            r = jt.ones((4, 64)).start_grad()
            w = jt.ones((64,)).start_grad()
            self.assertIs(
                select_kernel("nn.add_rms_norm.training", x, r, w, 1e-6),
                _add_rms_norm_training_cuda.__wrapped__,
            )
            with jt.no_grad():
                self.assertIsNone(select_kernel("nn.add_rms_norm.training", x, r, w, 1e-6))
            self.assertIsNone(
                select_kernel("nn.add_rms_norm.training", x, r.float16(), w, 1e-6)
            )
            self.assertIsNone(
                select_kernel("nn.add_rms_norm.training", x.float64(), r.float64(), w.float64(), 1e-6)
            )

    def test_forward_and_all_gradients_parity(self):
        shapes = (
            (4, 31),
            (128, 31),
            (128, 64),
            (8, 16, 256),
            (72, 768),
            (4, 2048),
            (32, 4096),
            (8, 5120),
            (4, 8200),
        )
        for dtype in ("float32", "float16", "bfloat16"):
            for shape in shapes:
                with self.subTest(dtype=dtype, shape=shape):
                    self._run_case(shape, dtype, mode="both")

    def test_all_three_gradient_paths(self):
        for dtype in ("float32", "float16", "bfloat16"):
            for shape in ((128, 256), (16, 2048), (4, 8200)):
                for mode in ("normed", "total", "both"):
                    with self.subTest(dtype=dtype, shape=shape, mode=mode):
                        self._run_case(shape, dtype, mode=mode)

    def test_distinct_consumers_and_aliased_inputs(self):
        shape = (96, 128)
        x_np = self.rng.standard_normal(shape).astype(np.float32) * 0.5
        r_np = self.rng.standard_normal(shape).astype(np.float32) * 0.5
        w_np = (self.rng.standard_normal((128,)) * 0.5 + 1.0).astype(np.float32)
        dy_np = self.rng.standard_normal(shape).astype(np.float32) * 0.5
        dt_np = self.rng.standard_normal(shape).astype(np.float32) * 0.5
        ax_np = self.rng.standard_normal(shape).astype(np.float32) * 0.5
        ar_np = self.rng.standard_normal(shape).astype(np.float32) * 0.5

        with jt.flag_scope(use_cuda=1, backend_fallback="error", auto_graph_replay=0):
            # Separate extra consumers on x and residual ensure shared d_sum
            # never cross-contaminates gx and gr.
            x = jt.array(x_np).start_grad()
            r = jt.array(r_np).start_grad()
            w = jt.array(w_np).start_grad()
            y, t = jt.nn.fused_add_rms_norm(x, r, w, eps=1e-6)
            loss = (
                (y * jt.array(dy_np)).sum()
                + (t * jt.array(dt_np)).sum()
                + (x * jt.array(ax_np)).sum()
                + (r * jt.array(ar_np)).sum()
            )
            gx, gr, gw = jt.grad(loss, [x, r, w])
            got = [v.numpy() for v in (gx, gr, gw)]

            # Aliased x and residual (x + x) must double the gradient into x.
            xa = jt.array(x_np).start_grad()
            wa = jt.array(w_np).start_grad()
            ya, ta = jt.nn.fused_add_rms_norm(xa, xa, wa, eps=1e-6)
            loss_a = (ya * jt.array(dy_np)).sum() + (ta * jt.array(dt_np)).sum()
            gxa, gwa = jt.grad(loss_a, [xa, wa])
            got_alias = [v.numpy() for v in (gxa, gwa)]

        _, _, ref_dx, ref_dr, ref_dw = _reference_add_rms_norm(
            x_np, r_np, w_np, 1e-6, dy_np, dt_np
        )
        np.testing.assert_allclose(got[0], ref_dx + ax_np, rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(got[1], ref_dr + ar_np, rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(got[2], ref_dw, rtol=2e-4, atol=2e-4)

        _, _, ref_a_dx, _, ref_a_dw = _reference_add_rms_norm(
            x_np, x_np, w_np, 1e-6, dy_np, dt_np
        )
        np.testing.assert_allclose(got_alias[0], 2.0 * ref_a_dx, rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(got_alias[1], ref_a_dw, rtol=2e-4, atol=2e-4)

    def test_frozen_inputs_weight_offset_and_strided(self):
        self._run_case((64, 256), "float32", mode="both", weight_offset=1.0)
        self._run_case((8, 16, 256), "float32", mode="both", strided=True)

        shape = (64, 128)
        x_np = self.rng.standard_normal(shape).astype(np.float32) * 0.5
        r_np = self.rng.standard_normal(shape).astype(np.float32) * 0.5
        w_np = (self.rng.standard_normal((128,)) * 0.5 + 1.0).astype(np.float32)
        dy_np = self.rng.standard_normal(shape).astype(np.float32) * 0.5
        dt_np = self.rng.standard_normal(shape).astype(np.float32) * 0.5
        _, _, ref_dx, ref_dr, ref_dw = _reference_add_rms_norm(
            x_np, r_np, w_np, 1e-6, dy_np, dt_np
        )
        for flags in ((True, False, False), (False, True, False), (False, False, True)):
            with self.subTest(trainable=flags):
                with jt.flag_scope(use_cuda=1, backend_fallback="error", auto_graph_replay=0):
                    vars_ = [jt.array(a) for a in (x_np, r_np, w_np)]
                    for v, req in zip(vars_, flags):
                        v.start_grad() if req else v.stop_grad()
                    y, t = jt.nn.fused_add_rms_norm(vars_[0], vars_[1], vars_[2], 1e-6)
                    loss = (y * jt.array(dy_np)).sum() + (t * jt.array(dt_np)).sum()
                    targets = [v for v, req in zip(vars_, flags) if req]
                    grads = [g.numpy() for g in jt.grad(loss, targets)]
                expected = [e for e, req in zip((ref_dx, ref_dr, ref_dw), flags) if req]
                for g, e in zip(grads, expected):
                    np.testing.assert_allclose(g, e, rtol=2e-4, atol=2e-4)

    def test_vectorized_inference_parity(self):
        for dtype in ("float32", "float16", "bfloat16"):
            for shape in ((16, 31), (16, 256), (8, 2048), (4, 4096), (4, 5120)):
                for round_before_weight in (True, False):
                    for fp32_residual in (False, True):
                        if fp32_residual and round_before_weight:
                            continue
                        with self.subTest(
                            dtype=dtype, shape=shape,
                            rbw=round_before_weight, fp32_res=fp32_residual
                        ):
                            x_np = (self.rng.standard_normal(shape) * 0.5).astype(np.float32)
                            r_np = (self.rng.standard_normal(shape) * 0.5).astype(np.float32)
                            w_np = (self.rng.standard_normal((shape[-1],)) * 0.5 + 1.0).astype(np.float32)
                            with jt.no_grad(), jt.flag_scope(use_cuda=1, auto_graph_replay=0):
                                x = jt.array(x_np).cast(dtype)
                                r = jt.array(r_np).cast(dtype)
                                w = jt.array(w_np).cast(dtype)
                                x_q = x.float32().numpy()
                                r_q = r.float32().numpy()
                                w_q = w.float32().numpy()
                                y, t = jt.nn.fused_add_rms_norm(
                                    x, r, w, eps=1e-6,
                                    round_before_weight=round_before_weight,
                                    fp32_residual=fp32_residual,
                                    weight_offset=0.5,
                                )
                                y_np, t_np = y.float32().numpy(), t.float32().numpy()
                            ref_y, ref_t, _, _, _ = _reference_add_rms_norm(
                                x_q, r_q, w_q, 1e-6, None, None, weight_offset=0.5
                            )
                            rtol, atol = self._tol(dtype)
                            np.testing.assert_allclose(y_np, ref_y, rtol=rtol, atol=atol * max(1.0, float(np.abs(ref_y).max())))
                            np.testing.assert_allclose(t_np, ref_t, rtol=rtol, atol=atol * max(1.0, float(np.abs(ref_t).max())))

    def test_materialized_grad_total_and_chained_layers(self):
        """Exercise the 5-input in-kernel grad_total path and 2-layer Transformer chaining."""
        class _DirectSeed(jt.Function):
            def __init__(self, dy, dt):
                self.dy, self.dt = dy, dt

            def execute(self, y, t):
                return jt.zeros((1,), "float32")

            def grad(self, _):
                return self.dy, self.dt

        # 1. Materialized grad_total (is_finished=True) across warp, block, and wide paths.
        for dtype in ("float32", "float16", "bfloat16"):
            rtol, atol = self._tol(dtype)
            for shape in ((128, 31), (128, 256), (16, 2048), (8, 4096), (4, 8200)):
                with self.subTest(dtype=dtype, shape=shape):
                    hidden = shape[-1]
                    x_np = (self.rng.standard_normal(shape) * 0.5).astype(np.float32)
                    r_np = (self.rng.standard_normal(shape) * 0.5).astype(np.float32)
                    w_np = (self.rng.standard_normal((hidden,)) * 0.5 + 1.0).astype(np.float32)
                    dy_np = (self.rng.standard_normal(shape) * 0.5).astype(np.float32)
                    dt_np = (self.rng.standard_normal(shape) * 0.5).astype(np.float32)
                    with jt.flag_scope(use_cuda=1, backend_fallback="error", auto_graph_replay=0):
                        xq = jt.array(x_np).cast(dtype).float32().numpy()
                        rq = jt.array(r_np).cast(dtype).float32().numpy()
                        wq = jt.array(w_np).cast(dtype).float32().numpy()
                        dyq = jt.array(dy_np).cast(dtype).float32().numpy()
                        dtq = jt.array(dt_np).cast(dtype).float32().numpy()
                        x = jt.array(xq).cast(dtype).start_grad()
                        r = jt.array(rq).cast(dtype).start_grad()
                        w = jt.array(wq).cast(dtype).start_grad()
                        dy = jt.array(dyq).cast(dtype).stop_grad()
                        dt = jt.array(dtq).cast(dtype).stop_grad()
                        jt.sync([x, r, w, dy, dt])
                        y, t = jt.nn.fused_add_rms_norm(x, r, w, eps=1e-6)
                        gx, gr, gw = jt.grad(_DirectSeed(dy, dt)(y, t), [x, r, w])
                        jt.sync([y, t, gx, gr, gw])
                        got = [v.float32().numpy() for v in (y, t, gx, gr, gw)]
                    ref = _reference_add_rms_norm(xq, rq, wq, 1e-6, dyq, dtq)
                    for name, actual, expected in zip(("normed", "total", "dx", "dr", "dw"), got, ref):
                        scale = max(1.0, float(np.abs(expected).max()))
                        np.testing.assert_allclose(
                            actual, expected, rtol=rtol, atol=atol * scale,
                            err_msg=f"direct-seed mismatch on {name} shape={shape} dtype={dtype}",
                        )

        # 2. Chained layers: layer 2 backward passes a jt.code grad_total into layer 1.
        shape = (8, 16, 2048)
        for dtype in ("float32", "bfloat16"):
            rtol, atol = self._tol(dtype)
            x_np = (self.rng.standard_normal(shape) * 0.5).astype(np.float32)
            r_np = (self.rng.standard_normal(shape) * 0.5).astype(np.float32)
            dx2_np = (self.rng.standard_normal(shape) * 0.5).astype(np.float32)
            w1_np = (self.rng.standard_normal((2048,)) * 0.5 + 1.0).astype(np.float32)
            w2_np = (self.rng.standard_normal((2048,)) * 0.5 + 1.0).astype(np.float32)
            g_np = (self.rng.standard_normal(shape) * 0.5).astype(np.float32)
            with jt.flag_scope(use_cuda=1, backend_fallback="error", auto_graph_replay=0):
                def run_chain(use_fused):
                    x = jt.array(x_np).cast(dtype).start_grad()
                    r = jt.array(r_np).cast(dtype).start_grad()
                    dx2 = jt.array(dx2_np).cast(dtype).start_grad()
                    w1 = jt.array(w1_np).cast(dtype).start_grad()
                    w2 = jt.array(w2_np).cast(dtype).start_grad()
                    g = jt.array(g_np).cast(dtype).stop_grad()
                    if use_fused:
                        y1, t1 = jt.nn.fused_add_rms_norm(x, r, w1, 1e-6)
                        y2, t2 = jt.nn.fused_add_rms_norm(y1 + dx2, t1, w2, 1e-6)
                    else:
                        t1 = x + r
                        y1 = jt.nn.rms_norm(t1, w1, 1e-6)
                        t2 = (y1 + dx2) + t1
                        y2 = jt.nn.rms_norm(t2, w2, 1e-6)
                    loss = ((y2 + t2) * g).sum()
                    grads = jt.grad(loss, [x, r, dx2, w1, w2])
                    jt.sync([y2, t2] + grads)
                    return [v.float32().numpy() for v in [y2, t2] + grads]
                got = run_chain(True)
                ref = run_chain(False)
            for a, b in zip(got, ref):
                scale = max(1.0, float(np.abs(b).max()))
                np.testing.assert_allclose(a, b, rtol=rtol, atol=atol * scale)


if __name__ == "__main__":
    unittest.main()

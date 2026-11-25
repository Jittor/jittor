# ***************************************************************
# Copyright (c) 2026 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Tests for trainable layer-level differentiable CUDA Graphs (`jt.graphed_module`)."""

import gc
import unittest

import numpy as np

import jittor as jt
from jittor import nn
from _helpers import capability as _test_capability
from _helpers.runtime_policy import preserve_policy as _test_preserve_policy


class _TransformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * mlp_ratio)
        self.fc2 = nn.Linear(d_model * mlp_ratio, d_model)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else None

    def execute(self, x):
        b, s, d = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape((b, s, 3, self.n_heads, self.head_dim))
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        attn = nn.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape((b, s, d))
        out = self.proj(attn)
        if self.drop is not None:
            out = self.drop(out)
        x = x + out
        h2 = self.norm2(x)
        mlp = self.fc2(nn.gelu(self.fc1(h2)))
        if self.drop is not None:
            mlp = self.drop(mlp)
        return x + mlp


class _TransformerStack(nn.Module):
    def __init__(self, n_layers=4, d_model=128, n_heads=4, graphed=False, warmup=1):
        super().__init__()
        raw_blocks = [_TransformerBlock(d_model=d_model, n_heads=n_heads) for _ in range(n_layers)]
        if graphed:
            raw_blocks = jt.graphed_module(raw_blocks, warmup=warmup)
        self.blocks = nn.ModuleList(raw_blocks)
        self.head = nn.Linear(d_model, 16)

    def execute(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


class _ResNetBlock(nn.Module):
    def __init__(self, channels=32, smooth=False):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.smooth = smooth

    def execute(self, x):
        act = nn.gelu if self.smooth else nn.relu
        h = act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return act(x + h)


def _copy_state(src, dst):
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    assert list(src_sd.keys()) == list(dst_sd.keys()), (
        list(src_sd.keys()), list(dst_sd.keys())
    )
    for k in src_sd:
        req = bool(dst_sd[k].requires_grad)
        arr = jt.array(src_sd[k].numpy().copy())
        if req:
            arr.start_grad()
        else:
            arr.stop_grad()
        dst_sd[k].update(arr)
    jt.sync_all(True)


@unittest.skipIf(
    not _test_capability.check_accelerator("cuda", backend=jt).enabled,
    "no usable CUDA in this build",
)
@_test_preserve_policy(jt, "use_cuda", "keep_graph", "auto_graph_replay")
class TestGraphedModule(unittest.TestCase):
    def setUp(self):
        jt.flags.use_cuda = 1
        jt.flags.auto_graph_replay = 0
        self.rng = np.random.default_rng(20261125)

    def test_multi_step_transformer_parity_with_loss_item_and_adamw(self):
        """Multi-layer Transformer with dynamic outer loss.item() and AdamW matches eager."""
        eager_model = _TransformerStack(n_layers=3, d_model=64, n_heads=4, graphed=False)
        graph_model = _TransformerStack(n_layers=3, d_model=64, n_heads=4, graphed=True, warmup=1)
        _copy_state(eager_model, graph_model)

        opt_eager = jt.optim.AdamW(eager_model.parameters(), lr=1e-3, weight_decay=0.01)
        opt_graph = jt.optim.AdamW(graph_model.parameters(), lr=1e-3, weight_decay=0.01)

        try:
            for step in range(8):
                x_np = self.rng.standard_normal((4, 16, 64)).astype(np.float32) * 0.5
                y_np = self.rng.standard_normal((4, 16, 16)).astype(np.float32) * 0.5

                # Eager step
                xe, ye = jt.array(x_np), jt.array(y_np)
                out_e = eager_model(xe)
                loss_e = ((out_e - ye) ** 2).mean()
                loss_e_val = float(loss_e.item())
                grads_e = jt.grad(loss_e, eager_model.parameters())
                grads_e_np = [g.numpy().copy() for g in grads_e]
                opt_eager.step(loss_e)
                jt.sync_all(True)

                # Graphed step (with outer loss.item() readback!)
                xg, yg = jt.array(x_np), jt.array(y_np)
                out_g = graph_model(xg)
                loss_g = ((out_g - yg) ** 2).mean()
                loss_g_val = float(loss_g.item())
                grads_g = jt.grad(loss_g, graph_model.parameters())
                grads_g_np = [g.numpy().copy() for g in grads_g]
                opt_graph.step(loss_g)
                jt.sync_all(True)

                np.testing.assert_allclose(
                    loss_g_val, loss_e_val, rtol=1e-4, atol=1e-5,
                    err_msg=f"loss mismatch at step {step}",
                )
                for idx, (gg, ge) in enumerate(zip(grads_g_np, grads_e_np)):
                    np.testing.assert_allclose(
                        gg, ge, rtol=5e-4, atol=5e-4,
                        err_msg=f"grad mismatch at step {step} param {idx}",
                    )
                for (ke, pe), (kg, pg) in zip(
                    eager_model.state_dict().items(), graph_model.state_dict().items()
                ):
                    self.assertEqual(ke, kg)
                    np.testing.assert_allclose(
                        pg.numpy(), pe.numpy(), rtol=5e-4, atol=5e-4,
                        err_msg=f"weight mismatch at step {step} {ke}",
                    )

            for blk in graph_model.blocks:
                self.assertIsNone(blk.refused)
                self.assertIsNone(blk.graph_refused)
                self.assertGreaterEqual(blk.stats["fwd_graph"], 3)
                self.assertGreaterEqual(blk.stats["bwd_graph"], 3)
        finally:
            for blk in graph_model.blocks:
                blk.invalidate()

    def test_resnet_batchnorm_training_parity(self):
        """ResNet blocks with BatchNorm running stats match eager across AdamW & ReLU+SGD."""
        class ResNet(nn.Module):
            def __init__(self, graphed=False, smooth=True):
                super().__init__()
                blocks = [_ResNetBlock(16, smooth=smooth) for _ in range(3)]
                if graphed:
                    blocks = jt.graphed_module(blocks, warmup=1)
                self.blocks = nn.ModuleList(blocks)

            def execute(self, x):
                for b in self.blocks:
                    x = b(x)
                return x

        # 1. Smooth activation + BatchNorm + AdamW (no cuDNN ReLU sign-flip amplification).
        eager_net = ResNet(graphed=False, smooth=True)
        graph_net = ResNet(graphed=True, smooth=True)
        _copy_state(eager_net, graph_net)
        opt_e = jt.optim.AdamW(eager_net.parameters(), lr=1e-3)
        opt_g = jt.optim.AdamW(graph_net.parameters(), lr=1e-3)

        try:
            for step in range(7):
                x_np = self.rng.standard_normal((4, 16, 8, 8)).astype(np.float32) * 0.5
                y_np = self.rng.standard_normal((4, 16, 8, 8)).astype(np.float32) * 0.5

                le = ((eager_net(jt.array(x_np)) - jt.array(y_np)) ** 2).mean()
                le_val = float(le.item())
                opt_e.step(le)
                jt.sync_all(True)

                lg = ((graph_net(jt.array(x_np)) - jt.array(y_np)) ** 2).mean()
                lg_val = float(lg.item())
                opt_g.step(lg)
                jt.sync_all(True)

                np.testing.assert_allclose(lg_val, le_val, rtol=1e-4, atol=1e-4)

            for (k, pe), (_, pg) in zip(
                eager_net.state_dict().items(), graph_net.state_dict().items()
            ):
                np.testing.assert_allclose(
                    pg.numpy(), pe.numpy(), rtol=5e-4, atol=5e-4,
                    err_msg=f"state mismatch on {k}",
                )
            for blk in graph_net.blocks:
                self.assertIsNone(blk.refused)
                self.assertGreaterEqual(blk.stats["fwd_graph"], 2)
                self.assertGreaterEqual(blk.stats["bwd_graph"], 2)
        finally:
            for blk in graph_net.blocks:
                blk.invalidate()

        # 2. ReLU + BatchNorm + SGD over 5 steps: exact loss, weight, and running stat parity.
        eager_relu = ResNet(graphed=False, smooth=False)
        graph_relu = ResNet(graphed=True, smooth=False)
        _copy_state(eager_relu, graph_relu)
        sgd_e = jt.optim.SGD(eager_relu.parameters(), lr=1e-2)
        sgd_g = jt.optim.SGD(graph_relu.parameters(), lr=1e-2)

        try:
            for step in range(5):
                x_np = self.rng.standard_normal((4, 16, 8, 8)).astype(np.float32) * 0.5
                y_np = self.rng.standard_normal((4, 16, 8, 8)).astype(np.float32) * 0.5

                le = ((eager_relu(jt.array(x_np)) - jt.array(y_np)) ** 2).mean()
                le_val = float(le.item())
                sgd_e.step(le)
                jt.sync_all(True)

                lg = ((graph_relu(jt.array(x_np)) - jt.array(y_np)) ** 2).mean()
                lg_val = float(lg.item())
                sgd_g.step(lg)
                jt.sync_all(True)

                np.testing.assert_allclose(lg_val, le_val, rtol=1e-4, atol=1e-4)

            for (k, pe), (_, pg) in zip(
                eager_relu.state_dict().items(), graph_relu.state_dict().items()
            ):
                np.testing.assert_allclose(
                    pg.numpy(), pe.numpy(), rtol=5e-4, atol=5e-4,
                    err_msg=f"relu+sgd state mismatch on {k}",
                )
            for blk in graph_relu.blocks:
                self.assertIsNone(blk.refused)
                self.assertGreaterEqual(blk.stats["fwd_graph"], 1)
                self.assertGreaterEqual(blk.stats["bwd_graph"], 1)
        finally:
            for blk in graph_relu.blocks:
                blk.invalidate()

    def test_gradient_accumulation_and_unfused_optimizer(self):
        """Microbatch gradient accumulation + unfused parameter rebinding (`p.update`)."""
        eager_blk = _TransformerBlock(d_model=64, n_heads=4)
        graph_blk = jt.graphed_module(_TransformerBlock(d_model=64, n_heads=4), warmup=1)
        _copy_state(eager_blk, graph_blk)

        lr = 0.02
        try:
            for step in range(6):
                micro_x = [
                    self.rng.standard_normal((4, 8, 64)).astype(np.float32) * 0.5
                    for _ in range(3)
                ]
                micro_t = [
                    self.rng.standard_normal((4, 8, 64)).astype(np.float32) * 0.5
                    for _ in range(3)
                ]

                acc_e = None
                for mx, mt in zip(micro_x, micro_t):
                    loss = ((eager_blk(jt.array(mx)) - jt.array(mt)) ** 2).mean() / 3.0
                    gs = jt.grad(loss, eager_blk.parameters())
                    acc_e = gs if acc_e is None else [a + g for a, g in zip(acc_e, gs)]
                jt.sync(acc_e)
                for p, g in zip(eager_blk.parameters(), acc_e):
                    p.update(p - lr * g)
                jt.sync_all(True)

                acc_g = None
                for mx, mt in zip(micro_x, micro_t):
                    loss = ((graph_blk(jt.array(mx)) - jt.array(mt)) ** 2).mean() / 3.0
                    gs = jt.grad(loss, graph_blk.parameters())
                    acc_g = gs if acc_g is None else [a + g for a, g in zip(acc_g, gs)]
                jt.sync(acc_g)
                for ge, gg in zip(acc_e, acc_g):
                    np.testing.assert_allclose(gg.numpy(), ge.numpy(), rtol=5e-4, atol=5e-4)
                # Deliberately unfused out-of-place `p.update(...)` rebinding!
                for p, g in zip(graph_blk.parameters(), acc_g):
                    p.update(p - lr * g)
                jt.sync_all(True)

                for pe, pg in zip(eager_blk.parameters(), graph_blk.parameters()):
                    np.testing.assert_allclose(pg.numpy(), pe.numpy(), rtol=5e-4, atol=5e-4)

            self.assertIsNone(graph_blk.refused)
            self.assertGreaterEqual(graph_blk.stats["fwd_graph"], 8)
            self.assertGreaterEqual(graph_blk.stats["bwd_graph"], 8)
        finally:
            graph_blk.invalidate()

    def test_overlapping_forward_weight_tied_recurrent_call_is_safe(self):
        """Calling the same graphed module twice before backward must not corrupt activations."""
        eager_blk = _TransformerBlock(d_model=64, n_heads=4)
        graph_blk = jt.graphed_module(_TransformerBlock(d_model=64, n_heads=4), warmup=0)
        _copy_state(eager_blk, graph_blk)

        try:
            # Prime capture with one normal fwd+bwd first.
            x0 = self.rng.standard_normal((2, 8, 64)).astype(np.float32) * 0.5
            for _ in range(4):
                g_out = graph_blk(jt.array(x0))
                gs = jt.grad((g_out ** 2).mean(), graph_blk.parameters())
                jt.sync(gs)
            self.assertGreaterEqual(graph_blk.stats["fwd_graph"], 1)

            # Now call twice in one forward before backward!
            x1_np = self.rng.standard_normal((2, 8, 64)).astype(np.float32) * 0.5
            x2_np = self.rng.standard_normal((2, 8, 64)).astype(np.float32) * 0.5
            xe1, xe2 = jt.array(x1_np).start_grad(), jt.array(x2_np).start_grad()
            loss_e = (eager_blk(xe1) ** 2).mean() + (eager_blk(xe2) ** 2).mean()
            grads_e = [g.numpy() for g in jt.grad(loss_e, [xe1, xe2] + eager_blk.parameters())]

            xg1, xg2 = jt.array(x1_np).start_grad(), jt.array(x2_np).start_grad()
            loss_g = (graph_blk(xg1) ** 2).mean() + (graph_blk(xg2) ** 2).mean()
            grads_g = [g.numpy() for g in jt.grad(loss_g, [xg1, xg2] + graph_blk.parameters())]

            for gg, ge in zip(grads_g, grads_e):
                np.testing.assert_allclose(gg, ge, rtol=5e-4, atol=5e-4)
        finally:
            graph_blk.invalidate()

    def test_shape_change_recapture_and_no_grad_switching(self):
        """Shape changes re-capture cleanly; switching train/eval+no_grad preserves accuracy."""
        eager_blk = _TransformerBlock(d_model=64, n_heads=4)
        graph_blk = jt.graphed_module(_TransformerBlock(d_model=64, n_heads=4), warmup=1)
        _copy_state(eager_blk, graph_blk)

        try:
            for shape in ((4, 8, 64), (2, 12, 64)):
                for _ in range(5):
                    x_np = self.rng.standard_normal(shape).astype(np.float32) * 0.5
                    xe = jt.array(x_np).start_grad()
                    xg = jt.array(x_np).start_grad()
                    ye = eager_blk(xe)
                    yg = graph_blk(xg)
                    np.testing.assert_allclose(yg.numpy(), ye.numpy(), rtol=5e-4, atol=5e-4)
                    ge = jt.grad((ye ** 2).mean(), [xe])[0].numpy()
                    gg = jt.grad((yg ** 2).mean(), [xg])[0].numpy()
                    np.testing.assert_allclose(gg, ge, rtol=5e-4, atol=5e-4)
            self.assertGreaterEqual(graph_blk.stats["captured"], 2)

            # Switch to eval + no_grad and back to train.
            eager_blk.eval()
            graph_blk.eval()
            with jt.no_grad():
                for _ in range(5):
                    x_np = self.rng.standard_normal((2, 12, 64)).astype(np.float32) * 0.5
                    ye = eager_blk(jt.array(x_np)).numpy()
                    yg = graph_blk(jt.array(x_np)).numpy()
                    np.testing.assert_allclose(yg, ye, rtol=5e-4, atol=5e-4)
            self.assertGreaterEqual(graph_blk.stats["eval_replayed"], 2)

            eager_blk.train()
            graph_blk.train()
            x_np = self.rng.standard_normal((2, 12, 64)).astype(np.float32) * 0.5
            xe = jt.array(x_np).start_grad()
            xg = jt.array(x_np).start_grad()
            ge = jt.grad((eager_blk(xe) ** 2).mean(), [xe])[0].numpy()
            gg = jt.grad((graph_blk(xg) ** 2).mean(), [xg])[0].numpy()
            np.testing.assert_allclose(gg, ge, rtol=5e-4, atol=5e-4)
        finally:
            graph_blk.invalidate()

    def test_dropout_philox_draws_fresh_masks_and_matches_backward(self):
        """Dropout inside graphed_module must NOT repeat the same mask across steps."""
        class DropNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(64, 64)
                self.drop = nn.Dropout(0.5)

            def execute(self, x):
                return self.drop(self.lin(x))

        net = jt.graphed_module(DropNet(), warmup=1)
        net.train()
        w_np = net.lin.weight.numpy().copy()
        x_np = np.ones((8, 64), dtype=np.float32)
        outs = []
        try:
            for _ in range(8):
                x = jt.array(x_np).start_grad()
                y = net(x)
                y_np = y.numpy().copy()
                # Check backward uses exact forward mask: d(sum(y))/dx = (mask * 2) @ W.
                gx = jt.grad(y.sum(), [x])[0].numpy()
                mask = (y_np != 0.0).astype(np.float32)
                expected_gx = (mask * 2.0) @ w_np
                np.testing.assert_allclose(gx, expected_gx, rtol=1e-4, atol=1e-4)
                outs.append(y_np)
            self.assertIsNone(net.refused)
            self.assertGreaterEqual(net.stats["fwd_graph"], 2)
            self.assertGreaterEqual(net.stats["bwd_graph"], 2)
            # Consecutive CUDA graph launches MUST produce different dropout masks!
            self.assertFalse(np.allclose(outs[-1], outs[-2]))
            self.assertFalse(np.allclose(outs[-2], outs[-3]))
        finally:
            net.invalidate()

    def test_retain_graph_repeated_backward_and_clip_grad_norm(self):
        """Repeated backward with different seeds and grad norm clipping match eager."""
        eager_blk = _TransformerBlock(d_model=64, n_heads=4)
        graph_blk = jt.graphed_module(_TransformerBlock(d_model=64, n_heads=4), warmup=1)
        _copy_state(eager_blk, graph_blk)

        try:
            # Prime CUDA graphs first.
            for _ in range(4):
                x_np = self.rng.standard_normal((4, 8, 64)).astype(np.float32) * 0.5
                xg = jt.array(x_np).start_grad()
                yg = graph_blk(xg)
                jt.sync(jt.grad((yg ** 2).mean(), [xg] + graph_blk.parameters()))
            self.assertGreaterEqual(graph_blk.stats["fwd_graph"], 1)
            self.assertGreaterEqual(graph_blk.stats["bwd_graph"], 1)

            x_np = self.rng.standard_normal((4, 8, 64)).astype(np.float32) * 0.5
            s1_np = self.rng.standard_normal((4, 8, 64)).astype(np.float32) * 0.5
            s2_np = self.rng.standard_normal((4, 8, 64)).astype(np.float32) * 0.5

            xe = jt.array(x_np).start_grad()
            ye = eager_blk(xe)
            ge1 = [g.numpy() for g in jt.grad((ye * jt.array(s1_np)).sum(), [xe] + eager_blk.parameters(), retain_graph=True)]
            ge2 = [g.numpy() for g in jt.grad((ye * jt.array(s2_np)).sum(), [xe] + eager_blk.parameters())]

            xg = jt.array(x_np).start_grad()
            yg = graph_blk(xg)
            gg1 = [g.numpy() for g in jt.grad((yg * jt.array(s1_np)).sum(), [xg] + graph_blk.parameters(), retain_graph=True)]
            gg2 = [g.numpy() for g in jt.grad((yg * jt.array(s2_np)).sum(), [xg] + graph_blk.parameters())]

            for a, b in zip(gg1, ge1):
                np.testing.assert_allclose(a, b, rtol=5e-4, atol=5e-4)
            for a, b in zip(gg2, ge2):
                np.testing.assert_allclose(a, b, rtol=5e-4, atol=5e-4)
        finally:
            graph_blk.invalidate()

    def test_refusal_and_fallback_on_host_readback_and_barrier(self):
        """Modules with host readbacks or graph_replay_barrier must refuse & run eagerly."""
        class ReadbackMod(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(16, 16)

            def execute(self, x):
                y = self.lin(x)
                if float(y.sum().item()) > 0:
                    return y * 2.0
                return y * 0.5

        class BarrierMod(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(16, 16)

            def execute(self, x):
                jt.graph_replay_barrier("custom opaque op")
                return self.lin(x)

        for cls in (ReadbackMod, BarrierMod):
            with self.subTest(cls=cls.__name__):
                eager = cls()
                wrapped = jt.graphed_module(cls(), warmup=1)
                _copy_state(eager, wrapped)
                for _ in range(4):
                    x_np = self.rng.standard_normal((4, 16)).astype(np.float32)
                    xe, xg = jt.array(x_np).start_grad(), jt.array(x_np).start_grad()
                    ye, yg = eager(xe), wrapped(xg)
                    np.testing.assert_allclose(yg.numpy(), ye.numpy(), rtol=1e-5, atol=1e-5)
                    ge = jt.grad((ye ** 2).mean(), [xe])[0].numpy()
                    gg = jt.grad((yg ** 2).mean(), [xg])[0].numpy()
                    np.testing.assert_allclose(gg, ge, rtol=1e-5, atol=1e-5)
                self.assertIsNotNone(wrapped.refused)
                self.assertEqual(wrapped.stats["fwd_graph"], 0)
                wrapped.invalidate()

    def test_param_migration_re_records_and_no_memory_leak(self):
        """Reading a parameter to host keeps exact parity; repeated reset()/GC does not leak GPU memory."""
        eager_blk = _TransformerBlock(d_model=64, n_heads=4)
        blk = jt.graphed_module(_TransformerBlock(d_model=64, n_heads=4), warmup=1)
        _copy_state(eager_blk, blk)
        x_np = self.rng.standard_normal((4, 8, 64)).astype(np.float32) * 0.5
        try:
            for _ in range(6):
                x = jt.array(x_np).start_grad()
                y = blk(x)
                gs = jt.grad((y ** 2).mean(), [x] + blk.parameters())
                jt.sync(gs)
            self.assertGreaterEqual(blk.stats["fwd_graph"], 2)
            # Trigger host readback of a parameter and verify exact post-migration parity.
            _ = blk.qkv.weight.numpy()
            xe = jt.array(x_np).start_grad()
            xg = jt.array(x_np).start_grad()
            ye = eager_blk(xe)
            yg = blk(xg)
            ge = [g.numpy() for g in jt.grad((ye ** 2).mean(), [xe] + eager_blk.parameters())]
            gg = [g.numpy() for g in jt.grad((yg ** 2).mean(), [xg] + blk.parameters())]
            np.testing.assert_allclose(yg.numpy(), ye.numpy(), rtol=5e-4, atol=5e-4)
            for a, b in zip(gg, ge):
                np.testing.assert_allclose(a, b, rtol=5e-4, atol=5e-4)

            # Verify repeated capture + reset() does not leak CUDA memory.
            blk.reset()
            gc.collect()
            jt.gc()
            jt.sync_all(True)
            base_used = int(jt.get_mem_info().total_cuda_used)
            for _ in range(4):
                for _ in range(4):
                    x = jt.array(x_np).start_grad()
                    y = blk(x)
                    jt.sync(jt.grad((y ** 2).mean(), [x] + blk.parameters()))
                blk.reset()
                gc.collect()
                jt.gc()
                jt.sync_all(True)
            after_used = int(jt.get_mem_info().total_cuda_used)
            self.assertLessEqual(after_used - base_used, 4 * 1024 * 1024)
        finally:
            blk.invalidate()
        del blk
        gc.collect()
        jt.gc()

    def test_stale_backward_after_overwrite_or_invalidate_raises(self):
        """Stale retain_graph backward after overwrite or invalidate must raise, never silently corrupt."""
        blk = jt.graphed_module(_TransformerBlock(d_model=64, n_heads=4), warmup=0)
        x1_np = self.rng.standard_normal((2, 8, 64)).astype(np.float32) * 0.5
        x2_np = self.rng.standard_normal((2, 8, 64)).astype(np.float32) * 0.5
        try:
            # Capture and run first backward with retain_graph=True.
            x1 = jt.array(x1_np).start_grad()
            y1 = blk(x1)
            jt.sync(jt.grad((y1 ** 2).mean(), [x1], retain_graph=True))

            # Subsequent forward overwrites static saved activations.
            x2 = jt.array(x2_np).start_grad()
            y2 = blk(x2)
            jt.sync(jt.grad((y2 ** 2).mean(), [x2]))

            # Re-running backward on stale y1 MUST raise RuntimeError.
            with self.assertRaises(RuntimeError):
                jt.grad((y1 ** 2).mean(), [x1])

            # Calling invalidate() between forward and backward MUST raise RuntimeError.
            x3 = jt.array(x1_np).start_grad()
            y3 = blk(x3)
            blk.invalidate()
            with self.assertRaises(RuntimeError):
                jt.grad((y3 ** 2).mean(), [x3])
        finally:
            blk.invalidate()

    def test_multi_output_partial_grad_mask_parity(self):
        """Multi-output modules support alternating partial and full backward masks."""
        class TwoOut(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin1 = nn.Linear(64, 64)
                self.lin2 = nn.Linear(64, 64)

            def execute(self, x):
                h = nn.gelu(self.lin1(x))
                return h, self.lin2(h)

        eager = TwoOut()
        graph = jt.graphed_module(TwoOut(), warmup=1)
        _copy_state(eager, graph)
        try:
            modes = ("both", "first", "second", "both", "first", "second", "both", "first", "second")
            for mode in modes:
                x_np = self.rng.standard_normal((4, 64)).astype(np.float32) * 0.5
                xe = jt.array(x_np).start_grad()
                xg = jt.array(x_np).start_grad()
                ye1, ye2 = eager(xe)
                yg1, yg2 = graph(xg)
                np.testing.assert_allclose(yg1.numpy(), ye1.numpy(), rtol=5e-4, atol=5e-4)
                np.testing.assert_allclose(yg2.numpy(), ye2.numpy(), rtol=5e-4, atol=5e-4)
                if mode == "first":
                    le, lg = (ye1 ** 2).mean(), (yg1 ** 2).mean()
                    t_e, t_g = [xe, eager.lin1.weight], [xg, graph.lin1.weight]
                elif mode == "second":
                    le, lg = (ye2 ** 2).mean(), (yg2 ** 2).mean()
                    t_e, t_g = [xe] + eager.parameters(), [xg] + graph.parameters()
                else:
                    le = (ye1 ** 2).mean() + (ye2 ** 2).mean()
                    lg = (yg1 ** 2).mean() + (yg2 ** 2).mean()
                    t_e, t_g = [xe] + eager.parameters(), [xg] + graph.parameters()
                ge = [g.numpy() for g in jt.grad(le, t_e)]
                gg = [g.numpy() for g in jt.grad(lg, t_g)]
                for a, b in zip(gg, ge):
                    np.testing.assert_allclose(a, b, rtol=5e-4, atol=5e-4)
            self.assertIsNone(graph.refused)
            self.assertGreaterEqual(graph.stats["fwd_graph"], 2)
            self.assertGreaterEqual(graph.stats["bwd_graph"], 3)
        finally:
            graph.invalidate()

    def test_fused_tail_activation_linear_gelu_parity(self):
        """Module ending in fused Linear+GELU must stage grad_outputs without pre-running cleared bwd."""
        class LinearGelu(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(64, 64)

            def execute(self, x):
                return nn.gelu(self.lin(x))

        for record in (False, True):
            with self.subTest(record=record):
                eager = LinearGelu()
                graph = jt.graphed_module(LinearGelu(), warmup=1, record=record)
                _copy_state(eager, graph)
                try:
                    for step in range(6):
                        x_np = self.rng.standard_normal((4, 64)).astype(np.float32) * 0.5
                        xe = jt.array(x_np).start_grad()
                        xg = jt.array(x_np).start_grad()
                        ye, yg = eager(xe), graph(xg)
                        np.testing.assert_allclose(yg.numpy(), ye.numpy(), rtol=1e-5, atol=1e-5)
                        ge = [g.numpy() for g in jt.grad((ye ** 2).mean(), [xe] + eager.parameters())]
                        gg = [g.numpy() for g in jt.grad((yg ** 2).mean(), [xg] + graph.parameters())]
                        for a, b in zip(gg, ge):
                            np.testing.assert_allclose(
                                a, b, rtol=1e-5, atol=1e-5,
                                err_msg=f"LinearGelu grad mismatch at step={step} record={record}",
                            )
                    if record:
                        self.assertGreaterEqual(graph.stats["fwd_graph"], 3)
                        self.assertGreaterEqual(graph.stats["bwd_graph"], 3)
                finally:
                    graph.invalidate()

    def test_bfloat16_transformer_with_async_execution_and_unfused_adamw(self):
        """bfloat16 Transformer + graphed_module + async_execution=1 must not break stream handoff."""
        eager_model = _TransformerStack(n_layers=2, d_model=64, n_heads=4, graphed=False)
        graph_model = _TransformerStack(n_layers=2, d_model=64, n_heads=4, graphed=True, warmup=1)
        _copy_state(eager_model, graph_model)
        for m in (eager_model, graph_model):
            for p in m.parameters():
                p.update(p.bfloat16())
        jt.sync_all(True)

        opt_e = jt.optim.AdamW(eager_model.parameters(), lr=1e-3, weight_decay=0.01)
        opt_g = jt.optim.AdamW(graph_model.parameters(), lr=1e-3, weight_decay=0.01)
        try:
            for step in range(6):
                x_np = self.rng.standard_normal((4, 16, 64)).astype(np.float32) * 0.5
                y_np = self.rng.standard_normal((4, 16, 16)).astype(np.float32) * 0.5
                xe, ye = jt.array(x_np).bfloat16(), jt.array(y_np).bfloat16()
                le = ((eager_model(xe) - ye) ** 2).mean()
                le_val = float(le.item())
                opt_e.step(le)

                xg, yg = jt.array(x_np).bfloat16(), jt.array(y_np).bfloat16()
                lg = ((graph_model(xg) - yg) ** 2).mean()
                lg_val = float(lg.item())
                opt_g.step(lg)

                np.testing.assert_allclose(lg_val, le_val, rtol=5e-2, atol=5e-2)
            jt.sync_all(True)
            for blk in graph_model.blocks:
                self.assertIsNone(blk.refused)
                self.assertGreaterEqual(blk.stats["fwd_graph"], 2)
                self.assertGreaterEqual(blk.stats["bwd_graph"], 2)
        finally:
            for blk in graph_model.blocks:
                blk.invalidate()


if __name__ == "__main__":
    unittest.main()

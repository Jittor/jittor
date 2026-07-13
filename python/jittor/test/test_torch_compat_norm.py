"""Torch-grade normalization-layer parity for ``import jittor as torch``.

Part of the torch-grade test-suite expansion. Like the sibling
``test_torch_compat_nn.py`` / ``test_torch_compat_math.py`` modules this is a structured
``unittest`` module: every check compares jittor-as-torch's normalization layers against
an INDEPENDENT, explicit numpy reference, and runs on BOTH CPU and CUDA (when the build
has it), so it locks torch *normalization semantics* rather than jittor self-consistency.

Covers:
  * ``BatchNorm1d``/``BatchNorm2d``/``BatchNorm3d``: train-mode normalization (BIASED
    batch var), the running-stats update (running_mean / running_var with the UNBIASED
    n/(n-1) correction torch applies), eval-mode output from running stats, affine
    weight/bias, and the ``affine=False`` path. Also the functional ``F.batch_norm``
    (eval).
  * ``LayerNorm``: single- and multi-dim ``normalized_shape``, affine weight/bias, the
    torch-2.1 ``bias=False`` path, custom ``eps``; plus functional ``F.layer_norm``.
  * ``GroupNorm``: affine, ``num_groups``==1 and ==num_channels limits, custom eps;
    plus functional ``F.group_norm``.
  * ``InstanceNorm1d``/``InstanceNorm2d``: per-(N,C) spatial normalization with/without
    affine; plus functional ``F.instance_norm``.

Notes:
  * torch's BatchNorm/LayerNorm/GroupNorm/InstanceNorm normalize with the BIASED variance
    (ddof=0) of the reduced axes; only the stored ``running_var`` of BatchNorm uses the
    unbiased (Bessel) correction. The references encode exactly that.
  * jittor modules default to train mode; we drive ``.train()`` / ``.eval()`` explicitly.

Run:  python -m jittor.test.test_torch_compat_norm
      python -m pytest python/jittor/test/test_torch_compat_norm.py
"""
import os
import unittest
import numpy as np
import jittor as torch          # the whole point: jittor IS torch here
import jittor as jt
from jittor import nn

F = nn.functional

_DEVICES = [("cpu", 0)] + ([("cuda", 1)] if jt.has_cuda else [])


def both_devices(fn):
    for name, use_cuda in _DEVICES:
        with jt.flag_scope(use_cuda=use_cuda):
            fn(name)


def t(a):
    return torch.array(a)


class Base(unittest.TestCase):
    def ac(self, got, ref, atol=1e-4, rtol=1e-5, msg=""):
        g = np.asarray(got); r = np.asarray(ref)
        self.assertEqual(tuple(g.shape), tuple(r.shape), f"shape {g.shape}!={r.shape}; {msg}")
        np.testing.assert_allclose(g, r, atol=atol, rtol=rtol, err_msg=msg)


# --------------------------------------------------------------------------- BatchNorm

class TestBatchNorm(Base):
    def test_bn1d_train_output(self):
        rng = np.random.RandomState(200)
        x = (rng.randn(8, 4) * 2 + 1).astype("float32")
        mu = x.mean(0); var = x.var(0)             # biased, as torch BN normalization
        ref = (x - mu) / np.sqrt(var + 1e-5)
        def body(dev):
            bn = nn.BatchNorm1d(4); bn.train()
            self.ac(bn(t(x)).numpy(), ref, atol=1e-4, msg=f"bn1d train out {dev}")
        both_devices(body)

    def test_bn1d_running_stats(self):
        # torch updates running_mean with the BATCH mean and running_var with the
        # UNBIASED (n/(n-1)) batch var, momentum 0.1, from init mean=0 var=1.
        rng = np.random.RandomState(201)
        x = (rng.randn(8, 4) * 2 + 1).astype("float32")
        n = 8
        mu = x.mean(0); var = x.var(0)
        rm = 0.0 + (mu - 0.0) * 0.1
        rv = 1.0 + (var * n / (n - 1) - 1.0) * 0.1
        def body(dev):
            bn = nn.BatchNorm1d(4); bn.train()
            bn(t(x))
            self.ac(bn.running_mean.numpy(), rm.astype("float32"), atol=1e-5,
                    msg=f"bn1d running_mean {dev}")
            self.ac(bn.running_var.numpy(), rv.astype("float32"), atol=1e-5,
                    msg=f"bn1d running_var {dev}")
        both_devices(body)

    def test_bn1d_eval(self):
        # eval normalizes with the stored running stats (not the batch's).
        rng = np.random.RandomState(202)
        x_train = (rng.randn(8, 4) * 2 + 1).astype("float32")
        x_eval = rng.randn(5, 4).astype("float32")
        def body(dev):
            bn = nn.BatchNorm1d(4); bn.train(); bn(t(x_train))
            rm = bn.running_mean.numpy(); rv = bn.running_var.numpy()
            bn.eval()
            ref = (x_eval - rm) / np.sqrt(rv + 1e-5)
            self.ac(bn(t(x_eval)).numpy(), ref, atol=1e-4, msg=f"bn1d eval {dev}")
        both_devices(body)

    def test_bn2d_train(self):
        rng = np.random.RandomState(203)
        x = rng.randn(4, 3, 5, 5).astype("float32")
        mu = x.mean((0, 2, 3), keepdims=True); var = x.var((0, 2, 3), keepdims=True)
        ref = (x - mu) / np.sqrt(var + 1e-5)
        def body(dev):
            bn = nn.BatchNorm2d(3); bn.train()
            self.ac(bn(t(x)).numpy(), ref, atol=1e-4, msg=f"bn2d train {dev}")
        both_devices(body)

    def test_bn2d_affine(self):
        rng = np.random.RandomState(204)
        x = rng.randn(4, 3, 4, 4).astype("float32")
        w = rng.randn(3).astype("float32"); b = rng.randn(3).astype("float32")
        mu = x.mean((0, 2, 3), keepdims=True); var = x.var((0, 2, 3), keepdims=True)
        ref = (x - mu) / np.sqrt(var + 1e-5) * w.reshape(1, 3, 1, 1) + b.reshape(1, 3, 1, 1)
        def body(dev):
            bn = nn.BatchNorm2d(3); bn.train()
            bn.weight.update(t(w)); bn.bias.update(t(b))
            self.ac(bn(t(x)).numpy(), ref, atol=1e-4, msg=f"bn2d affine {dev}")
        both_devices(body)

    def test_bn1d_no_affine(self):
        rng = np.random.RandomState(205)
        x = rng.randn(6, 4).astype("float32")
        mu = x.mean(0); var = x.var(0)
        ref = (x - mu) / np.sqrt(var + 1e-5)
        def body(dev):
            bn = nn.BatchNorm1d(4, affine=False); bn.train()
            self.ac(bn(t(x)).numpy(), ref, atol=1e-4, msg=f"bn1d no-affine {dev}")
        both_devices(body)

    def test_bn3d_train(self):
        rng = np.random.RandomState(206)
        x = rng.randn(2, 3, 2, 2, 2).astype("float32")
        mu = x.mean((0, 2, 3, 4), keepdims=True); var = x.var((0, 2, 3, 4), keepdims=True)
        ref = (x - mu) / np.sqrt(var + 1e-5)
        def body(dev):
            bn = nn.BatchNorm3d(3); bn.train()
            self.ac(bn(t(x)).numpy(), ref, atol=1e-4, msg=f"bn3d train {dev}")
        both_devices(body)

    def test_functional_batch_norm_eval(self):
        rng = np.random.RandomState(207)
        x = rng.randn(3, 4).astype("float32")
        rm = rng.randn(4).astype("float32")
        rv = (np.abs(rng.randn(4)) + 0.5).astype("float32")
        ref = (x - rm) / np.sqrt(rv + 1e-5)
        def body(dev):
            self.ac(F.batch_norm(t(x), t(rm), t(rv), training=False).numpy(), ref,
                    atol=1e-4, msg=f"F.batch_norm eval {dev}")
        both_devices(body)


# --------------------------------------------------------------------------- LayerNorm

class TestLayerNorm(Base):
    def test_ln_basic(self):
        rng = np.random.RandomState(210)
        x = rng.randn(4, 5).astype("float32")
        mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
        ref = (x - mu) / np.sqrt(var + 1e-5)
        def body(dev):
            ln = nn.LayerNorm(5)
            self.ac(ln(t(x)).numpy(), ref, atol=1e-4, msg=f"ln basic {dev}")
        both_devices(body)

    def test_ln_affine(self):
        rng = np.random.RandomState(211)
        x = rng.randn(4, 5).astype("float32")
        w = rng.randn(5).astype("float32"); b = rng.randn(5).astype("float32")
        mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
        ref = (x - mu) / np.sqrt(var + 1e-5) * w + b
        def body(dev):
            ln = nn.LayerNorm(5)
            ln.weight.update(t(w)); ln.bias.update(t(b))
            self.ac(ln(t(x)).numpy(), ref, atol=1e-4, msg=f"ln affine {dev}")
        both_devices(body)

    def test_ln_multidim(self):
        rng = np.random.RandomState(212)
        x = rng.randn(2, 3, 4).astype("float32")
        flat = x.reshape(2, -1)
        mu = flat.mean(1).reshape(2, 1, 1); var = flat.var(1).reshape(2, 1, 1)
        ref = (x - mu) / np.sqrt(var + 1e-5)
        def body(dev):
            ln = nn.LayerNorm((3, 4))
            self.ac(ln(t(x)).numpy(), ref, atol=1e-4, msg=f"ln multidim {dev}")
        both_devices(body)

    def test_ln_bias_false(self):
        # torch 2.1+: LayerNorm(..., bias=False) -> scale only, no shift param.
        rng = np.random.RandomState(213)
        x = rng.randn(4, 5).astype("float32")
        w = rng.randn(5).astype("float32")
        mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
        ref = (x - mu) / np.sqrt(var + 1e-5) * w
        def body(dev):
            ln = nn.LayerNorm(5, bias=False)
            ln.weight.update(t(w))
            self.ac(ln(t(x)).numpy(), ref, atol=1e-4, msg=f"ln bias=False {dev}")
        both_devices(body)

    def test_ln_custom_eps(self):
        rng = np.random.RandomState(214)
        x = rng.randn(4, 6).astype("float32")
        eps = 1e-3
        mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
        ref = (x - mu) / np.sqrt(var + eps)
        def body(dev):
            ln = nn.LayerNorm(6, eps=eps)
            self.ac(ln(t(x)).numpy(), ref, atol=1e-4, msg=f"ln eps {dev}")
        both_devices(body)

    def test_functional_layer_norm(self):
        rng = np.random.RandomState(215)
        x = rng.randn(2, 5).astype("float32")
        w = rng.randn(5).astype("float32"); b = rng.randn(5).astype("float32")
        mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
        ref = (x - mu) / np.sqrt(var + 1e-5) * w + b
        def body(dev):
            self.ac(F.layer_norm(t(x), (5,), t(w), t(b)).numpy(), ref, atol=1e-4,
                    msg=f"F.layer_norm {dev}")
        both_devices(body)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    def test_ln_no_grad_cuda_3d(self):
        rng = np.random.RandomState(216)
        x = rng.randn(2, 8, 32).astype("float32")
        w = rng.randn(32).astype("float32")
        b = rng.randn(32).astype("float32")
        mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
        ref = (x - mu) / np.sqrt(var + 1e-5) * w + b
        with jt.flag_scope(use_cuda=1), jt.no_grad():
            ln = nn.LayerNorm(32)
            ln.weight.update(t(w)); ln.bias.update(t(b))
            self.ac(ln(t(x)).numpy(), ref, atol=1e-4, msg="ln no_grad cuda 3d")

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    def test_ln_no_grad_cuda_dynamic_rows_share_source(self):
        rng = np.random.RandomState(219)
        arrays = [
            rng.randn(2, 8, 32).astype("float32"),
            rng.randn(3, 7, 32).astype("float32"),
        ]
        sources = []
        original_code = jt.code

        def capture_code(*args, **kwargs):
            sources.append(kwargs.get("cuda_src"))
            return original_code(*args, **kwargs)

        with jt.flag_scope(use_cuda=1), jt.no_grad():
            ln = nn.LayerNorm(32)
            inputs = [t(array) for array in arrays]
            try:
                jt.code = capture_code
                outputs = [ln(value) for value in inputs]
            finally:
                jt.code = original_code

            self.assertEqual(len(sources), 2)
            self.assertEqual(sources[0], sources[1])
            for array, output in zip(arrays, outputs):
                mean = array.mean(-1, keepdims=True)
                var = array.var(-1, keepdims=True)
                ref = (array - mean) / np.sqrt(var + 1e-5)
                self.ac(output.numpy(), ref, atol=1e-4,
                        msg="ln dynamic rows shared source")

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    def test_ln_no_grad_cuda_bfloat16_private_opt_in(self):
        rng = np.random.RandomState(220)
        x_np = rng.randn(2, 5, 1536).astype("float32")
        weight_np = (1 + 0.1 * rng.randn(1536)).astype("float32")
        bias_np = (0.1 * rng.randn(1536)).astype("float32")
        with jt.flag_scope(use_cuda=1), jt.no_grad():
            x = t(x_np).bfloat16()
            weight = t(weight_np)
            bias = t(bias_np)
            self.assertIsNone(nn._layer_norm_no_grad_cuda(
                x, (1536,), weight, bias, 1e-6))
            out = nn._layer_norm_no_grad_cuda(
                x, (1536,), weight, bias, 1e-6,
                allow_bfloat16=True)
            value = x.float32()
            mean = value.mean(dim=-1, keepdim=True)
            variance = ((value - mean) * (value - mean)).mean(
                dim=-1, keepdim=True)
            ref = ((value - mean) * jt.rsqrt(variance + 1e-6)
                   * weight + bias).bfloat16()
            out_np, ref_np = jt.fetch_sync([
                out.float32(), ref.float32(),
            ])
        self.assertEqual(str(out.dtype), "bfloat16")
        np.testing.assert_allclose(
            out_np, ref_np, atol=0.016, rtol=0.008)

        extreme_np = np.empty((4, 1536), dtype="float32")
        extreme_np[0] = 1e38
        extreme_np[1, 0::2] = 1e38
        extreme_np[1, 1::2] = -1e38
        extreme_np[2] = 0
        extreme_np[2, 0] = np.nan
        extreme_np[3] = 0
        extreme_np[3, 0] = np.inf
        with jt.flag_scope(use_cuda=1), jt.no_grad():
            extreme = t(extreme_np).bfloat16()
            out = nn._layer_norm_no_grad_cuda(
                extreme, (1536,), 1.0, 0.0, 1e-6,
                allow_bfloat16=True)
            affine_out = nn._layer_norm_no_grad_cuda(
                extreme, (1536,), jt.ones(1536), jt.zeros(1536), 1e-6,
                allow_bfloat16=True)
            out_np, affine_np = jt.fetch_sync([
                out.float32(), affine_out.float32(),
            ])
        self.assertTrue(np.isfinite(out_np[:2]).all())
        np.testing.assert_array_equal(out_np[0], np.zeros(1536))
        np.testing.assert_allclose(
            out_np[1, 0::2], 1.0, atol=0.008, rtol=0)
        np.testing.assert_allclose(
            out_np[1, 1::2], -1.0, atol=0.008, rtol=0)
        self.assertTrue(np.isnan(out_np[2:]).all())
        np.testing.assert_array_equal(affine_np[:2], out_np[:2])
        np.testing.assert_array_equal(
            np.isnan(affine_np[2:]), np.ones((2, 1536), dtype=bool))

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    def test_ln_no_grad_cuda_fast_path_float32_and_float16(self):
        rng = np.random.RandomState(218)
        original = nn._ln_normalize

        def reject_composite(*args, **kwargs):
            raise AssertionError("CUDA no-grad LayerNorm missed its fused path")

        try:
            nn._ln_normalize = reject_composite
            with jt.flag_scope(use_cuda=1), jt.no_grad():
                x = rng.randn(2, 8, 32).astype("float32")
                w = rng.randn(32).astype("float32")
                b = rng.randn(32).astype("float32")
                mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
                ref = (x - mu) / np.sqrt(var + 1e-5) * w + b
                ln = nn.LayerNorm(32)
                ln.weight = t(w)
                ln.bias = t(b)
                out = ln(t(x))
                self.assertEqual(str(out.dtype), "float32")
                self.ac(out.numpy(), ref, atol=1e-4,
                        msg="ln fused no_grad cuda float32")

                for shape in ((2, 77, 512), (2, 50, 768)):
                    hidden = shape[-1]
                    for low_variance in (False, True):
                        x = rng.randn(*shape).astype("float32")
                        if low_variance:
                            x = 1.0 + x * 1e-3
                        w = rng.randn(hidden).astype("float32")
                        b = rng.randn(hidden).astype("float32")
                        x = x.astype("float16").astype("float32")
                        w = w.astype("float16").astype("float32")
                        b = b.astype("float16").astype("float32")
                        mu = x.mean(-1, keepdims=True)
                        var = x.var(-1, keepdims=True)
                        ref = (x - mu) / np.sqrt(var + 1e-5) * w + b
                        ref = ref.astype("float16").astype("float32")
                        ln = nn.LayerNorm(hidden)
                        dtype = "float16"
                        ln.weight = t(w).cast(dtype)
                        ln.bias = t(b).cast(dtype)
                        out = ln(t(x).cast(dtype))
                        got = out.float32().numpy()
                        label = f"ln fused no_grad cuda fp16 h{hidden} lowvar={low_variance}"
                        self.assertEqual(str(out.dtype), dtype)
                        self.ac(got, ref, atol=2e-3, rtol=5e-4, msg=label)
                        rel_l2 = np.linalg.norm((got - ref).ravel()) / max(
                            np.linalg.norm(ref.ravel()), 1e-30)
                        self.assertLessEqual(rel_l2, 5e-4, label)
        finally:
            nn._ln_normalize = original

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    def test_ln_no_grad_cuda_scalar_affine_fast(self):
        rng = np.random.RandomState(217)
        x = rng.randn(4, 16, 128).astype("float32")
        old = os.environ.get("JITTOR_LAYERNORM_SCALAR_FAST")
        try:
            with jt.flag_scope(use_cuda=1), jt.no_grad():
                os.environ["JITTOR_LAYERNORM_SCALAR_FAST"] = "0"
                ref = F.layer_norm(t(x), (128,), 1.0, 0.0, eps=1e-6)
                os.environ["JITTOR_LAYERNORM_SCALAR_FAST"] = "1"
                out = F.layer_norm(t(x), (128,), 1.0, 0.0, eps=1e-6)
                self.ac(out.numpy(), ref.numpy(), atol=1e-4, msg="ln scalar affine fast")
        finally:
            if old is None:
                os.environ.pop("JITTOR_LAYERNORM_SCALAR_FAST", None)
            else:
                os.environ["JITTOR_LAYERNORM_SCALAR_FAST"] = old


# --------------------------------------------------------------------------- GroupNorm

class TestGroupNorm(Base):
    def test_gn_affine(self):
        rng = np.random.RandomState(220)
        x = rng.randn(2, 6, 4, 4).astype("float32")
        w = rng.randn(6).astype("float32"); b = rng.randn(6).astype("float32")
        groups = 3
        xg = x.reshape(2, groups, 6 // groups, 16)
        m = xg.mean((2, 3), keepdims=True); v = xg.var((2, 3), keepdims=True)
        xhat = ((xg - m) / np.sqrt(v + 1e-5)).reshape(2, 6, 4, 4)
        ref = xhat * w.reshape(1, 6, 1, 1) + b.reshape(1, 6, 1, 1)
        def body(dev):
            gn = nn.GroupNorm(groups, 6)
            gn.weight.update(t(w)); gn.bias.update(t(b))
            self.ac(gn(t(x)).numpy(), ref, atol=1e-4, msg=f"gn affine {dev}")
        both_devices(body)

    def test_gn_groups_limits(self):
        rng = np.random.RandomState(221)
        x = rng.randn(2, 6, 4, 4).astype("float32")
        for groups in [1, 6]:
            xg = x.reshape(2, groups, 6 // groups, 16)
            m = xg.mean((2, 3), keepdims=True); v = xg.var((2, 3), keepdims=True)
            ref = ((xg - m) / np.sqrt(v + 1e-5)).reshape(2, 6, 4, 4)
            def body(dev, groups=groups, ref=ref):
                gn = nn.GroupNorm(groups, 6)
                self.ac(gn(t(x)).numpy(), ref, atol=1e-4, msg=f"gn groups={groups} {dev}")
            both_devices(body)

    def test_gn_custom_eps(self):
        rng = np.random.RandomState(222)
        x = rng.randn(2, 4, 6).astype("float32")
        eps = 1e-3
        xg = x.reshape(2, 2, 2, 6)
        m = xg.mean((2, 3), keepdims=True); v = xg.var((2, 3), keepdims=True)
        ref = ((xg - m) / np.sqrt(v + eps)).reshape(2, 4, 6)
        def body(dev):
            gn = nn.GroupNorm(2, 4, eps=eps)
            self.ac(gn(t(x)).numpy(), ref, atol=1e-4, msg=f"gn eps {dev}")
        both_devices(body)

    def test_functional_group_norm(self):
        rng = np.random.RandomState(223)
        x = rng.randn(2, 4, 6).astype("float32")
        w = rng.randn(4).astype("float32"); b = rng.randn(4).astype("float32")
        xg = x.reshape(2, 2, 2, 6)
        m = xg.mean((2, 3), keepdims=True); v = xg.var((2, 3), keepdims=True)
        xhat = ((xg - m) / np.sqrt(v + 1e-5)).reshape(2, 4, 6)
        ref = xhat * w.reshape(1, 4, 1) + b.reshape(1, 4, 1)
        def body(dev):
            self.ac(F.group_norm(t(x), 2, t(w), t(b)).numpy(), ref, atol=1e-4,
                    msg=f"F.group_norm {dev}")
        both_devices(body)


# ------------------------------------------------------------------------ InstanceNorm

class TestInstanceNorm(Base):
    def test_in2d_no_affine(self):
        rng = np.random.RandomState(230)
        x = rng.randn(2, 3, 5, 5).astype("float32")
        m = x.mean((2, 3), keepdims=True); v = x.var((2, 3), keepdims=True)
        ref = (x - m) / np.sqrt(v + 1e-5)
        def body(dev):
            inn = nn.InstanceNorm2d(3, affine=False)
            self.ac(inn(t(x)).numpy(), ref, atol=1e-4, msg=f"in2d no-affine {dev}")
        both_devices(body)

    def test_in2d_affine(self):
        rng = np.random.RandomState(231)
        x = rng.randn(2, 3, 5, 5).astype("float32")
        w = rng.randn(3).astype("float32"); b = rng.randn(3).astype("float32")
        m = x.mean((2, 3), keepdims=True); v = x.var((2, 3), keepdims=True)
        ref = (x - m) / np.sqrt(v + 1e-5) * w.reshape(1, 3, 1, 1) + b.reshape(1, 3, 1, 1)
        def body(dev):
            inn = nn.InstanceNorm2d(3, affine=True)
            inn.weight.update(t(w)); inn.bias.update(t(b))
            self.ac(inn(t(x)).numpy(), ref, atol=1e-4, msg=f"in2d affine {dev}")
        both_devices(body)

    def test_in1d(self):
        rng = np.random.RandomState(232)
        x = rng.randn(2, 3, 8).astype("float32")
        m = x.mean(2, keepdims=True); v = x.var(2, keepdims=True)
        ref = (x - m) / np.sqrt(v + 1e-5)
        def body(dev):
            inn = nn.InstanceNorm1d(3, affine=False)
            self.ac(inn(t(x)).numpy(), ref, atol=1e-4, msg=f"in1d {dev}")
        both_devices(body)

    def test_functional_instance_norm(self):
        rng = np.random.RandomState(233)
        x = rng.randn(2, 3, 6).astype("float32")
        w = rng.randn(3).astype("float32"); b = rng.randn(3).astype("float32")
        m = x.mean(2, keepdims=True); v = x.var(2, keepdims=True)
        ref = (x - m) / np.sqrt(v + 1e-5) * w.reshape(1, 3, 1) + b.reshape(1, 3, 1)
        def body(dev):
            self.ac(F.instance_norm(t(x), weight=t(w), bias=t(b)).numpy(), ref,
                    atol=1e-4, msg=f"F.instance_norm {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)

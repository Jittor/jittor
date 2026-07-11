"""Torch-grade attention/transformer parity for ``import jittor as torch``.

The transformer surface is the core of the jittor-as-torch project. Compares
F.scaled_dot_product_attention and nn.MultiheadAttention against explicit numpy references.
CPU+CUDA.

Run:  python -m jittor.test.test_torch_compat_attention
"""
import unittest
import os
import numpy as np
import jittor as torch
import jittor as jt
from jittor import nn

_DEVICES = [("cpu", 0)] + ([("cuda", 1)] if jt.has_cuda else [])


def both_devices(fn):
    for name, use_cuda in _DEVICES:
        with jt.flag_scope(use_cuda=use_cuda):
            fn(name)


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _sdpa_ref(q, k, v, mask=None):
    d = q.shape[-1]
    s = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(d)
    if mask is not None:
        s = s + mask
    return _softmax(s, -1) @ v


class Base(unittest.TestCase):
    def ac(self, got, ref, atol=2e-4, rtol=2e-4, msg=""):
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref), atol=atol, rtol=rtol,
                                   err_msg=msg)


class TestSDPA(Base):
    def setUp(self):
        rng = np.random.RandomState(0)
        # (batch, heads, seq, dim)
        self.q = rng.randn(2, 3, 5, 8).astype("float32")
        self.k = rng.randn(2, 3, 5, 8).astype("float32")
        self.v = rng.randn(2, 3, 5, 8).astype("float32")

    def test_sdpa_no_mask(self):
        q, k, v = self.q, self.k, self.v
        def body(dev):
            out = torch.nn.functional.scaled_dot_product_attention(
                jt.array(q), jt.array(k), jt.array(v)).numpy()
            self.ac(out, _sdpa_ref(q, k, v), msg=f"sdpa {dev}")
        both_devices(body)

    def test_sdpa_causal(self):
        q, k, v = self.q, self.k, self.v
        seq = q.shape[-2]
        causal = np.triu(np.full((seq, seq), -np.inf, dtype="float32"), 1)
        def body(dev):
            out = torch.nn.functional.scaled_dot_product_attention(
                jt.array(q), jt.array(k), jt.array(v), is_causal=True).numpy()
            self.ac(out, _sdpa_ref(q, k, v, causal), msg=f"sdpa causal {dev}")
        both_devices(body)

    def test_sdpa_dropout_one(self):
        q, k, v = self.q, self.k, self.v

        def body(dev):
            out = torch.nn.functional.scaled_dot_product_attention(
                jt.array(q), jt.array(k), jt.array(v), dropout_p=1.0).numpy()
            self.ac(out, np.zeros_like(q), atol=0.0, rtol=0.0,
                    msg=f"sdpa dropout p=1 {dev}")

        both_devices(body)

    def test_native_sdpa_bool_mask(self):
        from jittor.attention import scaled_dot_product_attention

        q, k, v = self.q, self.k, self.v
        keep = np.tril(np.ones((q.shape[-2], k.shape[-2]), dtype=bool))
        broadcast_keep = np.broadcast_to(
            keep, q.shape[:-2] + keep.shape).copy()

        def body(dev):
            for name, mask in (("2d", keep), ("broadcast", broadcast_keep)):
                additive = np.where(mask, 0.0, -np.inf).astype("float32")
                for mask_type, value in (("bool", mask), ("additive", additive)):
                    out = scaled_dot_product_attention(
                        jt.array(q), jt.array(k), jt.array(v), jt.array(value)).numpy()
                    self.ac(out, _sdpa_ref(q, k, v, additive),
                            msg=f"native sdpa {name} {mask_type} mask {dev}")

        both_devices(body)

    def test_sdpa_fully_masked_row_is_zero_with_zero_grad(self):
        from jittor.attention import scaled_dot_product_attention as native_sdpa

        q, k, v = self.q, self.k, self.v
        keep = np.ones((q.shape[-2], k.shape[-2]), dtype=bool)
        keep[2, :] = False
        additive = np.where(keep, 0.0, -np.inf).astype("float32")
        expected = _sdpa_ref(q, k, v)
        expected[..., 2, :] = 0.0
        upstream = np.zeros_like(expected)
        upstream[..., 2, :] = 1.0

        def body(dev):
            for name, fn in (
                    ("torch_compat", torch.nn.functional.scaled_dot_product_attention),
                    ("native", native_sdpa)):
                for mask_name, mask in (("bool", keep), ("additive", additive)):
                    qv, kv, vv = jt.array(q), jt.array(k), jt.array(v)
                    out = fn(qv, kv, vv, jt.array(mask))
                    dq, dk, dv = jt.grad(
                        (out * jt.array(upstream)).sum(), [qv, kv, vv])
                    got = jt.fetch_sync([out, dq, dk, dv])
                    label = f"{name} fully masked {mask_name} {dev}"
                    self.ac(got[0], expected, atol=3e-4, rtol=3e-4,
                            msg=label)
                    for grad in got[1:]:
                        self.ac(grad, np.zeros_like(grad), atol=0.0, rtol=0.0,
                                msg=label + " grad")

        both_devices(body)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    def test_sdpa_native_flash_attn_fp16_cuda(self):
        rng = np.random.RandomState(23)
        q = rng.randn(2, 4, 8, 8).astype("float32")
        k = rng.randn(2, 4, 8, 8).astype("float32")
        v = rng.randn(2, 4, 8, 8).astype("float32")
        with jt.flag_scope(use_cuda=1), jt.no_grad():
            if hasattr(jt, "_torch_sdpa_flash_stats"):
                delattr(jt, "_torch_sdpa_flash_stats")
            out = torch.nn.functional.scaled_dot_product_attention(
                jt.array(q).float16(), jt.array(k).float16(), jt.array(v).float16())
            got = out.float32().numpy()
            stats = getattr(jt, "_torch_sdpa_flash_stats", {})
        self.assertGreaterEqual(stats.get("hits", 0), 1, "native flash-attn SDPA was not used")
        self.assertIn("flashattn_jittor", str(stats.get("backend", "")))
        self.ac(got, _sdpa_ref(q, k, v), atol=2e-3, rtol=2e-3, msg="sdpa native flash fp16 cuda")

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    def test_sdpa_native_flash_attn_float32_opt_in_cast_cuda(self):
        rng = np.random.RandomState(29)
        q = rng.randn(2, 4, 8, 8).astype("float32")
        k = rng.randn(2, 4, 8, 8).astype("float32")
        v = rng.randn(2, 4, 8, 8).astype("float32")
        old = os.environ.get("JITTOR_FLASH_ATTN_CAST_FLOAT32")
        os.environ["JITTOR_FLASH_ATTN_CAST_FLOAT32"] = "fp16"
        try:
            with jt.flag_scope(use_cuda=1), jt.no_grad():
                if hasattr(jt, "_torch_sdpa_flash_stats"):
                    delattr(jt, "_torch_sdpa_flash_stats")
                out = torch.nn.functional.scaled_dot_product_attention(
                    jt.array(q), jt.array(k), jt.array(v))
                self.assertEqual(str(out.dtype), "float32")
                got = out.numpy()
                stats = getattr(jt, "_torch_sdpa_flash_stats", {})
        finally:
            if old is None:
                os.environ.pop("JITTOR_FLASH_ATTN_CAST_FLOAT32", None)
            else:
                os.environ["JITTOR_FLASH_ATTN_CAST_FLOAT32"] = old
        self.assertGreaterEqual(stats.get("hits", 0), 1, "native flash-attn SDPA was not used")
        self.assertGreaterEqual(stats.get("casts", {}).get("float32_to_float16", 0), 1)
        self.ac(got, _sdpa_ref(q, k, v), atol=2e-3, rtol=2e-3, msg="sdpa native flash fp32 opt-in cast cuda")

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    def test_flash_attn_packed_split_cuda(self):
        from jittor.torch_shim import flashattn_jittor

        old = os.environ.get("JITTOR_FLASH_ATTN_FUSED_PACKED_SPLIT")
        os.environ["JITTOR_FLASH_ATTN_FUSED_PACKED_SPLIT"] = "1"
        try:
            with jt.flag_scope(use_cuda=1), jt.no_grad():
                cases = [
                    (
                        "qkv_varlen",
                        jt.array(np.arange(7 * 3 * 2 * 4, dtype=np.float16).reshape(7, 3, 2, 4)),
                        flashattn_jittor._split_qkvpacked_cuda,
                        lambda x: (x[:, 0], x[:, 1], x[:, 2]),
                    ),
                    (
                        "qkv_dense",
                        jt.array(np.arange(2 * 5 * 3 * 2 * 4, dtype=np.float16).reshape(2, 5, 3, 2, 4)),
                        flashattn_jittor._split_qkvpacked_cuda,
                        lambda x: (x[:, :, 0], x[:, :, 1], x[:, :, 2]),
                    ),
                    (
                        "kv_varlen",
                        jt.array(np.arange(7 * 2 * 2 * 4, dtype=np.float16).reshape(7, 2, 2, 4)),
                        flashattn_jittor._split_kvpacked_cuda,
                        lambda x: (x[:, 0], x[:, 1]),
                    ),
                    (
                        "kv_dense",
                        jt.array(np.arange(2 * 5 * 2 * 2 * 4, dtype=np.float16).reshape(2, 5, 2, 2, 4)),
                        flashattn_jittor._split_kvpacked_cuda,
                        lambda x: (x[:, :, 0], x[:, :, 1]),
                    ),
                ]
                start = dict(flashattn_jittor._PACKED_SPLIT_STATS)
                for name, packed, split_fn, ref_fn in cases:
                    outs = split_fn(packed)
                    self.assertIsNotNone(outs, name)
                    vals = jt.fetch_sync(list(outs) + list(ref_fn(packed)))
                    for i in range(len(outs)):
                        np.testing.assert_array_equal(vals[i], vals[i + len(outs)], err_msg=name)
                stats = flashattn_jittor._PACKED_SPLIT_STATS
                self.assertGreaterEqual(stats["qkv_cuda"] - start.get("qkv_cuda", 0), 2)
                self.assertGreaterEqual(stats["kv_cuda"] - start.get("kv_cuda", 0), 2)
        finally:
            if old is None:
                os.environ.pop("JITTOR_FLASH_ATTN_FUSED_PACKED_SPLIT", None)
            else:
                os.environ["JITTOR_FLASH_ATTN_FUSED_PACKED_SPLIT"] = old


class TestMultiheadAttention(Base):
    def test_mha_shapes_and_self_consistency(self):
        rng = np.random.RandomState(1)
        E, H, L, B = 16, 4, 6, 2
        x = rng.randn(L, B, E).astype("float32")
        def body(dev):
            mha = nn.MultiheadAttention(E, H)
            q = jt.array(x)
            out, w = mha(q, q, q)
            self.assertEqual(tuple(out.shape), (L, B, E), f"mha out shape {dev}")
            # attention weights rows sum to 1 (softmax)
            wsum = np.asarray(w.numpy()).sum(-1)
            self.ac(wsum, np.ones_like(wsum), atol=1e-4, msg=f"mha attn rows sum 1 {dev}")
            # backward produces finite gradients
            g = jt.grad(out.sum(), [p for p in mha.parameters() if not p.is_stop_grad()])
            self.assertTrue(all(bool(jt.isfinite(gi).all().item()) for gi in g),
                            f"mha grads finite {dev}")
        both_devices(body)

    def test_mha_dropout_training_and_eval(self):
        rng = np.random.RandomState(41)
        x = jt.array(rng.randn(5, 2, 8).astype("float32"))

        def body(dev):
            mha = nn.MultiheadAttention(8, 2, dropout=1.0, bias=False)
            mha.train()
            train_out, train_weights = mha(x, x, x, need_weights=True)
            train_no_weights, _ = mha(x, x, x, need_weights=False)
            got_train, got_weights, got_no_weights = jt.fetch_sync(
                [train_out, train_weights, train_no_weights])
            self.ac(got_train, np.zeros_like(got_train), atol=0.0, rtol=0.0,
                    msg=f"mha train dropout output {dev}")
            self.ac(got_weights, np.zeros_like(got_weights), atol=0.0, rtol=0.0,
                    msg=f"mha train dropout weights {dev}")
            self.ac(got_no_weights, np.zeros_like(got_no_weights), atol=0.0, rtol=0.0,
                    msg=f"mha train dropout no weights {dev}")

            mha.eval()
            eval_out, eval_weights = mha(x, x, x, need_weights=True)
            got_eval, got_eval_weights = jt.fetch_sync([eval_out, eval_weights])
            self.assertGreater(float(np.abs(got_eval).max()), 0.0,
                               f"mha eval output {dev}")
            self.ac(got_eval_weights.sum(-1),
                    np.ones_like(got_eval_weights.sum(-1)), atol=1e-5,
                    msg=f"mha eval weights {dev}")

        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)

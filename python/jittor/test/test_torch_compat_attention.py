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


if __name__ == "__main__":
    unittest.main(verbosity=2)

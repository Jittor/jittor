"""Torch-grade attention/transformer parity for ``import jittor as torch``.

The transformer surface is the core of the jittor-as-torch project. Compares
F.scaled_dot_product_attention and nn.MultiheadAttention against explicit numpy references.
CPU+CUDA.

Run:  python -m jittor.test.test_torch_compat_attention
"""
import unittest
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
from types import ModuleType
from unittest import mock
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

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    def test_dinov3_runtime_fuses_fp32_inference_rope(self):
        from jittor.torch_shim import trellis_runtime

        module = ModuleType(trellis_runtime._DINOV3_MODULE)
        calls = []

        def rotate_half(value):
            half = int(value.shape[-1]) // 2
            return jt.concat((-value[..., half:], value[..., :half]), dim=-1)

        def reference(q, k, cos, sin, **kwargs):
            calls.append(kwargs)
            patch_count = int(sin.shape[-2])
            prefix_count = int(q.shape[-2]) - patch_count
            q_prefix, q_patch = q[..., :prefix_count, :], q[..., prefix_count:, :]
            k_prefix, k_patch = k[..., :prefix_count, :], k[..., prefix_count:, :]
            q_patch = q_patch * cos + rotate_half(q_patch) * sin
            k_patch = k_patch * cos + rotate_half(k_patch) * sin
            return (
                jt.concat((q_prefix, q_patch), dim=-2),
                jt.concat((k_prefix, k_patch), dim=-2),
            )

        module.apply_rotary_pos_emb = reference
        self.assertTrue(trellis_runtime._patch_dinov3_module(module))
        patched = module.apply_rotary_pos_emb
        self.assertIs(patched, module.apply_rotary_pos_emb)
        self.assertTrue(trellis_runtime._patch_dinov3_module(module))

        rng = np.random.RandomState(109)
        q_base_np = rng.randn(2, 13, 16, 64).astype("float32")
        k_base_np = rng.randn(2, 13, 16, 64).astype("float32")
        q_np = q_base_np.transpose(0, 2, 1, 3)
        k_np = k_base_np.transpose(0, 2, 1, 3)
        cos_np = rng.randn(8, 64).astype("float32")
        sin_np = rng.randn(8, 64).astype("float32")
        with jt.flag_scope(use_cuda=1), jt.no_grad():
            q = jt.array(q_base_np).view(2, 13, 16, 64).transpose(1, 2)
            k = jt.array(k_base_np).view(2, 13, 16, 64).transpose(1, 2)
            cos, sin = jt.array(cos_np), jt.array(sin_np)
            expected = reference(q, k, cos, sin)
            call_count = len(calls)
            actual = patched(q, k, cos, sin)
            self.assertEqual(len(calls), call_count)
            fetched = jt.fetch_sync(list(expected) + list(actual))

        np.testing.assert_allclose(
            fetched[2], fetched[0], atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(
            fetched[3], fetched[1], atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(fetched[2][..., :5, :], q_np[..., :5, :])
        np.testing.assert_array_equal(fetched[3][..., :5, :], k_np[..., :5, :])

        call_count = len(calls)
        with jt.flag_scope(use_cuda=1), jt.no_grad():
            patched(
                jt.array(q_np).float16(), jt.array(k_np).float16(),
                jt.array(cos_np).float16(), jt.array(sin_np).float16(),
            )
        self.assertEqual(len(calls), call_count + 1)
        self.assertEqual(calls[-1], {})

        call_count = len(calls)
        with jt.flag_scope(use_cuda=1):
            patched(
                jt.array(q_np), jt.array(k_np),
                jt.array(cos_np), jt.array(sin_np),
            )
        self.assertEqual(len(calls), call_count + 1)
        self.assertEqual(calls[-1], {})

        with jt.flag_scope(use_cuda=1), jt.no_grad():
            patched(
                jt.array(q_np), jt.array(k_np),
                jt.array(cos_np), jt.array(sin_np),
                future_option=True,
            )
        self.assertEqual(calls[-1], {"future_option": True})

        call_count = len(calls)
        with jt.flag_scope(use_cuda=1), jt.no_grad(), mock.patch.dict(
                os.environ, {"JITTOR_DINOV3_FUSED_ROPE": "0"}, clear=False):
            patched(
                jt.array(q_np), jt.array(k_np),
                jt.array(cos_np), jt.array(sin_np),
            )
        self.assertEqual(len(calls), call_count + 1)
        self.assertEqual(calls[-1], {})

        replacement = ModuleType(trellis_runtime._DINOV3_MODULE)
        replacement._jittor_torch_fast_dinov3_rope = True
        replacement.apply_rotary_pos_emb = reference
        self.assertTrue(trellis_runtime._patch_dinov3_module(replacement))
        self.assertIsNot(replacement.apply_rotary_pos_emb, reference)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    def test_trellis_runtime_fuses_bf16_multihead_rms_norm(self):
        from jittor.torch_shim import trellis_runtime

        calls = []

        class MultiHeadRMSNorm:
            def __init__(self, gamma):
                self.gamma = gamma
                self.scale = 128 ** 0.5

            def forward(self, x, **kwargs):
                calls.append(kwargs)
                value = x.float32()
                norm = (value * value).sum(-1, keepdims=True).sqrt()
                return (value / norm.maximum(1e-12)
                        * self.gamma * self.scale).cast(str(x.dtype))

        module = ModuleType(trellis_runtime._ATTENTION_MODULE)
        module.MultiHeadRMSNorm = MultiHeadRMSNorm
        self.assertTrue(trellis_runtime._patch_attention_module(module))
        patched = MultiHeadRMSNorm.forward
        self.assertTrue(trellis_runtime._patch_attention_module(module))
        self.assertIs(patched, MultiHeadRMSNorm.forward)

        rng = np.random.RandomState(127)
        x_np = rng.randn(2, 7, 12, 128).astype("float32")
        gamma_np = (1.0 + 0.1 * rng.randn(12, 128)).astype("float32")
        with jt.flag_scope(use_cuda=1), jt.no_grad():
            x = jt.array(x_np).bfloat16()
            gamma = jt.array(gamma_np)
            instance = MultiHeadRMSNorm(gamma)
            expected = patched._jittor_torch_original(instance, x)
            call_count = len(calls)
            actual = patched(instance, x)
            self.assertEqual(len(calls), call_count)
            expected_np, actual_np = jt.fetch_sync([
                expected.float32(), actual.float32(),
            ])
        np.testing.assert_allclose(
            actual_np, expected_np, atol=0.016, rtol=0.008)

        call_count = len(calls)
        with jt.flag_scope(use_cuda=1), jt.no_grad():
            patched(instance, jt.array(x_np))
        self.assertEqual(len(calls), call_count + 1)
        self.assertEqual(calls[-1], {})

        call_count = len(calls)
        with jt.flag_scope(use_cuda=1), jt.no_grad():
            patched(instance, jt.array(x_np).bfloat16(), future_option=True)
        self.assertEqual(len(calls), call_count + 1)
        self.assertEqual(calls[-1], {"future_option": True})

        call_count = len(calls)
        with jt.flag_scope(use_cuda=1):
            patched(instance, jt.array(x_np).bfloat16())
        self.assertEqual(len(calls), call_count + 1)

        for unsupported_scale in (-128 ** 0.5, float("nan")):
            call_count = len(calls)
            instance.scale = unsupported_scale
            with jt.flag_scope(use_cuda=1), jt.no_grad():
                patched(instance, jt.array(x_np).bfloat16())
            self.assertEqual(len(calls), call_count + 1)
        instance.scale = 128 ** 0.5

        call_count = len(calls)
        with jt.flag_scope(use_cuda=1), jt.no_grad(), mock.patch.dict(
                os.environ, {"JITTOR_TRELLIS_FUSED_RMS_NORM": "0"}, clear=False):
            patched(instance, jt.array(x_np).bfloat16())
        self.assertEqual(len(calls), call_count + 1)

        replacement = ModuleType(trellis_runtime._ATTENTION_MODULE)
        replacement._jittor_torch_fast_trellis_rms_norm = True
        replacement.MultiHeadRMSNorm = type(
            "MultiHeadRMSNorm", (), {"forward": lambda self, x: x})
        original = replacement.MultiHeadRMSNorm.forward
        self.assertTrue(trellis_runtime._patch_attention_module(replacement))
        self.assertIsNot(replacement.MultiHeadRMSNorm.forward, original)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    def test_trellis_cross_kv_cache_is_opt_in_and_sampler_scoped(self):
        from jittor.torch_shim import trellis_runtime

        projection_calls = []
        received_dtypes = []

        class Projection(nn.Linear):
            def __init__(self):
                super().__init__(1024, 3072)
                self.weight = self.weight.bfloat16().stop_grad()
                self.bias = self.bias.bfloat16().stop_grad()
                self.is_train = False

            def execute(self, context):
                projection_calls.append(context)
                return super().execute(context)

        class Attention:
            _type = "cross"
            channels = 1536
            ctx_channels = 1024
            num_heads = 12
            head_dim = 128
            training = False

            def __init__(self):
                self.to_kv = Projection()

        class Model:
            training = False
            dtype = "bfloat16"

            def modules(self):
                return [attention]

        class FlowEulerSampler:
            def __init__(self):
                self.fail = False
                self.replace_weight = False

            def sample(self, model, noise, cond=None, *args, **kwargs):
                contexts = [cond, kwargs.get("neg_cond")]
                received_dtypes.append(tuple(str(value.dtype) for value in contexts))
                contexts = [value.bfloat16() for value in contexts]
                outputs = []
                for index, context in enumerate(contexts):
                    outputs.append(attention.to_kv(context))
                    if self.replace_weight and index == 0:
                        attention.to_kv.weight = jt.ones(
                            (3072, 1024), dtype="bfloat16").stop_grad()
                    outputs.append(attention.to_kv(context))
                if self.fail:
                    raise RuntimeError("expected test failure")
                return contexts, outputs

        class FlowEulerGuidanceIntervalSampler(FlowEulerSampler):
            def sample(self, model, noise, cond, neg_cond, steps=12):
                return super().sample(
                    model, noise, cond, steps, neg_cond=neg_cond)

        sampler_module = ModuleType(trellis_runtime._FLOW_EULER_MODULE)
        sampler_module.FlowEulerSampler = FlowEulerSampler
        self.assertTrue(trellis_runtime._patch_flow_euler_module(sampler_module))
        patched_sample = FlowEulerSampler.sample
        self.assertTrue(trellis_runtime._patch_flow_euler_module(sampler_module))
        self.assertIs(patched_sample, FlowEulerSampler.sample)

        model = Model()
        sampler = FlowEulerGuidanceIntervalSampler()
        with jt.flag_scope(use_cuda=1), jt.no_grad():
            attention = Attention()
            source = jt.randn((1, 1029, 1024)).stop_grad()
            neg_source = jt.randn((1, 1029, 1024)).stop_grad()
            with mock.patch.dict(os.environ, {
                    "JITTOR_TRELLIS_CROSS_KV_CACHE": "1",
                    "JITTOR_TRELLIS_CROSS_KV_CACHE_MB": "1",
            }, clear=False):
                sampler.sample(model, None, source, neg_source)
            self.assertEqual(received_dtypes[-1], ("float32", "float32"))
            self.assertEqual(len(projection_calls), 4)
            self.assertNotIn("forward", attention.to_kv.__dict__)

            with mock.patch.dict(os.environ, {
                    "JITTOR_TRELLIS_CROSS_KV_CACHE": "1",
                    "JITTOR_TRELLIS_CROSS_KV_CACHE_MB": "384",
            }, clear=False):
                result = sampler.sample(model, None, source, neg_source)
                self.assertEqual(received_dtypes[-1], ("bfloat16", "bfloat16"))
                self.assertIs(result[1][0], result[1][1])
                self.assertIs(result[1][2], result[1][3])
                self.assertEqual(len(projection_calls), 6)
                self.assertIsNone(trellis_runtime._CROSS_KV_CACHE_SCOPE.get())
                self.assertNotIn("forward", attention.to_kv.__dict__)

                sampler.sample(model, None, source, neg_source)
                self.assertEqual(len(projection_calls), 8)

                attention.to_kv.weight = attention.to_kv.weight.float32().stop_grad()
                sampler.sample(model, None, source, neg_source)
                self.assertEqual(received_dtypes[-1], ("float32", "float32"))
                self.assertEqual(len(projection_calls), 12)
                self.assertNotIn("forward", attention.to_kv.__dict__)
                attention.to_kv.weight = attention.to_kv.weight.bfloat16().stop_grad()

                sampler.replace_weight = True
                result = sampler.sample(model, None, source, neg_source)
                self.assertIsNot(result[1][0], result[1][1])
                self.assertEqual(len(projection_calls), 16)
                sampler.replace_weight = False

                sampler.fail = True
                with self.assertRaisesRegex(RuntimeError, "expected test failure"):
                    sampler.sample(model, None, source, neg_source)
                self.assertIsNone(trellis_runtime._CROSS_KV_CACHE_SCOPE.get())
                self.assertNotIn("forward", attention.to_kv.__dict__)
                sampler.fail = False
                self.assertEqual(len(projection_calls), 18)

            with mock.patch.dict(os.environ, {
                    "JITTOR_TRELLIS_CROSS_KV_CACHE": "0",
            }, clear=False):
                sampler.sample(model, None, source, neg_source)
            self.assertEqual(received_dtypes[-1], ("float32", "float32"))
            self.assertEqual(len(projection_calls), 22)

            model.training = True
            with mock.patch.dict(os.environ, {
                    "JITTOR_TRELLIS_CROSS_KV_CACHE": "1",
            }, clear=False):
                sampler.sample(model, None, source, neg_source)
            self.assertEqual(received_dtypes[-1], ("float32", "float32"))
            self.assertEqual(len(projection_calls), 26)

    def test_sdpa_causal(self):
        q, k, v = self.q, self.k, self.v
        seq = q.shape[-2]
        causal = np.triu(np.full((seq, seq), -np.inf, dtype="float32"), 1)
        def body(dev):
            out = torch.nn.functional.scaled_dot_product_attention(
                jt.array(q), jt.array(k), jt.array(v), is_causal=True).numpy()
            self.ac(out, _sdpa_ref(q, k, v, causal), msg=f"sdpa causal {dev}")
        both_devices(body)

    def test_sdpa_gqa_math_fallback(self):
        rng = np.random.RandomState(71)
        q = rng.randn(1, 4, 3, 8).astype("float32")
        k = rng.randn(1, 2, 5, 8).astype("float32")
        v = rng.randn(1, 1, 5, 8).astype("float32")
        repeated_k = np.repeat(k, 2, axis=1)
        repeated_v = np.repeat(v, 4, axis=1)

        def body(dev):
            out = torch.nn.functional.scaled_dot_product_attention(
                jt.array(q), jt.array(k), jt.array(v), enable_gqa=True).numpy()
            self.ac(out, _sdpa_ref(q, repeated_k, repeated_v),
                    msg=f"sdpa gqa fallback {dev}")

        both_devices(body)

    def test_sdpa_dropout_one(self):
        q, k, v = self.q, self.k, self.v

        def body(dev):
            out = torch.nn.functional.scaled_dot_product_attention(
                jt.array(q), jt.array(k), jt.array(v), dropout_p=1.0).numpy()
            self.ac(out, np.zeros_like(q), atol=0.0, rtol=0.0,
                    msg=f"sdpa dropout p=1 {dev}")

        both_devices(body)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    def test_sdpa_fp16_training_fallback_mask_and_causal(self):
        rng = np.random.RandomState(53)
        q = rng.randn(2, 4, 8, 16).astype("float32")
        k = rng.randn(2, 4, 8, 16).astype("float32")
        v = rng.randn(2, 4, 8, 16).astype("float32")
        keep = np.tril(np.ones((8, 8), dtype=bool))
        additive = np.where(keep, 0.0, -np.inf).astype("float32")

        with jt.flag_scope(use_cuda=1):
            for name, kwargs, ref_mask in (
                    ("causal", {"is_causal": True}, additive),
                    ("bool_mask", {"attn_mask": jt.array(keep)}, additive)):
                qv = jt.array(q).float16()
                kv = jt.array(k).float16()
                vv = jt.array(v).float16()
                out = torch.nn.functional.scaled_dot_product_attention(
                    qv, kv, vv, **kwargs)
                grads = jt.grad(out.float32().sum(), [qv, kv, vv])
                fetched = jt.fetch_sync([out.float32()] + [grad.float32() for grad in grads])
                got_out, got_grads = fetched[0], fetched[1:]
                self.assertEqual(str(out.dtype), "float16")
                self.ac(got_out, _sdpa_ref(q, k, v, ref_mask),
                        atol=3e-3, rtol=3e-3, msg="fp16 fallback " + name)
                for tensor_name, got_grad in zip(("q", "k", "v"), got_grads):
                    self.assertTrue(np.isfinite(got_grad).all(),
                                    name + " " + tensor_name + " grad")

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    def test_sdpa_short_square_inference_prefers_math(self):
        rng = np.random.RandomState(97)
        q = rng.randn(1, 12, 50, 64).astype("float32")
        k = rng.randn(1, 12, 50, 64).astype("float32")
        v = rng.randn(1, 12, 50, 64).astype("float32")
        old_inference = os.environ.get("JITTOR_TORCH_INFERENCE")
        os.environ["JITTOR_TORCH_INFERENCE"] = "1"
        try:
            with jt.flag_scope(use_cuda=1), jt.no_grad():
                if hasattr(jt, "_torch_sdpa_flash_stats"):
                    delattr(jt, "_torch_sdpa_flash_stats")
                out = torch.nn.functional.scaled_dot_product_attention(
                    jt.array(q).float16(), jt.array(k).float16(),
                    jt.array(v).float16())
                got = out.float32().numpy()
                stats = getattr(jt, "_torch_sdpa_flash_stats", {})
        finally:
            if old_inference is None:
                os.environ.pop("JITTOR_TORCH_INFERENCE", None)
            else:
                os.environ["JITTOR_TORCH_INFERENCE"] = old_inference
        self.assertEqual(stats.get("hits", 0), 0)
        self.assertGreaterEqual(
            stats.get("misses", {}).get("short_square_math", 0), 1)
        self.ac(got, _sdpa_ref(q, k, v), atol=3e-3, rtol=3e-3,
                msg="short square inference math SDPA")

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    def test_required_flash_backend_returning_none_raises(self):
        from jittor.torch_shim import flashattn_jittor

        q = jt.ones((1, 2, 4, 32), dtype="float16")
        backend = ModuleType("required_flash_backend")
        backend.flash_attn_func = lambda *args, **kwargs: None
        with jt.flag_scope(use_cuda=1), jt.no_grad(), \
                mock.patch.object(flashattn_jittor, "load_backend_for",
                                  return_value=(backend, None)), \
                mock.patch.object(flashattn_jittor, "required", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "returned no output"):
                torch.nn.functional.scaled_dot_product_attention(q, q, q)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    def test_native_flash_receives_compact_gqa_kv_heads(self):
        from jittor.torch_shim import flashattn_jittor

        backend = ModuleType("gqa_flash_backend")
        seen = {}

        def fake_flash(q, k, v, *args, **kwargs):
            seen["shapes"] = (tuple(q.shape), tuple(k.shape), tuple(v.shape))
            return jt.zeros(q.shape, dtype=q.dtype)

        backend.flash_attn_func = fake_flash

        with jt.flag_scope(use_cuda=1), jt.no_grad(), \
                mock.patch.object(flashattn_jittor, "load_backend_for",
                                  return_value=(backend, None)), \
                mock.patch.object(flashattn_jittor, "required", return_value=False), \
                mock.patch.object(flashattn_jittor, "backend_name",
                                  return_value="gqa_flash_backend"):
            q = jt.ones((1, 4, 2, 32), dtype="float16")
            k = jt.ones((1, 2, 3, 32), dtype="float16")
            v = jt.ones((1, 2, 3, 32), dtype="float16")
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, enable_gqa=True)

        self.assertEqual(tuple(out.shape), (1, 4, 2, 32))
        self.assertEqual(seen["shapes"], (
            (1, 2, 4, 32),
            (1, 3, 2, 32),
            (1, 3, 2, 32),
        ))

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    def test_native_flash_rejects_head_mismatch_without_gqa(self):
        from jittor.torch_shim import flashattn_jittor

        backend = ModuleType("unexpected_gqa_backend")
        backend.flash_attn_func = mock.Mock()
        with jt.flag_scope(use_cuda=1), jt.no_grad(), \
                mock.patch.object(flashattn_jittor, "load_backend_for",
                                  return_value=(backend, None)) as loader:
            q = jt.ones((1, 4, 2, 32), dtype="float16")
            k = jt.ones((1, 2, 3, 32), dtype="float16")
            v = jt.ones((1, 2, 3, 32), dtype="float16")
            with self.assertRaises(Exception):
                torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, enable_gqa=False).sync()
        loader.assert_not_called()
        backend.flash_attn_func.assert_not_called()

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    def test_static_inference_reuses_capability_checked_backend(self):
        from jittor.torch_shim import flashattn_jittor

        backend = ModuleType("cached_flash_backend")
        replacement = ModuleType("replacement_flash_backend")
        backend.flash_attn_func = lambda q, k, v, *args, **kwargs: jt.zeros(
            q.shape, dtype=q.dtype)
        replacement.flash_attn_func = backend.flash_attn_func
        cache = torch._torch_sdpa_flash_backend_cache
        cache.clear()
        backends = iter(((backend, None), (replacement, None),
                         (replacement, None), (replacement, None)))

        def load_backend(*args, **kwargs):
            # Real first-use loading publishes its required compile capability
            # through watched environment variables. The cache must save the
            # post-load token or the immediate next layer would miss.
            if not os.environ.get("JITTOR_FLASH_ATTN_HEAD_DIMS"):
                os.environ["JITTOR_FLASH_ATTN_HEAD_DIMS"] = "32"
            return next(backends)

        try:
            with jt.flag_scope(use_cuda=1), jt.no_grad(), \
                    mock.patch.dict(os.environ, {
                        "JITTOR_TORCH_INFERENCE": "1",
                        "JITTOR_FLASH_ATTN_HEAD_DIMS": "",
                        "JITTOR_FLASH_ATTN_JITTOR_SRC": "/cache-source-a",
                    }, clear=False), \
                    mock.patch.object(flashattn_jittor,
                                      "_BACKEND_LOAD_GENERATION", 37), \
                    mock.patch.object(flashattn_jittor, "load_backend_for",
                                      side_effect=load_backend) as loader, \
                    mock.patch.object(
                        flashattn_jittor, "backend_publication_token",
                        side_effect=lambda backend: flashattn_jittor.backend_cache_token()), \
                    mock.patch.object(flashattn_jittor, "backend_name",
                                      return_value="cached_flash_backend"):
                q = jt.ones((1, 2, 1, 32), dtype="float16")
                torch.nn.functional.scaled_dot_product_attention(q, q, q)
                torch.nn.functional.scaled_dot_product_attention(q, q, q)
                self.assertEqual(loader.call_count, 1)
                os.environ["JITTOR_FLASH_ATTN_JITTOR_SRC"] = "/replacement"
                torch.nn.functional.scaled_dot_product_attention(q, q, q)
                self.assertEqual(loader.call_count, 2)
                del os.environ["JITTOR_FLASH_ATTN_JITTOR_SRC"]
                torch.nn.functional.scaled_dot_product_attention(q, q, q)
                self.assertEqual(loader.call_count, 3)
                flashattn_jittor._BACKEND_LOAD_GENERATION = 38
                torch.nn.functional.scaled_dot_product_attention(q, q, q)
                self.assertEqual(loader.call_count, 4)
        finally:
            cache.clear()

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    def test_static_inference_does_not_cache_backend_across_env_race(self):
        from jittor.torch_shim import flashattn_jittor

        stale = ModuleType("stale_flash_backend")
        current = ModuleType("current_flash_backend")
        stale.flash_attn_func = lambda q, k, v, *args, **kwargs: jt.zeros(
            q.shape, dtype=q.dtype)
        current.flash_attn_func = stale.flash_attn_func
        old_token = (1, 9, 21)
        new_token = (1, 9, 22)
        tokens = iter((old_token, new_token, new_token, new_token, new_token))
        publications = iter((old_token, new_token))
        cache = torch._torch_sdpa_flash_backend_cache
        cache.clear()
        try:
            with jt.flag_scope(use_cuda=1), jt.no_grad(), \
                    mock.patch.dict(os.environ, {
                        "JITTOR_TORCH_INFERENCE": "1",
                    }, clear=False), \
                    mock.patch.object(flashattn_jittor, "backend_cache_token",
                                      side_effect=lambda: next(tokens)), \
                    mock.patch.object(flashattn_jittor, "load_backend_for",
                                      side_effect=((stale, None),
                                                   (current, None))) as loader, \
                    mock.patch.object(flashattn_jittor,
                                      "backend_publication_token",
                                      side_effect=lambda backend: next(publications)), \
                    mock.patch.object(flashattn_jittor, "backend_name",
                                      return_value="race_flash_backend"):
                q = jt.ones((1, 2, 1, 32), dtype="float16")
                torch.nn.functional.scaled_dot_product_attention(q, q, q)
                self.assertEqual(cache, {})
                torch.nn.functional.scaled_dot_product_attention(q, q, q)
                torch.nn.functional.scaled_dot_product_attention(q, q, q)
                self.assertEqual(loader.call_count, 2)
                self.assertIs(cache[(32, "float16")][1], current)
        finally:
            cache.clear()

    def test_flash_backend_environment_epoch_tracks_watched_mutations(self):
        from jittor.torch_shim import flashattn_jittor

        epoch = flashattn_jittor.backend_environment_epoch()
        if epoch is None:
            self.skipTest("Python audit hooks are unavailable")
        name = "JITTOR_FLASH_ATTN_JITTOR_SRC"
        old_value = os.environ.get(name)
        unrelated_name = "JITTOR_FLASH_ATTN_EPOCH_UNWATCHED_TEST"
        old_unrelated = os.environ.get(unrelated_name)
        try:
            os.environ[unrelated_name] = "unchanged-token"
            after_unrelated = flashattn_jittor.backend_environment_epoch()
            os.environ[name] = "/epoch-test-a"
            after_set = flashattn_jittor.backend_environment_epoch()
            os.environ[name] = "/epoch-test-b"
            after_replace = flashattn_jittor.backend_environment_epoch()
            del os.environ[name]
            after_delete = flashattn_jittor.backend_environment_epoch()
            if os.supports_bytes_environ:
                os.environb[os.fsencode(name)] = b"/epoch-test-bytes"
                after_bytes = flashattn_jittor.backend_environment_epoch()
                del os.environb[os.fsencode(name)]
            else:
                after_bytes = after_delete
        finally:
            if old_value is not None:
                os.environ[name] = old_value
            else:
                os.environ.pop(name, None)
            if old_unrelated is not None:
                os.environ[unrelated_name] = old_unrelated
            else:
                os.environ.pop(unrelated_name, None)

        self.assertEqual(after_unrelated, epoch)
        self.assertGreater(after_set, epoch)
        self.assertGreater(after_replace, after_set)
        self.assertGreater(after_delete, after_replace)
        if os.supports_bytes_environ:
            self.assertGreater(after_bytes, after_delete)

    def test_flash_backend_environment_epoch_hook_is_idempotent(self):
        from jittor.torch_shim import flashattn_jittor

        state = flashattn_jittor._BACKEND_ENV_EPOCH_STATE
        if state is None:
            self.skipTest("Python audit hooks are unavailable")
        self.assertIs(flashattn_jittor._install_backend_environment_epoch_hook(),
                      state)
        self.assertIs(flashattn_jittor._install_backend_environment_epoch_hook(),
                      state)
        before = flashattn_jittor.backend_environment_epoch()
        name = "JITTOR_FLASH_ATTN_DIRECT_ADAPTER"
        old_value = os.environ.get(name)
        try:
            os.environ[name] = "epoch-idempotency-test"
            after = flashattn_jittor.backend_environment_epoch()
        finally:
            if old_value is not None:
                os.environ[name] = old_value
            else:
                os.environ.pop(name, None)
        self.assertEqual(after, before + 1)

    def test_flash_backend_environment_epoch_survives_module_reload(self):
        from jittor.torch_shim import flashattn_jittor

        script = r'''
import os
import pathlib
import sys
import types

path = pathlib.Path(sys.argv[1])
code = compile(path.read_text(encoding="utf-8"), os.fspath(path), "exec")
module = types.ModuleType("_flashattn_jittor_reload_test")
module.__file__ = os.fspath(path)
exec(code, module.__dict__)
state = module._BACKEND_ENV_EPOCH_STATE
old_token = module.backend_cache_token()
assert state is not None and old_token is not None
exec(code, module.__dict__)
new_token = module.backend_cache_token()
assert module._BACKEND_ENV_EPOCH_STATE is state
assert new_token[0] > old_token[0]
name = "JITTOR_FLASH_ATTN_FUSED_PACKED_SPLIT"
before = module.backend_environment_epoch()
os.environ[name] = "reload-idempotency-test"
after = module.backend_environment_epoch()
assert after == before + 1, (before, after)
'''
        completed = subprocess.run(
            [sys.executable, "-c", script, flashattn_jittor.__file__],
            text=True, capture_output=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_flash_backend_environment_epoch_detects_silent_hook_rejection(self):
        from jittor.torch_shim import flashattn_jittor

        attr = flashattn_jittor._BACKEND_ENV_EPOCH_STATE_ATTR
        original = getattr(sys, attr)
        delattr(sys, attr)
        try:
            with mock.patch.object(sys, "addaudithook", return_value=None), \
                    mock.patch.object(sys, "audit", wraps=sys.audit) as probe:
                state = flashattn_jittor._install_backend_environment_epoch_hook()
                self.assertIsNone(state)
                self.assertFalse(getattr(sys, attr)["active"])
                probe.assert_called_once_with(
                    flashattn_jittor._BACKEND_ENV_EPOCH_PROBE)
        finally:
            setattr(sys, attr, original)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    def test_static_inference_cache_disables_without_environment_epoch(self):
        from jittor.torch_shim import flashattn_jittor

        backend = ModuleType("uncached_flash_backend")
        backend.flash_attn_func = lambda q, k, v, *args, **kwargs: jt.zeros(
            q.shape, dtype=q.dtype)
        cache = torch._torch_sdpa_flash_backend_cache
        cache.clear()
        try:
            with jt.flag_scope(use_cuda=1), jt.no_grad(), \
                    mock.patch.dict(os.environ, {
                        "JITTOR_TORCH_INFERENCE": "1",
                    }, clear=False), \
                    mock.patch.object(flashattn_jittor,
                                      "backend_cache_token",
                                      return_value=None), \
                    mock.patch.object(flashattn_jittor, "load_backend_for",
                                      return_value=(backend, None)) as loader, \
                    mock.patch.object(flashattn_jittor,
                                      "backend_publication_token",
                                      return_value=None), \
                    mock.patch.object(flashattn_jittor, "backend_name",
                                      return_value="uncached_flash_backend"):
                q = jt.ones((1, 2, 1, 32), dtype="float16")
                torch.nn.functional.scaled_dot_product_attention(q, q, q)
                torch.nn.functional.scaled_dot_product_attention(q, q, q)
                self.assertEqual(loader.call_count, 2)
                self.assertEqual(cache, {})
        finally:
            cache.clear()

    def test_flash_backend_reloads_for_expanded_capability(self):
        from jittor.torch_shim import flashattn_jittor

        first = ModuleType("first_flash_backend")
        first._flashattn_jittor_official = True
        first._flashattn_jittor_head_dims = (32,)
        first._flashattn_jittor_dtypes = ("fp16",)
        expanded = ModuleType("expanded_flash_backend")
        expanded._flashattn_jittor_official = True
        expanded._flashattn_jittor_head_dims = (32, 64)
        expanded._flashattn_jittor_dtypes = ("fp16", "bf16")

        capability_env = {
            "JITTOR_FLASH_ATTN_HEAD_DIMS": "",
            "FLASH_ATTN_HEAD_DIMS": "",
            "JITTOR_FLASH_ATTN_DTYPES": "",
            "FLASH_ATTN_DTYPES": "",
        }
        with mock.patch.dict(os.environ, capability_env, clear=False), \
                mock.patch.object(
                    flashattn_jittor, "load_backend",
                    side_effect=(first, expanded)) as loader:
            backend, miss = flashattn_jittor.load_backend_for(64, "bfloat16")
        self.assertIs(backend, expanded)
        self.assertIsNone(miss)
        self.assertEqual(loader.call_args_list, [mock.call(), mock.call(force=True)])

    def test_flash_success_cache_invalidates_on_build_environment_change(self):
        from jittor.torch_shim import flashattn_jittor

        old = ModuleType("cached_flash_backend")
        replacement = ModuleType("replacement_flash_backend")
        old_env = (("JTCUDA", "/old/cuda"),)
        new_env = (("JTCUDA", "/new/cuda"),)
        cached_key = (old_env, ("/flash/source",))
        current_key = (new_env, ("/flash/source",))
        import_identities = []

        def reload_source(root):
            self.assertEqual(root, "/flash/source")
            import_identities.append(flashattn_jittor._official_import_identity(
                "official-forward", "/extensions/0123456789abcdef",
                "flash_attn_2_cuda_jittor"))
            return replacement

        with mock.patch.object(flashattn_jittor, "_BACKEND", old), \
                mock.patch.object(flashattn_jittor, "_BACKEND_NAME", "cached"), \
                mock.patch.object(flashattn_jittor, "_BACKEND_CONFIG_KEY", cached_key), \
                mock.patch.object(flashattn_jittor, "_BACKEND_LOAD_GENERATION", 11), \
                mock.patch.object(flashattn_jittor, "_LOADING", False), \
                mock.patch.object(flashattn_jittor, "enabled", return_value=True), \
                mock.patch.object(flashattn_jittor, "_backend_environment_key",
                                  return_value=new_env), \
                mock.patch.object(flashattn_jittor, "_backend_config_key",
                                  return_value=current_key), \
                mock.patch.object(flashattn_jittor, "explicit_source_roots",
                                  return_value=["/flash/source"]), \
                mock.patch.object(flashattn_jittor, "_load_from_source_root",
                                  side_effect=reload_source) as loader:
            backend = flashattn_jittor.load_backend()
            generation = flashattn_jittor._BACKEND_LOAD_GENERATION

        self.assertIs(backend, replacement)
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(generation, 12)
        self.assertEqual(import_identities, [
            flashattn_jittor._official_import_identity(
                "official-forward", "/extensions/0123456789abcdef",
                "flash_attn_2_cuda_jittor", generation=12)
        ])

    def test_flash_backend_does_not_publish_across_build_environment_race(self):
        from jittor.torch_shim import flashattn_jittor

        backend = ModuleType("raced_flash_backend")

        def load_source(root):
            self.assertEqual(root, "/race/source")
            os.environ["JTCUDA"] = "/new/cuda"
            return backend

        with mock.patch.dict(os.environ, {"JTCUDA": "/old/cuda"}, clear=False), \
                mock.patch.object(flashattn_jittor, "_BACKEND",
                                  flashattn_jittor._UNSET), \
                mock.patch.object(flashattn_jittor, "_BACKEND_NAME", "math"), \
                mock.patch.object(flashattn_jittor, "_BACKEND_CONFIG_KEY", None), \
                mock.patch.object(flashattn_jittor, "_BACKEND_LOAD_GENERATION", 5), \
                mock.patch.object(flashattn_jittor, "_BACKEND_PUBLICATION_TOKEN", None), \
                mock.patch.object(flashattn_jittor, "_LOADING", False), \
                mock.patch.object(flashattn_jittor, "enabled", return_value=True), \
                mock.patch.object(flashattn_jittor, "explicit_source_roots",
                                  return_value=["/race/source"]), \
                mock.patch.object(flashattn_jittor, "_load_from_source_root",
                                  side_effect=load_source), \
                mock.patch.object(flashattn_jittor, "_backend_config_key",
                                  return_value=(("config",), ("/race/source",))):
            result = flashattn_jittor.load_backend(force=True)
            publication = flashattn_jittor.backend_publication_token(result)
            config_key = flashattn_jittor._BACKEND_CONFIG_KEY

        self.assertIs(result, backend)
        self.assertIsNone(publication)
        self.assertIsNone(config_key)

    def test_flash_backend_refreshes_publication_after_same_value_env_write(self):
        from jittor.torch_shim import flashattn_jittor

        backend = ModuleType("stable_flash_backend")
        with mock.patch.dict(os.environ, {"JTCUDA": "/same/cuda"}, clear=False):
            environment_key = flashattn_jittor._backend_environment_key()
            os.environ["JTCUDA"] = "/same/cuda"
            with mock.patch.object(flashattn_jittor, "_BACKEND", backend), \
                    mock.patch.object(flashattn_jittor, "_BACKEND_NAME", "stable"), \
                    mock.patch.object(flashattn_jittor, "_BACKEND_CONFIG_KEY",
                                      (environment_key, ())), \
                    mock.patch.object(flashattn_jittor,
                                      "_BACKEND_LOAD_GENERATION", 7), \
                    mock.patch.object(flashattn_jittor,
                                      "_BACKEND_PUBLICATION_TOKEN", None), \
                    mock.patch.object(flashattn_jittor, "_LOADING", False), \
                    mock.patch.object(flashattn_jittor, "enabled", return_value=True):
                result = flashattn_jittor.load_backend()
                publication = flashattn_jittor.backend_publication_token(result)
                current = flashattn_jittor.backend_cache_token()

        self.assertIs(result, backend)
        self.assertEqual(publication, current)

    def test_flash_backend_loader_exception_is_not_published(self):
        from jittor.torch_shim import flashattn_jittor

        old_backend = ModuleType("old_flash_backend")
        replacement = ModuleType("replacement_flash_backend")
        with mock.patch.object(flashattn_jittor, "_BACKEND", old_backend), \
                mock.patch.object(flashattn_jittor, "_BACKEND_CONFIG_KEY", None), \
                mock.patch.object(flashattn_jittor,
                                  "_BACKEND_PUBLICATION_TOKEN", (1, 1, 1)), \
                mock.patch.object(flashattn_jittor, "_LOADING", False), \
                mock.patch.object(flashattn_jittor, "enabled", return_value=True), \
                mock.patch.object(flashattn_jittor, "explicit_source_roots",
                                  return_value=["/error/source"]), \
                mock.patch.object(flashattn_jittor, "_load_from_source_root",
                                  side_effect=(RuntimeError("loader failed"),
                                               replacement)), \
                mock.patch.object(flashattn_jittor, "_backend_config_key",
                                  return_value=(("stable",), ("/error/source",))):
            with self.assertRaisesRegex(RuntimeError, "loader failed"):
                flashattn_jittor.load_backend(force=True)
            self.assertIsNone(flashattn_jittor._BACKEND_CONFIG_KEY)
            self.assertIsNone(flashattn_jittor._BACKEND_PUBLICATION_TOKEN)
            self.assertFalse(flashattn_jittor._LOADING)
            result = flashattn_jittor.load_backend()
            self.assertIs(result, replacement)
            self.assertIsNotNone(
                flashattn_jittor.backend_publication_token(replacement))

    def test_flash_backend_disabled_race_does_not_cache_enabled_miss(self):
        from jittor.torch_shim import flashattn_jittor

        backend = ModuleType("reenabled_flash_backend")
        enabled_calls = 0

        def enabled():
            nonlocal enabled_calls
            enabled_calls += 1
            if enabled_calls == 1:
                os.environ["JITTOR_FLASH_ATTN_JITTOR"] = "1"
                return False
            return True

        with mock.patch.dict(os.environ, {
                    "JITTOR_FLASH_ATTN_JITTOR": "0",
                }, clear=False), \
                mock.patch.object(flashattn_jittor, "_BACKEND",
                                  flashattn_jittor._UNSET), \
                mock.patch.object(flashattn_jittor, "_BACKEND_CONFIG_KEY", None), \
                mock.patch.object(flashattn_jittor,
                                  "_BACKEND_PUBLICATION_TOKEN", None), \
                mock.patch.object(flashattn_jittor, "_LOADING", False), \
                mock.patch.object(flashattn_jittor, "enabled",
                                  side_effect=enabled), \
                mock.patch.object(flashattn_jittor, "explicit_source_roots",
                                  return_value=["/enabled/source"]), \
                mock.patch.object(flashattn_jittor, "_load_from_source_root",
                                  return_value=backend) as loader:
            self.assertIsNone(flashattn_jittor.load_backend())
            self.assertIsNone(flashattn_jittor._BACKEND_CONFIG_KEY)
            result = flashattn_jittor.load_backend()

        self.assertIs(result, backend)
        self.assertEqual(loader.call_count, 1)

    def test_flash_explicit_python_source_switches_module_root(self):
        from jittor.torch_shim import flashattn_jittor

        module_name = "flashattn_jittor"
        old_modules = {
            key: value for key, value in sys.modules.items()
            if key == module_name or key.startswith(module_name + ".")
        }
        old_path = list(sys.path)
        try:
            for key in old_modules:
                sys.modules.pop(key, None)
            with tempfile.TemporaryDirectory() as tmp:
                base = pathlib.Path(tmp)
                roots = []
                for label in ("A", "B"):
                    root = base / label
                    package = root / module_name
                    package.mkdir(parents=True)
                    (package / "__init__.py").write_text(
                        "ORIGIN = %r\n"
                        "def flash_attn_func(*args, **kwargs):\n"
                        "    return ORIGIN\n" % label,
                        encoding="utf-8",
                    )
                    roots.append(root)

                with mock.patch.object(flashattn_jittor, "_BACKEND",
                                       flashattn_jittor._UNSET), \
                        mock.patch.object(flashattn_jittor, "_BACKEND_NAME", "math"), \
                        mock.patch.object(flashattn_jittor, "_BACKEND_CONFIG_KEY", None), \
                        mock.patch.object(flashattn_jittor, "_BACKEND_LOAD_GENERATION", 0), \
                        mock.patch.object(flashattn_jittor, "_LOADING", False), \
                        mock.patch.dict(os.environ, {
                            "JITTOR_FLASH_ATTN_JITTOR_SRC": os.fspath(roots[0]),
                        }, clear=False):
                    first = flashattn_jittor.load_backend(force=True)
                    os.environ["JITTOR_FLASH_ATTN_JITTOR_SRC"] = os.fspath(roots[1])
                    second = flashattn_jittor.load_backend()

                self.assertEqual(first.ORIGIN, "A")
                self.assertEqual(second.ORIGIN, "B")
                self.assertIsNot(first, second)
                self.assertEqual(
                    os.path.commonpath((
                        os.fspath(pathlib.Path(second.__file__).resolve()),
                        os.fspath(roots[1].resolve()),
                    )),
                    os.fspath(roots[1].resolve()),
                )
        finally:
            for key in list(sys.modules):
                if key == module_name or key.startswith(module_name + "."):
                    sys.modules.pop(key, None)
            sys.modules.update(old_modules)
            sys.path[:] = old_path

    def test_flash_backend_load_is_single_flight(self):
        from jittor.torch_shim import flashattn_jittor

        backend = ModuleType("single_flight_backend")
        backend.flash_attn_func = lambda *args, **kwargs: "ok"
        entered = threading.Event()
        release = threading.Event()
        second_started = threading.Event()
        results = []
        errors = []

        def slow_load(root):
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test loader timed out")
            return backend

        def run(force, started=None):
            if started is not None:
                started.set()
            try:
                results.append(flashattn_jittor.load_backend(force=force))
            except Exception as exc:
                errors.append(exc)

        env_key = (("test", "stable"),)
        config_key = (env_key, ("/single/source",))
        with mock.patch.object(flashattn_jittor, "_BACKEND",
                               flashattn_jittor._UNSET), \
                mock.patch.object(flashattn_jittor, "_BACKEND_NAME", "math"), \
                mock.patch.object(flashattn_jittor, "_BACKEND_CONFIG_KEY", None), \
                mock.patch.object(flashattn_jittor, "_BACKEND_LOAD_GENERATION", 0), \
                mock.patch.object(flashattn_jittor, "_LOADING", False), \
                mock.patch.object(flashattn_jittor, "enabled", return_value=True), \
                mock.patch.object(flashattn_jittor, "_backend_environment_key",
                                  return_value=env_key), \
                mock.patch.object(flashattn_jittor, "_backend_config_key",
                                  return_value=config_key), \
                mock.patch.object(flashattn_jittor, "explicit_source_roots",
                                  return_value=["/single/source"]), \
                mock.patch.object(flashattn_jittor, "_load_from_source_root",
                                  side_effect=slow_load) as loader:
            first = threading.Thread(target=run, args=(True,))
            second = threading.Thread(target=run, args=(False, second_started))
            first.start()
            self.assertTrue(entered.wait(timeout=2))
            second.start()
            self.assertTrue(second_started.wait(timeout=2))
            second.join(timeout=0.05)
            self.assertTrue(second.is_alive(), "second loader must wait")
            release.set()
            first.join(timeout=5)
            second.join(timeout=5)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(item is backend for item in results))

    def test_flash_capability_expansion_is_one_locked_transaction(self):
        from jittor.torch_shim import flashattn_jittor

        entered = threading.Event()
        release = threading.Event()
        second_started = threading.Event()
        snapshots = []
        results = []
        errors = []

        def parse(name):
            return tuple(part for part in os.environ.get(name, "").split(",")
                         if part)

        def slow_load(force=False):
            dims = tuple(int(part) for part in parse(
                "JITTOR_FLASH_ATTN_HEAD_DIMS"))
            dtypes = parse("JITTOR_FLASH_ATTN_DTYPES")
            snapshots.append((dims, dtypes, force))
            backend = ModuleType("capability_" + "_".join(map(str, dims)))
            backend._flashattn_jittor_official = True
            backend._flashattn_jittor_head_dims = dims
            backend._flashattn_jittor_dtypes = dtypes
            if len(snapshots) == 1:
                entered.set()
                if not release.wait(timeout=5):
                    raise RuntimeError("test capability loader timed out")
            return backend

        def run(dim, dtype, started=None):
            if started is not None:
                started.set()
            try:
                results.append(flashattn_jittor.load_backend_for(dim, dtype))
            except Exception as exc:
                errors.append(exc)

        capability_env = {
            "JITTOR_FLASH_ATTN_HEAD_DIMS": "",
            "FLASH_ATTN_HEAD_DIMS": "",
            "JITTOR_FLASH_ATTN_DTYPES": "",
            "FLASH_ATTN_DTYPES": "",
        }
        with mock.patch.dict(os.environ, capability_env, clear=False), \
                mock.patch.object(flashattn_jittor, "load_backend",
                                  side_effect=slow_load):
            first = threading.Thread(target=run, args=(32, "float16"))
            second = threading.Thread(
                target=run, args=(64, "bfloat16", second_started))
            first.start()
            self.assertTrue(entered.wait(timeout=2))
            second.start()
            self.assertTrue(second_started.wait(timeout=2))
            second.join(timeout=0.05)
            self.assertTrue(second.is_alive(),
                            "second capability request must wait for the build")
            self.assertEqual(os.environ["JITTOR_FLASH_ATTN_HEAD_DIMS"], "32")
            self.assertEqual(os.environ["JITTOR_FLASH_ATTN_DTYPES"], "fp16")
            release.set()
            first.join(timeout=5)
            second.join(timeout=5)
            final_dims = set(parse("JITTOR_FLASH_ATTN_HEAD_DIMS"))
            final_dtypes = set(parse("JITTOR_FLASH_ATTN_DTYPES"))

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.assertTrue(all(miss is None for _, miss in results))
        self.assertEqual(final_dims, {"32", "64"})
        self.assertEqual(final_dtypes, {"fp16", "bf16"})
        self.assertEqual(snapshots[0][:2], ((32,), ("fp16",)))
        self.assertEqual(snapshots[1][:2], ((32, 64), ("fp16", "bf16")))

    def test_flash_stub_function_cache_tracks_backend_instance(self):
        import flash_attn

        first = ModuleType("first_flash_backend")
        second = ModuleType("second_flash_backend")
        first.flash_attn_func = lambda *args, **kwargs: "first"
        second.flash_attn_func = lambda *args, **kwargs: "second"
        state = {"backend": first}
        loader = ModuleType("fake_flash_loader")
        loader.load_backend = lambda: state["backend"]
        loader.backend_name = lambda: state["backend"].__name__
        loader.required = lambda: False
        loader.last_error = lambda: None

        cache = flash_attn._NATIVE_FUNCTION_CACHE
        old_cache = dict(cache)
        try:
            cache.clear()
            with mock.patch.object(flash_attn, "_flashattn_jittor", loader):
                self.assertIs(
                    flash_attn._native_function("flash_attn_func"),
                    first.flash_attn_func)
                self.assertIs(
                    flash_attn._native_function("flash_attn_func"),
                    first.flash_attn_func)
                state["backend"] = second
                self.assertIs(
                    flash_attn._native_function("flash_attn_func"),
                    second.flash_attn_func)
                replacement = lambda *args, **kwargs: "replacement"
                second.flash_attn_func = replacement
                self.assertIs(
                    flash_attn._native_function("flash_attn_func"),
                    replacement)
        finally:
            cache.clear()
            cache.update(old_cache)

    def test_flash_stub_required_rejects_native_none(self):
        import flash_attn

        backend = ModuleType("required_none_backend")
        backend.flash_attn_func = lambda *args, **kwargs: None
        loader = ModuleType("required_none_loader")
        loader.load_backend = lambda: backend
        loader.backend_name = lambda: backend.__name__
        loader.required = lambda: True
        loader.last_error = lambda: None

        cache = flash_attn._NATIVE_FUNCTION_CACHE
        old_cache = dict(cache)
        try:
            cache.clear()
            q = jt.ones((1, 2, 1, 8))
            with mock.patch.object(flash_attn, "_flashattn_jittor", loader):
                with self.assertRaisesRegex(RuntimeError, "returned no output"):
                    flash_attn.flash_attn_func(q, q, q)
        finally:
            cache.clear()
            cache.update(old_cache)

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
    def test_sdpa_native_flash_attn_gqa_fp16_cuda(self):
        rng = np.random.RandomState(73)
        q = rng.randn(1, 4, 3, 32).astype("float32")
        k = rng.randn(1, 2, 5, 32).astype("float32")
        v = rng.randn(1, 2, 5, 32).astype("float32")
        with jt.flag_scope(use_cuda=1), jt.no_grad():
            if hasattr(jt, "_torch_sdpa_flash_stats"):
                delattr(jt, "_torch_sdpa_flash_stats")
            torch._torch_sdpa_flash_backend_cache.clear()
            out = torch.nn.functional.scaled_dot_product_attention(
                jt.array(q).float16(), jt.array(k).float16(),
                jt.array(v).float16(), enable_gqa=True)
            got = out.float32().numpy()
            stats = getattr(jt, "_torch_sdpa_flash_stats", {})
        expected = _sdpa_ref(
            q, np.repeat(k, 2, axis=1), np.repeat(v, 2, axis=1))
        self.assertGreaterEqual(stats.get("hits", 0), 1)
        self.assertEqual(stats.get("misses", {}), {})
        self.ac(got, expected, atol=3e-3, rtol=3e-3,
                msg="sdpa native flash gqa fp16 cuda")

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

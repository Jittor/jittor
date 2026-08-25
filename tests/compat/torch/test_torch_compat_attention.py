"""Torch-grade attention/transformer parity for ``import jittor as torch``.

The transformer surface is the core of the jittor-as-torch project. Compares
F.scaled_dot_product_attention and nn.MultiheadAttention against explicit numpy references.
CPU+CUDA.

Run:  python -m pytest tests/compat/torch/test_torch_compat_attention.py
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


def _sdpa_backward_ref(q, k, v, grad_out, mask=None):
    scale = q.shape[-1] ** -0.5
    scores = (q @ np.swapaxes(k, -1, -2)) * scale
    if mask is not None:
        scores = scores + mask
    probability = _softmax(scores, -1)
    grad_v = np.swapaxes(probability, -1, -2) @ grad_out
    grad_probability = grad_out @ np.swapaxes(v, -1, -2)
    grad_score = probability * (
        grad_probability
        - (grad_probability * probability).sum(axis=-1, keepdims=True))
    grad_q = (grad_score @ k) * scale
    grad_k = (np.swapaxes(grad_score, -1, -2) @ q) * scale
    return grad_q, grad_k, grad_v


def _native_flash_dtype_enabled(dtype):
    raw = (os.environ.get("JITTOR_FLASH_ATTN_DTYPES")
           or os.environ.get("FLASH_ATTN_DTYPES") or "")
    values = {item.strip().lower() for item in raw.replace(";", ",").split(",")}
    if values & {"all", "full", "*"}:
        return True
    aliases = {"bf16", "bfloat16"} if dtype == "bfloat16" else {"fp16", "float16"}
    return bool(values & aliases)


def _native_flash_head_dim_enabled(head_dim):
    raw = (os.environ.get("JITTOR_FLASH_ATTN_HEAD_DIMS")
           or os.environ.get("FLASH_ATTN_HEAD_DIMS") or "")
    values = {item.strip().lower() for item in raw.replace(";", ",").split(",")}
    return bool(values & {"all", "full", "*", str(int(head_dim))})


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
    def test_sdpa_short_training_prefers_math(self):
        from jittor.compat.shim.backends import flash_attention as flashattn_jittor

        rng = np.random.RandomState(167)
        q = rng.randn(2, 4, 32, 64).astype("float32")
        k = rng.randn(2, 4, 32, 64).astype("float32")
        v = rng.randn(2, 4, 32, 64).astype("float32")
        grad_out = rng.randn(2, 4, 32, 64).astype("float32")
        env = {
            "JITTOR_FLASH_ATTN_JITTOR_REQUIRED": "0",
            "JITTOR_FLASH_ATTN_TRAINING_MIN_SCORES": str(1 << 24),
        }
        with jt.flag_scope(use_cuda=1), \
                mock.patch.dict(os.environ, env, clear=False), \
                mock.patch.object(
                    flashattn_jittor, "load_backend_for") as loader:
            if hasattr(jt, "_torch_sdpa_flash_stats"):
                delattr(jt, "_torch_sdpa_flash_stats")
            qv, kv, vv = (
                jt.array(value).float16() for value in (q, k, v))
            out = torch.nn.functional.scaled_dot_product_attention(qv, kv, vv)
            grads = jt.grad(
                (out.float32() * jt.array(grad_out)).sum(), [qv, kv, vv])
            fetched = jt.fetch_sync(
                [out.float32()] + [grad.float32() for grad in grads])
            stats = getattr(jt, "_torch_sdpa_flash_stats", {})

        loader.assert_not_called()
        self.assertEqual(stats.get("hits", 0), 0)
        self.assertEqual(
            stats.get("misses", {}).get("short_training_math"), 1)
        self.ac(fetched[0], _sdpa_ref(q, k, v), atol=3e-3, rtol=3e-3,
                msg="short training math output")
        for name, got, expected in zip(
                ("q", "k", "v"), fetched[1:],
                _sdpa_backward_ref(q, k, v, grad_out)):
            self.ac(got, expected, atol=6e-3, rtol=6e-3,
                    msg="short training math %s gradient" % name)

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
        from jittor.compat.shim.backends import flash_attention as flashattn_jittor

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
        from jittor.compat.shim.backends import flash_attention as flashattn_jittor

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
        from jittor.compat.shim.backends import flash_attention as flashattn_jittor

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
        from jittor.compat.shim.backends import flash_attention as flashattn_jittor

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
        from jittor.compat.shim.backends import flash_attention as flashattn_jittor

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
        from jittor.compat.shim.backends import flash_attention as flashattn_jittor

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
        from jittor.compat.shim.backends import flash_attention as flashattn_jittor

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
        from jittor.compat.shim.backends import flash_attention as flashattn_jittor

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
        from jittor.compat.shim.backends import flash_attention as flashattn_jittor

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
        from jittor.compat.shim.backends import flash_attention as flashattn_jittor

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
        from jittor.compat.shim.backends import flash_attention as flashattn_jittor

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

    def test_flash_official_dropout_backward_capability(self):
        from jittor.compat.shim.backends import flash_attention as flashattn_jittor

        supported = flashattn_jittor._official_dropout_backward_supported
        self.assertTrue(supported(192, ()))
        self.assertTrue(supported(256, (80,)))
        self.assertTrue(supported(256, (90,)))
        self.assertTrue(supported(256, (80, 90)))
        self.assertFalse(supported(256, (86,)))
        self.assertFalse(supported(256, (89,)))
        self.assertFalse(supported(256, ()))
        self.assertFalse(supported(256, ("unknown",)))

    def test_flash_success_cache_invalidates_on_build_environment_change(self):
        from jittor.compat.shim.backends import flash_attention as flashattn_jittor

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
        from jittor.compat.shim.backends import flash_attention as flashattn_jittor

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
        from jittor.compat.shim.backends import flash_attention as flashattn_jittor

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
        from jittor.compat.shim.backends import flash_attention as flashattn_jittor

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
        from jittor.compat.shim.backends import flash_attention as flashattn_jittor

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
        from jittor.compat.shim.backends import flash_attention as flashattn_jittor

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
        from jittor.compat.shim.backends import flash_attention as flashattn_jittor

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
        from jittor.compat.shim.backends import flash_attention as flashattn_jittor

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

    def test_flash_stub_comes_from_the_bundled_package(self):
        import flash_attn

        expected = (
            pathlib.Path(jt.__file__).resolve().parent
            / "compat" / "shim" / "resources" / "stubs"
            / "flash_attn" / "__init__.py"
        )
        self.assertEqual(pathlib.Path(flash_attn.__file__).resolve(), expected)

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
    def test_sdpa_cuda_routes_masked_rows_through_safe_softmax(self):
        from jittor.nn.backends import softmax_cuda

        calls = []
        original = softmax_cuda.softmax_v1

        def traced(value, log=False, zero_all_neg_inf=False):
            calls.append(bool(zero_all_neg_inf))
            return original(value, log, zero_all_neg_inf)

        q = jt.ones((1, 2, 4, 8), dtype="float32")
        keep = jt.ones((4, 4), dtype="bool")
        keep[2, :] = False
        with jt.flag_scope(use_cuda=1), mock.patch.object(
                softmax_cuda, "softmax_v1", side_effect=traced):
            masked = torch.nn.functional.scaled_dot_product_attention(
                q, q, q, attn_mask=keep)
            masked.sync()
            self.assertEqual(calls, [True])

            calls.clear()
            causal = torch.nn.functional.scaled_dot_product_attention(
                q, q, q, is_causal=True)
            causal.sync()
            self.assertEqual(calls, [False])

        self.assertTrue(np.isfinite(masked.numpy()).all())
        self.assertTrue(np.isfinite(causal.numpy()).all())

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
    def test_sdpa_native_flash_attn_backward_fp16_cuda(self):
        rng = np.random.RandomState(101)
        q = rng.randn(1, 2, 8, 32).astype("float32")
        k = rng.randn(1, 2, 8, 32).astype("float32")
        v = rng.randn(1, 2, 8, 32).astype("float32")
        grad_out = rng.randn(1, 2, 8, 32).astype("float32")
        with jt.flag_scope(use_cuda=1):
            if hasattr(jt, "_torch_sdpa_flash_stats"):
                delattr(jt, "_torch_sdpa_flash_stats")
            qv, kv, vv = (
                jt.array(value).float16() for value in (q, k, v))
            out = torch.nn.functional.scaled_dot_product_attention(qv, kv, vv)
            grads = jt.grad(
                (out.float32() * jt.array(grad_out)).sum(), [qv, kv, vv])
            fetched = jt.fetch_sync(
                [out.float32()] + [grad.float32() for grad in grads])
            stats = getattr(jt, "_torch_sdpa_flash_stats", {})

        self.assertGreaterEqual(stats.get("hits", 0), 1)
        self.assertEqual(stats.get("misses", {}), {})
        self.ac(fetched[0], _sdpa_ref(q, k, v), atol=3e-3, rtol=3e-3,
                msg="sdpa native flash backward output")
        for name, got, expected in zip(
                ("q", "k", "v"), fetched[1:],
                _sdpa_backward_ref(q, k, v, grad_out)):
            self.ac(got, expected, atol=6e-3, rtol=6e-3,
                    msg="sdpa native flash %s gradient" % name)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("bfloat16"),
                         "native bf16 flash-attn capability not configured")
    def test_sdpa_native_flash_attn_backward_bf16_cuda(self):
        rng = np.random.RandomState(113)
        q = rng.randn(1, 2, 8, 32).astype("float32")
        k = rng.randn(1, 2, 8, 32).astype("float32")
        v = rng.randn(1, 2, 8, 32).astype("float32")
        grad_out = rng.randn(1, 2, 8, 32).astype("float32")
        with jt.flag_scope(use_cuda=1):
            if hasattr(jt, "_torch_sdpa_flash_stats"):
                delattr(jt, "_torch_sdpa_flash_stats")
            qv, kv, vv = (
                jt.array(value).bfloat16() for value in (q, k, v))
            out = torch.nn.functional.scaled_dot_product_attention(qv, kv, vv)
            grads = jt.grad(
                (out.float32() * jt.array(grad_out)).sum(), [qv, kv, vv])
            fetched = jt.fetch_sync(
                [out.float32()] + [grad.float32() for grad in grads])
            stats = getattr(jt, "_torch_sdpa_flash_stats", {})

        self.assertGreaterEqual(stats.get("hits", 0), 1)
        self.assertEqual(stats.get("misses", {}), {})
        self.ac(fetched[0], _sdpa_ref(q, k, v), atol=3e-2, rtol=3e-2,
                msg="sdpa native bf16 flash backward output")
        for name, got, expected in zip(
                ("q", "k", "v"), fetched[1:],
                _sdpa_backward_ref(q, k, v, grad_out)):
            self.ac(got, expected, atol=3e-2, rtol=3e-2,
                    msg="sdpa native bf16 flash %s gradient" % name)

    def _check_sdpa_native_flash_mask_fallback(self, dtype, out_tol,
                                                grad_tol):
        rng = np.random.RandomState(157)
        q = rng.randn(1, 2, 8, 32).astype("float32")
        k = rng.randn(1, 2, 8, 32).astype("float32")
        v = rng.randn(1, 2, 8, 32).astype("float32")
        grad_out = rng.randn(1, 2, 8, 32).astype("float32")
        keep = np.tril(np.ones((8, 8), dtype=bool))
        bool_bias = np.where(keep, 0.0, -np.inf).astype("float32")
        additive = (rng.randn(8, 8) * 0.125).astype("float32")

        with jt.flag_scope(use_cuda=1):
            for name, mask, reference_mask in (
                    ("bool", jt.array(keep), bool_bias),
                    ("additive", jt.array(additive), additive)):
                if hasattr(jt, "_torch_sdpa_flash_stats"):
                    delattr(jt, "_torch_sdpa_flash_stats")
                qv, kv, vv = (
                    jt.array(value).to(dtype) for value in (q, k, v))
                out = torch.nn.functional.scaled_dot_product_attention(
                    qv, kv, vv, attn_mask=mask)
                grads = jt.grad(
                    (out.float32() * jt.array(grad_out)).sum(), [qv, kv, vv])
                fetched = jt.fetch_sync(
                    [out.float32()] + [grad.float32() for grad in grads])
                stats = getattr(jt, "_torch_sdpa_flash_stats", {})

                self.assertEqual(stats.get("hits", 0), 0, name)
                self.assertEqual(stats.get("misses", {}), {"mask": 1}, name)
                self.assertIsNone(stats.get("backend"), name)
                self.ac(
                    fetched[0], _sdpa_ref(q, k, v, reference_mask),
                    atol=out_tol, rtol=out_tol,
                    msg="sdpa native flash mask fallback %s output" % name)
                for tensor_name, got, expected in zip(
                        ("q", "k", "v"), fetched[1:],
                        _sdpa_backward_ref(
                            q, k, v, grad_out, reference_mask)):
                    self.ac(
                        got, expected, atol=grad_tol, rtol=grad_tol,
                        msg="sdpa native flash mask fallback %s %s gradient"
                        % (name, tensor_name))

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("float16"),
                         "native fp16 flash-attn capability not configured")
    def test_sdpa_native_flash_attn_mask_fallback_fp16_cuda(self):
        self._check_sdpa_native_flash_mask_fallback("float16", 3e-3, 6e-3)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("bfloat16"),
                         "native bf16 flash-attn capability not configured")
    def test_sdpa_native_flash_attn_mask_fallback_bf16_cuda(self):
        self._check_sdpa_native_flash_mask_fallback("bfloat16", 3e-2, 3e-2)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("float16"),
                         "native fp16 flash-attn capability not configured")
    def test_native_flash_attn_higher_order_rejected_fp16_cuda(self):
        import flash_attn

        rng = np.random.RandomState(163)
        q = rng.randn(1, 8, 2, 32).astype("float32")
        k = rng.randn(1, 8, 2, 32).astype("float32")
        v = rng.randn(1, 8, 2, 32).astype("float32")

        def check(first, target, label):
            value = first.float32().numpy()
            self.assertTrue(np.isfinite(value).all(), label)
            self.assertGreater(float(np.abs(value).sum()), 0.0, label)
            with self.assertRaisesRegex(
                    RuntimeError, "Higher-order gradients.*first-order-only"):
                jt.grad(first.float32().sum(), target)

        with jt.flag_scope(use_cuda=1):
            qv, kv, vv = (
                jt.array(value).float16() for value in (q, k, v))
            dense = flash_attn.flash_attn_func(qv, kv, vv)
            dense_first = jt.grad(
                (dense.float32() * dense.float32()).sum(), qv)
            check(dense_first, qv, "dense")

            cu_seqlens = jt.array([0, 8], dtype="int32")
            q_varlen, k_varlen, v_varlen = (
                jt.array(value[0]).float16() for value in (q, k, v))
            varlen = flash_attn.flash_attn_varlen_func(
                q_varlen, k_varlen, v_varlen,
                cu_seqlens, cu_seqlens, 8, 8)
            varlen_first = jt.grad(
                (varlen.float32() * varlen.float32()).sum(), q_varlen)
            check(varlen_first, q_varlen, "varlen")

            packed = jt.array(np.stack((q, k, v), axis=2)).float16()
            packed_out = flash_attn.flash_attn_qkvpacked_func(packed)
            packed_first = jt.grad(
                (packed_out.float32() * packed_out.float32()).sum(), packed)
            check(packed_first, packed, "qkvpacked")

            trainable = jt.array(q).float16()
            fixed_k = jt.array(k).float16().stop_grad()
            fixed_v = jt.array(v).float16().stop_grad()
            optimizer = jt.optim.SGD([trainable], lr=1e-3)
            for _ in range(2):
                train_out = flash_attn.flash_attn_func(
                    trainable, fixed_k, fixed_v)
                optimizer.step(
                    (train_out.float32() * train_out.float32()).mean())
            trained = trainable.float32().numpy()
            self.assertTrue(np.isfinite(trained).all())
            self.assertGreater(float(np.max(np.abs(trained - q))), 0.0)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("bfloat16"),
                         "native bf16 flash-attn capability not configured")
    @unittest.skipUnless(_native_flash_head_dim_enabled(64),
                         "native hdim64 flash-attn capability not configured")
    def test_sdpa_native_flash_attn_backward_hdim64_bf16_cuda(self):
        rng = np.random.RandomState(151)
        q = rng.randn(1, 2, 8, 64).astype("float32")
        k = rng.randn(1, 2, 8, 64).astype("float32")
        v = rng.randn(1, 2, 8, 64).astype("float32")
        grad_out = rng.randn(1, 2, 8, 64).astype("float32")
        with jt.flag_scope(use_cuda=1):
            if hasattr(jt, "_torch_sdpa_flash_stats"):
                delattr(jt, "_torch_sdpa_flash_stats")
            qv, kv, vv = (
                jt.array(value).bfloat16() for value in (q, k, v))
            out = torch.nn.functional.scaled_dot_product_attention(qv, kv, vv)
            grads = jt.grad(
                (out.float32() * jt.array(grad_out)).sum(), [qv, kv, vv])
            fetched = jt.fetch_sync(
                [out.float32()] + [grad.float32() for grad in grads])
            stats = getattr(jt, "_torch_sdpa_flash_stats", {})

        self.assertGreaterEqual(stats.get("hits", 0), 1)
        self.assertEqual(stats.get("misses", {}), {})
        self.ac(fetched[0], _sdpa_ref(q, k, v), atol=3e-2, rtol=3e-2,
                msg="sdpa native hdim64 bf16 flash backward output")
        for name, got, expected in zip(
                ("q", "k", "v"), fetched[1:],
                _sdpa_backward_ref(q, k, v, grad_out)):
            self.ac(got, expected, atol=3e-2, rtol=3e-2,
                    msg="sdpa native hdim64 bf16 flash %s gradient" % name)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("bfloat16"),
                         "native bf16 flash-attn capability not configured")
    @unittest.skipUnless(_native_flash_head_dim_enabled(96),
                         "native hdim96 flash-attn capability not configured")
    def test_sdpa_native_flash_attn_backward_hdim96_bf16_cuda(self):
        rng = np.random.RandomState(157)
        q = rng.randn(1, 2, 8, 96).astype("float32")
        k = rng.randn(1, 2, 8, 96).astype("float32")
        v = rng.randn(1, 2, 8, 96).astype("float32")
        grad_out = rng.randn(1, 2, 8, 96).astype("float32")
        with jt.flag_scope(use_cuda=1):
            if hasattr(jt, "_torch_sdpa_flash_stats"):
                delattr(jt, "_torch_sdpa_flash_stats")
            qv, kv, vv = (
                jt.array(value).bfloat16() for value in (q, k, v))
            out = torch.nn.functional.scaled_dot_product_attention(qv, kv, vv)
            grads = jt.grad(
                (out.float32() * jt.array(grad_out)).sum(), [qv, kv, vv])
            fetched = jt.fetch_sync(
                [out.float32()] + [grad.float32() for grad in grads])
            stats = getattr(jt, "_torch_sdpa_flash_stats", {})

        self.assertGreaterEqual(stats.get("hits", 0), 1)
        self.assertEqual(stats.get("misses", {}), {})
        self.ac(fetched[0], _sdpa_ref(q, k, v), atol=3e-2, rtol=3e-2,
                msg="sdpa native hdim96 bf16 flash backward output")
        for name, got, expected in zip(
                ("q", "k", "v"), fetched[1:],
                _sdpa_backward_ref(q, k, v, grad_out)):
            self.ac(got, expected, atol=3e-2, rtol=3e-2,
                    msg="sdpa native hdim96 bf16 flash %s gradient" % name)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("bfloat16"),
                         "native bf16 flash-attn capability not configured")
    @unittest.skipUnless(_native_flash_head_dim_enabled(128),
                         "native hdim128 flash-attn capability not configured")
    def test_sdpa_native_flash_attn_backward_hdim128_bf16_cuda(self):
        rng = np.random.RandomState(163)
        q = rng.randn(1, 2, 8, 128).astype("float32")
        k = rng.randn(1, 2, 8, 128).astype("float32")
        v = rng.randn(1, 2, 8, 128).astype("float32")
        grad_out = rng.randn(1, 2, 8, 128).astype("float32")
        with jt.flag_scope(use_cuda=1):
            if hasattr(jt, "_torch_sdpa_flash_stats"):
                delattr(jt, "_torch_sdpa_flash_stats")
            qv, kv, vv = (
                jt.array(value).bfloat16() for value in (q, k, v))
            out = torch.nn.functional.scaled_dot_product_attention(qv, kv, vv)
            grads = jt.grad(
                (out.float32() * jt.array(grad_out)).sum(), [qv, kv, vv])
            fetched = jt.fetch_sync(
                [out.float32()] + [grad.float32() for grad in grads])
            stats = getattr(jt, "_torch_sdpa_flash_stats", {})

        self.assertGreaterEqual(stats.get("hits", 0), 1)
        self.assertEqual(stats.get("misses", {}), {})
        self.ac(fetched[0], _sdpa_ref(q, k, v), atol=3e-2, rtol=3e-2,
                msg="sdpa native hdim128 bf16 flash backward output")
        for name, got, expected in zip(
                ("q", "k", "v"), fetched[1:],
                _sdpa_backward_ref(q, k, v, grad_out)):
            self.ac(got, expected, atol=3e-2, rtol=3e-2,
                    msg="sdpa native hdim128 bf16 flash %s gradient" % name)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("bfloat16"),
                         "native bf16 flash-attn capability not configured")
    @unittest.skipUnless(_native_flash_head_dim_enabled(192),
                         "native hdim192 flash-attn capability not configured")
    def test_sdpa_native_flash_attn_backward_hdim192_bf16_cuda(self):
        rng = np.random.RandomState(167)
        q = rng.randn(1, 2, 8, 192).astype("float32")
        k = rng.randn(1, 2, 8, 192).astype("float32")
        v = rng.randn(1, 2, 8, 192).astype("float32")
        grad_out = rng.randn(1, 2, 8, 192).astype("float32")
        with jt.flag_scope(use_cuda=1):
            if hasattr(jt, "_torch_sdpa_flash_stats"):
                delattr(jt, "_torch_sdpa_flash_stats")
            qv, kv, vv = (
                jt.array(value).bfloat16() for value in (q, k, v))
            out = torch.nn.functional.scaled_dot_product_attention(qv, kv, vv)
            grads = jt.grad(
                (out.float32() * jt.array(grad_out)).sum(), [qv, kv, vv])
            fetched = jt.fetch_sync(
                [out.float32()] + [grad.float32() for grad in grads])
            stats = getattr(jt, "_torch_sdpa_flash_stats", {})

        self.assertGreaterEqual(stats.get("hits", 0), 1)
        self.assertEqual(stats.get("misses", {}), {})
        self.ac(fetched[0], _sdpa_ref(q, k, v), atol=3e-2, rtol=3e-2,
                msg="sdpa native hdim192 bf16 flash backward output")
        for name, got, expected in zip(
                ("q", "k", "v"), fetched[1:],
                _sdpa_backward_ref(q, k, v, grad_out)):
            self.ac(got, expected, atol=3e-2, rtol=3e-2,
                    msg="sdpa native hdim192 bf16 flash %s gradient" % name)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("bfloat16"),
                         "native bf16 flash-attn capability not configured")
    @unittest.skipUnless(_native_flash_head_dim_enabled(256),
                         "native hdim256 flash-attn capability not configured")
    def test_sdpa_native_flash_attn_backward_hdim256_bf16_cuda(self):
        rng = np.random.RandomState(173)
        q = rng.randn(1, 2, 8, 256).astype("float32")
        k = rng.randn(1, 2, 8, 256).astype("float32")
        v = rng.randn(1, 2, 8, 256).astype("float32")
        grad_out = rng.randn(1, 2, 8, 256).astype("float32")
        with jt.flag_scope(use_cuda=1):
            if hasattr(jt, "_torch_sdpa_flash_stats"):
                delattr(jt, "_torch_sdpa_flash_stats")
            qv, kv, vv = (
                jt.array(value).bfloat16() for value in (q, k, v))
            out = torch.nn.functional.scaled_dot_product_attention(qv, kv, vv)
            grads = jt.grad(
                (out.float32() * jt.array(grad_out)).sum(), [qv, kv, vv])
            fetched = jt.fetch_sync(
                [out.float32()] + [grad.float32() for grad in grads])
            stats = getattr(jt, "_torch_sdpa_flash_stats", {})

        self.assertGreaterEqual(stats.get("hits", 0), 1)
        self.assertEqual(stats.get("misses", {}), {})
        self.ac(fetched[0], _sdpa_ref(q, k, v), atol=3e-2, rtol=3e-2,
                msg="sdpa native hdim256 bf16 flash backward output")
        for name, got, expected in zip(
                ("q", "k", "v"), fetched[1:],
                _sdpa_backward_ref(q, k, v, grad_out)):
            self.ac(got, expected, atol=3e-2, rtol=3e-2,
                    msg="sdpa native hdim256 bf16 flash %s gradient" % name)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("float16"),
                         "native fp16 flash-attn capability not configured")
    @unittest.skipUnless(_native_flash_head_dim_enabled(64),
                         "native hdim64 flash-attn capability not configured")
    def test_sdpa_native_flash_attn_backward_hdim64_fp16_cuda(self):
        rng = np.random.RandomState(127)
        q = rng.randn(1, 2, 8, 64).astype("float32")
        k = rng.randn(1, 2, 8, 64).astype("float32")
        v = rng.randn(1, 2, 8, 64).astype("float32")
        grad_out = rng.randn(1, 2, 8, 64).astype("float32")
        with jt.flag_scope(use_cuda=1):
            if hasattr(jt, "_torch_sdpa_flash_stats"):
                delattr(jt, "_torch_sdpa_flash_stats")
            qv, kv, vv = (
                jt.array(value).float16() for value in (q, k, v))
            out = torch.nn.functional.scaled_dot_product_attention(qv, kv, vv)
            grads = jt.grad(
                (out.float32() * jt.array(grad_out)).sum(), [qv, kv, vv])
            fetched = jt.fetch_sync(
                [out.float32()] + [grad.float32() for grad in grads])
            stats = getattr(jt, "_torch_sdpa_flash_stats", {})

        self.assertGreaterEqual(stats.get("hits", 0), 1)
        self.assertEqual(stats.get("misses", {}), {})
        self.ac(fetched[0], _sdpa_ref(q, k, v), atol=3e-3, rtol=3e-3,
                msg="sdpa native hdim64 flash backward output")
        for name, got, expected in zip(
                ("q", "k", "v"), fetched[1:],
                _sdpa_backward_ref(q, k, v, grad_out)):
            self.ac(got, expected, atol=3e-3, rtol=3e-3,
                    msg="sdpa native hdim64 flash %s gradient" % name)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("float16"),
                         "native fp16 flash-attn capability not configured")
    @unittest.skipUnless(_native_flash_head_dim_enabled(96),
                         "native hdim96 flash-attn capability not configured")
    def test_sdpa_native_flash_attn_backward_hdim96_fp16_cuda(self):
        rng = np.random.RandomState(131)
        q = rng.randn(1, 2, 8, 96).astype("float32")
        k = rng.randn(1, 2, 8, 96).astype("float32")
        v = rng.randn(1, 2, 8, 96).astype("float32")
        grad_out = rng.randn(1, 2, 8, 96).astype("float32")
        with jt.flag_scope(use_cuda=1):
            if hasattr(jt, "_torch_sdpa_flash_stats"):
                delattr(jt, "_torch_sdpa_flash_stats")
            qv, kv, vv = (
                jt.array(value).float16() for value in (q, k, v))
            out = torch.nn.functional.scaled_dot_product_attention(qv, kv, vv)
            grads = jt.grad(
                (out.float32() * jt.array(grad_out)).sum(), [qv, kv, vv])
            fetched = jt.fetch_sync(
                [out.float32()] + [grad.float32() for grad in grads])
            stats = getattr(jt, "_torch_sdpa_flash_stats", {})

        self.assertGreaterEqual(stats.get("hits", 0), 1)
        self.assertEqual(stats.get("misses", {}), {})
        self.ac(fetched[0], _sdpa_ref(q, k, v), atol=3e-3, rtol=3e-3,
                msg="sdpa native hdim96 flash backward output")
        for name, got, expected in zip(
                ("q", "k", "v"), fetched[1:],
                _sdpa_backward_ref(q, k, v, grad_out)):
            self.ac(got, expected, atol=3e-3, rtol=3e-3,
                    msg="sdpa native hdim96 flash %s gradient" % name)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("float16"),
                         "native fp16 flash-attn capability not configured")
    @unittest.skipUnless(_native_flash_head_dim_enabled(128),
                         "native hdim128 flash-attn capability not configured")
    def test_sdpa_native_flash_attn_backward_hdim128_fp16_cuda(self):
        rng = np.random.RandomState(137)
        q = rng.randn(1, 2, 8, 128).astype("float32")
        k = rng.randn(1, 2, 8, 128).astype("float32")
        v = rng.randn(1, 2, 8, 128).astype("float32")
        grad_out = rng.randn(1, 2, 8, 128).astype("float32")
        with jt.flag_scope(use_cuda=1):
            if hasattr(jt, "_torch_sdpa_flash_stats"):
                delattr(jt, "_torch_sdpa_flash_stats")
            qv, kv, vv = (
                jt.array(value).float16() for value in (q, k, v))
            out = torch.nn.functional.scaled_dot_product_attention(qv, kv, vv)
            grads = jt.grad(
                (out.float32() * jt.array(grad_out)).sum(), [qv, kv, vv])
            fetched = jt.fetch_sync(
                [out.float32()] + [grad.float32() for grad in grads])
            stats = getattr(jt, "_torch_sdpa_flash_stats", {})

        self.assertGreaterEqual(stats.get("hits", 0), 1)
        self.assertEqual(stats.get("misses", {}), {})
        self.ac(fetched[0], _sdpa_ref(q, k, v), atol=3e-3, rtol=3e-3,
                msg="sdpa native hdim128 flash backward output")
        for name, got, expected in zip(
                ("q", "k", "v"), fetched[1:],
                _sdpa_backward_ref(q, k, v, grad_out)):
            self.ac(got, expected, atol=3e-3, rtol=3e-3,
                    msg="sdpa native hdim128 flash %s gradient" % name)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("float16"),
                         "native fp16 flash-attn capability not configured")
    @unittest.skipUnless(_native_flash_head_dim_enabled(192),
                         "native hdim192 flash-attn capability not configured")
    def test_sdpa_native_flash_attn_backward_hdim192_fp16_cuda(self):
        rng = np.random.RandomState(139)
        q = rng.randn(1, 2, 8, 192).astype("float32")
        k = rng.randn(1, 2, 8, 192).astype("float32")
        v = rng.randn(1, 2, 8, 192).astype("float32")
        grad_out = rng.randn(1, 2, 8, 192).astype("float32")
        with jt.flag_scope(use_cuda=1):
            if hasattr(jt, "_torch_sdpa_flash_stats"):
                delattr(jt, "_torch_sdpa_flash_stats")
            qv, kv, vv = (
                jt.array(value).float16() for value in (q, k, v))
            out = torch.nn.functional.scaled_dot_product_attention(qv, kv, vv)
            grads = jt.grad(
                (out.float32() * jt.array(grad_out)).sum(), [qv, kv, vv])
            fetched = jt.fetch_sync(
                [out.float32()] + [grad.float32() for grad in grads])
            stats = getattr(jt, "_torch_sdpa_flash_stats", {})

        self.assertGreaterEqual(stats.get("hits", 0), 1)
        self.assertEqual(stats.get("misses", {}), {})
        self.ac(fetched[0], _sdpa_ref(q, k, v), atol=4e-3, rtol=4e-3,
                msg="sdpa native hdim192 flash backward output")
        for name, got, expected in zip(
                ("q", "k", "v"), fetched[1:],
                _sdpa_backward_ref(q, k, v, grad_out)):
            self.ac(got, expected, atol=4e-3, rtol=4e-3,
                    msg="sdpa native hdim192 flash %s gradient" % name)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("float16"),
                         "native fp16 flash-attn capability not configured")
    @unittest.skipUnless(_native_flash_head_dim_enabled(256),
                         "native hdim256 flash-attn capability not configured")
    def test_sdpa_native_flash_attn_backward_hdim256_fp16_cuda(self):
        rng = np.random.RandomState(149)
        q = rng.randn(1, 2, 8, 256).astype("float32")
        k = rng.randn(1, 2, 8, 256).astype("float32")
        v = rng.randn(1, 2, 8, 256).astype("float32")
        grad_out = rng.randn(1, 2, 8, 256).astype("float32")
        with jt.flag_scope(use_cuda=1):
            if hasattr(jt, "_torch_sdpa_flash_stats"):
                delattr(jt, "_torch_sdpa_flash_stats")
            qv, kv, vv = (
                jt.array(value).float16() for value in (q, k, v))
            out = torch.nn.functional.scaled_dot_product_attention(qv, kv, vv)
            grads = jt.grad(
                (out.float32() * jt.array(grad_out)).sum(), [qv, kv, vv])
            fetched = jt.fetch_sync(
                [out.float32()] + [grad.float32() for grad in grads])
            stats = getattr(jt, "_torch_sdpa_flash_stats", {})

        self.assertGreaterEqual(stats.get("hits", 0), 1)
        self.assertEqual(stats.get("misses", {}), {})
        self.ac(fetched[0], _sdpa_ref(q, k, v), atol=4e-3, rtol=4e-3,
                msg="sdpa native hdim256 flash backward output")
        for name, got, expected in zip(
                ("q", "k", "v"), fetched[1:],
                _sdpa_backward_ref(q, k, v, grad_out)):
            self.ac(got, expected, atol=4e-3, rtol=4e-3,
                    msg="sdpa native hdim256 flash %s gradient" % name)

    def _check_native_flash_attn_dropout(self, dtype, head_dim=32):
        import flash_attn
        from jittor.compat.shim.backends import flash_attention as flashattn_jittor

        rng = np.random.RandomState(103)
        q = rng.randn(2, 64, 4, head_dim).astype("float32")
        k = rng.randn(2, 64, 4, head_dim).astype("float32")
        v = rng.randn(2, 64, 4, head_dim).astype("float32")
        grad_out = rng.randn(2, 64, 4, head_dim).astype("float32")

        def run(seed=None):
            if seed is not None:
                torch.manual_seed(seed)
            qv, kv, vv = (
                jt.array(value).to(dtype) for value in (q, k, v))
            out, _, probability = flash_attn.flash_attn_func(
                qv, kv, vv, dropout_p=0.25, deterministic=True,
                return_attn_probs=True)
            self.assertEqual(tuple(probability.shape), (2, 4, 128, 128))
            grads = jt.grad(
                (out.float32() * jt.array(grad_out)).sum(), [qv, kv, vv])
            return jt.fetch_sync(
                [out.float32()] + [grad.float32() for grad in grads])

        with jt.flag_scope(use_cuda=1):
            first = run(20260824)
            replay = run(20260824)
            advanced = run()
            backend = flashattn_jittor.load_backend()

        self.assertTrue(getattr(backend, "_flashattn_jittor_training", False))
        np.testing.assert_array_equal(first[0], replay[0])
        for first_grad, replay_grad in zip(first[1:], replay[1:]):
            np.testing.assert_array_equal(first_grad, replay_grad)
            self.assertTrue(np.isfinite(first_grad).all())
            self.assertGreater(float(np.abs(first_grad).sum()), 0.0)
        self.assertGreater(float(np.max(np.abs(replay[0] - advanced[0]))), 1e-3)

    def _check_native_flash_attn_dropout_backward_rejected(self, dtype,
                                                            head_dim):
        import flash_attn

        rng = np.random.RandomState(103)
        values = [
            rng.randn(1, 8, 2, head_dim).astype("float32") for _ in range(3)
        ]
        error = "dropout backward for head dimension %s.*sm" % head_dim
        with jt.flag_scope(use_cuda=1):
            qv, kv, vv = (jt.array(value).to(dtype) for value in values)
            with jt.no_grad():
                forward = flash_attn.flash_attn_func(
                    qv, kv, vv, dropout_p=0.25, deterministic=True)
                forward_value = forward.float32().numpy()
            with self.assertRaisesRegex(RuntimeError, error):
                flash_attn.flash_attn_func(
                    qv, kv, vv, dropout_p=0.25, deterministic=True)
            cu_seqlens = jt.array([0, 8], dtype="int32")
            with self.assertRaisesRegex(RuntimeError, error):
                flash_attn.flash_attn_varlen_func(
                    qv[0], kv[0], vv[0], cu_seqlens, cu_seqlens, 8, 8,
                    dropout_p=0.25, deterministic=True)
            packed = jt.array(np.stack(values, axis=2)).to(dtype)
            with self.assertRaisesRegex(RuntimeError, error):
                flash_attn.flash_attn_qkvpacked_func(
                    packed, dropout_p=0.25, deterministic=True)
        self.assertTrue(np.isfinite(forward_value).all())
        self.assertGreater(float(np.abs(forward_value).sum()), 0.0)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("float16"),
                         "native fp16 flash-attn capability not configured")
    def test_native_flash_attn_dropout_replays_seed_and_backward(self):
        self._check_native_flash_attn_dropout("float16")

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("bfloat16"),
                         "native bf16 flash-attn capability not configured")
    def test_native_flash_attn_dropout_replays_seed_and_backward_bf16(self):
        self._check_native_flash_attn_dropout("bfloat16")

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("float16"),
                         "native fp16 flash-attn capability not configured")
    @unittest.skipUnless(_native_flash_head_dim_enabled(64),
                         "native hdim64 flash-attn capability not configured")
    def test_native_flash_attn_dropout_hdim64_fp16(self):
        self._check_native_flash_attn_dropout("float16", 64)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("bfloat16"),
                         "native bf16 flash-attn capability not configured")
    @unittest.skipUnless(_native_flash_head_dim_enabled(64),
                         "native hdim64 flash-attn capability not configured")
    def test_native_flash_attn_dropout_hdim64_bf16(self):
        self._check_native_flash_attn_dropout("bfloat16", 64)

    def _check_native_flash_attn_varlen_backward(
            self, dtype, out_tol, grad_tol, head_dim=32):
        import flash_attn

        rng = np.random.RandomState(107)
        q = rng.randn(7, 2, head_dim).astype("float32")
        k = rng.randn(7, 2, head_dim).astype("float32")
        v = rng.randn(7, 2, head_dim).astype("float32")
        grad_out = rng.randn(7, 2, head_dim).astype("float32")
        cu_seqlens = np.array([0, 3, 7], dtype="int32")

        expected_out = []
        expected_grads = [[], [], []]
        for start, stop in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            segment = [
                np.transpose(value[start:stop], (1, 0, 2))[None]
                for value in (q, k, v, grad_out)]
            expected_out.append(np.transpose(
                _sdpa_ref(*segment[:3])[0], (1, 0, 2)))
            for bucket, gradient in zip(
                    expected_grads,
                    _sdpa_backward_ref(*segment)):
                bucket.append(np.transpose(gradient[0], (1, 0, 2)))

        with jt.flag_scope(use_cuda=1):
            qv, kv, vv = (
                jt.array(value).to(dtype) for value in (q, k, v))
            cu = jt.array(cu_seqlens)
            out = flash_attn.flash_attn_varlen_func(
                qv, kv, vv, cu, cu, 4, 4)
            grads = jt.grad(
                (out.float32() * jt.array(grad_out)).sum(), [qv, kv, vv])
            fetched = jt.fetch_sync(
                [out.float32()] + [grad.float32() for grad in grads])

        self.ac(fetched[0], np.concatenate(expected_out), atol=out_tol, rtol=out_tol,
                msg="native flash varlen output")
        for name, got, expected in zip(("q", "k", "v"), fetched[1:], expected_grads):
            self.ac(got, np.concatenate(expected), atol=grad_tol, rtol=grad_tol,
                    msg="native flash varlen %s gradient" % name)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("float16"),
                         "native fp16 flash-attn capability not configured")
    def test_native_flash_attn_varlen_backward_fp16_cuda(self):
        self._check_native_flash_attn_varlen_backward("float16", 3e-3, 6e-3)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("bfloat16"),
                         "native bf16 flash-attn capability not configured")
    def test_native_flash_attn_varlen_backward_bf16_cuda(self):
        self._check_native_flash_attn_varlen_backward("bfloat16", 3e-2, 3e-2)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("float16"),
                         "native fp16 flash-attn capability not configured")
    @unittest.skipUnless(_native_flash_head_dim_enabled(64),
                         "native hdim64 flash-attn capability not configured")
    def test_native_flash_attn_varlen_backward_hdim64_fp16_cuda(self):
        self._check_native_flash_attn_varlen_backward("float16", 3e-3, 6e-3, 64)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("bfloat16"),
                         "native bf16 flash-attn capability not configured")
    @unittest.skipUnless(_native_flash_head_dim_enabled(64),
                         "native hdim64 flash-attn capability not configured")
    def test_native_flash_attn_varlen_backward_hdim64_bf16_cuda(self):
        self._check_native_flash_attn_varlen_backward("bfloat16", 3e-2, 3e-2, 64)

    def _check_native_flash_attn_qkvpacked_backward(self, dtype, head_dim=32):
        import flash_attn

        rng = np.random.RandomState(109)
        qkv = rng.randn(1, 8, 3, 2, head_dim).astype("float32")
        grad_out = rng.randn(1, 8, 2, head_dim).astype("float32")
        with jt.flag_scope(use_cuda=1):
            packed = jt.array(qkv).to(dtype)
            packed_out = flash_attn.flash_attn_qkvpacked_func(packed)
            packed_grad = jt.grad(
                (packed_out.float32() * jt.array(grad_out)).sum(), packed)

            qv, kv, vv = (
                jt.array(qkv[:, :, index]).to(dtype) for index in range(3))
            dense_out = flash_attn.flash_attn_func(qv, kv, vv)
            dense_grads = jt.grad(
                (dense_out.float32() * jt.array(grad_out)).sum(), [qv, kv, vv])
            fetched = jt.fetch_sync([
                packed_out.float32(), packed_grad.float32(), dense_out.float32(),
                *[gradient.float32() for gradient in dense_grads],
            ])

        self.ac(fetched[0], fetched[2], atol=0.0, rtol=0.0,
                msg="native qkvpacked output")
        expected_grad = np.stack(fetched[3:], axis=2)
        self.ac(fetched[1], expected_grad, atol=0.0, rtol=0.0,
                msg="native qkvpacked gradient")

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("float16"),
                         "native fp16 flash-attn capability not configured")
    def test_native_flash_attn_qkvpacked_backward_matches_dense(self):
        self._check_native_flash_attn_qkvpacked_backward("float16")

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("bfloat16"),
                         "native bf16 flash-attn capability not configured")
    def test_native_flash_attn_qkvpacked_backward_matches_dense_bf16(self):
        self._check_native_flash_attn_qkvpacked_backward("bfloat16")

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("float16"),
                         "native fp16 flash-attn capability not configured")
    @unittest.skipUnless(_native_flash_head_dim_enabled(64),
                         "native hdim64 flash-attn capability not configured")
    def test_native_flash_attn_qkvpacked_backward_hdim64_fp16(self):
        self._check_native_flash_attn_qkvpacked_backward("float16", 64)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("bfloat16"),
                         "native bf16 flash-attn capability not configured")
    @unittest.skipUnless(_native_flash_head_dim_enabled(64),
                         "native hdim64 flash-attn capability not configured")
    def test_native_flash_attn_qkvpacked_backward_hdim64_bf16(self):
        self._check_native_flash_attn_qkvpacked_backward("bfloat16", 64)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("float16"),
                         "native fp16 flash-attn capability not configured")
    @unittest.skipUnless(_native_flash_head_dim_enabled(96),
                         "native hdim96 flash-attn capability not configured")
    def test_native_flash_attn_training_variants_hdim96_fp16(self):
        self._check_native_flash_attn_dropout("float16", 96)
        self._check_native_flash_attn_varlen_backward("float16", 3e-3, 6e-3, 96)
        self._check_native_flash_attn_qkvpacked_backward("float16", 96)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("bfloat16"),
                         "native bf16 flash-attn capability not configured")
    @unittest.skipUnless(_native_flash_head_dim_enabled(96),
                         "native hdim96 flash-attn capability not configured")
    def test_native_flash_attn_training_variants_hdim96_bf16(self):
        self._check_native_flash_attn_dropout("bfloat16", 96)
        self._check_native_flash_attn_varlen_backward("bfloat16", 3e-2, 3e-2, 96)
        self._check_native_flash_attn_qkvpacked_backward("bfloat16", 96)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("float16"),
                         "native fp16 flash-attn capability not configured")
    @unittest.skipUnless(_native_flash_head_dim_enabled(128),
                         "native hdim128 flash-attn capability not configured")
    def test_native_flash_attn_training_variants_hdim128_fp16(self):
        self._check_native_flash_attn_dropout("float16", 128)
        self._check_native_flash_attn_varlen_backward("float16", 3e-3, 6e-3, 128)
        self._check_native_flash_attn_qkvpacked_backward("float16", 128)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("bfloat16"),
                         "native bf16 flash-attn capability not configured")
    @unittest.skipUnless(_native_flash_head_dim_enabled(128),
                         "native hdim128 flash-attn capability not configured")
    def test_native_flash_attn_training_variants_hdim128_bf16(self):
        self._check_native_flash_attn_dropout("bfloat16", 128)
        self._check_native_flash_attn_varlen_backward("bfloat16", 3e-2, 3e-2, 128)
        self._check_native_flash_attn_qkvpacked_backward("bfloat16", 128)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("float16"),
                         "native fp16 flash-attn capability not configured")
    @unittest.skipUnless(_native_flash_head_dim_enabled(192),
                         "native hdim192 flash-attn capability not configured")
    def test_native_flash_attn_training_variants_hdim192_fp16(self):
        self._check_native_flash_attn_dropout("float16", 192)
        self._check_native_flash_attn_varlen_backward("float16", 3e-3, 6e-3, 192)
        self._check_native_flash_attn_qkvpacked_backward("float16", 192)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("bfloat16"),
                         "native bf16 flash-attn capability not configured")
    @unittest.skipUnless(_native_flash_head_dim_enabled(192),
                         "native hdim192 flash-attn capability not configured")
    def test_native_flash_attn_training_variants_hdim192_bf16(self):
        self._check_native_flash_attn_dropout("bfloat16", 192)
        self._check_native_flash_attn_varlen_backward("bfloat16", 3e-2, 3e-2, 192)
        self._check_native_flash_attn_qkvpacked_backward("bfloat16", 192)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("float16"),
                         "native fp16 flash-attn capability not configured")
    @unittest.skipUnless(_native_flash_head_dim_enabled(256),
                         "native hdim256 flash-attn capability not configured")
    def test_native_flash_attn_training_variants_hdim256_fp16(self):
        from jittor.compat.shim.backends import flash_attention as flashattn_jittor

        if flashattn_jittor._official_dropout_backward_supported(
                256, jt.flags.cuda_archs):
            self._check_native_flash_attn_dropout("float16", 256)
        else:
            self._check_native_flash_attn_dropout_backward_rejected("float16", 256)
        self._check_native_flash_attn_varlen_backward("float16", 3e-3, 6e-3, 256)
        self._check_native_flash_attn_qkvpacked_backward("float16", 256)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @unittest.skipIf(not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
                     "native flash-attn source not configured")
    @unittest.skipUnless(_native_flash_dtype_enabled("bfloat16"),
                         "native bf16 flash-attn capability not configured")
    @unittest.skipUnless(_native_flash_head_dim_enabled(256),
                         "native hdim256 flash-attn capability not configured")
    def test_native_flash_attn_training_variants_hdim256_bf16(self):
        from jittor.compat.shim.backends import flash_attention as flashattn_jittor

        if flashattn_jittor._official_dropout_backward_supported(
                256, jt.flags.cuda_archs):
            self._check_native_flash_attn_dropout("bfloat16", 256)
        else:
            self._check_native_flash_attn_dropout_backward_rejected("bfloat16", 256)
        self._check_native_flash_attn_varlen_backward("bfloat16", 3e-2, 3e-2, 256)
        self._check_native_flash_attn_qkvpacked_backward("bfloat16", 256)

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
        from jittor.compat.shim.backends import flash_attention as flashattn_jittor

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

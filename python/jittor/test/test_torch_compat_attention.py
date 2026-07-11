"""Torch-grade attention/transformer parity for ``import jittor as torch``.

The transformer surface is the core of the jittor-as-torch project. Compares
F.scaled_dot_product_attention and nn.MultiheadAttention against explicit numpy references.
CPU+CUDA.

Run:  python -m jittor.test.test_torch_compat_attention
"""
import unittest
import os
import pathlib
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
    def test_required_flash_backend_returning_none_raises(self):
        import flash_attn
        from jittor.torch_shim import flashattn_jittor

        q = jt.ones((1, 2, 4, 32), dtype="float16")
        backend = ModuleType("required_flash_backend")
        with jt.flag_scope(use_cuda=1), jt.no_grad(), \
                mock.patch.object(flashattn_jittor, "load_backend_for",
                                  return_value=(backend, None)), \
                mock.patch.object(flashattn_jittor, "required", return_value=True), \
                mock.patch.object(flash_attn, "flash_attn_func", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "returned no output"):
                torch.nn.functional.scaled_dot_product_attention(q, q, q)

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

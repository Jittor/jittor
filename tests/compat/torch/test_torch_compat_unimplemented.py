"""Negative tests for torch APIs that used to be signature-complete no-ops.

Task 7.01 of the 2.0 refactor.  Every API listed in the compat audit's
"looks supported, actually a no-op" / "distributed and FSDP2" / "dtype and
device mapping" sections is covered here by one of two shapes:

* **implemented** -- a test that asserts the API now really takes effect
  (``torch.autocast`` changes the dtype ops are computed in, ``load_state_dict``
  reports the real key difference, ...);
* **refused** -- a test that asserts the API raises ``NotImplementedError`` and
  that ``JITTOR_TORCH_ALLOW_STUB``/``torch.compat_allow_stub(True)`` restores
  the old silent behaviour.

The last test renders ``torch.compat_unimplemented_apis()`` into the generated
"unimplemented API list" the plan asks for, and fails if an API is refused
without a stated consequence.

Run: python -m pytest tests/compat/torch/test_torch_compat_unimplemented.py
"""
import os
import unittest
import warnings

import numpy as np

import jittor as jt
import jittor as torch
from jittor.compat import stub_policy

class StubPolicyBase(unittest.TestCase):
    """Every test runs with the escape hatch OFF unless it says otherwise."""

    def setUp(self):
        self._saved_override = stub_policy.set_allow_stub(False)
        self._saved_env = os.environ.pop(stub_policy.ENV_VAR, None)
        stub_policy.reset_warned()

    def tearDown(self):
        stub_policy.set_allow_stub(self._saved_override)
        if self._saved_env is not None:
            os.environ[stub_policy.ENV_VAR] = self._saved_env
        else:
            os.environ.pop(stub_policy.ENV_VAR, None)
        stub_policy.reset_warned()

    def assertRefuses(self, fn, *needles):
        """fn() must raise NotImplementedError naming the API and the damage."""
        with self.assertRaises(NotImplementedError) as cm:
            fn()
        msg = str(cm.exception)
        for needle in needles:
            self.assertIn(needle, msg)
        self.assertIn(stub_policy.ENV_VAR, msg,
                      "the message must document the escape hatch")
        return msg

    def assertStubFallback(self, fn):
        """With the hatch on, fn() must warn once and return the old value."""
        stub_policy.set_allow_stub(True)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = fn()
            self.assertTrue(any(issubclass(w.category, RuntimeWarning)
                                for w in caught),
                            "opting into the stub must warn at least once")
            return result
        finally:
            stub_policy.set_allow_stub(False)

class TestStubPolicy(StubPolicyBase):
    def test_hatch_is_off_by_default(self):
        stub_policy.set_allow_stub(None)
        os.environ.pop(stub_policy.ENV_VAR, None)
        self.assertFalse(stub_policy.allow_stub())

    def test_env_var_opens_the_hatch(self):
        stub_policy.set_allow_stub(None)
        os.environ[stub_policy.ENV_VAR] = "1"
        try:
            self.assertTrue(stub_policy.allow_stub())
        finally:
            os.environ.pop(stub_policy.ENV_VAR, None)

    def test_env_var_off_values_stay_closed(self):
        stub_policy.set_allow_stub(None)
        for value in ("0", "false", "no", "off", ""):
            os.environ[stub_policy.ENV_VAR] = value
            self.assertFalse(stub_policy.allow_stub(), value)
        os.environ.pop(stub_policy.ENV_VAR, None)

    def test_torch_namespace_exposes_the_switch(self):
        self.assertFalse(torch.compat_allow_stub())
        try:
            self.assertTrue(torch.compat_allow_stub(True))
        finally:
            torch.compat_allow_stub(False)

    def test_warns_once_per_api(self):
        stub_policy.set_allow_stub(True)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for _ in range(3):
                    stub_policy.unimplemented("demo.api", "lose the data")
            self.assertEqual(len(caught), 1)
        finally:
            stub_policy.set_allow_stub(False)

class TestAutocast(StubPolicyBase):
    """torch.autocast used to be a total no-op: mixed precision silently ran fp32."""

    def setUp(self):
        super().setUp()
        self._amp_reg = int(getattr(jt.flags, "amp_reg", 0))

    def tearDown(self):
        jt.flags.amp_reg = self._amp_reg
        super().tearDown()

    def test_autocast_actually_lowers_op_dtype(self):
        a = jt.random((4, 4), dtype="float32")
        b = jt.random((4, 4), dtype="float32")
        self.assertEqual(str((a @ b).dtype), "float32")
        with torch.autocast("cuda", dtype=torch.float16):
            inside = a @ b
        self.assertEqual(str(inside.dtype), "float16",
                         "autocast must change the dtype ops compute in")

    def test_autocast_reports_itself_enabled(self):
        self.assertFalse(torch.is_autocast_enabled())
        with torch.autocast("cuda", dtype=torch.float16):
            self.assertTrue(torch.is_autocast_enabled())
            self.assertEqual(str(torch.get_autocast_dtype("cuda")), "float16")
        self.assertFalse(torch.is_autocast_enabled())

    def test_autocast_restores_the_previous_register(self):
        before = int(jt.flags.amp_reg)
        with torch.autocast("cuda", dtype=torch.float16):
            self.assertNotEqual(int(jt.flags.amp_reg), before)
        self.assertEqual(int(jt.flags.amp_reg), before)

    def test_autocast_enabled_false_is_a_real_no_op(self):
        a = jt.random((4, 4), dtype="float32")
        with torch.autocast("cuda", dtype=torch.float16, enabled=False):
            self.assertFalse(torch.is_autocast_enabled())
            self.assertEqual(str((a + a).dtype), "float32")

    def test_autocast_as_decorator_takes_effect(self):
        @torch.autocast("cuda", dtype=torch.float16)
        def f(x):
            return x * x

        out = f(jt.random((4, 4), dtype="float32"))
        self.assertEqual(str(out.dtype), "float16")

    def test_autocast_float32_forces_fp32(self):
        a = jt.random((4, 4), dtype="float16")
        with torch.autocast("cuda", dtype=torch.float32):
            self.assertEqual(str((a + a).dtype), "float32")

    def test_autocast_rejects_a_dtype_it_cannot_express(self):
        self.assertRefuses(
            lambda: torch.autocast("cuda", dtype="float8_e4m3fn"),
            "torch.autocast", "float8_e4m3fn")

    def test_autocast_stub_fallback_restores_the_no_op(self):
        ctx = self.assertStubFallback(
            lambda: torch.autocast("cuda", dtype="float8_e4m3fn"))
        self.assertIsNotNone(ctx)

class TestLoadStateDict(StubPolicyBase):
    """Module.load_state_dict returned _IncompatibleKeys([], []) unconditionally."""

    def _model(self):
        return torch.nn.Linear(4, 3)

    def test_missing_keys_are_reported(self):
        model = self._model()
        sd = dict(model.state_dict())
        sd.pop("bias")
        result = model.load_state_dict(sd, strict=False)
        self.assertIn("bias", result.missing_keys)
        self.assertEqual(result.unexpected_keys, [])

    def test_unexpected_keys_are_reported(self):
        model = self._model()
        sd = dict(model.state_dict())
        sd["not_a_param"] = jt.zeros(3)
        result = model.load_state_dict(sd, strict=False)
        self.assertIn("not_a_param", result.unexpected_keys)
        self.assertEqual(result.missing_keys, [])

    def test_strict_true_rejects_a_missing_key(self):
        model = self._model()
        sd = dict(model.state_dict())
        sd.pop("weight")
        with self.assertRaises(RuntimeError) as cm:
            model.load_state_dict(sd)
        self.assertIn("Missing key", str(cm.exception))
        self.assertIn("weight", str(cm.exception))

    def test_strict_true_rejects_an_unexpected_key(self):
        model = self._model()
        sd = dict(model.state_dict())
        sd["extra.weight"] = jt.zeros(2)
        with self.assertRaises(RuntimeError) as cm:
            model.load_state_dict(sd)
        self.assertIn("Unexpected key", str(cm.exception))

    def test_shape_mismatch_raises_even_when_not_strict(self):
        model = self._model()
        sd = dict(model.state_dict())
        sd["weight"] = jt.zeros((7, 7))
        with self.assertRaises(RuntimeError) as cm:
            model.load_state_dict(sd, strict=False)
        self.assertIn("size mismatch", str(cm.exception))

    def test_a_matching_checkpoint_still_loads(self):
        src = self._model()
        dst = self._model()
        result = dst.load_state_dict(src.state_dict())
        self.assertEqual(list(result.missing_keys), [])
        self.assertEqual(list(result.unexpected_keys), [])
        np.testing.assert_allclose(dst.weight.numpy(), src.weight.numpy())

    def test_an_entirely_wrong_checkpoint_no_longer_loads_silently(self):
        # The regression this whole task exists for: before, this returned
        # IncompatibleKeys([], []) and left the model randomly initialised.
        model = self._model()
        with self.assertRaises(RuntimeError):
            model.load_state_dict({"encoder.layer.0.weight": jt.zeros((4, 4))})

class TestTorchLoad(StubPolicyBase):
    """torch.load ignored weights_only and map_location and faked unknown classes."""

    def setUp(self):
        super().setUp()
        import tempfile
        self._dir = tempfile.mkdtemp(prefix="jt_load_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._dir, ignore_errors=True)
        super().tearDown()

    def _path(self, name):
        return os.path.join(self._dir, name)

    def test_plain_tensor_checkpoint_still_round_trips(self):
        p = self._path("t.pkl")
        x = jt.array(np.arange(6, dtype="float32"))
        torch.save({"x": x, "step": 3}, p)
        got = torch.load(p)
        np.testing.assert_array_equal(got["x"].numpy(), x.numpy())
        self.assertEqual(got["step"], 3)

    def test_weights_only_rejects_an_arbitrary_class(self):
        import pickle
        p = self._path("obj.pkl")
        with open(p, "wb") as fh:
            pickle.dump({"cfg": _PayloadClass(5)}, fh)
        with self.assertRaises(pickle.UnpicklingError) as cm:
            torch.load(p)
        self.assertIn("weights_only", str(cm.exception))

    def test_weights_only_false_still_loads_the_real_class(self):
        import pickle
        p = self._path("obj2.pkl")
        with open(p, "wb") as fh:
            pickle.dump({"cfg": _PayloadClass(5)}, fh)
        got = torch.load(p, weights_only=False)
        self.assertIsInstance(got["cfg"], _PayloadClass)
        self.assertEqual(got["cfg"].value, 5)

    def test_unknown_class_is_no_longer_replaced_by_an_empty_type(self):
        # Hand-built pickle referring to a module that does not exist. The old
        # find_class returned `type(name, (), {})`, so the load "succeeded"
        # with an attribute-free placeholder holding none of the saved state.
        import pickle
        p = self._path("ghost.pkl")
        payload = (b"\x80\x04\x95\x00\x00\x00\x00\x00\x00\x00\x00"
                   b"c__jittor_missing_module__\nGhost\n)\x81.")
        with open(p, "wb") as fh:
            fh.write(payload)
        with self.assertRaises(pickle.UnpicklingError) as cm:
            torch.load(p, weights_only=False)
        self.assertIn("Ghost", str(cm.exception))

    def test_map_location_cpu_puts_tensors_on_the_host(self):
        from jittor.compat.torch.types import _var_is_cpu_resident
        p = self._path("m.pkl")
        torch.save({"w": jt.ones((4, 4))}, p)
        got = torch.load(p, map_location="cpu")
        self.assertTrue(_var_is_cpu_resident(got["w"]))

    def test_map_location_cuda_without_a_device_is_an_error(self):
        p = self._path("m2.pkl")
        torch.save({"w": jt.ones((2, 2))}, p)
        if jt.flags.use_cuda:
            self.skipTest("this asserts the CPU-only diagnosis")
        with self.assertRaises(RuntimeError) as cm:
            torch.load(p, map_location="cuda")
        self.assertIn("map_location", str(cm.exception))

    def test_map_location_unsupported_target_is_refused(self):
        p = self._path("m3.pkl")
        torch.save({"w": jt.ones((2, 2))}, p)
        self.assertRefuses(lambda: torch.load(p, map_location="mps"),
                           "map_location")

class _PayloadClass:
    """Module-level so pickle can find it; stands in for a config object."""

    def __init__(self, value):
        self.value = value

class TestDataLoaderWorkers(StubPolicyBase):
    """DataLoader recorded num_workers and then always went single-process."""

    def _loader(self, **kwargs):
        data = torch.utils.data
        items = list(range(16))

        class _DS(data.Dataset):
            def __len__(self):
                return len(items)

            def __getitem__(self, i):
                import threading
                return (items[i], threading.get_ident())

        return data.DataLoader(_DS(), batch_size=4,
                               collate_fn=lambda b: b, **kwargs)

    def test_num_workers_zero_stays_single_process(self):
        loader = self._loader(num_workers=0)
        it = iter(loader)
        self.assertEqual(type(it).__name__, "_SingleProcessDataLoaderIter")

    def test_num_workers_selects_the_worker_iterator(self):
        loader = self._loader(num_workers=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            it = iter(loader)
            self.assertEqual(type(it).__name__, "_MultiProcessingDataLoaderIter")
            list(it)

    def test_workers_actually_run_off_the_main_thread(self):
        import threading
        main = threading.get_ident()
        loader = self._loader(num_workers=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            batches = list(loader)
        threads = {tid for batch in batches for _, tid in batch}
        self.assertTrue(threads - {main},
                        "batches must be prepared off the calling thread")

    def test_worker_batches_arrive_in_order_and_complete(self):
        loader = self._loader(num_workers=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = [v for batch in loader for v, _ in batch]
        self.assertEqual(values, list(range(16)))

    def test_worker_init_fn_is_called(self):
        seen = []
        loader = self._loader(num_workers=2, worker_init_fn=seen.append)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            list(loader)
        self.assertTrue(seen, "worker_init_fn must run in each worker")

    def test_multi_worker_use_warns_about_threads_not_processes(self):
        loader = self._loader(num_workers=2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            list(loader)
        self.assertTrue(any("THREADS" in str(w.message) for w in caught))

if __name__ == "__main__":
    unittest.main()

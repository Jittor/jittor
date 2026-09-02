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


class TestDistributedDataParallel(StubPolicyBase):
    """DDP synchronises for real now (7.02); it is no longer a stub.

    This class used to assert the opposite -- that constructing DDP on more
    than one rank raises -- which was 7.01's holding position while DDP was a
    forwarding wrapper that never all-reduced anything. 7.02 implemented the
    broadcast and the all-reduce, so refusing is no longer the contract.

    The real multi-rank behaviour is checked where it can actually be observed:
    ``tests/compat/torch/test_torch_ddp_grad_sync.py`` runs two ranks under
    ``mpirun`` and compares their parameters. Faking ``jt.world_size`` here
    would only produce a process that believes in ranks that do not exist.
    """

    def test_single_rank_ddp_is_allowed(self):
        model = torch.nn.Linear(3, 2)
        wrapped = torch.nn.parallel.DistributedDataParallel(model)
        out = wrapped(jt.ones((2, 3)))
        self.assertEqual(tuple(out.shape), (2, 2))

    def test_single_rank_ddp_marks_its_parameters_for_synchronisation(self):
        # The marker is how installers/tensor.py finds these gradients without
        # importing DDP; the order is what keeps the collectives in the same
        # sequence on every rank. Both are set even on one rank, where the
        # all-reduce itself is skipped.
        model = torch.nn.Linear(3, 2)
        torch.nn.parallel.DistributedDataParallel(model)
        orders = [getattr(p, "_jittor_ddp_order", None)
                  for p in model.parameters()]
        self.assertEqual(orders, list(range(len(orders))))
        self.assertTrue(all(getattr(p, "_jittor_ddp_state", None) is not None
                            for p in model.parameters()))

    def test_no_sync_is_a_real_switch_not_a_nullcontext(self):
        model = torch.nn.Linear(3, 2)
        wrapped = torch.nn.parallel.DistributedDataParallel(model)
        state = wrapped._jittor_ddp_state
        self.assertTrue(state.sync_enabled)
        with wrapped.no_sync():
            self.assertFalse(state.sync_enabled)
            self.assertFalse(wrapped.require_backward_grad_sync)
        self.assertTrue(state.sync_enabled)
        self.assertTrue(wrapped.require_backward_grad_sync)

    def test_a_build_without_the_collectives_says_so(self):
        # If a multi-rank job ever reaches the collectives on a build that has
        # none, it has to stop and say why -- continuing would train every rank
        # on its own gradients, which is the failure 7.02 exists to remove.
        from jittor.compat import collectives

        class NoCollectives:
            pass

        saved = collectives._world_size
        collectives._world_size = lambda: 4
        try:
            for call in (collectives._all_reduce_mean,
                         collectives._broadcast_from_rank0):
                with self.subTest(call=call.__name__):
                    with self.assertRaises(RuntimeError) as caught:
                        call(NoCollectives())
                    self.assertIn("4 ranks", str(caught.exception))
        finally:
            collectives._world_size = saved


class TestBackwardGradient(StubPolicyBase):
    """Tensor.backward(gradient=...) dropped its argument."""

    def test_gradient_weights_the_backward_pass(self):
        x = jt.array(np.arange(4, dtype="float32"))
        x.requires_grad = True
        y = x * x                       # dy/dx = 2x
        weights = jt.array(np.array([1.0, 2.0, 3.0, 4.0], dtype="float32"))
        y.backward(gradient=weights)
        expect = 2 * np.arange(4, dtype="float32") * np.array([1., 2., 3., 4.])
        np.testing.assert_allclose(x.grad.numpy(), expect, rtol=1e-5)

    def test_unweighted_backward_is_unchanged(self):
        x = jt.array(np.arange(4, dtype="float32"))
        x.requires_grad = True
        y = x * x
        y.backward()
        np.testing.assert_allclose(x.grad.numpy(),
                                   2 * np.arange(4, dtype="float32"), rtol=1e-5)

    def test_gradient_of_the_wrong_shape_is_rejected(self):
        x = jt.array(np.arange(4, dtype="float32"))
        x.requires_grad = True
        y = x * x
        with self.assertRaises(RuntimeError):
            y.backward(gradient=jt.ones((3, 5, 7)))


class TestTreeMap(StubPolicyBase):
    """torch.utils._pytree.tree_map did not recurse."""

    def _pytree(self):
        import sys
        return sys.modules["torch.utils._pytree"]

    def test_tree_map_recurses_into_containers(self):
        pytree = self._pytree()
        tree = {"a": [1, 2], "b": (3, {"c": 4})}
        got = pytree.tree_map(lambda v: v * 10, tree)
        self.assertEqual(got, {"a": [10, 20], "b": (30, {"c": 40})})

    def test_tree_map_only_moves_nested_tensors(self):
        pytree = self._pytree()
        batch = {"x": jt.ones((2,)), "meta": ["keep", 3]}
        got = pytree.tree_map_only(jt.Var, lambda t: t + 1, batch)
        np.testing.assert_allclose(got["x"].numpy(), np.full((2,), 2.0))
        self.assertEqual(got["meta"], ["keep", 3])

    def test_tree_map_preserves_the_structure(self):
        pytree = self._pytree()
        tree = [[1], [2, [3]]]
        self.assertEqual(pytree.tree_map(lambda v: v, tree), tree)


class TestSummaryWriter(StubPolicyBase):
    """Every SummaryWriter method returned None and wrote nothing."""

    def _has_real_writer(self):
        try:
            import tensorboardX  # noqa: F401
            return True
        except Exception:
            return False

    def test_writer_without_tensorboard_is_refused(self):
        if self._has_real_writer():
            self.skipTest("tensorboardX is installed; the writer is real")
        import sys
        SummaryWriter = sys.modules["torch.utils.tensorboard"].SummaryWriter
        self.assertRefuses(lambda: SummaryWriter(log_dir="/tmp/jt-tb"),
                           "SummaryWriter")

    def test_stub_fallback_restores_the_silent_writer(self):
        if self._has_real_writer():
            self.skipTest("tensorboardX is installed; the writer is real")
        import sys
        SummaryWriter = sys.modules["torch.utils.tensorboard"].SummaryWriter
        writer = self.assertStubFallback(lambda: SummaryWriter(log_dir="/tmp/jt-tb"))
        self.assertIsNone(writer.add_scalar("loss", 1.0, 0))


class TestInitAndSwa(StubPolicyBase):
    """nn.init.dirac_/sparse_ and swa_utils.update_bn were identity/no-op."""

    def test_dirac_builds_an_identity_kernel(self):
        w = jt.zeros((4, 4, 3, 3))
        torch.nn.init.dirac_(w)
        arr = w.numpy()
        self.assertEqual(arr.sum(), 4.0)
        for c in range(4):
            self.assertEqual(arr[c, c, 1, 1], 1.0)

    def test_dirac_preserves_an_identity_signal(self):
        w = jt.zeros((2, 2, 3, 3))
        torch.nn.init.dirac_(w)
        x = jt.array(np.random.RandomState(0).randn(1, 2, 5, 5).astype("float32"))
        y = jt.nn.conv2d(x, w, padding=1)
        np.testing.assert_allclose(y.numpy(), x.numpy(), atol=1e-5)

    def test_dirac_rejects_a_2d_tensor(self):
        with self.assertRaises(ValueError):
            torch.nn.init.dirac_(jt.zeros((4, 4)))

    def test_sparse_actually_zeroes_rows(self):
        w = jt.zeros((10, 4))
        torch.nn.init.sparse_(w, sparsity=0.5)
        arr = w.numpy()
        for col in range(4):
            self.assertEqual(int((arr[:, col] == 0).sum()), 5)
        self.assertNotEqual(float(np.abs(arr).sum()), 0.0)

    def test_update_bn_recomputes_running_statistics(self):
        import sys
        swa = sys.modules["torch.optim.swa_utils"]
        model = torch.nn.BatchNorm(4)
        model.running_mean.assign(jt.ones(4) * 99.0)
        batches = [jt.ones((8, 4)) * 5.0 for _ in range(3)]
        swa.update_bn(batches, model)
        np.testing.assert_allclose(model.running_mean.numpy(),
                                   np.full(4, 5.0), atol=1e-4)


class TestOverridesAndDefaults(StubPolicyBase):
    """has_torch_function was constantly False; set_default_device did nothing."""

    def setUp(self):
        super().setUp()
        self._use_cuda = jt.flags.use_cuda

    def tearDown(self):
        jt.flags.use_cuda = self._use_cuda
        super().tearDown()

    def test_has_torch_function_is_false_for_plain_vars(self):
        import sys
        overrides = sys.modules["torch.overrides"]
        self.assertFalse(overrides.has_torch_function((jt.ones(2),)))

    def test_has_torch_function_sees_a_subclass_override(self):
        import sys
        overrides = sys.modules["torch.overrides"]

        class _Sub:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return "intercepted"

        self.assertTrue(overrides.has_torch_function((_Sub(),)))

    def test_handle_torch_function_calls_the_override(self):
        import sys
        overrides = sys.modules["torch.overrides"]

        class _Sub:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return "intercepted"

        got = overrides.handle_torch_function(lambda *a, **k: "plain", [_Sub()])
        self.assertEqual(got, "intercepted")

    def test_torch_function_mode_is_refused(self):
        import sys
        overrides = sys.modules["torch.overrides"]

        def _enter():
            with overrides.TorchFunctionMode():
                pass

        self.assertRefuses(_enter, "TorchFunctionMode")

    def test_set_default_device_cpu_agrees_with_get(self):
        torch.set_default_device("cpu")
        self.assertEqual(str(torch.get_default_device()), "cpu")

    def test_set_default_device_cuda_agrees_with_get(self):
        if not jt.has_cuda:
            self.skipTest("no accelerator on this box")
        torch.set_default_device("cuda")
        self.assertIn("cuda", str(torch.get_default_device()))

    def test_set_default_device_non_zero_index_is_refused(self):
        if not jt.has_cuda:
            self.skipTest("no accelerator on this box")
        self.assertRefuses(lambda: torch.set_default_device("cuda:1"),
                           "set_default_device")

    def test_set_default_device_unknown_backend_is_refused(self):
        self.assertRefuses(lambda: torch.set_default_device("mps"),
                           "set_default_device")


class TestCudaDeviceAndEvents(StubPolicyBase):
    """set_device was a no-op; Event.elapsed_time was a constant 0.0."""

    def test_set_device_zero_is_accepted(self):
        self.assertIsNone(torch.cuda.set_device(0))

    def test_set_device_non_zero_is_refused(self):
        self.assertRefuses(lambda: torch.cuda.set_device(1),
                           "torch.cuda.set_device", "device 0")

    def test_set_device_non_zero_stub_fallback(self):
        self.assertStubFallback(lambda: torch.cuda.set_device(3))

    def test_event_elapsed_time_measures_something(self):
        import time
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        time.sleep(0.02)
        end.record()
        elapsed = start.elapsed_time(end)
        self.assertGreater(elapsed, 5.0,
                           "elapsed_time used to be a constant 0.0")

    def test_event_without_enable_timing_refuses_to_time(self):
        start = torch.cuda.Event()
        end = torch.cuda.Event()
        start.record()
        end.record()
        with self.assertRaises(RuntimeError):
            start.elapsed_time(end)

    def test_unrecorded_event_refuses_to_time(self):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with self.assertRaises(RuntimeError):
            start.elapsed_time(end)


class TestLibraryOpcheck(StubPolicyBase):
    def test_opcheck_is_refused(self):
        import sys
        library = sys.modules["torch.library"]
        self.assertRefuses(lambda: library.opcheck(None, ()),
                           "torch.library.opcheck")

    def test_opcheck_stub_fallback_returns_none(self):
        import sys
        library = sys.modules["torch.library"]
        self.assertIsNone(self.assertStubFallback(
            lambda: library.opcheck(None, ())))


class TestDistributedStubs(StubPolicyBase):
    """dcp save/load, subgroups, stores and DTensor were identities."""

    def _dist(self):
        import sys
        return sys.modules["torch.distributed"]

    def test_dcp_save_is_refused(self):
        import sys
        dcp = sys.modules["torch.distributed.checkpoint"]
        self.assertRefuses(lambda: dcp.save({"w": jt.ones(2)}),
                           "checkpoint.save", "discarding the checkpoint")

    def test_dcp_load_is_refused(self):
        import sys
        dcp = sys.modules["torch.distributed.checkpoint"]
        self.assertRefuses(lambda: dcp.load({"w": jt.ones(2)}),
                           "checkpoint.load")

    def test_dcp_save_stub_fallback(self):
        import sys
        dcp = sys.modules["torch.distributed.checkpoint"]
        self.assertStubFallback(lambda: dcp.save({"w": jt.ones(2)}))

    def test_single_rank_store_still_works(self):
        # One rank has nobody to meet, so a per-process dict IS a correct store.
        dist = self._dist()
        store = dist.TCPStore()
        store.set("step", b"1")
        self.assertEqual(store.get("step"), b"1")

    def test_multi_rank_tcp_store_is_refused(self):
        dist = self._dist()
        self.assertRefuses(
            lambda: dist.TCPStore("127.0.0.1", 29500, 2, True),
            "TCPStore", "dictionary")

    def test_multi_rank_file_store_is_refused(self):
        dist = self._dist()
        self.assertRefuses(lambda: dist.FileStore("/tmp/jt-store", 2),
                           "FileStore")

    def test_whole_world_subgroup_enumeration_is_allowed(self):
        dist = self._dist()
        groups, cur = dist.new_subgroups_by_enumeration([[0]])
        self.assertIs(cur, dist.group.WORLD)

    def test_partitioning_subgroup_enumeration_is_refused(self):
        dist = self._dist()
        saved = getattr(jt, "world_size", 1)
        jt.world_size = 4
        os.environ.setdefault("JT_NCCL_WORLD_SIZE", "4")
        try:
            self.assertRefuses(
                lambda: dist.new_subgroups_by_enumeration([[0, 1], [2, 3]]),
                "new_subgroups_by_enumeration")
        finally:
            jt.world_size = saved
            os.environ.pop("JT_NCCL_WORLD_SIZE", None)

    def test_single_rank_autograd_all_reduce_is_the_identity(self):
        dist = self._dist()
        x = jt.ones((3,))
        np.testing.assert_allclose(dist.nn.all_reduce(x).numpy(), x.numpy())


class TestDeviceMeshAndDTensor(StubPolicyBase):
    def _mesh_mod(self):
        from jittor.compat.fsdp2 import dtensor
        return dtensor

    def test_one_dimensional_mesh_indexing_still_works(self):
        dtensor = self._mesh_mod()
        mesh = dtensor.DeviceMesh("cpu", (1,))
        self.assertIs(mesh["dp"], mesh)

    def test_two_dimensional_mesh_on_one_rank_is_harmless(self):
        # Every collective on a one-rank world is a no-op, so a collapsed mesh
        # cannot send anything to the wrong ranks.
        dtensor = self._mesh_mod()
        mesh = dtensor.DeviceMesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        self.assertIs(mesh["dp"], mesh)

    def test_two_dimensional_mesh_axis_selection_is_refused(self):
        dtensor = self._mesh_mod()
        saved = getattr(jt, "world_size", 1)
        jt.world_size = 4
        try:
            mesh = dtensor.DeviceMesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
            self.assertRefuses(lambda: mesh["dp"], "DeviceMesh")
        finally:
            jt.world_size = saved

    def test_two_dimensional_mesh_get_group_is_refused(self):
        dtensor = self._mesh_mod()
        saved = getattr(jt, "world_size", 1)
        jt.world_size = 4
        try:
            mesh = dtensor.DeviceMesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
            self.assertRefuses(lambda: mesh.get_group(), "DeviceMesh.get_group")
        finally:
            jt.world_size = saved

    def test_full_tensor_on_one_rank_returns_the_tensor(self):
        dtensor = self._mesh_mod()
        local = jt.ones((2, 2))
        dt = dtensor.DTensor(local, dtensor.DeviceMesh("cpu", (1,)),
                             (dtensor.Shard(0),))
        np.testing.assert_allclose(dt.full_tensor().numpy(), local.numpy())

    def test_sharded_full_tensor_on_many_ranks_is_refused(self):
        dtensor = self._mesh_mod()
        saved = getattr(jt, "world_size", 1)
        jt.world_size = 4
        try:
            dt = dtensor.DTensor(jt.ones((2, 2)),
                                 dtensor.DeviceMesh("cpu", (4,)),
                                 (dtensor.Shard(0),))
            self.assertRefuses(lambda: dt.full_tensor(),
                               "DTensor.full_tensor", "LOCAL SHARD")
        finally:
            jt.world_size = saved

    def test_replicated_full_tensor_on_many_ranks_is_exact(self):
        dtensor = self._mesh_mod()
        saved = getattr(jt, "world_size", 1)
        jt.world_size = 4
        try:
            local = jt.ones((2, 2))
            dt = dtensor.DTensor(local, dtensor.DeviceMesh("cpu", (4,)),
                                 (dtensor.Replicate(),))
            np.testing.assert_allclose(dt.full_tensor().numpy(), local.numpy())
        finally:
            jt.world_size = saved


class TestGeneratedUnimplementedList(StubPolicyBase):
    """The plan asks tests/compat/torch to publish the list of gaps."""

    def test_every_declared_gap_states_its_consequence(self):
        for api, info in torch.compat_unimplemented_apis().items():
            self.assertTrue(info["effect"],
                            "%s is refused without saying what breaks" % api)

    def test_the_list_is_rendered_for_humans(self):
        # Renders whatever this process has touched; the printout is the
        # "auto-generated unimplemented API list" the plan asks for.
        lines = ["# torch APIs Jittor refuses to fake",
                 "",
                 "| API | what a silent stub would do |",
                 "| --- | --- |"]
        for api, info in sorted(torch.compat_unimplemented_apis().items()):
            lines.append("| `%s` | %s |" % (api, info["effect"]))
        approximate = torch.compat_approximate_apis()
        if approximate:
            lines += ["", "# torch APIs Jittor approximates", "",
                      "| API | how it differs |", "| --- | --- |"]
            for api, info in sorted(approximate.items()):
                lines.append("| `%s` | %s |" % (api, info["effect"]))
        rendered = "\n".join(lines)
        print("\n" + rendered)
        self.assertIn("| API |", rendered)


if __name__ == "__main__":
    unittest.main()

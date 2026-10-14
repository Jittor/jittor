"""The torch frontend's native fast paths answer as the Python paths do.

`src/bindings/pyjt/py_compat_fast.h` and `py_module_call.h`: `torch.cat`, a
basic `tensor[index]`, `Tensor.dtype`, the arithmetic operators, `view`,
`reshape`, `unsqueeze`, `transpose` and `module(...)` -- with a bias-free
`nn.Linear` and a standard RMS norm computed directly -- built without the
Python frames between the call and the ops. Each takes only the case it
recognises and hands anything else back, so both halves are checked here:
values, dtypes, views and gradients where the fast path answers, and the
cases it must leave alone.
"""

import contextlib
import unittest
from unittest import mock

import numpy as np
import torch

import jittor as jt


@contextlib.contextmanager
def _refusing(module, name):
    """`module.name` raising for the duration: the path under test must not
    reach it."""
    original = getattr(module, name)

    def refuse(*args, **kwargs):
        raise AssertionError("%s.%s was reached" % (module.__name__, name))

    setattr(module, name, refuse)
    try:
        yield
    finally:
        setattr(module, name, original)


def _cuda():
    return "cuda" if torch.cuda.is_available() else "cpu"


#: The CUDA kernels these paths build. `torch.cuda.is_available()` is also
#: true for an NPU under the torch frontend, where ACL's own kernels serve.
_CUDA_KERNELS = torch.cuda.is_available() and bool(getattr(jt.compiler, "is_cuda", False))


class TestFastCat(unittest.TestCase):
    def test_values_dims_and_dtypes(self):
        device = _cuda()
        rng = np.random.RandomState(0)
        for dtype, numpy_dtype in ((torch.float32, np.float32), (torch.float16, np.float16),
                                   (torch.int64, np.int64), (torch.bool, np.bool_)):
            for dim in (0, 1, -1):
                with self.subTest(dtype=dtype, dim=dim):
                    a_np = rng.rand(2, 3, 4) > 0.5 if dtype is torch.bool else rng.rand(2, 3, 4) * 10
                    b_np = rng.rand(2, 3, 4) > 0.5 if dtype is torch.bool else rng.rand(2, 3, 4) * 10
                    # Convert NumPy's default float64 on the host: Ascend310P
                    # does not support a device float64-to-target cast.
                    a_np, b_np = a_np.astype(numpy_dtype), b_np.astype(numpy_dtype)
                    a = torch.tensor(a_np, dtype=dtype, device=device)
                    b = torch.tensor(b_np, dtype=dtype, device=device)
                    c = torch.cat([a, b], dim=dim)
                    self.assertIsInstance(c, torch.Tensor)
                    self.assertEqual(c.dtype, dtype)
                    np.testing.assert_array_equal(
                        c.numpy(), np.concatenate([a.numpy(), b.numpy()], axis=dim))

    def test_the_cases_it_hands_back(self):
        a = torch.ones(2, 3, dtype=torch.uint8)
        b = torch.zeros(2, 3, dtype=torch.uint8)
        self.assertIs(jt.core._fast_cat([a, b], 0), NotImplemented)
        self.assertEqual(torch.cat([a, b]).dtype, torch.uint8)
        f = torch.ones(2, 3)
        h = torch.ones(2, 3, dtype=torch.float16)
        self.assertIs(jt.core._fast_cat([f, h], 0), NotImplemented)
        self.assertEqual(torch.cat([f, h]).dtype, torch.float32)
        empty = torch.ones(0, 3)
        self.assertIs(jt.core._fast_cat([f, empty], 0), NotImplemented)
        self.assertEqual(tuple(torch.cat([f, empty]).shape), (2, 3))
        self.assertIs(jt.core._fast_cat([f], 0), NotImplemented)

    def test_gradients_reach_every_input(self):
        x = torch.randn(3, 2, requires_grad=True)
        y = torch.randn(4, 2, requires_grad=True)
        w = torch.randn(7, 2)
        (torch.cat([x, y]) * w).sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), w.numpy()[:3])
        np.testing.assert_allclose(y.grad.numpy(), w.numpy()[3:])


class TestFastCatRegisteredBackend(unittest.TestCase):
    def setUp(self):
        from jittor.ops import concatenation
        from jittor._runtime import dispatch
        from jittor.compat.torch.installers.tensor.methods import _bind_native_fast_paths
        self.concat, self.dispatch = concatenation, dispatch
        self.bind = _bind_native_fast_paths
        self.flags = jt.flag_scope(use_cuda=0, amp_reg=0)
        self.flags.__enter__()
        self.addCleanup(self.flags.__exit__, None, None, None)
        self.bind(jt)
        self.a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device="cpu")
        self.b = torch.tensor([[7., 8., 9.]], device="cpu")
        self.expected = np.arange(1, 10, dtype="float32").reshape(3, 3)

    def tearDown(self):
        self.bind(jt)

    def _continue(self, inputs, dim):
        dtype = str(inputs[0].dtype)
        if dtype.startswith("torch."):
            dtype = dtype[6:]
        return self.concat._concat_after_kernel(inputs, dim, dtype)

    def test_selected_override_called_once_in_frontend_and_amp_scope(self):
        seen = []

        def supports(inputs, dim):
            seen.append(("supports", type(inputs), dim, jt.flags.amp_reg))
            return True

        def provider(inputs, dim):
            seen.append(("provider", type(inputs), dim, jt.flags.amp_reg))
            return self._continue(inputs, dim)

        original_amp = jt.amp_flags.prefer32 | jt.amp_flags.keep_white | jt.amp_flags.array_prefer
        with jt.flag_scope(amp_reg=original_amp), self.dispatch.override_kernel(
                "tensor.concat", "cpu", provider, supports=supports, dtypes=("float32",)):
            with _refusing(self.concat, "_concat"):
                result = torch.cat((self.a, self.b), dim=-2)
            self.assertEqual(jt.flags.amp_reg, original_amp)
        self.assertIsInstance(result, torch.Tensor)
        np.testing.assert_array_equal(result.numpy(), self.expected)
        scoped_amp = original_amp | jt.amp_flags.keep_reduce
        self.assertEqual(seen, [("supports", list, 0, scoped_amp), ("provider", tuple, 0, scoped_amp)])

    def test_declining_provider_resumes_once_and_keeps_gradients(self):
        seen = []

        def decline(inputs, dim):
            seen.append((inputs, dim))
            return None

        self.a.requires_grad_(True)
        self.b.requires_grad_(True)
        with self.dispatch.override_kernel("tensor.concat", "cpu", decline):
            with _refusing(self.concat, "_concat"):
                result = torch.cat([self.a, self.b], dim=0)
        self.assertEqual(len(seen), 1)
        self.assertIsInstance(result, torch.Tensor)
        np.testing.assert_array_equal(result.numpy(), self.expected)
        result.sum().backward()
        np.testing.assert_array_equal(self.a.grad.numpy(), np.ones((2, 3)))
        np.testing.assert_array_equal(self.b.grad.numpy(), np.ones((1, 3)))

    def test_ineligible_provider_predicate_is_not_repeated(self):
        seen = []

        def supports(inputs, dim):
            seen.append(dim)
            return False

        def unexpected(*args):
            raise AssertionError("ineligible provider executed")

        with self.dispatch.override_kernel("tensor.concat", "cpu", unexpected, supports=supports):
            with _refusing(self.concat, "_concat"):
                result = torch.cat([self.a, self.b], dim=0)
        self.assertEqual(seen, [0])
        np.testing.assert_array_equal(result.numpy(), self.expected)

    def test_decline_preserves_the_tracing_continuation(self):
        seen = []

        def decline(inputs, dim):
            seen.append(dim)
            return None

        with self.dispatch.override_kernel("tensor.concat", "cpu", decline), mock.patch.object(
                self.concat, "_tracing", return_value=True), mock.patch.object(
                self.concat, "_concat_fused", wraps=self.concat._concat_fused) as fused:
            with _refusing(self.concat, "_concat"):
                result = torch.cat([self.a, self.b])
            self.assertEqual(fused.call_count, 1)
        self.assertEqual(seen, [0])
        np.testing.assert_array_equal(result.numpy(), self.expected)

    def test_provider_and_predicate_errors_restore_amp_and_placement(self):
        class ProviderError(ValueError):
            pass

        original_amp = jt.amp_flags.prefer32 | jt.amp_flags.keep_white
        for fail_in in ("provider", "predicate"):
            seen = []

            def failure(*args):
                seen.append(jt.flags.amp_reg)
                raise ProviderError("concat callback failed")

            provider = failure if fail_in == "provider" else lambda *args: None
            supports = failure if fail_in == "predicate" else None
            with self.subTest(fail_in=fail_in), jt.flag_scope(amp_reg=original_amp):
                placement = jt.core._current_tensor_placement()
                with self.dispatch.override_kernel("tensor.concat", "cpu", provider, supports=supports):
                    with self.assertRaisesRegex(ProviderError, "concat callback failed"):
                        torch.cat([self.a, self.b])
                self.assertEqual(jt.flags.amp_reg, original_amp)
                self.assertEqual(jt.core._current_tensor_placement(), placement)
                self.assertEqual(seen, [original_amp | jt.amp_flags.keep_reduce])

    def test_amp_conversion_keeps_the_existing_python_route(self):
        seen = []

        def provider(inputs, dim):
            seen.append([value.dtype for value in inputs])
            return self._continue(inputs, dim)

        with jt.flag_scope(amp_reg=jt.amp_flags.prefer16), self.dispatch.override_kernel(
                "tensor.concat", "cpu", provider, dtypes=("float32",)):
            self.assertIs(jt.core._fast_cat([self.a, self.b], 0), NotImplemented)
            result = torch.cat([self.a, self.b])
            self.assertEqual(jt.flags.amp_reg, jt.amp_flags.prefer16)
        self.assertEqual(seen, [[torch.float16, torch.float16]])
        self.assertEqual(result.dtype, torch.float32)
        np.testing.assert_array_equal(result.numpy(), self.expected)

    def test_provider_result_retains_torch_common_input_dtype(self):
        calls = []

        def provider(inputs, dim):
            calls.append(dim)
            return self._continue(inputs, dim).double()

        with self.dispatch.override_kernel("tensor.concat", "cpu", provider):
            with _refusing(self.concat, "_concat"):
                result = torch.cat([self.a, self.b])
        self.assertEqual(calls, [0])
        self.assertEqual(result.dtype, torch.float32)
        np.testing.assert_array_equal(result.numpy(), self.expected)

    def test_python_selection_overrides_leave_the_fast_path(self):
        calls = []

        def provider(inputs, dim):
            calls.append(dim)
            return self._continue(inputs, dim)

        with mock.patch.object(self.concat, "select_kernel", return_value=provider) as selector:
            self.assertIs(jt.core._fast_cat([self.a, self.b], 0), NotImplemented)
            result = torch.cat([self.a, self.b])
            self.assertEqual(selector.call_count, 1)
        self.assertEqual(calls, [0])
        np.testing.assert_array_equal(result.numpy(), self.expected)
        calls.clear()
        original = self.dispatch.dispatch_context
        with self.dispatch.override_kernel("tensor.concat", "cpu", provider), mock.patch.object(
                self.dispatch, "dispatch_context", side_effect=original) as context:
            self.assertIs(jt.core._fast_cat([self.a, self.b], 0), NotImplemented)
            result = torch.cat([self.a, self.b])
            # The continuation can also query placement for its setitem ops;
            # only one query belongs to the concat input sequence.
            self.assertEqual(sum(isinstance(call[0][0], list)
                                 for call in context.call_args_list), 1)
        self.assertEqual(calls, [0])
        np.testing.assert_array_equal(result.numpy(), self.expected)

    def test_legacy_binding_and_unbound_selector_use_the_python_path(self):
        from jittor._runtime.step_capture import tracing
        calls = []

        def provider(inputs, dim):
            calls.append(dim)
            return self._continue(inputs, dim)

        with self.dispatch.override_kernel("tensor.concat", "cpu", provider):
            jt.core._compat_fast_bind(tracing)
            self.assertIs(jt.core._fast_cat([self.a, self.b], 0), NotImplemented)
            self.bind(jt)
            with mock.patch.object(self.dispatch, "_NATIVE_SELECT", False):
                self.assertIs(jt.core._fast_cat([self.a, self.b], 0), NotImplemented)
                result = torch.cat([self.a, self.b])
        self.assertEqual(calls, [0])
        np.testing.assert_array_equal(result.numpy(), self.expected)


class TestFastGetitem(unittest.TestCase):
    def test_basic_indices(self):
        a = torch.randn(4, 5, 6, device=_cuda())
        n = a.numpy()
        for index in (1, -1, slice(1, 3), (Ellipsis, slice(None, 2)), (slice(None), None, 2),
                      (0, Ellipsis, None), (slice(None, None, 2), 1)):
            with self.subTest(index=index):
                np.testing.assert_array_equal(a[index].numpy(), n[index])

    def test_a_view_writes_through(self):
        a = torch.zeros(3, 4)
        row = a[1]
        row.add_(2.0)
        self.assertEqual(float(a.sum()), 8.0)

    def test_a_data_view_writes_through_its_index(self):
        x = torch.tensor([1., 2.], device=_cuda(), requires_grad=True)
        data = x.data
        self.assertIs(jt.core._fast_getitem(data, 0), NotImplemented)
        with torch.no_grad():
            data[0].fill_(3)
        np.testing.assert_allclose(x.numpy(), [3., 2.])

    def test_advanced_and_boolean_indices_take_the_python_path(self):
        a = torch.arange(12).reshape(3, 4)
        index = torch.tensor([2, 0])
        self.assertIs(jt.core._fast_getitem(a, index), NotImplemented)
        np.testing.assert_array_equal(a[index].numpy(), a.numpy()[[2, 0]])
        self.assertIs(jt.core._fast_getitem(a, (True,)), NotImplemented)

    def test_gradient(self):
        x = torch.randn(3, 4, requires_grad=True)
        x[1:, :2].sum().backward()
        expected = np.zeros((3, 4), "float32")
        expected[1:, :2] = 1
        np.testing.assert_array_equal(x.grad.numpy(), expected)


class TestFastGetitemRegisteredBackend(unittest.TestCase):
    def setUp(self):
        from jittor._runtime import dispatch
        from jittor.ops import indexing
        from jittor.compat.torch.frontend import tensor_frontend
        from jittor.compat.torch.installers.tensor import method_api
        from jittor.compat.torch.installers.tensor.methods import _bind_native_fast_paths
        self.dispatch, self.indexing = dispatch, indexing
        self.frontend, self.methods, self.bind = tensor_frontend, method_api, _bind_native_fast_paths
        self.flags = jt.flag_scope(use_cuda=0, no_grad=1, amp_reg=0, auto_graph_replay=0)
        self.flags.__enter__()
        self.addCleanup(self.flags.__exit__, None, None, None)
        self.bind(jt)
        self.values = np.arange(24, dtype="float32").reshape(4, 6)
        # No explicit CPU-residency hint: that hint deliberately retains the
        # Python path, even when the provider itself supports CPU execution.
        self.x = jt.array(self.values)
        self.assertLess(self.x.placement_backend, 0)
        self.index = (slice(1, 4, 2), slice(None, None, 2))

    def tearDown(self):
        self.bind(jt)

    def _get(self, value, index, slow=False):
        guard = (mock.patch.object(jt.core, "_fast_getitem", return_value=NotImplemented)
                 if slow else _refusing(self.indexing, "_getitem_result"))
        with guard, self.frontend(torch.Tensor, like=value, default_placement=False):
            return self.methods._torch_getitem(value, index)

    def test_selected_override_runs_once_with_normalized_index_and_frontend(self):
        seen = []

        def supports(value, index, return_x):
            seen.append(("supports", index, index is self.index, return_x))
            return True

        def provider(value, index, return_x):
            seen.append(("provider", index, index is self.index, return_x))
            result = jt.core.ops.getitem(value, index)
            self.assertIsInstance(result, torch.Tensor)
            return result

        with self.dispatch.override_kernel("tensor.getitem", "cpu", provider,
                                           supports=supports, dtypes=("float32",)):
            result = self._get(self.x, self.index)
        self.assertEqual(seen, [("supports", self.index, False, None),
                                ("provider", self.index, False, None)])
        self.assertTrue(result._is_view())
        np.testing.assert_array_equal(result.numpy(), self.values[self.index])

    def test_decline_is_once_on_fast_and_slow_paths_and_views_write_to_root(self):
        for slow in (False, True):
            with self.subTest(slow=slow):
                root = jt.array(self.values)
                seen = []

                def decline(*args):
                    seen.append(args[1])
                    return None

                # Check native view flattening without a frontend scope: a
                # Torch result on CPU acquires explicit CPU placement, which
                # intentionally makes its next indexing call use Python.
                getter = self.indexing.getitem if slow else jt.core._fast_getitem
                with self.dispatch.override_kernel("tensor.getitem", "cpu", decline):
                    view = getter(root, self.index)
                    self.assertIsNot(view, NotImplemented)
                    nested = getter(view, (slice(None), slice(1, 2)))
                    self.assertIsNot(nested, NotImplemented)
                self.assertEqual(len(seen), 2)
                np.testing.assert_array_equal(view.numpy(), self.values[self.index])
                self.assertEqual(nested._view_base_id(), view._view_base_id())
                nested.assign(nested + 100)
                expected = self.values.copy()
                expected[1:4:2, 2:3] += 100
                np.testing.assert_array_equal(root.numpy(), expected)

    def test_provider_existing_alternate_view_and_plain_results_match_slow_path(self):
        for slow in (False, True):
            for recorded in (False, True):
                with self.subTest(slow=slow, recorded=recorded):
                    source = jt.array(self.values)
                    alternate = jt.array(self.values + 100)
                    provider_result = jt.core.ops.getitem(
                        alternate if recorded else source, self.index)
                    if recorded:
                        provider_result._set_view_of(alternate, self.index)
                    original_base = provider_result._view_base_id()
                    calls = []

                    def provider(*args):
                        calls.append(args)
                        return provider_result

                    with self.dispatch.override_kernel("tensor.getitem", "cpu", provider):
                        result = self._get(source, self.index, slow)
                    self.assertIs(result, provider_result)
                    self.assertEqual(len(calls), 1)
                    self.assertTrue(result._is_view())
                    if recorded:
                        self.assertEqual(result._view_base_id(), original_base)
                    result.assign(result + 7)
                    expected_source, expected_alternate = self.values.copy(), self.values + 100
                    (expected_alternate if recorded else expected_source)[self.index] += 7
                    np.testing.assert_array_equal(source.numpy(), expected_source)
                    np.testing.assert_array_equal(alternate.numpy(), expected_alternate)

    def test_ineligible_provider_predicate_is_once_on_both_paths(self):
        for slow in (False, True):
            seen = []

            def supports(*args):
                seen.append(args[1])
                return False

            def unexpected(*args):
                raise AssertionError("ineligible provider executed")

            with self.subTest(slow=slow), self.dispatch.override_kernel(
                    "tensor.getitem", "cpu", unexpected, supports=supports):
                result = self._get(self.x, self.index, slow)
            self.assertEqual(seen, [self.index])
            np.testing.assert_array_equal(result.numpy(), self.values[self.index])

    def test_provider_and_predicate_errors_propagate_and_restore_context(self):
        class ProviderError(ValueError):
            pass

        for fail_in in ("provider", "predicate"):
            calls = []

            def fail(*args):
                calls.append(args)
                raise ProviderError("getitem callback failed")

            provider = fail if fail_in == "provider" else lambda *args: None
            supports = fail if fail_in == "predicate" else None
            with self.subTest(fail_in=fail_in), self.dispatch.override_kernel(
                    "tensor.getitem", "cpu", provider, supports=supports):
                placement = jt.core._current_tensor_placement()
                with self.assertRaisesRegex(ProviderError, "getitem callback failed"):
                    self._get(self.x, self.index)
                self.assertEqual(jt.core._current_tensor_placement(), placement)
                self.assertEqual(jt.flags.amp_reg, 0)
                self.assertEqual(len(calls), 1)

    def test_python_dispatch_overrides_leave_native_shortcut(self):
        calls = []

        def provider(value, index, return_x):
            calls.append(index)
            return jt.core.ops.getitem(value, index)

        overrides = (
            (self.indexing, "try_dispatch", lambda _op, *args: provider(*args)),
            (self.dispatch, "select_kernel", lambda *args, **kwargs: provider),
            (self.dispatch, "dispatch_context", self.dispatch.dispatch_context),
        )
        for owner, name, replacement in overrides:
            calls.clear()
            with self.subTest(name=name), self.dispatch.override_kernel(
                    "tensor.getitem", "cpu", provider), mock.patch.object(
                    owner, name, side_effect=replacement) as callback:
                self.assertIs(jt.core._fast_getitem(self.x, self.index), NotImplemented)
                result = self._get(self.x, self.index, slow=True)
                self.assertGreaterEqual(callback.call_count, 1)
            self.assertEqual(calls, [self.index])
            np.testing.assert_array_equal(result.numpy(), self.values[self.index])

    def test_legacy_binding_and_disabled_native_selector_keep_python_route(self):
        from jittor._runtime.step_capture import tracing
        seen = []

        def provider(value, index, return_x):
            seen.append(index)
            return jt.core.ops.getitem(value, index)

        with self.dispatch.override_kernel("tensor.getitem", "cpu", provider):
            jt.core._compat_fast_bind(tracing)
            self.assertIs(jt.core._fast_getitem(self.x, self.index), NotImplemented)
            self.bind(jt)
            with mock.patch.object(self.dispatch, "_NATIVE_SELECT", False):
                self.assertIs(jt.core._fast_getitem(self.x, self.index), NotImplemented)
                result = self._get(self.x, self.index, slow=True)
        self.assertEqual(seen, [self.index])
        np.testing.assert_array_equal(result.numpy(), self.values[self.index])

    def test_unsupported_spellings_training_and_owner_hints_keep_python_route(self):
        class DynamicBound:
            def __index__(self):
                return 1

        with self.dispatch.override_kernel("tensor.getitem", "cpu", lambda *args: None):
            for index in (True, (True,), slice(None, None, -1), slice(None, None, 0),
                          slice(DynamicBound(), None), slice(np.int64(1), None), [1, 2]):
                with self.subTest(index=index):
                    self.assertIs(jt.core._fast_getitem(self.x, index), NotImplemented)
            with jt.flag_scope(no_grad=0):
                self.assertIs(jt.core._fast_getitem(self.x, self.index), NotImplemented)
            for attribute, value in (("_torch_data_owner", self.x),
                                     ("_jittor_torch_force_cpu", True)):
                hinted = jt.array(self.values)
                setattr(hinted, attribute, value)
                self.assertIs(jt.core._fast_getitem(hinted, self.index), NotImplemented)
            explicit_cpu = torch.tensor(self.values, device="cpu")
            self.assertIs(jt.core._fast_getitem(explicit_cpu, self.index), NotImplemented)
            self.assertIs(jt.core._fast_getitem(explicit_cpu.data, self.index), NotImplemented)

    def test_integer_cascade_keeps_native_producer_without_provider(self):
        def unexpected(*args):
            raise AssertionError("cascade integer selected a provider")

        with self.dispatch.override_kernel("tensor.getitem", "cpu", unexpected):
            for slow in (False, True):
                for index in (1, (1,)):
                    result = self._get(self.x, index, slow)
                    self.assertTrue(result._is_view())
                    np.testing.assert_array_equal(result.numpy(), self.values[1])

    def test_training_uses_python_provider_and_retains_gradients(self):
        calls = []

        def provider(value, index, return_x):
            calls.append(index)
            return jt.core.ops.getitem(value, index)

        with jt.flag_scope(no_grad=0), self.dispatch.override_kernel(
                "tensor.getitem", "cpu", provider):
            source = jt.array(self.values).start_grad()
            self.assertIs(jt.core._fast_getitem(source, self.index), NotImplemented)
            result = self._get(source, self.index, slow=True)
            gradient = jt.grad(result.sum(), source)
        expected = np.zeros_like(self.values)
        expected[self.index] = 1
        self.assertEqual(calls, [self.index])
        np.testing.assert_array_equal(gradient.numpy(), expected)


class TestFastDtype(unittest.TestCase):
    def test_dtype_objects(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.bool):
            with self.subTest(dtype=dtype):
                self.assertIs(torch.zeros(2, dtype=dtype).dtype, dtype)
        self.assertIn(torch.ones(1).dtype, {torch.float32})


class TestNativeModuleCall(unittest.TestCase):
    def test_hooks_still_run(self):
        layer = torch.nn.Linear(3, 3)
        seen = []
        layer.register_forward_hook(lambda module, args, out: seen.append(out.shape))
        outer = torch.nn.Sequential(layer)
        outer(torch.randn(2, 3))
        self.assertEqual([tuple(s) for s in seen], [(2, 3)])

    def test_instance_forward_and_results(self):
        layer = torch.nn.Linear(3, 2)
        x = torch.randn(4, 3)
        expected = x.numpy() @ layer.weight.numpy().T + layer.bias.numpy()
        np.testing.assert_allclose(layer(x).numpy(), expected, rtol=1e-5, atol=1e-5)
        layer.forward = lambda value: value * 2
        np.testing.assert_allclose(layer(x).numpy(), x.numpy() * 2)

    def test_parameters_are_published_for_backward(self):
        layer = torch.nn.Linear(3, 1)
        outer = torch.nn.Sequential(layer)
        outer(torch.randn(5, 3)).sum().backward()
        self.assertIsNotNone(layer.weight.grad)
        self.assertEqual(tuple(layer.weight.grad.shape), (1, 3))



class TestFastBinary(unittest.TestCase):
    def test_same_dtype_pairs(self):
        rng = np.random.RandomState(1)
        for dtype in (torch.float32, torch.float16, torch.int32, torch.bool):
            a_np = rng.rand(3, 4) * 4
            b_np = rng.rand(3, 4) * 4 + 1
            a = torch.tensor(a_np, device=_cuda()).to(dtype)
            b = torch.tensor(b_np, device=_cuda()).to(dtype)
            an, bn = a.numpy(), b.numpy()
            ops = [("add", lambda x, y: x + y), ("mul", lambda x, y: x * y)]
            if dtype is not torch.bool:
                ops.append(("sub", lambda x, y: x - y))
            for name, op in ops:
                with self.subTest(dtype=dtype, op=name):
                    out = op(a, b)
                    self.assertEqual(out.dtype, dtype)
                    np.testing.assert_allclose(out.numpy().astype("float64"),
                                               op(an, bn).astype("float64"), rtol=1e-3)
        a = torch.tensor(rng.rand(5) + 1, dtype=torch.float32)
        b = torch.tensor(rng.rand(5) + 1, dtype=torch.float32)
        np.testing.assert_allclose((a / b).numpy(), a.numpy() / b.numpy(), rtol=1e-6)

    def test_python_scalars_keep_the_tensor_dtype(self):
        h = torch.ones(3, dtype=torch.float16, device=_cuda())
        self.assertEqual((h * 2.5).dtype, torch.float16)
        self.assertEqual((2.5 * h).dtype, torch.float16)
        self.assertEqual((h - 1).dtype, torch.float16)
        np.testing.assert_array_equal((1.5 - h).numpy(), np.full(3, 0.5, "float16"))
        i = torch.ones(3, dtype=torch.int32)
        self.assertEqual((i * 3).dtype, torch.int32)
        # These promote, which the Python operator decides.
        self.assertIsNone(jt.core._fast_binary(i, 2.5, 4))
        self.assertEqual((i * 2.5).dtype, torch.float32)
        self.assertIsNone(jt.core._fast_binary(h, True, 0))
        self.assertEqual((h / 2.0).dtype, torch.float16)
        np.testing.assert_array_equal((h / 2.0).numpy(), np.full(3, 0.5, "float16"))
        # A float divisor is widened natively as `_true_division` widens it
        # (bit for bit: test_torch_compat_promotion.py) -- everywhere but ACL,
        # whose widening stays in Python; an int one is not.
        if not getattr(jt.compiler, "has_acl", 0):
            self.assertEqual(jt.core._fast_binary(h, 2.0, 6).dtype, torch.float16)
        self.assertIsNone(jt.core._fast_binary(h, 2, 6))

    def test_the_pairs_it_hands_back(self):
        f = torch.ones(3)
        h = torch.ones(3, dtype=torch.float16)
        self.assertIsNone(jt.core._fast_binary(f, h, 0))
        self.assertEqual((f + h).dtype, torch.float32)
        u = torch.ones(3, dtype=torch.uint8)
        self.assertIsNone(jt.core._fast_binary(u, u, 0))
        self.assertEqual((u + u).dtype, torch.uint8)
        i = torch.ones(3, dtype=torch.int32)
        self.assertIsNone(jt.core._fast_binary(i, i, 6))
        self.assertEqual((i / i).dtype, torch.float32)
        self.assertIsNone(jt.core._fast_binary(f, [1.0, 2.0, 3.0], 4))

    def test_gradients(self):
        x = torch.randn(3, 4, requires_grad=True)
        y = torch.randn(3, 4, requires_grad=True)
        ((x * y + x) * 2.0 - y).sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), (y.numpy() + 1) * 2, rtol=1e-6)
        np.testing.assert_allclose(y.grad.numpy(), x.numpy() * 2 - 1, rtol=1e-6)


class TestFastViews(unittest.TestCase):
    def test_view_reshape_and_unsqueeze(self):
        a = torch.randn(2, 3, 4, device=_cuda())
        n = a.numpy()
        np.testing.assert_array_equal(a.view(6, 4).numpy(), n.reshape(6, 4))
        np.testing.assert_array_equal(a.view(-1).numpy(), n.reshape(-1))
        np.testing.assert_array_equal(a.reshape((4, -1)).numpy(), n.reshape(4, -1))
        np.testing.assert_array_equal(a.reshape([3, 8]).numpy(), n.reshape(3, 8))
        np.testing.assert_array_equal(a.unsqueeze(1).numpy(), n[:, None])
        np.testing.assert_array_equal(a.unsqueeze(-1).numpy(), n[..., None])
        np.testing.assert_array_equal(a.unsqueeze(0).numpy(), n[None])
        # A strided view has no dense buffer to reshape in place; the Python
        # path materializes it first.
        for view, expected in ((a[:, :, ::2], n[:, :, ::2]),
                               (a[:, :1].expand(2, 3, 4), np.broadcast_to(n[:, :1], (2, 3, 4))),
                               (a.transpose(0, 2), n.transpose(2, 1, 0))):
            if not view._storage_is_contiguous():
                self.assertIsNone(jt.core._fast_view(view, (-1,)))
            np.testing.assert_array_equal(view.reshape(-1).numpy(), expected.reshape(-1))

    def test_transposes(self):
        a = torch.randn(2, 3, 4, device=_cuda())
        n = a.numpy()
        np.testing.assert_array_equal(a.transpose(1, 2).numpy(), n.transpose(0, 2, 1))
        np.testing.assert_array_equal(a.transpose(-1, 0).numpy(), n.transpose(2, 1, 0))
        twice = a.transpose(1, 2).transpose(1, 2)
        np.testing.assert_array_equal(twice.numpy(), n)
        composed = a.transpose(0, 1).transpose(1, 2)
        np.testing.assert_array_equal(composed.numpy(), n.transpose(1, 2, 0))

    def test_permutes(self):
        # Built natively for a permutation of non-negative ints, as the
        # transposes above; anything else is `jittor.transpose`'s to report.
        a = torch.randn(2, 3, 4, 5, device=_cuda())
        n = a.numpy()
        heads = a.permute(0, 2, 1, 3)
        np.testing.assert_array_equal(heads.numpy(), n.transpose(0, 2, 1, 3))
        np.testing.assert_array_equal(a.permute((3, 0, 2, 1)).numpy(), n.transpose(3, 0, 2, 1))
        np.testing.assert_array_equal(a.permute([1, 0, 3, 2]).numpy(), n.transpose(1, 0, 3, 2))
        # Composed with the transpose it is a view of, and no transpose at
        # all when the two cancel.
        back = heads.permute(0, 2, 1, 3)
        self.assertNotEqual(back._producer_op(), "transpose")
        np.testing.assert_array_equal(back.numpy(), n)
        np.testing.assert_array_equal(heads.permute(1, 0, 3, 2).numpy(),
                                      n.transpose(2, 0, 3, 1))
        np.testing.assert_array_equal(a.permute(0, -2, 1, -1).numpy(), n.transpose(0, 2, 1, 3))
        with self.assertRaisesRegex(RuntimeError, "twice"):
            a.permute(0, 0, 1, 2)
        with self.assertRaises(RuntimeError):
            a.permute(0, 1, 2)
        z = torch.zeros(2, 3, 4)
        z.permute(2, 0, 1)[1, 0, 2] = 6.0
        self.assertEqual(float(z[0, 2, 1]), 6.0)

    def test_views_write_through(self):
        z = torch.zeros(2, 3)
        z.view(6)[1] = 5.0
        z.transpose(0, 1)[0, 1] = 7.0
        z.unsqueeze(0)[0, 1, 2] = 9.0
        np.testing.assert_array_equal(z.numpy(), [[0, 5, 0], [7, 0, 9]])

    def test_gradients(self):
        x = torch.randn(2, 3, requires_grad=True)
        w = torch.randn(3, 2)
        (x.transpose(0, 1).reshape(6).unsqueeze(0) * w.reshape(1, 6)).sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), w.numpy().reshape(3, 2).T)


class _ConvNormSilu(torch.nn.Module):
    # The native module calls and the functions, on one convolution. At
    # module level: the native module call keeps every class it has seen.
    def __init__(self, conv, norm, silu):
        super().__init__()
        self.conv, self.norm, self.silu = conv, norm, silu

    def forward(self, x, functions):
        c = self.conv(x)
        n = self.norm
        if functions:
            return (jt.nn.group_norm(c, n.num_groups, n.weight, n.bias, n.eps),
                    jt.nn.silu(jt.nn.group_norm(c, n.num_groups, n.weight, n.bias, n.eps)))
        return n(c), self.silu(n(c)), torch.nn.functional.silu(n(c))


class _ToyRMSNorm(torch.nn.Module):
    # Transformers' LlamaRMSNorm, by name and body.
    def __init__(self, size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.variance_epsilon = eps

    def forward(self, hidden):
        dtype = hidden.dtype
        hidden = hidden.to(torch.float32)
        variance = hidden.pow(2).mean(-1, keepdim=True)
        hidden = hidden * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden.to(dtype)


class TestRegisteredRmsInference(unittest.TestCase):
    def setUp(self):
        from jittor._runtime import dispatch
        from jittor.compat.torch.installers.nn import module_methods
        self.dispatch, self.methods = dispatch, module_methods
        # Check one dispatch at a time. Automatic replay deliberately runs an
        # eager warmup and a capture, then can execute without Python callbacks.
        self.flags = jt.flag_scope(use_cuda=0, amp_reg=0, auto_graph_replay=0)
        self.flags.__enter__()
        self.addCleanup(self.flags.__exit__, None, None, None)
        self.norm = _ToyRMSNorm(8).cpu().eval()
        self.outer = torch.nn.Sequential(self.norm).eval()
        self.values = np.arange(16, dtype="float32").reshape(2, 8) / 16 + 0.5
        self.x = torch.tensor(self.values, dtype=torch.float32, device="cpu")
        # Publish the nested module's parameters before exercising its shortcut.
        with torch.no_grad():
            self.outer(self.x).sync()

    @staticmethod
    def _inside_python_rms():
        import sys
        frame = sys._getframe(1)
        while frame is not None:
            if frame.f_code.co_name == "_standard_rms_norm":
                return True
            frame = frame.f_back
        return False

    def _reference(self):
        return self.values / np.sqrt((self.values ** 2).mean(-1, keepdims=True)
                                     + self.norm.variance_epsilon)

    def test_registered_provider_runs_once_and_reads_current_weight_and_epsilon(self):
        seen = []

        def provider(value, weight, epsilon):
            seen.append((epsilon, self._inside_python_rms()))
            return value * 2 + weight

        with torch.no_grad(), self.dispatch.override_kernel("nn.rms_norm.inference", "cpu", provider):
            first = self.outer(self.x).numpy()
            self.norm.weight.copy_(torch.full((8,), 0.25, device="cpu"))
            self.norm.variance_epsilon = 0.125
            second = self.outer(self.x).numpy()
        self.assertEqual(seen, [(1e-6, False), (0.125, False)])
        np.testing.assert_array_equal(first, self.values * 2 + 1)
        np.testing.assert_array_equal(second, self.values * 2 + 0.25)

    def test_training_provider_keeps_priority_even_under_no_grad(self):
        seen = []

        def training(value, weight, epsilon):
            seen.append(("training", self._inside_python_rms()))
            return value + 7

        def unexpected(*args):
            raise AssertionError("inference provider ran after an accepted training provider")

        with torch.no_grad(), self.dispatch.override_kernel("nn.rms_norm.training", "cpu", training), \
                self.dispatch.override_kernel("nn.rms_norm.inference", "cpu", unexpected):
            actual = self.outer(self.x).numpy()
        self.assertEqual(seen, [("training", False)])
        np.testing.assert_array_equal(actual, self.values + 7)

    def test_declining_providers_resume_original_forward_once(self):
        seen = []

        def training(*args):
            seen.append(("training", self._inside_python_rms()))
            return None

        def inference(*args):
            seen.append(("inference", self._inside_python_rms()))
            return None

        with torch.no_grad(), self.dispatch.override_kernel("nn.rms_norm.training", "cpu", training), \
                self.dispatch.override_kernel("nn.rms_norm.inference", "cpu", inference):
            actual = self.outer(self.x).numpy()
        self.assertEqual(seen, [("training", False), ("inference", False)])
        np.testing.assert_allclose(actual, self._reference(), rtol=1e-6, atol=1e-6)

    def test_callback_errors_are_not_retried_and_restore_placement(self):
        class RmsError(ValueError):
            pass

        for predicate in (False, True):
            seen = []

            def failure(*args):
                seen.append(self._inside_python_rms())
                raise RmsError("registered RMS failure")

            with self.subTest(predicate=predicate), torch.no_grad():
                placement = jt.core._current_tensor_placement()
                provider = (lambda *args: None) if predicate else failure
                with self.dispatch.override_kernel("nn.rms_norm.inference", "cpu", provider,
                                                   supports=failure if predicate else None):
                    with self.assertRaisesRegex(RmsError, "registered RMS failure"):
                        self.outer(self.x)
                self.assertEqual(seen, [False])
                self.assertEqual(jt.core._current_tensor_placement(), placement)

    def test_gradient_enabled_calls_keep_the_python_owner(self):
        seen = []

        def training(value, weight, epsilon):
            seen.append(self._inside_python_rms())
            return value * weight

        self.x.requires_grad_(True)
        with self.dispatch.override_kernel("nn.rms_norm.training", "cpu", training):
            result = self.outer(self.x)
            result.sum().backward()
        self.assertEqual(seen, [True])
        np.testing.assert_array_equal(result.detach().numpy(), self.values)
        np.testing.assert_array_equal(self.x.grad.numpy(), np.ones_like(self.values))

    def test_bfloat16_order_helper_and_module_hooks_stay_in_python(self):
        x_bf16 = self.x.to(torch.bfloat16)
        seen = []

        def bf16_order(value, weight, epsilon):
            seen.append(self._inside_python_rms())
            return value

        with torch.no_grad(), mock.patch.object(self.methods, "_acl_bfloat16_rms_norm", bf16_order):
            result = self.outer(x_bf16)
        self.assertEqual(seen, [True])
        self.assertEqual(result.dtype, torch.bfloat16)
        np.testing.assert_array_equal(result.float().numpy(), self.values)
        seen.clear()

        def provider(value, weight, epsilon):
            seen.append(self._inside_python_rms())
            return value * 2

        hooked = []
        handle = self.norm.register_forward_hook(lambda module, args, output: hooked.append(output))
        try:
            with torch.no_grad(), self.dispatch.override_kernel("nn.rms_norm.inference", "cpu", provider):
                output = self.outer(self.x)
            self.assertEqual(seen, [True])
            self.assertEqual(len(hooked), 1)
            np.testing.assert_array_equal(output.numpy(), self.values * 2)
        finally:
            handle.remove()

    def test_instance_forward_and_python_placement_overrides_are_preserved(self):
        seen = []

        def provider(value, weight, epsilon):
            seen.append(self._inside_python_rms())
            return value * 2

        with torch.no_grad(), self.dispatch.override_kernel("nn.rms_norm.inference", "cpu", provider):
            self.norm.forward = lambda value: value + 11
            result = self.outer(self.x)
            self.assertEqual(seen, [])
            np.testing.assert_array_equal(result.numpy(), self.values + 11)
            del self.norm.forward
            original = self.dispatch.dispatch_context
            with mock.patch.object(self.dispatch, "dispatch_context", side_effect=original):
                result = self.outer(self.x)
            self.assertEqual(seen, [True])
            np.testing.assert_array_equal(result.numpy(), self.values * 2)


class TestNativeModuleDispatch(unittest.TestCase):
    """What `_dispatch_module_call` does, taken natively inside a module."""

    def test_a_bias_free_linear(self):
        device = _cuda()
        for dtype in (torch.float32, torch.float16):
            layer = torch.nn.Linear(8, 5, bias=False).to(device=device, dtype=dtype)
            outer = torch.nn.Sequential(layer)
            x = torch.randn(2, 3, 8, device=device, dtype=dtype)
            with torch.no_grad():
                out = outer(x)
            self.assertEqual(out.dtype, dtype)
            expected = x.float().numpy() @ layer.weight.float().numpy().T
            np.testing.assert_allclose(out.float().numpy(), expected, rtol=2e-2, atol=2e-2)
        # Training takes the same op and keeps its gradient.
        layer = torch.nn.Linear(4, 2, bias=False)
        x = torch.randn(3, 4)
        torch.nn.Sequential(layer)(x).sum().backward()
        np.testing.assert_allclose(layer.weight.grad.numpy(),
                                   np.tile(x.numpy().sum(0), (2, 1)), rtol=1e-5)

    @unittest.skipUnless(_CUDA_KERNELS, "the cuBLASLt route is CUDA's")
    def test_a_linear_with_a_bias_at_inference(self):
        # Built natively as the same cuBLASLt operator `lt_linear_cuda` builds.
        from jittor.backends.cuda.kernels.cublas.lt_linear_cuda import lt_linear_cuda
        for dtype in (torch.float32, torch.float16):
            layer = torch.nn.Linear(256, 1024).to(device="cuda", dtype=dtype)
            outer = torch.nn.Sequential(layer)
            x = torch.randn(1, 8, 256, device="cuda", dtype=dtype)
            with torch.no_grad():
                got = outer(x)
                want = lt_linear_cuda(x, layer.weight, layer.bias)
            self.assertEqual(got.dtype, dtype)
            self.assertEqual(tuple(got.shape), (1, 8, 1024))
            np.testing.assert_array_equal(got.float().numpy(), want.float().numpy())
        # What it hands back: a product too small for that route, and a call
        # that records gradients.
        small = torch.nn.Linear(8, 4).cuda()
        xs = torch.randn(2, 8, device="cuda")
        with torch.no_grad():
            np.testing.assert_allclose(
                torch.nn.Sequential(small)(xs).numpy(),
                xs.numpy() @ small.weight.numpy().T + small.bias.numpy(), rtol=1e-4, atol=1e-4)
        layer = torch.nn.Linear(256, 1024).cuda()
        x = torch.randn(1, 8, 256, device="cuda")
        torch.nn.Sequential(layer)(x).sum().backward()
        np.testing.assert_allclose(layer.bias.grad.numpy(), np.full(1024, 8.0), rtol=1e-5)

    def test_a_dropout_that_does_not_train_hands_back_its_input(self):
        x = torch.randn(4, 8, device=_cuda())
        drop = torch.nn.Dropout(0.5)
        outer = torch.nn.Sequential(drop)
        outer.eval()
        with torch.no_grad():
            self.assertIs(outer(x), x)
        outer.train()
        with torch.no_grad():
            dropped = outer(x)
        self.assertIsNot(dropped, x)
        self.assertTrue(bool((dropped == 0).any()))
        none = torch.nn.Sequential(torch.nn.Dropout(0.0))
        none.train()
        with torch.no_grad():
            self.assertIs(none(x), x)

    @unittest.skipIf(getattr(jt.compiler, "has_acl", 0), "ACL launches its own relu kernel")
    def test_a_relu(self):
        # Built natively as `jittor.nn.relu` builds it -- one `unary` relu --
        # without reaching Python, once `inplace=True` has had its warning.
        from jittor.nn.functional import activation
        x = torch.randn(4, 8, device=_cuda())
        for inplace in (False, True):
            outer = torch.nn.Sequential(torch.nn.ReLU(inplace=inplace))
            outer(x)
            with _refusing(activation, "_fused_activation"):
                got = outer(x)
            np.testing.assert_array_equal(got.numpy(), np.maximum(x.numpy(), 0))
        # What it hands back to Python: a replaced `jittor.nn.relu` ...
        outer = torch.nn.Sequential(torch.nn.ReLU())
        original = jt.nn.relu
        jt.nn.relu = lambda value, inplace=False: value * 0 + 7
        try:
            np.testing.assert_array_equal(outer(x).numpy(), np.full((4, 8), 7.0))
        finally:
            jt.nn.relu = original
        # ... an input whose producer offered to apply the activation itself ...
        y = x + 0
        activation.offer_activation(y, lambda act: ("fused", act))
        self.assertEqual(outer(y), ("fused", "relu"))
        # ... and a gradient still flows.
        z = torch.randn(4, 8, device=_cuda(), requires_grad=True)
        outer(z).sum().backward()
        np.testing.assert_array_equal(z.grad.numpy(), (z.numpy() > 0).astype("float32"))

    @unittest.skipUnless(_CUDA_KERNELS, "the channels-last group norm is CUDA's")
    def test_an_inference_group_norm_and_the_silu_after_it(self):
        # Built natively as `_group_norm_nhwc` builds it, with the same offer
        # to take the activation into its pass -- which a SiLU module and
        # `F.silu` both take -- bit for bit what the functions give.
        from jittor.backends.cuda.kernels.nn import group_norm_cuda
        conv = torch.nn.Conv2d(64, 64, 3, padding=1).cuda().half().eval()
        norm = torch.nn.GroupNorm(32, 64).cuda().half().eval()
        with torch.no_grad():
            norm.weight.copy_(torch.randn(64, device="cuda").half())
            norm.bias.copy_(torch.randn(64, device="cuda").half())
        silu = torch.nn.SiLU()
        outer = torch.nn.Sequential(conv, norm)

        x = torch.randn(2, 64, 16, 16, device="cuda").half()
        both = _ConvNormSilu(conv, norm, silu)
        with torch.no_grad():
            both(x, False)
            with _refusing(group_norm_cuda, "_group_norm_nhwc"):
                y, activated, functional = both(x, False)
            want_y, want = both(x, True)
            np.testing.assert_array_equal(y.numpy(), want_y.numpy())
            np.testing.assert_array_equal(activated.numpy(), want.numpy())
            np.testing.assert_array_equal(functional.numpy(), want.numpy())
            # A SiLU with no offer to take: x * x.sigmoid().
            plain = torch.randn(2, 8, 4, 4, device="cuda").half()
            np.testing.assert_array_equal(torch.nn.Sequential(silu)(plain).numpy(),
                                          jt.nn.silu(plain).numpy())
            # A dense input goes the way it went.
            dense = torch.randn(2, 64, 8, 8, device="cuda").half()
            np.testing.assert_allclose(
                torch.nn.Sequential(norm)(dense).float().numpy(),
                jt.nn.group_norm(dense, 32, norm.weight, norm.bias, norm.eps).float().numpy(),
                rtol=1e-3, atol=1e-3)
            # A replaced `jittor.nn.group_norm` answers.
            original = jt.nn.group_norm
            jt.nn.group_norm = lambda value, *args: value * 0 + 3
            try:
                np.testing.assert_array_equal(outer(x).float().numpy(), np.full((2, 64, 16, 16), 3.0))
            finally:
                jt.nn.group_norm = original
        # With a gradient to record it is the Python path's.
        z = torch.randn(2, 64, 16, 16, device="cuda", requires_grad=True)
        train = torch.nn.GroupNorm(32, 64).cuda()
        torch.nn.Sequential(train, torch.nn.SiLU())(z).sum().backward()
        self.assertEqual(tuple(z.grad.shape), (2, 64, 16, 16))

    @unittest.skipUnless(_CUDA_KERNELS, "the RMSNorm kernel is CUDA's")
    def test_a_bfloat16_rms_norm_rounds_as_torch_does(self):
        # The normalized value is rounded to bfloat16 before the weight
        # multiplies it, as ATen and HF's `LlamaRMSNorm` do; rounding once at
        # the end differed by one bfloat16 quantum in a quarter of the outputs.
        def bf16(a):
            bits = np.asarray(a, np.float32).copy().view(np.uint32)
            bits += 0x7FFF + ((bits >> 16) & 1)
            bits &= 0xFFFF0000
            return bits.view(np.float32)

        rng = np.random.RandomState(0)
        x = bf16(rng.randn(64, 256) * 3)
        w = bf16(rng.randn(256) * 0.5 + 1)
        norm = _ToyRMSNorm(256).cuda().to(torch.bfloat16)
        with torch.no_grad():
            norm.weight.copy_(torch.tensor(w).to(torch.bfloat16))
        outer = torch.nn.Sequential(norm)
        x64 = x.astype(np.float64)
        want = bf16(bf16(x64 / np.sqrt((x64 * x64).mean(-1, keepdims=True) + 1e-6)) * w)
        xt = torch.tensor(x, device="cuda").to(torch.bfloat16)
        with torch.no_grad():
            np.testing.assert_array_equal(outer(xt).float().numpy(), want)
        np.testing.assert_array_equal(
            outer(xt.clone().requires_grad_(True)).float().detach().numpy(), want)

    def test_a_standard_rms_norm(self):
        class ToyRMSNorm(torch.nn.Module):
            def __init__(self, size, eps=1e-6):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(size) * 0.5)
                self.variance_epsilon = eps

            def forward(self, hidden):
                variance = hidden.float().pow(2).mean(-1, keepdim=True)
                return self.weight * (hidden.float() * torch.rsqrt(variance + self.variance_epsilon)).to(hidden.dtype)

        device = _cuda()
        norm = ToyRMSNorm(16).to(device)
        outer = torch.nn.Sequential(norm)
        x = torch.randn(3, 16, device=device)
        n = x.numpy()
        expected = n / np.sqrt((n ** 2).mean(-1, keepdims=True) + 1e-6) * 0.5
        with torch.no_grad():
            np.testing.assert_allclose(outer(x).numpy(), expected, rtol=1e-4, atol=1e-5)
        x.requires_grad_(True)
        out = outer(x)
        np.testing.assert_allclose(out.detach().numpy(), expected, rtol=1e-4, atol=1e-5)
        out.sum().backward()
        self.assertEqual(tuple(x.grad.shape), (3, 16))


    @unittest.skipUnless(_CUDA_KERNELS, "the native LayerNorm build is CUDA's")
    def test_a_layer_norm_at_inference(self):
        # Built natively from the source `_layer_norm_no_grad_cuda` builds, the
        # warp-a-row kernel and the block-a-row one alike: bit for bit.
        from jittor.backends.cuda.kernels.nn.layer_norm_cuda import _layer_norm_no_grad_cuda
        from jittor.nn.functional import normalization
        for dtype, rows in ((torch.float32, 6), (torch.float16, 6), (torch.float32, 2048)):
            norm = torch.nn.LayerNorm(320).to(device="cuda", dtype=dtype)
            with torch.no_grad():
                norm.weight.copy_(torch.randn(320, device="cuda", dtype=dtype))
                norm.bias.copy_(torch.randn(320, device="cuda", dtype=dtype))
            outer = torch.nn.Sequential(norm)
            x = torch.randn(rows, 320, device="cuda", dtype=dtype) * 3 + 1
            with torch.no_grad():
                # A module's first call publishes its parameters in Python;
                # from the second on it is native, and the functional's relay
                # is not called.
                outer(x)
                with _refusing(normalization, "_layer_norm_no_grad_cuda"):
                    got = outer(x)
                want = _layer_norm_no_grad_cuda(x, (320,), norm.weight, norm.bias, norm.eps)
            self.assertEqual(got.dtype, dtype)
            np.testing.assert_array_equal(got.float().numpy(), want.float().numpy())
        # A call that needs a gradient takes the Python way and gets one.
        norm = torch.nn.LayerNorm(16).cuda()
        x = torch.randn(3, 16, device="cuda", requires_grad=True)
        torch.nn.Sequential(norm)(x).sum().backward()
        self.assertEqual(tuple(norm.weight.grad.shape), (16,))
        self.assertEqual(tuple(x.grad.shape), (3, 16))
        # A shape `layer_norm` rejects is rejected as before.
        with torch.no_grad(), self.assertRaises(Exception):
            torch.nn.Sequential(torch.nn.LayerNorm(16).cuda())(torch.randn(3, 8, device="cuda"))


class TestNativeGelu(unittest.TestCase):
    def test_the_native_build_matches_the_python_one(self):
        from jittor.nn.functional import activation
        native = activation._FAST_GELU
        self.assertIsNotNone(native)
        for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
            x = torch.randn(4, 33, device=_cuda()).to(dtype) * 3
            got = torch.nn.functional.gelu(x)
            try:
                activation._FAST_GELU = None
                want = torch.nn.functional.gelu(x)
            finally:
                activation._FAST_GELU = native
            self.assertEqual(got.dtype, dtype)
            np.testing.assert_array_equal(got.double().numpy(), want.double().numpy())
        x = torch.randn(5, device=_cuda(), requires_grad=True)
        torch.nn.functional.gelu(x).sum().backward()
        xn = x.detach().double().numpy()
        from math import erf, pi, sqrt
        expected = [0.5 * (1 + erf(v / sqrt(2))) + v * np.exp(-v * v / 2) / sqrt(2 * pi) for v in xn]
        np.testing.assert_allclose(x.grad.numpy(), expected, rtol=1e-5, atol=1e-6)
        # The tanh form, and an integer tensor, stay on the Python path.
        y = torch.randn(7, device=_cuda())
        yn = y.double().numpy()
        expected = 0.5 * yn * (1 + np.tanh(np.sqrt(2 / np.pi) * (yn + 0.044715 * yn ** 3)))
        np.testing.assert_allclose(torch.nn.functional.gelu(y, approximate="tanh").numpy(),
                                   expected, rtol=1e-5, atol=1e-6)
        self.assertIsNone(jt.core._fast_gelu(torch.arange(4, device=_cuda())))


    @unittest.skipUnless(_CUDA_KERNELS, "the one-kernel GELU is built on CUDA")
    def test_a_gelu_of_a_linear_output_is_one_kernel_with_the_same_bits(self):
        from jittor.nn.functional import activation
        outer = torch.nn.Sequential(torch.nn.Linear(256, 512)).cuda()
        flags = jt.flags.auto_graph_replay
        jt.flags.auto_graph_replay = 0
        try:
            for dtype in (torch.float32, torch.float16):
                outer = outer.to(dtype)
                x = torch.randn(8, 256, device="cuda", dtype=dtype)
                with torch.no_grad():
                    outer(x)
                    hidden = outer(x)
                    self.assertEqual(hidden._producer_name(), "code")
                    got = torch.nn.functional.gelu(hidden)
                    self.assertEqual(got._producer_name(), "code")
                    native = activation._FAST_GELU
                    try:
                        activation._FAST_GELU = None
                        want = torch.nn.functional.gelu(hidden)
                    finally:
                        activation._FAST_GELU = native
                self.assertEqual(got.dtype, dtype)
                np.testing.assert_array_equal(got.float().numpy(), want.float().numpy())
        finally:
            jt.flags.auto_graph_replay = flags


class TestSdpaRoutes(unittest.TestCase):
    @unittest.skipUnless(_CUDA_KERNELS, "the routes remembered are CUDA kernels")
    def test_a_remembered_route_gives_the_walk_s_answer(self):
        from jittor.compat.torch.installers.nn import attention
        from jittor._runtime import dispatch
        from jittor.nn.functional import attention as native
        sdpa = torch.nn.functional.scaled_dot_product_attention
        q = torch.randn(1, 4, 16, 32, device="cuda")
        k = torch.randn(1, 4, 16, 32, device="cuda")
        v = torch.randn(1, 4, 16, 32, device="cuda")
        mask = torch.randn(1, 1, 16, 16, device="cuda")
        attention._ROUTES.clear()
        with torch.no_grad():
            first = sdpa(q, k, v, attn_mask=mask).numpy()
            self.assertEqual(len(attention._ROUTES), 1)
            second = sdpa(q, k, v, attn_mask=mask).numpy()
        np.testing.assert_array_equal(first, second)
        # A registry change is a new walk: the override answers.
        calls = []

        def kernel(query, key, value, **kwargs):
            calls.append(1)
            return None

        with dispatch.override_kernel("nn.fused_attention", "cuda", kernel), torch.no_grad():
            third = sdpa(q, k, v, attn_mask=mask).numpy()
        self.assertEqual(len(calls), 1)
        np.testing.assert_allclose(third, first, rtol=1e-4, atol=1e-5)
        # Training is never remembered.
        attention._ROUTES.clear()
        qg = q.clone().requires_grad_(True)
        sdpa(qg, k, v, attn_mask=mask).sum().backward()
        self.assertEqual(len(attention._ROUTES), 0)
        self.assertEqual(tuple(qg.grad.shape), (1, 4, 16, 32))
        native.LAST_FUSED_KERNEL[0] = None

    @unittest.skipUnless(_CUDA_KERNELS, "the routes remembered are CUDA kernels")
    def test_a_device_route_is_not_taken_by_host_tensors_of_the_same_shapes(self):
        from jittor.compat.torch.installers.nn import attention
        sdpa = torch.nn.functional.scaled_dot_product_attention
        rng = np.random.RandomState(3)
        q_np = rng.randn(2, 4, 16, 32).astype("float32")
        mask_np = np.zeros((2, 1, 16, 16), "float32")
        attention._ROUTES.clear()
        outs = []
        try:
            with torch.no_grad():
                for device in ("cuda", "cpu", "cuda"):
                    q = torch.tensor(q_np, device=device)
                    mask = torch.tensor(mask_np, device=device)
                    out = sdpa(q, q, q, attn_mask=mask)
                    self.assertEqual(out.device.type, q.device.type)
                    outs.append(out.numpy())
        finally:
            attention._ROUTES.clear()
        np.testing.assert_allclose(outs[1], outs[0], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(outs[2], outs[0], rtol=1e-4, atol=1e-5)

    @unittest.skipUnless(_CUDA_KERNELS, "the routes remembered are CUDA kernels")
    def test_a_float32_call_flash_declines_is_remembered_with_its_cast_switch(self):
        # Flash takes no float32 unless asked to cast it, which is part of
        # what the route is remembered under.
        import os
        from jittor.compat.torch.installers.nn import attention
        sdpa = torch.nn.functional.scaled_dot_product_attention
        q, k, v = (torch.randn(1, 2, 8, 32, device="cuda") for _ in range(3))
        attention._ROUTES.clear()
        saved = os.environ.pop("JITTOR_FLASH_ATTN_CAST_FLOAT32", None)
        try:
            with torch.no_grad():
                first = sdpa(q, k, v).numpy()
                if attention._LAST_MISS[0] != "dtype":
                    self.skipTest("flash declined float32 for %s" % attention._LAST_MISS[0])
                self.assertEqual(len(attention._ROUTES), 1)
                np.testing.assert_array_equal(sdpa(q, k, v).numpy(), first)
                os.environ["JITTOR_FLASH_ATTN_CAST_FLOAT32"] = "1"
                cast = sdpa(q, k, v).numpy()
            self.assertNotEqual(attention._LAST_MISS[0], "dtype")
            np.testing.assert_allclose(cast, first, rtol=2e-2, atol=2e-2)
        finally:
            os.environ.pop("JITTOR_FLASH_ATTN_CAST_FLOAT32", None)
            if saved is not None:
                os.environ["JITTOR_FLASH_ATTN_CAST_FLOAT32"] = saved
            attention._ROUTES.clear()


class TestNativeRules(unittest.TestCase):
    @unittest.skipUnless(_CUDA_KERNELS, "the LayerNorm relay is CUDA's")
    def test_the_layer_norm_relay_is_chosen_natively_as_before(self):
        from jittor._runtime.dispatch import select_kernel
        from jittor.backends.cuda.kernels.nn import layer_norm_cuda
        impl = layer_norm_cuda._layer_norm_no_grad_cuda.__wrapped__
        w = torch.ones(32, device="cuda")
        b = torch.zeros(32, device="cuda")
        cases = [
            (torch.randn(4, 32, device="cuda"), (32,), w, b),
            (torch.randn(4, 32, device="cuda").half(), (32,), w.half(), b.half()),
            (torch.randn(4, 32, device="cuda").bfloat16(), (32,), w, b),
            (torch.randn(4, 16, device="cuda"), (16,), w, b),
            (torch.randn(4, 32, device="cuda"), (4, 32), w, b),
            (torch.randn(4, 32, device="cuda"), (32,), w, 0.0),
        ]
        with torch.no_grad():
            for x, shape, weight, bias in cases:
                python = layer_norm_cuda._supports_layer_norm_inference(x, shape, weight, bias, 1e-5)
                chosen = select_kernel("nn.layer_norm.inference", x, shape, weight, bias, 1e-5)
                self.assertIs(chosen, impl if python else None)

    @unittest.skipUnless(_CUDA_KERNELS, "the conv2d kernels compared are CUDA's")
    def test_the_conv2d_kernels_are_chosen_natively_as_before(self):
        # The depthwise kernel, then cuDNN, as their Python predicates answer.
        from jittor._runtime.dispatch import select_kernel
        from jittor.nn.backends import cudnn
        from jittor.nn.modules import depthwise
        cudnn_kernel = getattr(cudnn._try_cudnn_conv2d, "__wrapped__", cudnn._try_cudnn_conv2d)
        x = torch.randn(1, 8, 9, 9, device="cuda")
        dense = torch.randn(8, 8, 3, 3, device="cuda")
        per_channel = torch.randn(8, 1, 3, 3, device="cuda")
        cases = [(x, dense, 1, {}), (x, per_channel, 8, {}), (x.half(), per_channel.half(), 8, {}),
                 (x, per_channel, 8, {"_depthwise_fast_path": False}), (x, dense.half(), 1, {}),
                 (x.half(), dense.half(), 1, {"_depthwise_fast_path": True})]
        for a, w, groups, kwargs in cases:
            args = (a, w, None, (1, 1), (1, 1), (1, 1), groups)
            if depthwise._supports_depthwise_conv2d(*args, **kwargs):
                want = depthwise._depthwise_conv2d
            elif cudnn._supports_conv2d(*args, **kwargs):
                want = cudnn_kernel
            else:
                want = None
            got = select_kernel("conv2d", *args, **kwargs)
            if want is None:
                self.assertNotIn(got, (depthwise._depthwise_conv2d, cudnn_kernel))
            else:
                self.assertIs(got, want, (a.dtype, w.dtype, groups, kwargs))

    @unittest.skipUnless(_CUDA_KERNELS, "the group norm kernel is CUDA's")
    def test_the_group_norm_kernel_is_chosen_natively_as_before(self):
        from jittor._runtime.dispatch import select_kernel
        from jittor.backends.cuda.kernels.nn import group_norm_cuda
        kernel = getattr(group_norm_cuda._group_norm_cuda, "__wrapped__", group_norm_cuda._group_norm_cuda)
        x = torch.randn(2, 8, 4, 4, device="cuda")
        w, b = torch.ones(8, device="cuda"), torch.zeros(8, device="cuda")
        cases = [(x, 4, w, b, 1e-5), (x, 3, w, b, 1e-5), (x, 4, w[:4], b, 1e-5),
                 (x, 4, w, b, 0.0), (x, 4, None, b, 1e-5), (x.reshape(2, 8, 16), 4, w, b, 1e-5),
                 (x.half(), 2, w.half(), b.half(), 1e-6)]
        for args in cases:
            want = kernel if group_norm_cuda._supports_group_norm(*args) else None
            got = select_kernel("nn.group_norm", *args)
            if want is None:
                self.assertIsNot(got, kernel, args[1:])
            else:
                self.assertIs(got, kernel)

    def test_the_matmul_relay_is_chosen_natively_as_before(self):
        from jittor._runtime.dispatch import select_kernel
        from jittor.nn.functional import matrix
        a = torch.randn(4, 8, device=_cuda())
        b = torch.randn(8, 3, device=_cuda())
        expected = matrix._supports_cublas(a, b) and matrix._cublas_matmul
        chosen = select_kernel("matmul", a, b, False, False)
        if expected:
            self.assertIs(chosen, matrix._cublas_matmul)
        mixed = select_kernel("matmul", a, b.half(), False, False)
        self.assertIsNot(mixed, matrix._cublas_matmul)


if __name__ == "__main__":
    unittest.main()

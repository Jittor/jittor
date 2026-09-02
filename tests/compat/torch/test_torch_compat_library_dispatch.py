"""torch.library dispatch, registered autograd, and integration overrides.

Task 7.09.  Three defects from the compat audit's "custom operators" section:

* every dispatch key collapsed to "whichever was registered last", so
  ``impl(..., "CPU")`` followed by ``impl(..., "CUDA")`` ran the CUDA kernel on
  CPU tensors, and the common ``impl(..., ("CPU", "CUDA", "Meta"))`` spelling
  put the *meta* kernel last -- every real call returned fake numbers;
* ``register_autograd`` stored a backward that nothing ever read, so an
  operator whose forward leaves the tape produced gradients of exactly zero;
* ``custom_op`` hard-coded one model's operator name and discarded the
  caller's implementation for it.

Run: python -m pytest tests/compat/torch/test_torch_compat_library_dispatch.py
"""
import unittest

import numpy as np

import jittor as jt
import jittor as torch


_COUNTER = [0]


def _fresh_ns():
    _COUNTER[0] += 1
    return "jittor_dispatch_%d" % _COUNTER[0]


class TestDispatchKeySelection(unittest.TestCase):
    def _op(self, keys_to_impls, schema="f(Tensor x) -> Tensor"):
        ns = _fresh_ns()
        lib = torch.library.Library(ns, "DEF")
        lib.define(schema)
        for key, fn in keys_to_impls:
            lib.impl("f", fn, dispatch_key=key)
        return getattr(torch.ops, ns).f

    def test_cpu_tensor_takes_the_cpu_kernel_registered_first(self):
        op = self._op([("CPU", lambda x: x + 1), ("CUDA", lambda x: x + 100)])
        jt.flags.use_cuda = 0
        self.assertEqual(op(jt.zeros(2)).numpy().tolist(), [1.0, 1.0])

    def test_cpu_tensor_takes_the_cpu_kernel_registered_last(self):
        op = self._op([("CUDA", lambda x: x + 100), ("CPU", lambda x: x + 1)])
        jt.flags.use_cuda = 0
        self.assertEqual(op(jt.zeros(2)).numpy().tolist(), [1.0, 1.0])

    def test_meta_kernel_never_serves_a_real_call(self):
        # `impl(qualname, ("CPU", "CUDA", "Meta"), fn)` is the shape that used
        # to leave Meta last in the dict and therefore always selected.
        marker = {}

        def real(x):
            marker["real"] = True
            return x + 1

        def meta(x):
            marker["meta"] = True
            return jt.empty(x.shape, dtype=x.dtype)

        ns = _fresh_ns()
        lib = torch.library.Library(ns, "DEF")
        lib.define("f(Tensor x) -> Tensor")
        lib.impl("f", real, dispatch_key="CPU")
        lib.impl("f", real, dispatch_key="CUDA")
        lib.impl("f", meta, dispatch_key="Meta")
        jt.flags.use_cuda = 0
        out = getattr(torch.ops, ns).f(jt.zeros(2))
        self.assertEqual(out.numpy().tolist(), [1.0, 1.0])
        self.assertTrue(marker.get("real"))
        self.assertNotIn("meta", marker)

    def test_meta_only_operator_refuses_to_run(self):
        op = self._op([("Meta", lambda x: jt.empty(x.shape, dtype=x.dtype))])
        jt.flags.use_cuda = 0
        with self.assertRaises(NotImplementedError) as cm:
            op(jt.zeros(2))
        self.assertIn("fake", str(cm.exception))

    def test_composite_key_is_the_fallback_for_either_residency(self):
        op = self._op([("CompositeExplicitAutograd", lambda x: x + 7)])
        jt.flags.use_cuda = 0
        self.assertEqual(op(jt.zeros(2)).numpy().tolist(), [7.0, 7.0])

    def test_backend_specific_kernel_beats_the_composite_fallback(self):
        op = self._op([("CompositeExplicitAutograd", lambda x: x + 7),
                       ("CPU", lambda x: x + 1)])
        jt.flags.use_cuda = 0
        self.assertEqual(op(jt.zeros(2)).numpy().tolist(), [1.0, 1.0])

    def test_operator_with_no_usable_kernel_names_the_registered_keys(self):
        op = self._op([("XLA", lambda x: x)])
        jt.flags.use_cuda = 0
        with self.assertRaises(NotImplementedError) as cm:
            op(jt.zeros(2))
        self.assertIn("XLA", str(cm.exception))

    @unittest.skipUnless(jt.has_cuda, "needs an accelerator")
    def test_cuda_tensor_takes_the_cuda_kernel(self):
        op = self._op([("CPU", lambda x: x + 1), ("CUDA", lambda x: x + 100)])
        saved = jt.flags.use_cuda
        jt.flags.use_cuda = 1
        try:
            x = jt.zeros(2)
            x.sync()
            self.assertEqual(op(x).numpy().tolist(), [100.0, 100.0])
        finally:
            jt.flags.use_cuda = saved


class TestRegisteredAutograd(unittest.TestCase):
    def _tape_leaving_double(self, x):
        """A forward that detaches -- exactly the case the audit describes."""
        return jt.array(x.numpy() * 2.0)

    def test_registered_backward_is_actually_called(self):
        ns = _fresh_ns()
        lib = torch.library.Library(ns, "DEF")
        lib.define("f(Tensor x) -> Tensor")
        lib.impl("f", self._tape_leaving_double,
                 dispatch_key="CompositeExplicitAutograd")
        called = []

        def backward(ctx, grad):
            called.append(True)
            return grad * 2.0

        torch.library.register_autograd("%s::f" % ns, backward)
        op = getattr(torch.ops, ns).f
        x = jt.array(np.array([1.0, 2.0], dtype="float32"))
        x.requires_grad = True
        y = op(x)
        g = jt.grad(y.sum(), [x])[0]
        self.assertTrue(called, "the registered backward must run")
        np.testing.assert_allclose(g.numpy(), [2.0, 2.0], rtol=1e-5)

    def test_without_the_fix_the_gradient_would_be_zero(self):
        # Same operator, no register_autograd: the tape-leaving forward really
        # does produce a zero gradient. This pins WHY the wiring matters.
        ns = _fresh_ns()
        lib = torch.library.Library(ns, "DEF")
        lib.define("f(Tensor x) -> Tensor")
        lib.impl("f", self._tape_leaving_double,
                 dispatch_key="CompositeExplicitAutograd")
        op = getattr(torch.ops, ns).f
        x = jt.array(np.array([1.0, 2.0], dtype="float32"))
        x.requires_grad = True
        g = jt.grad(op(x).sum(), [x])[0]
        np.testing.assert_allclose(g.numpy(), [0.0, 0.0], atol=1e-6)

    def test_setup_context_state_reaches_backward(self):
        ns = _fresh_ns()
        lib = torch.library.Library(ns, "DEF")
        lib.define("f(Tensor x, float k) -> Tensor")
        lib.impl("f", lambda x, k: jt.array(x.numpy() * k),
                 dispatch_key="CompositeExplicitAutograd")

        def setup_context(ctx, inputs, output):
            ctx.k = float(inputs[1])

        def backward(ctx, grad):
            return grad * ctx.k, None

        torch.library.register_autograd("%s::f" % ns, backward,
                                        setup_context=setup_context)
        op = getattr(torch.ops, ns).f
        x = jt.array(np.array([1.0, 1.0], dtype="float32"))
        x.requires_grad = True
        g = jt.grad(op(x, 3.0).sum(), [x])[0]
        np.testing.assert_allclose(g.numpy(), [3.0, 3.0], rtol=1e-5)

    def test_saved_tensors_detect_an_inplace_modification(self):
        ns = _fresh_ns()
        lib = torch.library.Library(ns, "DEF")
        lib.define("f(Tensor x) -> Tensor")

        holder = {}

        def forward(x):
            return jt.array(x.numpy() ** 2)

        def setup_context(ctx, inputs, output):
            ctx.save_for_backward(inputs[0])
            holder["ctx"] = ctx

        def backward(ctx, grad):
            return grad * 2.0 * ctx.saved_tensors[0]

        lib.impl("f", forward, dispatch_key="CompositeExplicitAutograd")
        torch.library.register_autograd("%s::f" % ns, backward,
                                        setup_context=setup_context)
        op = getattr(torch.ops, ns).f
        x = jt.array(np.array([3.0], dtype="float32"))
        x.requires_grad = True
        op(x)
        ctx = holder["ctx"]
        saved = ctx.saved_tensors[0]
        saved.update(saved + 1)          # in-place modification after saving
        with self.assertRaises(RuntimeError) as cm:
            ctx.saved_tensors
        self.assertIn("inplace", str(cm.exception))

    def test_non_grad_inputs_skip_the_autograd_wrapper(self):
        ns = _fresh_ns()
        lib = torch.library.Library(ns, "DEF")
        lib.define("f(Tensor x) -> Tensor")
        lib.impl("f", lambda x: x + 1, dispatch_key="CompositeExplicitAutograd")
        torch.library.register_autograd(
            "%s::f" % ns, lambda ctx, g: (_ for _ in ()).throw(
                AssertionError("backward must not be built here")))
        op = getattr(torch.ops, ns).f
        x = jt.zeros(2)
        x.requires_grad = False
        self.assertEqual(op(x).numpy().tolist(), [1.0, 1.0])


class TestIntegrationOverrides(unittest.TestCase):
    def test_generic_custom_op_keeps_the_callers_implementation(self):
        ns = _fresh_ns()

        @torch.library.custom_op("%s::mine" % ns, mutates_args=())
        def mine(x):
            return x + 5

        op = getattr(torch.ops, ns).mine
        self.assertEqual(op(jt.zeros(2)).numpy().tolist(), [5.0, 5.0])
        self.assertIsNone(op._overridden_by_integration)

    def test_library_module_carries_no_model_names(self):
        import inspect
        from jittor.compat.torch import library as library_mod
        source = inspect.getsource(library_mod)
        self.assertNotIn("grouped_mm_fallback", source,
                         "a generic registration API must not name a model op")

    def test_the_integration_override_still_applies(self):
        from jittor.compat.integrations import custom_op_overrides
        overrides = custom_op_overrides()
        self.assertIn("transformers::grouped_mm_fallback", overrides)

        @torch.library.custom_op("transformers::grouped_mm_fallback",
                                 mutates_args=())
        def caller_version(input, weight, offsets):
            raise AssertionError("must be replaced by the integration version")

        op = torch.ops.transformers.grouped_mm_fallback
        self.assertEqual(op._overridden_by_integration,
                         "transformers::grouped_mm_fallback")
        x = jt.array(np.ones((4, 2), dtype="float32"))
        w = jt.array(np.ones((2, 2, 3), dtype="float32"))
        offsets = jt.array(np.array([2, 4], dtype="int32"))
        out = op(x, w, offsets)
        self.assertEqual(tuple(out.shape), (4, 3))
        np.testing.assert_allclose(out.numpy(), np.full((4, 3), 2.0), rtol=1e-5)


if __name__ == "__main__":
    unittest.main()

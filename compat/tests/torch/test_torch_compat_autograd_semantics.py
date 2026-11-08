"""autograd semantics that used to differ silently from real PyTorch.

Task 7.11.  Every expectation below was pinned against real PyTorch 2.12
(a separate interpreter -- the `torch` in this one IS Jittor) before being
written down:

    needs_input_grad, all positional      -> (True, True, False)
    grad(create_graph=False)              -> requires_grad False, grad_fn None
    grad(create_graph=True)               -> requires_grad True
    grad(non-scalar output, no grad_out)  -> RuntimeError
                                             "grad can be implicitly created
                                              only for scalar outputs"
    saved tensor modified in place        -> RuntimeError "... modified by an
                                             inplace operation"
    Tensor.backward signature             -> (gradient=None, retain_graph=None,
                                              create_graph=False, inputs=None)

Run: python -m pytest compat/tests/torch/test_torch_compat_autograd_semantics.py
"""
import inspect
import unittest
import warnings

import numpy as np

import jittor as jt
import torch


class TestPythonScalarsNeverRequireGrad(unittest.TestCase):
    """A Python number in an expression never asks for a gradient, as in torch.

    `(1.0 - mask.float()) * min` is how Transformers turns a padding mask into
    an additive one. The `1.0` became a Var that "required grad", so the mask
    did too, and the fused attention kernels -- which take no mask gradient --
    declined every such mask for the O(L^2) math path.
    """

    def test_a_mask_built_from_an_integer_tensor_does_not_require_grad(self):
        mask = torch.ones(2, 4, dtype=torch.int64)
        additive = (1.0 - mask.float()) * -1e9
        self.assertFalse(additive.requires_grad)
        frozen = torch.randn(2, 4)
        self.assertFalse((2.0 * frozen + 1).requires_grad)

    def test_a_tensor_that_requires_grad_still_does(self):
        x = torch.randn(3, requires_grad=True)
        self.assertTrue((1.0 - x).requires_grad)
        leaf = torch.tensor(2.0, requires_grad=True)
        y = leaf * 3.0
        self.assertTrue(y.requires_grad)
        y.backward()
        self.assertEqual(float(leaf.grad), 3.0)


class TestInPlaceCopyPreservesRequiresGrad(unittest.TestCase):
    def test_copy_under_no_grad_preserves_frozen_and_trainable_parameters(self):
        with jt.flag_scope(use_cuda=0):
            frozen = torch.nn.Parameter(
                torch.tensor(np.array([1.0, 2.0], dtype=np.float32)),
                requires_grad=False,
            )
            trainable = torch.nn.Parameter(
                torch.tensor(np.array([3.0, 4.0], dtype=np.float32)),
                requires_grad=True,
            )

            with torch.no_grad():
                frozen.copy_(torch.tensor(np.array([5.0, 6.0], dtype=np.float32)))
                trainable.copy_(torch.tensor(np.array([7.0, 8.0], dtype=np.float32)))

            self.assertFalse(frozen.requires_grad)
            self.assertTrue(trainable.requires_grad)
            np.testing.assert_array_equal(frozen.numpy(), [5.0, 6.0])
            np.testing.assert_array_equal(trainable.numpy(), [7.0, 8.0])

            (trainable * trainable).sum().backward()
            self.assertIsNotNone(trainable.grad)
            self.assertIsNone(frozen.grad)


class TestNeedsInputGrad(unittest.TestCase):
    """ctx.needs_input_grad: one flag per argument PASSED, and no kwargs.

    The audit listed this as "counts positional arguments only, so a keyword
    call gets misaligned flags". Checked against real torch 2.12: the tuple
    follows the CALL, not the signature -- `apply(a, b, 3.0)` on
    `forward(ctx, a, b, c=1.0)` gives three flags and `apply(a, b)` gives two --
    and `apply()` rejects keyword arguments outright. So the positional tuple
    was already right; only the failure mode of a keyword call differed.
    """

    def _capture(self, call):
        seen = {}

        class F(torch.autograd.Function):
            def forward(ctx, a, b, scale=1.0):
                seen["flags"] = ctx.needs_input_grad
                return a * scale

            def backward(ctx, grad):
                return grad, None, None

        call(F, seen)
        return seen["flags"]

    def test_all_positional_matches_torch(self):
        a = jt.ones(2)
        a.requires_grad = True
        b = jt.ones(2)
        b.requires_grad = True
        flags = self._capture(lambda F, seen: F.apply(a, b, 3.0))
        self.assertEqual(flags, (True, True, False))

    def test_a_defaulted_argument_gets_no_slot_just_like_torch(self):
        a = jt.ones(2)
        a.requires_grad = True
        b = jt.ones(2)
        b.requires_grad = True
        flags = self._capture(lambda F, seen: F.apply(a, b))
        self.assertEqual(flags, (True, True))

    def test_a_frozen_tensor_argument_is_false(self):
        a = jt.ones(2)
        a.requires_grad = True
        b = jt.ones(2)
        b.requires_grad = False
        flags = self._capture(lambda F, seen: F.apply(a, b, 3.0))
        self.assertEqual(flags, (True, False, False))

    def test_a_keyword_call_is_rejected_with_torchs_message(self):
        a = jt.ones(2)
        a.requires_grad = True
        b = jt.ones(2)
        b.requires_grad = True
        with self.assertRaises(TypeError) as cm:
            self._capture(lambda F, seen: F.apply(a, b=b, scale=3.0))
        self.assertIn("no keyword arguments", str(cm.exception))


class TestInPlaceConstantWrites(unittest.TestCase):
    def test_cpu_fill_and_zero_preserve_aliases_and_parameter_gradients(self):
        with jt.flag_scope(use_cuda=0):
            for kind in ("tensor", "linear_weight"):
                for trainable in (False, True):
                    for operation, constant in (("fill_", 2.0), ("zero_", 0.0)):
                        with self.subTest(kind=kind, trainable=trainable, operation=operation):
                            if kind == "linear_weight":
                                module = torch.nn.Linear(2, 3, bias=False)
                                target = module.weight
                                target.requires_grad_(trainable)
                            else:
                                target = torch.ones(
                                    (3, 2), dtype=torch.float32, device="cpu",
                                    requires_grad=trainable)
                            aliases = {
                                "data": target.data,
                                "detach": target.detach(),
                                "view": target.view(6),
                            }
                            pointer = target.data_ptr()
                            alias_pointers = {name: alias.data_ptr()
                                              for name, alias in aliases.items()}
                            alias_grad = {name: bool(alias.requires_grad)
                                          for name, alias in aliases.items()}
                            self.assertTrue(all(value == pointer
                                                for value in alias_pointers.values()))
                            self.assertEqual(target.placement_backend, 0)
                            self.assertEqual(target.device_id, -1)
                            with torch.no_grad():
                                returned = (target.fill_(constant) if operation == "fill_"
                                            else target.zero_())
                            self.assertIs(returned, target)
                            self.assertEqual(target.data_ptr(), pointer)
                            self.assertEqual(target.placement_backend, 0)
                            self.assertEqual(target.device_id, -1)
                            self.assertEqual(target.device.type, "cpu")
                            self.assertEqual(target.dtype, torch.float32)
                            self.assertEqual(bool(target.requires_grad), trainable)
                            np.testing.assert_array_equal(
                                target.detach().numpy(), np.full((3, 2), constant))
                            for name, alias in aliases.items():
                                self.assertEqual(alias.data_ptr(), alias_pointers[name])
                                self.assertEqual(bool(alias.requires_grad), alias_grad[name])
                                np.testing.assert_array_equal(
                                    alias.detach().numpy(),
                                    np.full(tuple(alias.shape), constant))
                            if kind == "linear_weight":
                                inputs = torch.ones((1, 2), device="cpu", requires_grad=True)
                                output = module(inputs)
                                np.testing.assert_array_equal(
                                    output.detach().numpy(), np.full((1, 3), 2 * constant))
                                output.sum().backward()
                                np.testing.assert_array_equal(
                                    inputs.grad.numpy(), np.full((1, 2), 3 * constant))
                                if trainable:
                                    np.testing.assert_array_equal(
                                        target.grad.numpy(), np.ones((3, 2)))
                                else:
                                    self.assertIsNone(target.grad)
                            elif trainable:
                                (target * 3).sum().backward()
                                np.testing.assert_array_equal(
                                    target.grad.numpy(), np.full((3, 2), 3.0))

    def test_internal_constant_writes_preserve_follow_runtime_alias(self):
        # Swap an internal native holder to construct FollowRuntime placement.
        # Public Tensor(native) construction has separate placement semantics.
        from jittor._runtime.fallback import forbid_backend_fallbacks
        from jittor.compat.torch.installers.tensor.method_api import _api_fill, _api_zero
        before_fallback = jt.core.backend_fallback_count()
        with jt.runtime.scope(use_cuda=0, backend_fallback="error"), forbid_backend_fallbacks():
            native = jt.ones((3, 2), dtype="float32").stop_grad()
            alias = native.detach()
            alias_requires_grad = bool(alias.requires_grad)
            target = torch.ones((3, 2), device="cpu")
            self.assertIs(target.swap(native), target)
            self.assertIs(type(target), torch.Tensor)
            self.assertEqual(target.placement_backend, -1)
            self.assertFalse(target.requires_grad)
            self.assertEqual(bool(alias.requires_grad), alias_requires_grad)
            pointer = target._storage_address
            self.assertEqual(alias._storage_address, pointer)
            device_id = target.device_id
            for operation, constant in (("fill", 2.0), ("zero", 0.0)):
                with self.subTest(operation=operation), torch.no_grad():
                    returned = (_api_fill(target, constant) if operation == "fill"
                                else _api_zero(target))
                    self.assertIs(returned, target)
                    self.assertEqual(target.placement_backend, -1)
                    self.assertFalse(target.requires_grad)
                    self.assertEqual(bool(alias.requires_grad), alias_requires_grad)
                    self.assertEqual(target.device_id, device_id)
                    self.assertEqual(target._storage_address, pointer)
                    self.assertEqual(alias._storage_address, pointer)
                    self.assertEqual(tuple(jt.core.dispatch_context([target])), ("cpu", -1))
                    np.testing.assert_array_equal(target.numpy(), np.full((3, 2), constant))
                    np.testing.assert_array_equal(alias.numpy(), np.full((3, 2), constant))
        self.assertEqual(jt.core.backend_fallback_count(), before_fallback)

    def test_public_tensor_constructor_preserves_follow_runtime_alias(self):
        from jittor._runtime.fallback import forbid_backend_fallbacks
        with jt.runtime.scope(use_cuda=0, backend_fallback="error"), forbid_backend_fallbacks():
            native = jt.ones((3, 2), dtype="float32").stop_grad()
            target = torch.Tensor(native)
            self.assertEqual(native.placement_backend, -1)
            self.assertEqual(target.placement_backend, -1)
            explicit_cpu = torch.Tensor(native, device="cpu")
            self.assertEqual(explicit_cpu.placement_backend, 0)
            device_id = native.device_id
            pointer = target.data_ptr()
            self.assertEqual(native._storage_address, pointer)
            for operation, constant in (("fill_", 2.0), ("zero_", 0.0)):
                with self.subTest(operation=operation):
                    with torch.no_grad():
                        returned = (target.fill_(constant) if operation == "fill_"
                                    else target.zero_())
                    self.assertIs(returned, target)
                    self.assertEqual(target.data_ptr(), pointer)
                    self.assertEqual(native._storage_address, pointer)
                    self.assertEqual(target.placement_backend, -1)
                    self.assertEqual(native.placement_backend, -1)
                    self.assertEqual(target.device_id, device_id)
                    self.assertEqual(native.device_id, device_id)
                    np.testing.assert_array_equal(
                        native.numpy(), np.full((3, 2), constant))


    def test_cuda_public_tensor_constructor_preserves_follow_runtime_alias(self):
        from _helpers import capability
        capability.require_accelerator("cuda")
        if not jt.flags.use_cuda:
            self.skipTest("accelerator prerequisite: dedicated CUDA test mode required")
        from jittor._runtime.fallback import forbid_backend_fallbacks
        before_fallback = jt.core.backend_fallback_count()
        with jt.runtime.scope(use_cuda=1, backend_fallback="error"), forbid_backend_fallbacks():
            native = jt.ones((3, 2), dtype="float32").stop_grad()
            target = torch.Tensor(native)
            native.sync()
            target.sync()
            pointer = target.data_ptr()
            self.assertEqual(native._storage_address, pointer)
            self.assertEqual(native.placement_backend, -1)
            self.assertEqual(target.placement_backend, -1)
            self.assertEqual(native.device_id, target.device_id)
            self.assertEqual(tuple(jt.core.dispatch_context([native])), ("cuda", 0))
            self.assertEqual(tuple(jt.core.dispatch_context([target])), ("cuda", 0))
            for operation, constant in (("fill_", 2.0), ("zero_", 0.0)):
                with self.subTest(operation=operation), torch.no_grad():
                    returned = (target.fill_(constant) if operation == "fill_"
                                else target.zero_())
                    self.assertIs(returned, target)
                    target.sync()
                    native.sync()
                    self.assertEqual(target.data_ptr(), pointer)
                    self.assertEqual(target._storage_address, pointer)
                    self.assertEqual(native._storage_address, pointer)
                    self.assertEqual(target.device_id, native.device_id)
                    self.assertEqual(target.placement_backend, -1)
                    self.assertEqual(native.placement_backend, -1)
                    np.testing.assert_array_equal(
                        native.numpy(), np.full((3, 2), constant, dtype=np.float32))
        self.assertEqual(jt.core.backend_fallback_count(), before_fallback)
    def test_cuda_constant_writes_keep_explicit_cpu_and_cuda_storage(self):
        from _helpers import capability
        capability.require_accelerator("cuda")
        if not jt.flags.use_cuda:
            self.skipTest("accelerator prerequisite: dedicated CUDA test mode required")
        from jittor._runtime.fallback import forbid_backend_fallbacks
        with jt.runtime.scope(use_cuda=1, backend_fallback="error"), forbid_backend_fallbacks():
            for device in ("cpu", "cuda:0"):
                for kind in ("tensor", "parameter"):
                    for trainable in (False, True):
                        for operation, constant in (("fill_", 2.0), ("zero_", 0.0)):
                            with self.subTest(device=device, kind=kind, trainable=trainable,
                                              operation=operation):
                                target = torch.ones(
                                    (3, 2), dtype=torch.float32, device=device,
                                    requires_grad=trainable)
                                if kind == "parameter":
                                    target = torch.nn.Parameter(target, requires_grad=trainable)
                                aliases = (target.data, target.detach(), target.view(6))
                                pointer = target.data_ptr()
                                self.assertTrue(all(alias.data_ptr() == pointer for alias in aliases))
                                alias_grad = tuple(bool(alias.requires_grad) for alias in aliases)
                                backend = 0 if device == "cpu" else 1
                                index = -1 if device == "cpu" else 0
                                dispatch = ("cpu" if device == "cpu" else "cuda", index)
                                with torch.device("cuda"), torch.no_grad():
                                    returned = (target.fill_(constant) if operation == "fill_"
                                                else target.zero_())
                                self.assertIs(returned, target)
                                self.assertEqual(bool(target.requires_grad), trainable)
                                for value in (target,) + aliases:
                                    self.assertEqual(value.data_ptr(), pointer)
                                    self.assertEqual(value.placement_backend, backend)
                                    self.assertEqual(value.device_id, index)
                                    self.assertEqual(tuple(jt.core.dispatch_context([value])), dispatch)
                                    np.testing.assert_array_equal(
                                        value.detach().cpu().numpy(),
                                        np.full(tuple(value.shape), constant))
                                self.assertEqual(tuple(bool(alias.requires_grad) for alias in aliases),
                                                 alias_grad)


class TestBackwardLeafAndGradFn(unittest.TestCase):
    """The shim forwards leaf and producer identity to the core graph query."""

    def test_leaf_and_intermediate_report_torch_shape(self):
        # Torch factories cut their internal creation graph before publishing
        # the fresh leaf. Native scalar/JVP semantics remain independent.
        device = "cuda" if jt.introspection.policy.runtime.use_cuda else "cpu"
        x = torch.ones(3, device=device)
        self.assertEqual(x.device.type, device)
        x.requires_grad = True
        y = x * 2

        self.assertTrue(x.is_leaf)
        self.assertIsNone(x.grad_fn)
        # What the distinction is for still holds: an op output is not a leaf
        # and carries the producer identity.
        self.assertFalse(y.is_leaf)
        self.assertIsNotNone(y.grad_fn)
        self.assertEqual(y.grad_fn.node_id, y.grad_fn_node_id)
        self.assertEqual(y.grad_fn.op_id, y.grad_fn_op_id)
        self.assertEqual(y.grad_fn.name, y.grad_fn_name)
        self.assertEqual(y.grad_fn, y.grad_fn)
        y.sum().backward()
        np.testing.assert_array_equal(x.grad.detach().cpu().numpy(), [2., 2., 2.])


class TestCreateGraphVersusRetainGraph(unittest.TestCase):
    """The two were folded into one, so create_graph=False stayed differentiable."""

    def test_create_graph_false_returns_a_detached_gradient(self):
        x = jt.array(np.ones(3, dtype="float32"))
        x.requires_grad = True
        y = (x ** 3).sum()
        g = torch.autograd.grad(y, x, create_graph=False, retain_graph=True)[0]
        self.assertFalse(g.requires_grad)

    def test_create_graph_true_returns_a_differentiable_gradient(self):
        x = jt.array(np.ones(3, dtype="float32"))
        x.requires_grad = True
        y = (x ** 3).sum()
        g = torch.autograd.grad(y, x, create_graph=True)[0]
        self.assertTrue(g.requires_grad)

    def test_second_order_grad_works_with_create_graph(self):
        x = torch.tensor(np.full(3, 2.0, dtype="float32"), requires_grad=True)
        y = (x ** 3).sum()
        g = torch.autograd.grad(y, x, create_graph=True)[0]   # 3x^2
        gg = torch.autograd.grad(g.sum(), x)[0]               # 6x
        np.testing.assert_allclose(gg.numpy(), np.full(3, 12.0), rtol=1e-4)

    def test_retain_graph_alone_does_not_make_grads_differentiable(self):
        x = jt.array(np.ones(3, dtype="float32"))
        x.requires_grad = True
        y = (x ** 3).sum()
        first = torch.autograd.grad(y, x, retain_graph=True)[0]
        second = torch.autograd.grad(y, x)[0]
        self.assertFalse(first.requires_grad)
        np.testing.assert_allclose(first.numpy(), second.numpy())


class TestImplicitGradOutputs(unittest.TestCase):
    """A non-scalar output silently got a grad_output of ones."""

    def test_non_scalar_output_without_grad_outputs_raises(self):
        x = jt.array(np.ones(3, dtype="float32"))
        x.requires_grad = True
        y = x * 2
        with self.assertRaises(RuntimeError) as cm:
            torch.autograd.grad(y, x)
        self.assertIn("scalar outputs", str(cm.exception))

    def test_non_scalar_output_with_grad_outputs_is_fine(self):
        x = jt.array(np.ones(3, dtype="float32"))
        x.requires_grad = True
        y = x * 2
        g = torch.autograd.grad(y, x, grad_outputs=jt.ones(3))[0]
        np.testing.assert_allclose(g.numpy(), np.full(3, 2.0), rtol=1e-5)

    def test_scalar_output_still_works(self):
        x = jt.array(np.ones(3, dtype="float32"))
        x.requires_grad = True
        g = torch.autograd.grad((x * 2).sum(), x)[0]
        np.testing.assert_allclose(g.numpy(), np.full(3, 2.0), rtol=1e-5)

    def test_several_scalar_outputs_are_summed_like_torch(self):
        x = jt.array(np.ones(3, dtype="float32"))
        x.requires_grad = True
        a = (x * 2).sum()
        b = (x * 3).sum()
        g = torch.autograd.grad([a, b], x)[0]
        np.testing.assert_allclose(g.numpy(), np.full(3, 5.0), rtol=1e-5)


class TestSavedTensorVersions(unittest.TestCase):
    """torch raises when a saved tensor is modified in place before backward.

    ``save_for_backward``/``saved_tensors`` live on the ``ctx``, which in torch
    is a fresh object per call and is never the Function instance (torch's
    ``forward`` is a staticmethod; there is no instance to look at). jittor now
    matches that, so these tests capture the ctx from inside ``forward`` and
    assert on it -- reading ``fn.saved_tensors`` off the instance after the
    call would be reading a Function that saved nothing.
    """

    def _function(self, seen=None):
        class Square(torch.autograd.Function):
            def forward(ctx, a):
                if seen is not None:
                    seen.append(ctx)
                ctx.save_for_backward(a)
                return a * a

            def backward(ctx, grad):
                return grad * 2 * ctx.saved_tensors[0]

        return Square

    def test_untouched_saved_tensor_backs_propagates_normally(self):
        Square = self._function()
        x = torch.tensor(np.full(3, 2.0, dtype="float32"), requires_grad=True)
        g = jt.grad(Square.apply(x).sum(), [x])[0]
        np.testing.assert_allclose(g.numpy(), np.full(3, 4.0), rtol=1e-5)

    def test_inplace_modification_after_saving_is_detected(self):
        # jittor tapes a Function's inputs, so what forward saved is the taped
        # Var; this edits that object, which is the case the check can see.
        seen = []
        Square = self._function(seen)
        x = torch.tensor(np.full(3, 2.0, dtype="float32"), requires_grad=True)
        Square()(x)
        ctx = seen[0]
        saved = ctx.saved_tensors[0]
        saved.update(saved * 5)
        with self.assertRaises(RuntimeError) as cm:
            ctx.saved_tensors
        self.assertIn("inplace operation", str(cm.exception))

    def test_reading_the_saved_tensor_does_not_trip_the_check(self):
        seen = []
        Square = self._function(seen)
        x = torch.tensor(np.full(3, 2.0, dtype="float32"), requires_grad=True)
        Square()(x)
        _ = (x * 3).numpy()
        x.sync()
        self.assertEqual(len(seen[0].saved_tensors), 1)

    def test_each_call_of_one_instance_saves_its_own_tensors(self):
        # torch builds a ctx per apply(); two calls never share saved state.
        # When they did, the first call's backward silently used the second
        # call's tensors.
        seen = []
        Square = self._function(seen)
        fn = Square()
        a = jt.array(np.full(3, 2.0, dtype="float32"))
        a.requires_grad = True
        b = jt.array(np.full(3, 5.0, dtype="float32"))
        b.requires_grad = True
        out_a = fn(a)
        out_b = fn(b)
        self.assertEqual(len(seen), 2)
        self.assertIsNot(seen[0], seen[1])
        self.assertIsNot(seen[0], fn)
        np.testing.assert_allclose(seen[0].saved_tensors[0].numpy(),
                                   np.full(3, 2.0), rtol=1e-5)
        np.testing.assert_allclose(seen[1].saved_tensors[0].numpy(),
                                   np.full(3, 5.0), rtol=1e-5)
        # d(a^2)/da = 2a = 4, not 2b = 10
        np.testing.assert_allclose(jt.grad(out_a.sum(), [a])[0].numpy(),
                                   np.full(3, 4.0), rtol=1e-5)
        np.testing.assert_allclose(jt.grad(out_b.sum(), [b])[0].numpy(),
                                   np.full(3, 10.0), rtol=1e-5)


class TestCustomFunctionDoubleBackward(unittest.TestCase):
    """A backward written with differentiable ops differentiates again.

    torch's saved inputs keep their graph, so with ``create_graph=True`` the
    gradient of ``x ** 3`` through a custom Function differentiates to
    ``6 x``. jittor's Function saved stop-grad tapes and gave 0.
    """

    class Cube(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x ** 3

        @staticmethod
        def backward(ctx, grad):
            (x,) = ctx.saved_tensors
            return 3 * x * x * grad

    def test_second_derivative_through_save_for_backward(self):
        x = torch.tensor([2.0], requires_grad=True)
        (first,) = torch.autograd.grad(self.Cube.apply(x).sum(), x,
                                       create_graph=True)
        (second,) = torch.autograd.grad(first.sum(), x)
        np.testing.assert_allclose(first.numpy(), [12.0], rtol=1e-6)
        np.testing.assert_allclose(second.numpy(), [12.0], rtol=1e-6)

    def test_first_order_backward_is_unchanged(self):
        x = torch.tensor([2.0], requires_grad=True)
        self.Cube.apply(x).sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [12.0], rtol=1e-6)


class TestBackwardSignature(unittest.TestCase):
    def test_retain_graph_defaults_to_none_like_torch(self):
        # torch's ``backward`` is the torch frontend's method; the native Var
        # keeps jittor's own stub.
        sig = inspect.signature(torch.Tensor.backward)
        self.assertIsNone(sig.parameters["retain_graph"].default)
        self.assertIs(sig.parameters["create_graph"].default, False)

    def test_create_graph_true_retains_the_graph(self):
        x = torch.tensor(np.full(3, 2.0, dtype="float32"), requires_grad=True)
        y = (x ** 2).sum()
        y.backward(create_graph=True)
        y.backward()                 # would have raised on a freed graph
        self.assertIsNotNone(x.grad)


class TestMismatchedBackwardGrad(unittest.TestCase):
    """A grad whose element count cannot be reduced was silently zeroed."""

    def test_incompatible_grad_shape_warns_before_zeroing(self):
        class Bad(torch.autograd.Function):
            def forward(ctx, a):
                return a * 2

            def backward(ctx, grad):
                return jt.ones((7, 5))       # nothing like the input's shape

        x = jt.array(np.ones(3, dtype="float32"))
        x.requires_grad = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            g = jt.grad(Bad.apply(x).sum(), [x])[0]
        self.assertTrue(any("broadcast-compatible" in str(w.message)
                            for w in caught),
                        "a wrong backward must not be silently zeroed")
        np.testing.assert_allclose(g.numpy(), np.zeros(3), atol=1e-6)


class TestTorchGradientEdgeDtypes(unittest.TestCase):
    """Maintenance cases; R248's immutable full-array protocol is separate."""

    def test_low_raw_sum_to_precedes_fp32_input_dtype_cast(self):
        for dtype, half in ((torch.bfloat16, 2. ** -8),
                            (torch.float16, 2. ** -11)):
            returned = []
            class Expanded(torch.autograd.Function):
                @staticmethod
                def forward(ctx, value):
                    return value.expand(2)

                @staticmethod
                def backward(ctx, incoming):
                    raw = torch.tensor([1., half], dtype=dtype)
                    returned.append(raw)
                    return raw

            value = torch.tensor([.25], dtype=torch.float32, requires_grad=True)
            gradient, = torch.autograd.grad(Expanded.apply(value).sum(), (value,))
            self.assertEqual(len(returned), 1)
            self.assertEqual(returned[0].dtype, dtype)
            np.testing.assert_array_equal(returned[0].float().cpu().numpy(), [1., half])
            self.assertEqual(gradient.dtype, torch.float32)
            np.testing.assert_array_equal(gradient.cpu().numpy(), [1.])
            self.assertEqual(float(returned[0].float().sum().item()), 1. + half)
            self.assertIsNone(value.grad)

    def test_multi_axis_sum_to_does_not_round_between_axis_groups(self):
        for dtype, half in ((torch.bfloat16, 2. ** -8),
                            (torch.float16, 2. ** -11)):
            for control in (False, True):
                small = half * (2 if control else 1)
                returned = []
                class Expanded(torch.autograd.Function):
                    @staticmethod
                    def forward(ctx, value):
                        return value.expand(2, 2, 1)

                    @staticmethod
                    def backward(ctx, incoming):
                        raw = torch.tensor(
                            [[[1.], [-1.]], [[small], [0.]]], dtype=dtype)
                        returned.append(raw)
                        return raw

                value = torch.tensor([[.25]], dtype=torch.float32, requires_grad=True)
                gradient, = torch.autograd.grad(Expanded.apply(value).sum(), (value,))
                self.assertEqual(len(returned), 1)
                self.assertEqual(returned[0].dtype, dtype)
                np.testing.assert_array_equal(
                    returned[0].float().cpu().numpy(),
                    np.array([[[1.], [-1.]], [[small], [0.]]], dtype=np.float32))
                self.assertEqual(gradient.dtype, torch.float32)
                np.testing.assert_array_equal(
                    gradient.cpu().numpy(), np.full((1, 1), small, dtype=np.float32))
                staged = returned[0].sum(dim=0).sum(dim=0, keepdim=True).float()
                self.assertEqual(float(staged.item()), small if control else 0.)
                self.assertIsNone(value.grad)

    def test_one_node_two_outputs_complete_vjp_is_cast_once(self):
        class Pair(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value, positive, negative):
                ctx.save_for_backward(positive, negative)
                return value.float() * positive, value.float() * negative

            @staticmethod
            def backward(ctx, first, second):
                positive, negative = ctx.saved_tensors
                return first * positive + second * negative, None, None

        for dtype, half in ((torch.bfloat16, 2. ** -8),
                            (torch.float16, 2. ** -11)):
            value = torch.tensor([.25, .5], dtype=dtype, requires_grad=True)
            positive = torch.tensor([1. + half], dtype=torch.float32)
            negative = torch.tensor([-1.], dtype=torch.float32)
            first, second = Pair.apply(value, positive, negative)
            gradient, = torch.autograd.grad((first + second).sum(), (value,))
            self.assertEqual(gradient.dtype, dtype)
            np.testing.assert_array_equal(gradient.float().cpu().numpy(), [half, half])

    def test_each_edge_cast_and_shared_wide_cast_have_distinct_results(self):
        class FP32VJP(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value, coefficient):
                ctx.save_for_backward(coefficient)
                return value.float() * coefficient

            @staticmethod
            def backward(ctx, upstream):
                coefficient, = ctx.saved_tensors
                return upstream * coefficient, None

        for dtype, half_ulp in ((torch.bfloat16, 2. ** -8),
                                (torch.float16, 2. ** -11)):
            for family in ("custom", "ordinary_mixed",
                           "ordinary_separate_cast", "ordinary_shared_cast"):
                for reverse in (False, True):
                    with self.subTest(dtype=dtype, family=family, reverse=reverse):
                        value = torch.tensor([.25], dtype=dtype, requires_grad=True)
                        positive = torch.tensor([1. + half_ulp], dtype=torch.float32)
                        negative = torch.tensor([-1.], dtype=torch.float32)
                        if family == "custom":
                            left = FP32VJP.apply(value, positive)
                            right = FP32VJP.apply(value, negative)
                        elif family == "ordinary_mixed":
                            left, right = value * positive, value * negative
                        elif family == "ordinary_separate_cast":
                            left, right = value.float() * positive, value.float() * negative
                        else:
                            wide = value.float()
                            left, right = wide * positive, wide * negative
                        loss = (right + left).sum() if reverse else (left + right).sum()
                        gradient, = torch.autograd.grad(loss, (value,))
                        expected = half_ulp if family == "ordinary_shared_cast" else 0.
                        self.assertEqual(gradient.dtype, dtype)
                        self.assertEqual(float(gradient.float().item()), expected)

    def test_grouped_multiple_outputs_and_inputs_normalize_each_edge(self):
        class PairVJP(torch.autograd.Function):
            @staticmethod
            def forward(ctx, first, second, coefficient):
                ctx.save_for_backward(coefficient)
                return first.float() * coefficient, second.float() * coefficient

            @staticmethod
            def backward(ctx, first_gradient, second_gradient):
                coefficient, = ctx.saved_tensors
                return first_gradient * coefficient, second_gradient * coefficient, None

        for dtype, half_ulp in ((torch.bfloat16, 2. ** -8),
                                (torch.float16, 2. ** -11)):
            with self.subTest(dtype=dtype):
                first = torch.tensor([.25], dtype=dtype, requires_grad=True)
                second = torch.tensor([.5], dtype=dtype, requires_grad=True)
                positive = torch.tensor([1. + half_ulp], dtype=torch.float32)
                negative = torch.tensor([-1.], dtype=torch.float32)
                lp, rp = PairVJP.apply(first, second, positive)
                ln, rn = PairVJP.apply(first, second, negative)
                gradients = torch.autograd.grad((lp + rp + ln + rn).sum(), (first, second))
                for gradient in gradients:
                    self.assertEqual(gradient.dtype, dtype)
                    self.assertEqual(float(gradient.float().item()), 0.)

    def test_repeated_input_slots_are_distinct_dtype_edges(self):
        class RepeatedInput(torch.autograd.Function):
            @staticmethod
            def forward(ctx, first, second, positive, negative):
                ctx.save_for_backward(positive, negative)
                return first.float() * positive + second.float() * negative

            @staticmethod
            def backward(ctx, upstream):
                positive, negative = ctx.saved_tensors
                return upstream * positive, upstream * negative, None, None

        for dtype, half_ulp in ((torch.bfloat16, 2. ** -8),
                                (torch.float16, 2. ** -11)):
            with self.subTest(dtype=dtype):
                value = torch.tensor([.25], dtype=dtype, requires_grad=True)
                positive = torch.tensor([1. + half_ulp], dtype=torch.float32)
                negative = torch.tensor([-1.], dtype=torch.float32)
                gradient, = torch.autograd.grad(
                    RepeatedInput.apply(value, value, positive, negative).sum(), (value,))
                self.assertEqual(gradient.dtype, dtype)
                self.assertEqual(float(gradient.float().item()), 0.)

    def test_fp32_leaf_fan_in_is_not_reduced_to_forward_amp_dtype(self):
        for dtype, half_ulp in ((torch.bfloat16, 2. ** -8),
                                (torch.float16, 2. ** -11)):
            with self.subTest(dtype=dtype):
                weight = torch.eye(2, dtype=torch.float32, requires_grad=True)
                unit = torch.eye(2, dtype=torch.float32)
                small = unit * half_ulp
                # Two distinct cast edges into the FP32 leaf are required.
                with torch.autocast("cpu", dtype=dtype, cache_enabled=False):
                    loss = (weight @ unit + weight @ small).sum()
                gradient, = torch.autograd.grad(loss, (weight,))
                self.assertEqual(gradient.dtype, torch.float32)
                np.testing.assert_array_equal(
                    gradient.cpu().numpy(), np.full((2, 2), 1. + half_ulp, dtype=np.float32))

    def test_policy_restores_on_nested_scope_and_backward_exception(self):
        owner = inspect.getmodule(torch.autograd.grad)
        setter = jt.core._set_torch_grad_edge_dtypes
        original = setter(False)
        try:
            with owner._torch_grad_edge_dtypes():
                self.assertTrue(setter(True))
                with self.assertRaisesRegex(RuntimeError, "nested-edge-sentinel"):
                    with owner._torch_grad_edge_dtypes():
                        raise RuntimeError("nested-edge-sentinel")
                self.assertTrue(setter(True))
            self.assertFalse(setter(False))
            class Reject(torch.autograd.Function):
                @staticmethod
                def forward(ctx, value):
                    return value * 2

                @staticmethod
                def backward(ctx, gradient):
                    raise RuntimeError("backward-edge-sentinel")

            value = torch.tensor([2.], requires_grad=True)
            with self.assertRaisesRegex(RuntimeError, "backward-edge-sentinel"):
                torch.autograd.grad(Reject.apply(value).sum(), (value,))
            self.assertFalse(setter(False))
        finally:
            setter(original)


if __name__ == "__main__":
    unittest.main()

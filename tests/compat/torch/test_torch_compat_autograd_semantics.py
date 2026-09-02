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

Run: python -m pytest tests/compat/torch/test_torch_compat_autograd_semantics.py
"""
import inspect
import unittest
import warnings

import numpy as np

import jittor as jt
import jittor as torch


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
        x = jt.array(np.full(3, 2.0, dtype="float32"))
        x.requires_grad = True
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
        x = jt.array(np.full(3, 2.0, dtype="float32"))
        x.requires_grad = True
        g = jt.grad(Square.apply(x).sum(), [x])[0]
        np.testing.assert_allclose(g.numpy(), np.full(3, 4.0), rtol=1e-5)

    def test_inplace_modification_after_saving_is_detected(self):
        # jittor tapes a Function's inputs, so what forward saved is the taped
        # Var; this edits that object, which is the case the check can see.
        seen = []
        Square = self._function(seen)
        x = jt.array(np.full(3, 2.0, dtype="float32"))
        x.requires_grad = True
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
        x = jt.array(np.full(3, 2.0, dtype="float32"))
        x.requires_grad = True
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


class TestBackwardSignature(unittest.TestCase):
    def test_retain_graph_defaults_to_none_like_torch(self):
        sig = inspect.signature(jt.Var.backward)
        self.assertIsNone(sig.parameters["retain_graph"].default)
        self.assertIs(sig.parameters["create_graph"].default, False)

    def test_create_graph_true_retains_the_graph(self):
        x = jt.array(np.full(3, 2.0, dtype="float32"))
        x.requires_grad = True
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


if __name__ == "__main__":
    unittest.main()

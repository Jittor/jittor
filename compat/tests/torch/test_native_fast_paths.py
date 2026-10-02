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

import unittest

import numpy as np
import torch

import jittor as jt


def _cuda():
    return "cuda" if torch.cuda.is_available() else "cpu"


class TestFastCat(unittest.TestCase):
    def test_values_dims_and_dtypes(self):
        device = _cuda()
        rng = np.random.RandomState(0)
        for dtype in (torch.float32, torch.float16, torch.int64, torch.bool):
            for dim in (0, 1, -1):
                with self.subTest(dtype=dtype, dim=dim):
                    a_np = rng.rand(2, 3, 4) > 0.5 if dtype is torch.bool else rng.rand(2, 3, 4) * 10
                    b_np = rng.rand(2, 3, 4) > 0.5 if dtype is torch.bool else rng.rand(2, 3, 4) * 10
                    a = torch.tensor(a_np, device=device).to(dtype)
                    b = torch.tensor(b_np, device=device).to(dtype)
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
        self.assertIsNone(jt.core._fast_binary(h, 2.0, 6))

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


class TestNativeRules(unittest.TestCase):
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

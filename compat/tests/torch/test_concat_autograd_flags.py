"""Concat scratch must not turn stopped inference values into trainable data."""
import unittest
import sys

import torch


class TestConcatAutogradFlags(unittest.TestCase):
    def _prove(self, tensor, device):
        self.assertEqual(tensor.device.type, device)
        backend = sys.modules.get('jittor')
        if backend is not None and backend.core._is_tensor_carrier(tensor):
            tensor.sync()
            self.assertEqual(backend.core.dispatch_context([tensor]), (device, 0 if device == 'cuda' else -1))
            self.assertEqual(tensor.placement_backend, 1 if device == 'cuda' else 0)
            self.assertEqual(tensor.location(), 'device' if device == 'cuda' else 'cpu')
            self.assertEqual(backend.core.backend_fallback_count(), 0)
        elif device == 'cuda':
            self.assertTrue(tensor.is_cuda)
            torch.cuda.synchronize(tensor.device)

    def _check(self, device):
        for op in (torch.cat, torch.stack):
            for count in (2, 65):
                for container in (list, tuple):
                    stopped = torch.tensor([1., 2.], device=device, requires_grad=False)
                    output = op(container([stopped] * count), dim=0)
                    self.assertFalse(output.requires_grad)
                    self._prove(output, device)
                    self.assertFalse(stopped.requires_grad)
                    trainable = torch.tensor([1., 2.], device=device, requires_grad=True)
                    output = op(container([trainable] * count), dim=0)
                    self.assertTrue(output.requires_grad)
                    gradient, = torch.autograd.grad(
                        output, (trainable,), grad_outputs=torch.ones_like(output), allow_unused=False)
                    self._prove(gradient, device)
                    self.assertTrue(bool((gradient == count).all().item()))
                    self.assertTrue(trainable.requires_grad)
                    output = op([stopped, trainable], dim=0)
                    gradient, = torch.autograd.grad(
                        output, (trainable,), grad_outputs=torch.ones_like(output), allow_unused=False)
                    self._prove(gradient, device)
                    self.assertTrue(bool((gradient == 1).all().item()))
                    with torch.no_grad():
                        output = op([trainable, trainable], dim=0)
                    self.assertFalse(output.requires_grad)
                    self.assertTrue(trainable.requires_grad)
            empty = torch.empty((0, 2), device=device, requires_grad=False)
            self.assertFalse(op([empty, empty], dim=0).requires_grad)

    def _nonconstant_gradients(self, device):
        for op in (torch.cat, torch.stack):
            left = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device=device, requires_grad=True)
            right = torch.tensor([[6., 5., 4.], [3., 2., 1.]], device=device, requires_grad=True)
            output = op([left, right], dim=-1)
            cotangent = torch.arange(output.numel(), dtype=torch.float32, device=device).reshape(output.shape) / 8
            expected = ((cotangent[:, :3], cotangent[:, 3:]) if op is torch.cat else
                        (cotangent[..., 0], cotangent[..., 1]))
            gradients = torch.autograd.grad(output, (left, right), grad_outputs=cotangent, allow_unused=False)
            for gradient, reference in zip(gradients, expected):
                self._prove(gradient, device)
                self.assertTrue(torch.allclose(gradient, reference, atol=1e-5, rtol=1e-5))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
    def test_cuda_stack_int64_collate_carriers(self):
        for container in (list, tuple):
            for dim in (0, -1):
                left = torch.tensor([[(1 << 40) + 1, -7, 0], [9, 13, -(1 << 40)]],
                                    dtype=torch.int64, device='cuda')
                right = torch.tensor([[2, 5, -(1 << 39)], [(1 << 39) + 3, -1, 4]],
                                     dtype=torch.int64, device='cuda')
                output = torch.stack(container((left, right)), dim=dim)
                self.assertEqual(output.dtype, torch.int64)
                self.assertFalse(output.requires_grad)
                for value in (left, right, output):
                    self._prove(value, 'cuda')
                if dim == 0:
                    expected = [[[(1 << 40) + 1, -7, 0], [9, 13, -(1 << 40)]],
                                [[2, 5, -(1 << 39)], [(1 << 39) + 3, -1, 4]]]
                else:
                    expected = [[[(1 << 40) + 1, 2], [-7, 5], [0, -(1 << 39)]],
                                [[9, (1 << 39) + 3], [13, -1], [-(1 << 40), 4]]]
                reference = torch.tensor(expected, dtype=torch.int64, device='cuda')
                self._prove(reference, 'cuda')
                self.assertTrue(torch.equal(output, reference))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
    def test_cuda_stack_owned_carriers_full_input_vjp(self):
        for container in (list, tuple):
            for dim in (0, -1):
                left = torch.tensor([[1., -2., 3.], [4., -5., 6.]],
                                    device='cuda', requires_grad=True)
                right = torch.tensor([[-6., 5., -4.], [3., -2., 1.]],
                                     device='cuda', requires_grad=True)
                output = torch.stack(container((left, right)), dim=dim)
                cotangent = torch.arange(12, dtype=torch.float32, device='cuda').reshape(output.shape) / 8
                expected = ((cotangent[0], cotangent[1]) if dim == 0 else
                            (cotangent[..., 0], cotangent[..., 1]))
                gradients = torch.autograd.grad(output, (left, right), grad_outputs=cotangent,
                                                allow_unused=False)
                self.assertEqual(len(gradients), 2)
                for value in (left, right, output, cotangent) + gradients:
                    self._prove(value, 'cuda')
                self.assertTrue(left.requires_grad and right.requires_grad and output.requires_grad)
                for gradient, reference in zip(gradients, expected):
                    self.assertEqual(gradient.shape, reference.shape)
                    self.assertTrue(torch.allclose(gradient, reference, atol=1e-5, rtol=1e-5))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
    def test_cuda_array_owned_carrier_clone_cast_full_vjp(self):
        source = torch.tensor([[1., -2., 3.], [4., -5., 6.]],
                              dtype=torch.float32, device='cuda', requires_grad=True)
        backend = sys.modules.get('jittor')
        candidate = backend is not None and backend.core._is_tensor_carrier(source)
        for dtype in (None, 'float64'):
            if candidate:
                # Exercise the actual native factory called by original stack.
                output = backend.array(source, dtype=dtype)
            else:
                output = source.clone() if dtype is None else source.to(dtype=torch.float64)
            self._prove(source, 'cuda')
            self._prove(output, 'cuda')
            expected_dtype = torch.float32 if dtype is None else torch.float64
            self.assertEqual(output.dtype, expected_dtype)
            self.assertTrue(output.requires_grad)
            if dtype is None:
                self.assertIsNot(output, source)
                self.assertNotEqual(output.data_ptr(), source.data_ptr())
            reference = source.to(dtype=expected_dtype)
            self.assertTrue(torch.equal(output, reference))
            cotangent = torch.tensor([[.25, -.5, .75], [-1., 1.25, -1.5]],
                                     dtype=expected_dtype, device='cuda')
            gradient, = torch.autograd.grad(output, (source,), grad_outputs=cotangent,
                                           allow_unused=False)
            self._prove(gradient, 'cuda')
            self.assertEqual(gradient.dtype, source.dtype)
            self.assertTrue(torch.allclose(gradient, cotangent.to(dtype=source.dtype),
                                          atol=1e-5, rtol=1e-5))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
    def test_cuda_autograd_owned_carrier_direct_and_duplicate_inputs(self):
        # This method is inserted into TestConcatAutogradFlags, which owns _prove.
        def check_gradient(gradient, expected):
            self._prove(gradient, 'cuda')
            self._prove(expected, 'cuda')
            self.assertEqual(gradient.shape, expected.shape)
            self.assertEqual(gradient.dtype, expected.dtype)
            self.assertFalse(gradient.requires_grad)
            self.assertTrue(gradient.is_leaf)
            self.assertIsNone(gradient.grad_fn)
            self.assertTrue(torch.allclose(gradient, expected, atol=1e-5, rtol=1e-5))

        # Both scalar output and input are passed directly, not wrapped in tuples.
        x = torch.tensor([1., -2., 3.], device='cuda', requires_grad=True)
        y = x.square().sum()
        gradient, = torch.autograd.grad(y, x, create_graph=False, allow_unused=False)
        for value in (x, y):
            self._prove(value, 'cuda')
        check_gradient(gradient, torch.tensor([2., -4., 6.], device='cuda'))
        self.assertTrue(x.requires_grad)
        self.assertIsNone(x.grad)

        # A carrier must not bypass the non-scalar implicit-cotangent rejection.
        x = torch.tensor([1., -2., 3.], device='cuda', requires_grad=True)
        y = x.square()
        with self.assertRaises(RuntimeError):
            torch.autograd.grad(y, x, create_graph=False, allow_unused=False)
        self._prove(x, 'cuda')
        self._prove(y, 'cuda')
        self.assertIsNone(x.grad)

        # A direct Tensor grad_outputs is one complete cotangent, not an iterable.
        x = torch.tensor([1., -2., 3.], device='cuda', requires_grad=True)
        y = x.square()
        cotangent = torch.tensor([.5, -.25, 2.], device='cuda')
        gradient, = torch.autograd.grad(y, x, grad_outputs=cotangent,
                                       create_graph=False, allow_unused=False)
        for value in (x, y, cotangent):
            self._prove(value, 'cuda')
        check_gradient(gradient, torch.tensor([1., 1., 12.], device='cuda'))
        self.assertIsNone(x.grad)

        # Two positions alias the same public input; preserve both full gradients
        # and the existing one-result-per-node public-object identity contract.
        x = torch.tensor([1., -2., 3.], device='cuda', requires_grad=True)
        alias = x
        gradients = torch.autograd.grad(x.square().sum(), (x, alias),
                                        create_graph=False, allow_unused=False)
        self.assertEqual(len(gradients), 2)
        self.assertIs(gradients[0], gradients[1])
        for gradient in gradients:
            check_gradient(gradient, torch.tensor([2., -4., 6.], device='cuda'))
        self._prove(x, 'cuda')
        self.assertTrue(x.requires_grad)
        self.assertIsNone(x.grad)

    def test_cpu(self):
        self._check('cpu')
        self._nonconstant_gradients('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
    def test_cuda(self):
        self._check('cuda')
        self._nonconstant_gradients('cuda')


if __name__ == '__main__':
    unittest.main()

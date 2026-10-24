"""Public clipping/optimizer gradient storage regressions, one device/process.

Set JITTOR_GRAD_STORAGE_RUNTIME=oracle for an independent binary Torch run.
Full arrays stay in RAM. No model, VERL checkout, network, or performance claim.
"""
import os
import unittest

import numpy as np

_ORACLE = os.environ.get('JITTOR_GRAD_STORAGE_RUNTIME') == 'oracle'
if not _ORACLE:
    import jittor as jt
import torch
from torch.nn.utils.clip_grad import _get_total_norm, _clip_grads_with_norm_


class TestClipGradientStorage(unittest.TestCase):
    def setUp(self):
        self.device = os.environ.get('JITTOR_TEST_DEVICES', 'cpu')
        self.assertIn(self.device, ('cpu', 'cuda'))
        if _ORACLE:
            self.assertFalse(hasattr(torch, '_torch_compat_install_context'))
            if self.device == 'cuda' and not torch.cuda.is_available():
                self.skipTest('accelerator prerequisite: independent PyTorch CUDA unavailable')
        else:
            if self.device == 'cuda' and not jt.compiler.has_cuda:
                self.skipTest('accelerator prerequisite: candidate CUDA compiler unavailable')
            scope = jt.runtime.scope(use_cuda=int(self.device == 'cuda'),
                device_id=0 if self.device == 'cuda' else -1, backend_fallback='error')
            scope.__enter__()
            self.addCleanup(scope.__exit__, None, None, None)
            self.assertEqual(jt.core.backend_fallback_count(), 0)
            self.addCleanup(lambda: self.assertEqual(jt.core.backend_fallback_count(), 0))

    def resident(self, value):
        self.assertEqual(value.device.type, self.device)
        if not _ORACLE:
            native = getattr(value, '_var', value)
            native.sync()
            self.assertEqual(jt.core.dispatch_context([native]),
                (self.device, 0 if self.device == 'cuda' else -1))
            self.assertEqual(native.location(), 'device' if self.device == 'cuda' else 'cpu')
        elif self.device == 'cuda':
            torch.cuda.synchronize(value.device)

    def host(self, value):
        self.resident(value)
        with torch.no_grad():
            result = np.array(value.detach().cpu().numpy(), copy=True)
        self.resident(value)
        return result

    def test_clipping_keeps_public_gradient_storage(self):
        for adamw in (False, True):
            for dtype in (torch.float32, torch.float64):
                for layout in ('dense', 'transpose', 'offset_slice'):
                    with self.subTest(adamw=adamw, dtype=str(dtype), layout=layout):
                        parameter = torch.nn.Parameter(torch.zeros((2, 3), dtype=dtype, device=self.device))
                        opt = torch.optim.AdamW([parameter], lr=.001) if adamw else None
                        make = lambda x: torch.tensor(x, dtype=dtype, device=self.device)
                        if layout == 'dense':
                            gradient = make([[1., 2., 3.], [4., 5., 6.]])
                        elif layout == 'transpose':
                            gradient = make([[1., 4.], [2., 5.], [3., 6.]]).transpose(0, 1)
                        else:
                            gradient = make([[9., 1., 2., 3., 9.], [9., 4., 5., 6., 9.]])[:, 1:4]
                        parameter.grad = gradient
                        gradient = parameter.grad
                        alias = gradient.detach()
                        pointer, alias_pointer = gradient.data_ptr(), alias.data_ptr()
                        strides, offset = tuple(gradient.stride()), gradient.storage_offset()
                        pending = gradient * 3.0
                        total = _get_total_norm([gradient], 2.0, False, None)
                        _clip_grads_with_norm_([parameter], 1.0, total, None)
                        expected = np.arange(1, 7, dtype=np.float64).reshape(2, 3)
                        scale = 1.0 / (np.linalg.norm(expected) + 1e-6)
                        self.assertTrue(bool(np.allclose(self.host(gradient), expected * scale, atol=1e-5, rtol=1e-5)))
                        self.assertTrue(bool(np.allclose(self.host(alias), expected * scale, atol=1e-5, rtol=1e-5)))
                        self.assertTrue(bool(np.array_equal(self.host(pending), expected * 3.0)))
                        self.assertTrue(bool(np.array_equal(self.host(parameter), np.zeros((2, 3)))))
                        self.assertIs(parameter.grad, gradient)
                        self.assertEqual(gradient.data_ptr(), pointer)
                        self.assertEqual(alias.data_ptr(), alias_pointer)
                        self.assertEqual(gradient.data_ptr(), alias.data_ptr())
                        self.assertEqual(tuple(gradient.stride()), strides)
                        self.assertEqual(gradient.storage_offset(), offset)
                        self.assertFalse(gradient.requires_grad)
                        self.assertEqual(gradient.device, parameter.device)
                        del opt

    def test_optimizer_backward_preserves_gradient_layout_and_alias(self):
        for dtype in (torch.float32, torch.float64):
            for transposed in (False, True):
                with self.subTest(dtype=str(dtype), transposed=transposed):
                    value = torch.tensor([[1., 2.], [3., 4.]], dtype=dtype, device=self.device)
                    parameter = torch.nn.Parameter(value.transpose(0, 1) if transposed else value)
                    opt = torch.optim.AdamW([parameter], lr=.001)
                    (parameter.sum(dim=1) * 3.0).sum().backward()
                    gradient = parameter.grad
                    self.assertIsNotNone(gradient)
                    self.assertEqual(tuple(gradient.stride()), tuple(parameter.stride()))
                    pointer = gradient.data_ptr()
                    alias = gradient.detach()
                    self.assertTrue(bool(np.array_equal(self.host(gradient), np.full((2, 2), 3.0))))
                    (parameter.sum(dim=1) * 3.0).sum().backward()
                    self.assertIs(parameter.grad, gradient)
                    self.assertEqual(gradient.data_ptr(), pointer)
                    self.assertEqual(tuple(gradient.stride()), tuple(parameter.stride()))
                    self.assertTrue(bool(np.array_equal(self.host(gradient), np.full((2, 2), 6.0))))
                    self.assertTrue(bool(np.array_equal(self.host(alias), np.full((2, 2), 6.0))))
                    self.assertFalse(gradient.requires_grad)
                    del opt

    def test_expanded_gradient_rejected_before_storage_write(self):
        base = torch.tensor([[1., 2., 3.]], dtype=torch.float32, device=self.device)
        gradient = base.expand(2, 3)
        parameter = torch.nn.Parameter(torch.zeros((2, 3), device=self.device))
        parameter.grad = gradient
        original = self.host(base)
        total = _get_total_norm([gradient], 2.0, False, None)
        with self.assertRaises(RuntimeError):
            _clip_grads_with_norm_([parameter], 1.0, total, None)
        self.assertTrue(bool(np.array_equal(self.host(base), original)))
        self.assertIs(parameter.grad, gradient)

    def test_optimizer_create_graph_keeps_second_order_gradient(self):
        parameter = torch.nn.Parameter(torch.tensor([[1., 2.], [3., 4.]], device=self.device))
        opt = torch.optim.AdamW([parameter], lr=.001)
        (parameter ** 3).sum().backward(create_graph=True)
        self.assertTrue(parameter.grad.requires_grad)
        second, = torch.autograd.grad(parameter.grad.sum(), (parameter,))
        self.assertTrue(bool(np.array_equal(self.host(second), np.array([[6., 12.], [18., 24.]]))))
        parameter.grad = None
        del opt

    def test_public_backward_clip_3_4_5_preserves_gradient(self):
        parameter = torch.nn.Parameter(torch.tensor([.25, -.5], device=self.device))
        cotangent = torch.tensor([3., 4.], device=self.device)
        (parameter * cotangent).sum().backward()
        gradient = parameter.grad
        self.assertIsNotNone(gradient)
        alias = gradient.detach()
        pointer = gradient.data_ptr()
        total = torch.nn.utils.clip_grad_norm_([parameter], 1.0)
        self.assertIsInstance(total, torch.Tensor)
        self.resident(total)
        np.testing.assert_array_equal(self.host(total), np.asarray(5., dtype=np.float32))
        expected = np.asarray([3., 4.], dtype=np.float32) / np.float32(5. + 1e-6)
        np.testing.assert_allclose(self.host(gradient), expected, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(self.host(alias), expected, atol=1e-5, rtol=1e-5)
        self.assertIs(parameter.grad, gradient)
        self.assertEqual(gradient.data_ptr(), pointer)
        self.assertEqual(alias.data_ptr(), pointer)
        np.testing.assert_array_equal(self.host(parameter), np.asarray([.25, -.5], dtype=np.float32))
        np.testing.assert_array_equal(self.host(cotangent), np.asarray([3., 4.], dtype=np.float32))

    def test_public_nonfinite_classification_preserves_source(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=str(dtype)):
                values = np.asarray([0., np.nan, np.inf, -np.inf, 2.],
                                    dtype=np.float32 if dtype == torch.float32 else np.float64)
                source = torch.tensor(values, dtype=dtype, device=self.device)
                pointer = source.data_ptr()
                for function, expected in ((torch.isnan, np.isnan(values)),
                                           (torch.isinf, np.isinf(values)),
                                           (torch.isfinite, np.isfinite(values))):
                    result = function(source)
                    self.assertIsInstance(result, torch.Tensor)
                    self.assertIs(result.dtype, torch.bool)
                    np.testing.assert_array_equal(self.host(result), expected)
                    self.assertEqual(self.host(source).tobytes(), values.tobytes())
                    self.assertEqual(source.data_ptr(), pointer)


if __name__ == '__main__':
    unittest.main()

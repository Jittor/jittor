"""Public jagged_dim=2 padding layout, storage and complete packed input VJP.

CPU and CUDA selectors are independent; CUDA evidence selects only test_cuda.
The stride contract was reproduced against independent PyTorch CUDA, not
inferred from equal values or a contiguous layout hint.
"""
from contextlib import ExitStack
import os
import sys
import unittest

import numpy as np


class TestNestedPaddingLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.jt = None
        if os.environ.get('JITTOR_TORCH_SHIM') == '1':
            import jittor as jt
            from jittor.compat.torch import namespace_owner
            from jittor.compat.shim.runtime import activation_status
            assert namespace_owner(torch) is jt and activation_status(jt).active
            cls.jt = jt
        else:
            assert hasattr(torch, '_C') and not hasattr(torch, '_torch_compat_install_context')
            assert not any(n == 'jittor' or n.startswith('jittor.') for n in sys.modules)

    def _resident(self, tensor, device):
        self.assertEqual(tensor.device.type, device)
        if device == 'cuda':
            self.assertEqual(tensor.device.index, 0)
        if self.jt is not None:
            native = getattr(tensor, '_var', tensor)
            native.sync()
            ordinal = 0 if device == 'cuda' else -1
            self.assertEqual(self.jt.core.dispatch_context([native]), (device, ordinal))
            self.assertEqual(int(native.device_id), ordinal)
            self.assertEqual(int(native.placement_backend), int(device == 'cuda'))
            self.assertEqual(native.location(), 'device' if device == 'cuda' else 'cpu')

    def _snapshot(self, value, device):
        self._resident(value, device)
        with self.torch.no_grad():
            copy = value.clone()
            self._resident(copy, device)
            result = np.array(copy.cpu().numpy(), copy=True)
        self._resident(value, device)
        return result

    def _check(self, device):
        torch = self.torch
        if device == 'cuda' and not torch.cuda.is_available():
            self.skipTest('CUDA unavailable; no CUDA claim')
        with ExitStack() as stack:
            if self.jt is not None:
                from jittor._runtime.fallback import forbid_backend_fallbacks
                flags = dict(use_cuda=int(device == 'cuda'), backend_fallback='error')
                if device == 'cuda':
                    flags['device_id'] = 0
                stack.enter_context(self.jt.runtime.scope(**flags))
                stack.enter_context(forbid_backend_fallbacks())
                before = self.jt.core.backend_fallback_count()
                self.assertEqual(before, 0)
            target = 'cuda:0' if device == 'cuda' else 'cpu'
            for lengths in ((5,), (3, 5)):
                offsets = np.cumsum([0] + list(lengths), dtype=np.int64)
                for trainable in (False, True):
                    with self.subTest(lengths=lengths, trainable=trainable):
                        dtype = torch.float32 if trainable else torch.int64
                        np_dtype = np.float32 if trainable else np.int64
                        source = np.arange(4 * int(offsets[-1]), dtype=np_dtype).reshape(4, -1)
                        packed = torch.tensor(source, dtype=dtype, device=target, requires_grad=trainable)
                        off = torch.tensor(offsets, dtype=torch.int64, device=target)
                        nested = torch.nested.nested_tensor_from_jagged(packed, off, jagged_dim=2)
                        padded = torch.nested.to_padded_tensor(nested, -7, output_size=(len(lengths), 4, 5))
                        root = padded.transpose(0, 1)
                        text, rotary = root[0], root[1:]
                        objects = (padded, root, text, rotary)
                        wanted = ((20, 1, 4), (1, 20, 4), (20, 4), (1, 20, 4))
                        for tensor, stride in zip(objects, wanted):
                            self._resident(tensor, device)
                            self.assertEqual(tuple(tensor.stride()), stride)
                            self.assertEqual(tensor.requires_grad, trainable)
                            self.assertEqual(tensor.is_leaf, not trainable)
                            self.assertEqual(tensor.grad_fn is None, not trainable)
                        self.assertEqual(rotary.storage_offset(), 1)
                        origins = {t.data_ptr() - t.storage_offset() * t.element_size() for t in objects}
                        self.assertEqual(len(origins), 1)
                        self.assertNotIn(packed.data_ptr(), origins)
                        expected = np.full((len(lengths), 4, 5), -7, dtype=np_dtype)
                        for index, length in enumerate(lengths):
                            expected[index, :, :length] = source[:, offsets[index]:offsets[index + 1]]
                        np.testing.assert_array_equal(self._snapshot(padded, device), expected)
                        np.testing.assert_array_equal(self._snapshot(text, device), expected[:, 0, :])
                        np.testing.assert_array_equal(self._snapshot(rotary, device), expected.transpose(1, 0, 2)[1:])
                        if trainable:
                            cot = np.arange(1, expected.size + 1, dtype=np.float32).reshape(expected.shape)
                            dy = torch.tensor(cot, dtype=torch.float32, device=target)
                            gradient, = torch.autograd.grad(padded, (packed,), grad_outputs=dy)
                            expected_grad = np.concatenate([cot[i, :, :length] for i, length in enumerate(lengths)], axis=1)
                            self._resident(dy, device)
                            np.testing.assert_array_equal(self._snapshot(gradient, device), expected_grad)
                            self.assertIsNone(packed.grad)
                        np.testing.assert_array_equal(self._snapshot(packed, device), source)
                        for tensor, stride in zip(objects, wanted):
                            self._resident(tensor, device)
                            self.assertEqual(tuple(tensor.stride()), stride)
            if self.jt is not None:
                self.jt.sync_all(True)
                self.assertEqual(self.jt.core.backend_fallback_count(), before)

    def test_cpu(self):
        self._check('cpu')

    def test_cuda(self):
        self._check('cuda')


if __name__ == '__main__':
    unittest.main()

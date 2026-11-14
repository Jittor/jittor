"""FP32 attention reads no-grad views without QKV layout copies."""
import contextlib
import unittest
from unittest import mock

import numpy as np
import torch
import jittor as jt

from _helpers import capability
from _helpers.sdpa_reference import reference
from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor.backends.cuda.kernels.nn import fused_attention_f32_cuda as kernel


@unittest.skipUnless(capability.check_accelerator('cuda', backend=jt).enabled,
                     'needs a real CUDA runtime')
class TestAttentionStorageViews(unittest.TestCase):
    def setUp(self):
        self.scope = contextlib.ExitStack()
        self.scope.enter_context(jt.flag_scope(use_cuda=1, amp_reg=0))
        self.scope.enter_context(forbid_backend_fallbacks())
        self.addCleanup(self.scope.close)
        mask_cache = kernel._LAST_MASK[:]
        self.addCleanup(lambda: kernel._LAST_MASK.__setitem__(slice(None), mask_cache))

    def test_packed_qkv_with_offset_masks_and_partial_tiles(self):
        rng = np.random.default_rng(142)
        for shape in [(2, 129, 3, 4, 80), (1, 37, 3, 3, 64)]:
            array = rng.normal(0, .2, size=shape).astype('float32')
            # A nonzero offset in the backing allocation is part of the view.
            padded = np.pad(array.ravel(), (17, 11))
            source = torch.tensor(padded, device='cuda')[17:-11].reshape(shape)
            with torch.no_grad():
                q, k, v = source.permute(2, 0, 3, 1, 4).unbind(0)
                qa, ka, va = [array[:, :, index].transpose(0, 2, 1, 3) for index in range(3)]
                for causal in (False, True):
                    for kind in ('none', 'bool', 'float'):
                        with self.subTest(shape=shape, causal=causal, mask=kind):
                            mask = None
                            if kind != 'none':
                                keep = rng.random((1, 1, shape[1], shape[1])) > .2
                                keep[..., 0, :] = False
                                mask = keep if kind == 'bool' else np.where(keep, 0., -np.inf).astype('float32')
                            expected = reference(qa, ka, va, mask, causal, np.zeros_like(qa))[0]
                            jm = None if mask is None else torch.tensor(mask, device='cuda')
                            layout = kernel._dense_storage_layout(q)
                            self.assertIsNotNone(layout)
                            with mock.patch.object(kernel, '_forward', side_effect=AssertionError('unexpected dense-copy route')):
                                output = kernel._fused_attention_f32(q, k, v, jm, is_causal=causal)
                            output.sync()
                            self.assertEqual(output.location(), 'device')
                            self.assertEqual(output.device.type, 'cuda')
                            np.testing.assert_allclose(output.cpu().numpy(), expected, rtol=2e-5, atol=2e-6)

    def test_detached_views_do_not_restore_root_gradients(self):
        source = torch.randn(2, 37, 3, 4, 64, device='cuda', requires_grad=True)
        q, k, v = [piece.detach() for piece in source.permute(2, 0, 3, 1, 4).unbind(0)]
        output = kernel._fused_attention_f32(q, k, v)
        output.sync()
        self.assertFalse(output.requires_grad)
        self.assertEqual(output.location(), 'device')

    def test_rectangular_views_from_different_backing_tensors(self):
        rng = np.random.default_rng(62)
        qa = rng.normal(0, .2, (2, 37, 3, 2, 16)).astype('float32')
        kva = rng.normal(0, .2, (2, 53, 3, 2, 16)).astype('float32')
        with torch.no_grad():
            q = torch.tensor(qa, device='cuda').permute(2, 0, 3, 1, 4).unbind(0)[0]
            _, k, v = torch.tensor(kva, device='cuda').permute(2, 0, 3, 1, 4).unbind(0)
            arrays = [a[:, :, index].transpose(0, 2, 1, 3)
                      for a, index in ((qa, 0), (kva, 1), (kva, 2))]
            expected = reference(*arrays, None, True, np.zeros_like(arrays[0]))[0]
            output = kernel._fused_attention_f32(q, k, v, is_causal=True)
            output.sync()
            self.assertEqual(output.location(), 'device')
            np.testing.assert_allclose(output.cpu().numpy(), expected, rtol=2e-5, atol=2e-6)

    def test_training_keeps_all_input_gradients_on_the_existing_route(self):
        rng = np.random.default_rng(354)
        array = rng.normal(0, .2, (1, 37, 3, 2, 16)).astype('float32')
        source = torch.tensor(array, device='cuda', requires_grad=True)
        q, k, v = source.permute(2, 0, 3, 1, 4).unbind(0)
        qa, ka, va = [array[:, :, index].transpose(0, 2, 1, 3) for index in range(3)]
        expected = reference(qa, ka, va, None, False, np.ones_like(qa))
        with mock.patch.object(kernel, '_forward_view', side_effect=AssertionError('view route must not discard gradients')):
            output = kernel._fused_attention_f32(q, k, v)
            output.sum().backward()
        gradient = source.grad
        gradient.sync()
        self.assertEqual(gradient.location(), 'device')
        want = np.stack([g.transpose(0, 2, 1, 3) for g in expected[1:]], axis=2)
        np.testing.assert_allclose(gradient.cpu().numpy(), want, rtol=1e-4, atol=1e-5)

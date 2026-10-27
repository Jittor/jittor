"""Like factories allocate independent leaves and honor explicit gradient options."""
import contextlib
import os
import unittest

import numpy as np
import torch


@contextlib.contextmanager
def _device_scope(device):
    if not hasattr(torch, '_torch_compat_install_context'):
        yield None
        return
    import jittor as jt
    with jt.runtime.scope(use_cuda=int(device == 'cuda'), backend_fallback='error'):
        before = jt.core.backend_fallback_count()
        yield jt
        assert jt.core.backend_fallback_count() == before


def _make(name, source, **options):
    if name == 'full_like':
        return torch.full_like(source, fill_value=.5, **options)
    return getattr(torch, name)(source, **options)


def _check(test, native, execution_device, name, kind, requested, **options):
    base = torch.tensor(np.ones((2, 4), dtype='float32'), device=execution_device,
                        dtype=torch.float32, requires_grad=True)
    test.assertTrue(base.is_leaf)
    source = base if kind == 'leaf' else base * 2
    test.assertEqual(source.is_leaf, kind == 'leaf')
    if requested != 'default':
        options['requires_grad'] = requested
    output = _make(name, source, **options)
    expected_grad = requested is True
    test.assertEqual(output.requires_grad, expected_grad)
    test.assertTrue(output.is_leaf)
    test.assertIsNone(output.grad_fn)
    test.assertEqual(tuple(output.shape), tuple(source.shape))
    test.assertTrue(output.dtype == (options.get('dtype') or source.dtype))
    test.assertEqual(str(output.device).split(':')[0],
                     str(options.get('device') or source.device).split(':')[0])
    test.assertNotEqual(output.data_ptr(), source.data_ptr())
    peer = _make(name, source, **options)
    test.assertNotEqual(output.data_ptr(), peer.data_ptr())
    test.assertNotEqual(source.data_ptr(), peer.data_ptr())
    with torch.no_grad():
        peer.fill_(-3)
    if native is not None:
        output.sync()
        expected = 'device' if str(output.device).startswith('cuda') else 'cpu'
        test.assertEqual(output.location(), expected)
    if name != 'empty_like':
        expected_value = {'zeros_like': 0, 'ones_like': 1, 'full_like': .5}[name]
        np.testing.assert_array_equal(output.detach().cpu().numpy(),
                                      np.full((2, 4), expected_value))
    # Empty memory has no numerical value contract. Initialize all four
    # outputs uniformly before checking gradients and storage independence.
    with torch.no_grad():
        output.fill_(.5)
    np.testing.assert_array_equal(peer.detach().cpu().numpy(), np.full((2, 4), -3))
    np.testing.assert_array_equal(source.detach().cpu().numpy(),
                                  np.full((2, 4), 1 if kind == 'leaf' else 2))
    if expected_grad:
        (output * 2).sum().backward()
        test.assertIsNotNone(output.grad)
        np.testing.assert_array_equal(output.grad.detach().cpu().numpy(),
                                      np.full((2, 4), 2))
        (output * 3).sum().backward()
        np.testing.assert_array_equal(output.grad.detach().cpu().numpy(),
                                      np.full((2, 4), 5))
    test.assertIsNone(base.grad)
    test.assertIsNone(peer.grad)


class TestLikeFactoryLeaves(unittest.TestCase):
    def test_four_factories_leaf_and_nonleaf_default_and_explicit_grad(self, device=None):
        device = device or os.environ.get('JITTOR_TEST_DEVICE', 'cpu')
        with _device_scope(device) as native:
            for name in ('empty_like', 'zeros_like', 'ones_like', 'full_like'):
                for kind in ('leaf', 'nonleaf'):
                    for requested in ('default', True, False):
                        with self.subTest(factory=name, input=kind, grad=requested):
                            _check(self, native, device, name, kind, requested)

    def test_dtype_and_supported_options_preserve_leaf_contract(self, device=None):
        device = device or os.environ.get('JITTOR_TEST_DEVICE', 'cpu')
        with _device_scope(device) as native:
            for name in ('empty_like', 'zeros_like', 'ones_like', 'full_like'):
                with self.subTest(factory=name, options='explicit'):
                    _check(self, native, device, name, 'nonleaf', True, dtype=torch.float64,
                           device=device, layout=torch.strided, pin_memory=False,
                           memory_format=torch.contiguous_format)
                with self.subTest(factory=name, options='dtype=None'):
                    # Explicit None follows the input even if the global
                    # default dtype differs from the source's float32.
                    saved = torch.get_default_dtype()
                    try:
                        torch.set_default_dtype(torch.float64)
                        _check(self, native, device, name, 'leaf', False, dtype=None)
                    finally:
                        torch.set_default_dtype(saved)

    def test_explicit_device_overrides_like_input(self, device=None):
        device = device or os.environ.get('JITTOR_TEST_DEVICE', 'cpu')
        placements = [('cpu', 'cpu')]
        if device == 'cuda':
            placements.extend((('cpu', 'cuda'), ('cuda', 'cpu')))
        with _device_scope(device) as native:
            for name in ('empty_like', 'zeros_like', 'ones_like', 'full_like'):
                for source_device, target_device in placements:
                    with self.subTest(factory=name, source=source_device, target=target_device):
                        source = torch.tensor(np.ones((2, 4), dtype='float32'),
                                              device=source_device, requires_grad=True)
                        self.assertTrue(source.is_leaf)
                        output = _make(name, source, device=target_device, requires_grad=True)
                        self.assertEqual(str(output.device).split(':')[0], target_device)
                        self.assertTrue(output.is_leaf)
                        self.assertTrue(output.requires_grad)
                        self.assertIsNone(output.grad_fn)
                        if native is not None:
                            output.sync()
                            self.assertEqual(output.location(),
                                             'device' if target_device == 'cuda' else 'cpu')
                        with torch.no_grad():
                            output.fill_(.5)
                        output.sum().backward()
                        self.assertIsNotNone(output.grad)
                        np.testing.assert_array_equal(output.grad.detach().cpu().numpy(),
                                                      np.ones((2, 4)))
                        self.assertIsNone(source.grad)


if hasattr(torch, '_torch_compat_install_context'):
    from _helpers.device_types import instantiate_device_type_tests
    instantiate_device_type_tests(TestLikeFactoryLeaves, globals(), only_for=('cpu', 'cuda'))

if __name__ == '__main__':
    unittest.main()

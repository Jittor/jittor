"""empty_like is a fresh leaf, never an autograd alias of its input.

Run TestEmptyLikeFactorySemantics.test_cpu before test_cuda, in independent
Torch/Jittor interpreters. JITTOR_EMPTY_LIKE_RUNTIME=oracle|candidate overrides
selection from JITTOR_TORCH_SHIM. Each device covers 48 combinations.

Uninitialized output receives metadata checks only. Full byte/numeric checks
occur after no_grad copy_ initializes every element. BF16 is compared through
uint8 storage and an independent FP64 observation copy; no dtype exemptions.
Returned leaf.grad.requires_grad is reported separately, not repaired/asserted
as part of this factory contract. This is not full model compatibility proof.
"""
from contextlib import ExitStack, contextmanager, nullcontext
import json
import os
import sys
import unittest

import numpy as np


DTYPES = ('float16', 'bfloat16', 'float32', 'float64')
REQUESTS = ('default', 'false', 'true')
X = np.array([1., 2., 4., 8.], dtype=np.float64)
COTANGENT = np.array([1/8, 3/16, 5/32, 7/64], dtype=np.float64)
ATOL = RTOL = 1e-5


class TestEmptyLikeFactorySemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runtime = os.environ.get('JITTOR_EMPTY_LIKE_RUNTIME',
                                 'candidate' if os.environ.get('JITTOR_TORCH_SHIM') == '1' else 'oracle')
        cls.jt = None
        if runtime == 'candidate':
            import jittor as jt
            import torch
            from jittor.compat.torch import namespace_owner
            from jittor.compat.shim.runtime import activation_status
            assert namespace_owner(torch) is jt and activation_status(jt).active
            cls.jt = jt
        elif runtime == 'oracle':
            import torch
            assert hasattr(torch, '_C') and not hasattr(torch, '_torch_compat_install_context')
            assert not any(name == 'jittor' or name.startswith('jittor.') for name in sys.modules)
        else:
            raise ValueError('JITTOR_EMPTY_LIKE_RUNTIME must be oracle or candidate')
        cls.torch = torch

    @contextmanager
    def _device(self, device):
        with ExitStack() as stack:
            if device == 'cuda':
                if not self.torch.cuda.is_available():
                    self.skipTest('CUDA unavailable; no CUDA support claim')
                self.assertEqual(self.torch.cuda.current_device(), 0)
            if self.jt is not None:
                from jittor._runtime.fallback import forbid_backend_fallbacks
                stack.enter_context(self.jt.runtime.scope(use_cuda=int(device == 'cuda'), backend_fallback='error'))
                stack.enter_context(forbid_backend_fallbacks())
                before = self.jt.core.backend_fallback_count()
                self.assertEqual(before, 0)
            yield 'cuda:0' if device == 'cuda' else 'cpu'
            if self.jt is not None:
                self.jt.sync_all(True)
                self.assertEqual(self.jt.core.backend_fallback_count(), before)

    def _placement(self, tensor, target):
        device = target.split(':')[0]
        self.assertEqual(tensor.device.type, device)
        if self.jt is None:
            if device == 'cuda':
                self.assertEqual(tensor.device.index, 0)
                self.torch.cuda.synchronize(tensor.device)
            return
        native = getattr(tensor, '_var', tensor)
        native.sync()
        index = -1 if device == 'cpu' else 0
        self.assertEqual(self.jt.core.dispatch_context([native]), (device, index))
        self.assertEqual(int(native.device_id), index)
        self.assertEqual(int(native.placement_backend), 0 if device == 'cpu' else 1)
        self.assertEqual(native.location(), 'cpu' if device == 'cpu' else 'device')

    def _metadata(self, tensor, target):
        # Allocation/synchronization is allowed; uninitialized bytes are not read.
        self._placement(tensor, target)
        return dict(shape=tuple(tensor.shape), dtype=tensor.dtype,
                    requires_grad=bool(tensor.requires_grad), is_leaf=bool(tensor.is_leaf))

    def _snapshot_initialized(self, tensor, target):
        before = self._metadata(tensor, target)
        with self.torch.no_grad():
            clone = tensor.clone()
            self.assertIsNot(clone, tensor)
            self.assertFalse(clone.requires_grad)
            self._placement(clone, target)
            host = clone.cpu()
            self._placement(host, 'cpu')
            byte_view = host.contiguous().view(self.torch.uint8)
            self._placement(byte_view, 'cpu')
            raw = np.array(byte_view.numpy(), dtype=np.uint8, copy=True, order='C')
            widened = host.to(dtype=self.torch.float64)
            self._placement(widened, 'cpu')
            numbers = np.array(widened.numpy(), dtype=np.float64, copy=True, order='C')
        self.assertEqual(self._metadata(tensor, target), before)
        self.assertEqual(numbers.shape, before['shape'])
        self.assertTrue(np.isfinite(numbers).all(), 'nonfinite initialized contents')
        self.assertEqual(raw.size, numbers.size * tensor.element_size())
        return before, raw, numbers

    def _assert_numeric(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.isfinite(actual).all(), 'nonfinite tensor')
        bad = ~np.isclose(actual, expected, atol=ATOL, rtol=RTOL)
        difference = np.abs(actual.astype(np.float64)-expected)
        self.assertFalse(bool(bad.any()),
                         'full-array comparison: failed_elements=%d max_abs=%g' %
                         (int(bad.sum()), float(difference.max(initial=0))))

    def _case(self, target, dtype_name, input_grad, request, no_grad):
        torch = self.torch
        dtype = getattr(torch, dtype_name)
        source = torch.tensor(X, dtype=dtype, device=target, requires_grad=input_grad)
        stopped = torch.tensor(X, dtype=dtype, device=target, requires_grad=False)
        source_before, source_bytes, _ = self._snapshot_initialized(source, target)
        stopped_before, stopped_bytes, _ = self._snapshot_initialized(stopped, target)
        kwargs = {} if request == 'default' else {'requires_grad': request == 'true'}
        with torch.no_grad() if no_grad else nullcontext():
            output = torch.empty_like(source, **kwargs)
        fresh = self._metadata(output, target)
        with torch.no_grad():
            returned = output.copy_(stopped)
        self.assertIs(returned, output)
        filled, output_bytes, numbers = self._snapshot_initialized(output, target)
        with self.subTest(contract='filled_values'):
            self.assertTrue(np.array_equal(output_bytes, stopped_bytes), 'complete initialized bytes differ')
            self._assert_numeric(numbers, X)
        grad_flag = None
        if request == 'true':
            with self.subTest(contract='full_vjp'):
                self.assertTrue(output.requires_grad, 'requested True factory must permit backward even under no_grad')
                cot = torch.tensor(COTANGENT, dtype=dtype, device=target, requires_grad=False)
                self._placement(cot, target)
                loss = (output * cot).sum()
                self._placement(loss, target)
                loss.backward()
                gradient = output.grad
                self.assertIsNotNone(gradient, 'fresh leaf full gradient missing')
                grad_meta, _, grad_numbers = self._snapshot_initialized(gradient, target)
                self.assertEqual(grad_meta['shape'], X.shape)
                self.assertEqual(grad_meta['dtype'], dtype)
                self._assert_numeric(grad_numbers, COTANGENT)
                grad_flag = grad_meta['requires_grad']
        source_after, after_bytes, _ = self._snapshot_initialized(source, target)
        stopped_after, after_stopped_bytes, _ = self._snapshot_initialized(stopped, target)
        with self.subTest(contract='source_unchanged_and_unconnected'):
            self.assertEqual(source_after, source_before)
            self.assertEqual(stopped_after, stopped_before)
            self.assertTrue(np.array_equal(after_bytes, source_bytes))
            self.assertTrue(np.array_equal(after_stopped_bytes, stopped_bytes))
            self.assertIsNone(source.grad)
            self.assertIsNone(stopped.grad)
            self.assertEqual(source_before['requires_grad'], input_grad)
            self.assertTrue(source_before['is_leaf'])
            self.assertFalse(stopped_before['requires_grad'])
            self.assertTrue(stopped_before['is_leaf'])
        # Independent subtests allow a wrong flag to remain visible without
        # preventing full initialized bytes/VJP and the other 47 combinations.
        expected = dict(shape=X.shape, dtype=dtype, requires_grad=request == 'true', is_leaf=True)
        for stage, actual in (('fresh', fresh), ('filled', filled)):
            for key in expected:
                with self.subTest(stage=stage, contract=key):
                    self.assertEqual(actual[key], expected[key])
        return grad_flag

    def _check(self, device):
        observations = []
        attempted = 0
        with self._device(device) as target:
            for dtype in DTYPES:
                for input_grad in (False, True):
                    for request in REQUESTS:
                        for no_grad in (False, True):
                            label = f'{dtype}/{input_grad}/{request}/{no_grad}'
                            attempted += 1
                            with self.subTest(case=label, device=device):
                                flag = self._case(target, dtype, input_grad, request, no_grad)
                                if flag is not None:
                                    observations.append(dict(case=label, requires_grad=flag))
        self.assertEqual(attempted, 48)
        print('EMPTY_LIKE_RETURNED_GRAD_FLAGS ' + json.dumps(dict(
            device=device, combinations_attempted=attempted,
            scope='separate_first_order_gradient_flag_not_asserted_by_factory_contract',
            successful_vjp_observations=len(observations), expected_trainable_cases=16,
            observed=observations, L2_pass=False), sort_keys=True))

    def test_cpu(self):
        self._check('cpu')

    def test_cuda(self):
        self._check('cuda')


if __name__ == '__main__':
    unittest.main()

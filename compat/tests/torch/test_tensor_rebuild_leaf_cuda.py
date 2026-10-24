"""Real CUDA serialization restores a leaf before any output synchronization.

Host ndarrays are the production pickle transport, never CPU computation.
A skip is not CUDA acceptance. No serializer or gradient setter is patched.
"""
import ctypes
import pickle
import unittest

import pytest
import jittor as jt
import torch

pytestmark = pytest.mark.cuda


def run(torch, jt, pointer_probe):
    from jittor.compat.torch import frontend
    from jittor.compat.diagnostics import counts
    from jittor._runtime.fallback import forbid_backend_fallbacks

    def require(value):
        if not value:
            raise AssertionError('CUDA tensor rebuild leaf contract failed')

    before = counts()
    original_grad_mode = torch.is_grad_enabled()
    original_default_device = frontend.default_device()
    original_cuda_device = torch.cuda.current_device()
    report = dict(scope='actual_CUDA_tensor_rebuild_leaf_NOT_full_VERL', cases=[],
                  source_must_be_authenticated_by_caller=True,
                  production_serializers_unmodified=True,
                  full_PPO_GRPO_or_L_claims=False)

    def prove(value):
        value.sync()
        pointer_probe(jt, value)
        require((int(value.placement_backend), int(value.device_id)) == (1, 0)
                and str(value.device) == 'cuda:0')

    with forbid_backend_fallbacks(), jt.runtime.scope(
            use_cuda=1, device_id=0, backend_fallback='error'), torch.enable_grad():
        for dtype in (torch.float32, torch.bfloat16):
            for flag in (False, True):
                source = torch.ones((3, 4), device='cuda:0', dtype=dtype,
                                    requires_grad=flag)
                prove(source)
                source_pointer = int(source.device_raw_ptr)
                for operation in ('rebuild_tensor', 'pickle'):
                    mode_before = torch.is_grad_enabled()
                    default_before = frontend.default_device()
                    cuda_before = torch.cuda.current_device()
                    if operation == 'rebuild_tensor':
                        builder, arguments, state = frontend.reduce_tensor(source)
                        require(builder is frontend.rebuild_tensor and type(state) is dict)
                        value = frontend.rebuild_tensor(*arguments)
                    else:
                        value = pickle.loads(pickle.dumps(source, protocol=5))
                    # Must precede sync, data pointer access, equality, arithmetic,
                    # detach or any helper that might materialize the returned graph.
                    row = dict(dtype=str(dtype), requires_grad=flag, operation=operation,
                               immediate_is_leaf=bool(value.is_leaf),
                               immediate_grad_fn_none=value.grad_fn is None,
                               immediate_flag_restored=bool(value.requires_grad) is flag,
                               immediate_dtype_and_type=value.dtype == dtype
                                   and type(value) is type(source),
                               immediate_public_device=str(value.device),
                               grad_mode_restored=torch.is_grad_enabled() == mode_before,
                               default_device_restored=frontend.default_device() == default_before,
                               cuda_device_restored=torch.cuda.current_device() == cuda_before)
                    report['cases'].append(row)
                    require(row['immediate_is_leaf'] and row['immediate_grad_fn_none']
                            and row['immediate_flag_restored']
                            and row['immediate_dtype_and_type']
                            and row['immediate_public_device'] == 'cuda:0'
                            and row['grad_mode_restored'] and row['default_device_restored']
                            and row['cuda_device_restored'])
                    prove(value)
                    require(bool(torch.equal(value, source)))
                    row['independent_storage'] = int(value.device_raw_ptr) != source_pointer
                    require(row['independent_storage'])
                    if flag:
                        (value * 2).float().sum().backward()
                        gradient = value.grad
                        require(gradient is not None and gradient.dtype == dtype)
                        prove(gradient)
                        row['full_gradient_exact_CUDA'] = bool(
                            torch.equal(gradient, torch.full_like(value, 2)))
                        require(row['full_gradient_exact_CUDA'])
                    else:
                        row['no_gradient'] = value.grad is None
                        require(row['no_gradient'])
                    with torch.no_grad():
                        value.add_(1)
                    prove(value)
                    require(bool(torch.equal(value, torch.full_like(value, 2))))
                    row['source_unchanged'] = (source.grad is None
                        and bool(source.requires_grad) is flag
                        and bool(torch.equal(source, torch.ones_like(source))))
                    require(row['source_unchanged'])
                    row['passed'] = True

    report['checks'] = dict(
        all_eight_serialization_cases_passed=len(report['cases']) == 8
            and all(row['passed'] for row in report['cases']),
        grad_mode_restored=torch.is_grad_enabled() == original_grad_mode,
        default_device_restored=frontend.default_device() == original_default_device,
        cuda_device_restored=torch.cuda.current_device() == original_cuda_device,
        diagnostic_counts_unchanged=counts() == before,
        fallback_zero=int(jt.core.backend_fallback_count()) == 0)
    report['new_swallowed_count'] = sum(counts().values()) - sum(before.values())
    report['fallback_count'] = int(jt.core.backend_fallback_count())
    report['passed'] = all(report['checks'].values())
    require(report['passed'])
    return report


class TestTensorRebuildLeafCUDA(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('requires real CUDA; skip is not acceptance')
        torch.cuda.set_device(0)
        self.driver = ctypes.CDLL('libcuda.so.1')
        self.driver.cuInit.argtypes = [ctypes.c_uint]
        self.driver.cuInit.restype = ctypes.c_int
        self.driver.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.driver.cuPointerGetAttribute.restype = ctypes.c_int
        self.assertEqual(self.driver.cuInit(0), 0)

    def prove_cuda(self, backend, value):
        self.assertIs(backend, jt)
        self.assertEqual(tuple(jt.core.dispatch_context([value])), ('cuda', 0))
        self.assertEqual(value.location(), 'device')
        pointer = int(value.device_raw_ptr)
        self.assertGreater(pointer, 0)
        ordinal, memory_type = ctypes.c_int(-1), ctypes.c_int(-1)
        self.assertEqual(self.driver.cuPointerGetAttribute(
            ctypes.byref(ordinal), 9, ctypes.c_void_p(pointer)), 0)
        self.assertEqual(self.driver.cuPointerGetAttribute(
            ctypes.byref(memory_type), 2, ctypes.c_void_p(pointer)), 0)
        self.assertEqual((ordinal.value, memory_type.value), (0, 2))

    def test_rebuild_and_pickle_restore_independent_cuda_leaf(self):
        self.runtime_evidence = run(torch, jt, self.prove_cuda)
        self.assertTrue(self.runtime_evidence['passed'])

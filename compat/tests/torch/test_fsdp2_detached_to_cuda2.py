"""Actual CUDA2 detached-checkpoint Tensor.to storage regression, not full PPO."""
import ctypes
import gc
import os
import unittest
import weakref

import pytest
import jittor as jt
import torch
from jittor.compat.diagnostics import counts
from jittor._runtime.fallback import forbid_backend_fallbacks

pytestmark = pytest.mark.cuda


class TestFSDP2DetachedToCUDA2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if int(jt.world_size) != 2 or not torch.cuda.is_available():
            raise unittest.SkipTest('requires real CUDA NCCL2; skip is not device acceptance')
        cls.dist = torch.distributed
        cls.created_group = not cls.dist.is_initialized()
        if cls.created_group:
            cls.dist.init_process_group('nccl', rank=int(jt.rank), world_size=2)
        assert cls.dist.get_world_size() == int(jt.world_size) == 2
        assert cls.dist.get_rank() == int(jt.rank) and cls.dist.get_backend() == 'nccl'

    @classmethod
    def tearDownClass(cls):
        if cls.created_group:
            cls.dist.destroy_process_group()

    def setUp(self):
        torch.cuda.set_device(0)
        self.initial_counts = counts()
        self.runtime_evidence = dict(checks={}, initial_lazy_metadata={}, cases=[],
            startup_swallowed_total=sum(self.initial_counts.values()),
            global_environment_passed=sum(self.initial_counts.values()) == 0,
            full_PPO_or_L_claims=False,
            native_boundmethod_strong_holder_retention_supported=False,
            dead_owner_redistribute_identity_supported=False)
        self.driver = ctypes.CDLL('libcuda.so.1')
        self.driver.cuInit.argtypes = [ctypes.c_uint]
        self.driver.cuInit.restype = ctypes.c_int
        self.driver.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.driver.cuPointerGetAttribute.restype = ctypes.c_int
        self.assertEqual(self.driver.cuInit(0), 0)
        self.addCleanup(self.check_diagnostics)

    def check_diagnostics(self):
        current = counts()
        self.runtime_evidence.update(new_swallowed_count=sum(current.values()) - sum(self.initial_counts.values()),
            diagnostic_counts_unchanged=current == self.initial_counts,
            fallback_count=int(jt.core.backend_fallback_count()))
        self.assertEqual(current, self.initial_counts)
        self.assertEqual(jt.core.backend_fallback_count(), 0)

    def check(self, name, value):
        self.runtime_evidence['checks'][name] = bool(value)
        self.assertTrue(value, name)

    def descriptor(self, value):
        # Pure metadata reads: never heal the object being compared here.
        native = getattr(value, '_var', value)
        return dict(shape=tuple(int(d) for d in value.shape), dtype=str(value.dtype),
            public_device=(value.device.type, value.device.index), native_location=native.location(),
            backend=int(native.placement_backend), native_device=int(native.device_id),
            requires_grad=bool(value.requires_grad), version=int(value._version))

    def cuda_pointer(self, value):
        value.sync()
        self.assertEqual((value.device.type, value.device.index), ('cuda', 0))
        self.assertEqual((value.location(), int(value.placement_backend), int(value.device_id)), ('device', 1, 0))
        if not int(value.numel()):
            return
        pointer = int(value.device_raw_ptr)
        self.assertGreater(pointer, 0)
        ordinal, memory_type = ctypes.c_int(-1), ctypes.c_int(-1)
        self.assertEqual(self.driver.cuPointerGetAttribute(ctypes.byref(ordinal), 9, ctypes.c_void_p(pointer)), 0)
        self.assertEqual(self.driver.cuPointerGetAttribute(ctypes.byref(memory_type), 2, ctypes.c_void_p(pointer)), 0)
        self.assertEqual((ordinal.value, memory_type.value), (0, 2))

    def exact_cuda(self, actual, expected):
        self.cuda_pointer(actual)
        self.cuda_pointer(expected)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertTrue(bool(torch.equal(actual, expected)))

    def test_nonflat_detached_export_conversion_and_alias(self):
        self.run_case(False)

    def test_flat_detached_export_conversion_and_alias(self):
        self.run_case(True)

    def run_case(self, flat):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard
        from torch.distributed.tensor import DTensor
        previous = os.environ.get('JITTOR_FSDP2_FLAT')
        os.environ['JITTOR_FSDP2_FLAT'] = '1' if flat else '0'

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.arange(12, device='cuda:0', dtype=torch.float32).reshape(3, 4) / 16)
                self.bias = torch.nn.Parameter(torch.arange(3, device='cuda:0', dtype=torch.float32) / 8)
                self.frozen = torch.nn.Parameter(torch.arange(4, device='cuda:0', dtype=torch.float32) / 32,
                                                  requires_grad=False)

        try:
            with forbid_backend_fallbacks(), jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error'):
                for order in ('offload_then_capture', 'capture_then_offload'):
                    model = Model()
                    expected = {name: value.detach().clone() for name, value in model.named_parameters()}
                    for value in expected.values():
                        self.cuda_pointer(value)
                    fully_shard(model, mesh=init_device_mesh('cuda', (2,)), reshard_after_forward=True)
                    self.assertEqual(model._fsdp_state.true_fsdp_flat, flat)
                    with torch.no_grad():
                        checkpoint = model.state_dict() if order == 'capture_then_offload' else None
                        self.assertIs(model.to(device='cpu'), model)
                        if checkpoint is None:
                            checkpoint = model.state_dict()
                        self.assertEqual(set(checkpoint), {'weight', 'bias', 'frozen'})
                        canonical = dict(model.named_parameters())
                        # Retain original lazy observations, then explicitly stabilize
                        # the SOURCE baseline. The old diagnostic failures are separate
                        # evidence; source-baseline sync is not an export-placement proof.
                        lazy = {name: self.descriptor(value) for name, value in checkpoint.items()}
                        self.runtime_evidence['initial_lazy_metadata'][order] = lazy
                        for value in canonical.values():
                            value.sync()
                        for value in checkpoint.values():
                            snapshot = getattr(value, '_jittor_fsdp2_detached_snapshot')
                            value.sync()
                            snapshot.local.sync()
                            snapshot.gather_input.sync()
                        stable = {name: self.descriptor(value) for name, value in canonical.items()}
                        source_before = {name: self.descriptor(value) for name, value in checkpoint.items()}
                        for name, parameter in checkpoint.items():
                            label = order + ':' + name
                            self.assertIsInstance(parameter, DTensor)
                            converted = parameter.to(torch.cuda.current_device(), non_blocking=True)
                            self.assertIsInstance(converted, DTensor)
                            self.assertIsNone(getattr(converted, '_jittor_fsdp2_state', None))
                            self.assertIsNone(getattr(converted, '_jittor_fsdp2_entry', None))
                            self.assertFalse(bool(getattr(converted, '_is_param', False)))
                            self.assertFalse(converted.requires_grad)
                            self.assertIs(converted.full_tensor.owner(), converted)
                            self.assertIs(converted.to('cuda:0'), converted)
                            self.exact_cuda(converted.full_tensor(), expected[name])
                            # Exact downstream temporary chain: do not bind the
                            # converted owner before the public method is read.
                            self.exact_cuda(parameter.to(torch.cuda.current_device(),
                                non_blocking=True).full_tensor(), expected[name])
                            self.exact_cuda(parameter.to(torch.cuda.current_device(),
                                non_blocking=True, copy=True).to_local(), converted.to_local())
                            held = parameter.to(torch.cuda.current_device(), non_blocking=True, copy=True)
                            saved_full, saved_local = held.full_tensor, held.to_local
                            saved_redistribute = held.redistribute
                            self.assertIs(saved_redistribute(), held)
                            held_reference = weakref.ref(held)
                            del held
                            gc.collect()
                            self.assertIsNone(held_reference())
                            self.exact_cuda(saved_full(), expected[name])
                            self.exact_cuda(saved_local(), converted.to_local())
                            # This narrow legacy-protocol fix does not pretend
                            # to provide native boundmethod owner retention.
                            with self.assertRaises(ReferenceError):
                                saved_redistribute()
                            self.runtime_evidence['checks'][label + ':temporary_and_saved_snapshot_calls'] = True
                            del saved_full, saved_local, saved_redistribute, held_reference
                            # A same-device no-op intentionally returns the source.
                            # Test destructive alias writes on a real copied export
                            # in that control case, not on the source checkpoint.
                            working = converted if converted is not parameter else converted.to('cuda:0', copy=True)
                            snapshot = working._jittor_fsdp2_detached_snapshot
                            local = working.to_local()
                            self.assertFalse(isinstance(local, DTensor))
                            self.assertIs(local, snapshot.local)
                            self.cuda_pointer(local)
                            self.cuda_pointer(snapshot.gather_input)
                            if flat and int(local.numel()):
                                shard_size = int(snapshot.gather_input.numel())
                                first = int(self.dist.get_rank()) * shard_size
                                start = max(first, int(snapshot.offset)) - first
                                self.assertEqual(local.data_ptr(), snapshot.gather_input.data_ptr() + start * local.element_size())
                                self.assertEqual(local.storage_offset(), snapshot.gather_input.storage_offset() + start)
                            if int(local.numel()):
                                initial_version = int(working._version)
                                local.add_(.125)
                                self.assertEqual(int(working._version), initial_version + 1)
                            else:
                                local.add_(.125)
                            # All ranks modify their true local intersection, including
                            # rank-zero empty bias/frozen shards in the flat case.
                            self.exact_cuda(working.full_tensor(), expected[name] + .125)
                            local.sub_(.125)
                            self.exact_cuda(working.full_tensor(), expected[name])
                            copy = working.to('cuda:0', copy=True)
                            self.assertIsNot(copy, working)
                            self.exact_cuda(copy.full_tensor(), expected[name])
                            if int(local.numel()):
                                self.assertNotEqual(copy.to_local().data_ptr(), local.data_ptr())
                            double = working.to(dtype=torch.float64)
                            self.assertEqual(double.to_local().dtype, torch.float64)
                            self.assertEqual(double._jittor_fsdp2_detached_snapshot.gather_input.dtype, torch.float64)
                            self.exact_cuda(double.full_tensor(), expected[name].to(dtype=torch.float64))
                            self.check(label + ':source_checkpoint_metadata_unchanged', self.descriptor(parameter) == source_before[name])
                            self.check(label + ':canonical_metadata_unchanged', all(
                                self.descriptor(value) == stable[key] for key, value in model.named_parameters()))
                            self.check(label + ':canonical_holders_unchanged', all(
                                value is canonical[key] for key, value in model.named_parameters()))
                            self.runtime_evidence['cases'].append(dict(order=order, name=name,
                                converted_metadata=self.descriptor(converted), flat=flat,
                                exact_CUDA_values_and_alias_write_restore_passed=True))
                    self.check(order + ':three_parameter_export_cases_returned',
                        len([item for item in self.runtime_evidence['cases'] if item['order'] == order]) == 3)
        finally:
            if previous is None:
                os.environ.pop('JITTOR_FSDP2_FLAT', None)
            else:
                os.environ['JITTOR_FSDP2_FLAT'] = previous


if __name__ == '__main__':
    unittest.main()

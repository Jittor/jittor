"""Real CUDA2 cache lifetime regression, not PPO or peak-memory acceptance.

Run ranks in isolated caches under the maintained NCCL launcher. CPU tensor
arithmetic is not used. The callable also accepts Main's authenticated pointer
probe and an already initialized NCCL2 process group.
"""
from contextlib import ExitStack
import ctypes
import gc
import os
import unittest
import weakref

import pytest

pytestmark = pytest.mark.cuda


def require(condition, label):
    if not condition:
        raise AssertionError(label)


def run(torch, jt, pointer_probe, *, flat):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from jittor.compat.diagnostics import counts
    from jittor._runtime.fallback import forbid_backend_fallbacks

    require(type(flat) is bool and callable(pointer_probe), 'explicit_flat_and_pointer_authority')
    require(torch.cuda.is_available() and torch.distributed.is_initialized(), 'real_initialized_CUDA2_required')
    require(torch.distributed.get_world_size() == 2 and int(jt.world_size) == 2
            and torch.distributed.get_rank() == int(jt.rank)
            and torch.distributed.get_backend() == 'nccl', 'actual_NCCL_world2')
    before_counts = counts()
    old_flat = os.environ.get('JITTOR_FSDP2_FLAT')
    os.environ['JITTOR_FSDP2_FLAT'] = '1' if flat else '0'

    def cuda(value):
        value.sync()
        require(value.device.type == 'cuda' and value.device.index == 0
                and value.location() == 'device' and int(value.placement_backend) == 1
                and int(value.device_id) == 0
                and tuple(jt.core.dispatch_context([value])) == ('cuda', 0), 'actual_CUDA0_storage')
        if int(value.numel()): pointer_probe(jt, value)

    def close(value, expected):
        cuda(value); cuda(expected)
        require(tuple(value.shape) == tuple(expected.shape)
                and bool(torch.allclose(value, expected, atol=1e-5, rtol=1e-5)), 'complete_CUDA_values')

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.left = torch.nn.Parameter(torch.full((4, 4), .25, dtype=torch.float32, device='cuda:0'))
            self.right = torch.nn.Parameter(torch.full((4, 4), .5, dtype=torch.float32, device='cuda:0'))

        def forward(self, x):
            return x @ self.left + x @ self.right

    try:
        with ExitStack() as stack:
            stack.enter_context(forbid_backend_fallbacks())
            stack.enter_context(jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error'))
            torch.cuda.set_device(0)
            model = Model()
            fully_shard(model, mesh=init_device_mesh('cuda', (2,)), reshard_after_forward=True)
            state = model._fsdp_state
            entries = tuple(state.true_fsdp_params)
            require(state.true_fsdp_initialized and bool(state.true_fsdp_flat) is flat
                    and len(entries) == 2 and {entry.name for entry in entries} == {'left', 'right'},
                    'actual_flat_or_nonflat_state_and_entries')
            optimizer = torch.optim.SGD(list(model.parameters()), lr=.01)
            x = torch.ones((2, 4), dtype=torch.float32, device='cuda:0'); cuda(x)

            def backward():
                loss = model(x).sum(); cuda(loss); loss.backward()
                torch.cuda.synchronize(0)

            def gradients(number):
                for entry in entries:
                    require(entry.shard.grad is not None, 'real_public_gradient')
                    value = entry.shard.grad.full_tensor()
                    close(value, torch.full(tuple(entry.shape), number, dtype=torch.float32, device='cuda:0'))

            def references_and_snapshot():
                cached = tuple(state.true_fsdp_last_grads)
                require(len(cached) == 2 and all(value is not None for value in cached), 'complete_actual_gradient_cache')
                references = [weakref.ref(value) for value in cached]
                full = getattr(state, 'true_fsdp_last_flat_grad', None)
                if flat:
                    require(full is not None, 'actual_flat_gradient_owner'); references.append(weakref.ref(full))
                snapshot = next(value for value in cached if int(value.numel())).detach()
                cuda(snapshot)
                return references, snapshot

            for step in range(1, 4):
                optimizer.zero_grad(set_to_none=True)
                backward(); gradients(2.)
                references, snapshot = references_and_snapshot()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(0); gc.collect()
                require(all(reference() is None for reference in references), 'cleared_cache_holders_released')
                require(all(value is None for value in state.true_fsdp_last_grads)
                        and getattr(state, 'true_fsdp_last_flat_grad', None) is None, 'cache_metadata_released')
                close(snapshot, torch.full_like(snapshot, 2.))
                for entry in entries:
                    initial = .25 if entry.name == 'left' else .5
                    close(entry.shard.full_tensor(), torch.full(tuple(entry.shape), initial - .02 * step,
                        dtype=torch.float32, device='cuda:0'))
                del snapshot, references

            # Public zero tensors remain meaningful: the cleanup seam is not
            # called for set_to_none=False, even when numerical values are zero.
            def zero_tensor_case():
                backward()
                cached = state.true_fsdp_last_grads
                full = getattr(state, 'true_fsdp_last_flat_grad', None)
                optimizer.zero_grad(set_to_none=False)
                require(state.true_fsdp_last_grads is cached
                        and getattr(state, 'true_fsdp_last_flat_grad', None) is full, 'zero_tensor_cache_preserved')
                gradients(0.)
            zero_tensor_case(); optimizer.zero_grad(set_to_none=True)

            # Optimizers split one state. Clearing one must not erase the
            # other parameter's gradient/contribution or its common flat owner.
            left = torch.optim.SGD([entries[0].shard], lr=.01)
            right = torch.optim.SGD([entries[1].shard], lr=.01)
            def partial_case():
                backward()
                remaining = state.true_fsdp_last_grads[1]
                full = getattr(state, 'true_fsdp_last_flat_grad', None)
                left.zero_grad(set_to_none=True)
                require(state.true_fsdp_last_grads[0] is None
                        and state.true_fsdp_last_grads[1] is remaining
                        and entries[1].shard.grad is not None
                        and getattr(state, 'true_fsdp_last_flat_grad', None) is full,
                        'other_optimizer_contribution_preserved')
                right.zero_grad(set_to_none=True)
                require(all(value is None for value in state.true_fsdp_last_grads)
                        and getattr(state, 'true_fsdp_last_flat_grad', None) is None,
                        'last_partial_clear_releases_shared_cache')
            partial_case()

            # Accumulation is original FSDP math: two unsynchronized backwards
            # plus one synchronized backward must still yield all sixes.
            optimizer.zero_grad(set_to_none=True)
            model.set_requires_gradient_sync(False)
            backward(); backward()
            require(any(value is not None for value in state.true_fsdp_pending_full_grads),
                    'actual_no_sync_pending_contributions')
            model.set_requires_gradient_sync(True)
            backward(); gradients(6.)
            optimizer.zero_grad(set_to_none=True)

            # Explicitly zeroing the first optimizer discards only its own
            # pending contribution; the second optimizer's no_sync value lives.
            model.set_requires_gradient_sync(False)
            backward()
            def pending_partial_case():
                remaining = state.true_fsdp_pending_full_grads[1]
                require(remaining is not None, 'actual_other_pending_value')
                left.zero_grad(set_to_none=True)
                pending = state.true_fsdp_pending_full_grads
                require(pending[0] is None and pending[1] is remaining, 'other_no_sync_contribution_preserved')
            pending_partial_case()
            optimizer.zero_grad(set_to_none=True)
            require(state.true_fsdp_pending_full_grads is None, 'own_pending_clear_finishes')
            model.set_requires_gradient_sync(True)
            torch.cuda.synchronize(0)
            require(counts() == before_counts and jt.core.backend_fallback_count() == 0,
                    'no_new_swallowed_diagnostics_or_fallback')
            return dict(scope='real_CUDA2_cache_lifetime_regression_NOT_PPO_or_memory_benchmark',
                flat=flat, actual_optimizer_updates=3, complete_parameter_values_checked=True,
                external_detached_snapshot_preserved=True, old_cache_holders_released=True,
                partial_optimizer_and_no_sync_preserved=True, fallback_count=0,
                actual_device='cuda:0', actual_world_size=2, performance_proven=False, PPO_pass=False)
    finally:
        if old_flat is None: os.environ.pop('JITTOR_FSDP2_FLAT', None)
        else: os.environ['JITTOR_FSDP2_FLAT'] = old_flat


class TestFSDP2ClearedGradCacheCUDA2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import jittor as jt
        require(torch.cuda.is_available() and int(jt.world_size) == 2, 'real_CUDA2_launcher_required_no_skip')
        cls.torch, cls.jt = torch, jt
        cls.created = not torch.distributed.is_initialized()
        if cls.created: torch.distributed.init_process_group('nccl', rank=int(jt.rank), world_size=2)
        cls.driver = ctypes.CDLL('libcuda.so.1')
        cls.driver.cuInit.argtypes = [ctypes.c_uint]; cls.driver.cuInit.restype = ctypes.c_int
        cls.driver.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        cls.driver.cuPointerGetAttribute.restype = ctypes.c_int
        require(cls.driver.cuInit(0) == 0, 'actual_CUDA_driver')

    @classmethod
    def tearDownClass(cls):
        if cls.created: cls.torch.distributed.destroy_process_group()

    @staticmethod
    def pointer(jt, value):
        ordinal, kind = ctypes.c_int(-1), ctypes.c_int(-1)
        address = int(value.device_raw_ptr)
        require(address > 0, 'nonempty_actual_pointer')
        driver = TestFSDP2ClearedGradCacheCUDA2.driver
        require(driver.cuPointerGetAttribute(ctypes.byref(ordinal), 9, ctypes.c_void_p(address)) == 0
                and driver.cuPointerGetAttribute(ctypes.byref(kind), 2, ctypes.c_void_p(address)) == 0
                and (ordinal.value, kind.value) == (0, 2), 'actual_CUDA_device_pointer_ordinal0')

    def test_flat_three_updates_and_cache_lifetime(self):
        self.runtime_evidence = run(self.torch, self.jt, self.pointer, flat=True)

    def test_nonflat_three_updates_and_cache_lifetime(self):
        self.runtime_evidence = run(self.torch, self.jt, self.pointer, flat=False)


if __name__ == '__main__': unittest.main()

"""Actual CUDA2 FSDP usage flags must follow state, not a CPU factory default.

CPU is a storage/factory token only. All collective arithmetic executes on CUDA.
Run with separate masked rank caches under the maintained public NCCL2 launcher.
A skip is an environment fact, never accelerator or full-PPO acceptance.
"""
from contextlib import ExitStack
import ctypes
import os
import sys
import unittest
import uuid

import pytest
import jittor as jt
import torch
from jittor.compat.diagnostics import counts
from jittor._runtime.fallback import forbid_backend_fallbacks

pytestmark = pytest.mark.cuda


class TestFSDP2UsageFlagsPlacementCUDA2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if int(jt.world_size) != 2 or not torch.cuda.is_available():
            raise unittest.SkipTest('requires actual CUDA NCCL2; skip is not device acceptance')
        cls.dist = torch.distributed
        cls.created_group = not cls.dist.is_initialized()
        if cls.created_group:
            cls.dist.init_process_group('nccl', rank=int(jt.rank), world_size=2)
        assert cls.dist.get_backend() == 'nccl'
        assert cls.dist.get_world_size() == int(jt.world_size) == 2
        assert cls.dist.get_rank() == int(jt.rank)

    @classmethod
    def tearDownClass(cls):
        if cls.created_group:
            cls.dist.destroy_process_group()

    def setUp(self):
        self.previous_profile = sys.getprofile()
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(forbid_backend_fallbacks())
        stack.enter_context(jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error'))
        torch.cuda.set_device(0)
        self.assertEqual(int(jt.flags.auto_flush_ops), 0)
        self.assertEqual(torch.cuda.device_count(), 1, 'one actual masked physical GPU per rank required')
        self.driver = ctypes.CDLL('libcuda.so.1')
        self.driver.cuInit.argtypes = [ctypes.c_uint]
        self.driver.cuInit.restype = ctypes.c_int
        self.driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.driver.cuDeviceGet.restype = ctypes.c_int
        self.driver.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.driver.cuPointerGetAttribute.restype = ctypes.c_int
        self.assertEqual(self.driver.cuInit(0), 0)
        physical = ctypes.c_int(-1)
        self.assertEqual(self.driver.cuDeviceGet(ctypes.byref(physical), 0), 0)
        raw_uuid = (ctypes.c_ubyte * 16)()
        get_uuid = getattr(self.driver, 'cuDeviceGetUuid_v2', None)
        self.assertIsNotNone(get_uuid)
        self.assertEqual(get_uuid(ctypes.byref(raw_uuid), physical), 0)
        self.initial_counts = counts()
        self.runtime_evidence = dict(scope='actual_CUDA2_usage_flags_test_body_NOT_full_PPO',
            physical_uuid='GPU-' + str(uuid.UUID(bytes=bytes(raw_uuid))),
            rank=int(jt.rank), masked_logical_device=0, cases=[], checks={},
            startup_swallowed_total=sum(self.initial_counts.values()),
            startup_all_swallowed_zero=sum(self.initial_counts.values()) == 0,
            startup_nonzero_counts_preserved=True, CPU_storage_and_factory_context_only=True,
            full_PPO_or_L_claims=False)
        self.addCleanup(self.check_diagnostics)

    def check_diagnostics(self):
        self.assertEqual(sys.getprofile(), self.previous_profile)
        self.runtime_evidence.update(new_swallowed_count=sum(counts().values()) - sum(self.initial_counts.values()),
            diagnostic_counts_unchanged=counts() == self.initial_counts,
            fallback_count=int(jt.core.backend_fallback_count()))
        self.assertEqual(counts(), self.initial_counts)
        self.assertEqual(jt.core.backend_fallback_count(), 0)

    def prove_cuda(self, value):
        self.assertTrue(jt.core._is_tensor_carrier(value))
        value.sync()
        self.assertEqual((int(value.placement_backend), int(value.device_id)), (1, 0))
        self.assertEqual(tuple(jt.core.dispatch_context([value])), ('cuda', 0))
        public = value.device
        if type(public) is str:
            self.assertEqual(public, 'cuda:0')
            device_evidence = dict(representation='native_device_string', value=public)
        else:
            self.assertEqual((public.type, public.index), ('cuda', 0))
            device_evidence = dict(representation='public_device_object', type=public.type, index=public.index)
        empty = int(value.numel()) == 0
        # A zero-length operand has CUDA metadata but makes no allocation claim.
        evidence = dict(shape=tuple(int(d) for d in value.shape), dtype=str(value.dtype),
            native_backend=int(value.placement_backend), native_device=int(value.device_id),
            native_location=value.location(), public_device=device_evidence,
            empty=empty, actual_nonempty_device_pointer_verified=False)
        if not empty:
            self.assertEqual(value.location(), 'device')
            pointer = int(value.device_raw_ptr)
            self.assertGreater(pointer, 0)
            ordinal, memory_type = ctypes.c_int(-1), ctypes.c_int(-1)
            self.assertEqual(self.driver.cuPointerGetAttribute(ctypes.byref(ordinal), 9, ctypes.c_void_p(pointer)), 0)
            self.assertEqual(self.driver.cuPointerGetAttribute(ctypes.byref(memory_type), 2, ctypes.c_void_p(pointer)), 0)
            self.assertEqual((ordinal.value, memory_type.value), (0, 2))
            evidence['actual_nonempty_device_pointer_verified'] = True
        return evidence

    def prove_collective_host_readback(self, value):
        # _globally_used_grads has already called reduced.numpy(). That host
        # readback changes physical residency; it does not change the collective's
        # logical CUDA placement. Never migrate it back to manufacture a pointer.
        self.assertTrue(jt.core._is_tensor_carrier(value))
        self.assertEqual((int(value.placement_backend), int(value.device_id)), (1, 0))
        self.assertEqual(tuple(jt.core.dispatch_context([value])), ('cuda', 0))
        empty = int(value.numel()) == 0
        location = value.location()
        public = value.device
        if type(public) is str:
            public_evidence = dict(representation='native_device_string', value=public)
            if not empty:
                self.assertEqual(public, 'cpu')
        else:
            public_evidence = dict(representation='public_device_object', type=public.type, index=public.index)
        if not empty:
            self.assertEqual(location, 'cpu')
        return dict(shape=tuple(int(d) for d in value.shape), dtype=str(value.dtype),
            collective_native_backend=int(value.placement_backend),
            collective_native_device=int(value.device_id), collective_dispatch=('cuda', 0),
            native_location_after_original_numpy=location,
            public_device_after_original_numpy=public_evidence,
            original_helper_host_readback_returned=True,
            empty=empty, host_allocation_claimed=not empty,
            device_pointer_claimed=False, no_sync_or_migration_in_readback_observer=True)

    def test_nonflat_usage_flags_override_CPU_factory(self):
        self.run_case(False)

    def test_flat_usage_flags_override_CPU_factory(self):
        self.run_case(True)

    def test_nonflat_inactive_CPU_optimizer_during_CUDA_backward(self):
        self.run_inactive_optimizer_case(False)

    def test_flat_inactive_CPU_optimizer_during_CUDA_backward(self):
        self.run_inactive_optimizer_case(True)

    def run_inactive_optimizer_case(self, flat):
        from jittor._core.var import placement_scope_like
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard
        previous_flat = os.environ.get('JITTOR_FSDP2_FLAT')
        os.environ['JITTOR_FSDP2_FLAT'] = '1' if flat else '0'

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(
                    [[.5, .25], [-.25, .5]], device='cuda:0', dtype=torch.float32))
                self.bias = torch.nn.Parameter(torch.tensor(
                    [.125, -.125], device='cuda:0', dtype=torch.float32))

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias)

        try:
            mesh = init_device_mesh('cuda', (2,))
            actor = Model()
            fully_shard(actor, mesh=mesh, reshard_after_forward=True)
            actor_opt = torch.optim.SGD(actor.parameters(), lr=.125)
            actor.to('cpu')
            self.assertTrue(all(entry.full_param is None
                and int(entry.shard.placement_backend) == 0
                for entry in actor._fsdp_state.true_fsdp_params))
            critic = Model()
            fully_shard(critic, mesh=mesh, reshard_after_forward=True)
            critic_opt = torch.optim.SGD(critic.parameters(), lr=.125)
            self.assertIsNot(actor_opt, critic_opt)  # Keep both original optimizers alive.
            x = torch.tensor([[1., 2.], [3., 4.]], device='cuda:0',
                dtype=torch.float32, requires_grad=True)
            y = critic(x)
            self.runtime_evidence['forward'] = self.prove_cuda(y)
            self.assertTrue(torch.equal(y, torch.tensor(
                [[1.125, .625], [2.625, 1.125]], device='cuda:0', dtype=torch.float32)))
            loss = (y * float(int(jt.rank) + 1)).sum()
            cpu_token = torch.empty((1,), device='cpu')
            with placement_scope_like(cpu_token, override=True):
                loss.backward()
            self.runtime_evidence['input_gradient'] = self.prove_cuda(x.grad)
            factor = float(int(jt.rank) + 1)
            self.assertTrue(torch.equal(x.grad, torch.tensor(
                [[.25 * factor, .75 * factor]] * 2, device='cuda:0', dtype=torch.float32)))
            for entry in critic._fsdp_state.true_fsdp_params:
                self.assertIsNotNone(entry.shard.grad)
                actual = entry.shard.grad.full_tensor()
                self.runtime_evidence[entry.name + '_gradient'] = self.prove_cuda(actual)
                expected = [[6., 9.], [6., 9.]] if entry.name == 'weight' else [3., 3.]
                self.assertTrue(torch.equal(actual, torch.tensor(
                    expected, device='cuda:0', dtype=torch.float32)))
            self.assertTrue(all(entry.full_param is None and entry.shard.grad is None
                and int(entry.shard.placement_backend) == 0
                for entry in actor._fsdp_state.true_fsdp_params))
            self.runtime_evidence['checks'].update(original_backward_returned=True,
                inactive_CPU_state_not_unsharded=True, inactive_gradients_remain_None=True,
                complete_CUDA_gradients_exact=True)
        finally:
            if previous_flat is None:
                os.environ.pop('JITTOR_FSDP2_FLAT', None)
            else:
                os.environ['JITTOR_FSDP2_FLAT'] = previous_flat
            self.runtime_evidence['flat_environment_restored'] = (
                os.environ.get('JITTOR_FSDP2_FLAT') == previous_flat)
        self.assertTrue(self.runtime_evidence['flat_environment_restored'])

    def test_flat_completed_CPU_optimizer_forced_cached_identity_collision(self):
        self.run_completed_optimizer_identity_case(True)

    def test_nonflat_completed_CPU_optimizer_forced_cached_identity_collision(self):
        self.run_completed_optimizer_identity_case(False)

    def run_completed_optimizer_identity_case(self, flat):
        """Forced mechanism collision, NOT an observed CPython reuse/root-cause claim."""
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard
        previous_flat = os.environ.get('JITTOR_FSDP2_FLAT')
        os.environ['JITTOR_FSDP2_FLAT'] = '1' if flat else '0'
        saved = []

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(
                    [[.5, .25], [-.25, .5]], device='cuda:0', dtype=torch.float32))
                self.bias = torch.nn.Parameter(torch.tensor(
                    [.125, -.125], device='cuda:0', dtype=torch.float32))

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias)

        self.runtime_evidence.update(forced_identity_collision=True,
            natural_id_reuse_observed=False, observed_full_PPO_root_cause_claimed=False,
            mechanism_scope='completed_Critic_CPU_storage_then_Actor_CUDA_backward')
        try:
            mesh = init_device_mesh('cuda', (2,))
            critic = Model()
            fully_shard(critic, mesh=mesh, reshard_after_forward=True)
            critic_opt = torch.optim.SGD(critic.parameters(), lr=.125)
            x = torch.tensor([[1., 2.], [3., 4.]], device='cuda:0', dtype=torch.float32)
            (critic(x) * float(int(jt.rank) + 1)).sum().backward()
            for entry in critic._fsdp_state.true_fsdp_params:
                actual = entry.shard.grad.full_tensor()
                self.prove_cuda(actual)
                expected = [[6., 9.], [6., 9.]] if entry.name == 'weight' else [3., 3.]
                self.assertTrue(torch.equal(actual, torch.tensor(expected, device='cuda:0', dtype=torch.float32)))
            critic_opt.step()
            critic_opt.zero_grad(set_to_none=True)
            critic.to('cpu')
            critic_entries = critic._fsdp_state.true_fsdp_params
            self.assertTrue(all(entry.full_param is None and entry.shard.grad is None
                and int(entry.shard.placement_backend) == 0 for entry in critic_entries))
            # Host copies witness idle CPU storage only; no CPU model or loss is executed.
            critic_storage = [entry.shard.numpy().copy() for entry in critic_entries]
            self.runtime_evidence['checks']['original_Critic_step_zero_grad_CPU_returned'] = True
            actor = Model()
            fully_shard(actor, mesh=mesh, reshard_after_forward=True)
            actor_opt = torch.optim.SGD(actor.parameters(), lr=.125)
            actor_x = torch.tensor([[1., 2.], [3., 4.]], device='cuda:0',
                dtype=torch.float32, requires_grad=True)
            y = actor(actor_x)
            self.runtime_evidence['actor_forward'] = self.prove_cuda(y)
            self.assertTrue(torch.equal(y, torch.tensor([[1.125, .625], [2.625, 1.125]],
                device='cuda:0', dtype=torch.float32)))
            actor_entries = actor._fsdp_state.true_fsdp_params
            self.assertEqual(len(critic_entries), len(actor_entries))
            for previous, active in zip(critic_entries, actor_entries):
                self.assertIsNotNone(active.full_param)
                self.assertEqual(tuple(previous.shape), tuple(active.shape))
                had = hasattr(previous, '_jittor_fsdp_full_param_id')
                saved.append((previous, had, getattr(previous, '_jittor_fsdp_full_param_id', None)))
                # Only this stale integer metadata is forced; no tensor, gradient,
                # optimizer, native identity, placement or collective is replaced.
                object.__setattr__(previous, '_jittor_fsdp_full_param_id', id(active.full_param))
            self.runtime_evidence['forced_cached_identity_count'] = len(saved)
            self.assertGreater(len(saved), 0)
            (y * float(int(jt.rank) + 1)).sum().backward()
            self.runtime_evidence['actor_input_gradient'] = self.prove_cuda(actor_x.grad)
            factor = float(int(jt.rank) + 1)
            self.assertTrue(torch.equal(actor_x.grad, torch.tensor([[.25 * factor, .75 * factor]] * 2,
                device='cuda:0', dtype=torch.float32)))
            for entry in actor_entries:
                actual = entry.shard.grad.full_tensor()
                self.runtime_evidence[entry.name + '_Actor_gradient'] = self.prove_cuda(actual)
                expected = [[6., 9.], [6., 9.]] if entry.name == 'weight' else [3., 3.]
                self.assertTrue(torch.equal(actual, torch.tensor(expected, device='cuda:0', dtype=torch.float32)))
            self.assertTrue(all(entry.full_param is None and entry.shard.grad is None
                and int(entry.shard.placement_backend) == 0 for entry in critic_entries))
            for entry, before in zip(critic_entries, critic_storage):
                self.assertTrue((entry.shard.numpy() == before).all())
            actor_opt.step()
            expected_parameters = {'weight': [[-.25, -.875], [-1., -.625]], 'bias': [-.25, -.5]}
            for entry in actor_entries:
                value = entry.shard.full_tensor()
                self.prove_cuda(value)
                self.assertTrue(torch.equal(value, torch.tensor(expected_parameters[entry.name],
                    device='cuda:0', dtype=torch.float32)))
            self.runtime_evidence['checks'].update(Actor_backward_and_step_exact=True,
                idle_Critic_gradients_remain_None=True, idle_Critic_storage_unchanged=True,
                forced_stale_integer_did_not_claim_foreign_gradient=True)
        finally:
            for entry, had, old in saved:
                if had:
                    object.__setattr__(entry, '_jittor_fsdp_full_param_id', old)
                else:
                    object.__delattr__(entry, '_jittor_fsdp_full_param_id')
            if previous_flat is None:
                os.environ.pop('JITTOR_FSDP2_FLAT', None)
            else:
                os.environ['JITTOR_FSDP2_FLAT'] = previous_flat
            self.runtime_evidence['flat_environment_restored'] = os.environ.get('JITTOR_FSDP2_FLAT') == previous_flat
        self.assertTrue(self.runtime_evidence['flat_environment_restored'])

    def test_flat_two_optimizers_share_one_live_full_gradient_map(self):
        self.run_live_shared_full_identity_case(True)

    def test_nonflat_two_optimizers_share_one_live_full_gradient_map(self):
        self.run_live_shared_full_identity_case(False)

    def run_live_shared_full_identity_case(self, flat):
        from jittor._core.var import placement_scope_like
        from jittor.compat.fsdp2 import grad_sync
        from jittor.optim.base import optimizer_execution_groups
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard
        previous_flat = os.environ.get('JITTOR_FSDP2_FLAT')
        os.environ['JITTOR_FSDP2_FLAT'] = '1' if flat else '0'

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([[.5, .25], [-.25, .5]],
                    device='cuda:0', dtype=torch.float32))

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight)

        try:
            model = Model()
            fully_shard(model, mesh=init_device_mesh('cuda', (2,)), reshard_after_forward=True)
            opt1 = torch.optim.SGD(model.parameters(), lr=.125)
            opt2 = torch.optim.SGD(model.parameters(), lr=.125)
            x = torch.tensor([[1., 2.], [3., 4.]], device='cuda:0', dtype=torch.float32)
            loss = (model(x) * float(int(jt.rank) + 1)).sum()
            # Strong references are intentionally live. Both optimizers own the
            # very same actual full parameter, not an unrelated reused integer.
            targets = grad_sync.collect_fsdp_full_params_for_backward([opt1, opt2])
            self.assertEqual(len(targets), 1)
            with placement_scope_like(targets[0], override=True):
                grads = jt.grad(loss, targets)
            grad_map = {id(param): grad for param, grad in zip(targets, grads)}
            self.assertTrue(grad_sync.fill_fsdp_optimizer_grads_from_grad_map([opt1, opt2], grad_map))
            entry = model._fsdp_state.true_fsdp_params[0]
            actual = entry.shard.grad.full_tensor()
            self.runtime_evidence['shared_full_gradient'] = self.prove_cuda(actual)
            self.assertTrue(torch.equal(actual, torch.tensor([[6., 9.], [6., 9.]],
                device='cuda:0', dtype=torch.float32)))
            slot1 = optimizer_execution_groups(opt1)[0]['grads'][0]
            slot2 = optimizer_execution_groups(opt2)[0]['grads'][0]
            self.assertIs(slot1, slot2)
            self.assertIs(slot1, entry.shard.grad)
            self.assertTrue(all(entry.full_param is None for entry in model._fsdp_state.true_fsdp_params))
            self.assertEqual(len(targets), 1)
            # The identical grad map is deliberately published a second time
            # while targets strongly retain its original actual full parameter.
            # Two optimizer aliases still count once per call: exactly 2x, not 4x.
            self.assertTrue(grad_sync.fill_fsdp_optimizer_grads_from_grad_map([opt1, opt2], grad_map))
            accumulated = entry.shard.grad.full_tensor()
            self.runtime_evidence['repeated_shared_full_gradient'] = self.prove_cuda(accumulated)
            self.assertTrue(torch.equal(accumulated, torch.tensor([[12., 18.], [12., 18.]],
                device='cuda:0', dtype=torch.float32)))
            self.assertIs(optimizer_execution_groups(opt1)[0]['grads'][0],
                optimizer_execution_groups(opt2)[0]['grads'][0])
            self.runtime_evidence['checks'].update(shared_live_full_collected_once=True,
                both_optimizers_share_exact_gradient_object=True, exact_global_gradient_not_doubled=True,
                repeated_identical_gradmap_accumulates_exactly_twice=True)
        finally:
            if previous_flat is None:
                os.environ.pop('JITTOR_FSDP2_FLAT', None)
            else:
                os.environ['JITTOR_FSDP2_FLAT'] = previous_flat
            self.runtime_evidence['flat_environment_restored'] = os.environ.get('JITTOR_FSDP2_FLAT') == previous_flat
        self.assertTrue(self.runtime_evidence['flat_environment_restored'])


    def run_case(self, flat):
        from jittor._core.var import placement_scope_like
        from jittor.compat.fsdp2 import grad_sync
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard
        previous_flat = os.environ.get('JITTOR_FSDP2_FLAT')
        self.previous_profile = sys.getprofile()
        self.assertIsNone(self.previous_profile, 'do not replace a caller-owned profiler')
        os.environ['JITTOR_FSDP2_FLAT'] = '1' if flat else '0'

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([[.5, .25], [-.25, .5]],
                    device='cuda:0', dtype=torch.float32))

        try:
            model = Model()
            fully_shard(model, mesh=init_device_mesh('cuda', (2,)), reshard_after_forward=True)
            state = model._fsdp_state
            self.assertEqual(state.true_fsdp_flat, flat)
            self.assertEqual(len(state.true_fsdp_params), 1)
            self.runtime_evidence['state_shard'] = self.prove_cuda(state.true_fsdp_params[0].shard)
            cpu_token = torch.empty((1,), dtype=torch.int32, device='cpu')
            self.assertEqual(int(cpu_token.placement_backend), 0)
            cases = [('ordinary', [True, False], [True, False]),
                     ('all_unused', [False, False], [False, False]),
                     ('rank_disagreement', [True, False] if int(jt.rank) == 0 else [False, True], [True, True]),
                     ('empty_flags_artificial_boundary', [], [])]
            for name, local_used, expected in cases:
                captures = []

                def profile(frame, event, result):
                    # Retain references only. Inspect them after the call, never
                    # perform device/property operations inside the observer.
                    if frame.f_code is grad_sync._globally_used_grads.__code__ and event == 'return':
                        captures.append((frame.f_locals.get('flags'), frame.f_locals.get('reduced')))

                with placement_scope_like(cpu_token, override=True):
                    sys.setprofile(profile)
                    try:
                        actual = grad_sync._globally_used_grads(local_used, state)
                    finally:
                        sys.setprofile(self.previous_profile)
                self.assertEqual(actual, expected)
                self.assertEqual(len(captures), 1)
                flags, reduced = captures[0]
                self.assertIsNotNone(flags)
                self.assertIsNotNone(reduced)
                row = dict(name=name, local_used=local_used, expected=expected, actual=actual,
                    flags=self.prove_cuda(flags), reduced=self.prove_collective_host_readback(reduced), passed=True)
                self.runtime_evidence['cases'].append(row)
                self.runtime_evidence['checks'][name] = True
                del flags, reduced, captures
            self.assertEqual(len(self.runtime_evidence['cases']), 4)
            self.assertTrue(all(self.runtime_evidence['checks'].values()))
        finally:
            if previous_flat is None:
                os.environ.pop('JITTOR_FSDP2_FLAT', None)
            else:
                os.environ['JITTOR_FSDP2_FLAT'] = previous_flat
            self.runtime_evidence['flat_environment_restored'] = os.environ.get('JITTOR_FSDP2_FLAT') == previous_flat
        self.assertTrue(self.runtime_evidence['flat_environment_restored'])


if __name__ == '__main__':
    unittest.main()

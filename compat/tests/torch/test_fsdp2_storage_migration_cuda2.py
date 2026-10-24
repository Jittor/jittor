"""Real CUDA2 FSDP storage regression, not an environment or full-PPO acceptance.

Run only under the maintained native NCCL2 launcher with isolated rank caches.
CPU is a storage destination. All arithmetic, comparisons and gradients are CUDA.
No private conversion prototype is installed; this tests production behavior.
"""
from contextlib import ExitStack
import ctypes
import importlib
import os
import unittest

import pytest
import jittor as jt
import torch
from jittor.compat.diagnostics import counts
from jittor._runtime.fallback import forbid_backend_fallbacks

pytestmark = pytest.mark.cuda


def require(value):
    if not value:
        raise AssertionError('real CUDA2 FSDP storage regression guard failed')


class TestFSDP2StorageMigrationCUDA2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if int(jt.world_size) != 2:
            raise unittest.SkipTest('requires actual NCCL world2; skip is not acceptance')
        if not torch.cuda.is_available():
            raise unittest.SkipTest('requires actual CUDA; skip is not acceptance')
        cls.dist = torch.distributed
        cls.created_group = not cls.dist.is_initialized()
        if cls.created_group:
            cls.dist.init_process_group('nccl', rank=int(jt.rank), world_size=2)
        require(cls.dist.get_world_size() == 2 and cls.dist.get_backend() == 'nccl')
        require(cls.dist.get_rank() == int(jt.rank))

    @classmethod
    def tearDownClass(cls):
        if cls.created_group:
            cls.dist.destroy_process_group()

    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(forbid_backend_fallbacks())
        stack.enter_context(jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error'))
        torch.cuda.set_device(0)
        require(torch.cuda.current_device() == 0 and jt.core.backend_fallback_count() == 0)
        self.driver = ctypes.CDLL('libcuda.so.1')
        self.driver.cuInit.argtypes = [ctypes.c_uint]
        self.driver.cuInit.restype = ctypes.c_int
        self.driver.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.driver.cuPointerGetAttribute.restype = ctypes.c_int
        require(self.driver.cuInit(0) == 0)
        self.initial_counts = counts()
        startup = sum(self.initial_counts.values())
        self.runtime_evidence = dict(scope='test_body_only_NOT_startup_or_PPO_acceptance',
            startup_swallowed_total=startup, startup_all_swallowed_zero=startup == 0,
            startup_nonzero_counts_preserved=startup > 0, CPU_storage_only=True,
            original_freeze_RED_not_relabelled=True)
        self.addCleanup(self.assert_no_new_diagnostics)

    def assert_no_new_diagnostics(self):
        torch.cuda.synchronize(0)
        current = counts()
        self.runtime_evidence.update(new_swallowed_count=sum(current.values()) - sum(self.initial_counts.values()),
            diagnostic_counts_unchanged=current == self.initial_counts,
            fallback_count=int(jt.core.backend_fallback_count()))
        self.assertTrue(current == self.initial_counts, 'test body introduced swallowed diagnostics')
        self.assertEqual(jt.core.backend_fallback_count(), 0)

    def _cuda_pointer(self, value):
        require(jt.core._is_tensor_carrier(value))
        require(tuple(jt.core.dispatch_context([value])) == ('cuda', 0))
        ordinal, memory_type = ctypes.c_int(-1), ctypes.c_int(-1)
        pointer = int(value.device_raw_ptr)
        require(pointer > 0)
        require(self.driver.cuPointerGetAttribute(ctypes.byref(ordinal), 9, ctypes.c_void_p(pointer)) == 0)
        require(self.driver.cuPointerGetAttribute(ctypes.byref(memory_type), 2, ctypes.c_void_p(pointer)) == 0)
        require((ordinal.value, memory_type.value) == (0, 2))

    def test_flat_storage_roundtrip_two_updates(self):
        self._run_case(True)

    def test_nonflat_storage_roundtrip_two_updates(self):
        self._run_case(False)

    def _run_case(self, flat):
        require(type(flat) is bool)
        rank = int(self.dist.get_rank())
        require(rank in (0, 1) and int(jt.rank) == rank and self.dist.get_world_size() == 2)
        report = dict(checks={}, observations=[], phase='construct', execution_returned=False,
            actual_optimizer_updates_returned=0, flat_requested=flat,
            reversible_freeze_subcase_executed=False, original_freeze_RED_not_relabelled=True)

        def check(name, value):
            report['checks'][name] = bool(value)
            return bool(value)

        def guard():
            require(torch.cuda.current_device() == 0 and self.dist.get_world_size() == 2)
            require(type(jt.core.backend_fallback_count()) is int and jt.core.backend_fallback_count() == 0)

        def resident(value, kind):
            value.sync()
            public = value.device
            require(public.type == kind and value.location() == ('cpu' if kind == 'cpu' else 'device'))
            require(int(value.placement_backend) == (0 if kind == 'cpu' else 1))
            require(int(value.device_id) == (-1 if kind == 'cpu' else 0))
            if kind == 'cuda':
                require(public.index == 0)
                if int(value.numel()):
                    self._cuda_pointer(value)
            else:
                require(public.index is None)

        def metadata(value, kind):
            resident(value, kind)
            cls = type(value)
            return dict(python_type=dict(module=cls.__module__, qualname=cls.__qualname__),
                        mro=[dict(module=c.__module__, qualname=c.__qualname__) for c in cls.__mro__],
                        shape=tuple(int(x) for x in value.shape), dtype=str(value.dtype),
                        public_device=dict(type=value.device.type, index=value.device.index),
                        native_location=value.location(), native_device_id=int(value.device_id),
                        native_placement_backend=int(value.placement_backend),
                        requires_grad=bool(value.requires_grad), native_stop=bool(value.is_stop_grad()),
                        version=int(value._version), storage_offset=int(value._storage_offset()),
                        storage_strides=tuple(int(x) for x in value._storage_strides()),
                        view_provenance=bool(value._is_view()))

        def equal(value, expected):
            resident(value, 'cuda')
            expected = torch.tensor(expected, device='cuda:0', dtype=torch.float32)
            resident(expected, 'cuda')
            require(tuple(value.shape) == tuple(expected.shape))
            return bool(torch.allclose(value, expected, atol=1e-5, rtol=1e-5))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.full((3, 4), .5, device='cuda:0', dtype=torch.float32))
                self.bias = torch.nn.Parameter(torch.full((3,), .25, device='cuda:0', dtype=torch.float32))
                self.frozen = torch.nn.Parameter(torch.full((4,), .125, device='cuda:0', dtype=torch.float32), requires_grad=False)
                self.weight_alias = self.weight
                self.register_buffer('marker', torch.full((3,), .25, device='cuda:0', dtype=torch.float32))

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias) + self.frozen[:3] + self.marker

        def current_entries(state):
            entries = tuple(state.true_fsdp_params)
            require(len(entries) == 3 and {e.name for e in entries} == {'weight', 'bias', 'frozen'})
            return entries

        def quiescent(state, model, optimizer, tag):
            entries = current_entries(state)
            check(tag + '_resharded', state.true_fsdp_unsharded is False)
            check(tag + '_no_full_or_pending_grad', all(e.full_param is None and e.shard.grad is None for e in entries))
            require(state.true_fsdp_unsharded is False and all(e.full_param is None and e.shard.grad is None for e in entries))
            check(tag + '_full_flat_released', getattr(state, 'true_fsdp_flat_full_param', None) is None)
            check(tag + '_canonical_aliases', all(getattr(model, e.name) is e.shard for e in entries)
                  and model.weight_alias is model.weight)
            check(tag + '_optimizer_resolves_current_entries', len(optimizer.param_groups[0]['params']) == 3
                  and all(any(p is e.shard for p in optimizer.param_groups[0]['params']) for e in entries))

        def observe(state, model, kind, tag):
            entries = current_entries(state)
            row = dict(tag=tag, parameter_metadata={e.name: metadata(e.shard, kind) for e in entries},
                       marker_metadata=metadata(model.marker, kind), current_local_is_entry={})
            for e in entries:
                row['current_local_is_entry'][e.name] = e.shard.to_local() is e.shard
            check(tag + '_current_local_world2_owner', all(row['current_local_is_entry'].values()))
            if flat:
                owner = state.true_fsdp_flat_shard
                row['flat_metadata'] = metadata(owner, kind)
                row['flat_alias_relations'] = {}
                for e in entries:
                    start, length = importlib.import_module('jittor.compat.fsdp2.shard')._flat_local_overlap(state, e)
                    require(int(e.shard.numel()) == length)
                    if not length:
                        row['flat_alias_relations'][e.name] = {'empty': True}
                        continue
                    same = int(e.shard.data_ptr()) == int(owner.data_ptr()) + start * 4
                    offset = int(e.shard._storage_offset()) == int(owner._storage_offset()) + start
                    row['flat_alias_relations'][e.name] = dict(pointer_offset_matches=same, storage_offset_matches=offset)
                    check(tag + '_flat_alias_' + e.name, same and offset)
            report['observations'].append(row)

        def full_values(state, weight_row, bias, tag):
            expected = {'weight': [list(weight_row)] * 3, 'bias': [bias] * 3, 'frozen': [.125] * 4}
            for e in current_entries(state):
                check(tag + '_full_' + e.name, equal(e.shard.full_tensor(), expected[e.name]))

        def training_step(model, state, optimizer, step):
            guard()
            x = torch.tensor([[1., .5, -.25, 2.], [-.5, .25, 1., -1.]],
                             device='cuda:0', dtype=torch.float32, requires_grad=True)
            dy = torch.full((2, 3), float(rank + 1), device='cuda:0', dtype=torch.float32)
            y = model(x)
            ys = (2.25, .5) if step == 1 else (1.37109375, .18359375)
            check('step%d_full_Y' % step, equal(y, [[ys[0]] * 3, [ys[1]] * 3]))
            loss = (y * dy).sum()
            resident(loss, 'cuda')
            loss.backward()
            require(x.grad is not None)
            dx = [1.5] * 4 if step == 1 else [1.21875, 1.078125, 1.078125, .9375]
            check('step%d_full_dX' % step, equal(x.grad, [[z * (rank + 1) for z in dx]] * 2))
            expected_grads = {'weight': [[.75, 1.125, 1.125, 1.5]] * 3, 'bias': [3.] * 3}
            for e in current_entries(state):
                if e.name == 'frozen':
                    check('step%d_frozen_grad_None' % step, e.shard.grad is None)
                else:
                    require(e.shard.grad is not None)
                    check('step%d_full_d%s' % (step, e.name), equal(e.shard.grad.full_tensor(), expected_grads[e.name]))
            optimizer.step()
            torch.cuda.synchronize(0)
            report['actual_optimizer_updates_returned'] += 1
            expected_weight = ([.40625, .359375, .359375, .3125] if step == 1
                               else [.265625, .1484375, .1484375, .03125])
            full_values(state, expected_weight, -.125 if step == 1 else -.6875, 'step%d' % step)
            optimizer.zero_grad(set_to_none=True)
            del x, dy, y, loss
            quiescent(state, model, optimizer, 'step%d_quiescent' % step)
            guard()

        def instant_storage_return(state, model, kind):
            # No sync, data_ptr, dispatch_context, full_tensor or data reads here.
            # Read stored owners first, BEFORE either public to_local call can create a view.
            def read_physical(value, source_value=None):
                native_value = getattr(value, '_var', value)
                require(jt.core._is_tensor_carrier(native_value))
                public = value.device
                dimensions = tuple(int(d) for d in native_value.shape)
                elements = 1
                for dimension in dimensions:
                    elements *= dimension
                location = native_value.location()
                backend = int(native_value.placement_backend)
                ordinal = int(native_value.device_id)
                expected = ('cpu', 0, -1, 'cpu', None) if kind == 'cpu' else ('device', 1, 0, 'cuda', 0)
                actual = (location, backend, ordinal, public.type, public.index)
                cls = type(value)
                return dict(python_type=dict(module=cls.__module__, qualname=cls.__qualname__),
                    mro=[dict(module=c.__module__, qualname=c.__qualname__) for c in cls.__mro__],
                    native_shape=dimensions, numel=elements, native_location=location,
                    native_placement_backend=backend, native_device_id=ordinal,
                    public_device=dict(type=public.type, index=public.index),
                    native_is_stop_grad=bool(native_value.is_stop_grad()),
                    public_requires_grad=bool(value.requires_grad), version=int(value._version),
                    same_public_holder_as_entry=value is source_value if source_value is not None else None,
                    same_native_holder_as_entry=native_value is getattr(source_value, '_var', source_value)
                        if source_value is not None else None,
                    physical_check_applicable=elements > 0, physical_matches_requested=actual == expected,
                    empty_metadata_matches_requested=(backend, ordinal, public.type, public.index) == expected[1:],
                    zero_sized_storage_allocation_not_claimed=elements == 0,
                    no_sync_or_data_pointer_or_tensor_values_read=True)

            entries = current_entries(state)
            row = dict(tag='immediate_after_public_to_' + kind, requested_storage_kind=kind,
                caller_grad_enabled=bool(torch.is_grad_enabled()),
                owned_flat=[], entries={}, no_grad_public_locals={}, grad_enabled_public_locals={},
                observation_before_original_resident_or_sync=True,
                default_grad_unmaterialized_local_not_accepted_as_materialized=True)
            report['observations'].append(row)
            if flat:
                row['owned_flat'].append(read_physical(state.true_fsdp_flat_shard))
            for entry in entries:
                row['entries'][entry.name] = read_physical(entry.shard, entry.shard)
            with torch.no_grad():
                for entry in entries:
                    local = entry.shard.to_local()
                    row['no_grad_public_locals'][entry.name] = read_physical(local, entry.shard)
                    del local
            with torch.enable_grad():
                for entry in entries:
                    local = entry.shard.to_local()
                    row['grad_enabled_public_locals'][entry.name] = read_physical(local, entry.shard)
                    del local
            stored = row['owned_flat'] + list(row['entries'].values()) + list(row['no_grad_public_locals'].values())
            default = list(row['grad_enabled_public_locals'].values())
            require(any(item['physical_check_applicable'] for item in stored))
            stored_ok = all(item['physical_matches_requested'] for item in stored if item['physical_check_applicable'])
            default_ok = all(item['physical_matches_requested'] for item in default if item['physical_check_applicable'])
            empty_ok = all(item['empty_metadata_matches_requested']
                for item in stored + default if not item['physical_check_applicable'])
            row.update(all_nonempty_stored_and_no_grad_local_actual_storage=stored_ok,
                all_nonempty_grad_enabled_local_actual_storage=default_ok,
                all_empty_requested_metadata_only=empty_ok)
            check(kind + '_instant_stored_owner_entry_and_no_grad_local_physical', stored_ok)
            check(kind + '_instant_grad_enabled_public_local_physical', default_ok)
            check(kind + '_instant_empty_metadata_only_no_allocation_claim', empty_ok)
            # Retain the Full observer's physical requirement, not just intended placement flags.
            # A new default-grad lazy clone is reported distinctly and is NOT certified as stored.
            # Preserve strict failures while collecting the complete math gates.
            # The final assertion requires every recorded check, including these.

        previous_flat = os.environ.get('JITTOR_FSDP2_FLAT')
        os.environ['JITTOR_FSDP2_FLAT'] = '1' if flat else '0'
        try:
            model = Model()
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.fsdp import fully_shard
            mesh = init_device_mesh('cuda', (2,), mesh_dim_names=('dp',))
            fully_shard(model, mesh=mesh, reshard_after_forward=True)
            state = model._fsdp_state
            require(state.true_fsdp_world_size == 2 and state.true_fsdp_flat is flat)
            report['actual_flat'] = state.true_fsdp_flat
            report['actual_entry_order'] = [e.name for e in current_entries(state)]
            check('total_logical_numel19', sum(e.numel for e in current_entries(state)) == 19)
            if flat:
                check('flat_local_numel10_padding1', state.true_fsdp_flat_shard_numel == 10
                      and state.true_fsdp_flat_padded_numel == 20)
            optimizer = torch.optim.SGD(model.parameters(), lr=.125, momentum=.5)
            quiescent(state, model, optimizer, 'initial')
            observe(state, model, 'cuda', 'initial')
            report['phase'] = 'CUDA_optimizer_update1'
            training_step(model, state, optimizer, 1)
            observe(state, model, 'cuda', 'after_step1')
            parameters = {e.name: e.shard for e in current_entries(state)}
            group = optimizer.param_groups[0]
            saved_groups = tuple(group['params'])
            saved_values = tuple(group.get('values', ()))
            saved_steps = tuple(group.get('_torch_steps', ()))
            require(len(saved_groups) == len(saved_values) == len(saved_steps) == 3)
            report['reversible_freeze_subcase_executed'] = False
            report['original_freeze_RED_not_relabelled'] = True
            policy = {e.name: (bool(e.shard.requires_grad), bool(e.shard.is_stop_grad()), int(e.shard._version))
                      for e in current_entries(state)}
            check('active_weight_trainability_before_storage', policy['weight'][0] is True)
            for destination, kind in (('cpu', 'cpu'), ('cuda:0', 'cuda')):
                report['phase'] = 'public_Module_to_' + kind
                check(kind + '_returns_same_module', model.to(device=destination) is model)
                instant_storage_return(state, model, kind)
                observe(state, model, kind, 'after_' + kind)
                check(kind + '_same_Parameter_objects', all(e.shard is parameters[e.name] for e in current_entries(state)))
                check(kind + '_same_optimizer_params', all(a is b for a, b in zip(group['params'], saved_groups)))
                check(kind + '_same_optimizer_slots', len(group.get('values', ())) == len(saved_values)
                      and all(a is b for a, b in zip(group.get('values', ()), saved_values)))
                current = {e.name: (bool(e.shard.requires_grad), bool(e.shard.is_stop_grad()), int(e.shard._version))
                           for e in current_entries(state)}
                report['observations'].append(dict(tag=kind + '_policy', before=policy, after=current))
                check(kind + '_active_and_permanent_policy_and_version', current == policy)
                guard()
            full_values(state, [.40625, .359375, .359375, .3125], -.125, 'reload')
            if flat:
                report['phase'] = 'public_local_write_through_and_restore'
                e = next(e for e in current_entries(state) if int(e.shard.numel()))
                local = e.shard.to_local()
                owner = state.true_fsdp_flat_shard
                start, length = importlib.import_module('jittor.compat.fsdp2.shard')._flat_local_overlap(state, e)
                with torch.no_grad():
                    before = owner.clone()
                    positions = torch.arange(int(owner.numel()), device='cuda:0')
                    expected = before + ((positions >= start) & (positions < start + length)).float() * .125
                    local.add_(.125)
                    check('current_local_public_write_reaches_flat', bool(torch.equal(owner, expected)))
                    local.sub_(.125)
                    check('current_local_public_restore_reaches_flat', bool(torch.equal(owner, before)))
                del local, before, expected, positions
                full_values(state, [.40625, .359375, .359375, .3125], -.125, 'after_alias_restore')
            check('active_weight_trainability_after_storage', bool(model.weight.requires_grad))
            check('optimizer_steps_unchanged_by_storage', tuple(group.get('_torch_steps', ())) == saved_steps)
            report['phase'] = 'CUDA_optimizer_update2_existing_optimizer'
            training_step(model, state, optimizer, 2)
            observe(state, model, 'cuda', 'after_step2')
            check('two_actual_optimizer_updates', report['actual_optimizer_updates_returned'] == 2)
            check('original_optimizer_momentum_step2', tuple(group.get('_torch_steps', ())) == tuple(
                n + (0 if p is model.frozen else 1) for n, p in zip(saved_steps, group['params'])))
            report['phase'] = 'complete'
            report['execution_returned'] = True
            if flat and rank == 0:
                empty = [e for e in current_entries(state) if int(e.shard.numel()) == 0]
                check('actual_empty_bias_and_frozen_shards', {e.name for e in empty} == {'bias', 'frozen'})
                for entry in empty:
                    resident(entry.shard, 'cuda')
                check('actual_empty_frozen_grad_None', model.frozen.grad is None)
            guard()
            self.assertEqual(report['actual_optimizer_updates_returned'], 2)
            self.assertTrue(all(report['checks'].values()), 'complete CUDA2 storage/value/gradient/alias/policy check failed')
        finally:
            if previous_flat is None:
                os.environ.pop('JITTOR_FSDP2_FLAT', None)
            else:
                os.environ['JITTOR_FSDP2_FLAT'] = previous_flat
            self.runtime_evidence.update(report)
            self.runtime_evidence['flat_environment_restored'] = os.environ.get('JITTOR_FSDP2_FLAT') == previous_flat
        self.assertTrue(self.runtime_evidence['flat_environment_restored'])

    def test_flat_trainable_nograd_offload_lifetime(self):
        self._run_nograd_lifetime_case(True)

    def test_nonflat_trainable_nograd_offload_lifetime(self):
        self._run_nograd_lifetime_case(False)

    def _run_nograd_lifetime_case(self, flat):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard
        previous = os.environ.get('JITTOR_FSDP2_FLAT')
        os.environ['JITTOR_FSDP2_FLAT'] = '1' if flat else '0'
        evidence = self.runtime_evidence
        evidence.update(nograd_cases=[], actual_nograd_offload_returns=0,
                        all_original_storage_two_SGD_tests_retained=True)

        class Model(torch.nn.Module):
            def __init__(self, inner_grad=False):
                super().__init__()
                self.inner_grad = inner_grad
                self.weight = torch.nn.Parameter(torch.tensor([[.5, .25], [-.25, .5]],
                    device='cuda:0', dtype=torch.float32))
                self.bias = torch.nn.Parameter(torch.tensor([.125, -.125],
                    device='cuda:0', dtype=torch.float32))

            def forward(self, x):
                if self.inner_grad:
                    with torch.enable_grad():
                        return torch.nn.functional.linear(x, self.weight, self.bias)
                return torch.nn.functional.linear(x, self.weight, self.bias)

            def execute(self, x):
                return self.forward(x)

        def new_model(reshard=True, inner_grad=False):
            model = Model(inner_grad)
            fully_shard(model, mesh=init_device_mesh('cuda', (2,)), reshard_after_forward=reshard)
            self.assertEqual(model._fsdp_state.true_fsdp_flat, flat)
            return model

        def exact_cuda(value, expected):
            value.sync()
            self._cuda_pointer(value)
            reference = torch.tensor(expected, device='cuda:0', dtype=torch.float32)
            reference.sync()
            self._cuda_pointer(reference)
            self.assertTrue(bool(torch.equal(value, reference)))

        def fulls(state):
            return tuple(entry.full_param for entry in state.true_fsdp_params)

        try:
            for direct in (False, True):
                model = new_model()
                state = model._fsdp_state
                x = torch.tensor([[1., 2.], [3., 4.]], device='cuda:0', dtype=torch.float32)
                x.sync()
                self._cuda_pointer(x)
                row = dict(direct_execute=direct)
                evidence['nograd_cases'].append(row)
                with torch.no_grad():
                    y = model.execute(x) if direct else model(x)
                row['fulls_released_immediately'] = all(value is None for value in fulls(state))
                self.assertTrue(row['fulls_released_immediately'])
                self.assertIsNone(state.true_fsdp_flat_full_param)
                self.assertFalse(state.true_fsdp_unsharded)
                exact_cuda(y, [[1.125, .625], [2.625, 1.125]])
                identities = tuple(id(entry.shard) for entry in state.true_fsdp_params)
                with torch.no_grad():
                    model.to('cpu', non_blocking=True)
                evidence['actual_nograd_offload_returns'] += 1
                for entry in state.true_fsdp_params:
                    # Residency only: no arithmetic is executed on CPU.
                    self.assertEqual((entry.shard.device.type, entry.shard.location()), ('cpu', 'cpu'))
                    self.assertIsNone(entry.shard.grad)
                with torch.no_grad():
                    model.to('cuda:0', non_blocking=True)
                    reloaded = model.execute(x) if direct else model(x)
                exact_cuda(reloaded, [[1.125, .625], [2.625, 1.125]])
                self.assertEqual(identities, tuple(id(entry.shard) for entry in state.true_fsdp_params))
                self.assertTrue(all(value is None for value in fulls(state)))

                # A real grad-enabled graph still owns its exact full Vars.
                live = new_model()
                live_state = live._fsdp_state
                live_y = live.execute(x) if direct else live(x)
                retained = fulls(live_state)
                self.assertTrue(any(value is not None for value in retained))
                with self.assertRaises(NotImplementedError):
                    live.to('cpu')
                self.assertTrue(all(a is b for a, b in zip(retained, fulls(live_state))))
                # A later no-grad forward cannot overwrite that older graph.
                with torch.no_grad(), self.assertRaises(NotImplementedError):
                    live.execute(x) if direct else live(x)
                self.assertTrue(all(a is b for a, b in zip(retained, fulls(live_state))))
                exact_cuda(live_y, [[1.125, .625], [2.625, 1.125]])

                nested = new_model(inner_grad=True)
                nested_state = nested._fsdp_state
                with torch.no_grad():
                    nested_y = nested.execute(x) if direct else nested(x)
                self.assertTrue(nested_y.requires_grad)
                self.assertTrue(any(value is not None for value in fulls(nested_state)))
                with self.assertRaises(NotImplementedError):
                    nested.to('cpu')

                unresharded = new_model(reshard=False)
                unresharded_state = unresharded._fsdp_state
                with torch.no_grad():
                    unresharded_y = unresharded.execute(x) if direct else unresharded(x)
                self.assertTrue(unresharded_state.true_fsdp_unsharded)
                self.assertTrue(any(value is not None for value in fulls(unresharded_state)))
                with self.assertRaises(NotImplementedError):
                    unresharded.to('cpu')
                exact_cuda(unresharded_y, [[1.125, .625], [2.625, 1.125]])

                pending = new_model()
                pending_state = pending._fsdp_state
                entry = next(entry for entry in pending_state.true_fsdp_params if int(entry.shard.numel()))
                gradient = torch.zeros_like(entry.shard)
                entry.shard.grad = gradient
                stored_gradient = entry.shard.grad
                self.assertIsNotNone(stored_gradient)
                with torch.no_grad(), self.assertRaises(NotImplementedError):
                    pending.execute(x) if direct else pending(x)
                self.assertIs(entry.shard.grad, stored_gradient)
                self.assertTrue(all(value is None for value in fulls(pending_state)))
                entry.shard.grad = None
                row.update(older_live_graph_preserved=True, nested_enable_grad_preserved=True,
                           reshard_false_preserved=True, pending_gradient_preserved=True)
            self.assertEqual(evidence['actual_nograd_offload_returns'], 2)
        finally:
            if previous is None:
                os.environ.pop('JITTOR_FSDP2_FLAT', None)
            else:
                os.environ['JITTOR_FSDP2_FLAT'] = previous
            evidence['flat_environment_restored'] = os.environ.get('JITTOR_FSDP2_FLAT') == previous
        self.assertTrue(evidence['flat_environment_restored'])

if __name__ == '__main__':
    unittest.main()

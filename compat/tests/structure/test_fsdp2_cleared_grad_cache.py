"""Metadata-only FSDP cache lifetime controls; no framework import or JIT."""
import ast
from contextlib import nullcontext
import gc
from pathlib import Path
from types import SimpleNamespace
import unittest
import weakref

ROOT = Path(__file__).resolve().parents[2]


class Payload:
    pass


class OwnedCarrier:
    """Symbolic metadata only: no tensor math, framework or array imports."""
    shape = (1,)

    def __init__(self, label, dtype='float32', events=None):
        self.label, self.dtype = label, dtype
        self.events = [] if events is None else events

    def cast(self, dtype):
        self.events.append(('cast', self.label, dtype))
        return OwnedCarrier(('cast', self.label), dtype, self.events)

    def __add__(self, other):
        return OwnedCarrier(('add', self.label, other.label), self.dtype, self.events)

    def stop_grad(self):
        return self

    def sync(self):
        self.events.append(('sync', self.label))


class LegacyCarrier(OwnedCarrier):
    @staticmethod
    def copy(value):
        if not isinstance(value, LegacyCarrier):
            raise TypeError('legacy descriptor received a sibling carrier')
        return value


def accumulation_environment(pending=None, sync=False):
    events, calls = [], []
    state = SimpleNamespace(requires_gradient_sync=sync, true_fsdp_flat=False,
        true_fsdp_params=[SimpleNamespace(shard=OwnedCarrier('parameter')) for _ in range(2)],
        true_fsdp_pending_full_grads=pending, mp_policy=None)
    def copy(value):
        events.append(('copy', value.label))
        return OwnedCarrier(('copy', value.label), value.dtype, events)
    def delegate(*args, **kwargs):
        calls.append((args, kwargs)); return [None, None]
    tree = ast.parse((ROOT / 'fsdp2/grad_sync.py').read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
        and node.name in ('_accumulate_owned_full_grads', '_sync_sharded_grads_from_full_grads')]
    for node in selected:
        node.decorator_list = []
    namespace = dict(copy_native_tensor=copy, _native_sync_sharded_grads=delegate,
        common=SimpleNamespace(_factory_scope_like=lambda *args, **kwargs: nullcontext(), nullcontext=nullcontext),
        shard=SimpleNamespace(), jt=SimpleNamespace(Var=LegacyCarrier,
            core=SimpleNamespace(Var=LegacyCarrier,
                _is_tensor_carrier=lambda value: isinstance(value, OwnedCarrier))))
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'grad_sync.py', 'exec'), namespace)
    return namespace, state, events, calls


def environment(count=2, flat=True):
    entries = [SimpleNamespace(shard=SimpleNamespace(_torch_grad=None),
        full_param=None, last_grad=None, full_public_grad=None) for _ in range(count)]
    state = SimpleNamespace(true_fsdp_params=entries, true_fsdp_flat=flat,
        true_fsdp_last_grads=[Payload() for _ in entries],
        true_fsdp_pending_full_grads=None)
    if flat:
        state.true_fsdp_last_flat_grad = Payload()
    for entry in entries:
        entry.shard.state, entry.shard.entry = state, entry
    shard = SimpleNamespace(_fsdp_param_entry=lambda parameter: (parameter.state, parameter.entry))
    tree = ast.parse((ROOT / 'fsdp2/grad_sync.py').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == '_release_cleared_optimizer_grad_cache')
    namespace = {'shard': shard}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'grad_sync.py', 'exec'), namespace)
    opt = SimpleNamespace(param_groups=[{'params': [entry.shard for entry in entries]}])
    return namespace['_release_cleared_optimizer_grad_cache'], state, entries, opt


class TestClearedGradCache(unittest.TestCase):
    def test_original_metadata_reference_survives_public_grad_clear(self):
        _, state, entries, _ = environment()
        reference = weakref.ref(state.true_fsdp_last_grads[0])
        for entry in entries:
            entry.shard._torch_grad = entry.last_grad = entry.full_public_grad = None
        gc.collect()
        self.assertIsNotNone(reference())  # Original defect: the state is another owner.

    def test_complete_flat_release_drops_only_its_own_references(self):
        release, state, _, opt = environment()
        refs = [weakref.ref(value) for value in state.true_fsdp_last_grads]
        flat = weakref.ref(state.true_fsdp_last_flat_grad)
        release(opt); gc.collect()
        self.assertTrue(all(reference() is None for reference in refs))
        self.assertIsNone(flat())
        self.assertEqual(state.true_fsdp_last_grads, [None, None])

    def test_nonflat_external_snapshot_not_mutated(self):
        release, state, _, opt = environment(flat=False)
        external = state.true_fsdp_last_grads[:]
        original_list = state.true_fsdp_last_grads
        release(opt)
        self.assertEqual(original_list, external)
        self.assertTrue(all(value is not None for value in external))
        self.assertIsNot(state.true_fsdp_last_grads, original_list)

    def test_partial_optimizer_keeps_other_valid_gradient_and_flat_buffer(self):
        release, state, entries, opt = environment()
        opt.param_groups[0]['params'] = [entries[0].shard]
        entries[1].shard._torch_grad = entries[1].last_grad = state.true_fsdp_last_grads[1]
        other, flat = state.true_fsdp_last_grads[1], state.true_fsdp_last_flat_grad
        release(opt)
        self.assertIsNone(state.true_fsdp_last_grads[0])
        self.assertIs(state.true_fsdp_last_grads[1], other)
        self.assertIs(state.true_fsdp_last_flat_grad, flat)

    def test_two_partial_clears_release_flat_only_after_no_contributions_remain(self):
        release, state, entries, opt = environment()
        opt.param_groups[0]['params'] = [entries[0].shard]
        flat = state.true_fsdp_last_flat_grad
        release(opt); self.assertIs(state.true_fsdp_last_flat_grad, flat)
        opt.param_groups[0]['params'] = [entries[1].shard]
        release(opt); self.assertIsNone(state.true_fsdp_last_flat_grad)

    def test_pending_no_sync_contribution_is_not_erased(self):
        release, state, _, opt = environment()
        pending = [Payload(), None]; state.true_fsdp_pending_full_grads = pending
        old, flat = state.true_fsdp_last_grads[0], state.true_fsdp_last_flat_grad
        release(opt)
        self.assertIs(state.true_fsdp_last_grads[0], old)
        self.assertIsNone(state.true_fsdp_last_grads[1])
        self.assertIs(state.true_fsdp_pending_full_grads, pending)
        self.assertIs(state.true_fsdp_last_flat_grad, flat)

    def test_full_and_public_gradient_references_prevent_premature_clear(self):
        for field in ('last_grad', 'full_public_grad', 'full_param'):
            release, state, entries, opt = environment()
            value = Payload()
            setattr(entries[0], field, SimpleNamespace(_torch_grad=value) if field == 'full_param' else value)
            old = state.true_fsdp_last_grads[0]
            release(opt); self.assertIs(state.true_fsdp_last_grads[0], old)

    def test_foreign_state_is_not_reached(self):
        release, state, _, opt = environment()
        _, foreign, _, _ = environment()
        before = foreign.true_fsdp_last_grads
        release(opt); self.assertIs(foreign.true_fsdp_last_grads, before)

    def test_already_empty_cache_and_missing_cache_are_supported(self):
        for empty in ([], None):
            release, state, _, opt = environment()
            state.true_fsdp_last_grads = empty
            release(opt); self.assertIsNone(state.true_fsdp_last_flat_grad)

    def test_malformed_inventory_fails_before_rebinding(self):
        for name in ('true_fsdp_last_grads', 'true_fsdp_pending_full_grads'):
            release, state, _, opt = environment()
            setattr(state, name, [Payload()]); before = state.true_fsdp_last_grads
            with self.assertRaises(RuntimeError): release(opt)
            self.assertIs(state.true_fsdp_last_grads, before)

    def test_public_integration_runs_only_after_none_clear(self):
        tree = ast.parse((ROOT / 'torch/optimizer_api.py').read_text())
        zero = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_zero_grad_compat')
        branch = next(node for node in ast.walk(zero) if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name) and node.test.id == 'set_to_none'
            and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                    and child.func.attr == '_release_cleared_optimizer_grad_cache' for child in ast.walk(node)))
        self.assertIsInstance(branch.test, ast.Name)
        names = [node.func.attr for node in ast.walk(zero) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)]
        self.assertLess(names.index('refresh_visible_full_grads'), names.index('_release_cleared_optimizer_grad_cache'))
        source = (ROOT / 'fsdp_hooks.py').read_text()
        self.assertIn('"_release_cleared_optimizer_grad_cache"', source)
        source = (ROOT / 'fsdp2/__init__.py').read_text()
        self.assertIn('from .grad_sync import _release_cleared_optimizer_grad_cache', source)

    def test_private_seam_does_not_expand_fsdp_public_exports(self):
        for relative, name in (('fsdp2/grad_sync.py', '_EXPORTS'), ('fsdp2/__init__.py', '__all__')):
            tree = ast.parse((ROOT / relative).read_text())
            declaration = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
            self.assertNotIn('_release_cleared_optimizer_grad_cache', ast.literal_eval(declaration))

    def test_native_legacy_receiver_reproduces_owned_no_sync_failure(self):
        tree = ast.parse((ROOT.parent / 'python/jittor/distributed/fsdp/gradients.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
            and node.name == '_sync_sharded_grads_from_full_grads')
        namespace = {'jt': SimpleNamespace(Var=LegacyCarrier)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'original_native_gradients.py', 'exec'), namespace)
        state = SimpleNamespace(requires_gradient_sync=False, mp_policy=None)
        api = SimpleNamespace(common=SimpleNamespace())
        with self.assertRaises(TypeError):
            namespace[function.name](state, [OwnedCarrier('new')], api=api)
        self.assertFalse(hasattr(state, 'true_fsdp_pending_full_grads'))

    def test_owned_accumulation_uses_real_copy_contract_and_preserves_None(self):
        old = OwnedCarrier('old', dtype='bfloat16')
        namespace, state, events, calls = accumulation_environment([old, None])
        state.mp_policy = SimpleNamespace(reduce_dtype='bfloat16', param_dtype='float32')
        value = OwnedCarrier('new', events=events)
        result = namespace['_sync_sharded_grads_from_full_grads'](state, [value, None])
        self.assertEqual(result, [None, None]); self.assertEqual(calls, [])
        self.assertEqual(state.true_fsdp_pending_full_grads[0].label, ('copy', ('add', 'old', ('cast', 'new'))))
        self.assertIsNone(state.true_fsdp_pending_full_grads[1])
        self.assertEqual([row[0] for row in events], ['cast', 'copy', 'sync'])
        self.assertEqual(value.dtype, 'float32'); self.assertEqual(old.label, 'old')

    def test_owned_pending_routes_when_all_new_gradients_are_None(self):
        old = OwnedCarrier('old')
        namespace, state, events, calls = accumulation_environment([old, None])
        self.assertEqual(namespace['_sync_sharded_grads_from_full_grads'](state, [None, None]), [None, None])
        self.assertEqual(calls, []); self.assertEqual(events, [])
        self.assertIs(state.true_fsdp_pending_full_grads[0], old)

    def test_legacy_no_sync_and_all_synchronized_cases_keep_native_delegate(self):
        for sync, value in ((False, LegacyCarrier('legacy')), (True, OwnedCarrier('owned'))):
            namespace, state, events, calls = accumulation_environment(sync=sync)
            original = [value, None]
            self.assertEqual(namespace['_sync_sharded_grads_from_full_grads'](state, original), [None, None])
            self.assertEqual(len(calls), 1); self.assertIs(calls[0][0][1], original)
            self.assertEqual(events, [])

    def test_native_receiver_owner_is_core_Var_not_frontend_alias(self):
        namespace, state, events, calls = accumulation_environment()
        namespace['jt'].Var = OwnedCarrier
        namespace['_sync_sharded_grads_from_full_grads'](state, [OwnedCarrier('owned'), None])
        self.assertEqual(calls, [])
        self.assertEqual([row[0] for row in events], ['copy', 'sync'])

    def test_pending_inventory_is_checked_before_any_cast_or_copy(self):
        namespace, state, events, calls = accumulation_environment([OwnedCarrier('old')])
        state.mp_policy = SimpleNamespace(reduce_dtype='bfloat16')
        with self.assertRaises(RuntimeError):
            namespace['_sync_sharded_grads_from_full_grads'](state, [OwnedCarrier('new', events=events), None])
        self.assertEqual(events, []); self.assertEqual(calls, [])

    def test_failed_copy_does_not_publish_partial_pending_list(self):
        pending = [OwnedCarrier('a'), OwnedCarrier('b')]
        namespace, state, events, calls = accumulation_environment(pending)
        seen = []
        def failing_copy(value):
            seen.append(value)
            if len(seen) == 2: raise TypeError('inert copy failure')
            return OwnedCarrier('completed-but-not-published')
        namespace['copy_native_tensor'] = failing_copy
        with self.assertRaises(TypeError):
            namespace['_sync_sharded_grads_from_full_grads'](state, [OwnedCarrier('new-a'), OwnedCarrier('new-b')])
        self.assertIs(state.true_fsdp_pending_full_grads, pending)
        self.assertEqual([value.label for value in pending], ['a', 'b'])
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()

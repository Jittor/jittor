"""Canonical checkpoint planning contracts; stdlib symbols, no runtime/JIT."""
import copy
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]


class Parameter:
    requires_grad = True
    grad = None
    device = 'cpu'


class Model:
    def __init__(self):
        self.weight, self.bias = Parameter(), Parameter()
        self._is_fsdp_module = False

    def named_parameters(self, **kwargs):
        return iter((('weight', self.weight), ('bias', self.bias)))

    def modules(self):
        return (self,)


class Optimizer:
    def __init__(self, groups, state):
        self.param_groups = [dict(group) for group in groups]
        self.parameters = [parameter for group in groups for parameter in group['params']]
        self.state = copy.deepcopy(state)
        self.loads = 0

    def state_dict(self):
        pids = {id(parameter): index for index, parameter in enumerate(self.parameters)}
        return {'state': copy.deepcopy(self.state), 'param_groups': [dict(group,
            params=[pids[id(parameter)] for parameter in group['params']]) for group in self.param_groups]}

    def load_state_dict(self, state):
        self.loads += 1
        self.state = copy.deepcopy(state['state'])
        for current, saved in zip(self.param_groups, state['param_groups']):
            current.update((key, value) for key, value in saved.items() if key != 'params')


class CheckpointStateHostContracts(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        self.provider = None
        self.events = []
        self.dist = types.SimpleNamespace(is_initialized=lambda: False, get_world_size=lambda: 1,
            all_gather_object=lambda ranks, metadata: self.events.append('public-all-gather') or ranks.__setitem__(0, metadata))
        self.frontend = types.SimpleNamespace(distributed=self.dist,
            nn=types.SimpleNamespace(parallel=types.SimpleNamespace(DistributedDataParallel=type('DDP', (), {}))))
        native = types.ModuleType('jittor')
        native.Var, native.Module = type('Var', (), {}), type('Module', (), {})
        package = types.ModuleType('_checkpoint_state_contract')
        package.__path__ = []
        torch_package = types.ModuleType(package.__name__ + '.torch')
        torch_package.__path__ = []
        hooks = types.ModuleType(package.__name__ + '.fsdp_hooks')
        hooks.provider = lambda: self.provider
        context = types.ModuleType(torch_package.__name__ + '.context')
        context.get_install_context = lambda _: types.SimpleNamespace(target_namespace=self.frontend)
        self.modules = mock.patch.dict(sys.modules, {native.__name__: native, package.__name__: package,
            torch_package.__name__: torch_package, hooks.__name__: hooks, context.__name__: context})
        self.modules.start()
        spec = importlib.util.spec_from_file_location(torch_package.__name__ + '.checkpoint_state',
            ROOT / 'compat/torch/checkpoint_state.py')
        self.owner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.owner)

    def tearDown(self):
        self.modules.stop()

    def options(self, **kwargs):
        return types.SimpleNamespace(**dict(dict(full_state_dict=False, flatten_optimizer_state_dict=False,
            broadcast_from_rank0=False, strict=True, cpu_offload=False), **kwargs))

    def optimizer(self, state=None):
        return Optimizer([{'params': [self.model.weight], 'lr': .01},
                          {'params': [self.model.bias], 'lr': .015}],
                         {0: {'step': 1}, 1: {'step': 1}} if state is None else state)

    def save(self, optimizer):
        return self.owner.get_optimizer_state_dict(self.model, optimizer, options=self.options())

    def load(self, optimizer, state, **kwargs):
        return self.owner.set_optimizer_state_dict(self.model, optimizer, state, options=self.options(**kwargs))

    def test_initialized_optimizer_preserves_never_initialized_parameter(self):
        optimizer = self.optimizer({0: {'step': 1}})
        saved = self.save(optimizer)
        self.assertEqual(set(saved['state']), {'weight'})
        self.load(optimizer, saved)
        self.assertEqual(set(self.save(optimizer)['state']), {'weight'})

    def test_strict_missing_previously_initialized_state_refused_before_apply(self):
        optimizer = self.optimizer()
        saved = self.save(optimizer)
        del saved['state']['bias']
        saved['param_groups'][0]['lr'] = .9
        with self.assertRaisesRegex(RuntimeError, 'bias'):
            self.load(optimizer, saved)
        self.assertEqual(optimizer.loads, 0)
        self.assertEqual(optimizer.param_groups[0]['lr'], .01)

    def test_nonstrict_missing_state_resets_only_that_parameter(self):
        optimizer = self.optimizer()
        saved = self.save(optimizer)
        del saved['state']['bias']
        self.load(optimizer, saved, strict=False)
        self.assertEqual(optimizer.state, {0: {'step': 1}})

    def test_duplicate_group_membership_refused_before_any_optimizer_changes(self):
        optimizer = self.optimizer()
        saved = self.save(optimizer)
        saved['param_groups'][1]['params'] = ['weight']
        saved['param_groups'][0]['lr'] = .9
        with self.assertRaises(ValueError):
            self.load(optimizer, saved)
        self.assertEqual(optimizer.loads, 0)
        self.assertEqual(optimizer.param_groups[0]['lr'], .01)

    def test_reordered_groups_match_by_fqn(self):
        optimizer = self.optimizer()
        saved = self.save(optimizer)
        saved['param_groups'].reverse()
        saved['param_groups'][0]['lr'] = .8
        saved['param_groups'][1]['lr'] = .7
        self.load(optimizer, saved)
        self.assertEqual([group['lr'] for group in optimizer.param_groups], [.7, .8])

    def test_multiple_optimizer_strict_failure_is_atomic(self):
        first = Optimizer([{'params': [self.model.weight], 'lr': .01}], {0: {'step': 1}})
        second = Optimizer([{'params': [self.model.bias], 'lr': .02}], {0: {'step': 1}})
        saved = self.save((first, second))
        del saved['state']['bias']
        saved['param_groups'][0]['lr'] = .9
        with self.assertRaisesRegex(RuntimeError, 'bias'):
            self.load((first, second), saved)
        self.assertEqual(first.loads, 0)
        self.assertEqual(second.loads, 0)
        self.assertEqual(first.param_groups[0]['lr'], .01)

    def test_combined_checkpoint_can_load_one_selected_optimizer(self):
        first = Optimizer([{'params': [self.model.weight], 'lr': .01}], {0: {'step': 1}})
        second = Optimizer([{'params': [self.model.bias], 'lr': .02}], {0: {'step': 1}})
        saved = self.save((first, second))
        saved['param_groups'][0]['lr'] = .3
        self.load(first, saved)
        self.assertEqual(first.param_groups[0]['lr'], .3)

    def test_nonstrict_partial_group_retains_live_parameter_bindings(self):
        optimizer = Optimizer([{'params': [self.model.weight, self.model.bias], 'lr': .01}],
                              {0: {'step': 1}, 1: {'step': 1}})
        saved = self.save(optimizer)
        saved['param_groups'][0]['params'] = ['weight']
        saved['param_groups'][0]['lr'] = .3
        del saved['state']['bias']
        self.load(optimizer, saved, strict=False)
        self.assertEqual(optimizer.param_groups[0]['params'], [self.model.weight, self.model.bias])
        self.assertEqual(optimizer.param_groups[0]['lr'], .3)

    def test_fsdp_singleton_uses_declared_refresh_and_public_collective(self):
        optimizer = self.optimizer()
        self.model._is_fsdp_module = True
        self.provider = types.SimpleNamespace(
            refresh_optimizer_fsdp_params=lambda optimizer: self.events.append('refresh'),
            _gather_optimizer_state_dict=lambda model, state, ranks: self.events.append(('gather', len(ranks))) or state)
        # No undeclared/nonexistent _optimizer_checkpoint_parameter_names hook.
        saved = self.owner.get_optimizer_state_dict(self.model, optimizer, options=self.options(full_state_dict=True))
        self.assertEqual(set(saved['state']), {'weight', 'bias'})
        self.assertEqual(self.events, ['refresh', 'public-all-gather', ('gather', 1)])


if __name__ == '__main__':
    unittest.main()

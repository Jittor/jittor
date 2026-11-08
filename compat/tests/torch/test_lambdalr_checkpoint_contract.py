"""LambdaLR public checkpoints match the pinned Torch 2.7 schema and restore."""
import copy

import pytest
import torch

from _release_test_helpers import public_device_scope
from _scheduler_checkpoint_helpers import clone, optimizer, snapshot, update


@pytest.fixture
def lambdalr_device():
    with public_device_scope() as name:
        yield name


def schedule(opt):
    return torch.optim.lr_scheduler.LambdaLR(
        opt, [lambda epoch: 0.5 ** epoch, lambda epoch: 1.0 / (epoch + 1)])


def test_modern_checkpoint_preserves_complete_native_schema(lambdalr_device):
    opt = optimizer(lambdalr_device)
    scheduler = schedule(opt)
    assert scheduler.state_dict() == {
        'base_lrs': [.2, .4], 'last_epoch': 0, '_step_count': 1,
        '_get_lr_called_within_step': False, '_last_lr': [.2, .4],
        'lr_lambdas': [None, None]}
    assert [group['initial_lr'] for group in opt.param_groups] == [.2, .4]


def test_full_eight_updates_equal_fresh_three_plus_five(lambdalr_device):
    opt = optimizer(lambdalr_device)
    scheduler = schedule(opt)
    records = []
    for index in range(8):
        update(opt, scheduler)
        records.append(snapshot(opt, scheduler))
        if index == 2:
            saved_optimizer = clone(opt.state_dict())
            saved_scheduler = copy.deepcopy(scheduler.state_dict())
            saved_parameters = [p.detach().clone() for group in opt.param_groups
                                for p in group['params']]
    fresh = optimizer(lambdalr_device)
    restored = schedule(fresh)
    fresh.load_state_dict(saved_optimizer)
    restored.load_state_dict(saved_scheduler)
    with torch.no_grad():
        for p, saved in zip([p for group in fresh.param_groups for p in group['params']],
                            saved_parameters):
            p.copy_(saved)
    assert snapshot(fresh, restored) == records[2]
    for expected in records[3:]:
        update(fresh, restored)
        assert snapshot(fresh, restored) == expected


class Factor:
    def __init__(self, factor):
        self.factor = factor
        self.calls = 0

    def __call__(self, epoch):
        self.calls += 1
        return self.factor ** epoch


def test_callable_object_state_and_identity_restore(lambdalr_device):
    opt = optimizer(lambdalr_device)
    functions = [Factor(.5), Factor(.75)]
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, functions)
    update(opt, scheduler)
    saved = copy.deepcopy(scheduler.state_dict())
    assert saved['lr_lambdas'] == [dict(f.__dict__) for f in functions]
    functions[0].factor = .25
    functions[1].calls = 99
    scheduler.load_state_dict(saved)
    assert all(actual is expected for actual, expected in zip(scheduler.lr_lambdas, functions))
    assert [f.__dict__ for f in functions] == saved['lr_lambdas']


def test_loading_preserves_input_checkpoint(lambdalr_device):
    opt = optimizer(lambdalr_device)
    scheduler = schedule(opt)
    update(opt, scheduler)
    saved = copy.deepcopy(scheduler.state_dict())
    before = copy.deepcopy(saved)
    fresh = schedule(optimizer(lambdalr_device))
    fresh.load_state_dict(saved)
    assert saved == before
    assert fresh.state_dict() == saved


def test_legacy_additional_fields_remain_loadable(lambdalr_device):
    scheduler = schedule(optimizer(lambdalr_device))
    saved = copy.deepcopy(scheduler.state_dict())
    saved['verbose'] = True
    saved['legacy_marker'] = 17
    fresh = schedule(optimizer(lambdalr_device))
    fresh.load_state_dict(saved)
    assert fresh.state_dict() == saved


def test_loading_without_step_context_keeps_constructor_default(lambdalr_device):
    scheduler = schedule(optimizer(lambdalr_device))
    saved = copy.deepcopy(scheduler.state_dict())
    saved.pop('_get_lr_called_within_step')
    fresh = schedule(optimizer(lambdalr_device))
    fresh.load_state_dict(saved)
    assert fresh.state_dict()['_get_lr_called_within_step'] is False
    assert set(fresh.state_dict()) == set(saved) | {'_get_lr_called_within_step'}

"""Captured Adam counters preserve independent updates and exposed aliases."""

import pickle

import numpy as np
import pytest
import torch
import jittor as jt

from jittor.compat.torch.optimizer_adam import _AdamStep, _AdamSteps
from jittor._runtime import step_capture


def _steps(device, count=3):
    parameters = [torch.nn.Parameter(torch.ones(1, device=device)) for _ in range(count)]
    return _AdamSteps(dict(params=parameters, fused=True), [0] * count)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
@jt.flag_scope(backend_fallback='error')
def test_overlapping_counter_subsets_and_assignments(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is required')
    steps = _steps(device)
    expected = [0, 0, 0]
    for index, active in enumerate(([0, 1, 2], [0, 1], [2], [0, 1, 2], [1, 2], [0]) * 4):
        steps._advance_capture_steps(active)
        for slot in active:
            expected[slot] += 1
        assert [int(value) for value in steps] == expected
        assert steps._capture_steps_uniform([0, 1, 2]) == (len(set(expected)) == 1)
        if index == 9:
            steps[1] = 99
            expected[1] = 99


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
@jt.flag_scope(backend_fallback='error')
def test_exposed_counter_storage_remains_authoritative(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is required')
    steps = _steps(device)
    for _ in range(8):
        steps._advance_capture_steps([0, 1, 2])
    observed = steps[1].tensor()
    assert observed.device.type == device and observed.item() == 8
    observed.data.fill_(99)
    steps._advance_capture_steps([0, 1, 2])
    assert [int(value) for value in steps] == [9, 100, 9]
    assert steps[1].tensor() is observed


def test_unversioned_numpy_counter_alias_and_duplicate_indices():
    steps = _steps('cpu')
    for _ in range(3):
        steps._advance_capture_steps([0, 1, 2])
    array = np.array(3., dtype='float32')
    steps[1] = torch.from_numpy(array)
    array[...] = 13
    steps._advance_capture_steps([0, 1, 2])
    assert [int(value) for value in steps] == [4, 14, 4]
    steps._advance_capture_steps([0, 0, 2])
    assert [int(value) for value in steps] == [6, 14, 5]


def test_aliased_private_slots_are_advanced_separately():
    shared = _AdamStep(1)
    steps = _AdamSteps(dict(params=[], fused=False), [shared, shared])
    steps._advance_capture_steps([0, 1])
    assert int(shared) == 3


def test_counter_serialization_materializes_shared_increments():
    steps = _AdamSteps(dict(params=[], fused=False), [_AdamStep(1), _AdamStep(1)])
    for _ in range(9):
        steps._advance_capture_steps([0, 1])
    restored = pickle.loads(pickle.dumps(steps))
    assert [int(value) for value in restored] == [10, 10]
    restored._advance_capture_steps([0, 1])
    assert [int(value) for value in restored] == [11, 11]
    legacy = _AdamStep.__new__(_AdamStep)
    legacy.__setstate__(dict(value=7, counter=None, device=None, dtype=None))
    assert int(legacy) == 7


@pytest.mark.cuda
@pytest.mark.parametrize('kind', ['Adam', 'AdamW'])
@pytest.mark.parametrize('fused', [None, True])
@jt.flag_scope(backend_fallback='error')
def test_unequal_exposed_steps_do_not_share_bias_correction(kind, fused):
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required')

    def trajectory(captured):
        p = torch.nn.Parameter(torch.tensor([.2, -.3], device='cuda'))
        q = torch.nn.Parameter(torch.tensor([.4, .1], device='cuda'))
        options = dict(lr=.003, fused=fused)
        if kind == 'Adam':
            options['weight_decay'] = 0.
        optimizer = getattr(torch.optim, kind)([p, q], **options)

        def step():
            loss = (p * 2 + q).square().mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return loss.detach()

        runner = torch.compile(step, mode='reduce-overhead') if captured else step
        try:
            for _ in range(8):
                runner().item()
            if captured:
                assert runner.stats['graph'] > 0
            optimizer.state[p]['step'].data.fill_(50)
            runner().item()
            assert [int(optimizer.state[value]['step'].item()) for value in (p, q)] == [51, 9]
            return [value.detach().cpu().numpy().copy() for value in (p, q)]
        finally:
            if captured:
                runner.invalidate()

    for actual, expected in zip(trajectory(True), trajectory(False)):
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


@pytest.mark.cuda
@jt.flag_scope(backend_fallback='error')
def test_alternating_parameter_subsets_keep_independent_bias_correction():
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required')

    def trajectory(captured):
        p = torch.nn.Parameter(torch.tensor([.2, -.3], device='cuda'))
        q = torch.nn.Parameter(torch.tensor([.4, .1], device='cuda'))
        optimizer = torch.optim.AdamW([p, q], lr=.003)

        def both():
            loss = (p * 2 + q).square().mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return loss.detach()

        def only_p():
            loss = (p * .5).square().mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return loss.detach()

        runners = [torch.compile(fn, mode='reduce-overhead') for fn in (both, only_p)]
        try:
            for _ in range(8):
                (runners[0] if captured else both)().item()
            for _ in range(2):
                (runners[1] if captured else only_p)().item()
            (runners[0] if captured else both)().item()
            assert [int(optimizer.state[value]['step'].item()) for value in (p, q)] == [11, 9]
            return [value.detach().cpu().numpy().copy() for value in (p, q)]
        finally:
            for runner in runners:
                runner.invalidate()

    for actual, expected in zip(trajectory(True), trajectory(False)):
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


@pytest.mark.cuda
@jt.flag_scope(backend_fallback='error')
def test_refusal_after_executed_adam_takes_each_update_once():
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required')

    def trajectory(captured):
        p = torch.nn.Parameter(torch.tensor([.2, -.3], device='cuda'))
        optimizer = torch.optim.AdamW([p], lr=.003)

        def step():
            loss = p.square().mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step_capture.refuse('the test refuses after the update')
            return loss.detach()

        runner = torch.compile(step, mode='reduce-overhead') if captured else step
        try:
            losses = [runner().item() for _ in range(3)]
            return losses, p.detach().cpu().numpy().copy()
        finally:
            if captured:
                runner.invalidate()

    for actual, expected in zip(trajectory(True), trajectory(False)):
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)

"""Fail-closed full PPO LoRA observation inventories, without runtime imports."""
import copy
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def probe(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / 'agent/skills/torch-compat-repo-runbook/scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('ppo_lora_probe_test', scripts/'trl_ppo_lora_probe.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def complete_manifest():
    parameters = ['adapter_%d' % i for i in range(224)] + ['value_model.score.weight']
    arrays = {}
    for step in range(10):
        arrays['step/%d/loss/0' % step] = {}
        for name in parameters:
            for kind in ('gradient', 'updated_parameter', 'adam_step', 'exp_avg', 'exp_avg_sq'):
                arrays['step/%d/%s/%s/0' % (step, kind, name)] = {}
    for index in range(10):
        arrays['step/10/final_eval/%d/next_logits/0' % index] = {}
    return dict(protocol={"steps": 10}, optimizer_steps=10, trainable_parameters=dict.fromkeys(parameters, [1]),
                arrays=arrays, generation_calls=[{'draws': [{}, {}, {}, {}]} for _ in range(5)])


def test_complete_all_step_inventory_is_admitted(probe):
    probe.validate_manifest(complete_manifest())


@pytest.mark.parametrize('missing', ['loss', 'gradient', 'updated_parameter', 'adam_step', 'exp_avg', 'exp_avg_sq', 'next_logits'])
def test_mutually_missing_evidence_cannot_produce_false_success(probe, missing):
    data = complete_manifest()
    key = next(k for k in data['arrays'] if '/'+missing+'/' in k)
    del data['arrays'][key]
    with pytest.raises(AssertionError):
        probe.validate_manifest(data)


@pytest.mark.parametrize('kind', ['calls', 'parameter', 'rollout', 'draw'])
def test_skip_or_truncated_rollout_is_rejected(probe, kind):
    data = copy.deepcopy(complete_manifest())
    if kind == 'calls':
        data['optimizer_steps'] = 9
    elif kind == 'parameter':
        data['trainable_parameters'].popitem()
    elif kind == 'rollout':
        data['generation_calls'].pop()
    else:
        data['generation_calls'][0]['draws'].pop()
    with pytest.raises(AssertionError):
        probe.validate_manifest(data)

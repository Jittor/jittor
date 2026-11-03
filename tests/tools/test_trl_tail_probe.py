"""Offline controls for the five additional official text-Trainer probes."""
import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.fixture
def probe(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / 'agent/skills/torch-compat-repo-runbook/scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('tail_probe_test', scripts / 'trl_tail_probe.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scope_excludes_diffusion_and_existing_subset(probe, tmp_path, monkeypatch):
    base = ['probe', '--root', str(tmp_path/'root'), '--upstream', str(tmp_path/'upstream'),
            '--output', str(tmp_path/'trace'), '--runtime', 'torch', '--device', 'cpu']
    for method in ('ddpo', 'alignprop', 'sft'):
        monkeypatch.setattr(sys, 'argv', base + ['--algorithm', method])
        with pytest.raises(SystemExit):
            probe.options()
    for method in probe.ALGORITHMS:
        monkeypatch.setattr(sys, 'argv', base + ['--algorithm', method])
        assert probe.options().steps == 10


def test_real_process_labels_and_balanced_binary_preferences(probe):
    import trl_tail_cases as cases
    prm = cases.records('prm')
    assert len(prm) == 10
    assert all(len(row['completions']) == len(row['labels']) == 2 for row in prm)
    assert all(set(row['labels']) == {True, False} for row in prm)
    bco = cases.records('bco')
    assert sum(row['label'] for row in bco) == 5
    assert all(bco[i]['label'] != bco[i+1]['label'] for i in range(0, 10, 2))
    for method in cases.ALGORITHMS:
        rows = cases.records(method)
        assert len({row.get('prompt', row.get('text')) for row in rows}) == 10


def test_token_classifier_head_perturbation_rejected(probe, tmp_path):
    artifact = tmp_path/'policy.npz'
    initial = {'parameter/score.weight': np.arange(64, dtype=np.float32).reshape(2, 32)}
    np.savez(artifact, **initial)
    original = probe.digest(artifact)
    restored = probe.read_initial_state(artifact, original)
    np.testing.assert_array_equal(restored['parameter/score.weight'], initial['parameter/score.weight'])
    initial['parameter/score.weight'][1, 4] += .125
    np.savez(artifact, **initial)
    with pytest.raises(AssertionError, match='artifact changed'):
        probe.read_initial_state(artifact, original)


@pytest.mark.parametrize('field,value', [('role', 'policy'), ('inputs', {'input_ids': [[4, 6]]}),
                                       ('config', {'mixture_coef': .75})])
def test_geometric_mixture_replay_context_mismatch_rejected(probe, field, value):
    expected = dict(role='geometric_mixture', inputs={'input_ids': [[4, 5]]}, config={'mixture_coef': .5})
    probe.validate_generation_context(copy.deepcopy(expected), expected)
    changed = copy.deepcopy(expected)
    changed[field] = value
    with pytest.raises(AssertionError, match='Generation prompt/configuration'):
        probe.validate_generation_context(changed, expected)


def test_geometric_mixture_replay_token_tampering_rejected(probe):
    draw = dict(distribution_shape=[1, 256], shape=[1, 1], tokens=[[7]])
    draw['sha256'] = probe.replay_hash(draw)
    probe.check_replay_draw(draw)
    draw['tokens'][0][0] = 8
    with pytest.raises(AssertionError, match='replay hash mismatch'):
        probe.check_replay_draw(draw)

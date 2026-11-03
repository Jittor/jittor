"""Pure failure guards for localizing actual DDP repeat variation."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def probe(monkeypatch):
    scripts = Path(__file__).resolve().parents[2]/'agent/skills/torch-compat-repo-runbook/scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('trl_ddp_local_probe_tests', scripts/'trl_ddp_local_probe.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rank(probe, tmp_path):
    import numpy as np
    trace = tmp_path/'rank0'
    (trace/'arrays').mkdir(parents=True)
    manifest = dict(status='complete', world_size=2, backend='nccl', ddp_wrapped=True, fallback_count=0,
                    warm_only=False, runtime='jittor', initial={}, optimizer_steps=list(range(1, 11)), arrays={},
                    protocol=dict(precision='fp32', lora=dict(dropout=0.)),
                    trainable_parameters={'layer.%d.lora_B.weight' % i: [2] for i in range(112)})
    values = np.array([[2, 4, 5]], dtype='int64')
    np.save(trace/'arrays/input.npy', values)
    for name in probe.INPUTS:
        manifest['arrays']['step/1/input/'+name] = dict(path='arrays/input.npy', shape=[1, 3], dtype='int64',
            exact=True, sha256=probe.array_hash(values))
    probe.write_json(trace/'manifest.json', manifest)
    return trace


def mutate(probe, path, action):
    value = json.loads((path/'manifest.json').read_text())
    action(value)
    probe.write_json(path/'manifest.json', value)


def test_reads_actual_rank_inputs(probe, rank):
    manifest, batch, parameters = probe.read_rank(rank, 1, 0)
    assert manifest['world_size'] == 2 and set(batch) == probe.INPUTS and not parameters


@pytest.mark.parametrize('field,value', [('world_size', 1), ('fallback_count', 1), ('warm_only', True), ('ddp_wrapped', False)])
def test_refuses_nonactual_ddp_trace(probe, rank, field, value):
    mutate(probe, rank, lambda m: m.update({field: value}))
    with pytest.raises(ValueError, match='actual complete'):
        probe.read_rank(rank, 1, 0)


def test_missing_actual_input_fails(probe, rank):
    mutate(probe, rank, lambda m: m['arrays'].pop('step/1/input/labels'))
    with pytest.raises(ValueError, match='input inventory'):
        probe.read_rank(rank, 1, 0)


def test_requires_preceding_parameter_state(probe, rank):
    with pytest.raises(ValueError, match='precede'):
        probe.read_rank(rank, 1, 1)


def test_corrupt_retained_input_fails(probe, rank):
    import numpy as np
    np.save(rank/'arrays/input.npy', np.array([[2, 4, 6]], dtype='int64'))
    with pytest.raises(AssertionError, match='hash mismatch'):
        probe.read_rank(rank, 1, 0)


@pytest.fixture
def preceding_state(probe, rank, tmp_path):
    import numpy as np
    import shutil
    manifest = json.loads((rank/'manifest.json').read_text())
    for name in probe.INPUTS:
        manifest['arrays']['step/2/input/'+name] = manifest['arrays']['step/1/input/'+name].copy()
    values = np.array([0.25, -0.5], dtype='float32')
    np.save(rank/'arrays/parameters.npy', values)
    for name in manifest['trainable_parameters']:
        manifest['arrays']['step/1/parameter/'+name] = dict(
            path='arrays/parameters.npy', shape=[2], dtype='float32', exact=True,
            sha256=probe.array_hash(values))
    probe.write_json(rank/'manifest.json', manifest)
    other = tmp_path/'retained_parameters'
    shutil.copytree(rank, other)
    return rank, other


def test_reads_matching_preceding_rank_state(probe, preceding_state):
    rank, other = preceding_state
    _, _, parameters = probe.read_rank(rank, 2, 1, other)
    assert len(parameters) == 112


def test_other_valid_state_must_match_selected_rank_sha(probe, preceding_state):
    import numpy as np
    rank, other = preceding_state
    values = np.array([0.5, -0.5], dtype='float32')
    np.save(other/'arrays/parameters.npy', values)
    def replace_hashes(manifest):
        for key, info in manifest['arrays'].items():
            if key.startswith('step/1/parameter/'):
                info['sha256'] = probe.array_hash(values)
    mutate(probe, other, replace_hashes)
    with pytest.raises(ValueError, match='differs from selected rank'):
        probe.read_rank(rank, 2, 1, other)


def test_missing_selected_rank_state_metadata_fails(probe, preceding_state):
    rank, other = preceding_state
    mutate(probe, rank, lambda m: m['arrays'].pop('step/1/parameter/layer.0.lora_B.weight'))
    with pytest.raises(ValueError, match='selected-rank preceding adapter metadata'):
        probe.read_rank(rank, 2, 1, other)


def test_corrupt_preceding_payload_fails(probe, preceding_state):
    import numpy as np
    rank, other = preceding_state
    np.save(other/'arrays/parameters.npy', np.array([1., 2.], dtype='float32'))
    with pytest.raises(AssertionError, match='hash mismatch'):
        probe.read_rank(rank, 2, 1, other)


def local_result(probe):
    names = {'layer.%d.lora_B.weight' % i: [2] for i in range(112)}
    fields = {'loss', 'logits'} | {'gradient/'+name for name in names}
    return dict(status='complete', fallback_count=0, collective_calls=0, optimizer_updates=0, runtime='jittor',
                mode='local', protocol=dict(repetitions=2, boundaries=False), trainable_parameters=names,
                arrays={'repeat/%d/%s' % (i, name): {} for i in range(2) for name in fields},
                repetitions=[dict(index=0, comparison={}), dict(index=1, comparison={name: {} for name in fields})])


def test_complete_local_diagnostic_inventory(probe):
    probe.validate_result(local_result(probe))


@pytest.mark.parametrize('mutation', ['gradient', 'repeat', 'collective', 'update', 'trainables'])
def test_incomplete_local_diagnostic_fails(probe, mutation):
    result = local_result(probe)
    if mutation == 'gradient':
        result['arrays'].pop('repeat/0/gradient/layer.0.lora_B.weight')
    elif mutation == 'repeat':
        result['repetitions'].pop()
    elif mutation == 'collective':
        result['collective_calls'] = 1
    elif mutation == 'update':
        result['optimizer_updates'] = 1
    else:
        result['trainable_parameters'].popitem()
    with pytest.raises(ValueError):
        probe.validate_result(result)


def test_ce_requires_actual_saved_lse(probe):
    result = local_result(probe)
    result.update(mode='ce', arrays={'repeat/%d/%s' % (i, name): {} for i in range(2) for name in ('loss', 'gradient')})
    result['repetitions'][1]['comparison'] = {'loss': {}, 'gradient': {}}
    with pytest.raises(ValueError, match='observation inventory'):
        probe.validate_result(result)


def test_observer_inventory_cannot_silently_shrink(probe):
    result = local_result(probe)
    result['protocol'].update(boundaries=True, observed_modules=[])
    with pytest.raises(ValueError, match='observer inventory'):
        probe.validate_result(result)

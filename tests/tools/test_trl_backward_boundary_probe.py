"""Pure guards for complete, hash-bound backward captures and local VJPs."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def probe(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / 'agent/skills/torch-compat-repo-runbook/scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('trl_backward_boundary_probe_tests', scripts / 'trl_backward_boundary_probe.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def captured(probe):
    from trl_practical_compare import expected_adapters
    arrays, modules = {}, {}
    frozen = {'base_model.model.model.layers.0.input_layernorm',
              'base_model.model.model.layers.0.self_attn.k_norm'}
    for name, kind in probe.expected_modules().items():
        modules[name] = {'kind': kind}
        for suffix in ('x', 'weight', 'y'):
            arrays[name + '/' + suffix] = {'requires_grad': name not in frozen}
        if name not in frozen:
            arrays[name + '/dy_scaled'] = {}
    for name, shape in expected_adapters().items():
        arrays['parameter_gradient/' + name] = {'shape': shape}
    return dict(status='first_backward_observed_before_update', precision='fp16', backward_loss_scale=1024.,
                all112_gradients_match_formal_oracle=True, modules=modules, arrays=arrays,
                gradient_reference={name: {'bitwise_equal': True} for name in expected_adapters()})


def result_for(probe, captured):
    cases, arrays = {}, {}
    for name, info in probe.selected_modules(captured, ['lora_B', 'rms', 'lm_head']):
        for route in (('functional', 'module') if info['kind'] in ('lora_B', 'lm_head') else ('',)):
            prefix = name + '/' + route if route else name
            cases[prefix] = {'kind': info['kind']}
            for suffix in ('y', 'dx_scaled') + (('dw_unscaled',) if info['kind'] == 'lora_B' else ()):
                arrays[prefix + '/' + suffix] = {}
    return dict(cases=cases, arrays=arrays)


def test_complete_inventory_and_legitimate_frozen_outputs(probe, captured):
    selected = probe.selected_modules(captured, ['lora_B', 'rms', 'lm_head'])
    assert len(selected) == 168
    result = result_for(probe, captured)
    assert len(result['cases']) == 225 and len(result['arrays']) == 562
    probe.validate_replay_result(captured, ['lora_B', 'rms', 'lm_head'], result)


@pytest.mark.parametrize('kind,count', [('lora_B', 56), ('rms', 111), ('lm_head', 1)])
def test_selected_kinds_preserve_expected_active_count(probe, captured, kind, count):
    assert len(probe.selected_modules(captured, [kind])) == count


@pytest.mark.parametrize('mutation,match', [
    (lambda m: m.update(modules={}), 'module inventory'),
    (lambda m: m['modules'].pop('base_model.model.lm_head'), 'module inventory'),
    (lambda m: m['arrays'].pop('base_model.model.lm_head/dy_scaled'), 'active module cotangent'),
    (lambda m: m['arrays'].pop('base_model.model.lm_head/y'), 'output inventory'),
    (lambda m: m['arrays'].pop('base_model.model.lm_head/weight'), 'output inventory'),
    (lambda m: m['arrays']['base_model.model.lm_head/y'].pop('requires_grad'), 'requires_grad'),
    (lambda m: m['arrays']['base_model.model.lm_head/y'].update(requires_grad=False), 'inactive module cotangent'),
    (lambda m: m['gradient_reference'].pop(next(iter(m['gradient_reference']))), 'formal adapter gradient inventory'),
    (lambda m: m['arrays'].pop(next(k for k in m['arrays'] if k.startswith('parameter_gradient/'))), 'formal adapter gradient inventory'),
    (lambda m: m['arrays'][next(k for k in m['arrays'] if k.startswith('parameter_gradient/'))].update(shape=[1]), 'gradient shape'),
])
def test_refuses_truncated_or_falsely_active_capture(probe, captured, mutation, match):
    mutation(captured)
    with pytest.raises(AssertionError, match=match):
        probe.selected_modules(captured, ['lora_B', 'rms', 'lm_head'])


@pytest.mark.parametrize('kinds', [[], ['rms', 'rms'], ['unknown']])
def test_refuses_empty_or_invalid_selection(probe, captured, kinds):
    with pytest.raises(AssertionError, match='selected module kinds'):
        probe.selected_modules(captured, kinds)


@pytest.mark.parametrize('part', ['cases', 'arrays'])
def test_refuses_empty_replay_completion(probe, captured, part):
    result = result_for(probe, captured)
    result[part] = {}
    with pytest.raises(AssertionError, match='replay case/output/gradient inventory'):
        probe.validate_replay_result(captured, ['lora_B', 'rms', 'lm_head'], result)


@pytest.mark.parametrize('suffix', ['/y', '/dx_scaled', '/dw_unscaled'])
def test_refuses_missing_replay_output_or_gradient(probe, captured, suffix):
    result = result_for(probe, captured)
    result['arrays'].pop(next(k for k in result['arrays'] if k.endswith(suffix)))
    with pytest.raises(AssertionError, match='replay case/output/gradient inventory'):
        probe.validate_replay_result(captured, ['lora_B', 'rms', 'lm_head'], result)


@pytest.fixture
def artifact(probe, tmp_path):
    import numpy as np
    value = np.array([1., -2.], dtype='float32')
    np.save(tmp_path / 'value.npy', value, allow_pickle=False)
    return tmp_path, dict(path='value.npy', shape=[2], dtype='torch.float32', array_dtype='float32',
                         sha256=probe.array_hash(value), file_sha256=probe.digest(tmp_path / 'value.npy'))


def test_checks_float32_storage_for_bfloat16_capture(probe, artifact):
    path, info = artifact
    info['dtype'] = 'torch.bfloat16'
    assert probe.checked_capture(path, info).tolist() == [1., -2.]


@pytest.mark.parametrize('changes,match', [
    ({'shape': [1]}, 'shape/dtype mismatch'),
    ({'array_dtype': 'float16'}, 'logical/storage dtype'),
    ({'dtype': 'torch.float16', 'array_dtype': 'float16'}, 'shape/dtype mismatch'),
    ({'dtype': 'torch.unknown'}, 'logical/storage dtype'),
    ({'sha256': '0' * 64}, 'artifact hash mismatch'),
    ({'file_sha256': '0' * 64}, 'file hash mismatch'),
    ({'path': '../outside.npy'}, 'escapes'),
])
def test_refuses_corrupt_capture_array(probe, artifact, changes, match):
    path, info = artifact
    info.update(changes)
    with pytest.raises(AssertionError, match=match):
        probe.checked_capture(path, info)


def test_refuses_modified_file_even_with_new_payload_hash(probe, artifact):
    import numpy as np
    path, info = artifact
    value = np.array([3., -4.], dtype='float32')
    np.save(path / info['path'], value, allow_pickle=False)
    info['sha256'] = probe.array_hash(value)
    with pytest.raises(AssertionError, match='file hash mismatch'):
        probe.checked_capture(path, info)


@pytest.fixture
def locations(tmp_path):
    checkout, source = tmp_path / 'checkout', tmp_path / 'capture'
    checkout.mkdir()
    source.mkdir()
    return SimpleNamespace(root=checkout, source=source, output=tmp_path / 'output')


def test_resolves_paths_without_creating_output(probe, locations):
    probe.preflight_paths(locations)
    assert locations.root.is_absolute() and not locations.output.exists()


@pytest.mark.parametrize('target', ['checkout', 'source', 'nested_source', 'parent'])
def test_refuses_output_checkout_or_input_collision(probe, locations, target):
    locations.output = {'checkout': locations.root / 'output', 'source': locations.source,
                        'nested_source': locations.source / 'output', 'parent': locations.source.parent}[target]
    with pytest.raises(AssertionError, match='outside|collides'):
        probe.preflight_paths(locations)


def test_refuses_output_alias_through_symlink(probe, locations, tmp_path):
    alias = tmp_path / 'checkout_alias'
    alias.symlink_to(locations.root, target_is_directory=True)
    locations.output = alias / 'output'
    with pytest.raises(AssertionError, match='outside'):
        probe.preflight_paths(locations)


def test_refuses_existing_output(probe, locations):
    locations.output.mkdir()
    with pytest.raises(AssertionError, match='already exists'):
        probe.preflight_paths(locations)

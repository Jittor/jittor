"""Fail closed on lost gradients, changed provenance, device and precision."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pytest

from _helpers.vllm_installation import installation as installed_fixture
from _helpers.vllm_installation import contract as installation_contract
from _helpers.vllm_installation import adapter_binding

SCRIPTS = Path(__file__).resolve().parents[2] / 'agent/skills/torch-compat-repo-runbook/scripts'
@pytest.fixture
def installation(tmp_path):
    with installed_fixture(tmp_path) as value:
        yield value
sys.path.insert(0, str(SCRIPTS))
import _vllm_component_contract as contract  # noqa: E402
from component_measure import package_files  # noqa: E402


@pytest.fixture
def pair(request, tmp_path, monkeypatch):
    source, module, wheel, binding, _, _, _ = request.getfixturevalue('installation')
    binding = installation_contract.bind_installation(source, module, wheel, binding)
    monkeypatch.setattr(contract, 'WHEEL_SHA256', binding['wheel_sha256'])
    root = tmp_path / 'framework'
    root.mkdir()
    adapter = adapter_binding(root)
    captures = {}
    for name in ('native.cc', 'runner.py'):
        (root / name).write_text(name)
        captures[name] = hashlib.sha256(name.encode()).hexdigest()
    monkeypatch.setattr(contract, 'framework_sources', lambda _root: sorted(captures))
    transformer = tmp_path / 'transformers'
    transformer.mkdir()
    (transformer / '__init__.py').write_text('__version__ = "5.13.0"\n')
    dependencies = {'vllm': {'version': '0.29.0', 'origin': module.__file__,
                             'installed_files_sha256': binding['installed_files_sha256']},
                    'transformers': {'version': '5.13.0', 'origin': str(transformer / '__init__.py'),
                                     'installed_files_sha256': package_files(transformer)}}
    weights = tmp_path / 'weights.npz'
    np.savez(weights, **{'rotary.cos_sin_cache': np.ones((32, 8), dtype='float32')})
    binary = tmp_path / '_C.so'
    binary.write_bytes(b'independent test binary fixture')
    precision = {'matmul': False, 'cudnn': False, 'cudnn_benchmark': False, 'matmul_precision': 'highest'}
    values, paths = [], []
    arrays = {name: np.ones(shape, dtype='float32') for name, shape in contract.FIELDS.items()}
    for i, runtime in enumerate(('torch', 'jittor')):
        path = tmp_path / (runtime + '.npz')
        np.savez(path, **arrays)
        value = {'case': contract.CASE, 'device': 'cpu', 'seed': 0, 'jittor_commit': 'baseline',
                 'device_controls': {'cuda_visible_devices': '', 'cuda_available': False},
                 'upstream': source['upstream'], 'hf_hub_offline': True, 'seconds': .1,
                 'interpreter': {'is_shim': runtime == 'jittor', 'pid': i + 1, 'executable': str(tmp_path / runtime),
                                 'binary_extension': str(binary), 'binary_sha256': contract.digest(binary),
                                 'torch_version': '2.13.0+cu130'},
                 'downstream_distribution': binding, 'dependencies': dependencies,
                 'measurement_source_sha256': captures, 'weights': str(weights), 'weights_sha256': contract.digest(weights),
                 'tf32': precision, 'runtime_conditions': {'precision': precision, 'affinity': [0, 1], 'runtime_threads': 1,
                     'thread_env': {name: '1' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}},
                 'tensor_observations': [{'shape': list(shape), 'device': 'cpu', 'dtype': 'float32',
                                         'location': 'cpu', 'dispatch': 'cpu'} for shape in contract.FIELDS.values()],
                 'fallback_count': 0, 'fallback_policy': 'error', 'native_runtime_flags': {'use_cuda': 0}}
        if runtime == 'jittor':
            value['adapter_distribution'] = adapter
        values.append(copy.deepcopy(value))
        paths.append(path)
    def compare():
        for path, value in zip(paths, values):
            path.with_suffix('.json').write_text(json.dumps(value))
        return contract.compare_pair(*paths, source, 'baseline', 'cpu', 0, wheel, root)
    return compare, values, paths, arrays


def test_complete_pair_retains_all_four_gradients(pair):
    compare, _, _, _ = pair
    result = compare()
    assert len(result['rows']) == 5
    assert result['input_gradients'] == {'requested': 4, 'checked': 4}
    assert result['parameter_gradients'] == {'requested': 0, 'checked': 0}


@pytest.mark.parametrize('change', ['missing', 'bootstrap', 'metadata', 'record', 'entry_points',
                                    'direct_url', 'version', 'registration', 'source_root'])
def test_actual_maintained_adapter_identity_is_required(pair, change):
    compare, values, _, _ = pair
    value = values[1]
    binding = value['adapter_distribution']
    if change == 'missing':
        del value['adapter_distribution']
    elif change == 'bootstrap':
        binding['bootstrap'] = '/foreign/bootstrap.py'
    elif change in ('metadata', 'record', 'entry_points', 'direct_url'):
        binding[change + '_sha256'] = '0' * 64
    else:
        key = {'version': 'metadata', 'registration': 'entry_points', 'source_root': 'direct_url'}[change]
        path = Path(binding[key])
        text = path.read_text()
        text = (text.replace('2.0.0', '9.0.0') if change == 'version' else
                text.replace('jittor_vllm', 'other') if change == 'registration' else
                text.replace('adapters', 'foreign'))
        path.write_text(text)
        binding[key + '_sha256'] = contract.digest(path)
    with pytest.raises(ValueError, match='adapter|entry point'):
        compare()


@pytest.mark.parametrize('change', ['gradient', 'shape', 'dtype', 'nonfinite', 'extra', 'budget'])
def test_missing_or_invalid_numeric_evidence_is_rejected(pair, change):
    compare, _, paths, arrays = pair
    changed = {name: value.copy() for name, value in arrays.items()}
    if change == 'gradient':
        del changed['ingrad::key']
    elif change == 'shape':
        changed['ingrad::query'] = changed['ingrad::query'].reshape(-1)
    elif change == 'dtype':
        changed['ingrad::key'] = changed['ingrad::key'].astype('float64')
    elif change == 'nonfinite':
        changed['ingrad::key'][0, 0] = np.nan
    elif change == 'extra':
        changed['grad::invented'] = np.ones(1, dtype='float32')
    else:
        changed['ingrad::key'][0, 0] += .02
    np.savez(paths[1], **changed)
    with pytest.raises(ValueError):
        compare()


@pytest.mark.parametrize('change', ['source', 'incomplete_source', 'device', 'physical', 'observation', 'dtype',
                                    'fallback', 'shim', 'binary', 'installed', 'record', 'dependency', 'precision',
                                    'threads', 'timing', 'seed', 'weights', 'pid', 'cpu_cuda_visible'])
def test_provenance_runtime_and_scope_changes_are_rejected(pair, change):
    compare, values, _, _ = pair
    value = values[1]
    if change == 'source':
        value['measurement_source_sha256']['native.cc'] = '0' * 64
    elif change == 'incomplete_source':
        del value['measurement_source_sha256']['native.cc']
    elif change == 'device':
        value['device'] = 'cuda'
    elif change == 'physical':
        value['tensor_observations'][0]['dispatch'] = 'cuda'
    elif change == 'observation':
        value['tensor_observations'].pop()
    elif change == 'dtype':
        value['tensor_observations'][0]['dtype'] = 'float64'
    elif change == 'fallback':
        value['fallback_count'] = 1
    elif change == 'shim':
        value['interpreter']['is_shim'] = False
    elif change == 'binary':
        values[0]['interpreter']['binary_sha256'] = '0' * 64
    elif change == 'installed':
        del value['downstream_distribution']['installed_files_sha256']['kernels.bin']
    elif change == 'record':
        value['downstream_distribution']['record_sha256'] = '0' * 64
    elif change == 'dependency':
        value['dependencies']['transformers']['version'] = '5.12.0'
    elif change == 'precision':
        for packet in values:
            packet['tf32']['matmul'] = True
    elif change == 'threads':
        for packet in values:
            packet['runtime_conditions']['runtime_threads'] = 2
    elif change == 'timing':
        value['seconds'] = 0
    elif change == 'seed':
        value['seed'] = 7
    elif change == 'weights':
        value['weights_sha256'] = '0' * 64
    elif change == 'cpu_cuda_visible':
        value['device_controls']['cuda_visible_devices'] = '0'
    else:
        value['interpreter']['pid'] = values[0]['interpreter']['pid']
    with pytest.raises(ValueError):
        compare()


def test_input_perturbation_requires_real_output_and_gradient_change(tmp_path):
    a, b = tmp_path / 'base.npz', tmp_path / 'changed.npz'
    arrays = {name: np.ones(shape, dtype='float32') for name, shape in contract.FIELDS.items()}
    np.savez(a, **arrays)
    changed = {name: value.copy() for name, value in arrays.items()}
    changed['__output__'] += .1
    np.savez(b, **changed)
    with pytest.raises(ValueError, match='gradients'):
        contract.validate_perturbation(a, b)
    changed['ingrad::key'] += .1
    np.savez(b, **changed)
    assert contract.validate_perturbation(a, b) == ['ingrad::key']


@pytest.mark.parametrize('error,passes', [(9e-11, True), (2e-10, False)])
def test_zero_reference_gradient_uses_original_per_field_floor(tmp_path, error, passes):
    a, b = tmp_path / 'reference.npz', tmp_path / 'candidate.npz'
    expected = {name: np.zeros(shape, dtype='float32') for name, shape in contract.FIELDS.items()}
    expected['__output__'].fill(1e6)
    actual = {name: value.copy() for name, value in expected.items()}
    actual['ingrad::key'][0, 0] = error
    np.savez(a, **expected)
    np.savez(b, **actual)
    if passes:
        contract.compare_arrays(a, b)
    else:
        with pytest.raises(ValueError, match='budget'):
            contract.compare_arrays(a, b)


@pytest.mark.parametrize('failed_seed', [None, 0, 19])
def test_worker_exit_success_cannot_hide_failed_arrays_or_create_completion(tmp_path, monkeypatch, failed_seed):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / 'compat/tests/torch'))
    import _ecosystem_vllm_installed_harness as harness
    import registry_inventory

    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    retained = tmp_path / 'retained'
    monkeypatch.setattr(harness, 'ROOT', checkout)
    monkeypatch.setattr(harness.subprocess, 'check_output', lambda *a, **k: 'baseline\n')
    monkeypatch.setattr(registry_inventory, 'inventory', lambda *a: {})
    monkeypatch.setenv('JITTOR_VLLM_EVIDENCE_DIR', str(retained))
    calls = []

    def worker(python, case, command):
        path = Path(command[command.index('--') + 3])
        seed = int(command[command.index('--seed') + 1])
        runtime = command[command.index('--runtime') + 1]
        arrays = {name: np.full(shape, 1. + seed * .01, dtype='float32') for name, shape in contract.FIELDS.items()}
        if seed == failed_seed and runtime == 'jittor':
            arrays['ingrad::key'] += 1
        np.savez(path, **arrays)
        path.with_suffix('.json').write_text('{"scope": "offline transport fixture"}')
        calls.append((seed, runtime))
        return {'case': case}, 'worker exit zero'

    def comparison(a, b, *args):
        rows = contract.compare_arrays(a, b)
        return {'rows': rows, 'reference_result': {'weights_sha256': 'fixed'},
                'candidate_result': {'weights_sha256': 'fixed'}}

    monkeypatch.setattr(harness, '_run_command', worker)
    monkeypatch.setattr(contract, 'compare_pair', comparison)
    test = unittest.TestCase()
    test.device = 'cpu'
    output = retained / 'cpu' / contract.CASE
    if failed_seed is None:
        harness.compare_primitives(test, checkout, tmp_path / 'wheel')
        result = json.loads((output / 'complete-comparison.json').read_text())
        assert result['status'] == 'passed' and result['checked_tensors'] == 15
        assert calls == [(0, 'torch'), (0, 'jittor'), (7, 'torch'), (7, 'jittor'), (19, 'torch'), (19, 'jittor')]
    else:
        with pytest.raises(ValueError, match='budget'):
            harness.compare_primitives(test, checkout, tmp_path / 'wheel')
        assert (output / ('seed%d-jittor.npz' % failed_seed)).is_file()
        assert not (output / 'complete-comparison.json').exists()


@pytest.mark.parametrize('change', [None, 'extra_phase_precision', 'reused_pid', 'extra_phase_missing_gradient'])
def test_collector_revalidates_all_six_workers(pair, request, tmp_path, change):
    import collect_vllm_scorecard as collector

    _, values, _, arrays = pair
    source, _, wheel, _, _, _, _ = request.getfixturevalue('installation')
    source['axes'] = [{'axis': 'custom_op', 'members': ['silu_and_mul', 'rms_norm', 'rotary_embedding']}]
    folder = tmp_path / 'retained' / contract.CASE
    folder.mkdir(parents=True)
    for phase, seed in enumerate((0, 7, 19)):
        for side, runtime in enumerate(('torch', 'jittor')):
            value = copy.deepcopy(values[side])
            value.update(seed=seed)
            value['interpreter']['pid'] = 10 + phase * 2 + side
            if seed == 19 and runtime == 'jittor':
                if change == 'extra_phase_precision':
                    value['tf32']['matmul'] = True
                elif change == 'reused_pid':
                    value['interpreter']['pid'] = 10
            raw = {name: data + seed * .01 for name, data in arrays.items()}
            if seed == 19 and runtime == 'jittor' and change == 'extra_phase_missing_gradient':
                del raw['ingrad::key']
            path = folder / ('seed%d-%s.npz' % (seed, runtime))
            np.savez(path, **raw)
            path.with_suffix('.json').write_text(json.dumps(value))
    marker = {'case': contract.CASE, 'status': 'passed', 'baseline': 'baseline', 'device': 'cpu',
              'seeds': [0, 7, 19], 'checked_tensors': 15}
    (folder / 'complete-comparison.json').write_text(json.dumps(marker))
    platform = tmp_path / 'platform.json'
    platform.write_text(json.dumps({'precision': values[0]['tf32']}))
    args = (tmp_path / 'framework', source, wheel, folder.parent, platform, 'baseline',
            'docs/results/test.md', 'nox -s ecosystem_vllm')
    if change is None:
        record = collector.collect(*args)
        assert record['level'] == 2 and record['numerics']['checked_tensors'] == 15
        roles = {artifact['role'] for artifact in record['evidence']['artifacts']}
        assert {'perturb-comparison', 'perturb19-comparison', 'base-comparison', 'comparison', 'upstream-wheel'} <= roles
    else:
        with pytest.raises(ValueError):
            collector.collect(*args)

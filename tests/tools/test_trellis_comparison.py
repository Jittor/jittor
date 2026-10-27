"""Synthetic protocol rejection tests; these establish no TRELLIS runtime support."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def gate():
    import sys
    path = ROOT / 'compat/tests/torch'
    sys.path.insert(0, str(path))
    try:
        spec = importlib.util.spec_from_file_location('trellis_gate_contract', path / 'test_ecosystem_trellis.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(path))


@pytest.fixture
def pair(gate):
    metadata = {key: 'synthetic' for key in (
        'case', 'upstream', 'source_hashes', 'source_version', 'trellis_origin',
        'model_class', 'probe_sha256', 'weights_sha256', 'tf32', 'runtime_conditions', 'jittor_commit')}
    metadata.update(device=gate.DEVICE, steps=3, perturbation=0, perturb_input=None,
                    trainable_parameters=['weight', 'bias'], frozen_parameters=[],
                    input_gradients=['x', 'cond'], input_shapes={'x': [1], 'cond': [1]},
                    parameter_elements=2)
    theirs = dict(metadata, runtime='torch', torch_binary='/synthetic/torch/_C.so')
    ours = dict(metadata, runtime='jittor', fallback_policy='error', fallback_count=0,
                backend={'use_cuda': gate.DEVICE == 'cuda', 'use_acl': False})
    arrays = {}
    for step in range(3):
        for kind in ('output', 'loss'):
            arrays[f'{step}/{kind}'] = np.array([.1], dtype=np.float32)
        for kind in ('gradient', 'parameter', 'momentum'):
            for name in metadata['trainable_parameters']:
                arrays[f'{step}/{kind}/{name}'] = np.array([.1], dtype=np.float32)
        for kind in ('input', 'input_grad'):
            for name in metadata['input_gradients']:
                arrays[f'{step}/{kind}/{name}'] = np.array([.1], dtype=np.float32)
    return arrays, deepcopy(arrays), deepcopy(theirs), deepcopy(ours)


def test_complete_synthetic_protocol_can_compare(gate, pair):
    reference, actual, theirs, ours = pair
    result = gate._compare(reference, actual, theirs, ours)
    assert result['checked_tensors'] == 36
    assert result['max_absolute'] == 0


@pytest.mark.parametrize('field', ('upstream', 'tf32', 'runtime_conditions', 'weights_sha256',
                                   'probe_sha256', 'source_hashes', 'device', 'jittor_commit'))
def test_mismatched_measurement_identity_rejected(gate, pair, field):
    reference, actual, theirs, ours = pair
    ours[field] = 'changed'
    with pytest.raises(AssertionError, match='protocol mismatch'):
        gate._compare(reference, actual, theirs, ours)


@pytest.mark.parametrize('key', ('1/gradient/weight', '2/input_grad/cond', '0/momentum/bias'))
def test_both_runtimes_omitting_same_required_tensor_rejected(gate, pair, key):
    reference, actual, theirs, ours = pair
    del reference[key]
    del actual[key]
    with pytest.raises(AssertionError):
        gate._compare(reference, actual, theirs, ours)


@pytest.mark.parametrize('value', (np.nan, np.inf, 0.))
def test_nonfinite_or_disappearing_small_gradient_rejected(gate, pair, value):
    reference, actual, theirs, ours = pair
    reference['1/gradient/bias'] = np.array([1e-5], dtype=np.float32)
    actual['1/gradient/bias'] = np.array([value], dtype=np.float32)
    with pytest.raises(AssertionError):
        gate._compare(reference, actual, theirs, ours)


@pytest.mark.parametrize('change', ('fallback', 'oracle'))
def test_fallback_or_nonbinary_oracle_rejected(gate, pair, change):
    reference, actual, theirs, ours = pair
    if change == 'fallback':
        ours['fallback_count'] = 1
    else:
        theirs['torch_binary'] = '/synthetic/torch/__init__.py'
    with pytest.raises(AssertionError):
        gate._compare(reference, actual, theirs, ours)


@pytest.fixture(scope='module')
def collector():
    path = ROOT / 'agent/skills/trellis-torch-compat/scripts/collect_scorecard.py'
    spec = importlib.util.spec_from_file_location('trellis_scorecard_collector', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def gate_artifacts(tmp_path):
    import hashlib
    import json
    upstream = {'commit': 'a' * 40, 'version': 'source@' + 'a' * 40, 'remote': 'synthetic'}
    source = {'path': 'trellis2/models/public.py', 'sha256': 'b' * 64}
    inventory = {'repo': 'trellis', 'upstream': upstream, 'sources': [source],
                 'package_sources': [source], 'package_files': [source]}
    installation = {'repo': 'trellis', 'version': upstream['version'],
                    'installation_kind': 'source-checkout', 'package': '/synthetic/trellis2',
                    'origin': '/synthetic/trellis2/__init__.py',
                    'source_sha256': {source['path']: source['sha256']},
                    'installed_files_sha256': {'models/public.py': source['sha256']}}
    metadata = {'jittor_commit': 'c' * 40, 'upstream': upstream, 'probe_sha256': 'd' * 64,
                'source_hashes': {source['path']: source['sha256']},
                'downstream_distribution': installation, 'trellis_origin': installation['origin'],
                'case': 'trellis_SyntheticDeclaredModel', 'weights_sha256': 'e' * 64,
                'perturb_input': None,
                'trainable_parameters': ['weight'], 'input_gradients': ['x'],
                'model': 'SyntheticDeclaredModel', 'device': 'cpu', 'steps': 3,
                'source_version': upstream['version'], 'input_shapes': {'x': [1]},
                'frozen_parameters': [], 'fallback_policy': 'error', 'fallback_count': 0}
    errors = {'checked_tensors': 21, 'max_absolute': 0.,
              'fields': {'output': {'max_scaled_relative': 0., 'tolerance': .002}}}
    value = {'case': 'trellis_SyntheticDeclaredModel', 'device': 'cpu', 'status': 'passed',
             'jittor_result': dict(deepcopy(metadata), runtime='jittor'),
             'oracle_result': dict(deepcopy(metadata), runtime='torch'),
             'input_perturbation_verified': True, 'errors': errors,
             'input_perturbations': [{'input': 'x', 'errors': errors}]}
    directory = tmp_path / 'run/cpu/SyntheticDeclaredModel'
    directory.mkdir(parents=True)
    for name in ('reference.npz', 'candidate.npz', 'base-comparison.json', 'perturb-comparison.json'):
        (directory / name).write_bytes(b'synthetic contract fixture; no runtime support')
    changed = {}
    for stem, key in (('reference', 'oracle_result'), ('candidate', 'jittor_result')):
        (directory / (stem + '.json')).write_text(json.dumps(value[key]))
        changed[stem] = dict(deepcopy(value[key]), perturb_input='x')
        (directory / (stem + '-x.json')).write_text(json.dumps(changed[stem]))
    (directory / 'perturb1-comparison.json').write_text(json.dumps({
        'case': value['case'], 'status': 'passed', 'input_perturbation_verified': True,
        'input': 'x', 'reference_result': changed['reference'], 'candidate_result': changed['candidate']}))
    comparison = directory / 'comparison.json'
    comparison.write_text(json.dumps(value))
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir()}
    (directory / 'artifacts.json').write_text(json.dumps(hashes))
    platform = tmp_path / 'platform.json'
    platform.write_text('{}')
    return inventory, tmp_path / 'run', {'cpu': platform}, comparison


def _collect(collector, gate_artifacts):
    inventory, run, platforms, _ = gate_artifacts
    return collector.collect(inventory, [run], platforms, 'c' * 40,
                             'docs/results/synthetic-contract-only.md', 'synthetic')


def test_collector_only_awards_actual_named_model_axis(collector, gate_artifacts):
    records = _collect(collector, gate_artifacts)
    assert len(records) == 1 and records[0]['level'] == 3
    assert records[0]['registry_refs'] == [
        {'repo': 'trellis', 'axis': 'models', 'member': 'SyntheticDeclaredModel'}]
    assert records[0]['numerics']['checked_tensors'] == 42


@pytest.mark.parametrize('change', ('baseline', 'upstream', 'source_hashes', 'probe_sha256'))
def test_collector_rejects_identity_mismatch(collector, gate_artifacts, change):
    import json
    path = gate_artifacts[3]
    value = json.loads(path.read_text())
    value['jittor_result']['jittor_commit' if change == 'baseline' else change] = 'changed'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        _collect(collector, gate_artifacts)


def test_collector_rejects_tampered_raw_artifact(collector, gate_artifacts):
    (gate_artifacts[3].parent / 'candidate.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Changed TRELLIS artifact'):
        _collect(collector, gate_artifacts)


def test_collector_rejects_duplicate_case_device(collector, gate_artifacts):
    inventory, run, platforms, _ = gate_artifacts
    with pytest.raises(ValueError, match='Duplicate TRELLIS'):
        collector.collect(inventory, [run, run], platforms, 'c' * 40, 'synthetic', 'synthetic')


def test_collector_rejects_missing_source_file_hash(collector, gate_artifacts):
    import json
    path = gate_artifacts[3]
    value = json.loads(path.read_text())
    value['jittor_result']['source_hashes'] = {}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='source file hash mismatch'):
        _collect(collector, gate_artifacts)


@pytest.mark.parametrize('omitted', ('candidate.npz', 'comparison.json'))
def test_collector_rejects_incomplete_artifact_manifest(collector, gate_artifacts, omitted):
    import json
    path = gate_artifacts[3].parent / 'artifacts.json'
    hashes = json.loads(path.read_text())
    del hashes[omitted]
    path.write_text(json.dumps(hashes))
    with pytest.raises(ValueError, match='Incomplete TRELLIS artifact manifest'):
        _collect(collector, gate_artifacts)


@pytest.mark.parametrize('stem', ('reference', 'candidate'))
@pytest.mark.parametrize('phase', ('', '-x'))
@pytest.mark.parametrize('change', ('missing', 'source', 'payload', 'version'))
def test_collector_rechecks_every_runtime_phase_despite_honest_rehash(
        collector, gate_artifacts, stem, phase, change):
    import hashlib
    import json

    directory = gate_artifacts[3].parent
    path = directory / (stem + phase + '.json')
    value = json.loads(path.read_text())
    binding = value['downstream_distribution']
    if change == 'missing':
        value.pop('downstream_distribution')
    elif change == 'source':
        binding['source_sha256'] = {}
    elif change == 'payload':
        binding['installed_files_sha256'] = {}
    else:
        binding['version'] = 'a-false-release'
    path.write_text(json.dumps(value))
    manifest = {item.name: hashlib.sha256(item.read_bytes()).hexdigest()
                for item in directory.iterdir() if item.name != 'artifacts.json'}
    (directory / 'artifacts.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        _collect(collector, gate_artifacts)


def test_maintained_gate_retains_failed_numerics_without_success(gate, pair, tmp_path, monkeypatch):
    checkout = tmp_path / 'upstream'
    (checkout / 'trellis2/models').mkdir(parents=True)
    (checkout / 'trellis2/models/__init__.py').write_text('# synthetic host contract\n')
    output = tmp_path / 'evidence'
    monkeypatch.setenv('JITTOR_TRELLIS_CHECKOUT', str(checkout))
    monkeypatch.setenv('JITTOR_TRELLIS_EVIDENCE_DIR', str(output))
    monkeypatch.setattr(gate, 'REAL_TORCH_PYTHON', '/synthetic/binary-torch-python')
    reference, actual, theirs, ours = pair
    actual['0/output'] = np.array([9.], dtype=np.float32)
    theirs['weights'] = str(tmp_path / 'weights.npz')

    def completed_worker(runtime, case, checkout, target, weights=None, perturb_input=None):
        target.write_bytes(b'raw files from successful synthetic worker')
        return (reference, theirs) if runtime == 'torch' else (actual, ours)

    monkeypatch.setattr(gate, '_run', completed_worker)
    with pytest.raises(AssertionError):
        gate.test_real_trellis_models_training('SyntheticDeclaredModel')
    directory = output / gate.DEVICE / 'SyntheticDeclaredModel'
    assert (directory / 'reference.npz').is_file()
    assert (directory / 'candidate.npz').is_file()
    assert not (directory / 'comparison.json').exists()
    assert not (directory / 'artifacts.json').exists()

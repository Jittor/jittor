"""Actual TRELLIS.2 model training; does not certify the pretrained 3D pipeline."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pytest

from _helpers.child_process import child_env, run_python_child
from _ecosystem_harness import (
    REAL_TORCH_PYTHON, EcosystemComparison, _comparison_floor, _divergence, _enabled,
)

ROOT = Path(__file__).resolve().parents[3]
RUNNER = Path(__file__).with_name('_ecosystem_trellis_runner.py')
DEVICE = os.environ.get('JITTOR_TEST_DEVICES', 'cpu')
if DEVICE not in ('cpu', 'cuda'):
    raise ValueError('TRELLIS gate requires exactly cpu or cuda')


def _run(runtime, case, checkout, output, weights=None, perturb_input=None):
    arguments = [str(RUNNER), '--root', str(ROOT), '--checkout', str(checkout),
                 '--runtime', runtime, '--case', case, '--device', DEVICE,
                 '--output', str(output)]
    if weights:
        arguments.extend(('--weights', str(weights)))
    if perturb_input:
        arguments.extend(('--perturb-input', perturb_input, '--perturbation', '.125'))
    environment = os.environ.copy()
    if runtime == 'jittor':
        result = run_python_child(arguments, env=environment, inherit=False,
                                  merge_stderr=True, timeout=1800)
    else:
        environment.pop('LD_LIBRARY_PATH', None)
        for name in list(environment):
            if name.startswith(('JT_', 'JITTOR_', 'NCCL_')) or name in ('use_cuda', 'nvcc_path'):
                environment.pop(name)
        result = subprocess.run(
            [REAL_TORCH_PYTHON, *arguments],
            env=child_env(environment, inherit=False, repo_paths=False, without_torch_mode=True),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1800)
    output.with_suffix('.log').write_text(result.stdout)
    assert result.returncode == 0, result.stdout[-10000:]
    with np.load(output, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive}
    metadata = json.loads(output.with_suffix('.json').read_text())
    return arrays, metadata


def _compare(reference, candidate, theirs, ours):
    assert theirs['runtime'] == 'torch' and ours['runtime'] == 'jittor'
    assert theirs['torch_binary'].endswith(('.so', '.pyd'))
    for key in ('case', 'device', 'steps', 'perturbation', 'perturb_input', 'upstream',
                'source_hashes', 'source_version', 'trellis_origin', 'model_class',
                'probe_sha256', 'weights_sha256', 'trainable_parameters',
                'frozen_parameters', 'input_gradients', 'input_shapes',
                'parameter_elements', 'tf32', 'runtime_conditions', 'jittor_commit'):
        assert ours[key] == theirs[key], 'TRELLIS protocol mismatch: ' + key
    assert ours['device'] == DEVICE and ours['steps'] == 3
    assert ours['fallback_policy'] == 'error' and ours['fallback_count'] == 0
    assert ours['backend']['use_cuda'] == (DEVICE == 'cuda')
    assert ours['backend']['use_acl'] is False
    assert ours['parameter_elements'] > 0 and ours['trainable_parameters']
    expected = set()
    for step in range(3):
        expected.update('{}/{}'.format(step, name) for name in ('output', 'loss'))
        for kind in ('gradient', 'parameter', 'momentum'):
            expected.update('{}/{}/{}'.format(step, kind, name)
                            for name in ours['trainable_parameters'])
        for kind in ('input', 'input_grad'):
            expected.update('{}/{}/{}'.format(step, kind, name)
                            for name in ours['input_gradients'])
    assert set(reference) == set(candidate) == expected
    errors = {'checked_tensors': len(expected), 'max_absolute': 0., 'fields': {}}
    for field in ('output', 'loss', 'input', 'input_grad', 'gradient', 'parameter', 'momentum'):
        keys = sorted(key for key in expected if key.split('/')[1] == field)
        floor = _comparison_floor(reference, keys)
        tolerance = (EcosystemComparison.backward_tolerance if field in ('gradient', 'input_grad')
                     else EcosystemComparison.forward_tolerance)
        row = {'max_scaled_relative': 0., 'max_absolute': 0., 'tolerance': tolerance,
               'scale_floor': floor}
        for key in keys:
            assert candidate[key].shape == reference[key].shape, key
            assert np.isfinite(candidate[key]).all() and np.isfinite(reference[key]).all(), key
            difference = _divergence(candidate[key], reference[key], floor)
            assert difference <= tolerance, (key, difference, tolerance)
            absolute = float(np.max(np.abs(candidate[key] - reference[key])))
            row['max_scaled_relative'] = max(row['max_scaled_relative'], difference)
            row['max_absolute'] = max(row['max_absolute'], absolute)
        errors['fields'][field] = row
        errors['max_absolute'] = max(errors['max_absolute'], row['max_absolute'])
    return errors


@pytest.mark.parametrize('case', ('SparseStructureFlowModel', 'SparseStructureEncoder',
                                 'SparseStructureDecoder'))
def test_real_trellis_models_training(case):
    checkout = os.environ.get('JITTOR_TRELLIS_CHECKOUT', '')
    if not checkout or not REAL_TORCH_PYTHON:
        if _enabled('JITTOR_REQUIRE_TRELLIS') or _enabled('JITTOR_REQUIRE_REAL_TORCH'):
            pytest.fail('TRELLIS requires JITTOR_TRELLIS_CHECKOUT and REAL_TORCH_PYTHON')
        pytest.skip('TRELLIS pinned checkout and independent Torch are not configured')
    checkout = Path(checkout).resolve()
    assert (checkout / 'trellis2/models/__init__.py').is_file()
    persistent = os.environ.get('JITTOR_TRELLIS_EVIDENCE_DIR', '')
    with tempfile.TemporaryDirectory(prefix='trellis-training-') as temporary:
        directory = Path(persistent) / DEVICE / case if persistent else Path(temporary)
        if persistent:
            directory.mkdir(parents=True, exist_ok=False)
        reference, theirs = _run('torch', case, checkout, directory / 'reference.npz')
        candidate, ours = _run('jittor', case, checkout, directory / 'candidate.npz', theirs['weights'])
        errors = _compare(reference, candidate, theirs, ours)
        summary = {'case': ours['case'], 'status': 'passed', 'device': DEVICE,
                   'baseline': ours['jittor_commit'], 'errors': errors,
                   'oracle_result': theirs, 'jittor_result': ours}
        (directory / 'base-comparison.json').write_text(json.dumps({
            'baseline': ours['jittor_commit'], 'cases': [summary]}, indent=2) + '\n')
        perturbations = []
        for name in ours['input_gradients']:
            ref, meta_ref = _run('torch', case, checkout, directory / ('reference-' + name + '.npz'),
                                 theirs['weights'], name)
            actual, meta_actual = _run('jittor', case, checkout, directory / ('candidate-' + name + '.npz'),
                                      theirs['weights'], name)
            perturbed_errors = _compare(ref, actual, meta_ref, meta_actual)
            for changed, original in ((ref, reference), (actual, candidate)):
                assert not np.allclose(changed['0/output'], original['0/output'], atol=1e-8, rtol=0), name
                for other in ours['input_gradients']:
                    if other != name:
                        np.testing.assert_array_equal(changed['0/input/' + other], original['0/input/' + other])
            perturbations.append({'input': name, 'errors': perturbed_errors})
        summary['input_perturbation_verified'] = True
        summary['input_perturbations'] = perturbations
        (directory / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
        (directory / 'perturb-comparison.json').write_text(json.dumps({
            'case': ours['case'], 'status': 'passed', 'input_perturbation_verified': True,
            'inputs': perturbations, 'reference_result': meta_ref,
            'candidate_result': meta_actual}, indent=2) + '\n')
        artifacts = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(directory.iterdir()) if path.is_file()}
        (directory / 'artifacts.json').write_text(json.dumps(artifacts, indent=2) + '\n')
        print('TRELLIS_ACCEPTED', case, DEVICE, json.dumps(errors))

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

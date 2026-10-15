"""Reject component evidence missing provenance or requested numerical fields."""
import ast
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'agent/skills/torch-compat-repo-runbook/scripts'


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def collector():
    return load('collect_component_scorecard')


@pytest.fixture
def measurement():
    return load('component_measure')


def test_comparison_reaudit_preserves_existing_evidence(tmp_path, collector):
    path = tmp_path / 'complete-comparison.json'
    value = {'case': 'component', 'status': 'passed', 'rows': []}
    collector.write_comparison(path, value)
    original = path.read_bytes()
    # A fixed timestamp distinguishes reuse from rewriting identical content.
    os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    collector.write_comparison(path, value)
    assert path.read_bytes() == original
    assert path.stat().st_mtime_ns == 1_000_000_000
    with pytest.raises(ValueError, match='Existing comparison differs'):
        collector.write_comparison(path, dict(value, status='changed'))
    assert path.read_bytes() == original
    assert path.stat().st_mtime_ns == 1_000_000_000


def test_binding_rejects_missing_file_stale_baseline_and_changed_version(collector):
    source = {'repo': 'torchmetrics', 'upstream': {'version': '1.7.4', 'commit': 'a' * 40},
              'package_sources': [{'path': 'src/torchmetrics/__init__.py', 'sha256': 'b' * 64}]}
    value = {'jittor_commit': 'c' * 40, 'upstream': source['upstream'], 'hf_hub_offline': True,
             'downstream_distribution': {'repo': 'torchmetrics', 'version': '1.7.4',
                'source_sha256': {'src/torchmetrics/__init__.py': 'b' * 64},
                'installed_files_sha256': {'__init__.py': 'b' * 64}},
             'measurement_source_sha256': {'runner.py': 'd' * 64}}
    collector.validate_binding(value, source, 'c' * 40)
    for field in ('jittor_commit', 'upstream', 'measurement_source_sha256'):
        changed = copy.deepcopy(value)
        changed.pop(field)
        with pytest.raises(ValueError):
            collector.validate_binding(changed, source, 'c' * 40)
    for field in ('source_sha256', 'installed_files_sha256', 'version'):
        changed = copy.deepcopy(value)
        changed['downstream_distribution'].pop(field)
        with pytest.raises(ValueError):
            collector.validate_binding(changed, source, 'c' * 40)


@pytest.mark.parametrize('failure', ['missing', 'extra', 'shape', 'nonfinite', 'numerical'])
def test_array_comparison_refuses_incomplete_or_invalid_gradients(tmp_path, failure, collector):
    reference = {'__output__': np.array([1., 2.]), 'ingrad::sample': np.array([3., 4.])}
    candidate = {k: v.copy() for k, v in reference.items()}
    if failure == 'missing':
        candidate.pop('ingrad::sample')
    elif failure == 'extra':
        candidate['not-requested'] = np.ones(1)
    elif failure == 'shape':
        candidate['ingrad::sample'] = np.ones(1)
    elif failure == 'nonfinite':
        candidate['ingrad::sample'][0] = np.nan
    else:
        candidate['ingrad::sample'][0] = 100.
    paths = [tmp_path / 'ref.npz', tmp_path / 'cand.npz']
    for path, arrays in zip(paths, (reference, candidate)):
        np.savez(path, **arrays)
    with pytest.raises(ValueError):
        collector.compare_arrays(*paths, reference, .001, .01)


def test_actual_installed_sources_are_hashed_and_corruption_refused(tmp_path, measurement):
    package = tmp_path / 'torchmetrics'
    package.mkdir()
    path = package / '__init__.py'
    path.write_text('__version__ = "1.7.4"\n')
    (package / 'binary.so').write_bytes(b'actual installed bytes')
    source = {'repo': 'torchmetrics', 'upstream': {'version': '1.7.4'},
              'package_sources': [{'path': 'src/torchmetrics/__init__.py',
                                   'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}]}
    module = SimpleNamespace(__file__=str(path), __version__='1.7.4')
    observed = measurement.installed_distribution(source, module)
    assert set(observed['installed_files_sha256']) == {'__init__.py', 'binary.so'}
    path.write_text('changed source\n')
    with pytest.raises(ValueError, match='differs'):
        measurement.installed_distribution(source, module)


@pytest.mark.parametrize('failure', [None, 'new-training-pair', 'modified-file', 'missing-file'])
def test_every_phase_retains_the_actual_baseline_weight_snapshot(tmp_path, collector, failure):
    original = tmp_path / 'original.weights.npz'
    np.savez(original, weight=np.array([1., 2.], dtype='float32'))
    expected = hashlib.sha256(original.read_bytes()).hexdigest()
    values = [{'weights': str(original), 'weights_sha256': expected} for _ in range(2)]
    if failure == 'new-training-pair':
        # Both training runtimes agreeing with each other is insufficient when
        # they no longer use the baseline/perturbation's initial parameters.
        alternate = tmp_path / 'other.weights.npz'
        np.savez(alternate, weight=np.array([3., 4.], dtype='float32'))
        values = [{'weights': str(alternate), 'weights_sha256': hashlib.sha256(alternate.read_bytes()).hexdigest()}
                  for _ in range(2)]
    elif failure == 'modified-file':
        np.savez(original, weight=np.array([5., 6.], dtype='float32'))
    elif failure == 'missing-file':
        original.unlink()
    if failure is None:
        collector.validate_initial_weights(values, expected)
    else:
        with pytest.raises(ValueError):
            collector.validate_initial_weights(values, expected)


def test_registry_members_are_actually_called_by_maintained_builders(collector):
    tree = ast.parse((ROOT / 'compat/tests/torch/_ecosystem_cases.py').read_text())
    for spec in collector.SPECS.values():
        builder = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                       and n.name == spec.get('builder', '_' + spec['case']))
        calls = {n.func.id for n in ast.walk(builder) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert set(spec['members']) <= calls
    assert collector.SPECS['torchmetrics']['level'] == 2
    assert collector.SPECS['torchmetrics']['parameters'] == []
    assert collector.SPECS['tensordict']['level'] == 3

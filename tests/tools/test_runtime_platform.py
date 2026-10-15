"""The extracted runtime header preserves fields and rejects invented inputs."""

import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / 'agent/skills/torch-compat-repo-runbook/scripts'


@pytest.fixture
def capture(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import _runtime_platform as module

    monkeypatch.setattr(module.platform, 'platform', lambda: 'fixture Linux')
    monkeypatch.setattr(module, 'cpu_description', lambda: 'fixture CPU')
    monkeypatch.setattr(module, 'selected_nvidia_wheels', lambda: {'nvidia-runtime': '12.8'})
    monkeypatch.setattr(module, 'command', lambda executable, _: executable + ' version')
    return module


def inputs(device='cuda'):
    precision = {'matmul': False, 'cudnn': False, 'cudnn_benchmark': False, 'matmul_precision': 'highest'}
    flags = {'use_cuda': int(device == 'cuda'), 'use_acl': 0, 'graph_replay': 'default', 'tf32': False, 'hf32': False}
    return {'physical_devices': [['0', 'GPU-fixture-zero', 'fixture card zero', 'fixture driver'],
                                  ['3', 'GPU-fixture-three', 'fixture card three', 'fixture driver']]
                                 if device == 'cuda' else [],
            'card_indices': ['0', '3'] if device == 'cuda' else [],
            'precision': precision, 'jittor_flags': flags, 'nvcc': 'fixture-nvcc', 'cxx': 'fixture-cxx'}


@pytest.mark.parametrize('device', ('cpu', 'cuda'))
@pytest.mark.parametrize('regime', ('default', 'eager'))
def test_original_accelerate_platform_fields_remain_equal(capture, device, regime):
    values = inputs(device)
    values['jittor_flags']['graph_replay'] = regime
    result = capture.capture_platform(SimpleNamespace(version=SimpleNamespace(cuda='12.8')), device, **values)
    # This is the pre-extraction Capture.finish platform contract, including
    # scalar/list distinctions, original flag spellings and CPU sentinels.
    expected = {'os': 'fixture Linux', 'cpu': 'fixture CPU',
                'accelerator_model': ['fixture card zero', 'fixture card three'] if device == 'cuda' else 'not-applicable',
                'card_count': 2 if device == 'cuda' else 0, 'card_indices': values['card_indices'],
                'physical_devices': [{'index': '0', 'uuid': 'GPU-fixture-zero'},
                                     {'index': '3', 'uuid': 'GPU-fixture-three'}] if device == 'cuda' else [],
                'driver': 'fixture driver' if device == 'cuda' else 'not-applicable',
                'cuda': '12.8' if device == 'cuda' else 'not-applicable', 'cann': 'not-applicable',
                'nvcc': 'fixture-nvcc version' if device == 'cuda' else 'not-applicable',
                'gcc': 'fixture-cxx version', 'nvidia_wheels': {'nvidia-runtime': '12.8'},
                'jittor_flags': values['jittor_flags'], 'precision': values['precision']}
    assert result == expected


@pytest.mark.parametrize('change', ('missing-card', 'wrong-allocation', 'duplicate-allocation',
                                    'duplicate-uuid', 'missing-precision', 'string-precision',
                                    'missing-flags', 'wrong-device-flag', 'wrong-tf32'))
def test_incomplete_or_conflicting_observations_are_rejected(capture, change):
    values = inputs()
    if change == 'missing-card':
        values['physical_devices'].pop()
    elif change == 'wrong-allocation':
        values['card_indices'][0] = 'unallocated'
    elif change == 'duplicate-allocation':
        values['card_indices'][1] = values['card_indices'][0]
    elif change == 'duplicate-uuid':
        values['physical_devices'][1][1] = values['physical_devices'][0][1]
    elif change == 'missing-precision':
        del values['precision']['cudnn']
    elif change == 'string-precision':
        values['precision']['cudnn'] = 'false'
    elif change == 'missing-flags':
        del values['jittor_flags']['use_cuda']
    elif change == 'wrong-device-flag':
        values['jittor_flags']['use_cuda'] = 0
    else:
        values['jittor_flags']['tf32'] = True
    with pytest.raises(ValueError):
        capture.capture_platform(SimpleNamespace(version=SimpleNamespace(cuda='12.8')), 'cuda', **values)


def test_cpu_cannot_relabel_a_cuda_inventory(capture):
    with pytest.raises(ValueError):
        capture.capture_platform(SimpleNamespace(), 'cpu', **inputs())


def test_compiler_failure_remains_a_failure(capture, monkeypatch):
    def fail(*_):
        raise subprocess.CalledProcessError(1, ['compiler', '--version'])

    monkeypatch.setattr(capture, 'command', fail)
    with pytest.raises(subprocess.CalledProcessError):
        capture.capture_platform(SimpleNamespace(version=SimpleNamespace(cuda='12.8')), 'cuda', **inputs())


def test_new_source_closure_does_not_retrofit_old_commits(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location('accelerate_platform_source_test', SCRIPTS / '_accelerate_scorecard.py')
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)

    def git(*args):
        return subprocess.check_output(['git', '-C', str(tmp_path), *args], text=True).strip()

    git('init', '-q')
    git('config', 'user.name', 'Platform fixture')
    git('config', 'user.email', 'platform@example.invalid')
    for name in protocol.MEASUREMENT_SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# old source fixture\n', encoding='utf-8')
    git('add', '.')
    git('-c', 'commit.gpgsign=false', 'commit', '-qm', 'Old source closure')
    old = git('rev-parse', 'HEAD')
    helper = 'agent/skills/torch-compat-repo-runbook/scripts/_runtime_platform.py'
    (tmp_path / helper).write_text('# new platform source fixture\n', encoding='utf-8')
    git('add', helper)
    git('-c', 'commit.gpgsign=false', 'commit', '-qm', 'New source closure')
    new = git('rev-parse', 'HEAD')
    monkeypatch.setattr(protocol, 'ROOT', tmp_path)
    before, after = protocol.measurement_sources(old), protocol.measurement_sources(new)
    assert set(before) == set(protocol.MEASUREMENT_SOURCES)
    assert set(after) == set(before) | {helper}
    assert {name: after[name] for name in before} == before

"""Omni workers bind the shared header to actual allocation and native flags."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def observed(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'agent/skills/torch-compat-repo-runbook/scripts'))
    import _runtime_platform as shared

    spec = importlib.util.spec_from_file_location(
        'omni_platform_test', ROOT / 'agent/skills/vllm-omni-torch-compat/scripts/probe_h3.py')
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    monkeypatch.setattr(shared.platform, 'platform', lambda: 'observed OS')
    monkeypatch.setattr(shared, 'command', lambda path, _: str(path) + ' version')
    monkeypatch.setattr(shared, 'cpu_description', lambda: 'observed CPU')
    monkeypatch.setattr(shared, 'selected_nvidia_wheels', lambda: {'nvidia-cuda-runtime': '12.2'})
    monkeypatch.setattr(probe.shutil, 'which', lambda name: '/observed/' + name)
    monkeypatch.delenv('JT_BUILD_NVCC_PATH', raising=False)

    def inventory(command, **_):
        assert command == ['nvidia-smi', '--query-gpu=index,uuid,name,driver_version', '--format=csv,noheader']
        return '0, GPU-other, other device, 580\n3, GPU-observed, observed device, 580\n'

    monkeypatch.setattr(probe.subprocess, 'check_output', inventory)
    torch = SimpleNamespace(
        version=SimpleNamespace(cuda='12.2'),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                                 cudnn=SimpleNamespace(allow_tf32=False, benchmark=False)),
        get_float32_matmul_precision=lambda: 'highest')
    return probe, torch


def controls(device):
    return {'cuda_visible_devices': '3' if device == 'cuda' else '',
            'cuda_available': device == 'cuda',
            'cuda_device': {'uuid': 'GPU-observed', 'physical_index': 3, 'logical_index': 0}}


@pytest.mark.parametrize('device', ('cpu', 'cuda'))
@pytest.mark.parametrize('native', (False, True))
def test_worker_platform_uses_verified_allocation_and_native_compiler(observed, device, native):
    probe, torch = observed
    backend = SimpleNamespace(
        flags=SimpleNamespace(use_cuda=int(device == 'cuda'), auto_graph_replay=1),
        compiler=SimpleNamespace(cc_path='/native/g++', nvcc_path='/native/nvcc')) if native else None
    result = probe.observed_platform(torch, SimpleNamespace(device=device), controls(device), backend)
    assert result['gcc'] == ('/native/g++' if native else '/observed/g++') + ' version'
    assert result['card_count'] == int(device == 'cuda')
    assert result['jittor_flags']['graph_replay'] == ('default' if native else 'eager')
    assert result['jittor_flags']['use_cuda'] == int(device == 'cuda')
    if device == 'cuda':
        assert result['physical_devices'] == [{'index': '3', 'uuid': 'GPU-observed'}]
        assert result['accelerator_model'] == ['observed device']
        assert result['nvcc'] == ('/native/nvcc' if native else '/observed/nvcc') + ' version'
    else:
        assert result['physical_devices'] == [] and result['nvcc'] == 'not-applicable'


def test_inventory_cannot_replace_the_driver_observed_uuid(observed):
    probe, torch = observed
    actual = controls('cuda')
    actual['cuda_device']['uuid'] = 'GPU-not-observed'
    with pytest.raises(RuntimeError, match='actual CUDA driver UUID'):
        probe.observed_platform(torch, SimpleNamespace(device='cuda'), actual)


def test_native_cuda_flags_cannot_be_labeled_as_cpu(observed):
    probe, torch = observed
    backend = SimpleNamespace(flags=SimpleNamespace(use_cuda=1, auto_graph_replay=0),
                              compiler=SimpleNamespace(cc_path='/native/g++', nvcc_path='/native/nvcc'))
    with pytest.raises(ValueError, match='Backend flags disagree'):
        probe.observed_platform(torch, SimpleNamespace(device='cpu'), controls('cpu'), backend)


def test_missing_compiler_is_a_setup_failure(observed, monkeypatch):
    probe, torch = observed
    monkeypatch.setattr(probe.shutil, 'which', lambda name: None if name == 'nvcc' else '/observed/g++')
    with pytest.raises(RuntimeError, match='compiler executables'):
        probe.observed_platform(torch, SimpleNamespace(device='cuda'), controls('cuda'))

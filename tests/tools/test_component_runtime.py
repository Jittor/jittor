"""Shared device/runtime evidence rejects forged observations."""
import copy
import hashlib
from pathlib import Path
import pytest
SCRIPTS = Path(__file__).resolve().parents[2] / 'agent/skills/torch-compat-repo-runbook/scripts'

def sha(value):
    return hashlib.sha256(value).hexdigest()

@pytest.fixture(params=('cpu', 'cuda'))
def runtime_packet(tmp_path, monkeypatch, request):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import _component_runtime as component
    binary = tmp_path / '_C.so'
    binary.write_bytes(b'host-only identity fixture; not loaded')
    device = request.param
    precision = {'matmul': False, 'cudnn': False, 'cudnn_benchmark': False, 'matmul_precision': 'highest'}
    platform = {'precision': precision, 'card_count': int(device == 'cuda'), 'card_indices': ['3'] if device == 'cuda' else [], 'physical_devices': [{'index': '3', 'uuid': 'GPU-observed'}] if device == 'cuda' else []}
    observation = {'device': device, 'dispatch': device, 'location': 'device' if device == 'cuda' else 'cpu', 'pointer_device': 0}
    packet = {'runtime': 'jittor', 'device': device, 'interpreter': {'is_shim': True, 'pid': 100}, 'runtime_environment': {key: str(tmp_path / key) for key in ('HOME', 'JITTOR_HOME', 'TMPDIR', 'XDG_CACHE_HOME')}, 'platform': platform, 'tf32': precision, 'tensor_observations': [observation], 'fallback_policy': 'error', 'fallback_count': 0, 'native_runtime_flags': {'use_cuda': int(device == 'cuda')}, 'device_controls': {'cuda_visible_devices': '3' if device == 'cuda' else '', 'cuda_available': device == 'cuda', 'cuda_device': {'uuid': 'GPU-observed'}}}
    return (component, packet, binary)

def test_runtime_provenance_accepts_actual_cpu_or_allocated_cuda(runtime_packet):
    (component, packet, binary) = runtime_packet
    component.validate_runtime_metadata(packet, 'baseline', packet['platform'], source_validator=lambda *_: None)
    oracle = copy.deepcopy(packet)
    oracle.update(runtime='torch', interpreter={'is_shim': False, 'pid': 101, 'binary_extension': str(binary), 'binary_sha256': sha(binary.read_bytes()), 'torch_version': '2.11.0'})
    expected = dict(packet['platform'], oracle={'is_shim': False, 'torch_version': '2.11.0', 'binary_extension': str(binary)})
    component.validate_runtime_metadata(oracle, 'baseline', expected, source_validator=lambda *_: None)
    binary.write_bytes(b'changed actual oracle extension')
    with pytest.raises(ValueError, match='binary'):
        component.validate_runtime_metadata(oracle, 'baseline', expected, source_validator=lambda *_: None)

@pytest.mark.parametrize('change', ('shim', 'pid', 'home', 'precision', 'device', 'fallback', 'native-device', 'residency', 'visibility', 'record-platform'))
def test_missing_or_contradictory_runtime_evidence_is_rejected(runtime_packet, change):
    (component, packet, _) = runtime_packet
    expected = copy.deepcopy(packet['platform'])
    if change == 'shim':
        packet['interpreter']['is_shim'] = False
    elif change == 'pid':
        packet['interpreter'].pop('pid')
    elif change == 'home':
        packet['runtime_environment'].pop('HOME')
    elif change == 'precision':
        packet['tf32'] = dict(packet['tf32'], matmul=True)
    elif change == 'device':
        packet['tensor_observations'][0]['device'] = 'wrong-device'
    elif change == 'fallback':
        packet['fallback_count'] = 1
    elif change == 'native-device':
        packet['native_runtime_flags']['use_cuda'] = 1 - int(packet['device'] == 'cuda')
    elif change == 'residency':
        packet['tensor_observations'][0]['location'] = 'wrong-location'
    elif change == 'visibility':
        packet['device_controls']['cuda_available'] = packet['device'] != 'cuda'
    else:
        expected['card_count'] += 1
    with pytest.raises(ValueError):
        component.validate_runtime_metadata(packet, 'baseline', expected, source_validator=lambda *_: None)

@pytest.mark.parametrize('change', ('uuid', 'pointer'))
def test_cuda_allocation_and_actual_pointer_must_agree(runtime_packet, change):
    (component, packet, _) = runtime_packet
    if packet['device'] == 'cpu':
        return
    if change == 'uuid':
        packet['device_controls']['cuda_device']['uuid'] = 'GPU-not-allocated'
    else:
        packet['tensor_observations'][0]['pointer_device'] = 1
    with pytest.raises(ValueError, match='physical CUDA'):
        component.validate_runtime_metadata(packet, 'baseline', source_validator=lambda *_: None)

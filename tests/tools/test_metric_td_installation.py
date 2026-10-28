"""Synthetic host payloads test provenance; no fixture extension is loaded."""

import base64
import copy
import csv
import hashlib
import importlib.util
import io
from pathlib import Path
import subprocess
from types import SimpleNamespace
import zipfile

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / 'agent/skills/torch-compat-repo-runbook/scripts'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def record_bytes(rows):
    text = io.StringIO()
    csv.writer(text).writerows(rows)
    return text.getvalue().encode()


@pytest.fixture(params=('torchmetrics', 'tensordict'))
def installation(tmp_path, monkeypatch, request):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import _installed_source_proof as shared
    import _metric_td_installation as component

    repo = request.param
    version, commit, remote, _ = component.PINS[repo]
    package = tmp_path / 'site' / repo
    package.mkdir(parents=True)
    payload = {'__init__.py': ('__version__ = ' + repr(version) + '\n').encode(),
               'resource.pt': b'non-Python fixture bytes'}
    if repo == 'tensordict':
        payload['_version.py'] = b"__version__ = version = '0.10.0'\n__commit_id__ = None\n"
        payload['_C.so'] = b'host fixture only; never a loaded native extension'
    prefix = 'src/' if repo == 'torchmetrics' else ''
    source = {'repo': repo, 'upstream': {'version': version, 'commit': commit,
                                      'remote': 'https://github.com/' + remote + '.git'},
              'package_sources': [{'path': prefix + repo + '/__init__.py',
                                   'sha256': sha(payload['__init__.py'])}],
              'package_files': [{'path': prefix + repo + '/' + name, 'sha256': sha(data)}
                                for name, data in payload.items() if not name.startswith(('_version', '_C'))]}
    metadata_root = package.parent / (repo + '-' + version + '.dist-info')
    metadata_root.mkdir()
    header = ('Metadata-Version: 2.1\nName: ' + repo + '\nVersion: ' + version + '\n').encode()
    (metadata_root / 'METADATA').write_bytes(header)
    rows = []
    for name, data in payload.items():
        (package / name).write_bytes(data)
        rows.append([repo + '/' + name,
                     'sha256=' + base64.urlsafe_b64encode(bytes.fromhex(sha(data))).decode().rstrip('='),
                     str(len(data))])
    wheel = tmp_path / (repo + '.whl')
    with zipfile.ZipFile(wheel, 'w') as archive:
        for name, data in payload.items():
            archive.writestr(repo + '/' + name, data)
        archive.writestr(metadata_root.name + '/METADATA', header)
        archive.writestr(metadata_root.name + '/RECORD', record_bytes(rows))
    (metadata_root / 'RECORD').write_bytes(record_bytes(rows))
    monkeypatch.setitem(component.PINS, repo, (version, commit, remote, sha(wheel.read_bytes())))
    module = SimpleNamespace(__file__=str(package / '__init__.py'), __path__=[str(package)], __version__=version)
    dist = SimpleNamespace(_path=metadata_root, metadata={'Name': repo}, version=version,
                           locate_file=lambda name: package.parent / name)
    monkeypatch.setattr(shared.metadata, 'distribution', lambda _: dist)
    binding = {'repo': repo, 'version': version, 'package': str(package),
               'source_sha256': {row['path']: row['sha256'] for row in source['package_sources']},
               'installed_files_sha256': {name: sha(data) for name, data in payload.items()}}
    return component, shared, source, module, wheel, binding, dist, rows


def bind(installation):
    component, shared, source, module, wheel, binding, _, _ = installation
    return shared.bind_official_wheel(source, module, wheel, binding,
                                      expected_wheel_sha256=component.wheel_pin(source, wheel))


def test_actual_metadata_and_full_payload_survive_common_admission(installation):
    component, shared, source, _, wheel, original, _, _ = installation
    binding = bind(installation)
    assert 'metadata' not in original
    component.validate_captured(binding, source, wheel)
    proof = shared.installation_proofs({'repo': source['repo']}, {source['repo']: source},
                                      [{'role': 'upstream-wheel', 'path': str(wheel), 'sha256': sha(wheel.read_bytes())}])
    shared.validate_installations({'downstream_distribution': binding}, proof)
    assert binding['distribution_version'] == source['upstream']['version']
    assert binding['installed_files_sha256'] == original['installed_files_sha256']


@pytest.mark.parametrize('change', ('missing-wheel', 'source-commit', 'wheel-bytes'))
def test_missing_or_changed_official_identity_is_rejected(installation, change):
    component, shared, source, _, wheel, _, _, _ = installation
    artifacts = [{'role': 'upstream-wheel', 'path': str(wheel), 'sha256': sha(wheel.read_bytes())}]
    if change == 'missing-wheel':
        artifacts = []
    elif change == 'source-commit':
        source['upstream']['commit'] = '0' * 40
    else:
        wheel.write_bytes(wheel.read_bytes() + b'changed official artifact')
        artifacts[0]['sha256'] = sha(wheel.read_bytes())
    with pytest.raises(ValueError):
        shared.installation_proofs({'repo': source['repo']}, {source['repo']: source}, artifacts)


@pytest.mark.parametrize('change', ('metadata', 'record', 'payload', 'claim', 'version', 'missing-metadata',
                                    'missing-record', 'metadata-prefix', 'extra-payload'))
def test_actual_existing_distribution_fields_cannot_be_replaced(installation, change):
    component, _, source, _, wheel, _, dist, _ = installation
    binding = copy.deepcopy(bind(installation))
    if change == 'metadata':
        (dist._path / 'METADATA').write_bytes(b'Name: counterfeit\n')
    elif change == 'record':
        (dist._path / 'RECORD').write_bytes(b'changed actual RECORD\n')
    elif change == 'payload':
        (Path(binding['package']) / 'resource.pt').write_bytes(b'changed resource')
    elif change == 'claim':
        binding['installed_files_sha256']['resource.pt'] = '0' * 64
    elif change == 'version':
        binding['distribution_version'] = '9.9'
    elif change == 'missing-metadata':
        del binding['metadata_sha256']
    elif change == 'missing-record':
        del binding['record_sha256']
    elif change == 'metadata-prefix':
        binding['metadata'] = str(Path(binding['package']).parent / 'other.dist-info/METADATA')
    else:
        extra = Path(binding['package']) / 'unbound.json'
        extra.write_bytes(b'{}')
        binding['installed_files_sha256']['unbound.json'] = sha(extra.read_bytes())
    with pytest.raises((ValueError, FileNotFoundError)):
        component.validate_captured(binding, source, wheel)


def test_coherent_payload_and_real_record_replacement_still_refuses(installation):
    component, _, source, _, wheel, _, dist, _ = installation
    original = bind(installation)
    record = dist._path / 'RECORD'
    record_original = record.read_bytes()
    names = ['resource.pt']
    if source['repo'] == 'tensordict':
        names += ['_version.py', '_C.so']
    for name in names:
        binding = copy.deepcopy(original)
        path = Path(binding['package']) / name
        before = path.read_bytes()
        after = before + b'\n# coherent replacement preserving the release version\n'
        path.write_bytes(after)
        binding['installed_files_sha256'][name] = sha(after)
        rows = list(csv.reader(io.StringIO(record_original.decode())))
        for row in rows:
            if row[0] == source['repo'] + '/' + name:
                row[1:] = ['sha256=' + base64.urlsafe_b64encode(
                    bytes.fromhex(sha(after))).decode().rstrip('='), str(len(after))]
        record.write_bytes(record_bytes(rows))
        binding['record_sha256'] = sha(record.read_bytes())
        # All captured and on-disk claims agree. The immutable official wheel
        # must remain the authority for resources, generated version and binary.
        with pytest.raises(ValueError, match='complete package payload'):
            component.validate_captured(binding, source, wheel)
        path.write_bytes(before)
        record.write_bytes(record_original)


def test_components_refuse_python_only_installation_claim(installation):
    _, _, source, module, _, _, _, _ = installation
    spec = importlib.util.spec_from_file_location('component_measure_missing_official', SCRIPTS / 'component_measure.py')
    measurement = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(measurement)
    with pytest.raises(ValueError, match='official-wheel'):
        measurement.installed_distribution(source, module)


def test_descriptor_keeps_only_real_official_generated_version(installation, tmp_path, monkeypatch):
    component, _, source, _, wheel, _, _, _ = installation
    import registry_inventory

    checkout = tmp_path / 'upstream'
    checkout.mkdir()
    (checkout / 'pyproject.toml').write_text(
        '[tool.setuptools_scm]\nversion_file = "tensordict/_version.py"\n', encoding='utf-8')
    monkeypatch.setattr(registry_inventory, 'inventory', lambda *_: copy.deepcopy(source))
    monkeypatch.setattr(component.subprocess, 'check_output', lambda *_args, **_kwargs: b'')
    descriptor = component.source_descriptor(source['repo'], checkout, wheel)
    assert descriptor['official_wheel']['sha256'] == sha(wheel.read_bytes())
    assert descriptor['official_wheel']['package_files_sha256']['resource.pt']
    if source['repo'] == 'tensordict':
        generated, = descriptor['generated_sources']
        assert generated['path'] == 'tensordict/_version.py'
        assert generated['version'] == '0.10.0' and generated['commit_id'] is None
        assert generated['declaration']['sha256'] == sha((checkout / 'pyproject.toml').read_bytes())
        (checkout / 'pyproject.toml').write_text(
            '[tool.setuptools_scm]\nversion_file = "different.py"\n', encoding='utf-8')
        with pytest.raises(ValueError, match='generated'):
            component.source_descriptor(source['repo'], checkout, wheel)
    else:
        assert descriptor['generated_sources'] == []


def test_measured_framework_hashes_use_immutable_git_blobs(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import _metric_td_installation as component

    root = SCRIPTS.parents[3]
    baseline = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    hashes = component.framework_snapshot(root, baseline=baseline)
    helper = 'agent/skills/torch-compat-repo-runbook/scripts/_runtime_platform.py'
    expected = subprocess.check_output(['git', '-C', str(root), 'show', baseline + ':' + helper])
    assert hashes[helper] == sha(expected)
    assert 'agent/skills/torch-compat-repo-runbook/scripts/training_probe.py' in hashes
    assert 'agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py' in hashes
    symlink = 'python/jittor/compat'
    assert hashes[symlink] == sha(subprocess.check_output(['git', '-C', str(root), 'show', baseline + ':' + symlink]))
    value = {'jittor_commit': baseline, 'measurement_source_sha256': hashes}
    component.validate_measurement_source(value, baseline)
    hashes[helper] = '0' * 64
    with pytest.raises(ValueError, match='committed'):
        component.validate_measurement_source(value, baseline)
    # The failed caller cannot poison the canonical immutable cached mapping.
    assert component.framework_snapshot(root, baseline=baseline)[helper] == sha(expected)


@pytest.fixture(params=('cpu', 'cuda'))
def runtime_packet(tmp_path, monkeypatch, request):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import _metric_td_installation as component

    # Framework bytes have their separate real-Git test above. This fixture
    # isolates runtime evidence, including genuine binary-file verification.
    monkeypatch.setattr(component, 'validate_measurement_source', lambda *_: None)
    binary = tmp_path / '_C.so'
    binary.write_bytes(b'host-only identity fixture; not loaded')
    device = request.param
    precision = {'matmul': False, 'cudnn': False, 'cudnn_benchmark': False,
                 'matmul_precision': 'highest'}
    platform = {'precision': precision, 'card_count': int(device == 'cuda'),
                'card_indices': ['3'] if device == 'cuda' else [],
                'physical_devices': [{'index': '3', 'uuid': 'GPU-observed'}] if device == 'cuda' else []}
    observation = {'device': device, 'dispatch': device,
                   'location': 'device' if device == 'cuda' else 'cpu', 'pointer_device': 0}
    packet = {'runtime': 'jittor', 'device': device,
              'interpreter': {'is_shim': True, 'pid': 100},
              'runtime_environment': {key: str(tmp_path / key) for key in
                                      ('HOME', 'JITTOR_HOME', 'TMPDIR', 'XDG_CACHE_HOME')},
              'platform': platform, 'tf32': precision, 'tensor_observations': [observation],
              'fallback_policy': 'error', 'fallback_count': 0,
              'native_runtime_flags': {'use_cuda': int(device == 'cuda')},
              'device_controls': {'cuda_visible_devices': '3' if device == 'cuda' else '',
                                  'cuda_available': device == 'cuda',
                                  'cuda_device': {'uuid': 'GPU-observed'}}}
    return component, packet, binary


def test_runtime_provenance_accepts_actual_cpu_or_allocated_cuda(runtime_packet):
    component, packet, binary = runtime_packet
    component.validate_runtime_metadata(packet, 'baseline', packet['platform'])
    oracle = copy.deepcopy(packet)
    oracle.update(runtime='torch', interpreter={'is_shim': False, 'pid': 101,
                  'binary_extension': str(binary), 'binary_sha256': sha(binary.read_bytes()),
                  'torch_version': '2.11.0'})
    expected = dict(packet['platform'], oracle={'is_shim': False, 'torch_version': '2.11.0',
                                               'binary_extension': str(binary)})
    component.validate_runtime_metadata(oracle, 'baseline', expected)
    binary.write_bytes(b'changed actual oracle extension')
    with pytest.raises(ValueError, match='binary'):
        component.validate_runtime_metadata(oracle, 'baseline', expected)


@pytest.mark.parametrize('change', ('shim', 'pid', 'home', 'precision', 'device', 'fallback',
                                    'native-device', 'residency', 'visibility', 'record-platform'))
def test_missing_or_contradictory_runtime_evidence_is_rejected(runtime_packet, change):
    component, packet, _ = runtime_packet
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
        component.validate_runtime_metadata(packet, 'baseline', expected)


@pytest.mark.parametrize('change', ('uuid', 'pointer'))
def test_cuda_allocation_and_actual_pointer_must_agree(runtime_packet, change):
    component, packet, _ = runtime_packet
    if packet['device'] == 'cpu':
        return  # CPU has no CUDA pointer; its explicit-hide cases are above.
    if change == 'uuid':
        packet['device_controls']['cuda_device']['uuid'] = 'GPU-not-allocated'
    else:
        packet['tensor_observations'][0]['pointer_device'] = 1
    with pytest.raises(ValueError, match='physical CUDA'):
        component.validate_runtime_metadata(packet, 'baseline')

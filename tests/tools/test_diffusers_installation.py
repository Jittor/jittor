"""Synthetic host payloads exercise proof contracts, never model execution."""

import base64
import copy
import csv
import hashlib
import io
from pathlib import Path
import subprocess
from types import SimpleNamespace
import zipfile

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / 'agent/skills/torch-compat-repo-runbook/scripts'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def receipt(payload):
    result = io.StringIO()
    csv.writer(result).writerows([[name, 'sha256=' + base64.urlsafe_b64encode(
        hashlib.sha256(data).digest()).decode().rstrip('='), str(len(data))]
                                 for name, data in payload.items()])
    return result.getvalue().encode()


@pytest.fixture
def installation(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import _diffusers_installation as component
    import _installed_source_proof as shared

    package = tmp_path / 'site/diffusers'
    package.mkdir(parents=True)
    payload = {'diffusers/__init__.py': b"__version__ = '0.35.1'\n",
               'diffusers/resource.bin': b'host fixture only; not an imported binary'}
    for name, data in payload.items():
        (package.parent / name).write_bytes(data)
    distribution = package.parent / 'diffusers-0.35.1.dist-info'
    distribution.mkdir()
    header = b'Metadata-Version: 2.1\nName: diffusers\nVersion: 0.35.1\n'
    (distribution / 'METADATA').write_bytes(header)
    (distribution / 'RECORD').write_bytes(receipt(payload))
    wheel = tmp_path / 'diffusers.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
        archive.writestr(distribution.name + '/METADATA', header)
        archive.writestr(distribution.name + '/RECORD', receipt(payload))
    monkeypatch.setattr(component, 'WHEEL_SHA256', sha(wheel.read_bytes()))
    known = [{'path': 'src/' + name, 'sha256': sha(data)} for name, data in payload.items()]
    known += [{'path': 'src/diffusers/' + name, 'sha256': sha(name.encode())}
              for name in sorted(component.OMITTED_DOCUMENTATION)]
    source = {'repo': 'diffusers', 'upstream': {'version': '0.35.1', 'commit': component.COMMIT,
               'remote': 'https://github.com/huggingface/diffusers.git'},
              'package_sources': known[:1], 'package_files': known,
              'official_wheel': {'path': str(wheel), 'sha256': sha(wheel.read_bytes())}}
    module = SimpleNamespace(__file__=str(package / '__init__.py'), __path__=[str(package)],
                             __version__='0.35.1')
    dist = shared.metadata.PathDistribution(distribution)
    monkeypatch.setattr(shared.metadata, 'distribution', lambda _: dist)
    binding = {'repo': 'diffusers', 'version': '0.35.1', 'package': str(package),
               'source_sha256': {row['path']: row['sha256'] for row in source['package_sources']},
               'installed_files_sha256': {name.removeprefix('diffusers/'): sha(data)
                                          for name, data in payload.items()}}
    return component, shared, source, module, wheel, binding, dist


def bound(fixture):
    component, shared, source, module, wheel, binding, _ = fixture
    return shared.bind_official_wheel(source, module, wheel, binding,
                                      expected_wheel_sha256=component.wheel_pin(source, wheel))


def test_actual_receipt_package_and_wheel_survive_common_proof(installation):
    component, shared, source, _, wheel, original, _ = installation
    value = bound(installation)
    assert 'metadata' not in original
    component.validate_captured(value, source, wheel)
    proof = shared.installation_proofs({'repo': 'diffusers'}, {'diffusers': source},
                                       [{'role': 'upstream-wheel', 'path': str(wheel),
                                         'sha256': sha(wheel.read_bytes())}])
    shared.validate_installations({'downstream_distribution': value}, proof)
    assert value['distribution_name'] == 'diffusers'
    assert value['distribution_version'] == '0.35.1'


@pytest.mark.parametrize('change', ('commit', 'version', 'remote', 'artifact'))
def test_wrong_official_identity_is_refused(installation, change):
    component, _, source, _, wheel, _, _ = installation
    source = copy.deepcopy(source)
    if change == 'artifact':
        wheel.write_bytes(wheel.read_bytes() + b'changed input')
    else:
        source['upstream'][change] = 'unverified'
    with pytest.raises(ValueError):
        component.wheel_pin(source, wheel)


@pytest.mark.parametrize('change', ('unbound.py', 'unbound.json', 'missing-runtime', 'resource'))
def test_coherent_wheel_cannot_add_or_change_unbound_payload(installation, monkeypatch, change):
    component, _, source, _, wheel, _, _ = installation
    with zipfile.ZipFile(wheel) as archive:
        payload = {name: archive.read(name) for name in archive.namelist()}
    if change.startswith('unbound.'):
        payload['diffusers/' + change] = b'unbound upstream payload'
    elif change == 'missing-runtime':
        del payload['diffusers/__init__.py']
    else:
        payload['diffusers/resource.bin'] = b'coherent changed resource'
    with zipfile.ZipFile(wheel, 'w') as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
    # Even an updated artifact authority cannot authorize untracked package
    # contents or omit runtime source. Production retains the fixed PyPI pin.
    monkeypatch.setattr(component, 'WHEEL_SHA256', sha(wheel.read_bytes()))
    with pytest.raises(ValueError, match='payload|omissions'):
        component.wheel_pin(source, wheel)


@pytest.mark.parametrize('change', ('metadata', 'record', 'payload', 'receipt-size', 'missing-binding'))
def test_actual_installation_or_captured_receipt_tampering_is_refused(installation, change):
    component, _, source, module, wheel, _, dist = installation
    value = bound(installation)
    if change in ('metadata', 'record'):
        path = Path(dist._path) / change.upper()
        path.write_bytes(path.read_bytes() + b'changed\n')
    elif change == 'payload':
        (Path(module.__file__).parent / 'resource.bin').write_bytes(b'changed installed resource')
    elif change == 'receipt-size':
        path = Path(dist._path) / 'RECORD'
        rows = list(csv.reader(path.read_text().splitlines()))
        rows[0][2] = '999'
        out = io.StringIO()
        csv.writer(out).writerows(rows)
        path.write_text(out.getvalue())
    else:
        value.pop('metadata')
    with pytest.raises(ValueError):
        component.validate_captured(value, source, wheel)


def test_source_descriptor_retains_documentation_omissions_and_rejects_dirty_checkout(
        installation, tmp_path, monkeypatch):
    component, _, source, _, wheel, _, _ = installation
    import registry_inventory

    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    subprocess.run(['git', '-C', str(checkout), 'init', '-q'], check=True)
    monkeypatch.setattr(registry_inventory, 'inventory', lambda *_: copy.deepcopy(source))
    value = component.source_descriptor(checkout, wheel)
    assert len(value['excluded_package_sources']) == 6
    (checkout / 'untracked').write_text('dirty source')
    with pytest.raises(ValueError, match='Dirty'):
        component.source_descriptor(checkout, wheel)


def test_runtime_source_validator_defaults_to_existing_metric_contract(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import _metric_td_installation as component

    calls = []
    monkeypatch.setattr(component, 'validate_measurement_source', lambda value, baseline: calls.append(('default', value, baseline)))
    with pytest.raises(ValueError, match='runtime/device'):
        component.validate_runtime_metadata({}, 'a' * 40)
    assert calls == [('default', {}, 'a' * 40)]
    with pytest.raises(ValueError, match='runtime/device'):
        component.validate_runtime_metadata({}, 'b' * 40,
            source_validator=lambda value, baseline: calls.append(('explicit', value, baseline)))
    assert calls[-1] == ('explicit', {}, 'b' * 40)

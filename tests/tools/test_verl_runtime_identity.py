"""Host metadata contracts; real independent-prefix failure is retained externally."""

import copy
import importlib
from pathlib import Path
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'agent/skills/torch-compat-repo-runbook/scripts'


@pytest.fixture
def contract(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'compat/tests/torch'))
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module('_verl_component_contract')


@pytest.fixture
def metadata_pair(contract, tmp_path):
    versions = {'verl': '0.10.0.dev', 'numpy': '2.2.6', 'transformers': '5.5.3',
                'ray': '2.54.1', 'tensordict': '0.10.0'}
    values = []
    for side in ('oracle', 'candidate'):
        dependencies = {}
        for name, version in versions.items():
            package = tmp_path / side / name
            package.mkdir(parents=True)
            origin = package / '__init__.py'
            origin.write_text('__version__ = %r\n' % version)
            (package / 'payload.bin').write_bytes(name.encode())
            dependencies[name] = {'version': version, 'origin': str(origin),
                                  'installed_files_sha256': contract.package_files(package)}
        values.append({'dependencies': dependencies, 'device': 'cpu', 'tf32': {'enabled': False},
                       'runtime_conditions': {'precision': {'enabled': False}, 'threads': 2}})
    reference_package = Path(values[0]['dependencies']['verl']['origin']).parent
    payload = contract.package_files(reference_package)
    source = {'repo': 'verl', 'upstream': {'version': versions['verl']},
              'package_sources': [{'path': 'verl/__init__.py', 'sha256': payload['__init__.py']}],
              'package_files': [{'path': 'verl/' + name, 'sha256': digest} for name, digest in payload.items()]}
    wheel = tmp_path / 'verl.whl'
    with zipfile.ZipFile(wheel, 'w') as stream:
        for name in payload:
            stream.writestr('verl/' + name, (reference_package / name).read_bytes())
        stream.writestr('verl-0.10.0.dev0.dist-info/METADATA', 'Name: verl\nVersion: 0.10.0.dev0\n')
    proof = contract.installation_proofs({'repo': 'verl', 'registry_refs': []}, {'verl': source},
            [{'role': 'upstream-wheel', 'path': str(wheel), 'sha256': contract.digest(wheel)}])
    for value in values:
        entry = value['dependencies']['verl']
        value['downstream_distribution'] = {'repo': 'verl', 'version': versions['verl'],
            'distribution_version': '0.10.0.dev0', 'package': str(Path(entry['origin']).parent),
            'source_sha256': {'verl/__init__.py': payload['__init__.py']},
            'installed_files_sha256': dict(entry['installed_files_sha256']), 'wheel_sha256': contract.digest(wheel)}
    return values, proof


def checked(contract, value, proof):
    # compare_pair runs this proof validation before the runtime identity check.
    contract.validate_installations(value, proof)
    return contract.runtime_identity(value)


def test_distinct_prefixes_preserve_raw_locations_and_compare_full_identity(contract, metadata_pair):
    values, proof = metadata_pair
    original = copy.deepcopy(values)
    assert values[0]['dependencies']['verl']['origin'] != values[1]['dependencies']['verl']['origin']
    assert checked(contract, values[0], proof) == checked(contract, values[1], proof)
    assert values == original


@pytest.mark.parametrize('side', (0, 1))
@pytest.mark.parametrize('mutation', ('missing_origin', 'empty_origin', 'relative_origin', 'nonexistent_origin',
    'wrong_package_origin', 'payload_bytes', 'recorded_payload', 'missing_package', 'wrong_package_binding',
    'version', 'source', 'wheel', 'metadata_version', 'unknown_dependency_field', 'unknown_distribution_field',
    'missing_dependency', 'extra_dependency'))
def test_location_normalization_cannot_hide_content_or_identity_changes(contract, metadata_pair, side, mutation):
    values, proof = metadata_pair
    value = values[side]
    entry = value['dependencies']['verl']
    binding = value['downstream_distribution']
    if mutation == 'missing_origin':
        entry.pop('origin')
    elif mutation == 'empty_origin':
        entry['origin'] = ''
    elif mutation == 'relative_origin':
        entry['origin'] = 'relative/__init__.py'
    elif mutation == 'nonexistent_origin':
        entry['origin'] += '.missing'
    elif mutation == 'wrong_package_origin':
        entry['origin'] = value['dependencies']['transformers']['origin']
    elif mutation == 'payload_bytes':
        (Path(entry['origin']).parent / 'payload.bin').write_bytes(b'modified actual binary payload')
    elif mutation == 'recorded_payload':
        entry['installed_files_sha256']['payload.bin'] = '0' * 64
    elif mutation == 'missing_package':
        binding.pop('package')
    elif mutation == 'wrong_package_binding':
        binding['package'] = values[1 - side]['downstream_distribution']['package']
    elif mutation == 'version':
        value['dependencies']['ray']['version'] = 'changed'
    elif mutation == 'source':
        binding['source_sha256']['verl/__init__.py'] = '0' * 64
    elif mutation == 'wheel':
        binding['wheel_sha256'] = '0' * 64
    elif mutation == 'metadata_version':
        binding['distribution_version'] = '0.10.0.dev99'
    elif mutation == 'unknown_dependency_field':
        value['dependencies']['ray']['new_identity'] = {'value': 7}
    elif mutation == 'unknown_distribution_field':
        binding['new_identity'] = {'value': 7}
    elif mutation == 'missing_dependency':
        value['dependencies'].pop('numpy')
    else:
        value['dependencies']['unbound'] = copy.deepcopy(entry)
    with pytest.raises((ValueError, AssertionError)):
        assert checked(contract, values[0], proof) == checked(contract, values[1], proof)


@pytest.mark.parametrize('mutation', ('source', 'wheel', 'metadata_version'))
def test_matching_forged_primary_proofs_still_fail(contract, metadata_pair, mutation):
    values, proof = metadata_pair
    for value in values:
        binding = value['downstream_distribution']
        if mutation == 'source':
            binding['source_sha256']['verl/__init__.py'] = '0' * 64
        elif mutation == 'wheel':
            binding['wheel_sha256'] = '0' * 64
        else:
            binding['distribution_version'] = '0.10.0.dev99'
    for value in values:
        with pytest.raises(ValueError):
            checked(contract, value, proof)


def test_optional_distribution_binding_must_belong_to_its_own_prefix(contract, metadata_pair):
    values, proof = metadata_pair
    for value in values:
        entry = value['dependencies']['transformers']
        value['downstream_distributions'] = {'transformers': {'repo': 'transformers',
            'version': entry['version'], 'package': str(Path(entry['origin']).parent),
            'installed_files_sha256': dict(entry['installed_files_sha256']), 'new_identity': 11}}
    assert checked(contract, values[0], proof) == checked(contract, values[1], proof)
    values[1]['downstream_distributions']['transformers']['package'] = (
        values[0]['downstream_distributions']['transformers']['package'])
    with pytest.raises(ValueError, match='binding differs'):
        checked(contract, values[1], proof)


def test_unknown_fields_survive_and_plural_presence_is_not_dropped(contract, metadata_pair):
    values, proof = metadata_pair
    values[0]['dependencies']['ray']['new_identity'] = {'value': 7}
    identity = checked(contract, values[0], proof)
    assert identity['dependencies']['ray']['new_identity'] == {'value': 7}
    values[1]['dependencies']['ray']['new_identity'] = {'value': 7}
    assert checked(contract, values[0], proof) == checked(contract, values[1], proof)
    values[1]['downstream_distributions'] = {}
    assert checked(contract, values[0], proof) != checked(contract, values[1], proof)


@pytest.mark.parametrize('binding', (None, [], 'unbound'))
def test_present_optional_binding_cannot_be_unbound(contract, metadata_pair, binding):
    values, proof = metadata_pair
    values[0]['downstream_distributions'] = {'ray': binding}
    with pytest.raises(ValueError, match='Malformed VERL dependency distribution binding'):
        checked(contract, values[0], proof)

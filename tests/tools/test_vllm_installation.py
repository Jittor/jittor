"""Actual wheel, package and pip-rewritten RECORD provenance; no tensor runtime."""

import copy
from pathlib import Path

import pytest

from _helpers.vllm_installation import contract, sha
from _helpers.vllm_installation import installation as installed_fixture

@pytest.fixture
def installation(tmp_path):
    with installed_fixture(tmp_path) as value:
        yield value


def test_valid_pip_record_keeps_both_original_artifacts(installation):
    source, module, wheel, binding, dist, _, _ = installation
    result = contract.bind_installation(source, module, wheel, binding)
    assert result['wheel_sha256'] == sha(wheel.read_bytes())
    assert result['record_sha256'] == sha((dist._path / 'RECORD').read_bytes())
    assert result['metadata_sha256'] == sha((dist._path / 'METADATA').read_bytes())
    assert result['distribution_version'] == '0.29.0'
    assert result['installed_files_sha256'] == binding['installed_files_sha256']
    assert 'wheel_sha256' not in binding


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'wrong_hash', 'wrong_size',
                                    'extra_package', 'unknown_cache', 'cache_hash', 'malformed'])
def test_record_rejects_incomplete_or_changed_package_scope(installation, change):
    source, module, wheel, binding, dist, rows, record = installation
    rows = copy.deepcopy(rows)
    if change == 'missing':
        rows.pop()
    elif change == 'duplicate':
        rows.append(rows[0])
    elif change == 'wrong_hash':
        rows[0][1] = 'sha256=' + 'x' * 43
    elif change == 'wrong_size':
        rows[0][2] = '999'
    elif change == 'extra_package':
        rows.append(['vllm/unbound.py', '', ''])
    elif change == 'unknown_cache':
        rows.append(['vllm/__pycache__/unbound.cpython-311.pyc', '', ''])
    elif change == 'cache_hash':
        rows.append(['vllm/__pycache__/__init__.cpython-311.pyc', 'changed', ''])
    else:
        rows.append(['invalid'])
    (dist._path / 'RECORD').write_bytes(record(rows))
    with pytest.raises(ValueError, match='RECORD'):
        contract.bind_installation(source, module, wheel, binding)


@pytest.mark.parametrize('change', ['metadata', 'version', 'name', 'metadata_root', 'package_root',
                                    'extra_search_root', 'source', 'payload', 'commit', 'wheel'])
def test_distribution_and_source_identity_are_independently_required(installation, change):
    source, module, wheel, binding, dist, _, _ = installation
    if change == 'metadata':
        (dist._path / 'METADATA').write_bytes(b'Name: counterfeit\n')
    elif change == 'version':
        dist.version = '0.29.1'
    elif change == 'name':
        dist.metadata['Name'] = 'other'
    elif change == 'metadata_root':
        dist.locate_file = lambda name: Path('/other') / name
    elif change == 'package_root':
        binding['package'] += '-wrong'
    elif change == 'extra_search_root':
        module.__path__.append('/unbound')
    elif change == 'source':
        binding['source_sha256']['vllm/__init__.py'] = '0' * 64
    elif change == 'commit':
        source['upstream']['commit'] = '0' * 40
    elif change == 'wheel':
        contract.WHEEL_SHA256 = '0' * 64
    else:
        binding['installed_files_sha256']['unbound.py'] = '0' * 64
    with pytest.raises(ValueError):
        contract.bind_installation(source, module, wheel, binding)

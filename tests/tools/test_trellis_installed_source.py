"""Real file-byte proof and corrupt phase negatives; no tensor runtime needed."""

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

from _helpers.child_process import run_python_child

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'agent/skills/trellis-torch-compat/scripts'
COMMON = ROOT / 'agent/skills/torch-compat-repo-runbook/scripts'
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(COMMON))
from _source_binding import source_installation, validate_phase  # noqa: E402


@pytest.fixture
def installed(tmp_path):
    files = {'trellis2/__init__.py': b'# real file bytes, not executed\n',
             'trellis2/models/public.py': b'class Public: pass\n',
             'trellis2/data/schema.json': b'{"schema": 1}\n'}
    rows = []
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        rows.append({'path': name, 'sha256': hashlib.sha256(content).hexdigest()})
    source = {'repo': 'trellis', 'upstream': {'commit': 'a' * 40, 'version': 'source@' + 'a' * 40},
              'package_sources': [row for row in rows if row['path'].endswith('.py')],
              'package_files': rows}
    return source, tmp_path / 'trellis2/__init__.py'


def test_actual_source_checkout_payload_has_no_invented_release(installed):
    source, origin = installed
    binding = source_installation(source, origin)
    assert binding['version'] == source['upstream']['version']
    assert 'distribution_version' not in binding and 'wheel_sha256' not in binding
    assert len(binding['source_sha256']) == 2 and len(binding['installed_files_sha256']) == 3
    validate_phase({'downstream_distribution': binding, 'trellis_origin': str(origin)}, source)


@pytest.mark.parametrize('mutation', ('change', 'omit', 'extra_python', 'extra_data', 'escape'))
def test_source_installation_rejects_changed_or_partial_actual_files(installed, mutation):
    source, origin = installed
    package = origin.parent
    if mutation == 'change':
        (package / 'models/public.py').write_text('class Wrong: pass\n')
    elif mutation == 'omit':
        (package / 'data/schema.json').unlink()
    elif mutation.startswith('extra'):
        (package / ('untracked.py' if mutation == 'extra_python' else 'untracked.json')).write_text('extra')
    else:
        outside = package.parent / 'outside.py'
        outside.write_text('external')
        (package / 'escape.py').symlink_to(outside)
    with pytest.raises(ValueError, match='installation differs|escapes'):
        source_installation(source, origin)


@pytest.mark.parametrize('mutation', ('missing', 'version', 'origin', 'source', 'payload', 'extra_payload', 'kind'))
def test_each_phase_requires_independent_complete_actual_binding(installed, mutation):
    source, origin = installed
    binding = source_installation(source, origin)
    value = {'downstream_distribution': copy.deepcopy(binding), 'trellis_origin': str(origin)}
    actual = value['downstream_distribution']
    if mutation == 'missing':
        value.pop('downstream_distribution')
    elif mutation == 'version':
        actual['version'] = '1.0'
    elif mutation == 'origin':
        value['trellis_origin'] = '/another/trellis2/__init__.py'
    elif mutation == 'source':
        actual['source_sha256'].pop('trellis2/__init__.py')
    elif mutation == 'payload':
        actual['installed_files_sha256'].pop('data/schema.json')
    elif mutation == 'extra_payload':
        actual['installed_files_sha256']['extra.dat'] = 'b' * 64
    else:
        actual['installation_kind'] = 'official-pypi'
    with pytest.raises(ValueError):
        validate_phase(value, source)


def test_source_proof_runs_without_site_packages_or_framework_imports(installed):
    source, origin = installed
    snippet = ('import json,sys; sys.path[:0] = ' + repr([str(SCRIPTS), str(COMMON)]) + '; '
               'from _source_binding import source_installation,validate_phase; '
               'source=json.loads(' + repr(json.dumps(source)) + '); '
               'binding=source_installation(source,' + repr(str(origin)) + '); '
               'validate_phase({"downstream_distribution":binding,"trellis_origin":binding["origin"]},source); '
               'assert not {"torch","jittor","numpy","packaging"}.intersection(sys.modules); print("source-only passed")')
    result = run_python_child(['-S', '-c', snippet], without_torch_mode=True)
    assert result.returncode == 0, result.stdout
    assert 'source-only passed' in result.stdout

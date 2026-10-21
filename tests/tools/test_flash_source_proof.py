"""Real local wheel bytes and distinct implementation identities stay fail-closed."""

import base64
import csv
import hashlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest


@pytest.fixture
def proof(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / 'agent/skills/torch-compat-repo-runbook/scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('flash_source_proof_test', scripts / '_flash_attention_scorecard.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_wheel(path, *, corrupt_record=False, omit_native=False, payload_change=None):
    files = {'flash_attn/__init__.py': b'__version__ = "2.7.4.post1"\n',
             'flash_attn/ops/child.py': b'VALUE = 1\n',
             'flash_attn/declared_data.json': b'{"fixture_resource": true}\n',
             'flash_attn-2.7.4.post1.dist-info/METADATA': b'Name: flash_attn\nVersion: 2.7.4.post1\n'}
    if payload_change == 'add':
        files['flash_attn/untracked_payload.json'] = b'{"untracked": true}\n'
    elif payload_change == 'remove':
        del files['flash_attn/declared_data.json']
    if not omit_native:
        files['flash_attn_2_cuda.test.so'] = b'fixture binary payload; never loaded'
    rows = []
    for name, content in files.items():
        encoded = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip('=')
        rows.append([name, 'sha256=' + encoded, str(len(content))])
    record = 'flash_attn-2.7.4.post1.dist-info/RECORD'
    if corrupt_record:
        rows[0][1] = 'sha256=' + 'x' * 43
    rows.append([record, '', ''])
    output = io.StringIO()
    csv.writer(output, lineterminator='\n').writerows(rows)
    files[record] = output.getvalue().encode()
    with zipfile.ZipFile(path, 'w') as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return files


@pytest.mark.parametrize('change', ('add', 'remove'))
def test_coherent_wheel_prefix_record_repacking_cannot_change_fixed_artifact(proof, tmp_path, monkeypatch, change):
    original = tmp_path / 'original.whl'
    make_wheel(original)
    monkeypatch.setattr(proof, 'LOCAL_FLASH_WHEEL_SHA256', proof.digest(original))
    assert proof.verified_flash_wheel(original)['sha256'] == proof.digest(original)
    variant = tmp_path / 'variant.whl'
    make_wheel(variant, payload_change=change)
    members = proof.wheel_files(variant)  # The changed RECORD remains complete and valid.
    prefix = tmp_path / 'installed-variant'
    with zipfile.ZipFile(variant) as archive:
        archive.extractall(prefix)
    assert len(proof.verify_wheel_installation({'wheel_members': members}, prefix)) == len(members)
    with pytest.raises(ValueError, match='fixed artifact'):
        proof.verified_flash_wheel(variant)


def test_complete_record_and_real_installed_payload(proof, tmp_path):
    wheel = tmp_path / 'local.whl'
    files = make_wheel(wheel)
    hashes = proof.wheel_files(wheel)
    assert set(hashes) == set(files)
    prefix = tmp_path / 'installed'
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(prefix)
    source = {'wheel_members': hashes}
    assert len(proof.verify_wheel_installation(source, prefix)) == len(files)
    metadata = prefix / 'flash_attn-2.7.4.post1.dist-info/METADATA'
    assert len(proof.carrier_record_payload(metadata)) == len(files)
    (prefix / 'flash_attn/ops/child.py').write_bytes(b'VALUE = 2\n')
    with pytest.raises(ValueError, match='complete RECORD payload'):
        proof.verify_wheel_installation(source, prefix)
    with pytest.raises(ValueError, match='carrier RECORD mismatch'):
        proof.carrier_record_payload(metadata)


def test_corrupt_record_rejected_before_installation(proof, tmp_path):
    wheel = tmp_path / 'local.whl'
    make_wheel(wheel, corrupt_record=True)
    with pytest.raises(ValueError, match='RECORD mismatch'):
        proof.wheel_files(wheel)


def test_record_cannot_omit_native_member(proof, tmp_path):
    wheel = tmp_path / 'local.whl'
    make_wheel(wheel)
    with zipfile.ZipFile(wheel, 'a') as archive:
        archive.writestr('unrecorded.so', b'new native code')
    with pytest.raises(ValueError, match='Incomplete wheel RECORD'):
        proof.wheel_files(wheel)


def test_selected_provider_uses_actual_metadata_precedence(proof, monkeypatch):
    first = SimpleNamespace(metadata={'Name': 'nvidia-cuda-runtime-cu12'}, version='12.8.90')
    last = SimpleNamespace(metadata={'Name': 'nvidia-cuda-runtime-cu12'}, version='12.2.140')
    monkeypatch.setattr(proof.md, 'distributions', lambda: [first, last])
    monkeypatch.setattr(proof.md, 'version', lambda name: first.version)
    assert proof.selected_nvidia_wheels() == {'nvidia-cuda-runtime-cu12': '12.8.90'}


def test_platform_dependency_does_not_invent_an_absent_record(proof, monkeypatch, tmp_path):
    name = 'nvidia-cuda-runtime-cu12'
    distribution = SimpleNamespace(metadata={'Name': name}, version='12.8.90', _path=tmp_path)
    monkeypatch.setattr(proof.md, 'distributions', lambda: [distribution])
    monkeypatch.setattr(proof.md, 'version', lambda unused: distribution.version)
    monkeypatch.setattr(proof.md, 'distribution', lambda unused: distribution)
    (tmp_path / 'METADATA').write_text('Name: ' + name + '\nVersion: 12.8.90\n')
    observed = proof.selected_nvidia_metadata()[name]
    assert observed['record'] is None and observed['record_present'] is False
    record = tmp_path / 'RECORD'
    record.write_text('actual existing provider record\n')
    observed = proof.selected_nvidia_metadata()[name]
    assert observed['record']['sha256'] == proof.digest(record) and observed['record_present'] is True
    (tmp_path / 'METADATA').unlink()
    with pytest.raises(FileNotFoundError):
        proof.selected_nvidia_metadata()


@pytest.mark.parametrize('change', ('world', 'shape', 'causal', 'dtype', 'api', 'extra', 'boolean_integer'))
def test_unmeasured_protocol_cannot_borrow_original_scope(proof, change):
    spec = proof.protocol('flash_attn_func', 'float16')
    if change == 'world':
        spec['world_size'] = 2
    elif change == 'shape':
        spec['shape'][1] = 12
    elif change == 'causal':
        spec['causal'] = True
    elif change == 'dtype':
        spec['precision'] = 'float32'
    elif change == 'api':
        spec['api'] = 'flash_attn_with_kvcache'
    elif change == 'boolean_integer':
        spec['world_size'] = True
    else:
        spec['unmeasured'] = 'extension'
    record = {'record_kind': 'protocol', 'repo': 'flash-attention', 'registry_refs': [], 'protocol': spec}
    with pytest.raises(ValueError, match='Unsupported'):
        proof.validate_spec(record, {'repo': 'flash-attention', 'axes': []})


def test_protocol_dispatch_is_exact_and_registry_stays_separate(proof):
    import _protocol_scorecards as dispatcher
    source = {'repo': 'flash-attention', 'axes': []}
    record = {'repo': 'flash-attention', 'record_kind': 'protocol', 'registry_refs': [],
              'protocol': proof.protocol('flash_attn_func', 'bfloat16')}
    assert dispatcher.validator(record, source).__name__ == '_flash_attention_scorecard'
    for defect in ('registry', 'other_id', 'other_repo'):
        changed = dict(record, protocol=dict(record['protocol']))
        if defect == 'registry':
            changed['record_kind'] = 'registry'
        elif defect == 'other_id':
            changed['protocol']['id'] = 'unmaintained-flash'
        else:
            changed['repo'] = 'transformers'
        with pytest.raises(ValueError):
            dispatcher.validator(changed, source)


@pytest.mark.parametrize('configuration', (None, {}, {'path': '/nonexistent/flash-evidence-configuration.json'}))
def test_malformed_source_evidence_is_a_failed_admission(proof, configuration):
    record = {'record_kind': 'protocol', 'repo': 'flash-attention', 'registry_refs': [],
              'protocol': proof.protocol('flash_attn_func', 'float16')}
    with pytest.raises(ValueError, match='Invalid FlashAttention evidence'):
        proof.validate_spec(record, {'repo': 'flash-attention', 'axes': [], 'configuration': configuration})


def test_external_native_file_tampering_is_rejected(proof, tmp_path):
    native = tmp_path / 'native.so'
    native.write_bytes(b'original')
    binding = {str(native): proof.digest(native)}
    proof.recheck(binding)
    native.write_bytes(b'changed')
    with pytest.raises(ValueError, match='external file changed'):
        proof.recheck(binding)


def test_compiler_headers_include_nested_real_inputs_and_detect_changes(proof, tmp_path):
    nested = tmp_path / 'cuda/nested'
    nested.mkdir(parents=True)
    source = nested / 'runtime.cuh'
    source.write_text('using actual_header_type = float;\n')
    original = proof.compiler_headers([tmp_path])
    assert set(original) == {str(source)}
    source.write_text('using actual_header_type = double;\n')
    assert proof.compiler_headers([tmp_path]) != original
    with pytest.raises(ValueError, match='Missing actual compiler include directory'):
        proof.compiler_headers([tmp_path / 'absent'])


def test_real_public_deployment_metadata_is_owned_compatibility_surface(proof, monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('flash_public_deployment', root / 'compat/shim/deploy.py')
    deployment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deployment)
    deployment.deploy(tmp_path, resource_root=root / 'compat/shim/resources')
    monkeypatch.syspath_prepend(str(tmp_path))
    value = proof.facade_metadata(root, tmp_path / 'flash_attn')
    assert value['kind'] == 'owned facade compatibility metadata; not an upstream wheel installation'
    assert set(value['files']) == {'METADATA', 'top_level.txt'}
    metadata = Path(value['directory']) / 'METADATA'
    metadata.write_text(metadata.read_text().replace('compatibility surface', 'unrecorded replacement'))
    with pytest.raises(ValueError, match='differs from owned resources'):
        proof.facade_metadata(root, tmp_path / 'flash_attn')

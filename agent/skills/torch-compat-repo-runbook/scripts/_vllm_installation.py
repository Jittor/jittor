"""Bind the actual vLLM installation to its clean source and official wheel."""

import base64
import csv
import hashlib
from importlib import metadata
from importlib.util import source_from_cache
import io
import os
from pathlib import Path
import subprocess
import zipfile

from _installed_source_proof import installation_proofs, require, validate_installations

UPSTREAM_COMMIT = '98dff2a81d747d1dba01a47f939f48c3526d4206'
WHEEL_SHA256 = '09d48617fc2be9c6cdcd5db480651ab0d84817b257204f2cc2e3ecbb70bbb635'


def framework_sources(root):
    root = Path(root)
    tracked = subprocess.check_output(
        ['git', '-C', str(root), 'ls-files', 'python', 'compat', 'src', 'backends', 'adapters'], text=True).splitlines()
    require(tracked, 'Missing complete vLLM framework source inventory')
    tracked += ['noxfile.py', 'agent/skills/torch-compat-repo-runbook/scripts/_vllm_installation.py',
                'agent/skills/torch-compat-repo-runbook/scripts/_vllm_component_contract.py',
                'agent/skills/torch-compat-repo-runbook/scripts/component_measure.py',
                'agent/skills/torch-compat-repo-runbook/scripts/_installed_source_proof.py',
                'agent/skills/torch-compat-repo-runbook/scripts/registry_inventory.py',
                'agent/skills/torch-compat-repo-runbook/scripts/collect_vllm_scorecard.py',
                'tests/_helpers/artifacts.py']
    return sorted(set(tracked))


def device_controls(torch, device):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    result = {'cuda_visible_devices': visible, 'cuda_available': bool(torch.cuda.is_available())}
    if device == 'cpu':
        require(visible == '' and not result['cuda_available'], 'vLLM CPU must explicitly hide CUDA')
        return result
    import ctypes
    import uuid

    require(result['cuda_available'] and visible and ',' not in visible, 'vLLM needs one explicitly allocated CUDA device')
    driver = ctypes.CDLL('libcuda.so.1')
    ordinal, identifier = ctypes.c_int(), ctypes.create_string_buffer(16)
    require(driver.cuDeviceGet(ctypes.byref(ordinal), 0) == 0
            and driver.cuDeviceGetUuid(identifier, ordinal) == 0, 'Cannot query actual vLLM CUDA UUID')
    observed = 'GPU-' + str(uuid.UUID(bytes=identifier.raw))
    rows = [line.split(', ') for line in subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).splitlines()]
    selected = [row for row in rows if visible in row]
    require(len(selected) == 1 and selected[0][1] == observed, 'Actual vLLM CUDA allocation differs from request')
    result['cuda_device'] = {'logical_index': 0, 'physical_index': int(selected[0][0]), 'uuid': observed}
    return result


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def framework_snapshot(root):
    root = Path(root)
    return {name: (hashlib.sha256(os.fsencode(os.readlink(root / name))).hexdigest()
                   if (root / name).is_symlink() else digest_file(root / name))
            for name in framework_sources(root)}


def validate_adapter_installation(binding, root):
    """Check the actual editable carrier and the loaded maintained bootstrap."""
    import configparser
    from email.parser import BytesParser
    import json

    require(isinstance(binding, dict), 'Missing actual vLLM adapter installation')
    root = Path(root).resolve()
    bootstrap = root / 'adapters/jittor_adapters/vllm/bootstrap.py'
    require(binding.get('bootstrap') == str(bootstrap), 'Foreign vLLM adapter bootstrap')
    paths = {}
    for key, filename in (('metadata', 'METADATA'), ('record', 'RECORD'),
                          ('entry_points', 'entry_points.txt'), ('direct_url', 'direct_url.json'),
                          ('bootstrap', 'bootstrap.py')):
        path = Path(binding.get(key, ''))
        require(path.is_file() and path.name == filename
                and digest_file(path) == binding.get(key + '_sha256'), 'Changed actual vLLM adapter ' + key)
        paths[key] = path
    require(len({paths[key].parent for key in ('metadata', 'record', 'entry_points', 'direct_url')}) == 1,
            'Mixed vLLM adapter distribution metadata roots')
    metadata_value = BytesParser().parsebytes(paths['metadata'].read_bytes())
    require(metadata_value['Name'] == 'jittor-torch-adapters' and metadata_value['Version'] == '2.0.0',
            'Wrong actual vLLM adapter distribution identity')
    entries = configparser.ConfigParser()
    entries.read_string(paths['entry_points'].read_text())
    require(entries.get('jittor.module_patches', 'jittor_vllm', fallback='')
            == 'jittor_adapters.vllm:register', 'Missing actual maintained vLLM entry point')
    direct = json.loads(paths['direct_url'].read_text())
    require(direct.get('url') == (root / 'adapters').as_uri()
            and direct.get('dir_info', {}).get('editable') is True, 'Foreign actual vLLM adapter source')


def adapter_installation(root):
    import sys

    root = Path(root).resolve()
    module = sys.modules.get('jittor_adapters.vllm.bootstrap')
    require(module is not None, 'Maintained vLLM adapter bootstrap was not loaded')
    dist = metadata.distribution('jittor-torch-adapters')
    require(any(entry.group == 'jittor.module_patches' and entry.name == 'jittor_vllm'
                and entry.value == 'jittor_adapters.vllm:register' for entry in dist.entry_points),
            'Actual vLLM adapter entry point was not installed')
    paths = {name: Path(dist._path) / filename for name, filename in
             (('metadata', 'METADATA'), ('record', 'RECORD'), ('entry_points', 'entry_points.txt'),
              ('direct_url', 'direct_url.json'))}
    paths['bootstrap'] = Path(module.__file__).resolve()
    result = {key: str(path) for key, path in paths.items()}
    result.update({key + '_sha256': digest_file(path) for key, path in paths.items()})
    validate_adapter_installation(result, root)
    return result


def bind_installation(source, module, wheel, binding):
    """Verify original METADATA and installed RECORD's complete package entries.

    Pip legitimately rewrites RECORD when installing scripts and local metadata.
    The wheel and installed RECORD remain distinct artifacts; every wheel package
    member must retain its original hash and size in the actual installed RECORD.
    The common proof additionally compares the complete on-disk package payload.
    """
    wheel = Path(wheel)
    require(source['upstream'].get('commit') == UPSTREAM_COMMIT
            and source['upstream'].get('version') == '0.29.0', 'Unverified vLLM source pin')
    wheel_sha256 = digest_file(wheel)
    require(wheel_sha256 == WHEEL_SHA256, 'Unverified official vLLM wheel')
    package = Path(module.__file__).resolve().parent
    require({Path(p).resolve() for p in module.__path__} == {package},
            'Additional vLLM package search root')
    require(Path(binding['package']).resolve() == package, 'Changed vLLM package root')
    dist = metadata.distribution('vllm')
    require(Path(dist.locate_file('vllm/__init__.py')).resolve() == Path(module.__file__).resolve(),
            'vLLM distribution metadata belongs to another installation')
    require(dist.metadata['Name'] == 'vllm' and dist.version == str(module.__version__)
            == source['upstream']['version'], 'Actual vLLM distribution identity mismatch')
    metadata_file = Path(dist._path) / 'METADATA'
    record_file = Path(dist._path) / 'RECORD'
    actual_record = record_file.read_bytes()
    rows = list(csv.reader(io.StringIO(actual_record.decode('utf-8'))))
    require(all(len(row) == 3 for row in rows), 'Malformed installed vLLM RECORD')
    recorded = {row[0]: row[1:] for row in rows}
    require(len(recorded) == len(rows), 'Duplicate installed vLLM RECORD path')
    with zipfile.ZipFile(wheel) as archive:
        entries = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
        require(len(entries) == 1, 'Ambiguous vLLM wheel METADATA')
        require(metadata_file.read_bytes() == archive.read(entries[0]),
                'Installed vLLM METADATA differs from official wheel')
        package_entries = {n: archive.getinfo(n) for n in archive.namelist()
                           if n.startswith('vllm/') and not n.endswith('/')}
        installed_names = {n for n in recorded if n.startswith('vllm/')}
        require(package_entries and set(package_entries) <= installed_names,
                'Incomplete installed vLLM RECORD package scope')
        # Pip records generated bytecode with empty hashes/sizes. As in the
        # shared installed-source protocol, generated caches are not payload;
        # only cache entries mapping to an actual wheel Python source qualify.
        for name in installed_names - set(package_entries):
            try:
                original = source_from_cache(name)
            except ValueError as error:
                raise ValueError('Unbound installed vLLM RECORD package entry: ' + name) from error
            require(name.endswith('.pyc') and original in package_entries
                    and original.endswith('.py') and recorded[name] == ['', ''],
                    'Unbound installed vLLM RECORD bytecode entry: ' + name)
        for name, info in package_entries.items():
            relative = name[len('vllm/'):]
            digest = binding['installed_files_sha256'].get(relative)
            require(isinstance(digest, str) and len(digest) == 64,
                    'Missing actual vLLM package hash: ' + name)
            encoded = base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip('=')
            require(recorded[name] == ['sha256=' + encoded, str(info.file_size)],
                    'Installed vLLM RECORD hash/size mismatch: ' + name)
    result = dict(binding, wheel_sha256=wheel_sha256,
                  distribution_name=dist.metadata['Name'], distribution_version=dist.version,
                  metadata=str(metadata_file), metadata_sha256=hashlib.sha256(metadata_file.read_bytes()).hexdigest(),
                  record=str(record_file), record_sha256=hashlib.sha256(actual_record).hexdigest())
    artifact = {'role': 'upstream-wheel', 'path': str(wheel), 'sha256': result['wheel_sha256']}
    proof = installation_proofs({'repo': 'vllm'}, {'vllm': source}, [artifact])
    validate_installations({'downstream_distribution': result}, proof)
    return result

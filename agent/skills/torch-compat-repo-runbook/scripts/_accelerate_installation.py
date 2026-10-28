"""Pinned installed-distribution proof for the maintained Accelerate protocols."""

import base64
import csv
from email.parser import BytesParser
import hashlib
import importlib.metadata
import io
from pathlib import Path
import zipfile

from component_measure import installed_distribution, package_files
from _installed_source_proof import installation_proofs, validate_installations


VERSION = '1.10.1'
SOURCE_SHA = '623cc0ba588d192c77758939a6e340adf07edd4f'
WHEEL_SHA256 = '3621cff60b9a27ce798857ece05e2b9f56fcc71631cfb31ccf71f0359c311f11'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wheel_artifact(path):
    path = Path(path).resolve()
    require(path.is_file() and digest(path) == WHEEL_SHA256,
            'Accelerate requires the pinned official 1.10.1 wheel')
    return {'role': 'upstream-wheel', 'path': str(path), 'sha256': WHEEL_SHA256}


def proof(source, artifact):
    remote = source.get('upstream', {}).get('remote', '')
    require(source.get('repo') == 'accelerate' and source.get('axes') == []
            and source.get('upstream', {}).get('commit') == SOURCE_SHA
            and source['upstream'].get('version') == VERSION
            and remote in ('https://github.com/huggingface/accelerate',
                           'https://github.com/huggingface/accelerate.git',
                           'git@github.com:huggingface/accelerate.git'),
            'Unsupported Accelerate protocol source')
    require(artifact.get('role') == 'upstream-wheel'
            and artifact == wheel_artifact(artifact['path']), 'Wrong official Accelerate wheel artifact')
    result = installation_proofs({'repo': 'accelerate', 'registry_refs': []},
                                 {'accelerate': source}, [artifact])
    require(result['packages'][0]['payload'], 'Missing official Accelerate wheel payload')
    return result


def validate_binding(binding, source, artifact):
    """Validate actual metadata and package closure, preserving original paths."""
    expected = proof(source, artifact)
    validate_installations({'downstream_distribution': binding}, expected)
    require(binding.get('distribution_name') == 'accelerate'
            and binding.get('distribution_version') == VERSION,
            'Wrong actual Accelerate distribution name/version')
    location = binding.get('wheel_path')
    require(isinstance(location, str) and location and Path(location).is_absolute()
            and wheel_artifact(location)['sha256'] == artifact['sha256'],
            'Unbound actual Accelerate wheel location')
    package = Path(binding.get('package', ''))
    origin = Path(binding.get('origin', ''))
    require(package.is_absolute() and origin == package / '__init__.py' and origin.is_file()
            and package_files(package) == binding['installed_files_sha256'],
            'Accelerate module origin is not bound to its installed package')
    records = binding.get('distribution_metadata')
    require(isinstance(records, dict) and set(records) == {'METADATA', 'RECORD'},
            'Missing actual Accelerate METADATA/RECORD proof')
    with zipfile.ZipFile(artifact['path']) as archive:
        names = [name for name in archive.namelist() if name.endswith('.dist-info/METADATA')]
        require(len(names) == 1, 'Ambiguous official Accelerate METADATA')
        metadata_name = names[0]
        for name, record in records.items():
            require(isinstance(record, dict) and isinstance(record.get('text'), str)
                    and isinstance(record.get('path'), str) and record['path'],
                    'Missing actual Accelerate distribution metadata text')
            raw = record['text'].encode()
            path = Path(record.get('path', ''))
            require(path == package.parent / Path(metadata_name).parent / name
                    and path.is_file() and path.read_bytes() == raw
                    and hashlib.sha256(raw).hexdigest() == record.get('sha256'),
                    'Wrong actual Accelerate distribution metadata binding')
            if name == 'METADATA':
                require(raw == archive.read(metadata_name), 'Actual Accelerate METADATA differs from official wheel')
        header = BytesParser().parsebytes(records['METADATA']['text'].encode())
        require(header.get_all('Name') == ['accelerate'] and header.get_all('Version') == [VERSION],
                'Wrong actual Accelerate METADATA identity')
        entries = list(csv.reader(io.StringIO(records['RECORD']['text'])))
        require(all(len(row) == 3 for row in entries)
                and len({row[0] for row in entries}) == len(entries),
                'Malformed actual Accelerate RECORD')
        entries = {name: (checksum, size) for name, checksum, size in entries}
        for name, checksum in expected['packages'][0]['payload'].items():
            member = 'accelerate/' + name
            encoded = base64.urlsafe_b64encode(bytes.fromhex(checksum)).decode().rstrip('=')
            require(entries.get(member) == ('sha256=' + encoded, str(archive.getinfo(member).file_size)),
                    'Actual Accelerate RECORD does not bind complete wheel payload: ' + name)


def capture(module, source, wheel):
    """Called inside each measured worker, after the unchanged training protocol."""
    artifact = wheel_artifact(wheel)
    binding = installed_distribution(source, module)
    distribution = importlib.metadata.distribution('accelerate')
    origin = Path(module.__file__).resolve()
    require(Path(distribution.locate_file('accelerate/__init__.py')).resolve() == origin,
            'Accelerate METADATA belongs to another import prefix')
    records = {}
    for name in ('METADATA', 'RECORD'):
        paths = [path for path in distribution.files or ()
                 if str(path).endswith('.dist-info/' + name)]
        require(len(paths) == 1, 'Missing or ambiguous installed Accelerate ' + name)
        path = Path(distribution.locate_file(paths[0])).resolve()
        # Installed RECORD can use CRLF; retain exact bytes through UTF-8
        # rather than universal-newline text translation.
        text = path.read_bytes().decode()
        records[name] = {'path': str(path), 'sha256': digest(path), 'text': text}
    binding.update(origin=str(origin), distribution_name=distribution.metadata['Name'],
                   distribution_version=distribution.version, distribution_metadata=records,
                   wheel_sha256=artifact['sha256'], wheel_path=artifact['path'])
    validate_binding(binding, source, artifact)
    return binding

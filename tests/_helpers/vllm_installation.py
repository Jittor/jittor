"""Shared host-only installed vLLM wheel fixture for provenance contracts."""

import base64
from contextlib import contextmanager
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import types
from unittest import mock
import zipfile

SCRIPTS = Path(__file__).resolve().parents[2] / 'agent/skills/torch-compat-repo-runbook/scripts'
sys.path.insert(0, str(SCRIPTS))
import _vllm_installation as contract  # noqa: E402


def sha(data):
    return hashlib.sha256(data).hexdigest()


def adapter_binding(root):
    bootstrap = root / 'adapters/jittor_adapters/vllm/bootstrap.py'
    bootstrap.parent.mkdir(parents=True)
    bootstrap.write_text('# maintained adapter fixture\n')
    metadata_root = root / 'candidate-site/jittor_torch_adapters-2.0.0.dist-info'
    metadata_root.mkdir(parents=True)
    payloads = {'metadata': ('METADATA', 'Metadata-Version: 2.1\nName: jittor-torch-adapters\nVersion: 2.0.0\n'),
                'record': ('RECORD', ''),
                'entry_points': ('entry_points.txt', '[jittor.module_patches]\njittor_vllm = jittor_adapters.vllm:register\n'),
                'direct_url': ('direct_url.json', json.dumps({'url': (root / 'adapters').as_uri(),
                                                            'dir_info': {'editable': True}}))}
    paths = {'bootstrap': bootstrap}
    for key, (filename, content) in payloads.items():
        path = metadata_root / filename
        path.write_text(content)
        paths[key] = path
    result = {key: str(path) for key, path in paths.items()}
    result.update({key + '_sha256': sha(path.read_bytes()) for key, path in paths.items()})
    return result


@contextmanager
def installation(tmp_path):
    package = tmp_path / 'site/vllm'
    package.mkdir(parents=True)
    payload = {'__init__.py': b'__version__ = "0.29.0"\n', 'kernels.bin': b'actual payload'}
    meta = package.parent / 'vllm-0.29.0.dist-info'
    meta.mkdir()
    header = b'Metadata-Version: 2.1\nName: vllm\nVersion: 0.29.0\n'
    (meta / 'METADATA').write_bytes(header)
    rows = []
    for name, data in payload.items():
        (package / name).write_bytes(data)
        rows.append(['vllm/' + name, 'sha256=' + base64.urlsafe_b64encode(bytes.fromhex(sha(data))).decode().rstrip('='), str(len(data))])
    def record(entries):
        text = io.StringIO()
        csv.writer(text).writerows(entries)
        return text.getvalue().encode()
    wheel = tmp_path / 'vllm.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        for name, data in payload.items():
            archive.writestr('vllm/' + name, data)
        archive.writestr(meta.name + '/METADATA', header)
        archive.writestr(meta.name + '/RECORD', record(rows))
    # Real pip adds installation metadata; keep the distinct installed RECORD.
    (meta / 'RECORD').write_bytes(record(rows + [[meta.name + '/INSTALLER', '', ''],
                                                ['vllm/__pycache__/__init__.cpython-311.pyc', '', '']]))
    source = {'repo': 'vllm', 'upstream': {'version': '0.29.0', 'commit': contract.UPSTREAM_COMMIT},
              'package_sources': [{'path': 'vllm/__init__.py', 'sha256': sha(payload['__init__.py'])}],
              'package_files': [{'path': 'vllm/' + name, 'sha256': sha(data)} for name, data in payload.items()]}
    module = types.SimpleNamespace(__file__=str(package / '__init__.py'), __path__=[str(package)], __version__='0.29.0')
    dist = types.SimpleNamespace(_path=meta, metadata={'Name': 'vllm'}, version='0.29.0',
                                 locate_file=lambda name: package.parent / name)
    binding = {'repo': 'vllm', 'version': '0.29.0', 'package': str(package),
               'source_sha256': {'vllm/__init__.py': sha(payload['__init__.py'])},
               'installed_files_sha256': {name: sha(data) for name, data in payload.items()}}
    with mock.patch.object(contract.metadata, 'distribution', return_value=dist), \
            mock.patch.object(contract, 'WHEEL_SHA256', sha(wheel.read_bytes())):
        yield source, module, wheel, binding, dist, rows, record



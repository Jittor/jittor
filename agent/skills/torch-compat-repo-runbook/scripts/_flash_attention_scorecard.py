"""Independent FlashAttention implementations; never a same-package registry proof.

The oracle is a locally packaged, source-built upstream extension. The candidate
is the separately distributed Jittor facade and its native extension bridge.
Neither identity substitutes for the other, or for the missing CPU prerequisite.
"""

import ast
import base64
import csv
import functools
import hashlib
import importlib.metadata as md
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

VERSION = '2.7.4.post1'
COMMIT = '5231d95fe13733fb534c01895f7ea88c6a6c7793'
CUTLASS = 'c506e16788cb08416a4a57e11a9067beeee29420'
TORCH_WHEEL_SHA256 = 'c9a7ca4c74fae10a58e6175b4b2cea953f9322bb6562bbf339ad6a05f52190ad'
# This is the actual local source-built artifact, not a PyPI Flash binary.
# A new legitimate build requires a reviewed pin and fresh source-bound evidence.
LOCAL_FLASH_WHEEL_SHA256 = 'c0685ff498af6cde5376f9b24353241957f1ecc534fe8e8976cd8f49a5d65a74'
APIS = ('flash_attn_func', 'flash_attn_kvpacked_func', 'flash_attn_qkvpacked_func',
        'flash_attn_varlen_func', 'flash_attn_varlen_kvpacked_func', 'flash_attn_varlen_qkvpacked_func')
SCRIPT_DIR = 'agent/skills/flash-attention-torch-compat/scripts'


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def bound(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': digest(path)}


def verified_flash_wheel(path):
    artifact = bound(path)
    require(artifact['sha256'] == LOCAL_FLASH_WHEEL_SHA256,
            'Local source-built Flash wheel differs from the fixed artifact')
    return artifact


def recheck(files):
    require(isinstance(files, dict) and files, 'Missing actual external file bindings')
    for name, sha in files.items():
        require(Path(name).is_absolute() and Path(name).is_file() and digest(name) == sha,
                'Actual external file changed: ' + name)


def source_files(root):
    root = Path(root)
    paths = git(root, 'ls-files', 'python', 'compat', 'src', 'backends', SCRIPT_DIR,
                'agent/skills/torch-compat-repo-runbook/scripts', 'tests/_helpers').splitlines()
    paths += ['noxfile.py', 'tests/_helpers/artifacts.py',
              'agent/skills/torch-compat-repo-runbook/scripts/_flash_attention_scorecard.py',
              'agent/skills/torch-compat-repo-runbook/scripts/_protocol_scorecards.py']
    return {p: (hashlib.sha256(os.fsencode(os.readlink(root / p))).hexdigest()
                if (root / p).is_symlink() else digest(root / p)) for p in sorted(set(paths))}


def wheel_files(wheel):
    """Validate the real wheel's complete RECORD, including its native binary."""
    from _installed_source_proof import relative_path
    with zipfile.ZipFile(wheel) as archive:
        names = [n for n in archive.namelist() if not n.endswith('/')]
        require(len(names) == len(set(names)), 'Duplicate local wheel member')
        for name in names:
            relative_path(name)
        records = [n for n in names if n.endswith('.dist-info/RECORD')]
        require(len(records) == 1, 'Expected exactly one wheel RECORD')
        actual = {n: hashlib.sha256(archive.read(n)).hexdigest() for n in names}
        entries = list(csv.reader(io.StringIO(archive.read(records[0]).decode())))
        require(len(entries) == len(names) and {r[0] for r in entries} == set(names),
                'Incomplete wheel RECORD')
        for name, encoded, size in entries:
            if name == records[0]:
                require(encoded == size == '', 'Invalid RECORD self entry')
            else:
                expected = 'sha256=' + base64.urlsafe_b64encode(bytes.fromhex(actual[name])).decode().rstrip('=')
                require(encoded == expected and int(size) == len(archive.read(name)), 'Wheel RECORD mismatch: ' + name)
        return actual


def verify_wheel_installation(source, prefix):
    prefix = Path(prefix).resolve()
    actual = {str(prefix / name): digest(prefix / name) for name in source['wheel_members']}
    require(actual == {str(prefix / name): sha for name, sha in source['wheel_members'].items()},
            'Installed local wheel differs from complete RECORD payload')
    return actual


def verified_torch_proof(path):
    """Anchor the whole actual installation to the fixed official binary wheel."""
    proof = json.loads(Path(path).read_text())
    require(proof['version'] == '2.11.0+cu128' and proof['wheel_sha256'] == TORCH_WHEEL_SHA256
            and digest(proof['wheel']) == TORCH_WHEEL_SHA256, 'Unverified official Torch binary wheel')
    with zipfile.ZipFile(proof['wheel']) as wheel:
        members = {n for n in wheel.namelist() if not n.endswith('/')}
        records = [n for n in members if n.endswith('.dist-info/RECORD')]
        require(len(records) == 1, 'Missing official Torch RECORD')
        record = records[0]
        entries = list(csv.reader(io.StringIO(wheel.read(record).decode())))
        expected = {}
        for name, encoded, size in entries:
            if name == record:
                require(encoded == size == '', 'Invalid official Torch RECORD self entry')
                expected[name] = hashlib.sha256(wheel.read(record)).hexdigest()
            else:
                require(encoded.startswith('sha256='), 'Unhashed official Torch wheel member')
                encoded = encoded[len('sha256='):]
                expected[name] = base64.urlsafe_b64decode(encoded + '=' * (-len(encoded) % 4)).hex()
        require(len(entries) == len(expected) == len(members) and set(expected) == members
                and proof['payload_files_sha256'] == expected, 'Incomplete official Torch payload proof')
    return proof


def selected_nvidia_wheels():
    names = {d.metadata['Name'] for d in md.distributions()
             if (d.metadata.get('Name') or '').startswith('nvidia-')}
    return {name: md.version(name) for name in sorted(names)}


def selected_nvidia_metadata():
    result = {}
    for name, version in selected_nvidia_wheels().items():
        distribution = md.distribution(name)
        require(distribution.version == version, 'Selected provider metadata precedence changed')
        record = Path(distribution._path) / 'RECORD'
        # These are observed platform dependencies, not the two independently
        # proven downstream implementations. Do not invent missing historical
        # provider RECORDs or claim complete NVIDIA wheel installations.
        result[name] = {'version': version, 'metadata': bound(Path(distribution._path) / 'METADATA'),
                        'record': bound(record) if record.is_file() else None,
                        'record_present': record.is_file()}
    return result


def carrier_record_payload(metadata):
    metadata = Path(metadata)
    record = metadata.parent / 'RECORD'
    rows = list(csv.reader(io.StringIO(record.read_text())))
    require(len(rows) == len({row[0] for row in rows}), 'Duplicate carrier RECORD entry')
    result = {}
    for name, encoded, size in rows:
        if Path(name).suffix in ('.pyc', '.pyo'):
            continue
        path = (metadata.parent.parent / name).resolve()
        sha = digest(path)
        if path != record.resolve():
            expected = 'sha256=' + base64.urlsafe_b64encode(bytes.fromhex(sha)).decode().rstrip('=')
            require(encoded == expected and int(size) == path.stat().st_size,
                    'Actual carrier RECORD mismatch: ' + name)
        result[str(path)] = sha
    return result


def facade_metadata(root, package):
    distribution = md.distribution('flash-attn')
    directory = Path(distribution._path).resolve()
    require(directory.parent == Path(package).parent and distribution.version == VERSION,
            'Selected facade compatibility metadata belongs to another implementation')
    resource = Path(root) / 'compat/shim/resources/flash_attn_dist_info'
    declared = {p.name: digest(p) for p in resource.iterdir() if p.is_file()}
    deployed = {p.name: digest(p) for p in directory.iterdir() if p.is_file()}
    require(deployed == declared, 'Facade compatibility metadata differs from owned resources')
    return {'kind': 'owned facade compatibility metadata; not an upstream wheel installation',
            'directory': str(directory), 'files': deployed}


def compiler_headers(directories):
    result = {}
    for directory in directories:
        directory = Path(directory).resolve()
        require(directory.is_dir(), 'Missing actual compiler include directory: ' + str(directory))
        result.update({str(p.resolve()): digest(p) for p in directory.rglob('*')
                       if p.is_file() and p.suffix in ('.h', '.hpp', '.cuh', '.inc', '.inl', '.tpp')})
    require(result, 'Missing actual compiler header payload')
    return result


def capture_compiler_inputs():
    from jittor.compat.shim import cpp_extension
    configuration = cpp_extension.cfg()
    distribution = md.distribution('pybind11')
    metadata = bound(Path(distribution._path) / 'METADATA')
    record = bound(Path(distribution._path) / 'RECORD')
    include = Path(configuration['pybind_inc']).resolve()
    require(include == Path(distribution.locate_file('pybind11/include')).resolve(),
            'Actual compiler pybind headers do not belong to selected real distribution')
    directories = sorted({str(Path(p).resolve()) for p in
                          [*configuration['cuda_includes'], configuration['pybind_inc'], configuration['py_inc']]})
    return {'kind': 'actual declared compiler include resources', 'include_directories': directories,
            'headers': compiler_headers(directories), 'pybind_include': str(include),
            'pybind_distribution': {'name': distribution.metadata['Name'], 'version': distribution.version,
                                    'metadata': metadata, 'record': record,
                                    'installed_files': carrier_record_payload(metadata['path'])},
            'cuda_wheel_fingerprint': configuration['cuda_wheel_fingerprint']}


def source_descriptor(configuration):
    from registry_inventory import inventory
    from _installed_source_proof import wheel_payload
    config = json.loads(Path(configuration).read_text())
    require(set(config) == {'upstream', 'wheel', 'oracle_build', 'torch_wheel_proof'}, 'Unknown source configuration')
    upstream = Path(config['upstream']).resolve()
    result = inventory('flash-attention', upstream)
    require(result['upstream']['commit'] == COMMIT and result['upstream']['version'] == VERSION
            and result['upstream']['remote'] in ('git@github.com:Dao-AILab/flash-attention.git',
                                                'https://github.com/Dao-AILab/flash-attention.git'),
            'Unverified official source identity')
    require(not git(upstream, 'status', '--porcelain'), 'Dirty official upstream')
    require(git(upstream / 'csrc/cutlass', 'rev-parse', 'HEAD') == CUTLASS
            and not git(upstream / 'csrc/cutlass', 'status', '--porcelain'), 'Unverified CUTLASS source')
    require(git(upstream, 'ls-tree', COMMIT, 'csrc/cutlass').split()[2] == CUTLASS,
            'CUTLASS checkout differs from official upstream gitlink')
    setup = upstream / 'setup.py'
    package_names = [ast.literal_eval(n.value) for n in ast.parse(setup.read_text()).body
                     if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PACKAGE_NAME'
                                                         for t in n.targets)]
    require(package_names == ['flash_attn'], 'Unrecognized source-declared distribution name')
    wheel = verified_flash_wheel(config['wheel'])
    _, payload = wheel_payload(wheel, 'flash-attention', VERSION, 'flash_attn',
                               {r['path']: r['sha256'] for r in result['package_files']},
                               distribution_names=package_names)
    require(all(p['path'][len('flash_attn/'):] in payload for p in result['package_sources']),
            'Local wheel omits official Python source')
    members = wheel_files(wheel['path'])
    tracked = set(git(upstream, 'ls-files').splitlines())
    for name, sha in members.items():
        if name.endswith('.py'):
            require(name in tracked and digest(upstream / name) == sha,
                    'Local wheel Python file differs from official source: ' + name)
    native = [n for n in members if n.startswith('flash_attn_2_cuda.') and n.endswith('.so')]
    require(len(native) == 1, 'Local wheel lacks genuine native extension')
    build = json.loads(Path(config['oracle_build']).read_text())
    require(build['status'] == 'passed' and build['upstream_sha'] == COMMIT
            and build['module_sha256'] == members[native[0]], 'Unbound upstream native source build')
    require(len(build['sources']) == 85, 'Incomplete declared native source build')
    cuda_calls = [n for n in ast.walk(ast.parse(setup.read_text())) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Name) and n.func.id == 'CUDAExtension']
    declaration = next(n for n in cuda_calls if any(k.arg == 'name' and isinstance(k.value, ast.Constant)
                                                   and k.value.value == 'flash_attn_2_cuda' for k in n.keywords))
    declared = ast.literal_eval(next(k.value for k in declaration.keywords if k.arg == 'sources'))
    arguments = next(k.value for k in declaration.keywords if k.arg == 'extra_compile_args')
    nvcc = next(v for k, v in zip(arguments.keys, arguments.values) if ast.literal_eval(k) == 'nvcc')
    flags = ast.literal_eval(nvcc.args[0].left)
    require(set(declared) == set(build['sources']) and build['cuda_flags'] == flags,
            'Native source/flags differ from official CUDAExtension declaration')
    torch_proof = verified_torch_proof(config['torch_wheel_proof'])
    require(build['torch'] == torch_proof['version'] and build['cuda'] == '12.8', 'Native extension/Torch ABI mismatch')
    for name, sha in build['sources'].items():
        require(digest(upstream / name) == sha, 'Native build source mismatch: ' + name)
    actual_sources = git(upstream, 'ls-files', 'csrc/flash_attn').splitlines()
    actual_sources += ['setup.py', '.gitmodules', 'AUTHORS', 'LICENSE']
    actual_sources += [n for n in members if n.endswith('.py')]
    upstream_files = {str(upstream / p): digest(upstream / p) for p in actual_sources}
    upstream_files.update({str(p): digest(p) for p in sorted((upstream / 'csrc/cutlass/include').rglob('*'))
                           if p.is_file()})
    result.update(axes=[], canonical_attention_api_inventory=inventory('flash-attention', upstream)['axes'],
                  scope='Bounded independent FlashAttention training protocol; full public API declaration inventory retained separately, no registry coverage admission',
                  configuration=bound(configuration), local_wheel=wheel, wheel_members=members,
                  local_wheel_kind='local packaging of pinned upstream Python and previously source-built native binary; not a published PyPI binary',
                  oracle_build=bound(config['oracle_build']), torch_wheel_proof=bound(config['torch_wheel_proof']),
                  setup=bound(setup), distribution_name=package_names[0], native_member=native[0],
                  upstream_files=upstream_files, cutlass_commit=CUTLASS)
    result['excluded_from_local_wheel'] = [dict(row, reason='Unchanged upstream build_py excludes this non-Python package file')
                                           for row in result['package_files'] if row['path'][len('flash_attn/'):] not in payload]
    require(all(not row['path'].endswith('.py') for row in result['excluded_from_local_wheel']),
            'Cannot exclude upstream Python source')
    return result


def capture_implementation(root, source, runtime, torch, flash_attn, extension, backend=None):
    package = Path(flash_attn.__file__).resolve().parent
    external = dict(source['upstream_files'])
    for item in ('configuration', 'local_wheel', 'oracle_build', 'torch_wheel_proof', 'setup'):
        external[source[item]['path']] = source[item]['sha256']
    actual = {p.relative_to(package).as_posix(): digest(p) for p in sorted(package.rglob('*'))
              if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo')}
    compatibility_metadata = None
    compiler_inputs = None
    if runtime == 'torch':
        require(not hasattr(torch, '_torch_compat_install_context') and 'jittor' not in sys.modules,
                'Reference is not independent binary Torch')
        distribution = md.distribution(source['distribution_name'])
        require(distribution.version == VERSION and Path(distribution.locate_file('flash_attn')).resolve() == package,
                'Imported upstream package and selected metadata differ')
        prefix = Path(distribution.locate_file('')).resolve()
        external.update(verify_wheel_installation(source, prefix))
        expected = {n[len('flash_attn/'):]: h for n, h in source['wheel_members'].items() if n.startswith('flash_attn/')}
        require(actual == expected and Path(extension.__file__).resolve() == prefix / source['native_member'],
                'Actual imported upstream implementation differs from local wheel')
        proof = json.loads(Path(source['torch_wheel_proof']['path']).read_text())
        torch_prefix = Path(proof['actual_prefix']).resolve()
        require(Path(torch.__file__).resolve() == torch_prefix / 'torch/__init__.py'
                and torch.__version__ == proof['version'] and digest(proof['wheel']) == proof['wheel_sha256'],
                'Actual binary Torch identity mismatch')
        external[proof['wheel']] = proof['wheel_sha256']
        for name, sha in proof['payload_files_sha256'].items():
            require(digest(torch_prefix / name) == sha, 'Binary Torch installed payload changed: ' + name)
            external[str(torch_prefix / name)] = sha
        kind = 'upstream-local-wheel-with-source-built-native-extension'
        carrier = {'name': distribution.metadata['Name'], 'version': distribution.version,
                   'metadata': bound(Path(distribution._path) / 'METADATA'),
                   'record': bound(Path(distribution._path) / 'RECORD')}
        require(Path(torch._C.__file__).suffix == '.so', 'Missing binary Torch')
        external[str(Path(torch._C.__file__).resolve())] = digest(torch._C.__file__)
    else:
        require(hasattr(torch, '_torch_compat_install_context') and backend._flashattn_jittor_official,
                'Candidate did not use native official-source bridge')
        facade = Path(root) / 'compat/shim/resources/stubs/flash_attn'
        expected = {p.relative_to(facade).as_posix(): digest(p) for p in facade.rglob('*') if p.is_file()}
        require(actual == expected and len(actual) == 5, 'Installed facade differs, including nested modules')
        compatibility_metadata = facade_metadata(root, package)
        external.update({str(Path(compatibility_metadata['directory']) / n): h
                         for n, h in compatibility_metadata['files'].items()})
        distribution = md.distribution('jittor-torch')
        require(distribution.metadata['Name'] == 'jittor-torch', 'Missing real candidate carrier')
        carrier = {'name': distribution.metadata['Name'], 'version': distribution.version,
                   'metadata': bound(Path(distribution._path) / 'METADATA'),
                   'record': bound(Path(distribution._path) / 'RECORD')}
        for module in (extension, backend._flashattn_jittor_packed_low_level):
            require(module is not None and Path(module.__file__).suffix == '.so', 'Missing real native bridge extension')
            directory = Path(module.__file__).resolve().parent
            for path in directory.rglob('*'):
                if path.is_file() and path.suffix in ('.so', '.cpp', '.cc', '.cu', '.h', '.ninja'):
                    external[str(path.resolve())] = digest(path)
        kind = 'jittor-facade-and-native-official-source-bridge'
        compiler_inputs = capture_compiler_inputs()
        external.update(compiler_inputs['headers'])
        external.update(compiler_inputs['pybind_distribution']['installed_files'])
    for name, sha in actual.items():
        external[str(package / name)] = sha
    carrier_files = carrier_record_payload(carrier['metadata']['path'])
    require(carrier_files, 'Missing actual carrier installed files')
    carrier['installed_files'] = carrier_files
    external.update(carrier_files)
    for item in (carrier['metadata'], carrier['record']):
        external[item['path']] = item['sha256']
    return {'kind': kind, 'public_api_version': VERSION,
            'version_semantics': 'public API version; actual carrier identity is separate',
            'package': str(package), 'package_files': actual, 'carrier_distribution': carrier,
            'compatibility_metadata': compatibility_metadata,
            'compiler_inputs': compiler_inputs,
            'extension': bound(extension.__file__), 'external_files': external}


def validate_implementation(value, source, root, runtime):
    require(value.get('source_descriptor') == source, 'Actual process source descriptor mismatch')
    require(value.get('source_files') == source_files(root), 'Measured framework/tool bytes changed')
    proof = value.get('implementation', {})
    kind = ('upstream-local-wheel-with-source-built-native-extension' if runtime == 'torch'
            else 'jittor-facade-and-native-official-source-bridge')
    require(proof.get('kind') == kind and proof.get('public_api_version') == VERSION
            and proof.get('version_semantics') == 'public API version; actual carrier identity is separate',
            'Missing independent implementation identity')
    carrier = proof.get('carrier_distribution', {})
    require(carrier.get('name') == ('flash_attn' if runtime == 'torch' else 'jittor-torch'),
            'Actual distribution carrier mismatch')
    external = proof.get('external_files', {})
    require(source['upstream_files'].items() <= external.items(), 'Missing actual native source/CUTLASS binding')
    for field in ('configuration', 'local_wheel', 'oracle_build', 'torch_wheel_proof', 'setup'):
        require(external.get(source[field]['path']) == source[field]['sha256'], 'Missing external source identity: ' + field)
    for field in ('metadata', 'record'):
        require(external.get(carrier.get(field, {}).get('path')) == carrier.get(field, {}).get('sha256')
                and carrier.get(field, {}).get('sha256'), 'Missing actual carrier metadata/RECORD')
    require(carrier.get('installed_files') and carrier['installed_files'].items() <= external.items(),
            'Incomplete actual carrier installed-file binding')
    metadata_path, record_path = Path(carrier['metadata']['path']), Path(carrier['record']['path'])
    require(record_path == metadata_path.parent / 'RECORD', 'Carrier METADATA/RECORD prefix mismatch')
    from email.parser import BytesParser
    metadata = BytesParser().parsebytes(metadata_path.read_bytes())
    require(metadata.get_all('Name') == [carrier['name']] and metadata.get_all('Version') == [carrier['version']],
            'Carrier identity differs from actual METADATA')
    installed = carrier_record_payload(metadata_path)
    require(installed == carrier['installed_files'], 'Incomplete actual carrier RECORD payload')
    package = Path(proof.get('package', ''))
    require(package.is_absolute(), 'Missing imported package path')
    if runtime == 'torch':
        expected = {n[len('flash_attn/'):]: h for n, h in source['wheel_members'].items() if n.startswith('flash_attn/')}
        require(proof.get('package_files') == expected
                and proof.get('extension', {}).get('sha256') == source['wheel_members'][source['native_member']],
                'Actual oracle package/native extension mismatch')
        require(verify_wheel_installation(source, package.parent).items() <= external.items(),
                'Incomplete installed native wheel payload binding')
        torch_proof = json.loads(Path(source['torch_wheel_proof']['path']).read_text())
        expected_torch = {str(Path(torch_proof['actual_prefix']) / n): h
                          for n, h in torch_proof['payload_files_sha256'].items()}
        require(expected_torch.items() <= external.items()
                and external.get(torch_proof['wheel']) == torch_proof['wheel_sha256'],
                'Incomplete actual binary Torch wheel/payload binding')
    else:
        facade = Path(root) / 'compat/shim/resources/stubs/flash_attn'
        expected = {p.relative_to(facade).as_posix(): digest(p) for p in facade.rglob('*') if p.is_file()}
        require(proof.get('package_files') == expected, 'Incomplete recursive candidate facade binding')
        compatibility = proof.get('compatibility_metadata', {})
        resources = Path(root) / 'compat/shim/resources/flash_attn_dist_info'
        declared = {p.name: digest(p) for p in resources.iterdir() if p.is_file()}
        directory = Path(compatibility.get('directory', ''))
        require(compatibility.get('kind') == 'owned facade compatibility metadata; not an upstream wheel installation'
                and compatibility.get('files') == declared and directory.parent == package.parent
                and all(external.get(str(directory / n)) == h for n, h in declared.items()),
                'Missing honest facade compatibility metadata/resource binding')
        require(carrier.get('version') == '2.0.0', 'Wrong actual candidate carrier version')
        inputs = proof.get('compiler_inputs', {})
        require(inputs.get('kind') == 'actual declared compiler include resources'
                and inputs.get('headers') == compiler_headers(inputs.get('include_directories', []))
                and inputs['headers'].items() <= external.items(), 'Incomplete actual compiler header binding')
        pybind = inputs.get('pybind_distribution', {})
        require(pybind.get('name') == 'pybind11' and pybind.get('version')
                and carrier_record_payload(pybind['metadata']['path']) == pybind.get('installed_files')
                and pybind['installed_files'].items() <= external.items(), 'Incomplete real pybind distribution binding')
        require(str(Path(inputs['pybind_include']) / 'pybind11/pybind11.h') in inputs['headers']
                and any(Path(p).name == 'cuda_runtime.h' for p in inputs['headers']),
                'Missing actual pybind/CUDA compiler headers')
        require(any('flash_attn_2_cuda_jittor' in p and p.endswith('.so') for p in external)
                and any('flash_attn_2_cuda_jittor_packed' in p and p.endswith('.so') for p in external)
                and any(p.endswith(('.cpp', '.cu')) and 'flash' in p for p in external),
                'Missing actual generated native bridge source/binaries')
    require(all(external.get(str(package / n)) == h for n, h in proof.get('package_files', {}).items()),
            'Missing actual installed package file binding')
    recheck(proof.get('external_files'))
    require(value.get('runtime_pid', 0) > 0 and value.get('runtime') == runtime, 'Missing real runtime process identity')
    require(value.get('api') in APIS and value.get('head_dim') == 32 and value.get('causal') is False
            and value.get('dtype') in ('float16', 'bfloat16') and value.get('perturbation') in (0., .125),
            'Measured execution is outside the bounded training protocol')
    inputs = value.get('inputs', {})
    expected_protocol = protocol(value['api'], value['dtype'])
    require(inputs == {'seed': 20261015, 'shape_kind': 'tiny', 'shape': expected_protocol['shape'],
                       'batch': 2, 'heads': 2, 'sequence_lengths': expected_protocol['sequence_lengths'],
                       'hf_hub_offline': True, 'steps': 3, 'optimizer': 'SGD', 'learning_rate': .01,
                       'dropout': 0., 'parameter_dtype': 'float32', 'gradient_dtype': 'float32'},
            'Original input/training protocol changed')
    observations = value.get('device_observations', [])
    labels = {r['label'] for r in observations}
    required = {f'step{s}/{n}' for s in range(3) for n in ('q', 'k', 'v', 'output', 'loss',
                'grad0', 'grad1', 'grad2', 'input_grad', 'param0', 'param1', 'param2')}
    require(labels == required and len(observations) == len(required)
            and all(r['device'] == 'cuda' and r['pointer_device'] == 0 for r in observations),
            'Missing physical CUDA tensor observations')
    for row in observations:
        field = row['label'].split('/', 1)[1]
        dtype = 'torch.' + (value['dtype'] if field in ('q', 'k', 'v', 'output') else 'float32')
        # The Jittor Tensor frontend spells dtype names identically through str(dtype).
        require(row['dtype'] == dtype and row['shape'] == ([] if field == 'loss' else inputs['shape']),
                'Actual operand/gradient shape or dtype changed: ' + row['label'])
    require(value.get('physical_device_uuid', '').startswith('GPU-'), 'Missing physical device UUID')
    require(value.get('mapped_libraries') and all(Path(p).is_file() and digest(p) == sha
            for p, sha in value['mapped_libraries'].items()), 'Mapped CUDA libraries changed')
    require(value.get('cuda_runtime', {}).get('version', 0) > 0
            and value['cuda_runtime']['path'] in value['mapped_libraries'], 'Missing actual loaded CUDA runtime version')
    providers = value.get('platform', {}).get('nvidia_metadata', {})
    require(providers and {n: p['version'] for n, p in providers.items()} == value['platform']['nvidia_wheels'],
            'Missing actual selected provider metadata')
    for name, provider in providers.items():
        for item in (provider['metadata'], provider['record']):
            if item is None:
                continue
            require(digest(item['path']) == item['sha256'], 'Actual selected provider metadata changed')
        observed_record = Path(provider['metadata']['path']).parent / 'RECORD'
        require(provider.get('record_present') is observed_record.is_file()
                and (provider['record'] is not None) is observed_record.is_file(),
                'Missing/misreported observed provider RECORD availability')
        if provider['record'] is not None:
            require(provider['record']['path'] == str(observed_record.resolve()), 'Provider RECORD prefix mismatch')
        metadata = BytesParser().parsebytes(Path(provider['metadata']['path']).read_bytes())
        require(metadata['Name'] == name and metadata['Version'] == provider['version'], 'Actual provider identity mismatch')
    runtime = value['cuda_runtime']['version']
    runtime_names = [n for n in providers if n.startswith('nvidia-cuda-runtime')]
    require(len(runtime_names) == 1 and providers[runtime_names[0]]['version'].split('.')[:2]
            == [str(runtime // 1000), str((runtime % 1000) // 10)],
            'Loaded CUDA runtime differs from selected provider version')
    for item in value.get('platform', {}).get('compiler_executables', {}).values():
        require(Path(item['path']).is_file() and digest(item['path']) == item['sha256'], 'Compiler executable changed')
    require(set(value.get('platform', {}).get('compiler_executables', {})) == {'nvcc', 'cxx'}, 'Missing compiler identities')
    require(value.get('environment', {}).keys() >= {'HOME', 'TMPDIR', 'JITTOR_HOME', 'XDG_CACHE_HOME'},
            'Missing actual per-process environment')


def protocol(api, dtype):
    return {'id': 'flash-attention-training', 'api': api, 'precision': dtype, 'head_dim': 32,
            'causal': False, 'world_size': 1, 'seed': 20261015, 'input_perturbations': [0., .125],
            'updates': 3, 'optimizer': 'SGD', 'learning_rate': .01, 'dropout': 0.,
            'parameter_dtype': 'float32', 'gradient_dtype': 'float32',
            'shape': [11, 2, 32] if 'varlen' in api else [2, 11, 2, 32],
            'sequence_lengths': [4, 7] if 'varlen' in api else [11, 11],
            'tf32': False, 'matmul_precision': 'highest', 'num_threads': 2}


def admission_entrypoint(function):
    """Malformed/inaccessible host evidence is a rejection, never a runtime skip.

    Tensor execution and compilation are outside these host-only entrypoints.
    Keep the original exception as the cause and its text in the rejection.
    """
    @functools.wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (KeyError, TypeError, AttributeError, OSError, AssertionError, StopIteration,
                subprocess.CalledProcessError, zipfile.BadZipFile) as error:
            raise ValueError('Invalid FlashAttention evidence: ' + str(error)) from error
    return checked


@admission_entrypoint
def validate_spec(record, source):
    spec = record.get('protocol', {})
    require(spec.get('api') in APIS and spec.get('precision') in ('float16', 'bfloat16')
            and json.dumps(spec, sort_keys=True) == json.dumps(protocol(spec['api'], spec['precision']), sort_keys=True),
            'Unsupported FlashAttention training protocol')
    require(record.get('record_kind') == 'protocol' and record.get('repo') == source.get('repo') == 'flash-attention'
            and record.get('registry_refs') == source.get('axes') == [], 'Not a bounded independent protocol')
    require(source == source_descriptor(source['configuration']['path']), 'Source descriptor no longer matches actual sources')


@admission_entrypoint
def validate_measurements(record, artifacts, source, *, root=None):
    validate_spec(record, source)
    auditor_root = Path(__file__).resolve().parents[4]
    root = Path(root).resolve() if root is not None else auditor_root
    scripts = auditor_root / SCRIPT_DIR
    module_spec = importlib.util.spec_from_file_location('flash_source_collector', scripts / 'collect_scorecard.py')
    collector = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(collector)
    by_role = {a['role']: a for a in artifacts}
    require('candidate' in by_role and 'reference' in by_role, 'Missing actual runtime arrays')
    directory = Path(by_role['candidate']['path']).parent
    comparison_spec = importlib.util.spec_from_file_location('flash_source_comparison', scripts / 'compare_training.py')
    comparison = importlib.util.module_from_spec(comparison_spec)
    comparison_spec.loader.exec_module(comparison)
    regenerated = collector.collect_case(directory, source, record['jittor_commit'],
                                        collector.measured_probe_hash(root, record['jittor_commit']), comparison,
                                        record['evidence']['report'], record['evidence']['command'], None, root=root)
    for name in ('protocol', 'inputs', 'training', 'numerics', 'fallback', 'platform', 'downstream_versions'):
        require(record[name] == regenerated[name], 'Regenerated FlashAttention measurement differs: ' + name)
    supplied = {(a['role'], str(Path(a['path']).resolve()), a['sha256']) for a in artifacts}
    required = {(a['role'], a['path'], a['sha256']) for a in regenerated['evidence']['artifacts']}
    require(required <= supplied, 'Incomplete binding of actual phase/runtime evidence')

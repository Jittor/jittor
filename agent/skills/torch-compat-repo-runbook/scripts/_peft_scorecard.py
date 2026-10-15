"""Full source and numerical proof for the existing PEFT LoRA/Llama scenario.

This adds no model/optimizer mathematics and never upgrades historical raw files.
"""

from email.parser import BytesParser
import hashlib
import importlib.metadata as md
import os
from pathlib import Path
import subprocess
import zipfile

CASE = 'peft_lora_llama'
PINS = {
    'peft': ('0.17.1', '53c25fe4fdd7c6aa4b40db0560815ea570d32303',
             'huggingface/peft', '3d129d64def3d74779c32a080d2567e5f7b674e77d546e3585138216d903f99e'),
    'transformers': ('4.56.2', 'cd74917ffc3e8f84e4a886052c5ab32b7ac623cc',
                     'huggingface/transformers', '79c03d0e85b26cb573c109ff9eafa96f3c8d4febfd8a0774e8bba32702dd6dde'),
}
ATOL, RTOL = 2e-3, 1e-2


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(checkout, *args):
    return subprocess.check_output(['git', '-C', str(checkout), *args], text=True).strip()


def source_descriptor(checkouts, wheels):
    from registry_inventory import inventory
    result = {}
    for repo, (version, sha, remote, wheel_hash) in PINS.items():
        checkout, wheel = Path(checkouts[repo]).resolve(), Path(wheels[repo]).resolve()
        source = inventory(repo, checkout)
        require(not git(checkout, 'status', '--porcelain'), 'Dirty official upstream: ' + repo)
        require(source['upstream']['commit'] == sha and source['upstream']['version'] == version,
                'Unexpected pinned upstream: ' + repo)
        require(source['upstream']['remote'] in ('git@github.com:' + remote + '.git',
                                                'https://github.com/' + remote + '.git'),
                'Unexpected official upstream remote: ' + repo)
        require(digest(wheel) == wheel_hash, 'Wheel differs from pinned official PyPI bytes: ' + repo)
        tracked = git(checkout, 'ls-files', 'src/' + repo).splitlines()
        require(tracked, 'Empty upstream package: ' + repo)
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            require(len(names) == len(set(names)), 'Duplicate wheel member')
            metadata = [n for n in names if n.endswith('.dist-info/METADATA')]
            require(len(metadata) == 1, 'Expected one wheel distribution')
            package = BytesParser().parsebytes(archive.read(metadata[0]))
            require(package.get_all('Name') == [repo] and package.get_all('Version') == [version],
                    'Wrong wheel name/version')
            payload = {n[len(repo) + 1:]: hashlib.sha256(archive.read(n)).hexdigest()
                       for n in names if n.startswith(repo + '/') and not n.endswith('/')}
            included, excluded = [], []
            for name in tracked:
                relative = name[len('src/' + repo + '/'):]
                sha256 = digest(checkout / name)
                row = {'path': name, 'sha256': sha256}
                if relative not in payload:
                    require(Path(relative).suffix in ('.md', '.sh'),
                            'Official wheel omits runtime source: ' + name)
                    excluded.append(dict(row, reason='Official wheel excludes this non-runtime source file'))
                else:
                    require(payload[relative] == sha256, 'Wheel/source bytes differ: ' + name)
                    included.append(row)
            expected = {r['path'][len('src/' + repo + '/'):]: r['sha256'] for r in included}
            require(payload == expected, 'Unbound wheel package file: ' + repo)
        source.update(package_sources=included, excluded_package_sources=excluded,
                      wheel={'path': str(wheel), 'sha256': wheel_hash, 'package_files_sha256': payload})
        result[repo] = source
    return result


def installed_payload(source, module):
    """Inspect the package that this actual tensor process imported."""
    repo = source['repo']
    package = Path(module.__file__).resolve().parent
    require({Path(p).resolve() for p in module.__path__} == {package}, 'Additional package search root')
    actual = {p.relative_to(package).as_posix(): digest(p) for p in sorted(package.rglob('*'))
              if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo')}
    expected = source['wheel']['package_files_sha256']
    require(actual == expected, 'Actual installed package differs from official wheel: ' + repo)
    distribution = md.distribution(repo)
    require(md.version(repo) == str(module.__version__) == source['upstream']['version'],
            'Selected distribution/imported version mismatch: ' + repo)
    require(Path(distribution.locate_file(repo)).resolve() == package,
            'Selected distribution metadata belongs to another package: ' + repo)
    metadata = Path(distribution._path) / 'METADATA'
    record = Path(distribution._path) / 'RECORD'
    with zipfile.ZipFile(source['wheel']['path']) as wheel:
        prefix = next(n.rsplit('/', 1)[0] for n in wheel.namelist() if n.endswith('.dist-info/METADATA'))
        require(metadata.read_bytes() == wheel.read(prefix + '/METADATA')
                and record.read_bytes() == wheel.read(prefix + '/RECORD'), 'Installed wheel metadata/RECORD differs')
    return {'repo': repo, 'version': str(module.__version__), 'package': str(package),
            'source_sha256': {r['path']: r['sha256'] for r in source['package_sources']},
            'wheel_sha256': source['wheel']['sha256'], 'installed_files_sha256': actual,
            'metadata': str(metadata), 'metadata_sha256': digest(metadata),
            'record': str(record), 'record_sha256': digest(record)}


def source_files(root):
    root = Path(root)
    paths = git(root, 'ls-files', 'python', 'compat', 'src', 'backends').splitlines()
    paths += ['noxfile.py', 'tools/run_peft_gate.py',
              'agent/skills/torch-compat-repo-runbook/scripts/collect_peft_scorecards.py',
              'agent/skills/torch-compat-repo-runbook/scripts/_peft_scorecard.py',
              'agent/skills/torch-compat-repo-runbook/scripts/training_probe.py',
              'compat/tests/torch/_ecosystem_runner.py', 'compat/tests/torch/_ecosystem_cases.py']
    return {p: (hashlib.sha256(os.fsencode(os.readlink(root / p))).hexdigest()
                if (root / p).is_symlink() else digest(root / p)) for p in sorted(set(paths))}


def validate_binding(value, sources, baseline, runtime, device):
    require(value.get('case') == CASE and value.get('jittor_commit') == baseline,
            'Wrong case/source baseline')
    require(value.get('runtime') == runtime and value.get('device') == device,
            'Wrong runtime/device')
    identity = value.get('interpreter', {})
    require(identity.get('is_shim') is (runtime == 'jittor') and type(identity.get('pid')) is int,
            'Wrong actual interpreter identity')
    if runtime == 'torch':
        binary = identity.get('binary_extension')
        require(isinstance(binary, str) and Path(binary).is_file()
                and digest(binary) == identity.get('binary_sha256'), 'Invalid genuine Torch binary proof')
    for repo, source in sources.items():
        actual = value.get('downstream_distributions', {}).get(repo, {})
        require(actual.get('version') == source['upstream']['version']
                and actual.get('source_sha256') == {r['path']: r['sha256'] for r in source['package_sources']}
                and actual.get('wheel_sha256') == source['wheel']['sha256']
                and actual.get('installed_files_sha256') == source['wheel']['package_files_sha256'],
                'Missing full package/source/wheel binding: ' + repo)
        captured = value.get('dependencies', {}).get(repo, {})
        require(captured.get('installed_files_sha256') == actual['installed_files_sha256']
                and captured.get('version') == actual['version']
                and Path(captured.get('origin', '')).parent == Path(actual.get('package', '')),
                'Probe/runtime dependency capture differs: ' + repo)
    require(value.get('hf_hub_offline') is True and value.get('measurement_source_sha256'),
            'Missing offline/full runtime source proof')
    if runtime == 'jittor':
        require(value.get('fallback_count') == 0 and value.get('fallback_policy') == 'error',
                'Missing zero-fallback computation')
    observations = value.get('device_observations', [])
    require(observations and all(r.get('device') == device for r in observations),
            'Missing actual operand/output/gradient residency')
    if device == 'cuda':
        require(all(r.get('pointer_device') == 0 for r in observations), 'Not actual logical CUDA allocation zero')
    requested = set(value['model_scope']['trainable_parameters'] if value.get('mode') == 'train'
                    else value['model_scope']['parameters'])
    require({r['label'][len('gradient/'): ] for r in observations if r['label'].startswith('gradient/')} == requested,
            'Missing observed parameter gradient residency')


def compare_arrays(reference, candidate, expected):
    import numpy as np
    with np.load(reference) as ref, np.load(candidate) as cand:
        require(set(ref.files) == set(cand.files) == set(expected), 'Missing/unexpected numerical field')
        # This is the original verify_repo field-scale budget, not a new threshold.
        scale = max([1e-12, *[float(np.abs(ref[k].astype('float64')).max()) for k in ref.files if ref[k].size]])
        rows = []
        for name in sorted(ref.files):
            a, b = ref[name], cand[name]
            require(a.shape == b.shape and a.dtype == b.dtype and np.isfinite(a).all() and np.isfinite(b).all(),
                    'Invalid numerical array: ' + name)
            error = float(np.abs(a.astype('float64') - b.astype('float64')).max()) if a.size else 0.
            require(error <= ATOL or error / scale <= RTOL, 'Original numerical budget failed: ' + name)
            rows.append({'name': name, 'max_abs': error, 'max_scaled_relative': error / scale})
    return rows

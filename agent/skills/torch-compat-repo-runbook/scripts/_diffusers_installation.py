"""Fixed official Diffusers payload and actual-process installation authority."""

import hashlib
from importlib import metadata
from pathlib import Path
import subprocess
import zipfile

from _installed_source_proof import package_path, require, wheel_distribution_binding


VERSION = '0.35.1'
COMMIT = '0f252be0ed42006c125ef4429156cb13ae6c1d60'
WHEEL_SHA256 = 'fe29ff10200970c7c5934c6488c213e2a77a03dad5e6fa00bbd8e1d04234cb0e'
# The official release excludes these tracked documentation pages; no runtime
# source, generated Python file or other extra payload is exempted.
OMITTED_DOCUMENTATION = frozenset((
    'experimental/README.md', 'models/README.md', 'pipelines/README.md',
    'pipelines/deprecated/README.md', 'pipelines/stable_diffusion/README.md',
    'schedulers/README.md',
))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wheel_pin(source, wheel):
    upstream = source['upstream']
    require(source.get('repo') == 'diffusers' and upstream.get('version') == VERSION
            and upstream.get('commit') == COMMIT and upstream.get('remote') in (
                'https://github.com/huggingface/diffusers.git',
                'git@github.com:huggingface/diffusers.git'), 'Unverified official Diffusers source')
    require(digest(wheel) == WHEEL_SHA256, 'Diffusers wheel differs from fixed official PyPI bytes')
    known = {package_path(row['path'])[1]: row['sha256'] for row in source['package_files']}
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'Duplicate official Diffusers wheel member')
        payload = {name[len('diffusers/'):]: hashlib.sha256(archive.read(name)).hexdigest()
                   for name in names if name.startswith('diffusers/') and not name.endswith('/')}
    require(set(known) - set(payload) == OMITTED_DOCUMENTATION,
            'Unexpected official Diffusers source omissions')
    require(payload == {name: value for name, value in known.items() if name not in OMITTED_DOCUMENTATION},
            'Unbound or changed official Diffusers package payload')
    return WHEEL_SHA256


def source_descriptor(checkout, wheel):
    from registry_inventory import inventory

    checkout, wheel = Path(checkout).resolve(), Path(wheel).resolve()
    source = inventory('diffusers', checkout)
    require(not subprocess.check_output(['git', '-C', str(checkout), 'status', '--porcelain']),
            'Dirty official Diffusers checkout')
    wheel_pin(source, wheel)
    source['official_wheel'] = {'path': str(wheel), 'sha256': WHEEL_SHA256}
    source['excluded_package_sources'] = [dict(row, reason='Official wheel excludes tracked documentation')
                                          for row in source['package_files']
                                          if package_path(row['path'])[1] in OMITTED_DOCUMENTATION]
    return source


def validate_captured(binding, source, wheel):
    from component_measure import package_files

    expected = wheel_pin(source, wheel)
    package = Path(binding.get('package', ''))
    receipt = Path(binding.get('metadata', ''))
    require(package.is_absolute() and receipt.is_absolute() and receipt.name == 'METADATA'
            and receipt.parent.parent == package.parent, 'Foreign actual Diffusers distribution root')
    require(package_files(package) == binding.get('installed_files_sha256'),
            'Actual Diffusers payload differs from captured installation')
    actual = wheel_distribution_binding(source, wheel, binding, metadata.PathDistribution(receipt.parent),
                                       expected_wheel_sha256=expected)
    for name in ('wheel_sha256', 'distribution_name', 'distribution_version', 'metadata',
                 'metadata_sha256', 'record', 'record_sha256'):
        require(binding.get(name) == actual[name], 'Missing or changed actual Diffusers distribution: ' + name)


def framework_snapshot(root, *, baseline=None):
    from _metric_td_installation import framework_snapshot as component_snapshot

    root = Path(root)
    result = component_snapshot(root, baseline=baseline)
    for name in ('agent/skills/torch-compat-repo-runbook/scripts/_diffusers_installation.py',
                 'tools/run_diffusers_gate.py', 'tools/run_transformers_gate.py'):
        data = (subprocess.check_output(['git', '-C', str(root), 'show', baseline + ':' + name])
                if baseline is not None else (root / name).read_bytes())
        result[name] = hashlib.sha256(data).hexdigest()
    return dict(sorted(result.items()))


def validate_measurement_source(value, baseline):
    root = Path(__file__).resolve().parents[4]
    require(value.get('jittor_commit') == baseline
            and value.get('measurement_source_sha256') == framework_snapshot(root, baseline=baseline),
            'Diffusers measurement differs from immutable committed framework/probe bytes')


def validate_runtime_metadata(value, baseline, expected_platform=None):
    from _metric_td_installation import validate_runtime_metadata as component_runtime

    component_runtime(value, baseline, expected_platform, source_validator=validate_measurement_source)

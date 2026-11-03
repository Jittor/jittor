"""Historical Metrics/TensorDict evidence decoding and legacy source identities.

Current execution/provisioning is maintained by the two independent ecosystem
repositories. This reader remains for existing report/source proofs; Diffusers
also retains the historical source-closure contract when reviewing old runs.
"""

import ast
from functools import lru_cache
import hashlib
from importlib import metadata
import os
from pathlib import Path
import re
import subprocess
import zipfile

from _installed_source_proof import package_path, require, wheel_distribution_binding


PINS = {
    'torchmetrics': ('1.7.4', '986eb900b8bc660ff8174c95ec8f702e0e5cb6ca',
                     'Lightning-AI/torchmetrics',
                     '9298ad0e893b0cf2956bee95b0f7eecdc65205ab84ddb4f6762eff157c240518'),
    'tensordict': ('0.10.0', 'c34264a58b12471dc3e2913ea586b32431872d4c',
                   'pytorch/tensordict',
                   'bccb6a7d7f02a8c83ca41815943bfa284da7d28019ab4276135809bda5ba05a6'),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wheel_pin(source, wheel):
    repo = source['repo']
    version, commit, remote, expected = PINS[repo]
    upstream = source['upstream']
    require(upstream.get('version') == version and upstream.get('commit') == commit
            and upstream.get('remote') in ('git@github.com:' + remote + '.git',
                                           'https://github.com/' + remote + '.git'),
            'Unexpected official component source: ' + repo)
    require(digest(wheel) == expected, 'Component wheel differs from fixed official bytes: ' + repo)
    return expected


def source_descriptor(repo, checkout, wheel):
    from registry_inventory import inventory

    checkout, wheel = Path(checkout).resolve(), Path(wheel).resolve()
    source = inventory(repo, checkout)
    require(not subprocess.check_output(['git', '-C', str(checkout), 'status', '--porcelain']),
            'Dirty official component checkout')
    expected = wheel_pin(source, wheel)
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'Duplicate official component wheel member')
        payload = {name[len(repo) + 1:]: hashlib.sha256(archive.read(name)).hexdigest()
                   for name in names if name.startswith(repo + '/') and not name.endswith('/')}
        known = {package_path(row['path'])[1]: row['sha256'] for row in source['package_files']}
        require(all(payload.get(package_path(row['path'])[1]) == row['sha256']
                    for row in source['package_sources']), 'Official component wheel/source mismatch')
        generated = set(payload) - set(known)
        allowed = {'_version.py', '_C.so'} if repo == 'tensordict' else set()
        require(generated <= allowed, 'Unbound official component package payload')
        require(all(payload[name] == sha for name, sha in known.items() if name in payload),
                'Official component resource/source mismatch')
        source['generated_sources'] = []
        if repo == 'tensordict':
            import tomllib

            declaration = checkout / 'pyproject.toml'
            settings = tomllib.loads(declaration.read_text())['tool']['setuptools_scm']
            require(settings.get('version_file') == 'tensordict/_version.py'
                    and generated == {'_version.py', '_C.so'}, 'Unexpected TensorDict generated payload')
            assignments = {}
            for node in ast.parse(archive.read('tensordict/_version.py')).body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in ('__version__', '__commit_id__'):
                            assignments[target.id] = ast.literal_eval(node.value)
            require(assignments == {'__version__': PINS[repo][0], '__commit_id__': None},
                    'Unexpected official TensorDict generated version')
            source['generated_sources'] = [{
                'path': 'tensordict/_version.py', 'sha256': payload['_version.py'],
                'kind': 'official-wheel setuptools-scm version_file',
                'declaration': {'path': 'pyproject.toml', 'sha256': digest(declaration)},
                'version': assignments['__version__'], 'commit_id': None,
            }]
    source['official_wheel'] = {'path': str(wheel), 'sha256': expected,
                                'package_files_sha256': payload}
    return source


def validate_captured(binding, source, wheel):
    """Use real captured prefixes; do not substitute the auditor's installation."""
    from component_measure import package_files

    expected = wheel_pin(source, wheel)
    package = Path(binding.get('package', ''))
    metadata_path = Path(binding.get('metadata', ''))
    require(package.is_absolute() and metadata_path.is_absolute()
            and metadata_path.name == 'METADATA' and metadata_path.parent.parent == package.parent,
            'Component metadata belongs to another installation')
    require(package_files(package) == binding.get('installed_files_sha256'),
            'Actual component payload differs from captured installation')
    actual = wheel_distribution_binding(source, wheel, binding, metadata.PathDistribution(metadata_path.parent),
                                       expected_wheel_sha256=expected)
    for key in ('wheel_sha256', 'distribution_name', 'distribution_version', 'metadata',
                'metadata_sha256', 'record', 'record_sha256'):
        require(binding.get(key) == actual[key], 'Missing or changed actual component distribution: ' + key)


from _source_layout import source_path

def framework_snapshot(root, *, baseline=None):
    """Reuse the existing complete framework snapshot, adding this probe's tools."""
    from _vllm_installation import framework_sources

    root = Path(root)
    if baseline is not None:
        require(isinstance(baseline, str) and re.fullmatch(r'[0-9a-f]{40}', baseline),
                'Component measurement requires a full historical commit SHA')
    paths = framework_sources(root, baseline=baseline)
    paths += ['tools/ecosystem/_metric_td_installation.py',
              'tools/ecosystem/collect_component_scorecard.py',
              'tools/ecosystem/training_probe.py',
              'tools/ecosystem/verify_repo.py',
              'tools/ecosystem/_runtime_platform.py']
    paths = [source_path(root, name, baseline) for name in paths]
    runtime_helper = source_path(root, 'tools/ecosystem/_component_runtime.py', baseline)
    helper_selection = (['ls-files', '--', runtime_helper] if baseline is None else
                        ['ls-tree', '-r', '--name-only', baseline, '--', runtime_helper])
    helper_tracked = bool(subprocess.check_output(
        ['git', '-C', str(root), *helper_selection], text=True).strip())
    consumer = source_path(root, 'tools/ecosystem/_metric_td_installation.py', baseline)
    consumer_bytes = (subprocess.check_output(['git', '-C', str(root), 'show', baseline + ':' + consumer])
                      if baseline is not None else (root / consumer).read_bytes())
    imports_helper = any(isinstance(node, ast.ImportFrom) and node.module == '_component_runtime'
                         or isinstance(node, ast.Import) and any(alias.name == '_component_runtime'
                                                               for alias in node.names)
                         for node in ast.walk(ast.parse(consumer_bytes, filename=consumer)))
    require(not imports_helper or helper_tracked,
            'Component runtime helper is imported but absent from the selected source inventory')
    if helper_tracked:
        paths.append(runtime_helper)
    # A historical measurement predating the maintained runner retains exactly
    # its original closure. File existence on today's disk cannot choose it.
    tools = ('tools/run_metric_td_gate.py', 'tools/run_transformers_gate.py')
    selection = (['ls-files', '--', *tools] if baseline is None else
                 ['ls-tree', '-r', '--name-only', baseline, '--', *tools])
    tracked = set(subprocess.check_output(
        ['git', '-C', str(root), *selection], text=True).splitlines())
    if tools[0] in tracked:
        require(set(tools) <= tracked, 'Incomplete maintained component runner closure')
        if baseline is None:
            committed = set(subprocess.check_output(
                ['git', '-C', str(root), 'ls-tree', '-r', '--name-only', 'HEAD', '--', *tools],
                text=True).splitlines())
            require(set(tools) <= committed, 'Maintained component runner must be committed')
            for name in tools:
                require((root / name).is_file() and digest(root / name) == hashlib.sha256(
                    subprocess.check_output(['git', '-C', str(root), 'show', 'HEAD:' + name])).hexdigest(),
                    'Missing or changed committed maintained component tool: ' + name)
        paths += list(tools)
    if baseline is not None:
        return dict(_committed_sources(str(root.resolve()), baseline, tuple(sorted(set(paths)))))
    return {name: (hashlib.sha256(os.fsencode(os.readlink(root / name))).hexdigest()
                   if (root / name).is_symlink() else digest(root / name)) for name in sorted(set(paths))}


@lru_cache(maxsize=8)
def _committed_sources(root, baseline, paths):
    # Git commit blobs are immutable. Return tuples so callers cannot mutate
    # canonical cached identities shared by the phase/rank validation calls.
    return tuple((name, hashlib.sha256(subprocess.check_output(
                 ['git', '-C', root, 'show', baseline + ':' + name])).hexdigest()) for name in paths)


def validate_measurement_source(value, baseline):
    require(value.get('jittor_commit') == baseline, 'Component runtime baseline mismatch')
    root = Path(__file__).resolve().parents[2]
    require(value.get('measurement_source_sha256') == framework_snapshot(root, baseline=baseline),
            'Component measurement source differs from committed framework bytes')


def validate_runtime_metadata(value, baseline, expected_platform=None, *, source_validator=None):
    """Read historical component evidence; current execution lives downstream."""
    from _component_runtime import validate_runtime_metadata as validate

    return validate(value, baseline, expected_platform,
                    source_validator=source_validator or validate_measurement_source)

#!/usr/bin/env python3
"""Provision isolated pinned CPU downstream gates, then run the maintained nox session.

All mutable files belong to --state-root, outside the checkout. The wheelhouse,
resolved package lock and command logs are retained for replay and diagnosis.
Selected distributions' base dependency closures are resolved and checked;
optional downstream feature or backend extras are not enabled.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'agent/skills/torch-compat-repo-runbook/scripts'
TORCH = '2.11.0+cpu'
SWIFT_SHA = 'ff5128777dd3dc5538f1a95fbcc29442c327d7d2'
SWIFT_REMOTE = 'https://github.com/modelscope/ms-swift.git'
VERL_SHA = 'e5a627cf55c94d85406556854c2c1a22db4ff5e7'
VERL_REMOTE = 'https://github.com/volcengine/verl.git'
COMMON = ('numpy==1.26.4', 'accelerate==1.10.1', 'huggingface-hub==0.35.3',
          'tokenizers==0.22.2', 'safetensors==0.8.0', 'astunparse==1.6.3',
          'pillow==11.0.0', 'tqdm==4.67.1')
SWIFT_DEPS = ('datasets==3.6.0', 'pyarrow==19.0.1', 'modelscope==1.33.0', 'addict==2.4.0',
              'dacite==1.9.2', 'einops==0.8.1', 'pandas==2.2.3',
              'simplejson==4.2.0', 'sentencepiece==0.2.1', 'rouge==1.0.1', 'trl==0.21.0')
# The two registered VERL components use the base distribution's complete
# Requires-Dist closure. Do not enable the separate CPU/backend extras, which
# describe full trainers and a different Torch profile.
VERL_DEPS = ('numpy==2.2.6', 'transformers==5.5.3', 'ray[default]==2.54.1',
             'tensordict==0.10.0', 'huggingface-hub==1.5.0', 'tokenizers==0.22.2',
             'safetensors==0.8.0', 'peft==0.17.1', 'accelerate==1.10.1', 'datasets==3.6.0',
             'pandas==2.2.3', 'pyarrow==19.0.1', 'astunparse==1.6.3',
             'pillow==11.0.0', 'tqdm==4.67.1')
# The official Torch 2.11 CPU wheel requires setuptools<82. Canonical Stage 2
# tools keep setuptools==83 in a separate controller/build environment.
RUNTIME_SETUPTOOLS = 'setuptools==80.9.0'
OPENMMLAB_DEPS = ('numpy==1.26.4', 'opencv-python==4.11.0.86',
                  'astunparse==1.6.3', 'pillow==11.0.0', 'tqdm==4.67.1')
DIFFUSERS_DEPS = ('numpy==1.26.4', 'huggingface-hub==0.35.3',
                  'astunparse==1.6.3', 'pillow==11.0.0', 'tqdm==4.67.1')
METRIC_TD_DEPS = ('numpy==1.26.4', 'astunparse==1.6.3',
                  'pillow==11.0.0', 'tqdm==4.67.1')


def runtime_roots(library):
    if library == 'metric_tensordict':
        return list(METRIC_TD_DEPS)
    if library == 'diffusers':
        return list(DIFFUSERS_DEPS)
    if library == 'verl':
        return list(VERL_DEPS)
    if library == 'openmmlab':
        return list(OPENMMLAB_DEPS)
    require(library in ('peft', 'swift', 'transformers'), 'Unknown maintained library profile')
    return list(COMMON) + (list(SWIFT_DEPS) if library == 'swift' else [])


def session_name(library):
    require(library in ('peft', 'swift', 'verl', 'transformers', 'openmmlab', 'diffusers',
                        'metric_tensordict'), 'Unknown maintained library profile')
    return {'verl': 'ecosystem_verl_training',
            'metric_tensordict': 'ecosystem_metric_tensordict_source',
            'diffusers': 'ecosystem_diffusers_source',
            'transformers': 'ecosystem_transformers_source',
            'openmmlab': 'ecosystem_openmmlab_installed'}.get(library, 'ecosystem_' + library)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def external(path):
    path = Path(path).expanduser().resolve()
    require(path != ROOT and ROOT not in path.parents,
            'State, caches, checkouts and environments must be outside the source checkout')
    return path


def environment(state):
    # Do not inherit an editable source, package overlay, shim or selected GPU.
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(('JITTOR_', 'JT_', 'VERL_')) or key in (
                'PYTHONPATH', 'PYTHONHOME', 'REAL_TORCH_SITE', 'REAL_TORCH_PYTHON',
                'cache_name', 'use_cuda', 'nvcc_path', 'backend_fallback', 'VIRTUAL_ENV',
                'PYTEST_ADDOPTS', 'PIP_INDEX_URL', 'PIP_EXTRA_INDEX_URL', 'PIP_FIND_LINKS',
                'PIP_CONFIG_FILE', 'PIP_TRUSTED_HOST'):
            env.pop(key)
    for name, folder in (('HOME', 'home'), ('TMPDIR', 'tmp'), ('XDG_CACHE_HOME', 'xdg'),
                         ('PIP_CACHE_DIR', 'pip-cache'), ('HF_HOME', 'huggingface'),
                         ('TORCH_HOME', 'torch-home'), ('JITTOR_HOME', 'unit-cache')):
        path = state / folder
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    env.update(PYTHONPATH='', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               JITTOR_TORCH_SHIM='0', CUDA_VISIBLE_DEVICES='', JT_BACKEND='cpu',
               JT_USE_CUDA='0', use_cuda='0', JT_BUILD_NVCC_PATH='', nvcc_path='',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', USE_TF='0', USE_FLAX='0',
               PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_NO_INPUT='1',
               PIP_CONFIG_FILE=os.devnull,
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    return env


class Commands:
    def __init__(self, state, env):
        self.state, self.env = state, env
        history = state / 'commands.json'
        self.rows = json.loads(history.read_text()) if history.exists() else []

    def run(self, *command, cwd=ROOT):
        command = [str(value) for value in command]
        log = self.state / 'logs' / ('%03d.log' % len(self.rows))
        log.parent.mkdir(parents=True, exist_ok=True)
        row = {'argv': command, 'cwd': str(cwd), 'log': str(log.relative_to(self.state))}
        self.rows.append(row)
        print('Run', len(self.rows), command[:4], flush=True)
        with log.open('w') as stream:
            result = subprocess.run(command, cwd=cwd, env=self.env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=7200)
        row['returncode'] = result.returncode
        (self.state / 'commands.json').write_text(json.dumps(self.rows, indent=2) + '\n')
        require(result.returncode == 0, 'Command failed; see ' + str(log))
        return log


def checkout(commands, directory, remote, sha, cache=None):
    require(len(sha) == 40, 'Full upstream commit required')
    if cache is not None:
        def cached_git(*args):
            return subprocess.check_output(['git', '-C', str(cache), *args], text=True).strip()
        cached_remote = cached_git('remote', 'get-url', 'origin')
        cached_remote = cached_remote.replace('git@github.com:', 'https://github.com/')
        require(cached_remote == remote and cached_git('rev-parse', 'HEAD') == sha
                and not cached_git('status', '--porcelain'),
                'Read-only Git cache must have the exact clean official source')
    if not directory.exists():
        commands.run('git', 'init', directory)
        commands.run('git', '-C', directory, 'remote', 'add', 'origin', remote)
        commands.run('git', '-C', directory, 'fetch', '--depth', '1', cache or 'origin', sha)
        commands.run('git', '-C', directory, 'checkout', '--detach', 'FETCH_HEAD')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(directory), *args], text=True).strip()
    require(git('remote', 'get-url', 'origin') == remote and git('rev-parse', 'HEAD') == sha
            and not git('status', '--porcelain'), 'Upstream must be clean and exactly pinned')


def official_wheel(repo, version, expected_hash, folder):
    with urllib.request.urlopen('https://pypi.org/pypi/%s/%s/json' % (repo, version), timeout=60) as stream:
        release = json.load(stream)
    candidates = [item for item in release['urls'] if item['packagetype'] == 'bdist_wheel'
                  and item['digests']['sha256'] == expected_hash]
    require(len(candidates) == 1, 'Missing exact pinned official wheel: ' + repo)
    item = candidates[0]
    require(item['url'].startswith('https://files.pythonhosted.org/'), 'Unexpected wheel host')
    wheel = folder / item['filename']
    if not wheel.exists():
        temporary = wheel.with_suffix('.download')
        with urllib.request.urlopen(item['url'], timeout=120) as stream, temporary.open('wb') as out:
            shutil.copyfileobj(stream, out)
        require(digest(temporary) == expected_hash, 'Downloaded wheel SHA256 mismatch')
        temporary.rename(wheel)
    require(digest(wheel) == expected_hash, 'Pinned wheel bytes changed')
    return wheel


def install_wheels(commands, candidate, oracle, wheels, requirements):
    all_wheels = sorted(Path(wheels).glob('*.whl'))
    torch_wheels = [path for path in all_wheels if path.name.startswith('torch-')]
    require(len(torch_wheels) == 1 and torch_wheels[0].name.startswith('torch-' + TORCH),
            'Exactly one pinned official CPU Torch wheel required')
    require('torch==' + TORCH in requirements, 'Pinned CPU Torch must remain an install root')
    # Real distribution metadata is required by downstream availability checks.
    # The candidate activates the official shim only at gate execution; the
    # oracle stays independent. Never synthesize Torch dist-info.
    for interpreter in (oracle, candidate):
        commands.run(interpreter, '-m', 'pip', 'install', '--no-index',
                     '--find-links', wheels, *requirements)


def unpack_pinned_wheel(wheel, site):
    """Retain the official METADATA/RECORD bytes required by the existing gate.

    These two purelib wheels contain no scripts or .data relocation scheme.
    Dependencies were provisioned separately, with exactly the same wheels.
    """
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'Duplicate wheel member')
        for name in names:
            relative = Path(name)
            require(not relative.is_absolute() and '..' not in relative.parts and '\\' not in name
                    and not any(part.endswith('.data') for part in relative.parts),
                    'Unsafe or non-purelib pinned wheel layout')
            mode = (archive.getinfo(name).external_attr >> 16) & 0o170000
            require(mode != 0o120000,
                    'Symlinks are not permitted in pinned purelib wheels')
        archive.extractall(site)


def canonical_tools(selected):
    pins = [line.strip() for line in (ROOT / 'requirements/dev-tools.txt').read_text().splitlines()
            if line.split('==')[0].strip().lower() in selected]
    require(len(pins) == len(selected), 'Missing canonical runtime test pins')
    return pins


def runtime_tools():
    return canonical_tools({'pytest', 'pytest-timeout', 'pytest-xdist', 'scipy', 'wheel'})


def controller_tools():
    return canonical_tools({'nox', 'build', 'setuptools', 'wheel'})


def stage_build_source(source, build):
    """Preserve tracked symlink bytes, including upstream documentation links."""
    if not build.exists():
        shutil.copytree(source, build, symlinks=True, ignore=shutil.ignore_patterns('.git'))


def metric_td_checkouts(commands, state, source_cache, wheels):
    """Fetch real official source identities, including TensorDict's SCM tag."""
    from _metric_td_installation import PINS, source_descriptor

    checkouts, pinned, sources = {}, {}, {}
    for repo, (version, sha, remote, wheel_hash) in PINS.items():
        path = state / 'upstream' / repo
        cache = source_cache / repo if source_cache else None
        checkout(commands, path, 'https://github.com/' + remote + '.git', sha, cache)
        if repo == 'tensordict':
            tag = 'refs/tags/v' + version
            commands.run('git', '-C', path, 'fetch', '--depth', '1', cache or 'origin', tag + ':' + tag)
            peeled = subprocess.check_output(
                ['git', '-C', str(path), 'rev-parse', tag + '^{}'], text=True).strip()
            require(peeled == sha, 'Official TensorDict release tag differs from the pinned source')
        checkouts[repo] = path
        pinned[repo] = official_wheel(repo, version, wheel_hash, wheels)
        sources[repo] = source_descriptor(repo, path, pinned[repo])
    return checkouts, pinned, sources


def provision(library, state, python, source_cache=None, wheel_cache=None):
    require(sys.version_info[:2] == (3, 11), 'Nightly provisioning requires Python 3.11')
    require(not (state / 'provision.json').exists(), 'Use a new state root after successful provisioning')
    env = environment(state)
    if library == 'verl':
        env['VERL_USE_EXTERNAL_PLUGINS'] = 'none'
    elif library == 'openmmlab':
        env.update(MMCV_WITH_OPS='0', MMENGINE_LITE='0')
    commands = Commands(state, env)
    cached = ['--find-links', wheel_cache] if wheel_cache else []
    wheels = state / 'wheels'
    wheels.mkdir(exist_ok=True)
    candidate, oracle, controller = (state / side / 'bin/python' for side in ('candidate', 'oracle', 'controller'))
    for interpreter in (candidate, oracle, controller):
        if not interpreter.exists():
            commands.run(python, '-m', 'venv', interpreter.parents[1])
    sys.path.insert(0, str(SCRIPTS))
    checkouts, pinned = {}, {}
    if library == 'metric_tensordict':
        checkouts, pinned, sources = metric_td_checkouts(commands, state, source_cache, wheels)
    elif library == 'diffusers':
        from _diffusers_installation import COMMIT, VERSION, WHEEL_SHA256, source_descriptor
        path = state / 'upstream/diffusers'
        checkout(commands, path, 'https://github.com/huggingface/diffusers.git', COMMIT,
                 source_cache / 'diffusers' if source_cache else None)
        checkouts['diffusers'] = path
        pinned['diffusers'] = official_wheel('diffusers', VERSION, WHEEL_SHA256, wheels)
        sources = source_descriptor(path, pinned['diffusers'])
    elif library == 'verl':
        from registry_inventory import inventory
        path = state / 'upstream/verl'
        checkout(commands, path, VERL_REMOTE, VERL_SHA,
                 source_cache / 'verl' if source_cache else None)
        checkouts['verl'] = path
        sources = inventory('verl', path)
    elif library == 'openmmlab':
        from _openmmlab_component_contract import UPSTREAM_COMMITS
        from registry_inventory import inventory
        for repo, sha in UPSTREAM_COMMITS.items():
            path = state / 'upstream' / repo
            checkout(commands, path, 'https://github.com/open-mmlab/' + repo + '.git', sha,
                     source_cache / repo if source_cache else None)
            checkouts[repo] = path
        sources = {repo: inventory(repo, path) for repo, path in checkouts.items()}
    else:
        from _peft_scorecard import PINS, source_descriptor
        selected = {'transformers': PINS['transformers']} if library == 'transformers' else PINS
        for repo, (version, sha, remote, wheel_hash) in selected.items():
            path = state / 'upstream' / repo
            checkout(commands, path, 'https://github.com/' + remote + '.git', sha,
                     source_cache / repo if source_cache else None)
            checkouts[repo] = path
            pinned[repo] = official_wheel(repo, version, wheel_hash, wheels)
        if library == 'transformers':
            from _transformers_scorecard import source_descriptor as transformers_source
            sources = transformers_source(checkouts['transformers'], pinned['transformers'])
        else:
            sources = source_descriptor(checkouts, pinned)
    (state / 'upstream-proof.json').write_text(json.dumps(sources, indent=2) + '\n')
    # Both private runtimes receive complete real distribution metadata. Only
    # the candidate later activates the official Jittor Torch shim.
    commands.run(oracle, '-m', 'pip', 'download', '--index-url', 'https://download.pytorch.org/whl/cpu',
                 '--only-binary=:all:', '--no-deps', '--dest', wheels, *cached, 'torch==' + TORCH)
    tool_wheels = state / 'tool-wheels'
    tool_wheels.mkdir(exist_ok=True)
    commands.run(controller, '-m', 'pip', 'download', '--index-url', 'https://pypi.org/simple',
                 '--only-binary=:all:', '--dest', tool_wheels, *cached, *controller_tools())
    commands.run(controller, '-m', 'pip', 'install', '--no-index', '--no-deps', *sorted(tool_wheels.glob('*.whl')))
    if library == 'openmmlab':
        # The original setup.py files import pkg_resources. Keep the canonical
        # controller on setuptools 83; build in the pinned compatible runtime.
        commands.run(oracle, '-m', 'pip', 'install', '--index-url', 'https://pypi.org/simple',
                     *cached, RUNTIME_SETUPTOOLS, 'wheel==0.45.1')
        for repo, source in checkouts.items():
            build = state / (repo + '-build-source')
            stage_build_source(source, build)
            commands.run(oracle, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
                         '--wheel-dir', wheels, build)
            pattern = 'mmengine-0.10.7-*.whl' if repo == 'mmengine' else 'mmcv_lite-2.2.0-*.whl'
            built = list(wheels.glob(pattern))
            require(len(built) == 1, 'Expected one source-built base wheel: ' + repo)
            pinned[repo] = built[0]
    built_source, built_wheel = None, None
    if library in ('swift', 'verl'):
        if library == 'swift':
            built_source = state / 'upstream/ms-swift'
            checkout(commands, built_source, SWIFT_REMOTE, SWIFT_SHA,
                     source_cache / 'ms-swift' if source_cache else None)
        else:
            built_source = checkouts['verl']
        build = state / (library + '-build-source')
        if not build.exists():
            shutil.copytree(built_source, build, ignore=shutil.ignore_patterns('.git'))
        commands.run(controller, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
                     '--wheel-dir', wheels, build)
        pattern = 'ms_swift-4.5.2-*.whl' if library == 'swift' else 'verl-0.10.0.dev0-*.whl'
        built = list(wheels.glob(pattern))
        require(len(built) == 1, 'Expected one pinned source-built wheel: ' + library)
        built_wheel = built[0]
        if library == 'swift':
            from registry_inventory import inventory
            swift_inventory = inventory('ms-swift', built_source)
            (state / 'swift-inventory.json').write_text(json.dumps(swift_inventory, indent=2) + '\n')
    # Build source-only dependency wheels using the compatible runtime build
    # tools. The complete selected distributions (including Swift METADATA)
    # participate in resolution. Installation resolves the same roots offline:
    # a wheelhouse may also contain superseded versions from an earlier attempt.
    # pip check below must pass in every venv.
    commands.run(oracle, '-m', 'pip', 'install', '--index-url', 'https://pypi.org/simple',
                 *cached, RUNTIME_SETUPTOOLS, 'wheel==0.45.1')
    roots = runtime_roots(library)
    requirements = [*pinned.values(), *([built_wheel] if built_wheel else []),
                    'torch==' + TORCH, RUNTIME_SETUPTOOLS, *runtime_tools(), *roots]
    commands.run(oracle, '-m', 'pip', 'wheel', '--index-url', 'https://pypi.org/simple',
                 '--no-build-isolation', '--wheel-dir', wheels, '--find-links', wheels,
                 *cached, *requirements)
    install_wheels(commands, candidate, oracle, wheels, requirements)
    # The historical PEFT/TF producer requires archive RECORD bytes. The MM
    # Diffusers and Metrics/TensorDict accept legitimate pip RECORD entries; retain
    # their real receipts without extracting over the installed distributions.
    if library not in ('openmmlab', 'diffusers', 'metric_tensordict'):
        for side in ('candidate', 'oracle'):
            site = state / side / 'lib/python3.11/site-packages'
            for wheel in pinned.values():
                unpack_pinned_wheel(wheel, site)
    commands.run(candidate, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation',
                 '--config-settings', 'editable_mode=compat', '-e', ROOT)
    commands.run(candidate, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '-e', ROOT / 'compat')
    for interpreter in (candidate, oracle, controller):
        commands.run(interpreter, '-m', 'pip', 'check')
    commands.run(oracle, '-c', "import torch; assert not hasattr(torch, '_torch_compat_install_context'); "
                 "assert torch.__version__ == '" + TORCH + "'; assert not torch.cuda.is_available(); "
                 "assert torch._C.__file__; print(torch.__version__, torch._C.__file__)")
    if library == 'swift':
        # PyArrow 26's binary requires NumPy 2 despite its unconstrained wheel
        # metadata. Exercise the pinned data ABI before paying for first JIT.
        commands.run(oracle, '-c', "import numpy as np, pyarrow as pa, datasets; "
                     "assert np.__version__ == '1.26.4'; assert pa.__version__ == '19.0.1'; "
                     "assert datasets.__version__ == '3.6.0'; "
                     "x=np.arange(7, dtype=np.float32); np.testing.assert_array_equal(pa.array(x).to_numpy(), x); "
                     "assert len(datasets.Dataset.from_dict({'values': [1, 2]})) == 2; "
                     "print('Data ABI', np.__version__, pa.__version__, datasets.__version__)")
    elif library == 'verl':
        commands.run(oracle, '-c', "import importlib.metadata as m; "
                     "import numpy, pyarrow, datasets, transformers, ray, tensordict, verl; "
                     "from packaging.version import Version; "
                     "assert numpy.__version__ == '2.2.6'; assert transformers.__version__ == '5.5.3'; "
                     "assert ray.__version__ == '2.54.1'; assert tensordict.__version__ == '0.10.0'; "
                     "assert verl.__version__ == '0.10.0.dev'; assert m.version('verl') == '0.10.0.dev0'; "
                     "assert Version(verl.__version__) == Version(m.version('verl')); "
                     "assert pyarrow.__version__ == '19.0.1'; "
                     "x=numpy.arange(7, dtype=numpy.float32); "
                     "numpy.testing.assert_array_equal(pyarrow.array(x).to_numpy(), x); "
                     "assert len(datasets.Dataset.from_dict({'values': [1, 2]})) == 2; "
                     "print('VERL base component profile', verl.__version__, m.version('verl'))")
    commands.run(candidate, '-c', "import importlib.metadata as m; assert m.version('torch') == '" + TORCH + "'")
    for side, interpreter in (('candidate', candidate), ('oracle', oracle), ('controller', controller)):
        log = commands.run(interpreter, '-m', 'pip', 'list', '--format=freeze')
        (state / ('resolved-lock-' + side + '.txt')).write_bytes(log.read_bytes())
    env.update(JITTOR_CI_PYTHON=str(candidate), REAL_TORCH_PYTHON=str(oracle),
               JITTOR_LAB_ROOT=str(state / 'lab'), JITTOR_NOX_SHARED_CACHE='0',
               JITTOR_TEST_DEVICES='cpu', JITTOR_SOURCE_ROOT=str(ROOT),
               PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTEST_PLUGINS='pytest_timeout,xdist.plugin')
    for repo in pinned:
        env['JITTOR_' + repo.upper() + '_CHECKOUT'] = str(checkouts[repo])
        env['JITTOR_' + repo.upper() + '_WHEEL'] = str(pinned[repo])
    if library == 'swift':
        env.update(JITTOR_SWIFT_UPSTREAM=str(built_source), JITTOR_SWIFT_WHEEL=str(built_wheel),
                   JITTOR_ECOSYSTEM_PACKAGE_SITE=str(state / 'candidate/lib/python3.11/site-packages'),
                   JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE=str(state / 'oracle/lib/python3.11/site-packages'))
    elif library == 'verl':
        env.update(JITTOR_VERL_UPSTREAM=str(built_source), JITTOR_VERL_WHEEL=str(built_wheel),
                   JITTOR_ECOSYSTEM_PACKAGE_SITE=str(state / 'candidate/lib/python3.11/site-packages'),
                   JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE=str(state / 'oracle/lib/python3.11/site-packages'),
                   VERL_USE_EXTERNAL_PLUGINS='none')
    if library == 'metric_tensordict':
        env['JITTOR_METRIC_TD_EVIDENCE_DIR'] = str(state / 'maintained-output')
    elif library == 'diffusers':
        env['JITTOR_DIFFUSERS_EVIDENCE_DIR'] = str(state / 'maintained-output')
    elif library == 'openmmlab':
        env.update(JITTOR_OPENMMLAB_EVIDENCE_DIR=str(state / 'maintained-output'),
                   JITTOR_ECOSYSTEM_PACKAGE_SITE=str(state / 'candidate/lib/python3.11/site-packages'),
                   JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE=str(state / 'oracle/lib/python3.11/site-packages'))
    manifest = {'library': library, 'source_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'wheels': {path.name: digest(path) for path in sorted(wheels.glob('*.whl'))},
        'tool_wheels': {path.name: digest(path) for path in sorted(tool_wheels.glob('*.whl'))},
        'source_clean': not bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
        'provisioner_sha256': digest(__file__),
        'session': session_name(library), 'oracle_torch': TORCH,
        'upstream_transport': 'read-only-local-git-cache' if source_cache else 'official-remote',
        'source_cache': str(source_cache) if source_cache else None,
        'wheel_cache': str(wheel_cache) if wheel_cache else None,
        'environment': {key: value for key, value in env.items() if key.startswith(('JITTOR_', 'JT_'))
                        or key in ('HOME', 'TMPDIR', 'CUDA_VISIBLE_DEVICES', 'REAL_TORCH_PYTHON',
                                   'VERL_USE_EXTERNAL_PLUGINS')}}
    (state / 'provision.json').write_text(json.dumps(manifest, indent=2) + '\n')
    commands.env = env
    return commands, controller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', choices=('peft', 'swift', 'verl', 'transformers', 'openmmlab',
                                             'diffusers', 'metric_tensordict'), required=True)
    parser.add_argument('--state-root', type=Path, required=True)
    parser.add_argument('--python', default=sys.executable, help='Healthy Python 3.11 for new private venvs')
    parser.add_argument('--source-cache', type=Path,
                        help='Optional read-only exact clean official Git checkouts for local replay')
    parser.add_argument('--wheel-cache', type=Path,
                        help='Optional read-only wheelhouse for replay; dependency resolution and checks remain required')
    parser.add_argument('--run', action='store_true', help='Run the complete maintained nox gate after provisioning')
    args = parser.parse_args()
    state = external(args.state_root)
    state.mkdir(parents=True, exist_ok=True)
    commands, python = provision(args.library, state, args.python, args.source_cache,
                                 external(args.wheel_cache) if args.wheel_cache else None)
    if args.run:
        require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(),
                'Formal nightly execution requires clean measured source')
        commands.run(python, '-m', 'nox', '-s', session_name(args.library))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

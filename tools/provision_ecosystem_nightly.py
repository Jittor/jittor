#!/usr/bin/env python3
"""Provision isolated pinned CPU PEFT/Swift gates, then run the maintained nox session.

All mutable files belong to --state-root, outside the checkout. The wheelhouse,
resolved package lock and command logs are retained for replay and diagnosis.
Selected distributions' base dependency closures are resolved and checked;
optional Swift feature extras are not enabled.
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
COMMON = ('numpy==1.26.4', 'accelerate==1.10.1', 'huggingface-hub==0.35.3',
          'tokenizers==0.22.2', 'safetensors==0.8.0', 'astunparse==1.6.3',
          'pillow==11.0.0', 'tqdm==4.67.1')
SWIFT_DEPS = ('datasets==3.6.0', 'modelscope==1.33.0', 'addict==2.4.0',
              'dacite==1.9.2', 'einops==0.8.1', 'pandas==2.2.3',
              'simplejson==4.2.0', 'sentencepiece==0.2.1', 'rouge==1.0.1', 'trl==0.21.0')
# The official Torch 2.11 CPU wheel requires setuptools<82. Canonical Stage 2
# tools keep setuptools==83 in a separate controller/build environment.
RUNTIME_SETUPTOOLS = 'setuptools==80.9.0'


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
        if key.startswith(('JITTOR_', 'JT_')) or key in (
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


def install_wheels(commands, candidate, oracle, wheels, preserved=()):
    all_wheels = sorted(Path(wheels).glob('*.whl'))
    torch_wheels = [path for path in all_wheels if path.name.startswith('torch-')]
    require(len(torch_wheels) == 1 and torch_wheels[0].name.startswith('torch-' + TORCH),
            'Exactly one pinned official CPU Torch wheel required')
    packages = [path for path in all_wheels if path not in preserved]
    # Real distribution metadata is required by downstream availability checks.
    # The candidate activates the official shim only at gate execution; the
    # oracle stays independent. Never synthesize Torch dist-info.
    for interpreter in (oracle, candidate):
        commands.run(interpreter, '-m', 'pip', 'install', '--no-index', '--no-deps', *packages)


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


def runtime_tools():
    selected = {'pytest', 'pytest-timeout', 'pytest-xdist', 'scipy', 'wheel'}
    pins = [line.strip() for line in (ROOT / 'requirements/dev-tools.txt').read_text().splitlines()
            if line.split('==')[0].strip().lower() in selected]
    require(len(pins) == len(selected), 'Missing canonical runtime test pins')
    return pins


def provision(library, state, python, source_cache=None):
    require(sys.version_info[:2] == (3, 11), 'Nightly provisioning requires Python 3.11')
    require(not (state / 'provision.json').exists(), 'Use a new state root after successful provisioning')
    env = environment(state)
    commands = Commands(state, env)
    wheels = state / 'wheels'
    wheels.mkdir(exist_ok=True)
    candidate, oracle, controller = (state / side / 'bin/python' for side in ('candidate', 'oracle', 'controller'))
    for interpreter in (candidate, oracle, controller):
        if not interpreter.exists():
            commands.run(python, '-m', 'venv', interpreter.parents[1])
    sys.path.insert(0, str(SCRIPTS))
    from _peft_scorecard import PINS, source_descriptor
    checkouts, pinned = {}, {}
    for repo, (version, sha, remote, wheel_hash) in PINS.items():
        path = state / 'upstream' / repo
        checkout(commands, path, 'https://github.com/' + remote + '.git', sha,
                 source_cache / repo if source_cache else None)
        checkouts[repo] = path
        pinned[repo] = official_wheel(repo, version, wheel_hash, wheels)
    sources = source_descriptor(checkouts, pinned)
    (state / 'upstream-proof.json').write_text(json.dumps(sources, indent=2) + '\n')
    # Both private runtimes receive complete real distribution metadata. Only
    # the candidate later activates the official Jittor Torch shim.
    commands.run(oracle, '-m', 'pip', 'download', '--index-url', 'https://download.pytorch.org/whl/cpu',
                 '--only-binary=:all:', '--no-deps', '--dest', wheels, 'torch==' + TORCH)
    tool_wheels = state / 'tool-wheels'
    tool_wheels.mkdir(exist_ok=True)
    commands.run(controller, '-m', 'pip', 'download', '--index-url', 'https://pypi.org/simple',
                 '--only-binary=:all:', '--dest', tool_wheels, '-r', ROOT / 'requirements/dev-tools.txt')
    commands.run(controller, '-m', 'pip', 'install', '--no-index', '--no-deps', *sorted(tool_wheels.glob('*.whl')))
    swift_source, swift_wheel = None, None
    if library == 'swift':
        swift_source = state / 'upstream/ms-swift'
        checkout(commands, swift_source, SWIFT_REMOTE, SWIFT_SHA,
                 source_cache / 'ms-swift' if source_cache else None)
        build = state / 'swift-build-source'
        if not build.exists():
            shutil.copytree(swift_source, build, ignore=shutil.ignore_patterns('.git'))
        commands.run(controller, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
                     '--wheel-dir', wheels, build)
        built = list(wheels.glob('ms_swift-4.5.2-*.whl'))
        require(len(built) == 1, 'Expected one Swift 4.5.2 source-built wheel')
        swift_wheel = built[0]
        from registry_inventory import inventory
        swift_inventory = inventory('ms-swift', swift_source)
        (state / 'swift-inventory.json').write_text(json.dumps(swift_inventory, indent=2) + '\n')
    # Build source-only dependency wheels using the compatible runtime build
    # tools. The complete selected distributions (including Swift METADATA)
    # participate in resolution; --no-deps is only used for the resulting
    # complete wheel closure, and pip check below must pass in every venv.
    commands.run(oracle, '-m', 'pip', 'install', '--index-url', 'https://pypi.org/simple',
                 RUNTIME_SETUPTOOLS, 'wheel==0.45.1')
    roots = list(COMMON) + (list(SWIFT_DEPS) if library == 'swift' else [])
    commands.run(oracle, '-m', 'pip', 'wheel', '--index-url', 'https://pypi.org/simple',
                 '--no-build-isolation', '--wheel-dir', wheels, '--find-links', wheels,
                 *pinned.values(), *([swift_wheel] if swift_wheel else []),
                 'torch==' + TORCH, RUNTIME_SETUPTOOLS, *runtime_tools(), *roots)
    install_wheels(commands, candidate, oracle, wheels, preserved=tuple(pinned.values()))
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
    commands.run(candidate, '-c', "import importlib.metadata as m; assert m.version('torch') == '" + TORCH + "'")
    for side, interpreter in (('candidate', candidate), ('oracle', oracle), ('controller', controller)):
        log = commands.run(interpreter, '-m', 'pip', 'list', '--format=freeze')
        (state / ('resolved-lock-' + side + '.txt')).write_bytes(log.read_bytes())
    env.update(JITTOR_CI_PYTHON=str(candidate), REAL_TORCH_PYTHON=str(oracle),
               JITTOR_LAB_ROOT=str(state / 'lab'), JITTOR_NOX_SHARED_CACHE='0',
               JITTOR_TEST_DEVICES='cpu', JITTOR_SOURCE_ROOT=str(ROOT),
               PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTEST_PLUGINS='pytest_timeout,xdist.plugin')
    for repo in PINS:
        env['JITTOR_' + repo.upper() + '_CHECKOUT'] = str(checkouts[repo])
        env['JITTOR_' + repo.upper() + '_WHEEL'] = str(pinned[repo])
    if library == 'swift':
        env.update(JITTOR_SWIFT_UPSTREAM=str(swift_source), JITTOR_SWIFT_WHEEL=str(swift_wheel),
                   JITTOR_ECOSYSTEM_PACKAGE_SITE=str(state / 'candidate/lib/python3.11/site-packages'),
                   JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE=str(state / 'oracle/lib/python3.11/site-packages'))
    manifest = {'library': library, 'source_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'wheels': {path.name: digest(path) for path in sorted(wheels.glob('*.whl'))},
        'tool_wheels': {path.name: digest(path) for path in sorted(tool_wheels.glob('*.whl'))},
        'session': 'ecosystem_' + library, 'oracle_torch': TORCH,
        'upstream_transport': 'read-only-local-git-cache' if source_cache else 'official-remote',
        'source_cache': str(source_cache) if source_cache else None,
        'environment': {key: value for key, value in env.items() if key.startswith(('JITTOR_', 'JT_'))
                        or key in ('HOME', 'TMPDIR', 'CUDA_VISIBLE_DEVICES', 'REAL_TORCH_PYTHON')}}
    (state / 'provision.json').write_text(json.dumps(manifest, indent=2) + '\n')
    commands.env = env
    return commands, controller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', choices=('peft', 'swift'), required=True)
    parser.add_argument('--state-root', type=Path, required=True)
    parser.add_argument('--python', default=sys.executable, help='Healthy Python 3.11 for new private venvs')
    parser.add_argument('--source-cache', type=Path,
                        help='Optional read-only exact clean official Git checkouts for local replay')
    parser.add_argument('--run', action='store_true', help='Run the complete maintained nox gate after provisioning')
    args = parser.parse_args()
    state = external(args.state_root)
    state.mkdir(parents=True, exist_ok=True)
    commands, python = provision(args.library, state, args.python, args.source_cache)
    if args.run:
        require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(),
                'Formal nightly execution requires clean measured source')
        commands.run(python, '-m', 'nox', '-s', 'ecosystem_' + args.library)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

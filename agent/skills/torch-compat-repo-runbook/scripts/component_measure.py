"""Run a maintained component probe and bind the files loaded by that interpreter.

Use a task-owned executable launcher (one per interpreter/repository) that runs
``python component_measure.py --root ROOT --repo REPO --checkout UPSTREAM --``
before forwarding the arguments supplied by verify_repo. Only the shared runner
and training_probe are accepted. This adds provenance; it does not grant a level.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def installed_distribution(source, module):
    package = Path(module.__file__).resolve().parent
    expected = {row['path']: row['sha256'] for row in source['package_sources']}
    actual = {}
    for name, wanted in expected.items():
        relative = Path(name)
        if relative.parts[0] == 'src':
            relative = Path(*relative.parts[1:])
        if relative.parts[0] != source['repo']:
            raise ValueError('Unexpected upstream package path: ' + name)
        path = package.joinpath(*relative.parts[1:])
        if not path.is_file() or digest(path) != wanted:
            raise ValueError('Installed source differs from clean upstream: ' + name)
        actual[name] = digest(path)
    version = str(module.__version__)
    if version != source['upstream']['version']:
        raise ValueError('Installed release version differs from upstream')
    files = {path.relative_to(package).as_posix(): digest(path)
             for path in sorted(package.rglob('*'))
             if path.is_file() and '__pycache__' not in path.parts
             and path.suffix not in ('.pyc', '.pyo')}
    return {'repo': source['repo'], 'version': version,
            'source_sha256': actual, 'installed_files_sha256': files,
            'package': str(package)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repo', choices=('torchmetrics', 'tensordict'), required=True)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('probe', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.probe[1:] if args.probe[:1] == ['--'] else args.probe
    root = args.root.resolve()
    script = Path(command[0]).resolve() if command else None
    allowed = {root / 'compat/tests/torch/_ecosystem_runner.py',
               root / 'agent/skills/torch-compat-repo-runbook/scripts/training_probe.py'}
    if script not in allowed:
        parser.error('Only the maintained ecosystem runner or training probe is allowed')
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from registry_inventory import inventory
    source = inventory(args.repo, args.checkout)
    baseline = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Measurements require a clean source checkout')
    sys.path.insert(0, str(script.parent))
    sys.argv = command
    captured = io.StringIO()
    with redirect_stdout(captured):
        runpy.run_path(str(script), run_name='__main__')
    module = importlib.import_module(args.repo)
    distribution = installed_distribution(source, module)
    torch = sys.modules['torch']
    runtime = command[command.index('--runtime') + 1]
    is_shim = hasattr(torch, '_torch_compat_install_context')
    if is_shim != (runtime == 'jittor'):
        raise ValueError('Actual interpreter is not the requested runtime')
    binary = getattr(getattr(torch, '_C', None), '__file__', None) if not is_shim else None
    if not is_shim and (not binary or not Path(binary).is_file()):
        raise ValueError('Oracle must load an actual binary PyTorch extension')
    binding = {'jittor_commit': baseline, 'upstream': source['upstream'],
               'hf_hub_offline': os.environ.get('HF_HUB_OFFLINE') == '1',
               'downstream_distribution': distribution,
               'measurement_source_sha256': {str(path.relative_to(root)): digest(path)
                   for path in (script, root / 'compat/tests/torch/_ecosystem_cases.py', Path(__file__).resolve())},
               'interpreter': {'executable': sys.executable, 'pid': os.getpid(), 'is_shim': is_shim,
                               'torch_version': torch.__version__, 'binary_extension': binary,
                               'binary_sha256': digest(binary) if binary else None},
               'measurement_command': [sys.executable, *command]}
    if is_shim:
        jt = sys.modules['jittor']
        if Path(jt.__file__).resolve() != root / 'python/jittor/__init__.py':
            raise ValueError('Measured Jittor does not belong to the requested worktree')
        binding['native_source'] = str(Path(jt.__file__).resolve())
        binding['native_runtime_flags'] = {name: int(getattr(jt.flags, name)) for name in
                                          ('use_cuda', 'auto_graph_replay', 'auto_graph_replay_retain_bytes')}
    output = Path(command[command.index('--output') + 1]) if '--output' in command else Path(command[2])
    lines = captured.getvalue().splitlines()
    for index, line in enumerate(lines):
        for marker in ('ECOSYSTEM_RESULT ', 'TRAINING_RESULT '):
            if line.startswith(marker):
                value = json.loads(line[len(marker):])
                value.update(binding)
                value['seed'] = int(command[command.index('--seed') + 1]) if '--seed' in command else 0
                weight = (Path(command[command.index('--weights') + 1]) if '--weights' in command
                          else output.with_name(output.stem + '.weights.npz'))
                value.update(weights=str(weight), weights_sha256=digest(weight))
                output.with_suffix('.json').write_text(json.dumps(value, indent=2) + '\n')
                lines[index] = marker + json.dumps(value)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()

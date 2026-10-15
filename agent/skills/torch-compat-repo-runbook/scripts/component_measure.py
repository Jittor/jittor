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
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def package_files(package):
    return {path.relative_to(package).as_posix(): digest(path)
            for path in sorted(package.rglob('*'))
            if path.is_file() and '__pycache__' not in path.parts
            and path.suffix not in ('.pyc', '.pyo')}


class SnapshotObserver:
    """Inspect actual tensors at the maintained probe's serialization boundary."""

    def __init__(self, runtime, device, script, *, allow_scalar_counter=False, verify_cuda_pointer=False):
        self.runtime, self.device, self.script = runtime, device, str(script)
        self.allow_scalar_counter = allow_scalar_counter
        self.verify_cuda_pointer = verify_cuda_pointer
        self.records = []

    def __call__(self, frame, event, _argument):
        if event != 'call' or frame.f_code.co_filename != self.script or frame.f_code.co_name not in ('snapshot', '_numpy_snapshot'):
            return
        value = frame.f_locals['value']
        if value.device.type != self.device:
            raise AssertionError('Actual VERL tensor device mismatch before serialization')
        dtype = str(value.dtype)
        if dtype not in ('float32', 'torch.float32'):
            if not (self.allow_scalar_counter and dtype in ('int64', 'torch.int64') and tuple(value.shape) == ()):
                raise AssertionError('Actual VERL tensor is not FP32 before serialization')
        row = {'device': value.device.type, 'dtype': dtype, 'shape': list(value.shape)}
        if self.runtime == 'jittor':
            # sync evaluates pending work; it does not request a device transfer.
            value.sync()
            row.update(location=value.location(), dispatch=sys.modules['jittor'].core.dispatch_context([value])[0])
            if row['location'] != ('device' if self.device == 'cuda' else 'cpu') or row['dispatch'] != self.device:
                raise AssertionError('Physical VERL tensor residency/dispatch mismatch before serialization')
        if self.verify_cuda_pointer and self.device == 'cuda':
            import ctypes

            ordinal = ctypes.c_int(-1)
            status = ctypes.CDLL('libcuda.so.1').cuPointerGetAttribute(
                ctypes.byref(ordinal), 9, ctypes.c_void_p(value.data_ptr()))
            if status != 0 or ordinal.value != value.device.index:
                raise AssertionError('Actual CUDA pointer belongs to another device')
            row['pointer_device'] = ordinal.value
        self.records.append(row)


def installed_distribution(source, module):
    package = Path(module.__file__).resolve().parent
    import_name = {'ms-swift': 'swift'}.get(source['repo'], source['repo'])
    expected = {row['path']: row['sha256'] for row in source['package_sources']}
    actual = {}
    for name, wanted in expected.items():
        relative = Path(name)
        if relative.parts[0] == 'src':
            relative = Path(*relative.parts[1:])
        if relative.parts[0] != import_name:
            raise ValueError('Unexpected upstream package path: ' + name)
        path = package.joinpath(*relative.parts[1:])
        if not path.is_file() or digest(path) != wanted:
            raise ValueError('Installed source differs from clean upstream: ' + name)
        actual[name] = digest(path)
    version = str(module.__version__)
    if version != source['upstream']['version']:
        raise ValueError('Installed release version differs from upstream')
    files = package_files(package)
    result = {'repo': source['repo'], 'version': version,
              'source_sha256': actual, 'installed_files_sha256': files,
              'package': str(package)}
    if source['repo'] in ('torchmetrics', 'tensordict'):
        from _installed_source_proof import bind_official_wheel
        from _metric_td_installation import wheel_pin

        if 'official_wheel' not in source:
            raise ValueError('Metrics/TensorDict require a fixed official-wheel source descriptor')
        wheel = Path(source['official_wheel']['path'])
        result = bind_official_wheel(source, module, wheel, result,
                                    expected_wheel_sha256=wheel_pin(source, wheel))
    if source['repo'] == 'verl':
        from importlib.metadata import distribution

        metadata = distribution('verl')
        if Path(metadata.locate_file('verl/__init__.py')).resolve() != Path(module.__file__).resolve():
            raise ValueError('VERL distribution metadata belongs to another installation')
        result['distribution_version'] = metadata.version
        wheel = Path(os.environ['JITTOR_VERL_WHEEL'])
        result['wheel_sha256'] = digest(wheel)
    if source['repo'] in ('mmengine', 'mmcv'):
        from importlib.metadata import PackageNotFoundError, distribution
        from _distribution_names import validate_name_declaration

        names = validate_name_declaration(source['distribution_names'], source['sources'])
        matches = []
        for name in names:
            try:
                metadata = distribution(name)
            except PackageNotFoundError:
                continue
            if Path(metadata.locate_file(source['repo'] + '/__init__.py')).resolve() == Path(module.__file__).resolve():
                matches.append(metadata)
        if len(matches) != 1:
            raise ValueError('Missing or ambiguous actual OpenMMLab distribution metadata')
        metadata = matches[0]
        result.update(distribution_name=metadata.metadata['Name'], distribution_version=metadata.version,
                      wheel_sha256=digest(Path(os.environ['JITTOR_' + source['repo'].upper() + '_WHEEL'])))
    if source['repo'] == 'vllm':
        from _vllm_installation import bind_installation

        result = bind_installation(source, module, Path(os.environ['JITTOR_VLLM_WHEEL']), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repo', choices=('torchmetrics', 'tensordict', 'diffusers', 'verl', 'mmengine', 'mmcv', 'vllm'), required=True)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--wheel', type=Path,
                        help='Required fixed official wheel for Metrics/TensorDict source proof')
    parser.add_argument('--mmengine-checkout', type=Path)
    parser.add_argument('probe', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.probe[1:] if args.probe[:1] == ['--'] else args.probe
    root = args.root.resolve()
    script = Path(command[0]).resolve() if command else None
    allowed = {root / 'compat/tests/torch/_ecosystem_runner.py',
               root / 'agent/skills/torch-compat-repo-runbook/scripts/training_probe.py'}
    if args.repo in ('mmengine', 'mmcv'):
        allowed.add(root / 'agent/skills/torch-compat-repo-runbook/scripts/openmmlab_training_probe.py')
    if args.mmengine_checkout is not None and args.repo != 'mmcv':
        parser.error('--mmengine-checkout is only valid for the real MMCV training-owner dependency')
    if args.repo == 'mmcv' and args.mmengine_checkout is None:
        parser.error('MMCV measurements also require --mmengine-checkout')
    if args.wheel is not None and args.repo not in ('torchmetrics', 'tensordict'):
        parser.error('--wheel is only valid for Metrics/TensorDict official-wheel proof')
    if script not in allowed:
        parser.error('Only the maintained ecosystem runner or training probe is allowed')
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from registry_inventory import inventory
    source = inventory(args.repo, args.checkout)
    if args.repo in ('torchmetrics', 'tensordict'):
        from _metric_td_installation import source_descriptor

        if args.wheel is None:
            parser.error('Metrics/TensorDict measurements require --wheel')
        source = source_descriptor(args.repo, args.checkout, args.wheel)
    sources = {args.repo: source}
    if args.repo == 'mmcv':
        sources['mmengine'] = inventory('mmengine', args.mmengine_checkout)
    baseline = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Measurements require a clean source checkout')
    frozen_sources = None
    if args.repo == 'vllm':
        from _vllm_installation import framework_snapshot

        frozen_sources = framework_snapshot(root)
    elif args.repo in ('torchmetrics', 'tensordict'):
        from _metric_td_installation import framework_snapshot

        frozen_sources = framework_snapshot(root)
    sys.path.insert(0, str(script.parent))
    runtime = command[command.index('--runtime') + 1]
    before_files, before_package = {}, {}
    if args.repo in ('verl', 'mmengine', 'mmcv', 'vllm', 'torchmetrics', 'tensordict'):
        from importlib.machinery import PathFinder

        # Hash the real import target without importing VERL or Torch ahead of
        # the probe's original device/runtime initialization order.
        for repo in sources:
            spec = PathFinder.find_spec(repo, sys.path)
            if spec is None or not spec.origin:
                raise ValueError('Missing actual component installation: ' + repo)
            before_package[repo] = Path(spec.origin).resolve().parent
            before_files[repo] = package_files(before_package[repo])
    sys.argv = command
    captured = io.StringIO()
    observer = (SnapshotObserver(runtime, command[command.index('--device') + 1], script,
                                 allow_scalar_counter=args.repo == 'mmcv',
                                 verify_cuda_pointer=args.repo in ('vllm', 'torchmetrics', 'tensordict'))
                if args.repo in ('verl', 'mmengine', 'mmcv', 'vllm', 'torchmetrics', 'tensordict') else None)
    previous_profile = sys.getprofile()
    try:
        if observer is not None:
            if previous_profile is not None:
                raise ValueError('VERL snapshot observations need an unprofiled worker')
            sys.setprofile(observer)
        with redirect_stdout(captured):
            runpy.run_path(str(script), run_name='__main__')
    finally:
        if observer is not None:
            sys.setprofile(previous_profile)
    module = importlib.import_module(args.repo)
    distribution = installed_distribution(source, module)
    distributions = {args.repo: distribution}
    for repo, entry in sources.items():
        if repo != args.repo:
            distributions[repo] = installed_distribution(entry, importlib.import_module(repo))
        if repo in before_files and (distributions[repo]['installed_files_sha256'] != before_files[repo]
                                     or Path(distributions[repo]['package']) != before_package[repo]):
            raise ValueError('Actual component installation changed during measurement: ' + repo)
    torch = sys.modules['torch']
    is_shim = hasattr(torch, '_torch_compat_install_context')
    if is_shim != (runtime == 'jittor'):
        raise ValueError('Actual interpreter is not the requested runtime')
    binary = getattr(getattr(torch, '_C', None), '__file__', None) if not is_shim else None
    if not is_shim and (not binary or not Path(binary).is_file()):
        raise ValueError('Oracle must load an actual binary PyTorch extension')
    source_paths = [script, root / 'compat/tests/torch/_ecosystem_cases.py', Path(__file__).resolve()]
    if args.repo == 'verl':
        source_paths.append(root / 'compat/tests/torch/_ecosystem_verl_cases.py')
        if root / 'compat/tests/torch/_ecosystem_runner.py' not in source_paths:
            source_paths.append(root / 'compat/tests/torch/_ecosystem_runner.py')
    if args.repo in ('mmengine', 'mmcv'):
        source_paths.append(root / 'compat/tests/torch/_ecosystem_openmmlab_cases.py')
        if root / 'compat/tests/torch/_ecosystem_runner.py' not in source_paths:
            source_paths.append(root / 'compat/tests/torch/_ecosystem_runner.py')
    if args.repo == 'vllm':
        from _vllm_installation import framework_sources

        source_paths = [root / path for path in framework_sources(root)]
        if (framework_snapshot(root) != frozen_sources
                or subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() != baseline
                or subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip()):
            raise ValueError('vLLM framework source changed during measurement')
    elif args.repo in ('torchmetrics', 'tensordict'):
        from _metric_td_installation import framework_snapshot

        source_paths = [root / path for path in frozen_sources]
        if (framework_snapshot(root) != frozen_sources
                or subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() != baseline
                or subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip()):
            raise ValueError('Metrics/TensorDict framework source changed during measurement')
    binding = {'jittor_commit': baseline, 'upstream': source['upstream'],
               'hf_hub_offline': os.environ.get('HF_HUB_OFFLINE') == '1',
               'downstream_distribution': distribution,
               'measurement_source_sha256': {
                   str(path.relative_to(root)): (hashlib.sha256(os.fsencode(os.readlink(path))).hexdigest()
                                                 if path.is_symlink() else digest(path))
                   for path in source_paths},
               'interpreter': {'executable': sys.executable, 'pid': os.getpid(), 'is_shim': is_shim,
                               'torch_version': torch.__version__, 'binary_extension': binary,
                               'binary_sha256': digest(binary) if binary else None},
               'measurement_command': [sys.executable, *command]}
    if args.repo == 'vllm':
        from _vllm_installation import adapter_installation, device_controls

        binding['device_controls'] = device_controls(torch, command[command.index('--device') + 1])
        if is_shim:
            binding['adapter_distribution'] = adapter_installation(root)
    elif args.repo in ('torchmetrics', 'tensordict'):
        from _vllm_installation import device_controls

        binding['runtime'] = runtime
        binding['device_controls'] = device_controls(torch, command[command.index('--device') + 1])
        binding['runtime_environment'] = {name: os.environ.get(name) for name in
                                          ('HOME', 'JITTOR_HOME', 'TMPDIR', 'XDG_CACHE_HOME')}
    if args.repo in ('mmengine', 'mmcv'):
        binding['upstreams'] = {repo: entry['upstream'] for repo, entry in sources.items()}
        binding['downstream_distributions'] = distributions
    if observer is not None:
        binding['tensor_observations'] = observer.records
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
                if args.repo in ('torchmetrics', 'tensordict'):
                    import shutil
                    from _runtime_platform import capture_platform

                    device = value['device']
                    allocation = os.environ['CUDA_VISIBLE_DEVICES'].split(',') if device == 'cuda' else []
                    rows = ([line.split(', ') for line in subprocess.check_output(
                        ['nvidia-smi', '--query-gpu=index,uuid,name,driver_version', '--format=csv,noheader'],
                        text=True).splitlines()] if device == 'cuda' else [])
                    selected = [row for row in rows if row[0] in allocation or row[1] in allocation]
                    flags = binding.get('native_runtime_flags', {})
                    native = sys.modules.get('jittor') if is_shim else None
                    value['platform'] = capture_platform(
                        torch, device, physical_devices=selected, card_indices=allocation,
                        precision=value['tf32'],
                        jittor_flags={'use_cuda': flags.get('use_cuda', int(device == 'cuda')), 'use_acl': 0,
                            'graph_replay': 'default' if flags.get('auto_graph_replay') else 'eager',
                            'tf32': value['tf32']['matmul'], 'hf32': False},
                        nvcc=(native.compiler.nvcc_path if native else os.environ.get('JT_BUILD_NVCC_PATH')),
                        cxx=native.compiler.cc_path if native else shutil.which('g++') or 'g++')
                value['seed'] = int(command[command.index('--seed') + 1]) if '--seed' in command else 0
                weight = (Path(command[command.index('--weights') + 1]) if '--weights' in command
                          else output.with_name(output.stem + '.weights.npz'))
                value.update(weights=str(weight), weights_sha256=digest(weight))
                output.with_suffix('.json').write_text(json.dumps(value, indent=2) + '\n')
                lines[index] = marker + json.dumps(value)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()

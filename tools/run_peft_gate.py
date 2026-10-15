#!/usr/bin/env python3
"""Re-run the complete existing PEFT LoRA case with real per-process provenance."""

import argparse
from contextlib import redirect_stdout
import ctypes
import importlib
import io
import json
import os
from pathlib import Path
import platform
import runpy
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'agent/skills/torch-compat-repo-runbook/scripts'
sys.path.insert(0, str(SCRIPTS))


def external_path(path):
    resolved = Path(path).resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        raise ValueError('Measurement/cache paths must be outside the source checkout')
    return resolved


def environment(cache, runtime, device):
    env = dict(os.environ)
    folder = external_path(cache) / ('candidate' if runtime == 'jittor' else 'oracle') / device
    for key, name in {'HOME': 'home', 'REAL_HOME': 'home', 'JITTOR_HOME': 'jittor-home',
                      'TMPDIR': 'tmp', 'XDG_CACHE_HOME': 'xdg-cache', 'TORCH_HOME': 'torch-home',
                      'JITTOR_TORCH_CACHE_ROOT': 'shim-cache', 'JITTOR_TORCH_RUNTIME_ROOT': 'shim-runtime',
                      'JITTOR_TORCH_HOME': 'home', 'TORCH_EXTENSIONS_DIR': 'torch-extensions'}.items():
        path = folder / name
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env.update(PYTHONPATH=str(ROOT / 'python') if runtime == 'jittor' else '',
               JITTOR_SOURCE_ROOT=str(ROOT) if runtime == 'jittor' else '',
               JITTOR_TORCH_SHIM='1' if runtime == 'jittor' else '0',
               JITTOR_TORCH_KEEP_HOME='1', JITTOR_TORCH_KEEP_TMPDIR='1',
               PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               JT_BACKEND=device, JT_USE_CUDA=str(int(device == 'cuda')), use_cuda=str(int(device == 'cuda')),
               JT_BUILD_USE_MKL='0', JT_BUILD_USE_NCCL='0', JT_BUILD_USE_MPI='0', JT_BUILD_USE_CUTT='0',
               JT_USE_PARALLEL_OP_COMPILER='0', DISABLE_MULTIPROCESSING='1', JT_CUDA_KERNEL_MATH='strict',
               JITTOR_ECOSYSTEM_TF32='1', JITTOR_ECOSYSTEM_CUDNN_BENCHMARK='0',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               cache_name='peft-source-bound-' + device)
    if device == 'cpu':
        env.update(CUDA_VISIBLE_DEVICES='', nvcc_path='', JT_BUILD_NVCC_PATH='')
    else:
        if not env.get('CUDA_VISIBLE_DEVICES', '').strip() or ',' in env['CUDA_VISIBLE_DEVICES']:
            raise ValueError('Allocate exactly one explicit CUDA device')
        nvcc = env.get('JT_BUILD_NVCC_PATH') or env.get('nvcc_path') or shutil.which('nvcc')
        if not nvcc:
            raise ValueError('CUDA requires a real NVCC toolchain')
        env.update(nvcc_path=nvcc, JT_BUILD_NVCC_PATH=nvcc)
    return env


def worker(args):
    from _peft_scorecard import (CASE, digest, git, installed_payload, require,
                                source_descriptor, source_files)
    require(not git(ROOT, 'status', '--porcelain'), 'Measurements require a clean source checkout')
    baseline = git(ROOT, 'rev-parse', 'HEAD')
    sources = source_descriptor({'peft': args.peft_checkout, 'transformers': args.transformers_checkout},
                                {'peft': args.peft_wheel, 'transformers': args.transformers_wheel})
    source_hashes = source_files(ROOT)
    started = time.time()
    sys.path.insert(0, str(ROOT / 'compat/tests/torch'))
    import _ecosystem_runner as runner
    import _ecosystem_cases as cases
    torch = runner._import_torch(args.runtime)
    shim = hasattr(torch, '_torch_compat_install_context')
    require(shim == (args.runtime == 'jittor'), 'Requested interpreter is not actual runtime')
    binary = getattr(getattr(torch, '_C', None), '__file__', None) if not shim else None
    require(shim or (binary and Path(binary).is_file()), 'Oracle has no real Torch binary extension')
    if shim:
        require(Path(sys.modules['jittor'].__file__).resolve() == ROOT / 'python/jittor/__init__.py',
                'Candidate imported another native source checkout')
    modules = {repo: importlib.import_module(repo) for repo in sources}
    distributions = {repo: installed_payload(source, modules[repo]) for repo, source in sources.items()}
    original, requirements = cases.CASES[CASE]
    observations, handles, scope = [], [], {}
    hooked = set()

    def observe(label, value):
        require(value.device.type == args.device, 'Actual tensor is on the wrong device: ' + label)
        row = {'label': label, 'shape': list(value.shape), 'dtype': str(value.dtype), 'device': value.device.type}
        if args.device == 'cuda':
            ordinal = ctypes.c_int(-1)
            rc = ctypes.CDLL('libcuda.so.1').cuPointerGetAttribute(
                ctypes.byref(ordinal), 9, ctypes.c_void_p(value.data_ptr()))
            require(rc == 0 and ordinal.value == 0, 'Tensor is not a real allocation on logical CUDA zero')
            row['pointer_device'] = ordinal.value
        observations.append(row)

    def build(runtime_torch):
        model, inputs = original(runtime_torch)
        scope.update(parameters=sorted(n for n, _ in model.named_parameters()),
                     buffers=sorted(n for n, _ in model.named_buffers()),
                     trainable_parameters=sorted(n for n, p in model.named_parameters() if p.requires_grad),
                     frozen_parameters=sorted(n for n, p in model.named_parameters() if not p.requires_grad),
                     input_spec=inputs,
                     config={n: getattr(model.config, n) for n in ('hidden_size', 'intermediate_size',
                             'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads', 'vocab_size')})
        require(len(scope['parameters']) == 29 and len(scope['trainable_parameters']) == 8
                and len(scope['frozen_parameters']) == 21, 'Original PEFT parameter scope changed')

        def before(module, positional, keyword):
            for name, value in module.named_parameters():
                observe('parameter/' + name, value)
                if value.requires_grad and name not in hooked:
                    def hook(gradient, name=name):
                        observe('gradient/' + name, gradient)
                    handles.append(value.register_hook(hook))
                    hooked.add(name)
            for name, value in keyword.items():
                if isinstance(value, torch.Tensor):
                    observe('input/' + name, value)

        def after(module, positional, output):
            observe('output', runner._primary_output(output))
        handles.extend((model.register_forward_pre_hook(before, with_kwargs=True),
                        model.register_forward_hook(after)))
        return model, inputs

    cases.CASES[CASE] = (build, requirements)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.phase == 'train':
        script = SCRIPTS / 'training_probe.py'
        command = [str(script), '--root', str(ROOT), '--case', CASE, '--runtime', args.runtime,
                   '--device', args.device, '--output', str(output), '--steps', '3',
                   '--learning-rate', '0.001', '--seed', '0', '--weights', str(args.weights)]
        marker = 'TRAINING_RESULT '
    else:
        script = ROOT / 'compat/tests/torch/_ecosystem_runner.py'
        command = [str(script), CASE, str(output), '--runtime', args.runtime, '--device', args.device,
                   '--seed', str(args.seed), '--repeats', '1']
        if args.weights:
            command.extend(('--weights', str(args.weights)))
        marker = 'ECOSYSTEM_RESULT '
    captured = io.StringIO()
    try:
        sys.argv = command
        with redirect_stdout(captured):
            runpy.run_path(str(script), run_name='__main__')
    finally:
        output.with_suffix('.probe-stdout.log').write_text(captured.getvalue())
        cases.CASES[CASE] = (original, requirements)
        for handle in handles:
            handle.remove()
    result_lines = [line[len(marker):] for line in captured.getvalue().splitlines() if line.startswith(marker)]
    require(len(result_lines) == 1, 'Probe did not produce one actual result')
    result = json.loads(result_lines[0])
    require(source_files(ROOT) == source_hashes and git(ROOT, 'rev-parse', 'HEAD') == baseline
            and not git(ROOT, 'status', '--porcelain'), 'Source changed during measurement')
    require(distributions == {repo: installed_payload(source, modules[repo]) for repo, source in sources.items()},
            'Installed package changed during measurement')
    weights = Path(args.weights) if args.weights else output.with_name(output.stem + '.weights.npz')
    result.update(case=CASE, runtime=args.runtime, jittor_commit=baseline, seed=args.seed,
                  hf_hub_offline=os.environ.get('HF_HUB_OFFLINE') == '1',
                  downstream_distributions=distributions, measurement_source_sha256=source_hashes,
                  model_scope=scope, device_observations=observations,
                  weights=str(weights), weights_sha256=digest(weights),
                  measurement_command=[sys.executable, *sys.orig_argv[1:]],
                  interpreter={'executable': sys.executable, 'pid': os.getpid(), 'is_shim': shim,
                               'torch_version': str(torch.__version__), 'binary_extension': binary,
                               'binary_sha256': digest(binary) if binary else None},
                  started_at=started, finished_at=time.time())
    import _accelerate_protocol as platform_helpers
    flags = {n: int(getattr(sys.modules['jittor'].flags, n)) for n in
             ('use_cuda', 'auto_graph_replay')} if shim else {}
    result['platform'] = {'os': platform.platform(), 'cpu': platform_helpers.cpu_description(),
        'accelerator_model': torch.cuda.get_device_name(0) if args.device == 'cuda' else 'not-applicable',
        'card_count': int(args.device == 'cuda'),
        'card_indices': os.environ['CUDA_VISIBLE_DEVICES'].split(',') if args.device == 'cuda' else [],
        'driver': platform_helpers.command('nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader').splitlines()[0]
                  if args.device == 'cuda' else 'not-applicable',
        'cuda': str(torch.version.cuda) if args.device == 'cuda' else 'not-applicable',
        'cann': 'not-applicable',
        'nvcc': platform_helpers.command(os.environ['JT_BUILD_NVCC_PATH'], '--version')
                if args.device == 'cuda' else 'not-applicable',
        'gcc': platform_helpers.command(shutil.which('g++') or 'g++', '--version'),
        'nvidia_wheels': platform_helpers.selected_nvidia_wheels(),
        'jittor_flags': {'use_cuda': flags.get('use_cuda', int(args.device == 'cuda')), 'use_acl': 0,
                         'graph_replay': 'default' if flags.get('auto_graph_replay') else 'eager',
                         'tf32': result['tf32']['matmul'], 'hf32': False}, 'precision': result['tf32']}
    maps = Path('/proc/self/maps')
    paths = {line.split()[-1] for line in maps.read_text().splitlines()
             if '/' in line and any(s in line.split()[-1] for s in ('libcu', 'libnv', 'libnccl'))} if maps.is_file() else set()
    result['mapped_libraries'] = [{'path': str(Path(p).resolve()), 'sha256': digest(p)}
                                  for p in sorted(paths) if Path(p).is_file()]
    if args.device == 'cuda':
        driver = ctypes.CDLL('libcuda.so.1')
        device, identifier = ctypes.c_int(), ctypes.create_string_buffer(16)
        require(driver.cuDeviceGet(ctypes.byref(device), 0) == 0
                and driver.cuDeviceGetUuid(identifier, device) == 0, 'Cannot query actual CUDA driver UUID')
        observed_uuid = 'GPU-' + str(uuid.UUID(bytes=identifier.raw))
        rows = [line.split(', ') for line in platform_helpers.command('nvidia-smi',
                '--query-gpu=index,uuid,name,driver_version', '--format=csv,noheader').splitlines()]
        selected = [row for row in rows if os.environ['CUDA_VISIBLE_DEVICES'] in row[:2]]
        require(len(selected) == 1 and selected[0][1] == observed_uuid,
                'Actual CUDA driver allocation differs from requested physical device')
        result['cuda_device'] = {'logical_index': 0, 'physical_index': int(selected[0][0]), 'uuid': observed_uuid}
        runtime_versions = {}
        for item in result['mapped_libraries']:
            if Path(item['path']).name.startswith('libcudart.so'):
                version = ctypes.c_int()
                require(ctypes.CDLL(item['path']).cudaRuntimeGetVersion(ctypes.byref(version)) == 0,
                        'Cannot query actually mapped CUDA runtime version')
                runtime_versions[item['path']] = version.value
        require(runtime_versions, 'CUDA computation has no actually mapped CUDA runtime')
        result['platform']['cuda'] = {'frontend': str(torch.version.cuda), 'mapped_runtime_api': runtime_versions}

    output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PEFT_RESULT ' + json.dumps({'case': CASE, 'runtime': args.runtime, 'phase': args.phase,
                                      'device': args.device, 'output': str(output), 'pid': os.getpid()}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-python', type=Path)
    parser.add_argument('--oracle-python', type=Path)
    for repo in ('peft', 'transformers'):
        parser.add_argument('--' + repo + '-checkout', type=Path, required=True)
        parser.add_argument('--' + repo + '-wheel', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--preflight', action='store_true', help='Run only the two actual base workers; never an L3 gate')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--runtime', choices=('torch', 'jittor'))
    parser.add_argument('--phase', choices=('base', 'perturb', 'train'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--weights', type=Path)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    from _peft_scorecard import git, require, source_descriptor
    require(not git(ROOT, 'status', '--porcelain'), 'Gate requires clean measured source')
    require(args.candidate_python and args.candidate_python.is_file()
            and args.oracle_python and args.oracle_python.is_file(), 'Two explicit interpreters are required')
    require(not args.output.exists() or not any(args.output.iterdir()), 'Refusing a nonempty measurement directory')
    require(ROOT not in args.output.resolve().parents, 'Measurements belong outside checkout')
    external_path(args.output)
    if args.cache_root:
        external_path(args.cache_root)
    args.output.mkdir(parents=True, exist_ok=True)
    sources = source_descriptor({'peft': args.peft_checkout, 'transformers': args.transformers_checkout},
                                {'peft': args.peft_wheel, 'transformers': args.transformers_wheel})
    baseline = git(ROOT, 'rev-parse', 'HEAD')
    execution = {'baseline': baseline, 'device': args.device, 'sources': sources, 'results': [],
                 'scope': 'preflight-base-only' if args.preflight else 'complete-original-L3'}
    common = []
    for repo in ('peft', 'transformers'):
        common.extend(('--' + repo + '-checkout', str(getattr(args, repo + '_checkout')),
                       '--' + repo + '-wheel', str(getattr(args, repo + '_wheel'))))
    phases = (('base', 0),) if args.preflight else (('base', 0), ('perturb', 7), ('train', 0))
    for phase, seed in phases:
        for runtime, python in (('torch', args.oracle_python), ('jittor', args.candidate_python)):
            output = args.output / (phase + '.' + runtime + '.npz')
            command = [str(python), str(Path(__file__).resolve()), '--worker', '--runtime', runtime,
                       '--phase', phase, '--seed', str(seed), '--device', args.device, '--output', str(output), *common]
            if phase != 'base' or runtime == 'jittor':
                command.extend(('--weights', str(args.output / 'base.torch.weights.npz')))
            env = environment(args.cache_root or args.output / 'benchmark-cache', runtime, args.device)
            with output.with_suffix('.log').open('w') as stream:
                process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=3600)
            execution['results'].append({'runtime': runtime, 'phase': phase, 'seed': seed,
                                         'returncode': process.returncode, 'command': command})
            (args.output / 'execution.json').write_text(json.dumps(execution, indent=2) + '\n')
            if process.returncode:
                raise RuntimeError('Real PEFT worker failed; see ' + str(output.with_suffix('.log')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

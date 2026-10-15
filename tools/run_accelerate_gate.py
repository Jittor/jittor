#!/usr/bin/env python3
"""Run source-bound Accelerate checkpoint protocols in independent interpreters.

CPU and CUDA world-one checkpoint rows are distinct from world-two FSDP2 FULL
rows. CUDA requires an explicit two-card allocation. No environment is installed
or changed by this runner; use prepared private interpreters with matching
Accelerate versions. Raw observations and independent rank caches stay external.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / 'compat/tests/torch'
SINGLE = 'test_optimizer_checkpoint_state.py::TestOptimizerCheckpoint::test_accelerate_save_load_continues_model_optimizer_scheduler'
FSDP = 'test_accelerate_distributed_integration.py::test_fsdp2_full_model_optimizer_checkpoint_resume'


def environment(runtime, device, state, rank, devices, upstream):
    """Set canonical and legacy flags before importing either tensor runtime."""
    env = dict(os.environ)
    for name in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'JT_NCCL_RANK', 'JT_NCCL_WORLD_SIZE',
                 'JT_NCCL_LOCAL_RANK', 'JT_NCCL_LOCAL_WORLD_SIZE', 'JITTOR_DIST_RENDEZVOUS_DIR',
                 'REAL_TORCH_SITE', 'JITTOR_TORCH_PROJECT_ROOT', 'JITTOR_TORCH_RUNTIME_ROOT',
                 'MASTER_ADDR', 'MASTER_PORT', 'JITTOR_ACCELERATE_INTEGRATION_OUTPUT', 'CHECKPOINT_REPORT',
                 'JITTOR_ACCELERATE_CHECKPOINT_PHASE', 'JITTOR_ACCELERATE_CHECKPOINT_DIRECTORY'):
        env.pop(name, None)
    env.update({'JITTOR_TORCH_SHIM': '1' if runtime == 'jittor' else '0',
                'JITTOR_SOURCE_ROOT': str(ROOT) if runtime == 'jittor' else '',
                'PYTHONPATH': str(ROOT / 'python') if runtime == 'jittor' else '',
                'PYTHONDONTWRITEBYTECODE': '1', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
                'PYTEST_PLUGINS': 'pytest_timeout,xdist.plugin',
                'JITTOR_REQUIRE_REAL_TORCH': '1', 'JITTOR_REQUIRE_ACCELERATE': '1',
                'JITTOR_TEST_REQUIRE_EXECUTION': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'JT_BACKEND': device, 'JT_USE_CUDA': str(int(device == 'cuda')), 'use_cuda': str(int(device == 'cuda')),
                'JT_BUILD_USE_MPI': '0', 'JT_BUILD_USE_NCCL': str(int(device == 'cuda' and runtime == 'jittor')), 'JT_BUILD_USE_MKL': '0', 'JT_BUILD_USE_CUTT': '0',
                'JT_USE_PARALLEL_OP_COMPILER': '0', 'DISABLE_MULTIPROCESSING': '1', 'JT_CUDA_KERNEL_MATH': 'strict',
                'JITTOR_ACCELERATE_CHECKOUT': str(upstream), 'JITTOR_ACCELERATE_DEVICES': ','.join(devices),
                'CUDA_VISIBLE_DEVICES': ','.join(devices) if device == 'cuda' else ''})
    env.update({name: '2' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
    if device == 'cpu':
        env.update({'JT_BUILD_NVCC_PATH': '', 'nvcc_path': ''})
    else:
        nvcc = env.get('JT_BUILD_NVCC_PATH') or env.get('nvcc_path') or shutil.which('nvcc')
        if not nvcc:
            raise ValueError('CUDA gate requires an explicit or discoverable nvcc')
        env.update({'JT_BUILD_NVCC_PATH': nvcc, 'nvcc_path': nvcc})
    rank_environment(env, state, runtime, device, rank)
    return env


def rank_environment(env, state, runtime, device, rank):
    cache = Path(state) / 'benchmark-cache' / runtime / device / ('rank%d' % rank)
    for name in ('home', 'jittor-home', 'tmp'):
        (cache / name).mkdir(parents=True, exist_ok=True)
    env.update({'HOME': str(cache / 'home'),
                'JITTOR_HOME': str(cache / 'jittor-home'), 'TMPDIR': str(cache / 'tmp'),
                'JT_BUILD_CACHE_NAME': 'accelerate-protocol-' + device + '-rank%d' % rank,
                'cache_name': 'accelerate-protocol-' + device + '-rank%d' % rank})


def worker_rank(args, env):
    # Warm workers are single-process, with one visible card. Their cache rank
    # must survive without pretending that a communicator has been launched.
    if args.warm_only:
        return args.warm_rank
    return int(env.get('JT_NCCL_RANK', env.get('RANK', '0')))


def worker(args):
    rank = worker_rank(args, os.environ)
    rank_environment(os.environ, args.cache_root or args.output, args.runtime, args.device, rank)
    if args.runtime == 'jittor':
        import jittor  # noqa: F401
    import torch
    assert hasattr(torch, '_torch_compat_install_context') == (args.runtime == 'jittor')
    if args.runtime == 'torch':
        assert torch._C.__file__.endswith(('.so', '.pyd', '.dylib'))
    if args.device == 'cuda':
        assert torch.cuda.is_available()
        torch.cuda.set_device(0 if args.runtime == 'jittor' else int(os.environ.get('LOCAL_RANK', '0')))
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    import numpy as np
    import random
    random.seed(0)
    np.random.seed(0)
    if args.warm_only:
        if args.runtime == 'jittor' and args.device == 'cuda':
            # The one-rank public launcher supplies a real singleton bootstrap.
            # Compile the NCCL extension serially here, before either two-rank
            # process can reach its first collective compilation.
            from jittor.build import compile_extern
            compile_extern.setup_nccl()
            if compile_extern.get_library_ops('nccl') is None:
                raise RuntimeError('Serial CUDA warm did not build the real NCCL provider')
        layer = torch.nn.Linear(4, 3).to(args.device)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=.01)
        for dtype in ([torch.float32, torch.bfloat16] if args.device == 'cuda' else [torch.float32]):
            compute = layer.to(dtype=dtype)
            # Match the maintained checkpoint node's explicit tensor factory.
            # Size-based ones has a separately recorded leaf-gradient limitation.
            x = torch.tensor(np.ones((4, 4), dtype=np.float32), device=args.device, requires_grad=True)
            assert x.is_leaf and x.requires_grad
            compute(x.to(dtype=dtype)).float().square().mean().backward()
            assert x.grad is not None and x.grad.device.type == args.device
            if dtype == torch.float32:
                optimizer.step()
            optimizer.zero_grad()
        if args.device == 'cuda':
            torch.cuda.synchronize()
        return 0
    if args.cpu_provider_probe:
        # Probe actual provider initialization, then the actual FSDP2 CPU path.
        import torch.distributed as dist
        record = {'runtime': args.runtime, 'device': 'cpu', 'world_size': 2, 'rank': rank}
        try:
            dist.init_process_group('gloo')
            from accelerate import Accelerator, FullyShardedDataParallelPlugin
            accelerator = Accelerator(cpu=True, fsdp_plugin=FullyShardedDataParallelPlugin(
                fsdp_version=2, state_dict_type='FULL_STATE_DICT', auto_wrap_policy='NO_WRAP'))
            model = accelerator.prepare(torch.nn.Linear(4, 3))
            assert accelerator.is_fsdp2, 'CPU world-two Accelerator did not use FSDP2'
            model(torch.ones((2, 4))).sum().backward()
            record['status'] = 'executed'
        except (RuntimeError, ValueError, NotImplementedError, AssertionError) as error:
            record.update(status='blocked', error_type=type(error).__name__, error=str(error))
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
        Path(args.measurement).mkdir(parents=True, exist_ok=True)
        (Path(args.measurement) / ('cpu_provider_rank%d.json' % rank)).write_text(json.dumps(record, indent=2))
        return 0
    sys.path.insert(0, str(TESTS))
    os.environ['CHECKPOINT_DEVICE'] = args.device
    os.environ['JITTOR_ACCELERATE_PROTOCOL_OUTPUT'] = str(Path(args.measurement).resolve())
    os.environ['JITTOR_ACCELERATE_INPUT_SEED'] = str(args.seed)
    if args.checkpoint_phase:
        if not args.checkpoint_directory:
            raise ValueError('Fresh process requires a persisted checkpoint directory')
        os.environ['JITTOR_ACCELERATE_CHECKPOINT_PHASE'] = args.checkpoint_phase
        os.environ['JITTOR_ACCELERATE_CHECKPOINT_DIRECTORY'] = str(Path(args.checkpoint_directory).resolve())
    os.environ['JITTOR_ACCELERATE_DISTRIBUTED_REQUIRED'] = '1' if args.world_size == 2 else '0'
    test = SINGLE if args.world_size == 1 else FSDP + '[' + args.precision + ']'
    filename, selector = test.split('::', 1)
    suite = Path(args.output) / 'source-suite'
    suite.mkdir(parents=True, exist_ok=True)
    link = suite / filename
    if not link.exists():
        try:
            link.symlink_to(TESTS / filename)
        except FileExistsError:
            pass  # Another rank created this exact shared source link.
    if link.resolve() != TESTS / filename:
        raise ValueError('External test source link points to the wrong checkout')
    import pytest
    junit = Path(args.measurement) / ('junit-rank%d.xml' % rank)
    junit.parent.mkdir(parents=True, exist_ok=True)
    rc = pytest.main([str(link) + '::' + selector, '-q', '-s', '--confcutdir=' + str(suite),
                      '--rootdir=' + str(suite), '--timeout=1800', '--junitxml=' + str(junit)])
    return rc


def run(command, env, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as stream:
        result = subprocess.run(command, env=env, cwd=str(log.parent), stdout=stream,
                                stderr=subprocess.STDOUT, timeout=7200, check=False)
    return result.returncode


def gate(args):
    state = args.output.resolve()
    if ROOT == state or ROOT in state.parents:
        raise ValueError('Measurements and caches must be outside the source checkout')
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Freeze and commit gate sources before actual measurements')
    if (state / 'execution.json').exists():
        raise ValueError('Choose a fresh output directory; historical measurements are immutable')
    if getattr(args, 'fresh_process', False) and state.exists() and any(state.iterdir()):
        raise ValueError('Choose an empty output directory; failed phase artifacts are immutable too')
    state.mkdir(parents=True, exist_ok=True)
    cache_state = (args.cache_root or state).resolve()
    if ROOT == cache_state or ROOT in cache_state.parents:
        raise ValueError('Benchmark caches must be outside the source checkout')
    devices = args.devices.split(',') if args.device == 'cuda' else []
    if args.device == 'cuda' and (len(devices) != 2 or len(set(devices)) != 2 or not all(devices)):
        raise ValueError('CUDA gate requires exactly two explicit distinct devices')
    commands, results = [], []
    for runtime, python in (('torch', args.oracle_python), ('jittor', args.jittor_python)):
        env = environment(runtime, args.device, cache_state, 0, devices, args.upstream.resolve())
        # First core/extension compilation is deliberately serial, including rank1.
        for rank in range(2 if args.device == 'cuda' else 1):
            warm_env = dict(env)
            rank_environment(warm_env, cache_state, runtime, args.device, rank)
            if devices:
                warm_env['CUDA_VISIBLE_DEVICES'] = devices[rank]
            warm = [python, str(Path(__file__).resolve()), '--worker', '--runtime', runtime,
                    '--device', args.device, '--output', str(state), '--cache-root', str(cache_state),
                    '--warm-only', '--warm-rank', str(rank)]
            if runtime == 'jittor' and args.device == 'cuda':
                warm = [python, '-m', 'jittor.distributed.launch', '--standalone', '--backend=nccl',
                        '--nproc-per-node=1', '--device-ids=' + devices[rank],
                        '--log-dir=' + str(state / 'warm-launcher-logs' / ('rank%d' % rank)), *warm[1:]]
            for attempt in range(3):
                rc = run(warm, warm_env, state / 'logs' / ('warm-%s-rank%d-%d.log' % (runtime, rank, attempt)))
                if rc != 3:  # Explicit jit_utils rebuild/restart protocol.
                    break
            if rc:
                raise RuntimeError('Warmup failed: ' + ' '.join(warm))
        scenarios = [(1, 'no')] + ([(2, 'no'), (2, 'bf16')] if args.device == 'cuda' else [])
        for world, precision in scenarios:
            for seed in (0, 7):
                measurement = state / 'measurements' / runtime / ('world%d-%s-seed%d' % (world, precision, seed))
                phases = ('produce', 'consume') if getattr(args, 'fresh_process', False) else (None,)
                checkpoint = state / 'checkpoints' / runtime / ('world%d-%s-seed%d' % (world, precision, seed))
                phase_results = []
                for phase in phases:
                    folder = measurement / phase if phase else measurement
                    base = [str(Path(__file__).resolve()), '--worker', '--runtime', runtime, '--device', args.device,
                            '--output', str(state), '--cache-root', str(cache_state), '--world-size', str(world), '--precision', precision,
                            '--seed', str(seed), '--measurement', str(folder)]
                    if phase:
                        base.extend(['--checkpoint-phase', phase, '--checkpoint-directory', str(checkpoint)])
                    if world == 1:
                        command = [python, *base]
                    elif runtime == 'torch':
                        command = [python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2', *base]
                    else:
                        command = [python, '-m', 'jittor.distributed.launch', '--standalone', '--backend=nccl',
                                   '--nproc-per-node=2', '--device-ids=' + ','.join(devices),
                                   '--log-dir=' + str(folder / 'launcher-logs'), *base]
                    scenario_env = environment(runtime, args.device, cache_state, 0,
                                               devices[:1] if world == 1 else devices, args.upstream.resolve())
                    log = '%s-world%d-%s-seed%d%s.log' % (runtime, world, precision, seed, '-' + phase if phase else '')
                    # run() waits for the complete launcher/rank process tree.
                    # A consumer is never started while its producer is alive.
                    started = time.monotonic_ns()
                    rc = run(command, scenario_env, state / 'logs' / log)
                    finished = time.monotonic_ns()
                    launch = {'command': command, 'runtime': runtime, 'returncode': rc}
                    if phase:
                        launch.update(phase=phase, world_size=world, precision=precision, seed=seed,
                                      started_monotonic_ns=started, finished_monotonic_ns=finished,
                                      producer_exited=(phase == 'consume' and phase_results[0]['returncode'] == 0))
                    commands.append(launch)
                    phase_results.append(launch)
                    if rc:
                        break
                    for rank in range(world):
                        tree = ET.parse(folder / ('junit-rank%d.xml' % rank))
                        cases = list(tree.iter('testcase'))
                        if len(cases) != 1 or any(list(case) and any(node.tag in ('skipped', 'failure', 'error') for node in case) for case in cases):
                            raise ValueError('Required gate did not execute exactly one passed test per rank')
                result = {'runtime': runtime, 'world_size': world, 'precision': precision, 'seed': seed,
                          'returncode': rc, 'measurement': str(measurement), 'status': 'failed' if rc else 'passed'}
                if getattr(args, 'fresh_process', False):
                    result.update(checkpoint_mode='fresh-process', phase_launches=phase_results)
                results.append(result)
                (state / 'execution.json').write_text(json.dumps({'commands': commands, 'results': results}, indent=2))
    return int(any(row['status'] != 'passed' for row in results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, help='External benchmark-only cache root; independent of immutable output directories')
    parser.add_argument('--jittor-python')
    parser.add_argument('--oracle-python')
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--devices', default='')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--runtime', choices=('torch', 'jittor'))
    parser.add_argument('--world-size', type=int, choices=(1, 2), default=1)
    parser.add_argument('--precision', choices=('no', 'bf16'), default='no')
    parser.add_argument('--seed', type=int, choices=(0, 7), default=0)
    parser.add_argument('--measurement')
    parser.add_argument('--warm-only', action='store_true')
    parser.add_argument('--fresh-process', action='store_true', help='Save, terminate producers, then launch new checkpoint consumers')
    parser.add_argument('--checkpoint-phase', choices=('produce', 'consume'))
    parser.add_argument('--checkpoint-directory', type=Path)
    parser.add_argument('--warm-rank', type=int, choices=(0, 1), default=0)
    parser.add_argument('--cpu-provider-probe', action='store_true')
    args = parser.parse_args()
    if args.worker:
        if not args.runtime:
            parser.error('--worker requires --runtime')
        return worker(args)
    if not all((args.jittor_python, args.oracle_python, args.upstream)):
        parser.error('gate requires --jittor-python, --oracle-python and --upstream')
    return gate(args)


if __name__ == '__main__':
    raise SystemExit(main())

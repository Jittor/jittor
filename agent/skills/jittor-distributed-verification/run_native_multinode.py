#!/usr/bin/env python3
"""Run the maintained world-two native FSDP protocol, one command per host.

This tool does not contact hosts or provision interpreters. Both nodes need the
same clean source commit and a genuinely shared output directory. Local state
directories contain private caches and must be outside that shared directory.
Loopback execution is useful evidence of the runner, not multi-host acceptance.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import time


STAGES = ('plan', 'oracle', 'warm', 'collectives', 'parity', 'baseline', 'resume', 'seal')
ENV_KEYS = (
    'HOME', 'JITTOR_HOME', 'TMPDIR', 'PYTHONPATH', 'JITTOR_TORCH_SHIM',
    'JT_BACKEND', 'JT_USE_CUDA', 'use_cuda', 'CUDA_VISIBLE_DEVICES',
    'JT_BUILD_USE_MPI', 'JT_BUILD_USE_NCCL', 'JT_BUILD_USE_MKL', 'JT_BUILD_USE_CUTT',
    'JT_BUILD_NVCC_PATH', 'nvcc_path', 'JITTOR_CUDA_WHEEL_STRICT',
    'JT_BUILD_CACHE_NAME', 'cache_name', 'JT_CUDA_KERNEL_MATH',
    'JT_USE_PARALLEL_OP_COMPILER', 'DISABLE_MULTIPROCESSING',
    'NCCL_SOCKET_IFNAME', 'NCCL_P2P_DISABLE', 'LD_PRELOAD', 'LD_LIBRARY_PATH',
    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'PYTEST_DISABLE_PLUGIN_AUTOLOAD', 'PYTEST_PLUGINS', 'PYTHONDONTWRITEBYTECODE',
    'JITTOR_REQUIRE_NATIVE_FSDP', 'JITTOR_REQUIRE_MULTINODE_CONTROL',
    'JITTOR_NATIVE_FSDP_TEST_DIR', 'JITTOR_NATIVE_FSDP_ORACLE_DIR',
    'JITTOR_NATIVE_FSDP_PHASE', 'JITTOR_TEST_REQUIRE_EXECUTION',
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def git(root, *arguments):
    return subprocess.check_output(['git', '-C', str(root), *arguments], text=True).strip()


def source_snapshot(root, expected):
    if git(root, 'rev-parse', 'HEAD') != expected or git(root, 'status', '--porcelain'):
        raise ValueError('the explicit source SHA must identify this clean checkout')
    files, modes = {}, {}
    entries = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-s', '-z'])
    for entry in entries.split(b'\0'):
        if not entry:
            continue
        descriptor, name = entry.decode().split('\t', 1)
        mode, _blob, stage = descriptor.split()
        if stage != '0' or mode not in ('100644', '100755', '120000'):
            raise ValueError('unsupported/unmerged Git entry: ' + name)
        path = root / name
        content = os.readlink(path).encode() if mode == '120000' else path.read_bytes()
        files[name] = hashlib.sha256(content).hexdigest()
        modes[name] = mode
    return {'source_sha': expected, 'status': '', 'files': files, 'git_modes': modes,
            'host_id': socket.gethostname(), 'source_root': str(root)}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--stage', choices=STAGES, required=True)
    result.add_argument('--source-root', type=Path, required=True)
    result.add_argument('--source-sha', required=True)
    result.add_argument('--python', type=Path, required=True)
    result.add_argument('--oracle-python', type=Path, required=True)
    result.add_argument('--node-rank', type=int, choices=(0, 1), required=True)
    result.add_argument('--master-addr', required=True)
    result.add_argument('--master-port', type=int, required=True)
    result.add_argument('--job-id', required=True)
    result.add_argument('--device-id', required=True)
    result.add_argument('--state-root', type=Path, required=True)
    result.add_argument('--shared-output', type=Path, required=True)
    result.add_argument('--nvcc', type=Path, required=True)
    result.add_argument('--timeout', type=float, default=2400)
    result.add_argument('--rendezvous-timeout', type=float, default=120)
    return result


def validate(args):
    for name in ('source_root', 'state_root', 'shared_output'):
        setattr(args, name, getattr(args, name).resolve())
    # Resolving a venv's bin/python symlink silently selects the base environment.
    # Preserve executable spellings while making relative CLI paths absolute.
    for name in ('python', 'oracle_python', 'nvcc'):
        setattr(args, name, getattr(args, name).absolute())
    if (not re.fullmatch(r'[0-9a-f]{40}', args.source_sha)
            or not re.fullmatch(r'[A-Za-z0-9_.-]{1,96}', args.job_id)):
        raise ValueError('source SHA/job id has an invalid form')
    if not 0 < args.master_port < 65536 or not args.master_addr.strip():
        raise ValueError('an explicit reachable master and nonzero port are required')
    if not args.device_id.strip() or ',' in args.device_id or any(c.isspace() for c in args.device_id):
        raise ValueError('this protocol requires exactly one explicitly allocated device per node')
    if any(not math.isfinite(value) or value <= 0 for value in (args.timeout, args.rendezvous_timeout)):
        raise ValueError('timeouts must be finite and positive')
    for name in ('python', 'oracle_python', 'nvcc'):
        if not getattr(args, name).is_file():
            raise ValueError(name + ' must name an existing executable')
    for path in (args.state_root, args.shared_output):
        if path == args.source_root or args.source_root in path.parents:
            raise ValueError('state and output must be outside the source checkout')
    if (args.state_root == args.shared_output or args.state_root in args.shared_output.parents
            or args.shared_output in args.state_root.parents):
        raise ValueError('private cache state and shared evidence must be disjoint')
    if args.stage in ('oracle', 'seal') and args.node_rank != 0:
        raise ValueError('oracle and seal run once, on node zero')
    return source_snapshot(args.source_root, args.source_sha)


def environment(args, *, oracle=False):
    env = dict(os.environ)
    for name in tuple(env):
        if name.startswith(('JT_NCCL_', 'JT_HCCL_', 'JITTOR_LAUNCHER_', 'OMPI_', 'PMI_')) or name in (
                'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
            env.pop(name, None)
    base = args.state_root / ('oracle' if oracle else 'worker')
    env.update({name: str(base / leaf) for name, leaf in (
        ('HOME', 'home'), ('JITTOR_HOME', 'jittor'), ('TMPDIR', 'tmp'))})
    env.update({
        'PYTHONPATH': '' if oracle else str(args.source_root / 'python'),
        'JITTOR_TORCH_SHIM': '0', 'JT_BACKEND': 'cuda', 'JT_USE_CUDA': '1', 'use_cuda': '1',
        'CUDA_VISIBLE_DEVICES': args.device_id,
        'JT_BUILD_USE_MPI': '0', 'JT_BUILD_USE_NCCL': '0' if oracle else '1',
        'JT_BUILD_USE_MKL': '0', 'JT_BUILD_USE_CUTT': '0',
        'JT_BUILD_NVCC_PATH': str(args.nvcc), 'nvcc_path': str(args.nvcc),
        'JITTOR_CUDA_WHEEL_STRICT': '1', 'JT_CUDA_KERNEL_MATH': 'strict',
        'JT_BUILD_CACHE_NAME': 'native-multinode-unit', 'cache_name': 'native-multinode-unit',
        'JT_USE_PARALLEL_OP_COMPILER': '0', 'DISABLE_MULTIPROCESSING': '1',
        'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1', 'PYTEST_PLUGINS': 'pytest_timeout,xdist.plugin',
        'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2',
        'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
        'JITTOR_TEST_REQUIRE_EXECUTION': '1',
    })
    if args.stage in ('parity', 'baseline', 'resume'):
        env.update({
            'JITTOR_REQUIRE_NATIVE_FSDP': '1', 'JITTOR_NATIVE_FSDP_PHASE': args.stage,
            'JITTOR_NATIVE_FSDP_TEST_DIR': str(args.shared_output / 'cuda-unit/fsdp-results'),
            'JITTOR_NATIVE_FSDP_ORACLE_DIR': str(args.shared_output / 'cuda-unit/oracle-results'),
        })
    if args.stage == 'collectives':
        env['JITTOR_REQUIRE_MULTINODE_CONTROL'] = '1'
    return env


def launcher_command(args):
    tool = args.source_root / 'agent/skills/jittor-distributed-verification/native_multinode_worker.py'
    address = '127.0.0.1' if args.stage == 'warm' else args.master_addr
    port = 0 if args.stage == 'warm' else args.master_port
    command = [str(args.python), '-m', 'jittor.distributed.launch', '--backend=nccl',
               '--nproc-per-node=1', '--device-ids=' + args.device_id,
               '--master-addr=' + address, '--master-port=' + str(port),
               '--timeout=' + str(args.rendezvous_timeout),
               '--log-dir=' + str(args.shared_output / 'logs' / args.stage / ('node' + str(args.node_rank)))]
    if args.stage == 'warm':
        command += ['--standalone']
    else:
        command += ['--nnodes=2', '--node-rank=' + str(args.node_rank),
                    '--rdzv-id=' + args.job_id + '-' + args.stage]
    return command + ['--', str(args.python), str(tool), '--stage', args.stage,
                      '--source-root', str(args.source_root), '--source-sha', args.source_sha,
                      '--node-rank', str(args.node_rank), '--output', str(args.shared_output)]


def run_command(args, command, env, label):
    directory = args.shared_output / 'commands'
    record = directory / (label + '.json')
    claim = directory / (label + '.claim')
    claim.parent.mkdir(parents=True, exist_ok=True)
    claim.open('x').close()
    for name in ('HOME', 'JITTOR_HOME', 'TMPDIR'):
        Path(env[name]).mkdir(parents=True, exist_ok=True)
    log = directory / (label + '.log')
    value = {'argv': command, 'cwd': str(args.source_root), 'source_sha': args.source_sha,
             'host_id': socket.gethostname(), 'node_rank': args.node_rank,
             'env': {name: env[name] for name in ENV_KEYS if name in env},
             'start_ns': time.time_ns()}
    write_json(directory / (label + '-started.json'), value)
    with log.open('x') as stream:
        result = subprocess.run(command, cwd=str(args.source_root), env=env, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=args.timeout)
    value.update(returncode=result.returncode, end_ns=time.time_ns(), log_sha256=sha256(log))
    write_json(record, value)
    if result.returncode:
        raise RuntimeError('command failed; partial evidence preserved: ' + str(record))


def seal(args):
    output = args.shared_output
    nodes = [json.loads((output / ('source-node%d.json' % node)).read_text()) for node in (0, 1)]
    for source in nodes:
        if (source['source_sha'] != args.source_sha or source['status'] != ''
                or source['files'] != nodes[0]['files'] or source['git_modes'] != nodes[0]['git_modes']):
            raise ValueError('nodes do not have identical clean source trees')
    for stage in ('warm', 'collectives', 'parity', 'baseline', 'resume'):
        for node in (0, 1):
            record = json.loads((output / 'commands' / ('%s-node%d.json' % (stage, node))).read_text())
            if record['returncode'] or record['source_sha'] != args.source_sha:
                raise ValueError('missing successful actual node execution')
    for kind in ('sgd', 'adamw'):
        for dtype in ('float32', 'bfloat16'):
            record = json.loads((output / 'commands' / ('oracle-%s-%s.json' % (kind, dtype))).read_text())
            if record['returncode'] or record['source_sha'] != args.source_sha:
                raise ValueError('missing independent oracle execution')
    files = {}
    for path in sorted(output.rglob('*')):
        if path.is_symlink():
            raise ValueError('evidence directory may not contain symlinks')
        if path.is_file():
            files[path.relative_to(output).as_posix()] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    write_json(output / 'raw-bindings.json', {'source_sha': args.source_sha, 'source_status': '',
                                            'files': files})
    script = args.source_root / 'agent/skills/jittor-distributed-verification/recheck_native_fsdp.py'
    command = [str(args.oracle_python), str(script), '--bindings', str(output / 'raw-bindings.json'),
               '--source-manifest', str(output / 'source-node0.json'),
               '--expected-source', args.source_sha, '--source-root', str(args.source_root),
               '--results', str(output / 'cuda-unit/fsdp-results'),
               '--oracle', str(output / 'cuda-unit/oracle-results'),
               '--identities', str(output / 'cuda-unit/worker-identities')]
    run_command(args, command, environment(args, oracle=True), 'seal-recheck')
    result = {'bound_files': len(files), 'hosts': [n['host_id'] for n in nodes],
              'raw_bindings_sha256': sha256(output / 'raw-bindings.json'),
              'recheck_log_sha256': sha256(output / 'commands/seal-recheck.log'),
              'source_sha': args.source_sha,
              'multi_host_acceptance': 'requires external topology verification'}
    write_json(output / 'verification.json', result)
    print(json.dumps(result, sort_keys=True))


def main():
    args = parser().parse_args()
    snapshot = validate(args)
    if args.stage == 'plan':
        print(json.dumps({'world_size': 2, 'local_world_size': 1, 'node_rank': args.node_rank,
                          'source_sha': args.source_sha, 'host_id': socket.gethostname(),
                          'native_python': str(args.python), 'oracle_python': str(args.oracle_python),
                          'scope': 'plan only; no device or connectivity claim'}, sort_keys=True))
        return
    args.shared_output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'seal':
        seal(args)
        return
    if args.stage == 'warm':
        write_json(args.shared_output / ('source-node%d.json' % args.node_rank), snapshot)
    else:
        prior = json.loads((args.shared_output / ('source-node%d.json' % args.node_rank)).read_text())
        warm = json.loads((args.shared_output / 'commands' / ('warm-node%d.json' % args.node_rank)).read_text())
        if (prior['source_sha'] != args.source_sha or prior['files'] != snapshot['files']
                or prior['git_modes'] != snapshot['git_modes'] or warm['returncode'] != 0):
            raise ValueError('this node needs a successful warm with the same clean source tree')
    if args.stage == 'oracle':
        for kind in ('sgd', 'adamw'):
            for dtype in ('float32', 'bfloat16'):
                command = [str(args.oracle_python), str(args.source_root / 'agent/skills/jittor-distributed-verification/native_fsdp_oracle.py'),
                           '--source-root', str(args.source_root), '--device', 'cuda', '--kind', kind,
                           '--dtype', dtype, '--output-dir', str(args.shared_output / 'cuda-unit/oracle-results')]
                run_command(args, command, environment(args, oracle=True), 'oracle-' + kind + '-' + dtype)
    else:
        run_command(args, launcher_command(args), environment(args), args.stage + '-node' + str(args.node_rank))


if __name__ == '__main__':
    main()

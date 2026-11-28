#!/usr/bin/env python3
"""Run one or two example benchmark ranks, with independent JIT caches.

The subprocess watchdog also works when backend code holds Python's GIL.
Pass all benchmark-specific arguments after --. Every rank sees the same
ordered GPU UUIDs; local rank selects its physical device.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import time

import numpy as np


def aggregate(ranks, warmup):
    lengths = {len(rank['records']) for rank in ranks}
    if len(lengths) != 1 or not all(rank['completed'] for rank in ranks):
        raise ValueError('All ranks must finish the same number of steps')
    profiled = {rank.get('profile_step', -1) for rank in ranks}
    indices = [i for i in range(warmup, next(iter(lengths))) if i not in profiled]
    times = [max(rank['records'][i]['total'] for rank in ranks) * 1000 for i in indices]
    if not times:
        raise ValueError('No measured steps after warmup')
    batch = ranks[0].get('local_batch_size', 16) * len(ranks)
    samples = sum(rank['records'][i].get('batch_size', rank.get('local_batch_size', 16))
                  for rank in ranks for i in indices)
    tail = times[-max(1, len(times)//2):]
    return dict(world_size=len(ranks), global_batch_size=batch, measured_steps=len(times),
                excluded_profile_steps=sorted(i for i in profiled if i >= warmup),
                steady_mean_ms=statistics.mean(times), steady_median_ms=statistics.median(times),
                tail_mean_ms=statistics.mean(tail), tail_median_ms=statistics.median(tail),
                p95_ms=float(np.percentile(times, 95)), global_samples_per_second=samples*1000/sum(times),
                timing='per-step maximum across ranks, including fetch/transfer/update/synchronization',
                max_rank_mean_ms=max(statistics.mean(rank['records'][i]['total']*1000 for i in indices) for rank in ranks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True)
    parser.add_argument('--backend', required=True, choices=('jittor', 'torch'))
    parser.add_argument('--example', required=True, choices=('nlp', 'cv', 'cv-parity'))
    parser.add_argument('--gpus', required=True, help='One UUID, or two comma separated UUIDs')
    parser.add_argument('--cpus', required=True, help='One CPU affinity per rank, separated by /')
    parser.add_argument('--cache-roots', required=True, help='One cache per rank, separated by comma')
    parser.add_argument('--output', required=True)
    parser.add_argument('--timeout', type=float, default=7200)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    gpus, cpus, caches = args.gpus.split(','), args.cpus.split('/'), args.cache_roots.split(',')
    if len(gpus) not in (1, 2) or len(cpus) != len(gpus) or len(caches) != len(gpus) or len(set(caches)) != len(caches):
        parser.error('Use one or two GPUs, with distinct caches and a CPU affinity for each rank')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    with socket.socket() as channel:
        channel.bind(('127.0.0.1', 0))
        port = channel.getsockname()[1]
    processes = []
    started = time.perf_counter()
    try:
        for rank in range(len(gpus)):
            env = dict(os.environ)
            env.update(CUDA_VISIBLE_DEVICES=args.gpus, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       JITTOR_HOME=caches[rank], RANK=str(rank), WORLD_SIZE=str(len(gpus)),
                       LOCAL_WORLD_SIZE=str(len(gpus)), MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
            if len(gpus) > 1:
                env['LOCAL_RANK'] = str(rank)
            else:
                env.pop('LOCAL_RANK', None)
            if args.backend == 'jittor':
                env.update(JITTOR_TORCH_SHIM='1', JITTOR_TORCH_SKIP_EXT_BUILD='1', JITTOR_TORCH_KEEP_HOME='1',
                           JT_BUILD_USE_MPI='0', JT_BUILD_USE_NCCL=str(int(len(gpus)>1)),
                           JITTOR_TORCH_DISTRIBUTED_AUTO_INIT=str(int(len(gpus)>1)),
                           JT_NCCL_WORLD_SIZE=str(len(gpus)), JT_NCCL_RANK=str(rank), JT_NCCL_LOCAL_RANK=str(rank),
                           JT_NCCL_ROOTINFO_FILE=str(out/'nccl.bin'), JITTOR_DIST_RENDEZVOUS_DIR=str(out))
            else:
                env.pop('JITTOR_TORCH_SHIM', None)
            script_name = 'cv_parity.py' if args.example == 'cv-parity' else 'bench_' + args.example + '.py'
            script = Path(__file__).with_name(script_name)
            command = ['uclampset', '-m', '1024', '-M', '1024', 'taskset', '-c', cpus[rank], args.python,
                       str(script), '--backend', args.backend, '--output', str(out / ('rank%d' % rank)), *arguments]
            log = (out / ('rank%d.log' % rank)).open('w')
            processes.append((subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True), log))
        deadline = time.monotonic() + args.timeout
        while any(process.poll() is None for process, _ in processes):
            if time.monotonic() > deadline:
                raise TimeoutError('Example benchmark exceeded %s seconds' % args.timeout)
            if any(process.poll() not in (None, 0) for process, _ in processes):
                raise RuntimeError('A benchmark rank failed; see rank logs in ' + str(out))
            time.sleep(.25)
        if any(process.returncode for process, _ in processes):
            raise RuntimeError('A benchmark rank failed; see ' + str(out))
    finally:
        for process, log in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            log.close()
        (out/'processes.json').write_text(json.dumps(
            [dict(pid=process.pid, exit_code=process.returncode) for process, _ in processes], indent=2)+'\n')
    ranks = [json.loads((out / ('rank%d/result.json' % rank)).read_text()) for rank in range(len(gpus))]
    if args.example == 'cv-parity':
        assert all(rank['completed'] and rank['frozen_unchanged'] for rank in ranks)
        result = dict(world_size=len(ranks), completed=True, numerical_snapshots=True)
    else:
        result = aggregate(ranks, ranks[0]['warmup_steps'])
    result.update(example=args.example, backend=args.backend, gpus=gpus, cpus=cpus,
                  rank_results=[str(out / ('rank%d/result.json' % rank)) for rank in range(len(gpus))],
                  wall_seconds=time.perf_counter()-started)
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

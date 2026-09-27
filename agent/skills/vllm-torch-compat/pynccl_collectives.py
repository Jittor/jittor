"""Direct vLLM PyNccl foreign-write ordering and buffer reuse diagnostic.

actual: pinned library behavior; ptds: pass handle 2 only to NCCL (Jittor only);
synchronize: preserve the stream but finish NCCL before creating a consumer.
No production modules are patched. Compare actual vs interventions separately.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace


def worker(args):
    rank = int(os.environ['RANK'])
    local = int(os.environ['LOCAL_RANK'])
    report = dict(backend=args.backend, rank=rank, local_rank=local, mode=args.mode,
                  status='running', checks=[], mismatches=[],
                  sizes=args.sizes, steps=args.steps, rank_skew_ms=args.rank_skew_ms)
    try:
        if args.backend == 'jittor':
            import jittor as jt
            jt.flags.use_parallel_op_compiler = 0
            jt.flags.use_cuda = 1
        import numpy as np
        import torch
        import torch.distributed as dist
        assert hasattr(torch, '_torch_compat_install_context') == (args.backend == 'jittor')
        torch.cuda.set_device(local)
        if not dist.is_initialized():
            from datetime import timedelta
            dist.init_process_group('nccl', timeout=timedelta(seconds=120))
        cpu = dist.new_group([0, 1], backend='gloo')
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.utils.torch_utils import current_stream
        comm = PyNcclCommunicator(cpu, device=torch.device('cuda:%d' % local))
        assert comm.available and not comm.disabled, 'real PyNccl is required'
        report.update(nccl_version=comm.nccl_version,
                      advertised_cuda_stream=int(current_stream().cuda_stream),
                      default_device=str(torch.get_default_device()))
        if args.mode == 'ptds':
            assert args.backend == 'jittor', 'PTDS intervention is only meaningful for Jittor producers'
            stream = SimpleNamespace(cuda_stream=2)
        else:
            stream = None
        report['passed_stream_override'] = None if stream is None else stream.cuda_stream

        def save():
            Path(args.output).write_text(json.dumps(report, indent=2))

        def check_tensor(label, value, reference, dtype_name, count, step):
            # Consumer executes on the backend's normal compute stream.
            # No sync/pointer query is inserted between NCCL and this compute.
            consumed = value * 2 + 1
            actual = consumed.cpu().numpy()
            expected = (reference * 2 + 1).astype(actual.dtype)
            error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
            row = dict(operation=label, dtype=dtype_name, count=count, step=step,
                       max_abs_error=error, matches=bool(np.array_equal(actual, expected)))
            report['checks'].append(row)
            if not row['matches'] and len(report['mismatches']) < 12:
                bad = np.flatnonzero(actual.reshape(-1) != expected.reshape(-1))
                row = dict(row, indices=bad[:8].tolist(),
                           actual=actual.reshape(-1)[bad[:8]].astype(float).tolist(),
                           expected=expected.reshape(-1)[bad[:8]].astype(float).tolist())
                report['mismatches'].append(row)
            assert value.device.type == 'cuda'
            driver = ctypes.CDLL('libcuda.so.1')
            memory, ordinal = ctypes.c_uint(), ctypes.c_int()
            ptr = ctypes.c_uint64(value.data_ptr())
            assert driver.cuPointerGetAttribute(ctypes.byref(memory), 2, ptr) == 0
            assert driver.cuPointerGetAttribute(ctypes.byref(ordinal), 9, ptr) == 0
            assert memory.value == 2 and ordinal.value == local
            report['driver_device'] = ordinal.value
            save()

        def after_collective():
            if args.mode == 'synchronize':
                torch.cuda.synchronize()

        for dtype_name in ('float16', 'float32'):
            dtype = getattr(torch, dtype_name)
            for count in args.sizes:
                reusable = torch.empty((count,), dtype=dtype, device='cuda:%d' % local)
                for step in range(args.steps):
                    phase = step % 16
                    reference = np.arange(count, dtype=np.float32) % 17
                    # Values remain exactly representable in both dtypes.
                    source = (torch.arange(count, device='cuda:%d' % local, dtype=torch.float32) % 17
                              + rank * 32 + phase * 3).to(dtype=dtype)
                    if rank == 1:
                        time.sleep(args.rank_skew_ms / 1000)  # inter-rank producer skew
                    reduced = comm.all_reduce(source, stream=stream)
                    after_collective()
                    check_tensor('all_reduce_new', reduced, reference * 2 + 32 + phase * 6,
                                 dtype_name, count, step)
                    reused = comm.all_reduce(source, out_tensor=reusable, stream=stream)
                    after_collective()
                    check_tensor('all_reduce_reused', reused, reference * 2 + 32 + phase * 6,
                                 dtype_name, count, step)
                    gathered = torch.empty((2 * count,), dtype=dtype, device='cuda:%d' % local)
                    comm.all_gather(gathered, source, stream=stream)
                    after_collective()
                    expected = np.concatenate([reference + r * 32 + phase * 3 for r in range(2)])
                    check_tensor('all_gather', gathered, expected, dtype_name, count, step)
                    broadcast = source.clone()
                    root_rank = step % 2
                    comm.broadcast(broadcast, src=root_rank, stream=stream)
                    after_collective()
                    check_tensor('broadcast', broadcast, reference + root_rank * 32 + phase * 3,
                                 dtype_name, count, step)
        report['status'] = 'completed' if not report['mismatches'] else 'failed'
        report['checks_count'] = len(report['checks'])
        save()
        comm.destroy()
        dist.destroy_process_group(cpu)
        dist.destroy_process_group()
        assert not report['mismatches'], 'PyNccl GPU consumer read incorrect collective output'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2))


def parent(args):
    root = Path(args.output).resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    prefix = args.backend + '-' + args.mode
    paths = [root / ('%s-rank%d.%s' % (prefix, rank, ext)) for rank in range(2) for ext in ('log', 'json')]
    if Path(args.output).exists() or any(path.exists() for path in paths):
        raise FileExistsError('Use a new output directory to retain earlier evidence')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    processes, logs = [], []
    report = dict(backend=args.backend, mode=args.mode, status='running',
                  cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
    try:
        for rank in range(2):
            env = dict(os.environ, MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port),
                       RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='2')
            if args.backend == 'jittor':
                env['JITTOR_HOME'] = os.environ['TP2_RANK%d_CACHE' % rank]
            logfile = open(root / ('%s-rank%d.log' % (prefix, rank)), 'x'); logs.append(logfile)
            command = [sys.executable, __file__, '--backend', args.backend, '--mode', args.mode,
                       '--worker', '--steps', str(args.steps), '--rank-skew-ms', str(args.rank_skew_ms),
                       '--sizes', *[str(size) for size in args.sizes],
                       '--output', str(root / ('%s-rank%d.json' % (prefix, rank)))]
            processes.append(subprocess.Popen(command, env=env, stdout=logfile,
                                              stderr=subprocess.STDOUT, start_new_session=True))
        deadline = time.monotonic() + args.timeout
        while any(proc.poll() is None for proc in processes):
            if any(proc.poll() not in (None, 0) for proc in processes):
                raise RuntimeError('A PyNccl rank failed; inspect per-rank JSON/logs')
            if time.monotonic() > deadline:
                raise TimeoutError('PyNccl smoke exceeded %s seconds' % args.timeout)
            time.sleep(.5)
        assert all(proc.returncode == 0 for proc in processes)
        rows = [json.loads((root / ('%s-rank%d.json' % (prefix, rank))).read_text()) for rank in range(2)]
        assert all(row['status'] == 'completed' for row in rows)
        report.update(status='completed', ranks=rows)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        for proc in processes:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait()
        for logfile in logs: logfile.close()
        Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['jittor', 'oracle'], required=True)
    parser.add_argument('--mode', choices=['actual', 'ptds', 'synchronize'], default='actual')
    parser.add_argument('--output', required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[16, 1024])
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--rank-skew-ms', type=float, default=10)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--worker', action='store_true')
    arguments = parser.parse_args()
    if min(arguments.sizes) < 1 or arguments.steps < 1 or arguments.rank_skew_ms < 0 or arguments.timeout <= 0:
        parser.error('sizes/steps/timeout must be positive; rank skew must be nonnegative')
    if arguments.mode == 'ptds' and arguments.backend != 'jittor':
        parser.error('PTDS override is a Jittor-only diagnostic')
    if not arguments.worker and arguments.backend == 'jittor':
        caches = [os.environ.get('TP2_RANK%d_CACHE' % rank) for rank in range(2)]
        if not all(caches) or Path(caches[0]).resolve() == Path(caches[1]).resolve():
            parser.error('set distinct prewarmed TP2_RANK0_CACHE and TP2_RANK1_CACHE')
    worker(arguments) if arguments.worker else parent(arguments)

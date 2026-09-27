"""External two-rank CPU-control + CUDA-data collective placement regression."""
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


def pointer_placement(tensor, expected, local_rank):
    driver = ctypes.CDLL('libcuda.so.1')
    pointer = ctypes.c_uint64(tensor.data_ptr())
    memory = ctypes.c_uint()
    code = driver.cuPointerGetAttribute(ctypes.byref(memory), 2, pointer)
    assert tensor.device.type == expected, (str(tensor.device), expected)
    if expected == 'cuda':
        assert code == 0 and memory.value == 2, (code, memory.value)
        ordinal = ctypes.c_int()
        assert driver.cuPointerGetAttribute(ctypes.byref(ordinal), 9, pointer) == 0
        assert ordinal.value == local_rank, ordinal.value
        return dict(device=str(tensor.device), memory_type=memory.value, driver_device=ordinal.value)
    # Pageable host allocations need not be registered with the CUDA driver.
    assert code == 1 or (code == 0 and memory.value == 1), (code, memory.value)
    return dict(device=str(tensor.device), driver_status=code,
                memory_type=memory.value if code == 0 else 'unregistered_host')


def worker(args):
    import faulthandler
    faulthandler.dump_traceback_later(25, repeat=True)
    rank = int(os.environ['RANK'])
    local = int(os.environ['LOCAL_RANK'])
    report = dict(backend=args.backend, rank=rank, local_rank=local, status='running', cpu_checks=[], gpu_checks=[])
    initialized = False
    try:
        if args.backend == 'jittor':
            import jittor as jt
            jt.flags.use_parallel_op_compiler = 0
            jt.flags.use_cuda = 1
        import torch
        import torch.distributed as dist
        assert hasattr(torch, '_torch_compat_install_context') == (args.backend == 'jittor')
        torch.cuda.set_device(local)
        report['original_default_device'] = str(torch.get_default_device())
        if args.backend == 'oracle':
            # Stress the same explicit-CPU operations under a CUDA factory
            # default; this is a diagnostic, not the normal native engine setup.
            torch.set_default_device('cuda:%d' % local)
        report['tested_default_device'] = str(torch.get_default_device())

        def check_defaults():
            assert torch.get_default_device().type == 'cuda'
            if args.backend == 'jittor':
                assert jt.flags.use_cuda == 1

        def check_cpu(label, tensor, expected):
            check_defaults()
            assert tensor.tolist() == expected, (label, tensor.tolist(), expected)
            check = pointer_placement(tensor, 'cpu', local)
            check.update(label=label, values=expected, dtype=str(tensor.dtype))
            report['cpu_checks'].append(check)
            check_defaults()

        def gpu_sum(label, increment):
            check_defaults()
            tensor = torch.tensor([rank + increment, 2 * (rank + increment)],
                                  dtype=torch.float32, device='cuda:%d' % local)
            pointer_placement(tensor, 'cuda', local)
            dist.all_reduce(tensor)
            torch.cuda.synchronize()
            expected = [1 + 2 * increment, 2 * (1 + 2 * increment)]
            assert tensor.cpu().tolist() == expected
            check = pointer_placement(tensor, 'cuda', local)
            check.update(label=label, values=expected)
            report['gpu_checks'].append(check)

        check_defaults()
        print('before_init', rank, dist.is_initialized(), flush=True)
        if not dist.is_initialized():
            from datetime import timedelta
            dist.init_process_group('nccl', timeout=timedelta(seconds=120))
        print('after_init', rank, dist.distributed_c10d._get_default_store(), flush=True)
        initialized = True
        assert dist.get_backend() == 'nccl'
        print('before_cpu_group', rank, flush=True)
        cpu = dist.new_group([0, 1], backend='gloo')
        assert dist.get_backend(cpu) == 'gloo'
        print('before_gpu_sum', rank, flush=True)
        gpu_sum('before_cpu_control', 1)
        print('before_cpu_sum', rank, flush=True)

        value = torch.tensor([rank + 1, rank + 3], dtype=torch.int64, device='cpu')
        dist.all_reduce(value, group=cpu, async_op=True).wait()
        check_cpu('sum', value, [3, 7])

        source = torch.tensor([rank, rank + 2], dtype=torch.int32, device='cpu')
        gathered = [torch.empty((2,), dtype=torch.int32, device='cpu') for _ in range(2)]
        dist.all_gather(gathered, source, group=cpu)
        for index, tensor in enumerate(gathered):
            check_cpu('all_gather_%d' % index, tensor, [index, index + 2])
        flat = torch.empty((4,), dtype=torch.int32, device='cpu')
        dist.all_gather_into_tensor(flat, source, group=cpu)
        check_cpu('all_gather_into_tensor', flat, [0, 2, 1, 3])
        dist.broadcast(source, src=1, group=cpu)
        check_cpu('broadcast', source, [1, 3])

        objects = [None, None]
        dist.all_gather_object(objects, {'rank': rank, 'payload': 'x' * (rank + 1)}, group=cpu)
        expected_objects = [{'rank': r, 'payload': 'x' * (r + 1)} for r in range(2)]
        assert objects == expected_objects
        broadcast = [{'rank': 1}, [2, 3]] if rank == 1 else [None, None]
        dist.broadcast_object_list(broadcast, src=1, group=cpu)
        assert broadcast == [{'rank': 1}, [2, 3]]
        collected = [None, None] if rank == 1 else None
        dist.gather_object({'rank': rank}, collected, dst=1, group=cpu)
        if rank == 1:
            assert collected == [{'rank': 0}, {'rank': 1}]
        report['object_collectives'] = dict(all_gather=objects, broadcast=broadcast, gather=collected)
        dist.barrier(group=cpu)
        check_defaults()
        dist.destroy_process_group(cpu)
        assert dist.is_initialized() and dist.get_backend() == 'nccl'
        gpu_sum('after_cpu_subgroup_destroy', 10)
        check_defaults()
        report.update(status='completed', final_default_device=str(torch.get_default_device()),
                      final_use_cuda=int(jt.flags.use_cuda) if args.backend == 'jittor' else None,
                      world_backend=str(dist.get_backend()))
        dist.destroy_process_group()
        initialized = False
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2))
        # On failure the parent terminates the paired process group. Avoid
        # another collective or uncertain teardown obscuring the first error.


def parent(args):
    root = Path(args.output).parent
    root.mkdir(parents=True, exist_ok=True)
    paths = [root / ('%s-mixed-rank%d.%s' % (args.backend, rank, ext))
             for rank in range(2) for ext in ('log', 'json')]
    if Path(args.output).exists() or any(path.exists() for path in paths):
        raise FileExistsError('Use a new output directory to retain earlier evidence')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    processes, logs = [], []
    report = dict(backend=args.backend, status='running', cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
    try:
        for rank in range(2):
            env = dict(os.environ, MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port),
                       RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='2')
            if args.backend == 'jittor':
                env['JITTOR_HOME'] = os.environ['TP2_RANK%d_CACHE' % rank]
            logfile = open(root / ('%s-mixed-rank%d.log' % (args.backend, rank)), 'x')
            logs.append(logfile)
            command = [sys.executable, __file__, '--backend', args.backend, '--worker',
                       '--output', str(root / ('%s-mixed-rank%d.json' % (args.backend, rank)))]
            processes.append(subprocess.Popen(command, env=env, stdout=logfile,
                                              stderr=subprocess.STDOUT, start_new_session=True))
        deadline = time.monotonic() + 600
        while any(proc.poll() is None for proc in processes):
            if any(proc.poll() not in (None, 0) for proc in processes):
                raise RuntimeError('A mixed-collective rank failed; inspect per-rank JSON/logs')
            if time.monotonic() > deadline:
                raise TimeoutError('mixed-collective smoke exceeded 600 seconds')
            time.sleep(.5)
        assert all(proc.returncode == 0 for proc in processes)
        rows = [json.loads((root / ('%s-mixed-rank%d.json' % (args.backend, rank))).read_text())
                for rank in range(2)]
        assert all(row['status'] == 'completed' for row in rows)
        assert {row['gpu_checks'][0]['driver_device'] for row in rows} == {0, 1}
        report.update(status='completed', ranks=rows)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        for proc in processes:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
        for logfile in logs:
            logfile.close()
        Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['jittor', 'oracle'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--worker', action='store_true')
    arguments = parser.parse_args()
    worker(arguments) if arguments.worker else parent(arguments)

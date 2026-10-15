"""Private-environment entry for the public native multinode acceptance runner."""

import argparse
import os
from pathlib import Path
import socket
import sys
import time

from run_native_multinode import source_snapshot, write_json


def warm(root):
    import numpy as np
    import jittor as jt
    from jittor import distributed as dist
    from jittor.distributed.fsdp import FullyShardedDataParallel, MixedPrecisionPolicy

    assert root / 'python' in Path(jt.__file__).resolve().parents
    assert 'torch' not in sys.modules
    assert jt.compiler.has_cuda and jt.flags.use_cuda and jt.world_size == 1 and jt.rank == 0
    jt.compile_extern.get_library('nccl', load=True)
    dist.init_process_group('nccl', init_method='env://')
    before = jt.core.backend_fallback_count()
    assert before == 0
    sys.path.insert(0, str(root / 'tests'))
    from _helpers.native_fsdp import model, optimizer, samples

    with jt.flag_scope(backend_fallback='error'):
        for dtype in ('float32', 'bfloat16'):
            value = jt.array(np.arange(8, dtype='float32')).cast(dtype).cuda(0)
            for operation in ('all_reduce', 'broadcast', 'all_gather', 'reduce_scatter'):
                function = getattr(jt.compile_extern.nccl_ops, 'nccl_' + operation)
                output = function(value, 0) if operation == 'broadcast' else function(value)
                np.testing.assert_array_equal(output.float32().numpy(), np.arange(8))
        # The model, initialization, samples and optimizer remain owned by the
        # maintained fixture. This singleton pass only serializes kernel JIT.
        for kind in ('sgd', 'adamw'):
            for dtype in ('float32', 'bfloat16'):
                policy = {} if dtype == 'float32' else {'mixed_precision': MixedPrecisionPolicy(
                    param_dtype=dtype, reduce_dtype='float32', output_dtype='float32')}
                module = FullyShardedDataParallel(model(jt, native=True, dropout=0.25), **policy)
                opt = optimizer(jt, module, kind)
                for step in range(3):
                    x, y = samples(step, 0)
                    x, y = jt.array(x).start_grad(), jt.array(y).stop_grad()
                    loss = ((module(x) - y) ** 2).mean()
                    assert np.isfinite(jt.grad(loss, x, retain_graph=True).numpy()).all()
                    opt.step(loss)
                    for parameter in module.parameters():
                        assert np.isfinite(parameter.numpy()).all()
        jt.sync_all(True)
    assert jt.core.backend_fallback_count() == before
    print('WARM-READY singleton rank=0 world=1 fallback_delta=0', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('warm', 'collectives', 'parity', 'baseline', 'resume'), required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--node-rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()
    source_snapshot(root, args.source_sha)
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    expected_world = 1 if args.stage == 'warm' else 2
    expected_rank = 0 if args.stage == 'warm' else args.node_rank
    if (world != expected_world or rank != expected_rank
            or int(os.environ['LOCAL_WORLD_SIZE']) != 1 or int(os.environ['LOCAL_RANK']) != 0):
        raise ValueError('public launcher did not assign the expected real ranks')
    phase = 'warm-node%d' % args.node_rank if args.stage == 'warm' else args.stage
    identity = {
        'pid': os.getpid(), 'parent_pid': os.getppid(), 'start_ns': time.time_ns(),
        'host_id': socket.gethostname(), 'rank': rank, 'world_size': world,
        'node_rank': args.node_rank, 'local_world_size': 1,
        'external_store': os.environ.get('JITTOR_LAUNCHER_EXTERNAL_STORE'),
        'home': os.environ['HOME'], 'jittor_home': os.environ['JITTOR_HOME'],
        'tmpdir': os.environ['TMPDIR'], 'visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
        'source_sha': args.source_sha, 'phase': phase,
    }
    write_json(args.output / 'cuda-unit/worker-identities' / phase / ('rank%d.json' % rank), identity)
    if args.stage == 'warm':
        warm(root)
        return
    if args.stage == 'collectives':
        targets = ['tests/backends/comm/nccl/test_launcher_store_groups.py',
                   'tests/backends/comm/nccl/test_native_collectives.py']
    else:
        targets = ['tests/backends/comm/nccl/test_native_fsdp_nccl.py::test_oracle_and_fresh_launcher_resume']
    # Preserve the recorded PID through exec into the unmodified maintained test.
    os.execvpe(sys.executable, [sys.executable, '-m', 'pytest', '-q', *targets], os.environ)


if __name__ == '__main__':
    main()

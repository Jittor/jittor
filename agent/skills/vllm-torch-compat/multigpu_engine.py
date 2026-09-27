"""TP=2 diagnostic wrapper: per-worker cache env and real CUDA placement RPC.

Uses the unchanged acceptance.py TP=2 options and generation, isolates worker
caches, checks placement, and explicitly verifies worker/shared-memory cleanup.
"""
import gc, json, os, runpy, sys
from contextlib import contextmanager
from pathlib import Path

def placement(worker_wrapper, install=False):
    import ctypes
    import torch
    binary_paths = {}
    if hasattr(torch, '_torch_compat_install_context'):
        import jittor as jt
        binary_paths = dict(core=jt.core.__file__, nccl=jt.compile_extern.nccl.__file__)
        expected = os.environ['JITTOR_HOME'] + '/'
        assert all(path.startswith(expected) for path in binary_paths.values()), binary_paths
    worker = worker_wrapper.worker
    runner = worker.model_runner
    replays = []
    if binary_paths and not install:
        for name, module in runner.model.named_modules():
            state = module.__dict__.get('_auto_graph_replay')
            if state is not None:
                replay = state.replay
                replays.append(dict(name=name, seen=state.seen, give_up=state.give_up,
                    stats=None if replay is None else replay.stats,
                    refused=None if replay is None else replay.refused))
    params = list(runner.model.named_parameters())
    caches = list(runner.kv_caches)
    assert params and caches, 'Missing model/KV tensors'
    tensors = [t for _, t in params] + caches
    assert all(t.device.type == 'cuda' for t in tensors)
    driver = ctypes.CDLL('libcuda.so.1')
    ordinals = set()
    for tensor in (params[0][1], caches[0]):
        memory, ordinal = ctypes.c_uint(), ctypes.c_int()
        ptr = ctypes.c_uint64(tensor.data_ptr())
        assert driver.cuPointerGetAttribute(ctypes.byref(memory), 2, ptr) == 0
        assert driver.cuPointerGetAttribute(ctypes.byref(ordinal), 9, ptr) == 0
        assert memory.value == 2, 'Model/KV pointer is not CUDA device memory'
        ordinals.add(ordinal.value)
    assert len(ordinals) == 1
    if install:
        runner._tp2_placement_calls = 0
        runner._tp2_placement_outputs = set()
        def check(module, args, output):
            outputs = [output] if isinstance(output, torch.Tensor) else list(output)
            real = [t for t in outputs if isinstance(t, torch.Tensor)]
            assert real and all(t.device.type == 'cuda' for t in real)
            runner._tp2_placement_calls += 1
            runner._tp2_placement_outputs.update(str(t.device) for t in real)
        runner._tp2_placement_hook = runner.model.register_forward_hook(check)
    elif install is False:
        assert runner._tp2_placement_calls > 0, 'No actual model forward observed'
    return dict(binary_paths=binary_paths, auto_replays=replays, rank=worker.rank, local_rank=worker.local_rank, parameters=len(params), kv_caches=len(caches),
                reported_devices=sorted({str(t.device) for t in tensors}), driver_devices=sorted(ordinals),
                forward_calls=getattr(runner, '_tp2_placement_calls', 0), forward_output_devices=sorted(getattr(runner, '_tp2_placement_outputs', ())),
                jittor_home=os.environ.get('JITTOR_HOME'))

@contextmanager
def isolated_workers(backend, root):
    """Spawn each Jittor rank from its own warm cache and extension path."""
    root = Path(root)
    from vllm.v1.executor.multiproc_executor import WorkerProc
    make_worker = WorkerProc.make_worker_process
    def isolated_worker(*args, **kwargs):
        local_rank = kwargs['local_rank'] if 'local_rank' in kwargs else args[1]
        previous = os.environ.get('JITTOR_HOME')
        previous_path = sys.path[:]
        if backend == 'jittor':
            os.environ['JITTOR_HOME'] = os.environ.get('TP2_RANK%d_CACHE' % local_rank, str(root / ('cache-rank%d' % local_rank)))
        if backend == 'jittor':
            sys.path[:] = [item for item in sys.path if '/.cache/jittor/' not in item]
        try:
            return make_worker(*args, **kwargs)
        finally:
            sys.path[:] = previous_path
            if previous is None: os.environ.pop('JITTOR_HOME', None)
            else: os.environ['JITTOR_HOME'] = previous
    WorkerProc.make_worker_process = staticmethod(isolated_worker)
    try:
        yield
    finally:
        WorkerProc.make_worker_process = staticmethod(make_worker)


def shutdown_engine(llm):
    """Use the pinned vLLM shutdown API and retain observable cleanup evidence."""
    client = llm.llm_engine.engine_core
    executor = client.engine_core.model_executor
    processes = [worker.proc for worker in executor.workers]
    def names():
        queues = [executor.rpc_broadcast_mq, *executor.response_mqs]
        for worker in executor.workers:
            queues.append(worker.worker_response_mq)
            queues.extend(getattr(worker, 'peer_worker_response_mqs', ()) or ())
        return sorted({queue.buffer.shared_memory.name for queue in queues
                       if queue is not None and getattr(queue, 'buffer', None) is not None})
    shared_memory_names = names()
    client.shutdown()
    gc.collect()
    rows = [dict(pid=process.pid, exitcode=process.exitcode, alive=process.is_alive())
            for process in processes]
    remaining = [name for name in shared_memory_names
                 if (Path('/dev/shm') / name.lstrip('/')).exists()]
    return dict(workers=rows, shared_memory_names=shared_memory_names,
                remaining_shared_memory=remaining,
                passed=len(rows) == 2 and all(row['exitcode'] == 0 and not row['alive']
                                             for row in rows) and not remaining)


def main():
    backend = sys.argv[sys.argv.index('--backend') + 1]
    root = Path(os.environ['TP2_RUN_ROOT'])
    if backend == 'jittor':
        import jittor as jt
        jt.flags.use_parallel_op_compiler = 0
        jt.flags.use_cuda = 1
    from vllm import LLM
    instances = []
    original_init, original_generate = LLM.__init__, LLM.generate
    def observed_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        instances.append(self)
        executor = self.llm_engine.engine_core.engine_core.model_executor
        rows = executor.collective_rpc(placement, args=(True,), timeout=120)
        assert len(rows) == 2 and {row['rank'] for row in rows} == {0, 1}
        assert {tuple(row['driver_devices']) for row in rows} == {(0,), (1,)}
        (root / (backend + '-placement-before.json')).write_text(json.dumps(rows, indent=2))
        if int(os.environ.get('TP2_TRACE_STEPS', '0')):
            from multigpu_trace import install_trace
            executor.collective_rpc(install_trace, timeout=120)
    def observed_generate(self, *args, **kwargs):
        out = original_generate(self, *args, **kwargs)
        rows = self.llm_engine.engine_core.engine_core.model_executor.collective_rpc(placement, args=(False,), timeout=120)
        (root / (backend + '-placement-after.json')).write_text(json.dumps(rows, indent=2))
        return out
    LLM.__init__, LLM.generate = observed_init, observed_generate
    try:
        with isolated_workers(backend, root):
            runpy.run_path(str(Path(__file__).with_name('acceptance.py')), run_name='__main__')
    finally:
        try:
            for instance in instances:
                lifecycle = shutdown_engine(instance)
                (root / (backend + '-lifecycle.json')).write_text(json.dumps(lifecycle, indent=2))
                assert lifecycle['passed'], lifecycle
        finally:
            LLM.__init__, LLM.generate = original_init, original_generate
if __name__ == '__main__':
    main()

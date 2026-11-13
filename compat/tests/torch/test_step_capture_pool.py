"""Shape captures share scratch, while returned values and state stay valid."""
import gc
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
import jittor as jt


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graphs are required')
def test_shape_captures_share_scratch_and_preserve_previous_results():
    def trajectory(compiled, shared=True):
        parameter = torch.nn.Parameter(torch.full((512,), .2, device='cuda'))
        optimizer = torch.optim.AdamW([parameter], lr=.01)

        def step(x):
            loss = (x * parameter).tanh().sin().exp().mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return loss.detach(), parameter[:2].detach().clone()

        functions = [step, step] if shared else [lambda x: step(x), lambda x: step(x)]
        runners = [torch.compile(fn, mode='reduce-overhead') for fn in functions] if compiled else functions
        outputs, history = [], []
        for index in range(14):
            x = torch.full((1024 + 256 * (index % 2), 512), 1. + .03 * index, device='cuda')
            result = runners[index % 2](x)
            torch.cuda.synchronize()
            outputs.append(result)
            history.append((result[0].item(), parameter.detach().cpu().numpy().copy()))
        # The copied return of an earlier call must survive both recordings
        # writing their shared scratch on later calls.
        for result, (loss, values) in zip(outputs, history):
            np.testing.assert_allclose(result[0].cpu().numpy(), loss)
            np.testing.assert_allclose(result[1].cpu().numpy(), values[:2])
        reserved = 0
        if compiled:
            assert all(r.stats['graph'] >= 3 and r.refused is None for r in runners)
            pools = [r._pool for r in runners]
            assert (pools[0] is pools[1]) == shared
            reserved = sum(jt.core._graph_pool_stats(p.handle)[1] for p in set(pools))
            runners[0].invalidate()
            # Releasing one executable cannot invalidate the other's arena.
            runners[1](torch.full((1280, 512), 1.5, device='cuda'))
            torch.cuda.synchronize()
            assert runners[1].stats['graph'] >= 4
            runners[1].invalidate()
            del pools
        return history, reserved

    expected, _ = trajectory(False)
    shared, shared_bytes = trajectory(True)
    separate, separate_bytes = trajectory(True, shared=False)
    for actual in (shared, separate):
        for (loss, values), (want_loss, want_values) in zip(actual, expected):
            np.testing.assert_allclose(loss, want_loss, rtol=2e-6, atol=1e-7)
            np.testing.assert_allclose(values, want_values, rtol=2e-6, atol=1e-7)
    assert shared_bytes < separate_bytes
    gc.collect()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph arenas are required')
def test_live_tensor_outlives_a_released_pool_handle():
    pool = jt.core._graph_pool_create()
    token = jt.core._graph_pool_scope_begin(pool, True)
    try:
        value = torch.full((128,), 3., device='cuda')
        torch.cuda.synchronize()
    finally:
        jt.core._graph_pool_scope_end(token)
    jt.core._graph_pool_release(pool)
    jt.gc()
    assert value.device.type == 'cuda'
    np.testing.assert_array_equal(value.cpu().numpy(), np.full(128, 3.))
    with pytest.raises(RuntimeError, match='invalid or released graph pool'):
        jt.core._graph_pool_release(pool)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph arenas are required')
@torch.no_grad()
def test_shared_scratch_orders_graphs_launched_from_different_threads():
    def step(x):
        return x.tanh().sin().exp().sum().detach()

    runners = [torch.compile(step, mode='reduce-overhead') for _ in range(2)]
    shapes = [(1024, 512), (1280, 512)]
    for _ in range(7):
        for runner, shape in zip(runners, shapes):
            runner(torch.full(shape, .5, device='cuda'))
    torch.cuda.synchronize()
    assert runners[0]._pool is runners[1]._pool
    assert all(r.stats['graph'] >= 2 for r in runners), [
        (r.stats, r.refused, r._graph_refused) for r in runners]

    outputs = []
    # Each worker has its own CUDA per-thread stream. Keep returned values
    # until both workers have repeatedly reused the same scratch addresses.
    with ThreadPoolExecutor(max_workers=1) as first, ThreadPoolExecutor(max_workers=1) as second:
        for index in range(12):
            value = .1 + index * .03
            futures = [worker.submit(runner, torch.full(shape, value, device='cuda'))
                       for worker, runner, shape in zip((first, second), runners, shapes)]
            outputs.extend((future.result(timeout=30), shape, value)
                           for future, shape in zip(futures, shapes))
    torch.cuda.synchronize()
    for output, shape, value in outputs:
        expected = np.exp(np.sin(np.tanh(np.float32(value)))) * np.prod(shape)
        np.testing.assert_allclose(output.cpu().numpy(), expected, rtol=2e-6)
    for runner in runners:
        runner.invalidate()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph arenas are required')
def test_pool_scope_errors_restore_the_original_allocation_scope():
    first, second = jt.core._graph_pool_create(), jt.core._graph_pool_create()
    token = jt.core._graph_pool_scope_begin(first, True)
    try:
        with pytest.raises(RuntimeError, match='different graph pools cannot nest'):
            jt.core._graph_pool_scope_begin(second, True)
        with pytest.raises(RuntimeError, match='close in order'):
            jt.core._graph_pool_scope_end(token + 1)
        value = torch.full((128,), 4., device='cuda')
        torch.cuda.synchronize()
    finally:
        jt.core._graph_pool_scope_end(token)
    jt.core._graph_pool_release(first)
    jt.core._graph_pool_release(second)
    np.testing.assert_array_equal(value.cpu().numpy(), np.full(128, 4.))

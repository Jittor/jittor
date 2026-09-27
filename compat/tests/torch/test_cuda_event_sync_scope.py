"""An event wait must not execute graphs constructed after its record point.

CPU runs verify shim control semantics using real lazy Jittor graphs. Set
JITTOR_EVENT_TEST_DEVICE=cuda for the corresponding real-device verification.
The background wait reproduces vLLM's async output thread, which must not take
over execution of the model thread's next, potentially incomplete graph.
"""
import os
import threading

import pytest
import torch


@pytest.mark.parametrize('background', [False, True])
@pytest.mark.parametrize('recorded', [False, True])
def test_event_wait_does_not_submit_future_graphs(background, recorded):
    import jittor as jt

    device = os.environ.get('JITTOR_EVENT_TEST_DEVICE', 'cpu')
    if device not in ('cpu', 'cuda'):
        raise ValueError('JITTOR_EVENT_TEST_DEVICE must be cpu or cuda')
    if device == 'cuda' and not jt.has_cuda:
        pytest.skip('CUDA unavailable')
    with jt.runtime.scope(use_cuda=int(device == 'cuda')):
        event = torch.cuda.Event()
        if recorded:
            prior = (torch.arange(8, dtype=torch.float32, device=device) * 3).sum()
            assert not prior.is_finished
            event.record()
            assert prior.is_finished, 'record must complete the preceding graph'
            assert prior.item() == 84

        future = (torch.arange(8, dtype=torch.float32, device=device) * 5 + 2).sum()
        assert not future.is_finished
        executions = jt.flags.exec_called
        failures = []

        def wait():
            try:
                event.synchronize()
            except BaseException as error:
                failures.append(error)

        if background:
            worker = threading.Thread(target=wait)
            worker.start()
            worker.join(timeout=30)
            assert not worker.is_alive(), 'event wait did not finish'
        else:
            wait()
        assert not failures, failures
        assert jt.flags.exec_called == executions, (
            'event wait submitted work created after its record point')
        assert not future.is_finished, 'event wait executed a future graph'
        assert future.item() == 156

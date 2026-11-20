"""Nonblocking sync delivers completed fetches without a blocking flush."""
from threading import Event
import time

import jittor as jt
import numpy as np
import pytest


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',['int32','int64'])
def test_ready_fetch_progresses_with_only_nonblocking_sync(device,dtype):
    if device=='cuda' and not jt.has_cuda:
        pytest.skip('real CUDA device required')
    with jt.flag_scope(use_cuda=int(device=='cuda'),auto_graph_replay=0):
        expected=np.arange(32,dtype=dtype).reshape(8,4)
        source=jt.array(expected)
        source.sync(True)
        if device=='cuda':
            assert source.location()=='device'
        ready=Event()
        arrays=[]
        def complete(array):
            arrays.append(np.array(array,copy=True))
            ready.set()
        jt.fetch(source,complete)
        deadline=time.monotonic()+5.0
        try:
            while not ready.is_set() and time.monotonic()<deadline:
                jt.sync_all(False)
                ready.wait(.0001)
            delivered_before_blocking_cleanup=ready.is_set()
        finally:
            jt.sync_all(True)
        assert delivered_before_blocking_cleanup
        assert len(arrays)==1
        np.testing.assert_array_equal(arrays[0],expected)

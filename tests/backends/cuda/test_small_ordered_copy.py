"""Compute-ordered storage copies preserve bytes and pending readers."""
import jittor as jt
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def cuda_scope():
    if not jt.has_cuda:
        pytest.skip('real CUDA device required')
    with jt.flag_scope(use_cuda=1, auto_graph_replay=0):
        yield


@pytest.mark.parametrize('count', [0, 1, 32, 1024, 16384, 16385, 262144])
def test_exact_bytes_at_small_copy_boundary(count):
    # Raw integer bits also cover signed zero, infinities and NaN payloads.
    bits=np.resize(np.array([0,0x80000000,0x7f800000,0x7fc12345,
                             0xff800000,0xffffffff],dtype=np.uint32),count)
    source=jt.array(bits.view(np.int32)).cuda()
    destination=jt.empty(source.shape,dtype='int32')
    jt.sync([source,destination],True)
    address=destination._storage_address
    destination._copy_into(source,False)
    assert destination._storage_address==address
    if count:
        assert destination.device_id==source.device_id>=0
    np.testing.assert_array_equal(destination.numpy(),bits.view(np.int32))


@pytest.mark.parametrize('count', [32,16384,16385])
def test_pending_producer_copy_and_consumer_keep_compute_order(count):
    source=jt.zeros((count,),dtype='int32').cuda()
    destination=jt.zeros_like(source)
    jt.sync([source,destination],True)
    consumers=[]
    for step in range(12):
        source=source+1
        source.sync(False)
        destination._copy_into(source,False)
        value=destination*2
        value.sync(False)
        consumers.append(value)
    for step,value in enumerate(consumers):
        np.testing.assert_array_equal(value.numpy(),
            np.full(count,2*(step+1),dtype=np.int32))

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_event_queue_only_exposes_async_queue_and_worker_lifecycle():
    header = (ROOT / "python/jittor/src/event_queue.h").read_text()
    implementation = (ROOT / "python/jittor/src/event_queue.cc").read_text()
    for dead in ("run_sync", "run_sync_done", "worker_caller", "volatile int"):
        assert dead not in header
        assert dead not in implementation
    assert "inline void push(Func func)" in header
    assert "inline void flush()" in header
    assert "void EventQueue::Worker::start()" in implementation
    assert "void EventQueue::Worker::stop()" in implementation


def test_nccl_device_selection_does_not_depend_on_event_queue_sync_shim():
    source = (ROOT / "python/jittor/extern/cuda/nccl/src/nccl_wrapper.cc").read_text()
    assert '"event_queue.h"' not in source
    assert "event_queue.run_sync" not in source
    assert "checkCudaErrors(cudaSetDevice(nccl_device_id));" in source

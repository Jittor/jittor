"""CPU Tensor prefetch preserves order, lifetimes, errors and worker cleanup."""
import os
import time
import gc
import socket

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not hasattr(os, 'memfd_create') or not hasattr(socket, 'SOCK_SEQPACKET'),
    reason='CPU process transport requires Linux memfd and SOCK_SEQPACKET')


class ProcessSamples(torch.utils.data.Dataset):
    def __len__(self):
        return 9

    def __getitem__(self, index):
        if index % 3 == 0:
            time.sleep(.01)
        worker = torch.utils.data.get_worker_info()
        return {'value': torch.tensor([index, index + 1], dtype=torch.float32, device='cpu'),
                'pid': os.getpid(), 'worker': -1 if worker is None else worker.id,
                'cuda': torch.cuda.is_available()}


def test_process_tensor_batches_preserve_order_duplicates_and_retained_results():
    order = [[3, 1, 3], [2], [8, 0], [7], [4, 6]]
    loader = torch.utils.data.DataLoader(ProcessSamples(), batch_sampler=order, num_workers=2)
    iterator = iter(loader)
    batches = [next(iterator)]
    processes = [process for _, process in iterator._process_pool.actors]
    batches.extend(iterator)
    assert all(process.poll() is not None for process in processes)
    worker_ids = set()
    for batch, indices in zip(batches, order):
        assert batch['value'].device.type == 'cpu'
        np.testing.assert_array_equal(batch['value'].numpy(), [[i, i + 1] for i in indices])
        if not np.all(batch['pid'].numpy() == os.getpid()):
            assert not batch['cuda'].numpy().any()
            worker_ids.update(batch['worker'].numpy().tolist())
    assert worker_ids == {0, 1}


class MixedSamples(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return (torch.tensor(index, dtype=torch.int64, device='cpu'),
                torch.tensor(index % 2 == 0, dtype=torch.bool, device='cpu'),
                np.array([index], dtype='float64'))


def test_unbatched_process_results_preserve_scalar_dtype_and_numpy_leaves():
    loader = torch.utils.data.DataLoader(MixedSamples(), batch_size=None, num_workers=2,
                                         collate_fn=lambda sample: sample)
    batches = list(loader)
    for index, (integer, boolean, numpy_value) in enumerate(batches):
        assert tuple(integer.shape) == tuple(boolean.shape) == ()
        assert integer.dtype == torch.int64 and boolean.dtype == torch.bool
        assert isinstance(numpy_value, np.ndarray) and numpy_value.dtype == np.float64
        assert integer.item() == index and boolean.item() == (index % 2 == 0)
        np.testing.assert_array_equal(numpy_value, [index])


class FailingSamples(ProcessSamples):
    def __getitem__(self, index):
        if index >= 2:
            raise ValueError('sample decoding failed')
        return super().__getitem__(index)


def test_process_error_keeps_original_exception_and_stops_all_workers():
    loader = torch.utils.data.DataLoader(FailingSamples(), batch_size=2, num_workers=2)
    iterator = iter(loader)
    next(iterator)
    processes = [process for _, process in iterator._process_pool.actors]
    with pytest.raises(ValueError, match='sample decoding failed'):
        next(iterator)
    assert all(process.poll() is not None for process in processes)


class CrashingSamples(ProcessSamples):
    def __getitem__(self, index):
        if index:
            os._exit(17)
        return super().__getitem__(index)


def test_unexpected_worker_exit_is_reported_and_remaining_workers_stop():
    iterator = iter(torch.utils.data.DataLoader(CrashingSamples(), batch_size=1, num_workers=2))
    next(iterator)
    processes = [process for _, process in iterator._process_pool.actors]
    with pytest.raises(RuntimeError, match='worker (channel failed|processes exited)'):
        next(iterator)
    assert all(process.poll() is not None for process in processes)


class SlowSamples(ProcessSamples):
    def __getitem__(self, index):
        if index:
            time.sleep(10)
        return super().__getitem__(index)


def test_timeout_and_abandoned_iterator_stop_processes():
    iterator = iter(torch.utils.data.DataLoader(SlowSamples(), batch_size=1, num_workers=2, timeout=.1))
    next(iterator)
    processes = [process for _, process in iterator._process_pool.actors]
    with pytest.raises(RuntimeError, match='timed out'):
        next(iterator)
    assert all(process.poll() is not None for process in processes)
    iterator = iter(torch.utils.data.DataLoader(ProcessSamples(), batch_size=1, num_workers=2))
    next(iterator)
    processes = [process for _, process in iterator._process_pool.actors]
    iterator._shutdown()
    assert all(process.poll() is not None for process in processes)


def test_transport_reuses_mapping_only_after_all_numpy_aliases_are_released():
    from jittor.compat.torch._dataloader_ipc import send_frame, receive_frame
    from jittor.compat.torch._dataloader_shared_memory import SharedMemoryPool
    pool = SharedMemoryPool()
    writer, reader = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    mappings, released = {}, []
    try:
        send_frame(writer, 'batch', 0, np.arange(8, dtype='float32'), pool)
        frame = receive_frame(reader, mappings, released.append)
        first_mapping = frame.memory
        original = frame.decode()
        assert type(original) is np.ndarray
        alias = original[1:]
        del original, frame
        gc.collect()
        assert not released
        send_frame(writer, 'batch', 1, np.full(8, 19, dtype='float32'), pool)
        frame = receive_frame(reader, mappings, released.append)
        retained = frame.decode()
        assert frame.memory is not first_mapping
        np.testing.assert_array_equal(alias, np.arange(1, 8))
        del alias
        gc.collect()
        assert released == [0]
        pool.release(released.pop())
        send_frame(writer, 'batch', 2, np.full(8, 23, dtype='float32'), pool)
        third = receive_frame(reader, mappings, released.append)
        assert third.memory is first_mapping
        np.testing.assert_array_equal(third.decode(), np.full(8, 23))
        np.testing.assert_array_equal(retained, np.full(8, 19))
    finally:
        writer.close()
        reader.close()
        pool.close()


def test_process_tensor_leases_survive_reuse_and_iterator_shutdown():
    order = [[index % 9] for index in range(80)]
    iterator = iter(torch.utils.data.DataLoader(ProcessSamples(), batch_sampler=order, num_workers=2))
    retained = []
    for index, batch in enumerate(iterator):
        if torch.cuda.is_available():
            transferred = batch['value'].to('cuda')
            assert transferred.device.type == 'cuda'
            np.testing.assert_array_equal((transferred + 1).cpu().numpy(),
                                          [[order[index][0] + 1, order[index][0] + 2]])
        if index % 11 == 0:
            retained.append((batch['value'].detach(), order[index][0]))
    gc.collect()
    for value, expected in retained:
        np.testing.assert_array_equal(value.numpy(), [[expected, expected + 1]])

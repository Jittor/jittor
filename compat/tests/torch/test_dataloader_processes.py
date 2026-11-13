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
    reason='CPU process transport requires os.memfd_create and socket.SOCK_SEQPACKET')


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


class NumpySamples(torch.utils.data.Dataset):
    def __len__(self):
        return 9

    def __getitem__(self, index):
        return {'image': np.full((3, 4), index/10, dtype='float32'),
                'wide': np.array([index+.25], dtype='float64'),
                'label': np.int64(index), 'mask': np.bool_(index % 2),
                'complex': np.array([index+2j], dtype='complex64')}


def test_numpy_collation_preserves_dtypes_and_retained_batches_after_worker_reuse():
    order = [[0, 3, 1], [7], [6, 2], [8, 4, 5]]
    loader = torch.utils.data.DataLoader(NumpySamples(), batch_sampler=order,
                                         num_workers=2, persistent_workers=True)
    try:
        first = list(loader)
        order.reverse()
        later = list(loader)
        for batches, indices in ((first, list(reversed(order))), (later, order)):
            for batch, selected in zip(batches, indices):
                assert batch['image'].device.type == 'cpu' and not batch['image'].requires_grad
                assert batch['image'].dtype == torch.float32
                assert batch['wide'].dtype == torch.float64
                assert batch['label'].dtype == torch.int64
                assert batch['mask'].dtype == torch.bool
                assert batch['complex'].dtype == torch.complex64
                np.testing.assert_allclose(batch['image'].numpy(),
                                           np.array([np.full((3, 4), i/10) for i in selected]))
                np.testing.assert_array_equal(batch['label'].numpy(), selected)
                np.testing.assert_array_equal(batch['complex'].numpy().reshape(-1),
                                              np.array(selected)+2j)
        if torch.cuda.is_available():
            result = first[0]['image'].to('cuda')*2
            assert result.device.type == 'cuda'
            np.testing.assert_allclose(result.cpu().numpy(), first[0]['image'].numpy()*2)
    finally:
        del loader
        gc.collect()
    np.testing.assert_array_equal(first[0]['label'].numpy(), [0, 3, 1])


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


@pytest.mark.parametrize('dtype,shape', [('float32', (3, 4)), ('int64', ()),
                                         ('complex64', (5,))])
def test_process_encoding_borrows_cpu_storage_until_transport_copies_it(dtype, shape):
    from jittor.compat.torch._dataloader_ipc import encode_batch, send_frame, receive_frame, decode_batch
    source = np.arange(max(1, int(np.prod(shape))), dtype='float32').astype(dtype).reshape(shape)
    tensor = torch.from_numpy(source)
    encoded = encode_batch(tensor)
    assert np.shares_memory(encoded[2], source)
    source[...] = 7
    np.testing.assert_array_equal(encoded[2], np.full(shape, 7))
    # The borrowed view keeps its tensor owner alive, including scalar and
    # complex storage. Transmission still produces an independent snapshot.
    del tensor
    gc.collect()
    writer, reader = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        send_frame(writer, 'batch', 0, encoded)
        source[...] = 11
        frame = receive_frame(reader)
        restored = decode_batch(frame.decode())
        assert restored.device.type == 'cpu'
        np.testing.assert_array_equal(restored.numpy(), np.full(shape, 7))
        np.testing.assert_array_equal(encoded[2], np.full(shape, 11))
    finally:
        writer.close()
        reader.close()


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


def test_persistent_process_workers_reuse_pids_and_preserve_retained_batches():
    batches = [[3], [1], [8], [2], [6]]
    loader = torch.utils.data.DataLoader(ProcessSamples(), batch_sampler=batches,
                                         num_workers=2, persistent_workers=True)
    iterator = iter(loader)
    retained = [next(iterator)]
    actors = [process for _, process in iterator._process_pool.actors]
    try:
        retained.extend(iterator)
        assert all(process.poll() is None for process in actors)
        batches[:] = [[6], [3], [1], [8], [2]]
        later = list(loader)
        for i, batch in enumerate(later):
            assert batch['pid'].item() == actors[i % len(actors)].pid
            index, = batches[i]
            np.testing.assert_array_equal(batch['value'].numpy(), [[index, index+1]])
        if torch.cuda.is_available():
            transferred = later[0]['value'].to('cuda') + 1
            assert transferred.device.type == 'cuda'
            np.testing.assert_array_equal(transferred.cpu().numpy(), [[7, 8]])
        for index, batch in zip((3, 1, 8, 2, 6), retained):
            np.testing.assert_array_equal(batch['value'].numpy(), [[index, index+1]])
    finally:
        del iterator, loader
        gc.collect()
    assert all(process.poll() is not None for process in actors)


def test_persistent_workers_retire_an_abandoned_pass_before_restart():
    batches = [[3], [1], [8], [2], [6]]
    loader = torch.utils.data.DataLoader(ProcessSamples(), batch_sampler=batches,
                                         num_workers=2, persistent_workers=True)
    iterator = iter(loader)
    next(iterator)
    actors = [process for _, process in iterator._process_pool.actors]
    try:
        batches[:] = [[6], [2], [4]]
        later = list(loader)
        assert all(process.poll() is not None for process in actors)
        assert [batch['value'].numpy()[0, 0] for batch in later] == [6, 2, 4]
    finally:
        loader._iterator._shutdown()

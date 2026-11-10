"""CPU batch transport. Receiving a frame never constructs backend tensors."""
import array
import mmap
import os
import pickle
import socket

import numpy as np


class Frame:
    def __init__(self, kind, index, memory, metadata_size, buffers, lease=None):
        self.kind, self.index = kind, index
        self.memory, self.metadata_size, self.buffers = memory, metadata_size, buffers
        self.lease = lease

    def decode(self):
        import cloudpickle
        if self.lease is None:
            views = [memoryview(self.memory)[offset:offset + size] for offset, size in self.buffers]
        else:
            from ._dataloader_shared_memory import leased_buffer
            views = [leased_buffer(self.memory, offset, size, self.lease) for offset, size in self.buffers]
        return cloudpickle.loads(self.memory[:self.metadata_size], buffers=views)


def send_frame(channel, kind, index, value, pool=None):
    import cloudpickle
    buffers = []
    metadata = cloudpickle.dumps(value, protocol=5, buffer_callback=buffers.append)
    offsets = []
    total = (len(metadata) + 63) // 64 * 64
    for buffer in buffers:
        size = buffer.raw().nbytes
        offsets.append((total, size))
        total = (total + size + 63) // 64 * 64
    if pool is None:
        descriptor = os.memfd_create('jittor-loader', os.MFD_CLOEXEC)
        slot = None
    else:
        slot, descriptor, memory = pool.acquire(total)
    try:
        if pool is None:
            os.ftruncate(descriptor, max(total, 1))
            memory = mmap.mmap(descriptor, max(total, 1))
        try:
            memory[:len(metadata)] = metadata
            for buffer, (offset, size) in zip(buffers, offsets):
                memory[offset:offset + size] = buffer.raw()
        finally:
            if pool is None:
                memory.close()
        header = pickle.dumps((kind, index, len(metadata), offsets, slot), protocol=5)
        channel.sendmsg([header], [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                                   array.array('i', [descriptor]))])
    finally:
        if pool is None:
            os.close(descriptor)


def receive_frame(channel, mappings=None, release=None):
    header, ancillary, flags, _ = channel.recvmsg(1 << 20, socket.CMSG_SPACE(4))
    if not header:
        raise EOFError('DataLoader worker channel closed')
    descriptors = array.array('i')
    for level, kind, data in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            descriptors.frombytes(data[:len(data) // descriptors.itemsize * descriptors.itemsize])
    try:
        if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or len(descriptors) != 1:
            raise RuntimeError('incomplete DataLoader worker frame')
        stat = os.fstat(descriptors[0])
        key = (stat.st_dev, stat.st_ino)
        memory = None if mappings is None else mappings.get(key)
        new_mapping = memory is None
        if new_mapping:
            memory = mmap.mmap(descriptors[0], 0, access=mmap.ACCESS_WRITE)
    finally:
        for descriptor in descriptors:
            os.close(descriptor)
    # Do page faults on the IO thread. NumPy's reduction releases the GIL;
    # MAP_POPULATE inside mmap() would block the training thread on the GIL.
    if new_mapping and len(memory) > 1 << 20:
        np.frombuffer(memory, dtype=np.uint8)[::4096].sum(dtype=np.uint64)
    kind, index, metadata_size, buffers, slot = pickle.loads(header)
    lease = None
    if slot is not None:
        if mappings is None or release is None:
            raise RuntimeError('pooled DataLoader frame needs a lease receiver')
        from ._dataloader_shared_memory import BatchLease
        mappings[key] = memory
        lease = BatchLease(slot, release)
    return Frame(kind, index, memory, metadata_size, buffers, lease)


def encode_batch(value):
    import jittor as jt
    if isinstance(value, jt.Var):
        if value.device.type != 'cpu' or value.requires_grad:
            raise RuntimeError('DataLoader process workers require detached CPU tensors')
        dtype = str(value.dtype).rsplit('.', 1)[-1]
        if dtype == 'bfloat16':
            import torch
            data = value.view(torch.int16).numpy()
        else:
            data = value.numpy()
        return ('tensor', dtype, data)
    if type(value) is dict:
        return ('dict', [(key, encode_batch(item)) for key, item in value.items()])
    if type(value) in (tuple, list):
        return (type(value).__name__, [encode_batch(item) for item in value])
    return ('value', value)


def decode_batch(value):
    import jittor as jt
    from .context import get_install_context
    torch = get_install_context(jt).target_namespace
    kind = value[0]
    if kind == 'tensor':
        dtype, data = value[1:]
        if hasattr(data, '__dlpack__') and data.dtype.kind in 'fiuc':
            result = torch.Tensor(jt.from_dlpack(data, copy=False).stop_grad())
        else:
            result = torch.from_numpy(data)
        return result.view(torch.bfloat16) if dtype == 'bfloat16' else result
    if kind == 'dict':
        return {key: decode_batch(item) for key, item in value[1]}
    if kind == 'tuple':
        return tuple(decode_batch(item) for item in value[1])
    if kind == 'list':
        return [decode_batch(item) for item in value[1]]
    return value[1]

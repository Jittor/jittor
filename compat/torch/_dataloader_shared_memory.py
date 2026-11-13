"""Reusable CPU transport storage, leased until the last batch view dies."""
import ctypes
import mmap
import os

import numpy as np


class SharedMemoryPool:
    """Worker-owned allocations. A release message makes a slot reusable.

    Never resize a shared allocation: a retained parent array can still map
    it. Retained batches naturally require additional slots, like retained
    shared-memory tensors in a native process DataLoader.
    """
    def __init__(self):
        self.slots = []

    def acquire(self, size):
        for index, slot in enumerate(self.slots):
            if not slot[3] and slot[2] >= size:
                slot[3] = True
                return index, slot[0], slot[1]
        descriptor = os.memfd_create('jittor-loader-pool', os.MFD_CLOEXEC)
        try:
            os.ftruncate(descriptor, max(size, 1))
            memory = mmap.mmap(descriptor, max(size, 1))
        except BaseException:
            os.close(descriptor)
            raise
        self.slots.append([descriptor, memory, max(size, 1), True])
        return len(self.slots)-1, descriptor, memory

    def release(self, index):
        slot = self.slots[index]
        if not slot[3]:
            raise RuntimeError('DataLoader shared storage released twice')
        slot[3] = False

    def close(self):
        for descriptor, memory, _, _ in self.slots:
            memory.close()
            os.close(descriptor)
        self.slots.clear()


class BatchLease:
    def __init__(self, index, release):
        self.index, self.release = index, release

    def __del__(self):
        self.release(self.index)


class _BufferOwner:
    """An ndarray base that owns both its mapping and the batch's lease.

    ndarray's buffer exporter retains this base when protocol-5 reconstructs
    the final arrays. DLPack then retains the array; no tensor subclass or
    change in the public NumPy result type is needed.
    """
    def __init__(self, memory, offset, size, lease):
        self.memory, self.lease = memory, lease
        address = ctypes.addressof(ctypes.c_char.from_buffer(memory)) + offset
        self.__array_interface__ = dict(shape=(size,), typestr='|u1',
                                        data=(address, False), version=3)


def leased_buffer(memory, offset, size, lease):
    return np.asarray(_BufferOwner(memory, offset, size, lease))

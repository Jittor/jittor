"""Small CPU control collectives over a distributed rendezvous Store.

This is a host transport, not a binding to libgloo. Accelerator model data must
continue to use its native communicator; CUDA inputs are deliberately rejected.
"""

import pickle
import threading

import numpy as np


class HostCollectives:
    def __init__(self, store, ranks, global_rank):
        self.store = store
        self.ranks = tuple(ranks)
        self.rank = (self.ranks.index(global_rank)
                     if global_rank in self.ranks else -1)
        self.sequence = 0
        self.closed = False
        self.lock = threading.Lock()
        transport = store
        while hasattr(transport, "store"):
            transport = transport.store
        self.owns_server = getattr(transport, "_server", None) is not None

    def exchange(self, operation, value):
        """Publish/read each rank once, then release all per-call store keys."""
        if self.closed:
            raise RuntimeError("CPU process group has been destroyed")
        if self.rank < 0:
            raise RuntimeError("rank is not a member of the CPU process group")
        if len(self.ranks) == 1:
            return [value]
        with self.lock:
            prefix = str(self.sequence) + "/"
            self.sequence += 1
            keys = [prefix + str(rank) for rank in range(len(self.ranks))]
            self.store.set(keys[self.rank], pickle.dumps(
                (operation, value, self.owns_server), protocol=pickle.HIGHEST_PROTOCOL))
            values = [pickle.loads(self.store.get(key)) for key in keys]
            # The TCP server's owner must leave last, even in reordered groups.
            # Otherwise it could close the server (or hold the GIL in a GPU
            # collective) while the cleanup coordinator still needs replies.
            coordinator = next((rank for rank, item in enumerate(values)
                                if item[2]), 0)
            done = {rank: prefix + "done/" + str(rank)
                    for rank in range(len(self.ranks)) if rank != coordinator}
            if self.rank != coordinator:
                # TCPStore replies before publishing arrival. Once root sees
                # every arrival, peers need no further access to these keys.
                self.store.arrive(done[self.rank])
            else:
                self.store.wait(list(done.values()))
                for key in keys + list(done.values()):
                    self.store.delete_key(key)
            if any(name != operation for name, _, _ in values):
                raise RuntimeError("CPU collective order differs between ranks")
            return [item for _, item, _ in values]

    @staticmethod
    def array(tensor):
        if str(tensor.device) != "cpu":
            raise ValueError("CPU process-group collectives require CPU tensors")
        return np.array(tensor.numpy(), copy=True)

    @staticmethod
    def tensor(array, like):
        import jittor as jt

        # Explicit host placement belongs to this operation, not the engine's
        # default device. Park the result on CPU before restoring CUDA flags.
        with jt.runtime.scope(use_cuda=0):
            return jt.array(array, dtype=like.dtype).cpu()

    def all_gather(self, tensor):
        arrays = self.exchange("all_gather", self.array(tensor))
        self._check_arrays(arrays)
        return [self.tensor(array, tensor) for array in arrays]

    @staticmethod
    def _check_arrays(arrays):
        first = arrays[0]
        if any(array.shape != first.shape or array.dtype != first.dtype
               for array in arrays):
            raise ValueError("CPU collective tensor shapes/dtypes differ between ranks")

    def all_reduce(self, tensor, operation):
        arrays = self.exchange("all_reduce/" + operation, self.array(tensor))
        self._check_arrays(arrays)
        result = arrays[0].copy()
        operators = {"sum": np.add, "mean": np.add, "max": np.maximum,
                     "min": np.minimum, "product": np.multiply}
        if operation not in operators:
            raise NotImplementedError("unsupported CPU reduction: " + operation)
        if operation == "mean" and result.dtype.kind not in "fc":
            raise ValueError("CPU mean reduction requires floating-point tensors")
        for array in arrays[1:]:
            operators[operation](result, array, out=result)
        if operation == "mean":
            result /= len(arrays)
        return self.tensor(result, tensor)

    def broadcast(self, tensor, root):
        if not 0 <= root < len(self.ranks):
            raise ValueError("CPU broadcast root is outside the process group")
        arrays = self.exchange("broadcast/" + str(root), self.array(tensor))
        self._check_arrays(arrays)
        return self.tensor(arrays[root], tensor)

    def barrier(self):
        self.exchange("barrier", None)

    def all_gather_object(self, value):
        return self.exchange("all_gather_object", value)

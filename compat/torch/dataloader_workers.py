"""Ordered CPU process prefetch for batches that contain frontend tensors."""
import importlib
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import weakref

import numpy as np

from ._dataloader_ipc import decode_batch, receive_frame, send_frame
from ..diagnostics import swallowed


def supports_process_batch(value):
    if sys.version_info < (3, 8) or not hasattr(os, 'memfd_create') or not hasattr(socket, 'SOCK_SEQPACKET'):
        return False
    import jittor as jt
    if jt.compiler.has_accelerator and not jt.compiler.is_cuda:
        return False
    from .context import get_install_context
    tensor_type = get_install_context(jt).target_namespace.Tensor
    if isinstance(value, jt.Var):
        return (type(value) is tensor_type and value.device.type == 'cpu'
                and not value.requires_grad and value._storage_is_contiguous())
    if type(value) is dict:
        return all(supports_process_batch(item) for item in value.values())
    if type(value) in (tuple, list):
        return all(supports_process_batch(item) for item in value)
    return isinstance(value, (np.ndarray, np.generic, int, float, complex, str, bytes, bool, type(None)))


def _worker_environment():
    env = dict(os.environ)
    for name in list(env):
        if name.startswith(('OMPI_', 'PMI_', 'PMIX_', 'MV2_', 'TORCHELASTIC_',
                            'JT_NCCL_', 'JITTOR_DIST_')):
            env.pop(name)
    env.update(JT_USE_CUDA='0', use_cuda='0', CUDA_VISIBLE_DEVICES='',
               JITTOR_TORCH_SHIM='1',
               JT_BUILD_USE_MPI='0', JT_BUILD_USE_NCCL='0',
               JITTOR_TORCH_DISTRIBUTED_AUTO_INIT='0',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               RANK='0', LOCAL_RANK='0', WORLD_SIZE='1', LOCAL_WORLD_SIZE='1',
               USE_TORCH_XLA='0')
    # Spawn must be able to import the user's Dataset module, including paths
    # an application or pytest added after interpreter startup.
    env['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(
        os.path.abspath(path or os.getcwd()) for path in sys.path))
    return env


class ProcessPrefetch:
    def __init__(self, loader, indices, seed):
        self.indices = indices
        self.maximum = max(1, int(loader.num_workers)) * max(1, int(loader.prefetch_factor or 2))
        self.timeout = float(loader.timeout or 0)
        self.persistent = bool(loader.persistent_workers)
        self.results = queue.Queue()
        self.pending = {}
        self.current = self.sent = self.active = 0
        self.actors = []
        self.closed = False
        environment = _worker_environment()
        try:
            for worker_id in range(int(loader.num_workers)):
                parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
                try:
                    process = subprocess.Popen(
                        [sys.executable, '-m', 'jittor.compat.torch._dataloader_worker', str(child.fileno())],
                        env=environment, pass_fds=[child.fileno()])
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
                actor = (parent, process)
                self.actors.append(actor)
                configuration = dict(dataset=loader.dataset, collate_fn=loader.collate_fn,
                                     batch_size=loader.batch_size, _iterable_dataset=False,
                                     num_workers=loader.num_workers, worker_init_fn=loader.worker_init_fn,
                                     worker_id=worker_id, seed=seed + worker_id)
                send_frame(parent, 'initialize', None, configuration)
                threading.Thread(target=self._receive, args=(actor,), daemon=True,
                                 name='jt-loader-io-%d' % worker_id).start()
            self._fill()
        except BaseException:
            self.close()
            raise

    def _receive(self, actor):
        channel, process = actor
        mappings = {}
        owner = weakref.ref(self)

        def release(index):
            pool = owner()
            if pool is None or pool.closed:
                return
            try:
                send_frame(channel, 'release', None, index)
            except OSError as error:
                pool.results.put(('closed', process.pid, str(error)))

        try:
            while True:
                self.results.put(receive_frame(channel, mappings, release))
        except (EOFError, OSError, RuntimeError) as error:
            self.results.put(('closed', process.pid, str(error)))
        finally:
            mappings.clear()

    def _fill(self):
        while self.active < self.maximum:
            try:
                indices = next(self.indices)
            except StopIteration:
                return
            send_frame(self.actors[self.sent % len(self.actors)][0], 'fetch', self.sent, indices)
            self.sent += 1
            self.active += 1

    def __iter__(self):
        return self

    def reset(self, batch_sampler):
        """Reuse drained workers and their RNG state for a new sampler pass."""
        if self.closed or self.active:
            return False
        dead = [(process.pid, process.returncode) for _, process in self.actors
                if process.poll() is not None]
        if dead:
            raise RuntimeError('DataLoader worker processes exited: %s' % (dead,))
        self.indices = iter(batch_sampler)
        self.current = self.sent = 0
        # Batch indices can restart after a drain. Shared-memory slot IDs are
        # independent and retain their leases across epochs.
        self._fill()
        return True

    def __next__(self):
        if self.active == 0:
            if not self.persistent:
                self.close()
            raise StopIteration
        deadline = time.monotonic() + self.timeout if self.timeout else None
        try:
            while self.current not in self.pending:
                remaining = .25 if deadline is None else min(.25, deadline - time.monotonic())
                if remaining <= 0:
                    raise RuntimeError('DataLoader timed out after %s seconds' % self.timeout)
                try:
                    frame = self.results.get(timeout=remaining)
                except queue.Empty:
                    dead = [(process.pid, process.poll()) for _, process in self.actors
                            if process.poll() is not None]
                    if dead:
                        raise RuntimeError('DataLoader worker processes exited: %s' % (dead,))
                    continue
                if isinstance(frame, tuple):
                    raise RuntimeError('DataLoader worker channel failed: %s' % (frame,))
                if frame.kind == 'ready':
                    continue
                if frame.kind == 'error' and frame.index is None:
                    _raise_worker_error(frame.decode())
                self.pending[frame.index] = frame
            frame = self.pending.pop(self.current)
            self.current += 1
            self.active -= 1
            self._fill()
            if frame.kind == 'error':
                _raise_worker_error(frame.decode())
            return decode_batch(frame.decode())
        except BaseException:
            self.close()
            raise

    def close(self):
        if getattr(self, 'closed', True):
            return
        self.closed = True
        for channel, process in self.actors:
            try:
                channel.shutdown(socket.SHUT_RDWR)
            except OSError as error:
                swallowed('DataLoader process cleanup: channel.shutdown', error,
                          'the worker channel was already closed')
            channel.close()
            if process.poll() is None:
                process.terminate()
        for _, process in self.actors:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        self.pending.clear()

    def __del__(self):
        self.close()


def _raise_worker_error(record):
    detail = 'DataLoader worker %s raised %s: %s\nWorker traceback:\n%s' % (
        record['worker_id'], record['name'], record['message'], record['traceback'])
    try:
        error_type = importlib.import_module(record['module'])
        for part in record['name'].split('.'):
            error_type = getattr(error_type, part)
        error = error_type(detail)
    except (ImportError, AttributeError, TypeError):
        raise RuntimeError(detail)
    raise error

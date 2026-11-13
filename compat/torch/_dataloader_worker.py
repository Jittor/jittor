"""Entry point for one isolated CPU DataLoader worker."""
import os
import random
import signal
import socket
import sys
import traceback
import types

import numpy as np

from ._dataloader_ipc import encode_batch, receive_frame, send_frame
from ._dataloader_shared_memory import SharedMemoryPool


def main():
    import jittor as jt
    from .context import get_install_context
    from .installers import data as loader_api
    torch = get_install_context(jt).target_namespace
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    channel = socket.socket(fileno=int(sys.argv[1]))
    channel.set_inheritable(False)
    initialization = receive_frame(channel).decode()
    worker_id = initialization.pop('worker_id')
    seed = initialization.pop('seed')
    worker_init = initialization.pop('worker_init_fn')
    loader = types.SimpleNamespace(**initialization)
    try:
        jt.core.set_num_threads(1)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        loader_api._worker_state.info = loader_api._WorkerInfo(
            worker_id, loader.num_workers, seed, loader.dataset)
        if worker_init is not None:
            worker_init(worker_id)
        if jt.flags.use_cuda:
            raise RuntimeError('DataLoader worker initialized an accelerator')
        send_frame(channel, 'ready', None, os.getpid())
    except Exception as error:
        send_frame(channel, 'error', None, _error_record(error, worker_id))
        return
    pool = SharedMemoryPool()
    try:
        while True:
            try:
                frame = receive_frame(channel)
            except EOFError:
                return
            if frame.kind == 'stop':
                return
            if frame.kind == 'release':
                pool.release(frame.decode())
                continue
            try:
                batch = loader_api._fetch_batch(loader, frame.decode())
                send_frame(channel, 'batch', frame.index, encode_batch(batch), pool)
            except Exception as error:
                # Exception transport: the parent re-raises with the traceback.
                send_frame(channel, 'error', frame.index, _error_record(error, worker_id))
    finally:
        pool.close()


def _error_record(error, worker_id):
    return dict(module=type(error).__module__, name=type(error).__qualname__,
                message=str(error), traceback=traceback.format_exc(), worker_id=worker_id)


if __name__ == '__main__':
    main()

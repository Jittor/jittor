"""Explicit unsupported worker signal boundary; no Native torch._C is loaded."""


def _error_if_any_worker_fails():
    raise NotImplementedError('DataLoader worker-process failure monitoring is not implemented')


def _remove_worker_pids(key):
    raise NotImplementedError('DataLoader worker-process PID ownership is not implemented')


def _set_worker_pids(key, pids):
    raise NotImplementedError('DataLoader worker-process PID ownership is not implemented')


def _set_worker_signal_handlers():
    raise NotImplementedError('DataLoader worker-process signal handlers are not implemented')


def _set_SIGCHLD_handler():
    raise NotImplementedError('DataLoader worker-process SIGCHLD handler is not implemented')

"""Importable unsupported pinning protocol, not successful placeholders."""


def pin_memory(data, device=None):
    raise NotImplementedError('DataLoader pinned-memory protocol is not implemented')


def _pin_memory_loop(in_queue, out_queue, device_id, done_event, device):
    raise NotImplementedError('DataLoader pinning threads are not implemented')

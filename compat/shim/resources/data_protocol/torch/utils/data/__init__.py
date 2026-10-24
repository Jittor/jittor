"""Compatibility data protocol: map loading, not complete Torch DataPipes.

Actual third-party torchdata packages import this public Torch protocol. Their
private checkpoint state machine is not copied or replaced here.
"""
import importlib
import sys
from jittor.compat.torch.installers.data import _DATA_APIS

globals().update(_DATA_APIS)
from . import _utils, dataloader, dataset, distributed, sampler

default_convert = _utils.collate.default_convert
default_collate = _utils.collate.default_collate
get_worker_info = _utils.worker.get_worker_info


def __getattr__(name):
    # Lazy real protocol classes keep plain DataLoader import available on older
    # Python; genuine torchdata0.11 itself requires modern Python.
    if name in ('IterDataPipe', 'MapDataPipe', 'DataChunk', 'DFIterDataPipe'):
        if sys.version_info < (3, 9):
            raise NotImplementedError('This DataPipe protocol requires Python3.9 or later')
        module = importlib.import_module(__name__ + '.datapipes.datapipe')
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError('module %r has no attribute %r' % (__name__, name))

"""Checkpoint implementations and their historical utility entry points.

Legacy ``jittor_utils`` modules query services registered by Jittor bootstrap.
Import Jittor first when using those paths directly. The writer remains lazy so
registration does not import Torch or initialize another runtime.
"""

from functools import partial as _partial
from importlib import import_module as _import_module

from jittor_utils.runtime_services import register_runtime_module as _register


_LOADERS = {
    name: _partial(_import_module, __name__ + "." + name)
    for name in ("load_pytorch", "load_pytorch_old", "save_pytorch")
}


def register_compatibility_services():
    for name, loader in _LOADERS.items():
        _register(name, loader)

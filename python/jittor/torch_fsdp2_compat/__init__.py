"""Stable facade for Jittor's Torch FSDP2 and DTensor compatibility layer."""

import contextlib
import enum
import os
import sys
import types

import numpy as np

import jittor as jt
from jittor import nn

from .._torch_fsdp2.runtime import bind_runtime as _bind_runtime
from .._torch_fsdp2.runtime import preserve_facade_origins as _preserve_origins

_bind_runtime(jt, nn, sys.modules[__name__])

from .._torch_fsdp2 import compat_types as _compat_types
from .._torch_fsdp2 import config as _config
from .._torch_fsdp2 import dtensor as _dtensor
from .._torch_fsdp2 import fsdp_api as _fsdp_api
from .._torch_fsdp2 import grad_sync as _grad_sync
from .._torch_fsdp2 import installer as _installer
from .._torch_fsdp2 import optimizer as _optimizer
from .._torch_fsdp2 import shard_common as _shard_common
from .._torch_fsdp2 import shard_runtime as _shard_runtime

_IMPLEMENTATION_MODULES = (
    _shard_common,
    _dtensor,
    _config,
    _shard_runtime,
    _grad_sync,
    _optimizer,
    _fsdp_api,
    _compat_types,
    _installer,
)
_exported_symbols = []
_export_owners = {}
for _module in _IMPLEMENTATION_MODULES:
    for _name in _module.FACADE_EXPORTS:
        if _name in _export_owners:
            raise RuntimeError(
                "duplicate FSDP2 facade export %r from %s and %s"
                % (_name, _export_owners[_name].__name__, _module.__name__)
            )
        _value = getattr(_module, _name)
        globals()[_name] = _value
        _export_owners[_name] = _module
        _exported_symbols.append(_value)

_preserve_origins(_exported_symbols)

del _bind_runtime, _exported_symbols, _module, _name, _preserve_origins, _value

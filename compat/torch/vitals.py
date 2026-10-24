"""Process-owned set_vital(str,str,str) compatibility subset.

Algorithm source: PyTorch v2.8.0 aten/src/ATen/core/Vitals.cpp and
torch/csrc/Module.cpp. This is the constructor/setVital subset only. No
read_vitals, destructor output, CUDA-initialization event, or complete vitals
compatibility is implemented or claimed. The initial CUDA.used='False' is the
original unconditional constructor registration, NOT device observation.

One owner is initialized at the first core.install boundary, not when this
private module is merely imported and not per loader/install-context. The
constructor accepts an environment reader for independent stdlib controls.
"""
import os
from threading import RLock


class _SetVitalOwner:
    def __init__(self, getenv):
        self._getenv = getenv
        self._lock = RLock()
        self._enabled = False
        self._values = {}
        # Native constructor's forced write calls torchVitalEnabled first.
        self._set('CUDA', 'used', 'False', force=True)

    def _is_enabled(self):
        if self._getenv('TORCH_VITAL'):
            self._enabled = True
        return self._enabled

    def _set(self, vital, attr, value, *, force=False):
        with self._lock:
            if not (self._is_enabled() or force):
                return False
            self._values.setdefault(vital, {})[attr] = value
            return True

    def set_vital(self, vital, attr, value):
        # This draft's stated callable domain is the three str arguments used
        # by genuine loader. Bytes/implicit conversion are NOT yet contracted.
        if not all(isinstance(arg, str) for arg in (vital, attr, value)):
            raise TypeError('set_vital requires three str arguments')
        return self._set(vital, attr, value)


# Initialize at the first core.install boundary, not when a private compat
# module happens to be imported before public activation. Retain one owner for
# the process; repeated installation never resets its sticky gate or values.
_OWNER = None
_INIT_LOCK = RLock()


def initialize_vitals():
    global _OWNER
    with _INIT_LOCK:
        if _OWNER is None:
            _OWNER = _SetVitalOwner(os.environ.get)
    return _OWNER


def set_vital(vital, attr, value):
    if _OWNER is None:
        raise RuntimeError('vitals owner is not initialized by Torch compatibility activation')
    return _OWNER.set_vital(vital, attr, value)

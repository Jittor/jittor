"""Native operator namespace and Python tensor-operation implementation domains.

Native callables remain owned by ``jittor_core.ops``. Python implementations
live in child modules, so their names cannot replace low-level operators.
"""

from jittor_core import ops as _native_ops


def __getattr__(name):
    return getattr(_native_ops, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_native_ops)))


__all__ = [name for name in dir(_native_ops) if not name.startswith("_")]

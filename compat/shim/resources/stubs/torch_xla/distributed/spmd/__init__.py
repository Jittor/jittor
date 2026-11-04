"""Import-compatible placeholder for ``torch_xla.distributed.spmd``.

Transformers imports this namespace when the emulated ``torch_xla`` version is
new enough for XLA FSDPv2.  SPMD sharding has no Jittor equivalent here, so the
names exist only to fail clearly when an FSDPv2 configuration actually asks for
them.
"""

from __future__ import absolute_import


def _unsupported(name):
    def unavailable(*args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "torch_xla emulation: %s is part of XLA SPMD sharding, which is "
            "not implemented on Jittor" % (name,)
        )

    unavailable.__name__ = name
    return unavailable


Mesh = _unsupported("Mesh")
set_global_mesh = _unsupported("set_global_mesh")
get_global_mesh = _unsupported("get_global_mesh")
mark_sharding = _unsupported("mark_sharding")
clear_sharding = _unsupported("clear_sharding")

__all__ = ["Mesh", "set_global_mesh", "get_global_mesh", "mark_sharding", "clear_sharding"]

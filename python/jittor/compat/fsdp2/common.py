"""Low-level sharding and collective helpers for FSDP2 compatibility."""

import os

import numpy as np

import jittor as jt
from ..diagnostics import EXPECTED, swallowed
# Rank/world queries and the two collectives moved down to jittor/compat/
# collectives.py: compat/torch/installers/distributed.py needs
# `_all_gather_shards` for plain `all_gather`, and borrowing it from here made
# the distributed installer depend on FSDP2. Re-exported so that every
# `common._all_gather_shards(...)` inside fsdp2 keeps working unchanged.
from ..collectives import (          # noqa: F401
    _all_gather_shards,
    _in_true_distributed,
    _nccl_ops,
    _rank,
    _reduce_scatter_padded,
    _slice_flat,
    _world_size,
)


def _prod(xs):
    out = 1
    for x in xs:
        try:
            out *= int(x)
        except EXPECTED as exc:
            swallowed("fsdp2/common.py _prod: out *= int(x)", exc)
    return out


def _flatten_var(v):
    return v.reshape((-1,))


def _ceil_div(a, b):
    return (int(a) + int(b) - 1) // int(b)


def _pad_flat(flat, padded_numel):
    n = int(flat.numel()) if callable(getattr(flat, "numel", None)) else int(np.prod(flat.shape))
    if n == int(padded_numel):
        return flat
    pad = jt.zeros((int(padded_numel) - n,), dtype=flat.dtype)
    return jt.concat([flat, pad], dim=0)


def _param_numel(v):
    return int(np.prod(tuple(int(x) for x in v.shape)))


def _fsdp2_flat_enabled(world_size, total_numel):
    value = os.environ.get("JITTOR_FSDP2_FLAT", "auto").lower()
    if value in ("0", "false", "no"):
        return False
    if value in ("1", "true", "yes"):
        return True
    # Flat sharding removes several tiny NCCL launches and is consistently
    # faster on 2 ranks. On 4 ranks it helps small models but the extra
    # flatten/slice work slows medium-size cases, so keep the legacy path there.
    return int(world_size) <= 2 or int(total_numel) <= 1_000_000


# The rank/world queries and the two collectives are re-exported above for the
# fsdp2 code that says `common._all_gather_shards(...)`, but they are no longer
# *owned* here -- they live in jittor/compat/collectives.py, below both fsdp2
# and the distributed installer. Listing them here would make the fsdp2 package
# re-publish somebody else's functions as its own.
_EXPORTS = (
    "_prod",
    "_flatten_var",
    "_ceil_div",
    "_pad_flat",
    "_param_numel",
    "_fsdp2_flat_enabled",
)

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


#: Where "auto" switches flat sharding off, and how to move it.
#:
#: Flat sharding removes several tiny NCCL launches and was consistently faster
#: on 2 ranks; on 4 ranks it helps small models but the extra flatten/slice
#: work slows medium-size cases. Both numbers come from one set of measurements
#: on one machine, so they are defaults rather than facts: a different
#: interconnect or a different model size moves them, and before this they were
#: literals in the middle of a boolean with no way to try another value short
#: of editing the source. Overriding either variable does not reach for
#: JITTOR_FSDP2_FLAT=1/0, which still forces the answer outright.
_FLAT_MAX_WORLD_SIZE = 2
_FLAT_MAX_NUMEL = 1_000_000


def _flat_threshold(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            "{}={!r} is not an integer. It is the parameter count (or rank "
            "count) at which FSDP2 stops using flat sharding; leave it unset "
            "for {}.".format(name, raw, default))


def _fsdp2_flat_enabled(world_size, total_numel):
    """Whether to shard this module flat, and why.

    ``JITTOR_FSDP2_FLAT`` forces the answer (1/0). Left at "auto", the two
    thresholds above decide, and each is itself overridable -- the point of
    8.11 is that a policy tuned on one machine should be reachable from the
    environment on another, not that these particular numbers are right.
    """
    value = os.environ.get("JITTOR_FSDP2_FLAT", "auto").lower()
    if value in ("0", "false", "no"):
        return False
    if value in ("1", "true", "yes"):
        return True
    max_world = _flat_threshold("JITTOR_FSDP2_FLAT_MAX_WORLD_SIZE",
                                _FLAT_MAX_WORLD_SIZE)
    max_numel = _flat_threshold("JITTOR_FSDP2_FLAT_MAX_NUMEL",
                                _FLAT_MAX_NUMEL)
    return int(world_size) <= max_world or int(total_numel) <= max_numel


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

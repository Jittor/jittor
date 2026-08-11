"""Low-level sharding and collective helpers for FSDP2 compatibility."""

import os

import numpy as np

from .runtime import facade, jt, preserve_facade_origins


def _prod(xs):
    out = 1
    for x in xs:
        try:
            out *= int(x)
        except Exception:
            pass
    return out


def _world_size():
    try:
        return int(getattr(jt, "world_size", 1))
    except Exception:
        return 1


def _rank():
    try:
        return int(getattr(jt, "rank", 0))
    except Exception:
        return 0


def _in_true_distributed():
    return facade._world_size() > 1 and (
        os.environ.get("JT_NCCL_WORLD_SIZE") is not None
        or os.environ.get("OMPI_COMM_WORLD_SIZE") is not None
        or getattr(jt, "in_mpi", False)
    )


def _nccl_ops():
    try:
        ops = getattr(jt.compile_extern, "nccl_ops", None)
        if ops is not None:
            return ops
        if os.environ.get("JT_NCCL_WORLD_SIZE") is not None:
            os.environ.setdefault("use_nccl", "1")
            setup = getattr(jt.compile_extern, "setup_nccl", None)
            if callable(setup):
                setup()
            return getattr(jt.compile_extern, "nccl_ops", None)
    except Exception:
        return None
    return None


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


def _slice_flat(flat, start, length):
    start = int(start)
    length = int(length)
    return flat[start:start + length]


def _all_gather_shards(local_shard):
    ops = facade._nccl_ops()
    if ops is not None and callable(getattr(ops, "nccl_all_gather", None)):
        return ops.nccl_all_gather(local_shard)
    if callable(getattr(local_shard, "mpi_all_gather", None)):
        return local_shard.mpi_all_gather()
    raise RuntimeError("Jittor NCCL all_gather is not available; launch with jittor.distributed.launch and use_nccl=1")


def _reduce_scatter_padded(full_grad):
    ops = facade._nccl_ops()
    if ops is not None and callable(getattr(ops, "nccl_reduce_scatter", None)):
        return ops.nccl_reduce_scatter(full_grad)
    # Correct fallback for environments with all_reduce but without native
    # reduce_scatter.  It communicates more than needed, but preserves semantics.
    reduced = full_grad.mpi_all_reduce("sum")
    shard = int(reduced.shape[0]) // max(facade._world_size(), 1)
    return facade._slice_flat(reduced, facade._rank() * shard, shard)


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


FACADE_EXPORTS = (
    "_prod",
    "_world_size",
    "_rank",
    "_in_true_distributed",
    "_nccl_ops",
    "_flatten_var",
    "_ceil_div",
    "_pad_flat",
    "_slice_flat",
    "_all_gather_shards",
    "_reduce_scatter_padded",
    "_param_numel",
    "_fsdp2_flat_enabled",
)

preserve_facade_origins(
    (globals()[name] for name in FACADE_EXPORTS),
    __name__,
)

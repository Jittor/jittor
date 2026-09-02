"""Rank/world queries and the two collectives the sharded paths are built on.

These lived in ``fsdp2/common.py``, whose docstring already called them
"low-level collective helpers" -- but nothing about ``_all_gather_shards`` is
specific to FSDP, and ``compat/torch/installers/distributed.py`` was reaching
*up* into ``fsdp2.common`` to borrow it for ``all_gather``,
``all_gather_object`` and ``all_gather_into_tensor``. That made the distributed
installer depend on FSDP2, the exact inversion of the intended layering
(``core -> tensor -> nn/optim -> distributed -> fsdp``).

Both sides now depend on this module instead, which depends on nothing but
jittor itself. ``fsdp2.common`` re-exports every name here, so the FSDP2 code
that says ``common._all_gather_shards(...)`` keeps working unchanged.
"""

from __future__ import absolute_import

import os

import jittor as jt

from .diagnostics import EXPECTED, swallowed

__all__ = ["_world_size", "_rank", "_in_true_distributed", "_nccl_ops",
           "_slice_flat", "_all_gather_shards", "_reduce_scatter_padded",
           "_all_reduce_mean", "_broadcast_from_rank0"]


def _world_size():
    # Deliberately not wrapped in try/except. If jittor's world_size cannot be
    # read, returning 1 here does not degrade gracefully -- it turns an N-rank
    # job into N independent single-rank jobs that each train the full model and
    # never exchange anything, which looks like it is working. 6.B04.
    return int(getattr(jt, "world_size", 1))


def _rank():
    # Same reasoning as _world_size: a swallowed error here makes every rank
    # believe it is rank 0 and shard identically.
    return int(getattr(jt, "rank", 0))


def _in_true_distributed():
    return _world_size() > 1 and (
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
    except EXPECTED as exc:
        swallowed("collectives.py _nccl_ops: ops = getattr(jt.compile_extern, 'nccl_ops', None)", exc)
        return None
    return None


def _slice_flat(flat, start, length):
    start = int(start)
    length = int(length)
    return flat[start:start + length]


def _all_gather_shards(local_shard):
    # On one rank the gather is the local shard itself. Say so before reaching
    # for a collective: ``fully_shard`` on a single process is a supported
    # configuration -- it is how the FSDP2 paths in ms-swift and verl run on
    # CPU -- and demanding NCCL there turns a no-op into a hard failure.
    if _world_size() <= 1:
        return local_shard
    ops = _nccl_ops()
    if ops is not None and callable(getattr(ops, "nccl_all_gather", None)):
        return ops.nccl_all_gather(local_shard)
    if callable(getattr(local_shard, "mpi_all_gather", None)):
        return local_shard.mpi_all_gather()
    raise RuntimeError("Jittor NCCL all_gather is not available; launch with jittor.distributed.launch and use_nccl=1")


def _reduce_scatter_padded(full_grad):
    # Likewise the identity on one rank: nothing to reduce against, and rank 0's
    # shard is the whole padded gradient.
    if _world_size() <= 1:
        return full_grad
    ops = _nccl_ops()
    if ops is not None and callable(getattr(ops, "nccl_reduce_scatter", None)):
        return ops.nccl_reduce_scatter(full_grad)
    # Correct fallback for environments with all_reduce but without native
    # reduce_scatter.  It communicates more than needed, but preserves semantics.
    reduced = full_grad.mpi_all_reduce("sum")
    shard = int(reduced.shape[0]) // max(_world_size(), 1)
    return _slice_flat(reduced, _rank() * shard, shard)


def _all_reduce_mean(var):
    """Average ``var`` across every rank. The identity on one rank."""
    if _world_size() <= 1:
        return var
    if not callable(getattr(var, "mpi_all_reduce", None)):
        raise RuntimeError(
            "this Jittor build has no mpi_all_reduce, so gradients cannot be "
            "synchronised across the %d ranks it was launched with. Build "
            "with MPI (mpicc on PATH at build time) or launch with NCCL."
            % _world_size())
    return var.mpi_all_reduce("mean")


def _broadcast_from_rank0(var):
    """Replace ``var`` with rank 0's copy in place. The identity on one rank."""
    if _world_size() <= 1:
        return var
    if not callable(getattr(var, "mpi_broadcast", None)):
        raise RuntimeError(
            "this Jittor build has no mpi_broadcast, so parameters cannot be "
            "broadcast to the %d ranks it was launched with." % _world_size())
    var.assign(var.mpi_broadcast())
    return var

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
           "_all_reduce_mean", "_broadcast_from_rank0", "_reduce_scalar"]


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
    """NCCL, setting it up on demand if this is a real multi-rank CUDA job.

    The shim's preflight used to default ``use_nccl=0`` unconditionally, and
    ``compile_extern.setup_nccl()`` reads that variable once, during ``import
    jittor``. The only other place setting it back (installers/distributed.py)
    runs after that import. So under ``import jittor as torch`` NCCL was always
    absent and every FSDP2 shard gather raised "Jittor NCCL all_gather is not
    available": FSDP2 could not run multi-rank in torch mode at all.

    That is fixed where it belongs -- shim/preflight.py now leaves NCCL on when
    the launcher says there is more than one rank, before the core loads.
    """
    try:
        ops = getattr(jt.compile_extern, "nccl_ops", None)
        if ops is not None:
            return ops
        if os.environ.get("JT_NCCL_WORLD_SIZE") is None:
            return None
        # Only the MPI-free rendezvous is set up from here, and only because
        # that launcher sets its variable after the shim's preflight has run.
        # NCCL is otherwise decided in shim/preflight.py, *before* the core is
        # imported, which is the only safe place: `setup_nccl()` reads
        # `use_nccl` once during that import, and initialising a communicator
        # any later -- from inside a collective, underneath a graph that is
        # already executing -- kills the next matmul with
        # CUBLAS_STATUS_EXECUTION_FAILED.
        os.environ.setdefault("use_nccl", "1")
        setup = getattr(jt.compile_extern, "setup_nccl", None)
        if callable(setup):
            setup()
        return getattr(jt.compile_extern, "nccl_ops", None)
    except EXPECTED as exc:
        swallowed("collectives.py _nccl_ops: setting NCCL up for a %d-rank "
                  "CUDA job" % _world_size(), exc,
                  "the sharded collectives will refuse instead of running")
        return None


def _slice_flat(flat, start, length):
    start = int(start)
    length = int(length)
    return flat[start:start + length]


def _world_process_group():
    # Native ProcessGroup owns backend selection, communicator identity and
    # device collectives. In particular, MPI-free HCCL must not depend on a
    # historical Var.mpi_* alias that may not exist.
    from jittor.distributed.process_group import get_default_group
    group = get_default_group()
    expected = _world_size()
    if group.size() != expected:
        raise RuntimeError(
            "collectives were requested for %d ranks but native WORLD has %d"
            % (expected, group.size()))
    return group


def _all_gather_shards(local_shard):
    if _world_size() <= 1:
        return local_shard
    return _world_process_group().all_gather(local_shard)


def _reduce_scatter_padded(full_grad):
    if _world_size() <= 1:
        return full_grad
    return _world_process_group().reduce_scatter(full_grad)


def _all_reduce_mean(var):
    """Average ``var`` across every rank. The identity on one rank."""
    if _world_size() <= 1:
        return var
    return _world_process_group().all_reduce(var, "mean")


def _broadcast_from_rank0(var):
    """Replace ``var`` with rank 0's copy in place."""
    if _world_size() <= 1:
        return var
    result = _world_process_group().broadcast(var, 0)
    var.assign(result)
    return var


def _reduce_scalar(var, how):
    """Combine a 0-d/1-element Var across ranks. The identity on one rank.

    Built on ``_all_gather_shards`` rather than ``mpi_all_reduce`` because
    Jittor's all-reduce implements only ``mean`` and ``add``
    (``mpi_all_reduce_op.cc`` asserts on anything else), while a p-norm needs
    ``max`` and ``min`` as well for the infinity norms. The payload is one
    number per rank, so gathering and reducing locally costs nothing.
    """
    if _world_size() <= 1:
        return var
    gathered = _all_gather_shards(var.reshape((1,)))
    if how == "sum":
        return gathered.sum()
    if how == "max":
        return gathered.max()
    if how == "min":
        return gathered.min()
    raise ValueError("unsupported cross-rank reduction %r" % (how,))

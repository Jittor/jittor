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
from contextlib import contextmanager

import jittor as jt

from .._core.dtypes import dtype_name as _dtype_name

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


def _mpi_cpu_all_gather_shards(local_shard):
    """Gather equal-shaped CPU shards over MPI when no native all-gather exists.

    MPI all-reduce is available in Jittor CPU builds, while the matching
    mpi_all_gather Var op is not. A source-at-a-time sum gives every rank the
    same shard sequence. Materialize each collective before issuing the next
    one so the ranks cannot flush their lazy graphs in different orders.
    """
    compile_extern = getattr(jt, "compile_extern", None)
    if not bool(getattr(compile_extern, "in_mpi", False)):
        return None
    location = getattr(local_shard, "location", None)
    if not callable(location) or location() not in ("cpu", "none"):
        return None
    reducer = getattr(local_shard, "mpi_all_reduce", None)
    if not callable(reducer):
        return None

    rank = _rank()
    size = _world_size()
    dtype_name = _dtype_name(local_shard.dtype)
    flat = local_shard.reshape((-1,))
    # Jittor's MPI wrapper has no bfloat16 datatype. Since a gather has only
    # one nonzero contributor per output shard, widening to float32 is exact;
    # cast the gathered shard back before returning it.
    if dtype_name == "bfloat16":
        flat = flat.float32()
    zero = jt.zeros_like(flat)
    gathered = []
    for source_rank in range(size):
        contribution = flat if rank == source_rank else zero
        part = contribution.mpi_all_reduce("add")
        part.sync()
        if dtype_name == "bfloat16":
            part = part.bfloat16()
        gathered.append(part)
    return jt.concat(gathered, dim=0)


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
    mpi_result = _mpi_cpu_all_gather_shards(local_shard)
    if mpi_result is not None:
        return mpi_result
    raise RuntimeError(
        "Jittor all_gather requires NCCL, HCCL, or the CPU MPI fallback; "
        "launch with a matching distributed backend")


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


@contextmanager
def _broadcast_placement_scope(var):
    """Make a collective result follow the tensor, not an unrelated ambient scope.

    mpi_broadcast creates a result using Jittor's ambient placement. Torch DDP
    can reach it while a CPU placement scope is active even though a parameter
    has already been moved to CUDA. In that case, keep both the collective
    result and its assignment on the parameter's actual device.
    """
    location = var.location()
    if location == "device":
        backend = 1
    elif location == "cpu":
        backend = 0
    else:
        # Preserve FollowRuntime/metadata-only behavior when the Var has no
        # concrete allocation from which to infer a backend.
        try:
            backend = int(var.placement_backend)
        except (AttributeError, TypeError, ValueError):
            backend = -1
        if backend < 0:
            yield
            return

    try:
        device_id = int(var.device_id)
    except (AttributeError, TypeError, ValueError):
        device_id = -1
    if device_id < 0:
        if backend == 1:
            try:
                device_id = int(jt.current_device())
            except (AttributeError, TypeError, ValueError, RuntimeError):
                device_id = 0
        else:
            device_id = 0

    token = jt.core._set_tensor_placement(backend, max(device_id, 0))
    try:
        yield
    finally:
        jt.core._reset_tensor_placement(token)


def _broadcast_from_rank0(var):
    """Replace var with rank 0's copy in place. The identity on one rank."""
    if _world_size() <= 1:
        return var
    if not callable(getattr(var, "mpi_broadcast", None)):
        raise RuntimeError(
            "this Jittor build has no mpi_broadcast, so parameters cannot be "
            "broadcast to the %d ranks it was launched with." % _world_size())
    with _broadcast_placement_scope(var):
        var.assign(var.mpi_broadcast())
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

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
           "_hccl_collective", "_all_reduce_mean", "_broadcast_from_rank0",
           "_reduce_scalar"]


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


def _hccl_collective(function, value, *args):
    """Create an HCCL op with an explicit ACL output placement.

    Autograd returns a lazy Var before its producer graph has necessarily been
    assigned a backend.  Such a Var can still report CPU here even in an ACL
    worker.  HCCL ops have no CPU implementation (by design), so leaving the
    output unplaced lets the lazy executor select CPU for the collective.  An
    explicit output placement makes the collective the accelerator anchor for
    that pending producer graph, just as an explicitly placed model parameter
    does for forward all-gather.
    """
    device_id = int(getattr(value, "device_id", -1))
    if device_id < 0:
        device_id = max(int(getattr(jt.flags, "device_id", 0)), 0)
    trace = os.environ.get("JITTOR_VERL_FATAL_TRACE") == "1"
    if trace:
        print(
            "JITTOR_VERL_HCCL_INPUT placement=%d device=%d dispatch=%s use_acl=%d" % (
                int(getattr(value, "placement_backend", -1)),
                int(getattr(value, "device_id", -1)),
                jt.core.dispatch_context([value]),
                int(bool(getattr(jt.flags, "use_acl", 0))),
            ),
            flush=True,
        )
    token = jt.core._set_tensor_placement(2, device_id)  # BackendId::Acl
    try:
        # A collective is a communication boundary, not an elementwise node.
        # If it fuses with the following divide/slice/stop-grad chain, the
        # fused group's terminal (host-visible) output can make the executor
        # request a CPU implementation for this internal HCCL op. Mark the
        # collective output as a fusion barrier while its ACL placement is
        # active so backend selection is made for the communication op itself.
        with jt.flag_scope(no_fuse=1):
            result = function(value, *args)
        if trace:
            print(
                "JITTOR_VERL_HCCL_OUTPUT placement=%d device=%d dispatch=%s" % (
                    int(getattr(result, "placement_backend", -1)),
                    int(getattr(result, "device_id", -1)),
                    jt.core.dispatch_context([result]),
                ),
                flush=True,
            )
        # FSDP consumes this as detached optimizer-gradient storage. Execute
        # the communication while the ACL placement scope is still active;
        # otherwise a later host-visible unused-gradient flag can become the
        # first graph synchronization point and make the lazy execution plan
        # inherit CPU for this already-created HCCL node. This intentionally
        # prioritizes correctness over overlap until the planner can preserve
        # per-op backend placement across that deferred graph.
        result.sync()
        return result
    finally:
        jt.core._reset_tensor_placement(token)


def _all_gather_shards(local_shard):
    # On one rank the gather is the local shard itself. Say so before reaching
    # for a collective: ``fully_shard`` on a single process is a supported
    # configuration -- it is how the FSDP2 paths in ms-swift and verl run on
    # CPU -- and demanding NCCL there turns a no-op into a hard failure.
    if _world_size() <= 1:
        return local_shard
    if not jt.flags.use_cuda:
        from jittor import distributed
        if distributed.get_backend() == "mpi":
            return distributed.all_gather(local_shard)
    # MPI-free Ascend publishes native HCCL ops, but no Var.mpi_all_gather.
    # Select by this tensor's backend rather than whichever communicator was
    # imported first; an ACL tensor must never enter NCCL or a host fallback.
    backend, _device_id = jt.core.dispatch_context([local_shard])
    if backend in ("acl", "acl_legacy"):
        ops = getattr(jt.compile_extern, "hccl_ops", None)
        gather = getattr(ops, "hccl_all_gather", None)
        if not callable(gather):
            raise RuntimeError("Jittor HCCL all_gather is not available for the ACL tensor")
        return gather(local_shard)
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
    backend, _device_id = jt.core.dispatch_context([full_grad])
    ops = getattr(jt.compile_extern, "hccl_ops", None)
    reduce_scatter = getattr(ops, "hccl_reduce_scatter", None)
    # A freshly-created gradient is still a lazy graph value here. On ACL its
    # producer may not have been placed yet, so dispatch_context can temporarily
    # report CPU even though the active runtime and collective backend are HCCL.
    # The ambient ACL flag is authoritative for a WORLD collective; relying only
    # on the lazy value made large FSDP gradients fall through to mpi_all_reduce
    # and then fail because hccl_all_reduce has (correctly) no CPU kernel.
    use_hccl = bool(getattr(jt.flags, "use_acl", 0)) and callable(reduce_scatter)
    if backend in ("acl", "acl_legacy") or use_hccl:
        if not callable(reduce_scatter):
            raise RuntimeError(
                "Jittor HCCL reduce_scatter is not available for the ACL tensor")
        return _hccl_collective(reduce_scatter, full_grad)
    ops = _nccl_ops()
    if ops is not None and callable(getattr(ops, "nccl_reduce_scatter", None)):
        return ops.nccl_reduce_scatter(full_grad)
    # Correct fallback for environments with all_reduce but without native
    # reduce_scatter.  It communicates more than needed, but preserves semantics.
    reduced = full_grad.mpi_all_reduce("sum")
    rank, world_size = _rank(), max(_world_size(), 1)
    total = int(reduced.shape[0])
    if total % world_size:
        raise RuntimeError(
            "reduce-scatter input length %s is not divisible by world size %s" %
            (total, world_size)
        )
    shard = total // world_size
    backend, _device_id = jt.core.dispatch_context([reduced])
    if backend in ("acl", "acl_legacy"):
        # CANN 9 SliceV2 is off by one on some very large tail slices. The
        # padded input is an exact partition, so SplitWithSize is both more
        # direct and reliable. This is detached optimizer-gradient storage.
        # Materialize HCCL first so the lazy graph placer cannot inherit the
        # inference-only split's host context and put the collective on CPU.
        reduced.sync()
        with jt.flag_scope(no_grad=1):
            return jt.split(reduced, [shard] * world_size, dim=0)[rank]
    return _slice_flat(reduced, rank * shard, shard)


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

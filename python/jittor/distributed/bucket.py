"""Bucketing several collectives into one NCCL group, with an overlap window.

Two things a collective could not do before 8.02, both needed by DDP (7.02):

* **Group them.** N small all-reduces cost N launches. ``ncclGroupStart`` /
  ``ncclGroupEnd`` submit the whole bucket at once.
* **Overlap them with compute.** Every collective used to join the default
  stream back to the communication stream the instant it was enqueued, which
  orders all later default-stream work behind the collective and makes overlap
  impossible by construction. ``defer_join`` leaves that join outstanding
  until :func:`comm_wait`.

The C++ side is in ``extern/cuda/nccl/src/nccl_wrapper.cc``; the docstring on
``nccl_bucket_begin`` there carries the full contract. The part callers have to
respect:

    with bucket_scope():
        reduced = [g.mpi_all_reduce() for g in grads]
        jt.sync(reduced)          # <- inside the scope, and only these
    other_compute()               # overlaps with the collectives
    comm_wait()                   # before anything reads `reduced`

``jt.sync`` has to be *inside* the scope because the group has to be open
around the NCCL calls, and those happen when the graph executes, not when the
scope object is built. And it has to sync exactly the collective outputs:
inside a group the NCCL calls have not run yet, so anything that consumes a
collective's output within the scope reads a buffer that nothing has written.
Syncing the collective outputs makes them the sinks of the executed graph,
which rules that out.
"""

from __future__ import absolute_import

from contextlib import contextmanager

__all__ = ["bucket_scope", "comm_wait", "join_pending"]

_COMMUNICATION_STREAM = 1


def _jittor():
    # Imported on use, not at module scope: this package is reachable from
    # `python -m jittor.distributed.launch`, which must not drag the whole
    # runtime in just to read argv.
    import jittor
    return jittor


def _module():
    # The bucket entry points are free functions on the NCCL module itself,
    # not operators, so they live on `nccl` rather than `nccl.ops`.
    module = _jittor().compile_extern.nccl
    if module is None:
        raise RuntimeError(
            "collective bucketing needs NCCL, which is not set up in this "
            "process; check use_nccl and that this is a CUDA build")
    return module


@contextmanager
def bucket_scope(defer_join=True):
    """Group every collective executed inside into one NCCL submission.

    With ``defer_join`` (the default) the default stream is left free to run
    ahead of the bucket; call :func:`comm_wait` before reading the results.
    With ``defer_join=False`` the bucket is still submitted as one group but
    joins immediately, which is the useful setting for measuring what grouping
    alone buys.
    """
    module = _module()
    module.nccl_bucket_begin(defer_join)
    try:
        yield
    finally:
        # Closing the group is not optional: an ncclGroupStart left open makes
        # the next unrelated collective part of this bucket.
        module.nccl_bucket_end()


def comm_wait():
    """Order the default stream behind any outstanding collective.

    Returns True if there was something to wait for, which is also the
    evidence that an overlap window existed.
    """
    return bool(_module().nccl_comm_wait())


def join_pending(device=None):
    """Whether a deferred join is still outstanding on the given device."""
    jt = _jittor()
    if device is None:
        device = jt.current_device()
    return bool(jt.core._cuda_stream_join_pending(
        _COMMUNICATION_STREAM, device))

"""Tools Jittor offers its users, as opposed to tools that build Jittor.

``jittor/utils`` used to hold both, plus a source translator and a Flask app.
The split is by audience:

* here — things a user imports or runs against their own model: an NVTX range
  marker for profiler timelines, and ``jtune`` for re-running one generated
  kernel by hand;
* repository ``tools/`` — things a maintainer runs against the checkout;
* ``jittor/utils`` — the few files the compiler and the C++ core reach by
  hard-coded *path*, which is why they cannot move without a C++ change.

Nothing is imported here: ``jittor.tools.nvtx`` loads the NVTX shared library
at import, and it must not be paid for by ``import jittor``.
"""
